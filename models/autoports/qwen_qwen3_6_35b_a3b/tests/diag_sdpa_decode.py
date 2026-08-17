# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Isolate ``ttnn.transformer.paged_scaled_dot_product_attention_decode`` at long context.

Why: ``test_longest_decode_context[full]`` is the only case in the stage materially below the
rest (PCC 0.9986 at position 262143 vs ~0.99999 everywhere else). ``diag_long_decode.py``
narrowed it to the attention branch and ruled out operand quantisation (the device diverges from
an *exact* bf16-operand reference too). This script strips away everything else — no
projections, no RoPE, no output gate, no o_proj, no MoE — and drives the op directly with random
K/V over a paged cache.

Omitting the program config changes **four** things at once, so all four are swept independently:

1. the core grid (device grid vs whatever is passed);
2. ``k_chunk_size`` (dynamic vs explicit);
3. ``exp_approx_mode`` (defaults to ``true`` with no config);
4. ``max_cores_per_head_batch`` -- the one that is easy to miss. It is a *struct default*
   (``sdpa_config.hpp:18``, value 16), and the factory reads
   ``program_config.has_value() ? program_config->max_cores_per_head_batch : num_cores_available``
   (``sdpa_decode_program_factory.cpp:192-193``). So on this 11x10 part with batch 1 and 2 KV
   heads, **no config gives 55 cores per head while any explicit config defaults to 16** --
   passing a config silently changes the parallel decomposition and the depth of the
   partial-softmax tree reduction, not just the knob you set.

Sweeping (4) is what separates "a program config is present" from "the op runs on N cores per
head". The ``max_cores=110`` row is the control: it should reproduce the no-config row exactly.

    python models/autoports/qwen_qwen3_6_35b_a3b/tests/diag_sdpa_decode.py
"""

import torch

import ttnn

NH, NKV, HD, BLOCK = 16, 2, 256, 64
MAX_CONTEXT = 262144

#: contexts chosen to include tiny, tile/block boundaries, non-multiples and the advertised max
CONTEXTS = [1, 32, 64, 128, 257, 1024, 4096, 32768, 131072, 262143, 262144]
#: (grid, k_chunk_size, exp_approx, max_cores_per_head_batch). ``k_chunk_size=0`` keeps the
#: dynamic chunk path and is legal for the paged *causal* decode op (only the non-causal branch
#: requires > 0, sdpa_decode_device_operation.cpp:321-327), so chunking can be held fixed while
#: the other axes move. ``max_cores=None`` means "leave the struct default", i.e. 16.
COMBOS = [
    # --- reference: no program config at all (this is what the layer shipped) ---
    (None, 0, None, None),
    # --- axis 4, everything else held at the device grid + dynamic chunk + exact exp ---
    ((11, 10), 0, False, 1),
    ((11, 10), 0, False, 2),
    ((11, 10), 0, False, 4),
    ((11, 10), 0, False, 8),
    ((11, 10), 0, False, 16),
    ((11, 10), 0, False, 32),
    ((11, 10), 0, False, 55),
    ((11, 10), 0, False, 110),  # <- the missing control: should reproduce the no-config row
    # --- axis 3 (exp_approx) at both ends of axis 4, to show it is still irrelevant ---
    ((11, 10), 0, True, 16),
    ((11, 10), 0, True, 110),
    # --- axis 1 (grid) at a fixed max_cores, to show the grid itself is not the variable ---
    ((8, 8), 0, False, 16),
    ((8, 8), 0, False, 110),
    # --- axis 2 (chunking) at a fixed max_cores ---
    ((11, 10), 128, False, 16),
    ((11, 10), 512, False, 16),
    ((11, 10), 128, False, 110),
    # --- the original small-grid rows, kept so the earlier table stays comparable ---
    ((1, 1), 128, False, None),
    ((2, 1), 128, False, None),
    ((8, 1), 64, False, None),
    ((8, 8), 512, False, None),
]


def cores_per_head(grid, max_cores, device_grid_cores, batch=1, num_kv_heads=NKV):
    """The factory's own arithmetic (sdpa_decode_program_factory.cpp:192-197), so the table can be
    read without the source open. ``grid is None`` == no program config."""
    avail = device_grid_cores if grid is None else grid[0] * grid[1]
    max_cores_per_head = avail if grid is None else (16 if max_cores is None else max_cores)
    uncapped = min(avail, max_cores_per_head * batch * num_kv_heads) // batch
    return max(1, uncapped // num_kv_heads)


def pcc(a, b):
    x = a.reshape(-1).double()
    y = b.reshape(-1).double()
    x = x - x.mean()
    y = y - y.mean()
    return float((x @ y) / (x.norm() * y.norm() + 1e-30))


#: (label, max_cores_per_head_batch or None-for-no-config) timed at TIME_CONTEXT. Accuracy is only
#: half the decision: 1 core/head serialises each KV head's keys onto a single core, so the cost
#: has to be measured rather than assumed.
TIMED = [("no-config (55/head)", "none"), ("16/head", 16), ("4/head", 4), ("1/head", 1)]
TIME_CONTEXT = 262144
TIME_ITERS = 20


def main():
    torch.set_num_threads(16)
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    dev_grid = device.compute_with_storage_grid_size()
    grid_cores = dev_grid.x * dev_grid.y
    hifi = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    try:
        gen = torch.Generator().manual_seed(7)
        k = torch.randn(1, NKV, MAX_CONTEXT, HD, generator=gen).to(torch.bfloat16).float()
        v = torch.randn(1, NKV, MAX_CONTEXT, HD, generator=gen).to(torch.bfloat16).float()
        q = torch.randn(1, 1, NH, HD, generator=gen).to(torch.bfloat16).float()

        blocks = MAX_CONTEXT // BLOCK
        as_cache = lambda t: ttnn.from_torch(  # noqa: E731 - local shorthand
            t.reshape(NKV, blocks, BLOCK, HD).permute(1, 0, 2, 3).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cache_k, cache_v = as_cache(k), as_cache(v)
        page_table = ttnn.from_torch(
            torch.arange(blocks).reshape(1, blocks).int(),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_q = ttnn.from_torch(
            q,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        print(
            "DIAG2 "
            + f"{'grid':>6} {'kchunk':>7} {'exp':>7} {'maxcore':>7} {'/head':>5} "
            + " ".join(f"{c:>8}" for c in CONTEXTS)
        )
        for grid, kc, exp_approx, max_cores in COMBOS:
            row = []
            for ctx in CONTEXTS:
                want = torch.nn.functional.scaled_dot_product_attention(
                    q.permute(0, 2, 1, 3).double(),
                    k[:, :, :ctx].repeat_interleave(NH // NKV, 1).double(),
                    v[:, :, :ctx].repeat_interleave(NH // NKV, 1).double(),
                    scale=HD**-0.5,
                ).float()
                cur = ttnn.from_torch(
                    torch.tensor([ctx - 1], dtype=torch.int32),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                kwargs = {"compute_kernel_config": hifi}
                if grid is not None:
                    pc_kwargs = {}
                    if max_cores is not None:
                        pc_kwargs["max_cores_per_head_batch"] = max_cores
                    kwargs["program_config"] = ttnn.SDPAProgramConfig(
                        compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
                        q_chunk_size=32,
                        k_chunk_size=kc,
                        exp_approx_mode=exp_approx,
                        **pc_kwargs,
                    )
                try:
                    out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                        tt_q,
                        cache_k,
                        cache_v,
                        page_table_tensor=page_table,
                        cur_pos_tensor=cur,
                        scale=HD**-0.5,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        **kwargs,
                    )
                    got = ttnn.to_torch(out).float()[:, :, :NH, :].permute(0, 2, 1, 3)
                    row.append(f"{pcc(got, want):.4f}")
                    ttnn.deallocate(out)
                except Exception as exc:  # noqa: BLE001 - diagnostic reports, never raises
                    row.append(f"ERR:{type(exc).__name__}")
                ttnn.deallocate(cur)
            chunk_label = "dynamic" if kc == 0 else str(kc)
            grid_label = "op-dflt" if grid is None else f"{grid[0]}x{grid[1]}"
            exp_label = "op-dflt" if exp_approx is None else ("approx" if exp_approx else "exact")
            per_head = cores_per_head(grid, max_cores, grid_cores)
            mc_label = "avail" if grid is None else str(16 if max_cores is None else max_cores)
            print(
                f"DIAG2 {grid_label:>6} {chunk_label:>7} {exp_label:>7} {mc_label:>7} {per_head:>5} "
                + " ".join(f"{r:>8}" for r in row)
            )
        # ---- cost of the accuracy: time the op itself at the advertised context ----
        import time

        cur = ttnn.from_torch(
            torch.tensor([TIME_CONTEXT - 1], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        print(f"DIAG2 timing at context {TIME_CONTEXT}, {TIME_ITERS} warmed iterations")
        for label, max_cores in TIMED:
            kwargs = {"compute_kernel_config": hifi}
            if max_cores != "none":
                kwargs["program_config"] = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(dev_grid.x, dev_grid.y),
                    q_chunk_size=32,
                    k_chunk_size=0,
                    exp_approx_mode=False,
                    max_cores_per_head_batch=max_cores,
                )
            call = lambda: ttnn.transformer.paged_scaled_dot_product_attention_decode(  # noqa: E731
                tt_q,
                cache_k,
                cache_v,
                page_table_tensor=page_table,
                cur_pos_tensor=cur,
                scale=HD**-0.5,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                **kwargs,
            )
            ttnn.deallocate(call())  # compile
            ttnn.synchronize_device(device)
            start = time.perf_counter()
            for _ in range(TIME_ITERS):
                ttnn.deallocate(call())
            ttnn.synchronize_device(device)
            per_iter_ms = (time.perf_counter() - start) * 1000.0 / TIME_ITERS
            print(f"DIAG2 TIME {label:>22}  {per_iter_ms:8.3f} ms/call")
        ttnn.deallocate(cur)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
