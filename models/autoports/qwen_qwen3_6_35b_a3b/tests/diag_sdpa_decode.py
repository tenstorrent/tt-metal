# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Isolate ``ttnn.transformer.paged_scaled_dot_product_attention_decode`` at long context.

Why: ``test_longest_decode_context[full]`` is the only case in the stage materially below the
rest (PCC 0.9986 at position 262143 vs ~0.99999 everywhere else). ``diag_long_decode.py``
narrowed it to the attention branch and ruled out operand quantisation (the device diverges from
an *exact* bf16-operand reference too). This script strips away everything else — no
projections, no RoPE, no output gate, no o_proj, no MoE — and drives the op directly with random
K/V over a paged cache, sweeping context length x core grid x ``k_chunk_size``.

What it establishes:

* With ``k_chunk_size`` left unset the op takes its **dynamic** path
  (``get_dynamic_Sk_chunk_t`` in
  ``ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp``, which
  picks ``nearest_pow_of_2_up_to_8(seq_len_in_tiles)`` from ``cur_pos`` and carries an in-source
  "seeing PCC issues" caveat). That path is structurally correct at every position but loses
  accuracy as the context grows.
* Explicit configs trade the two failure modes against each other: larger grids are more
  accurate at long context but return structurally wrong results at particular short contexts
  (too few k-chunks to feed the cross-core reduction), and small grids are correct everywhere
  but both slower and less accurate at long context.
* Because ``cur_pos`` is a runtime *device* value, the layer cannot select a config per call
  without a host read, so the functional decoder keeps the dynamic path. See
  ``doc/functional_decoder/README.md`` section 3.8.

    python models/autoports/qwen_qwen3_6_35b_a3b/tests/diag_sdpa_decode.py
"""

import torch

import ttnn

NH, NKV, HD, BLOCK = 16, 2, 256, 64
MAX_CONTEXT = 262144

#: contexts chosen to include tiny, tile/block boundaries, non-multiples and the advertised max
CONTEXTS = [1, 32, 64, 128, 257, 1024, 4096, 32768, 131072, 262143, 262144]
#: (grid, k_chunk_size); k_chunk_size 0 means "leave unset" -> the op's dynamic path
COMBOS = [
    ((8, 8), 0),
    ((1, 1), 128),
    ((2, 1), 128),
    ((4, 1), 128),
    ((8, 1), 64),
    ((8, 1), 128),
    ((8, 8), 128),
    ((8, 8), 512),
]


def pcc(a, b):
    x = a.reshape(-1).double()
    y = b.reshape(-1).double()
    x = x - x.mean()
    y = y - y.mean()
    return float((x @ y) / (x.norm() * y.norm() + 1e-30))


def main():
    torch.set_num_threads(16)
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
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

        print("DIAG2 " + f"{'grid':>5} {'kchunk':>7} " + " ".join(f"{c:>8}" for c in CONTEXTS))
        for grid, kc in COMBOS:
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
                if kc:
                    kwargs["program_config"] = ttnn.SDPAProgramConfig(
                        compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
                        q_chunk_size=32,
                        k_chunk_size=kc,
                        exp_approx_mode=False,
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
            label = "dynamic" if kc == 0 else str(kc)
            print(f"DIAG2 {grid[0]}x{grid[1]:<3} {label:>7} " + " ".join(f"{r:>8}" for r in row))
        print(
            "DIAG2 note: the same 262144-key attention computed by the *prefill* op "
            "(chunked_scaled_dot_product_attention, explicit q=k=128 config) scores 0.9999891 "
            "at the layer level, so this is specific to the decode op's dynamic chunk path."
        )
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
