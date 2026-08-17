# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Isolate ``ttnn.transformer.paged_scaled_dot_product_attention_decode`` at long context.

Why: ``test_longest_decode_context[full]`` is the only case in the stage materially below the
rest. ``diag_long_decode.py`` narrowed it to the attention branch and ruled out operand
quantisation (the device diverges from an *exact* bf16-operand reference too). This script strips
away everything else -- no projections, no RoPE, no output gate, no o_proj, no MoE -- and drives
the op directly with random K/V over a paged cache.

**What "no program config" actually means for the paged decode op.** The paged entry point does
not leave the config empty: ``paged_scaled_dot_product_attention_decode`` substitutes its own
default before the device op ever sees it (``sdpa_decode.cpp:122-129``)::

    if (!program_config.has_value()) {
        program_config = SDPAProgramConfig{
            device->compute_with_storage_grid_size(),   // grid
            std::nullopt,                               // sub_core_grids
            kDefaultDecodeChunkSize,                    // q_chunk_size = 32
            kDefaultDecodeChunkSize,                    // k_chunk_size = 32
            std::nullopt,                               // exp_approx_mode -> resolves to *true*
            kDefaultMaxCoresPerHeadBatch};              // max_cores_per_head_batch = 1

So ``program_config.has_value()`` is already true inside the factory, and the branch
``program_config.has_value() ? max_cores_per_head_batch : num_cores_available``
(``sdpa_decode_program_factory.cpp:192-193``) is **unreachable from this entry point**. The
struct default of 16 (``sdpa_config.hpp:18``) is unreachable too. Only the *non-paged*
``scaled_dot_product_attention_decode`` leaves the config empty and can reach 110 cores/head.

That makes ``max_cores_per_head_batch`` a two-value question here, not a wide axis: the op already
runs at **1 core per (slot, KV head)** by default, and the only reason to set it is to ask for
*more*. The axis that actually differs between "no config" and any config this stage wrote is
``k_chunk_size``: the op default is 32, while ``k_chunk_size=0`` selects the kernel's dynamic chunk
(``kernels/rt_args_common.hpp:96-107``, ``nearest_pow_of_2_up_to_8``). On this shape the dynamic path
resolves to **128 keys**, which the grid below shows rather than assumes -- the ``dynamic`` and
``128`` rows agree to the last digit at every context. At 262144 keys that is 8192 sequential
accumulation steps for the op default against 2048 for dynamic: a 4x difference in bf16 accumulation
depth on one core, in the direction that predicts both the accuracy and the latency gap.

So the sweep is a **2-D grid over (max_cores_per_head_batch, k_chunk_size)**, because those two
interact: the context at which more-than-one-core-per-head starts returning a silently wrong answer
moves with the chunk size. Sweeping cores at a single chunk size -- which an earlier version of this
file did -- cannot separate the two and is what made the previous conclusion wrong. The grid does
*not* identify the mechanism behind the wrong cells; see ``chunks_per_core`` for the one candidate
this script can already refute.

Row 0 is the **identity control**: no config versus an explicit config spelling out the substituted
default. It must be *bit*-identical, not merely close. If it is not, the model of the op above is
wrong and nothing else in the table can be attributed.

    python models/autoports/qwen_qwen3_6_35b_a3b/tests/diag_sdpa_decode.py
"""

import time

import torch

import ttnn

NH, NKV, HD, BLOCK = 16, 2, 256, 64
GROUP = NH // NKV
MAX_CONTEXT = 262144

#: contexts chosen to include tiny, tile/block boundaries, non-multiples and the advertised max
CONTEXTS = [1, 32, 64, 128, 257, 1024, 4096, 32768, 131072, 262143, 262144]

#: The op's substituted default, spelled out (``sdpa_decode.cpp:122-129``). ``exp_approx_mode`` is
#: ``None`` here, not ``False``, because that is what the substitution passes -- and the factory
#: resolves a missing ``exp_approx_mode`` to **true**, not false
#: (``sdpa_decode_program_factory.cpp:211-213``). An earlier version of this file used ``False`` and
#: claimed it was the same program; it is not, and the identity control has to be literally the
#: substituted config or it proves nothing. (The held-axis rows show approx and exact are
#: bit-identical for this shape, which is why the earlier control still came out identical.)
OP_DEFAULT = dict(k_chunk=32, exp=None, max_cores=1)

#: ``(k_chunk_size, max_cores_per_head_batch)``. ``k_chunk_size=0`` is the dynamic chunk path and is
#: legal for the paged *causal* op (only the non-causal branch requires > 0,
#: ``sdpa_decode_device_operation.cpp:321-327``).
K_CHUNKS = [0, 32, 64, 128, 256, 512, 1024, 2048]
MAX_CORES = [1, 2, 4, 8, 16, 55]

#: Held-axis checks, run once each at the op default's chunk size so they are comparable to it.
HELD = [
    ("grid 8x8", dict(grid=(8, 8), k_chunk=32, exp=False, max_cores=1)),
    ("exp approx", dict(k_chunk=32, exp=True, max_cores=1)),
    ("exp unset -> true", dict(k_chunk=32, exp=None, max_cores=1)),
    ("exp approx, 16 cores", dict(k_chunk=32, exp=True, max_cores=16)),
    ("max_cores 110", dict(k_chunk=32, exp=False, max_cores=110)),
]

#: Timed at ``TIME_CONTEXT``: the op default, the shipped-so-far setting, and every candidate the
#: accuracy table leaves viable. Accuracy is only half the decision.
TIMED = [
    ("op default (k32, 1 core)", dict(k_chunk=32, exp=False, max_cores=1)),
    ("no config", None),
    ("k0 dynamic, 1 core", dict(k_chunk=0, exp=False, max_cores=1)),
    ("k32, 4 cores", dict(k_chunk=32, exp=False, max_cores=4)),
    ("k32, 16 cores", dict(k_chunk=32, exp=False, max_cores=16)),
    ("k32, 55 cores", dict(k_chunk=32, exp=False, max_cores=55)),
    ("k256, 1 core", dict(k_chunk=256, exp=False, max_cores=1)),
    ("k512, 1 core", dict(k_chunk=512, exp=False, max_cores=1)),
    ("k1024, 1 core", dict(k_chunk=1024, exp=False, max_cores=1)),
    ("k2048, 1 core", dict(k_chunk=2048, exp=False, max_cores=1)),
    ("k256, 16 cores", dict(k_chunk=256, exp=False, max_cores=16)),
    ("k0 dynamic, 16 cores", dict(k_chunk=0, exp=False, max_cores=16)),
]
TIME_CONTEXT = 262144
TIME_ITERS = 20


def cores_per_head(max_cores, grid_cores, batch=1, num_kv_heads=NKV):
    """The factory's arithmetic (``sdpa_decode_program_factory.cpp:192-197``) for a config that is
    always present -- which, for this op, it always is."""
    uncapped = min(grid_cores, max_cores * batch * num_kv_heads) // batch
    return max(1, uncapped // num_kv_heads)


def chunks_per_core(ctx, k_chunk, per_head):
    """How many k-chunks each active core gets.

    Printed next to the PCC grid as raw data for whoever narrows the upstream bug, **not** as an
    explanation. "Fewer than one chunk per core" is refuted by this script's own grid: `k32` at
    2 cores/head and context 128 has 2.00 chunks per core and is still wrong (0.7250), while `k512`
    at 2 cores/head and context 257 has 0.50 and is fine (0.9998). Whatever the mechanism is, this
    ratio alone does not predict it.
    """
    if k_chunk == 0:  # dynamic; empirically 128 keys on this shape (see the grid)
        k_chunk = 128
    return -(-ctx // k_chunk) / per_head


def pcc(a, b):
    x = a.reshape(-1).double()
    y = b.reshape(-1).double()
    x = x - x.mean()
    y = y - y.mean()
    return float((x @ y) / (x.norm() * y.norm() + 1e-30))


def reference(q, k, v, ctx):
    """Exact fp64 GQA attention, one KV group at a time.

    Per group instead of ``repeat_interleave`` because the repeated fp64 K/V at 262144 keys is
    ~8.6 GB per tensor; per group it is 17 MB, which is what makes the full 2-D sweep affordable.
    """
    out = torch.empty(NH, HD, dtype=torch.float64)
    for h in range(NKV):
        qh = q[0, 0, h * GROUP : (h + 1) * GROUP].double()
        s = (qh @ k[0, h, :ctx].double().T) * (HD**-0.5)
        out[h * GROUP : (h + 1) * GROUP] = torch.softmax(s, dim=-1) @ v[0, h, :ctx].double()
    return out


def main():
    torch.set_num_threads(16)
    cur_of = {}
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
            q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        def cur_pos(ctx):
            if ctx not in cur_of:
                cur_of[ctx] = ttnn.from_torch(
                    torch.tensor([ctx - 1], dtype=torch.int32),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            return cur_of[ctx]

        def run_op(ctx, setting):
            """``setting is None`` means pass no program config at all. Returns the device tensor,
            so the timing loop can measure the op without a per-call host readback."""
            kwargs = {"compute_kernel_config": hifi}
            if setting is not None:
                kwargs["program_config"] = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(*setting.get("grid", (dev_grid.x, dev_grid.y))),
                    q_chunk_size=32,
                    k_chunk_size=setting["k_chunk"],
                    exp_approx_mode=setting["exp"],
                    max_cores_per_head_batch=setting["max_cores"],
                )
            return ttnn.transformer.paged_scaled_dot_product_attention_decode(
                tt_q,
                cache_k,
                cache_v,
                page_table_tensor=page_table,
                cur_pos_tensor=cur_pos(ctx),
                scale=HD**-0.5,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                **kwargs,
            )

        def run(ctx, setting):
            out = run_op(ctx, setting)
            got = ttnn.to_torch(out).float()[0, 0, :NH, :]
            ttnn.deallocate(out)
            return got

        # ---- row 0: the identity control. Must be bit-identical, not merely close. ----
        print("DIAG2 --- identity control: no config vs the substituted default (must be exact) ---")
        all_same = True
        for ctx in CONTEXTS:
            a, b = run(ctx, None), run(ctx, OP_DEFAULT)
            same = bool(torch.equal(a, b))
            all_same &= same
            print(
                f"DIAG2 IDENT ctx={ctx:>7}  bit_identical={str(same):>5}  "
                f"max_abs_diff={float((a - b).abs().max()):.3e}"
            )
        print(f"DIAG2 IDENT  all contexts bit-identical: {all_same}")

        # ---- the 2-D grid ----
        refs = {ctx: reference(q, k, v, ctx) for ctx in CONTEXTS}
        print()
        print("DIAG2 --- op PCC vs exact fp64, grid = (k_chunk_size) x (max_cores_per_head_batch) ---")
        print("DIAG2 " + f"{'kchunk':>7} {'maxcore':>7} {'/head':>5} " + " ".join(f"{c:>8}" for c in CONTEXTS))
        for kc in K_CHUNKS:
            for mc in MAX_CORES:
                row = []
                for ctx in CONTEXTS:
                    try:
                        row.append(f"{pcc(run(ctx, dict(k_chunk=kc, exp=False, max_cores=mc)), refs[ctx]):.4f}")
                    except Exception as exc:  # noqa: BLE001 - diagnostic reports, never raises
                        row.append(f"ERR:{type(exc).__name__}"[:8])
                per_head = cores_per_head(mc, grid_cores)
                print(
                    f"DIAG2 {'dynamic' if kc == 0 else kc:>7} {mc:>7} {per_head:>5} " + " ".join(f"{r:>8}" for r in row)
                )

        # ---- chunks per core, to show where the silent-wrong-answer boundary is ----
        print()
        print("DIAG2 --- k-chunks per active core (raw data for narrowing; does NOT predict the wrong cells) ---")
        print("DIAG2 " + f"{'kchunk':>7} {'maxcore':>7} " + " ".join(f"{c:>8}" for c in CONTEXTS))
        for kc in K_CHUNKS:
            for mc in MAX_CORES:
                per_head = cores_per_head(mc, grid_cores)
                print(
                    f"DIAG2 {'dynamic' if kc == 0 else kc:>7} {mc:>7} "
                    + " ".join(f"{chunks_per_core(c, kc, per_head):>8.2f}" for c in CONTEXTS)
                )

        # ---- held axes ----
        print()
        print("DIAG2 --- held-axis checks, all at the op default's k_chunk unless stated ---")
        for label, setting in HELD:
            row = []
            for ctx in CONTEXTS:
                try:
                    row.append(f"{pcc(run(ctx, setting), refs[ctx]):.4f}")
                except Exception as exc:  # noqa: BLE001
                    row.append(f"ERR:{type(exc).__name__}"[:8])
            print(f"DIAG2 HELD {label:>22} " + " ".join(f"{r:>8}" for r in row))

        # ---- cost ----
        print()
        print(f"DIAG2 timing at context {TIME_CONTEXT}, {TIME_ITERS} warmed iterations")
        for label, setting in TIMED:
            try:
                ttnn.deallocate(run_op(TIME_CONTEXT, setting))  # compile
                ttnn.synchronize_device(device)
                start = time.perf_counter()
                for _ in range(TIME_ITERS):
                    ttnn.deallocate(run_op(TIME_CONTEXT, setting))
                ttnn.synchronize_device(device)
                per_iter_ms = (time.perf_counter() - start) * 1000.0 / TIME_ITERS
                acc = pcc(run(TIME_CONTEXT, setting), refs[TIME_CONTEXT])
                print(f"DIAG2 TIME {label:>24}  {per_iter_ms:8.3f} ms/call   pcc {acc:.4f}")
            except Exception as exc:  # noqa: BLE001
                print(f"DIAG2 TIME {label:>24}  ERR:{type(exc).__name__}: {exc}")
    finally:
        for t in cur_of.values():
            ttnn.deallocate(t)
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
