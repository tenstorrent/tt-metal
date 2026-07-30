# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What DG_NORM_FULLCANVAS actually changes, measured — the number the tree never had.

`l1_residency.md` and `norm_fullcanvas_flip_gate.md` both attribute the flag's non-bit-identity to
"a ~2e-6 bf16 reduction/accumulation-ORDER difference between block_h=8 and 8x block_h=1", citing
PCC 0.999998 from `doc/optimize_perf/bench_norm_fullcanvas.py`. Two problems with that number:

* that bench computes PCC as an fp32 dot product over ~720K elements and reports values ABOVE 1.0
  elsewhere in the same table (1.000015, 1.000050), which a Pearson correlation cannot be — so its
  resolution floor is ~5e-5, i.e. 25x coarser than the 2e-6 it is cited as proving;
* a static reading of the sharded-LayerNorm kernel says `block_h` feeds only CB sizes, loop trip
  counts and `num_rows_per_all_to_all_worker` (1 for both 1 and 8) and does not change `num_blocks`,
  `winv` or `cinv` -- from which one predicts the weighted norms are BIT-IDENTICAL.

MEASURED 2026-07-30 on QB2, and the prediction is WRONG. Both paths differ, at a few bf16 ULPs:

    weighted (block_h=8 vs 8x block_h=1, same 88-core grid)
        61/256 rows, 19.43% of elements; rel p50 0, p99 1.14e-2, max 2.24e-2; ULP p99 2.91 max 5.73
    scaleless (8-core/block_w=11 vs 88-core/block_w=1)
        79/256 rows, 24.80% of elements; rel p50 0, p99 1.06e-2, max 1.56e-2; ULP p99 2.71 max 4.00

So the documented "~2e-6" understates the real delta by about FOUR ORDERS OF MAGNITUDE, and the
block_h mechanism it names is real after all -- the static argument above does not survive contact
with the kernel. Note the weighted delta is the LARGER of the two, so correcting the scaleless
dispatch (which stopped DG_NORM_FULLCANVAS re-sharding the MoE router's norm) does not make the flag
bit-identical and was never going to. A few-ULP perturbation in every weighted norm, compounded over
30 layers x 16-48 steps through the accept/renoise loop, is a sufficient mechanism for the ~85%
committed-token divergence `norm_fullcanvas_flip_gate.md` measured.

This test asserts only that both paths really are an RMSNorm of the input; the deltas are REPORTED,
not gated, because their value is the record, not a pass/fail. Re-run it if either path changes.

It measures at the SHIPPED hidden size. 2816 is not decoration: the
core-grid search picks 88 cores (11x8) for the full-canvas path and 8 cores (8x1) for the scaleless
path *because* 2816/32 = 88, and at any other width the topologies under test would not be the ones
that ship.

Run on QB2::

    DG_RUN_DEVICE=1 pytest models/experimental/diffusion_gemma/tests/test_device_norm_fullcanvas.py -s
"""

import os

import pytest
import torch

import ttnn
from models.experimental.diffusion_gemma.tt.denoise_forward import _build_fullcanvas_norm_cfg
from models.experimental.diffusion_gemma.tt.self_conditioning import _rms_norm_dram

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("DG_RUN_DEVICE") != "1",
        reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device",
    ),
    pytest.mark.use_module_device,  # one device open/teardown — avoid QB2 erisc cycling
]

_H = 2816  # the shipped hidden size; 2816/32 = 88 is what selects both grids under test
_S = 256  # the shipped canvas
_EPS = 1e-6


def _to_device(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def _report(name, a, b):
    """Print and return a characterisation of the difference between two torch tensors.

    max_rel alone is a bad summary here: it is maximised by elements where the denominator is near
    zero, so it overstates. Reported alongside are the percentiles over elements that are not tiny,
    and the distance in bf16 ULPs -- bf16 has an 8-bit mantissa, so 1 ULP is ~3.9e-3 relative and
    "a few ULPs" is the floor for any two different reduction trees.
    """
    same = torch.equal(a, b)
    af, bf = a.float(), b.float()
    diff = (af - bf).abs()
    scale = bf.abs()
    big = scale > scale.max() * 1e-3  # ignore near-zero denominators
    rel = (diff[big] / scale[big]) if big.any() else torch.zeros(1)
    ulp = rel / (2.0**-8)  # bf16 mantissa
    elems_differing = int((diff > 0).sum())
    rows_differing = int((diff.amax(dim=-1) > 0).sum())
    print(
        f"\n[{name}]\n"
        f"  bit_identical = {same}\n"
        f"  rows differing = {rows_differing}/{a.shape[-2]}   elements differing = "
        f"{elems_differing}/{a.numel()} ({100.0*elems_differing/a.numel():.2f}%)\n"
        f"  max_abs = {float(diff.max()):.3e}\n"
        f"  relative (over non-tiny elements): p50={float(rel.median()):.3e} "
        f"p99={float(rel.quantile(0.99)):.3e} max={float(rel.max()):.3e}\n"
        f"  in bf16 ULPs: p99={float(ulp.quantile(0.99)):.2f}  max={float(ulp.max()):.2f}"
    )
    return same, float(diff.max()), float(rel.max())


def test_scaleless_norm_topology_delta_is_what_the_flag_used_to_inject(device):
    """The router-norm hijack: _rms_norm_dram (8 cores, block_w=11) vs _fullcanvas_norm (88, 1).

    Until the dispatch order was corrected, DG_NORM_FULLCANVAS=1 sent the MoE router's weightless
    norm down the second path. These are structurally different summation trees, so this is the one
    place the flag genuinely changed a reduction. Measured here so the magnitude is on record rather
    than asserted — the test does not require them to agree, it requires the number to be known.
    """
    torch.manual_seed(0)
    x = torch.randn(1, 1, _S, _H)
    x_dev = _to_device(x, device)

    chunked = _rms_norm_dram(x_dev, weight=None, epsilon=_EPS, chunk_size=32)
    # The 88-core/block_w=1 arm, built the way the flag builds it for a weightless norm. Built
    # explicitly rather than through _fullcanvas_norm so this test keeps measuring the TOPOLOGY pair
    # even if the dispatch around it changes.
    memcfg, pc = _build_fullcanvas_norm_cfg(device, _S, _H)
    assert memcfg is not None, "the full-canvas config must be available at the shipped hidden size"
    x_sh = ttnn.to_memory_config(x_dev, memcfg)
    full_sh = ttnn.rms_norm(x_sh, weight=None, epsilon=_EPS, program_config=pc, memory_config=memcfg)
    full = ttnn.sharded_to_interleaved(full_sh, ttnn.DRAM_MEMORY_CONFIG)
    x_sh.deallocate(True)
    full_sh.deallocate(True)

    a = ttnn.to_torch(chunked)
    b = ttnn.to_torch(full)
    same, max_abs, max_rel = _report("scaleless: 8-core/block_w=11  vs  88-core/block_w=1", a, b)

    # Sanity: both must be a real RMSNorm of the same input, not garbage.
    ref = x / torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + _EPS)
    for label, got in (("chunked", a), ("fullcanvas", b)):
        rel = ((got.float() - ref).abs() / ref.abs().clamp_min(1e-30)).max()
        assert rel < 5e-2, f"{label} is not an RMSNorm of the input (max rel {rel:.3e})"

    print(f"[scaleless] VERDICT: the corrected dispatch keeps this path on the 8-core topology; "
          f"the flag used to swap it, worth {max_rel:.3e} relative.")


def test_weighted_norm_block_h_8_vs_1_is_the_documented_mechanism(device):
    """block_h=8 (one 256-row call) vs block_h=1 (8x 32-row calls), SAME 88-core grid.

    This is the mechanism l1_residency.md blames for the flag's decision changes. Both paths are
    built here from the same shipped config builders, so the only difference is block_h and the
    surrounding slice/concat. If this comes out bit-identical, the documented "~2e-6 reduction-order"
    story has no mechanism left and the flag's real delta was the scaleless hijack above.
    """
    from models.experimental.diffusion_gemma.tt.denoise_forward import _build_fullcanvas_norm_cfg

    torch.manual_seed(1)
    x = torch.randn(1, 1, _S, _H)
    w = torch.randn(1, 1, 1, _H) * 0.1 + 1.0
    x_dev = _to_device(x, device)
    w_dev = ttnn.from_torch(w.reshape(1, 1, -1, ttnn.TILE_SIZE), dtype=ttnn.bfloat16,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    cfg = _build_fullcanvas_norm_cfg(device, _S, _H)
    assert cfg is not None
    full_memcfg, full_pc = cfg
    x_sh = ttnn.to_memory_config(x_dev, full_memcfg)
    full_sh = ttnn.rms_norm(x_sh, weight=w_dev, epsilon=_EPS, program_config=full_pc, memory_config=full_memcfg)
    full = ttnn.sharded_to_interleaved(full_sh, ttnn.DRAM_MEMORY_CONFIG)

    # The chunked arm: the same builder at 32 rows is exactly gemma4 RMSNorm._build_sharded_cfg's
    # choice (same grid search, block_h=1), which is the config norm.forward() takes for a 32-row
    # slice -- so this reproduces the default path without constructing a gemma4 module.
    chunk_cfg = _build_fullcanvas_norm_cfg(device, ttnn.TILE_SIZE, _H)
    assert chunk_cfg is not None
    chunk_memcfg, chunk_pc = chunk_cfg
    parts = []
    for start in range(0, _S, ttnn.TILE_SIZE):
        sl = ttnn.slice(x_dev, [0, 0, start, 0], [1, 1, start + ttnn.TILE_SIZE, _H],
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
        sl_sh = ttnn.to_memory_config(sl, chunk_memcfg)
        out_sh = ttnn.rms_norm(sl_sh, weight=w_dev, epsilon=_EPS, program_config=chunk_pc,
                               memory_config=chunk_memcfg)
        parts.append(ttnn.sharded_to_interleaved(out_sh, ttnn.DRAM_MEMORY_CONFIG))
        sl.deallocate(True)
        sl_sh.deallocate(True)
        out_sh.deallocate(True)
    chunked = ttnn.concat(parts, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    a = ttnn.to_torch(chunked)
    b = ttnn.to_torch(full)
    same, max_abs, max_rel = _report("weighted: block_h=1 x8  vs  block_h=8 (same 88-core grid)", a, b)
    print(f"[weighted] VERDICT: {'BIT-IDENTICAL -> the documented block_h mechanism does not exist' if same else f'differs by {max_rel:.3e} relative'}")


@pytest.mark.parametrize(
    "label, fp32, approx",
    [
        ("default (fp32_acc=False, approx=True)", False, True),
        ("fp32_dest_acc_en=True", True, True),
        ("fp32_dest_acc_en=True + approx off", True, False),
    ],
)
def test_can_fp32_accumulation_make_the_two_paths_agree(device, label, fp32, approx):
    """Is the flag FIXABLE? Raise the reduction precision and see if the paths converge.

    `rmsnorm_default_compute_config` (ttnn/cpp/.../rmsnorm.cpp:16-20) is HiFi4 with
    `approx_mode=true, fp32_acc=FALSE`, so every one of the 88 per-core partials is rounded to bf16
    before the cross-core combine. That is what makes the summation TREE matter: block_h moves
    `num_cores_all_to_all` (8 -> 64), regrouping the combine, and in bf16 that regrouping is worth
    the few ULPs measured above.

    If the partials accumulate in fp32 the regrouping should stop mattering to ~2^-24 and the two
    paths should agree once the result is rounded back to bf16. If this comes out bit-identical,
    DG_NORM_FULLCANVAS becomes a free win like the bounded sliding read; if it does not, the flag is
    inherently decision-changing and only an absolute HF-vs-TT gate can clear it.

    No DG caller passes a compute_kernel_config to any norm today, so this costs nothing to try.
    """
    from models.experimental.diffusion_gemma.tt.denoise_forward import _build_fullcanvas_norm_cfg

    torch.manual_seed(1)
    x = torch.randn(1, 1, _S, _H)
    w = torch.randn(1, 1, 1, _H) * 0.1 + 1.0
    x_dev = _to_device(x, device)
    w_dev = ttnn.from_torch(w.reshape(1, 1, -1, ttnn.TILE_SIZE), dtype=ttnn.bfloat16,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    ckcfg = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=approx,
        fp32_dest_acc_en=fp32,
        packer_l1_acc=False,
    )

    def run(seq_rows):
        memcfg, pc = _build_fullcanvas_norm_cfg(device, seq_rows, _H)
        outs = []
        for start in range(0, _S, seq_rows):
            sl = ttnn.slice(x_dev, [0, 0, start, 0], [1, 1, start + seq_rows, _H],
                            memory_config=ttnn.DRAM_MEMORY_CONFIG) if seq_rows != _S else x_dev
            sh = ttnn.to_memory_config(sl, memcfg)
            o = ttnn.rms_norm(sh, weight=w_dev, epsilon=_EPS, program_config=pc,
                              memory_config=memcfg, compute_kernel_config=ckcfg)
            outs.append(ttnn.sharded_to_interleaved(o, ttnn.DRAM_MEMORY_CONFIG))
            sh.deallocate(True)
            o.deallocate(True)
            if sl is not x_dev:
                sl.deallocate(True)
        return outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    full = ttnn.to_torch(run(_S))
    chunked = ttnn.to_torch(run(ttnn.TILE_SIZE))
    same, _, max_rel = _report(f"FIX PROBE [{label}]", chunked, full)
    print(f"[fix probe] {label}: {'BIT-IDENTICAL -- the flag is fixable this way' if same else f'still differs, max_rel {max_rel:.3e}'}")


def test_what_fp32_accumulation_actually_costs(device):
    """Price the bit-identity fix. "fp32 is slow" is a general heuristic; this config may not pay it.

    fp32_dest_acc_en halves DST capacity, which hurts when the per-core output block is large. Here
    it is not: the full-canvas config lands on block_w=1 / subblock_w=1 (88 cores over 88 tile-cols),
    so there is little DST pressure to lose. Measured rather than assumed -- the alternative to a
    bit-identical fix is an absolute HF-vs-TT gate, which costs far more than any per-norm overhead.
    """
    import time
    from models.experimental.diffusion_gemma.tt.denoise_forward import _build_fullcanvas_norm_cfg

    torch.manual_seed(2)
    x = torch.randn(1, 1, _S, _H)
    w = torch.randn(1, 1, 1, _H) * 0.1 + 1.0
    x_dev = _to_device(x, device)
    w_dev = ttnn.from_torch(w.reshape(1, 1, -1, ttnn.TILE_SIZE), dtype=ttnn.bfloat16,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    fp32_cfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True, fp32_dest_acc_en=True, packer_l1_acc=False)

    def build(seq_rows):
        memcfg, pc = _build_fullcanvas_norm_cfg(device, seq_rows, _H)
        return memcfg, pc

    def timed(seq_rows, ckcfg, iters=50):
        memcfg, pc = build(seq_rows)
        def once():
            outs = []
            for start in range(0, _S, seq_rows):
                sl = ttnn.slice(x_dev, [0, 0, start, 0], [1, 1, start + seq_rows, _H],
                                memory_config=ttnn.DRAM_MEMORY_CONFIG) if seq_rows != _S else x_dev
                sh = ttnn.to_memory_config(sl, memcfg)
                o = ttnn.rms_norm(sh, weight=w_dev, epsilon=_EPS, program_config=pc,
                                  memory_config=memcfg, compute_kernel_config=ckcfg)
                outs.append(ttnn.sharded_to_interleaved(o, ttnn.DRAM_MEMORY_CONFIG))
                sh.deallocate(True)
                o.deallocate(True)
                if sl is not x_dev:
                    sl.deallocate(True)
            out = outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            out.deallocate(True)
        once()  # warm the program cache
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            once()
        ttnn.synchronize_device(device)
        return (time.perf_counter() - t0) / iters * 1e3  # ms per 256-row norm

    chunk_bf16 = timed(ttnn.TILE_SIZE, None)
    full_bf16 = timed(_S, None)
    full_fp32 = timed(_S, fp32_cfg)
    chunk_fp32 = timed(ttnn.TILE_SIZE, fp32_cfg)

    print(f"\n[cost] per 256-row norm, mean of 50 (includes slice/concat/sharding, i.e. what the model pays)")
    print(f"  chunked 8x32, bf16 acc (TODAY'S DEFAULT) : {chunk_bf16:8.3f} ms")
    print(f"  full-canvas,  bf16 acc (the flag)        : {full_bf16:8.3f} ms   {chunk_bf16/full_bf16:5.2f}x vs default")
    print(f"  full-canvas,  fp32 acc (BIT-IDENTICAL)   : {full_fp32:8.3f} ms   {chunk_bf16/full_fp32:5.2f}x vs default"
          f"   fp32 costs {100*(full_fp32/full_bf16-1):+.1f}% vs bf16 full-canvas")
    print(f"  chunked 8x32, fp32 acc                   : {chunk_fp32:8.3f} ms")
    print(f"[cost] VERDICT: the bit-identical option is "
          f"{chunk_bf16/full_fp32:.2f}x faster than today's default." if full_fp32 < chunk_bf16 else
          f"[cost] VERDICT: fp32 full-canvas is SLOWER than today's default -- the fix is not viable this way.")


def test_is_fp32_accumulation_more_ACCURATE_or_merely_different(device):
    """The crux for adopting fp32 accumulation: closer to the truth, or just a different bf16 point?

    Bit-identity between full-canvas+fp32 and chunked+fp32 does NOT make a flip free on its own,
    because today's shipped output is chunked+BF16. Adopting fp32 changes the norm once. That is only
    worth doing if fp32 is more ACCURATE, not merely different -- so measure all three against an
    fp32 torch reference of the same RMSNorm.
    """
    from models.experimental.diffusion_gemma.tt.denoise_forward import _build_fullcanvas_norm_cfg

    torch.manual_seed(3)
    x = torch.randn(1, 1, _S, _H)
    w = torch.randn(1, 1, 1, _H) * 0.1 + 1.0
    x_dev = _to_device(x, device)
    w_dev = ttnn.from_torch(w.reshape(1, 1, -1, ttnn.TILE_SIZE), dtype=ttnn.bfloat16,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    fp32_cfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True, fp32_dest_acc_en=True, packer_l1_acc=False)

    # Reference from the SAME bf16 inputs the device sees, accumulated in fp64 -- so the only thing
    # being measured is the device's accumulation, not the input quantisation.
    xq = ttnn.to_torch(x_dev).double()
    wq = ttnn.to_torch(w_dev).reshape(1, 1, 1, _H).double()
    ref = xq / torch.sqrt(xq.pow(2).mean(-1, keepdim=True) + _EPS) * wq

    def run(seq_rows, ckcfg):
        memcfg, pc = _build_fullcanvas_norm_cfg(device, seq_rows, _H)
        outs = []
        for start in range(0, _S, seq_rows):
            sl = ttnn.slice(x_dev, [0, 0, start, 0], [1, 1, start + seq_rows, _H],
                            memory_config=ttnn.DRAM_MEMORY_CONFIG) if seq_rows != _S else x_dev
            sh = ttnn.to_memory_config(sl, memcfg)
            o = ttnn.rms_norm(sh, weight=w_dev, epsilon=_EPS, program_config=pc,
                              memory_config=memcfg, compute_kernel_config=ckcfg)
            outs.append(ttnn.sharded_to_interleaved(o, ttnn.DRAM_MEMORY_CONFIG))
            sh.deallocate(True); o.deallocate(True)
            if sl is not x_dev:
                sl.deallocate(True)
        return ttnn.to_torch(outs[0] if len(outs) == 1 else
                             ttnn.concat(outs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)).double()

    arms = {
        "chunked bf16 (TODAY)": run(ttnn.TILE_SIZE, None),
        "chunked fp32": run(ttnn.TILE_SIZE, fp32_cfg),
        "fullcanvas bf16": run(_S, None),
        "fullcanvas fp32": run(_S, fp32_cfg),
    }
    print("\n[accuracy] error vs an fp64 reference over the SAME bf16 inputs:")
    denom = ref.abs().clamp_min(1e-30)
    for name, got in arms.items():
        rel = ((got - ref).abs() / denom)
        print(f"  {name:<22} rel p50={float(rel.median()):.3e}  p99={float(rel.quantile(0.99)):.3e}  "
              f"max={float(rel.max()):.3e}  rmse={float((got-ref).pow(2).mean().sqrt()):.3e}")
    print("[accuracy] VERDICT: fp32 accumulation is an IMPROVEMENT if its p99/rmse are lower than "
          "chunked bf16's; if they are equal, adopting it is a lateral move and needs its own gate.")
