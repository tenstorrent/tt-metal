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
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.experimental.diffusion_gemma.tt.denoise_forward import _fullcanvas_norm
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
    full = _fullcanvas_norm(SimpleNamespace(tt_weight=None, eps=_EPS), x_dev)
    assert full is not None, "the full-canvas config must be available at the shipped hidden size"

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
