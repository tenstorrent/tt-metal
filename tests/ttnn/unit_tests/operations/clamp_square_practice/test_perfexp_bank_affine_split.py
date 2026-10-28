# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""bank_affine_split bake-off: op baseline vs bank-affine page ownership.

test_correct : bit-exact (torch.equal) candidate vs the op's own baseline output, and baseline vs
               torch, over ragged / tiny / non-multiple-of-8 page counts. Run WITHOUT --profile.
test_perf    : one generic_op per (variant, shape). Perf only, never asserts a direction.
               Filters: BAS_VARIANTS=baseline,col_near  BAS_SHAPES=1x1x4096x4096
"""
import os

import pytest
import torch
import ttnn

from ttnn.operations.clamp_square_practice.perf_experiments.bank_affine_split import variant as V

LO, HI = -1.0, 1.0


def _io(device, shape, seed=0):
    torch.manual_seed(seed)
    t = torch.randn(shape, dtype=torch.bfloat16) * 2
    x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def alloc():
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )

    return t, x, alloc


def _run(x, out, desc, device):
    ttnn.generic_op([x, out], desc)
    ttnn.synchronize_device(device)
    return ttnn.to_torch(out)


CORRECT_SHAPES = [
    (1, 1, 32, 32),  # 1 tile
    (1, 1, 32, 96),  # 3 tiles (< num banks)
    (1, 1, 64, 160),  # 10 tiles (not multiple of 8)
    (3, 1, 96, 160),  # 45 tiles
    (1, 1, 32, 121 * 32),  # 121 tiles (> cores, ragged g2)
    (1, 1, 256, 256),
    (2, 4, 256, 512),
    (1, 1, 1000, 1000),  # padded H/W, 1024 tiles
    (8, 1, 1024, 1024),
    (1, 1, 4096, 4096),
]


@pytest.mark.parametrize("mode", ["bank_near", "pair_near", "col_near", "all_near", "col_far", "bank_nearest"])
@pytest.mark.parametrize("shape", CORRECT_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_correct(device, shape, mode):
    t, x, alloc = _io(device, shape)
    n = x.buffer_num_pages()
    if (
        mode.endswith("nearest")
        and n < device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y
    ):
        pytest.skip("nearest needs n_pages >= num cores")
    base_out = alloc()
    base = _run(x, base_out, V.build_baseline(x, base_out, LO, HI), device)
    ref = (torch.clamp(t.float(), LO, HI) ** 2).bfloat16()
    assert torch.equal(base, ref)
    cand_out = alloc()
    cand = _run(x, cand_out, V.build(x, cand_out, LO, HI, mode=mode), device)
    assert torch.equal(cand, base), f"mismatch {(cand != base).sum().item()} elems"


PERF_SHAPES = [
    (1, 1, 4096, 4096),
    (8, 1, 1024, 1024),
    (2, 4, 256, 512),
    (1, 1, 256, 256),
    (1, 1, 32, 32),
    (1, 1, 32, 8192),
    (64, 128),
]
PERF_VARIANTS = [
    "baseline",
    "all_near",
    "bank_near",
    "pair_near",
    "col_near",
    "col_far",
    "col_nearrd",
    "col_nearest",
    "baseline_dm",
    "col_near_dm",
    "pair_near_dm",
    # "_again": same program re-launched (program-cache hit) -> isolates the first-launch effect
    "baseline_again",
    "col_near_again",
    "baseline_dm_again",
    "col_near_dm_again",
]


@pytest.mark.parametrize("shape", PERF_SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("variant", PERF_VARIANTS)
def test_perf(device, shape, variant):
    vs, ss = os.environ.get("BAS_VARIANTS"), os.environ.get("BAS_SHAPES")
    if (vs and variant not in vs.split(",")) or (ss and "x".join(map(str, shape)) not in ss.split(",")):
        pytest.skip("filtered")
    _, x, alloc = _io(device, shape)
    if variant.endswith("nearest") and x.buffer_num_pages() < 120:
        pytest.skip("nearest needs n_pages >= num cores")
    out = alloc()
    v = variant.removesuffix("_again")
    dm = v.endswith("_dm")
    v = v.removesuffix("_dm")
    desc = V.build_baseline(x, out, LO, HI, dm) if v == "baseline" else V.build(x, out, LO, HI, mode=v, skip_compute=dm)
    ttnn.generic_op([x, out], desc)
    ttnn.synchronize_device(device)
    print(f"BENCH_ROW variant={variant} shape={shape}")


def _graduated_pd():
    import importlib.util
    from pathlib import Path

    p = Path(V.__file__).parent / "graduate" / "clamp_square_practice_program_descriptor.py"
    spec = importlib.util.spec_from_file_location("cs_graduated_pd", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize("shape", CORRECT_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_graduate(device, shape):
    """The drop-in descriptor+kernels (graduate/) are bit-exact vs the op and build the SAME
    RT args as the measured col_near variant (so the measured numbers apply to it)."""
    gpd = _graduated_pd()
    t, x, alloc = _io(device, shape)
    base_out = alloc()
    base = _run(x, base_out, V.build_baseline(x, base_out, LO, HI), device)
    out = alloc()
    desc = gpd.create_program_descriptor(x, out, LO, HI)
    ref_desc = V.build(x, out, LO, HI, mode="col_near")
    for k_new, k_ref in zip(desc.kernels[:2], ref_desc.kernels[:2]):
        assert list(k_new.compile_time_args) == list(k_ref.compile_time_args)
        for c in ttnn.corerange_to_cores(k_new.core_ranges, row_wise=True):
            assert list(k_new.runtime_args[c.x][c.y]) == list(k_ref.runtime_args[c.x][c.y]), (c.x, c.y)
    cand = _run(x, out, desc, device)
    assert torch.equal(cand, base)
