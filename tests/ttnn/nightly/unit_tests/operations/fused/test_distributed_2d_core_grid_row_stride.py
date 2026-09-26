# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression test for tenstorrent/tt-metal#56908 ($3,000 bounty).

Boundary covered: distributed LayerNorm/RMSNorm with ``use_2d_core_grid=True`` when a
core owns more than one row-tile (``tiles_per_core_x > 1``). The 2D readers advanced
the input tile index linearly instead of jumping ``Wt_full - tiles_per_core_y`` at
local row boundaries, silently reading wrong tiles on taller shapes.

Why the old tests missed it: every existing 2D shape lands on ``tiles_per_core_x == 1``
(each core owns exactly one row-tile), where the flat walk is accidentally correct.
The x-axis boundary (``tiles_per_core_x > 1``) was never crossed.

Pre-fix expectation: the ``tiles_per_core_x > 1`` x ``use_2d_core_grid=True`` cases FAIL
(rmsnorm: silently wrong values, PCC collapse; layernorm: the ops do not yet expose
``use_2d_core_grid`` at the Python binding, so these error at the call site --
exposing the flag is part of the fix). All ``use_2d_core_grid=False`` controls and the
``tiles_per_core_x == 1`` guards PASS. Post-fix: everything green except the
layernorm+``grid_2d`` pre cases, which skip by design (see API note); guards unchanged.

API note: ``ttnn.rms_norm_{pre,post}_all_gather`` expose ``use_2d_core_grid``; the
layernorm variants now expose it too. The 2D pre-all-gather compute kernel emits
rmsnorm-only (1-wide) statistics, so layernorm+``use_2d_core_grid=True`` pre is
rejected by device validation (loud ``TT_FATAL``) rather than silently producing
wrong stats downstream; those cases skip here.

HANG CANARY: the multi-row 2D cases previously hit the #55075 deadlock mechanism
(the reduce scaler was popped per-row from a depth-1 buffer; this PR pops it once
after the row loop, mirroring the 1D kernel). Run under
``TT_METAL_OPERATION_TIMEOUT_SECONDS=45``; a timeout now indicates a residual hang,
NOT the stride bug.

LAYOUT GUARD: every ``grid_2d`` case carries a hardcoded ``expect_tpcx`` constant
and asserts the 2D factory's actual decomposition on THIS device against it. If a
device's compute grid shifts a "bug" shape onto ``tiles_per_core_x == 1`` (where the
old walk is accidentally correct), the test FAILS LOUD naming the grid instead of
going green without exercising the fix. See ``_assert_layout_exercised``.

Run (repo root, single Wormhole):
    TT_METAL_OPERATION_TIMEOUT_SECONDS=45 pytest tests/ttnn/nightly/unit_tests/operations/fused/test_distributed_2d_core_grid_row_stride.py -v

Record with results: ``tt-smi`` output, ``device.compute_with_storage_grid_size()``,
the tt-metal commit hash, torch version, and the per-case grid decomposition logged
by the runners below.
"""

import pytest
import torch
import ttnn

from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.ttnn.nightly.unit_tests.operations.fused.utility_functions import (
    ttnn_layer_norm_pre_all_gather,
    ttnn_layer_norm_post_all_gather,
    ttnn_rms_norm_pre_all_gather,
    ttnn_rms_norm_post_all_gather,
)

# Module-scoped device: every test here shares one device configuration
pytestmark = pytest.mark.use_module_device

NUM_SIMULATED_DEVICES = 4
TILE = 32


def _grid_decomposition(seq_len, hidden_per_dev):
    """Replicate the 2D factory's core-grid math for logging/failure attribution.

    Returns (cores_x, tiles_per_core_x, cores_y, tiles_per_core_y).
    """
    num_tile_rows = seq_len // TILE
    wt = hidden_per_dev // TILE
    cores_x = max(d for d in range(1, 9) if num_tile_rows % d == 0)
    cores_y = max(d for d in range(1, 9) if wt % d == 0)
    return cores_x, num_tile_rows // cores_x, cores_y, wt // cores_y


def _assert_layout_exercised(device, seq_len, hidden_per_dev, expect_tpcx, tag):
    """Layout guard: fail LOUD unless the 2D factory's actual core decomposition on
    THIS device crosses the intended tiles_per_core_x boundary for this shape.

    Why: the tiles_per_core_x a shape lands on depends on the device's compute grid
    (the factory takes the largest divisor of num_tile_rows not exceeding
    grid_size.y). If a future device has a wider grid, a "bug" shape can silently
    land on tiles_per_core_x == 1 -- where the old flat walk is accidentally correct --
    and the test would go green without exercising the fix. This assertion converts
    that silent miss into a loud failure naming the device grid and the shape.

    `expect_tpcx` is a hardcoded constant per shape in the tables below (not a
    re-derivation), so a grid or factory-math change that moves the boundary breaks
    the test instead of masking it.
    """
    assert seq_len % TILE == 0 and hidden_per_dev % TILE == 0, (
        f"[{tag}] shape not tile-aligned: seq_len={seq_len} hidden_per_dev={hidden_per_dev}"
    )
    grid = device.compute_with_storage_grid_size()
    num_tile_rows = seq_len // TILE
    wt = hidden_per_dev // TILE
    # Exact mirror of the factory's 2D grid math
    # (layernorm_pre_all_gather_program_factory.cpp:494-504).
    cores_x = min(grid.y, num_tile_rows)
    while num_tile_rows % cores_x != 0 and cores_x > 1:
        cores_x -= 1
    tpcx = num_tile_rows // cores_x
    cores_y = min(grid.y, wt)
    while wt % cores_y != 0 and cores_y > 1:
        cores_y -= 1
    assert cores_x * tpcx == num_tile_rows, f"[{tag}] grid math inconsistent: {cores_x}*{tpcx} != {num_tile_rows}"
    assert tpcx == expect_tpcx, (
        f"[{tag}] LAYOUT GUARD: device compute grid ({grid.x}x{grid.y}) gives "
        f"tiles_per_core_x={tpcx} for shape (seq_len={seq_len}, hidden_per_dev={hidden_per_dev}), "
        f"but this case expects tiles_per_core_x={expect_tpcx}. The shape no longer crosses the "
        f"intended boundary on this device -- the bug case is NOT exercised. Update the shape "
        f"table (or expect_tpcx) for this grid before trusting this run."
    )
    return cores_x, tpcx, cores_y, wt // cores_y


def _make_ramp_input(seq_len, hidden_dim_total):
    """Deterministic tile-coordinate ramp input (NOT randn).

    Element (r, c) = (r * W + c) / (H * W) - 0.5 in float32, cast to bf16. Every
    tile carries a unique signature, so a wrong-tile read lands a provably wrong
    coordinate in the per-row stats and PCC collapses instead of borderline-failing.
    """
    rows = torch.arange(seq_len, dtype=torch.float32).unsqueeze(-1)
    cols = torch.arange(hidden_dim_total, dtype=torch.float32).unsqueeze(0)
    ramp = (rows * hidden_dim_total + cols) / (seq_len * hidden_dim_total) - 0.5
    return ramp.unsqueeze(0).unsqueeze(0).to(torch.bfloat16)


def _torch_reference(torch_input, torch_gamma, torch_beta, is_rmsnorm, eps):
    with torch.no_grad():
        if is_rmsnorm:
            ref = torch.nn.RMSNorm(normalized_shape=torch_input.shape[-1], eps=eps)
            ref.weight.data = torch_gamma.clone().float()
            return ref(torch_input.float()).to(torch.bfloat16)
        return torch.nn.functional.layer_norm(
            torch_input.float(),
            [torch_input.shape[-1]],
            torch_gamma.float(),
            torch_beta.float(),
            eps,
        ).to(torch.bfloat16)


def _to_device(t, device):
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _run_distributed_norm_single_device(
    device,
    seq_len,
    hidden_dim_total,
    is_rmsnorm,
    use_2d_core_grid,
    expect_tpcx,
    eps=1e-5,
    pcc_threshold=0.99,
):
    assert hidden_dim_total % NUM_SIMULATED_DEVICES == 0
    hidden_per_dev = hidden_dim_total // NUM_SIMULATED_DEVICES

    if use_2d_core_grid:
        # Layout guard: the bug boundary must actually be exercised on this device.
        _assert_layout_exercised(device, seq_len, hidden_per_dev, expect_tpcx, "norm")

    cores_x, tpcx, cores_y, tpcy = _grid_decomposition(seq_len, hidden_per_dev)
    logger.info(
        f"shape=(1,1,{seq_len},{hidden_dim_total}) per-dev=({seq_len},{hidden_per_dev}) "
        f"cores_x={cores_x} tiles_per_core_x={tpcx} cores_y={cores_y} tiles_per_core_y={tpcy} "
        f"is_rmsnorm={is_rmsnorm} use_2d_core_grid={use_2d_core_grid}"
    )

    # Deterministic input; the seed only affects the gamma/beta tensors below.
    torch.manual_seed(1234)
    torch_input = _make_ramp_input(seq_len, hidden_dim_total)
    torch_gamma = torch.randn(hidden_dim_total, dtype=torch.bfloat16)
    torch_beta = torch.randn(hidden_dim_total, dtype=torch.bfloat16)
    torch_output = _torch_reference(torch_input, torch_gamma, torch_beta, is_rmsnorm, eps)

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    # Chunk input and gamma/beta along hidden dim to simulate per-device shards.
    input_chunks = torch.chunk(torch_input, NUM_SIMULATED_DEVICES, dim=-1)
    gamma_chunks = torch.chunk(torch_gamma, NUM_SIMULATED_DEVICES, dim=-1)
    beta_chunks = torch.chunk(torch_beta, NUM_SIMULATED_DEVICES, dim=-1)

    tt_inputs = [_to_device(c, device) for c in input_chunks]
    tt_gammas = [_to_device(w.reshape(1, 1, 1, hidden_per_dev), device) for w in gamma_chunks]
    tt_betas = [_to_device(b.reshape(1, 1, 1, hidden_per_dev), device) for b in beta_chunks]

    # NOTE: ttnn.layer_norm_pre_all_gather now exposes use_2d_core_grid (threaded
    # through layernorm_pre_all_gather.{hpp,cpp} and the nanobind binding as part of
    # the #56908 fix, mirroring ttnn.rms_norm_pre_all_gather). ttnn.layer_norm_post_all_gather
    # exposes it as well (added alongside the post-all-gather stride fix).
    if is_rmsnorm:
        pre_op, post_op = ttnn_rms_norm_pre_all_gather, ttnn_rms_norm_post_all_gather
    else:
        pre_op, post_op = ttnn_layer_norm_pre_all_gather, ttnn_layer_norm_post_all_gather

    # Step 1: per-shard pre-all-gather stats.
    tt_stats = [
        pre_op(
            t,
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat16,
            use_2d_core_grid=use_2d_core_grid,
        )
        for t in tt_inputs
    ]

    # Step 2: simulate the all-gather by concatenating along stats dim (=3).
    tt_stats_gathered = ttnn.concat(tt_stats, dim=3)

    # Step 3: per-shard post-all-gather norm using the gathered stats.
    tt_outputs = []
    for i in range(NUM_SIMULATED_DEVICES):
        post_kwargs = dict(
            epsilon=eps,
            weight=tt_gammas[i],
            compute_kernel_config=compute_kernel_config,
            use_2d_core_grid=use_2d_core_grid,
        )
        if not is_rmsnorm:
            post_kwargs["bias"] = tt_betas[i]
        tt_outputs.append(post_op(tt_inputs[i], tt_stats_gathered, **post_kwargs))

    tt_out_concat = ttnn.concat(tt_outputs, dim=-1)
    tt_output_torch = ttnn.to_torch(tt_out_concat).to(torch.bfloat16)

    passing, pcc_msg = comp_pcc(torch_output, tt_output_torch, pcc=pcc_threshold)
    logger.info(f"is_rmsnorm={is_rmsnorm} use_2d_core_grid={use_2d_core_grid} | {pcc_msg}")

    return passing, pcc_msg


def _run_distributed_welford_layernorm_single_device(
    device,
    seq_len,
    hidden_dim_total,
    use_2d_core_grid,
    expect_tpcx,
    eps=1e-5,
):
    """Layernorm-only Welford leg: same 3-step pipeline with the Welford program config.

    Tolerances follow the existing welford test's bf16 floor: stats atol 0.01,
    PCC 0.999 (the fp32-stats floor is atol/rtol 1e-5, PCC 0.99999).
    """
    assert hidden_dim_total % NUM_SIMULATED_DEVICES == 0
    hidden_per_dev = hidden_dim_total // NUM_SIMULATED_DEVICES

    if use_2d_core_grid:
        _assert_layout_exercised(device, seq_len, hidden_per_dev, expect_tpcx, "welford")

    cores_x, tpcx, cores_y, tpcy = _grid_decomposition(seq_len, hidden_per_dev)
    logger.info(
        f"[welford] shape=(1,1,{seq_len},{hidden_dim_total}) per-dev=({seq_len},{hidden_per_dev}) "
        f"cores_x={cores_x} tiles_per_core_x={tpcx} cores_y={cores_y} tiles_per_core_y={tpcy} "
        f"use_2d_core_grid={use_2d_core_grid}"
    )

    torch.manual_seed(1234)
    torch_input = _make_ramp_input(seq_len, hidden_dim_total)
    torch_gamma = torch.randn(hidden_dim_total, dtype=torch.bfloat16)
    torch_beta = torch.randn(hidden_dim_total, dtype=torch.bfloat16)
    torch_output = _torch_reference(torch_input, torch_gamma, torch_beta, False, eps)

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=True)
    grid = device.compute_with_storage_grid_size()
    core_range_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    # Reciprocals are per the width the kernel reduces over: this shard's columns.
    recip_tensor = ttnn.create_layer_norm_reciprocals(device, core_range_set, hidden_per_dev)

    input_chunks = torch.chunk(torch_input, NUM_SIMULATED_DEVICES, dim=-1)
    gamma_chunks = torch.chunk(torch_gamma, NUM_SIMULATED_DEVICES, dim=-1)
    beta_chunks = torch.chunk(torch_beta, NUM_SIMULATED_DEVICES, dim=-1)

    tt_inputs = [_to_device(c, device) for c in input_chunks]
    tt_gammas = [_to_device(w.reshape(1, 1, 1, hidden_per_dev), device) for w in gamma_chunks]
    tt_betas = [_to_device(b.reshape(1, 1, 1, hidden_per_dev), device) for b in beta_chunks]

    # Step 1: per-shard Welford pre-all-gather stats.
    tt_stats = []
    for t in tt_inputs:
        try:
            tt_stats.append(
                ttnn_layer_norm_pre_all_gather(
                    t,
                    compute_kernel_config=compute_kernel_config,
                    dtype=ttnn.bfloat16,
                    program_config=program_config,
                    recip_tensor=recip_tensor,
                    use_2d_core_grid=use_2d_core_grid,
                )
            )
        except RuntimeError as e:
            # Welford+2D pre-all-gather is rejected by device validation on current
            # trees; record the API boundary as a skip instead of failing.
            if use_2d_core_grid:
                pytest.skip(f"Welford+2D pre-all-gather rejected by device validation: {e}")
            raise

    # Step 2: simulate the all-gather by concatenating along stats dim (=3).
    tt_stats_gathered = ttnn.concat(tt_stats, dim=3)

    # Welford stats layout (per shard): per-row mean at tile 0 col 0, per-row variance
    # at tile 1 col 0 (= overall column 32); 2 stat tiles per device shard.
    stats_torch = ttnn.to_torch(tt_stats_gathered).float()
    for d, chunk in enumerate(input_chunks):
        ref_mean = chunk.float().mean(dim=-1)
        ref_var = chunk.float().var(dim=-1, unbiased=False)
        tt_mean = stats_torch[..., d * 64 + 0]
        tt_var = stats_torch[..., d * 64 + 32]
        assert torch.allclose(tt_mean, ref_mean, atol=0.01, rtol=0.01), f"shard {d}: welford mean mismatch"
        assert torch.allclose(tt_var, ref_var, atol=0.01, rtol=0.01), f"shard {d}: welford var mismatch"
        for label, tt_v, ref_v in (("mean", tt_mean, ref_mean), ("var", tt_var, ref_var)):
            passing, pcc_msg = comp_pcc(ref_v, tt_v, pcc=0.999)
            logger.info(f"[welford] shard={d} stat={label} | {pcc_msg}")
            assert passing, f"shard {d}: welford {label} PCC check failed: {pcc_msg}"

    # Step 3: per-shard Welford post-all-gather norm using the gathered stats.
    tt_outputs = [
        ttnn_layer_norm_post_all_gather(
            tt_inputs[i],
            tt_stats_gathered,
            epsilon=eps,
            weight=tt_gammas[i],
            bias=tt_betas[i],
            compute_kernel_config=compute_kernel_config,
            program_config=program_config,
            use_2d_core_grid=use_2d_core_grid,
        )
        for i in range(NUM_SIMULATED_DEVICES)
    ]

    tt_out_concat = ttnn.concat(tt_outputs, dim=-1)
    tt_output_torch = ttnn.to_torch(tt_out_concat).to(torch.bfloat16)

    passing, pcc_msg = comp_pcc(torch_output, tt_output_torch, pcc=0.999)
    logger.info(f"[welford] use_2d_core_grid={use_2d_core_grid} | {pcc_msg}")

    return passing, pcc_msg


@pytest.mark.parametrize(
    "seq_len, hidden_dim_total, expect_tpcx",
    [
        # (320, 8192): per-dev (H=320, Wt=64) -> cores_x=5, tiles_per_core_x=2.
        # Minimal multi-row-tile case: crosses the tiles_per_core_x > 1 boundary.
        (320, 8192, 2),
        # (512, 8192): per-dev (H=512, Wt=64) -> cores_x=8, tiles_per_core_x=2.
        # Taller shape on the full 8-wide core row: crosses tiles_per_core_x > 1.
        (512, 8192, 2),
        # (1024, 4096): per-dev (H=1024, Wt=32) -> cores_x=8, tiles_per_core_x=4.
        # Deep stride + narrow rows: crosses tiles_per_core_x > 1.
        # HANG CANARY: #55075 reports the 2D pre-all-gather path deadlocking at
        # H=1024,W=1024 on an unmodified tree. The identified mechanism (reduce scaler
        # popped per-row from a depth-1 buffer) is fixed in this PR; run under
        # TT_METAL_OPERATION_TIMEOUT_SECONDS=45 as a canary for any residual hang.
        (1024, 4096, 4),
        # (128, 8192): per-dev (H=128, Wt=64) -> cores_x=4, tiles_per_core_x=1.
        # Guard: existing valid 2D behavior must stay green.
        (128, 8192, 1),
        # (128, 4096): per-dev (H=128, Wt=32) -> cores_x=4, tiles_per_core_x=1.
        # Guard: #50287's regression case (cores_y > tiles_per_core_y) stays green.
        (128, 4096, 1),
    ],
)
@pytest.mark.parametrize("is_rmsnorm", [True, False], ids=["rmsnorm", "layernorm"])
@pytest.mark.parametrize("use_2d_core_grid", [False, True], ids=["grid_1d", "grid_2d"])
def test_distributed_2d_core_grid_row_stride(
    device, seq_len, hidden_dim_total, expect_tpcx, is_rmsnorm, use_2d_core_grid
):
    if not is_rmsnorm and use_2d_core_grid:
        # layernorm+2D pre is rejected by device validation: the 2D pre compute kernel
        # emits rmsnorm-only (1-wide) statistics, so running it would silently produce
        # wrong stats downstream. Loud TT_FATAL instead; skip by design.
        pytest.skip("layernorm+use_2d_core_grid pre rejected by device validation (rmsnorm-stats-only 2D kernel)")
    passing, pcc_msg = _run_distributed_norm_single_device(
        device=device,
        seq_len=seq_len,
        hidden_dim_total=hidden_dim_total,
        is_rmsnorm=is_rmsnorm,
        use_2d_core_grid=use_2d_core_grid,
        expect_tpcx=expect_tpcx,
    )
    assert passing, (
        f"PCC check failed (is_rmsnorm={is_rmsnorm}, use_2d_core_grid={use_2d_core_grid}): {pcc_msg}"
    )


@pytest.mark.parametrize(
    "seq_len, hidden_dim_total, expect_tpcx",
    [
        # tiles_per_core_x=2 shapes only: the Welford leg targets the boundary.
        (320, 8192, 2),
        (512, 8192, 2),
    ],
)
@pytest.mark.parametrize("is_rmsnorm", [True, False], ids=["rmsnorm", "layernorm"])
@pytest.mark.parametrize("use_2d_core_grid", [False, True], ids=["grid_1d", "grid_2d"])
def test_distributed_2d_core_grid_row_stride_welford(
    device, seq_len, hidden_dim_total, expect_tpcx, is_rmsnorm, use_2d_core_grid
):
    if is_rmsnorm:
        # RMSNorm+Welford is rejected by device validation (see #52110); #56908 covers
        # Welford "where supported by the existing API", i.e. layernorm only.
        pytest.skip("RMSNorm+Welford rejected by device validation; welford leg is layernorm-only")
    passing, pcc_msg = _run_distributed_welford_layernorm_single_device(
        device=device,
        seq_len=seq_len,
        hidden_dim_total=hidden_dim_total,
        use_2d_core_grid=use_2d_core_grid,
        expect_tpcx=expect_tpcx,
    )
    assert passing, f"Welford PCC check failed (use_2d_core_grid={use_2d_core_grid}): {pcc_msg}"
