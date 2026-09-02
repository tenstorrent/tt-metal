# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for the single ttnn.linear the tt-forge ResNet-50 graph issues (the fc classifier), run on
Quasar via ttnn.experimental.quasar.linear.

  in   [1,2048]     L1   INTERLEAVED TILE
  w    [2048,1000]  DRAM INTERLEAVED TILE   (already K x N -- not torch's N x K)
  b    [1000]       DRAM INTERLEAVED TILE   (rank 1 -- see below)
  out  [1,1000]     L1   WIDTH_SHARDED TILE, 32 cores (8x4 grid, (0,0)-(7,3)), shard [32, 32]
  MathFidelity.HiFi2, transpose_a=False, transpose_b=False
  program config: MatmulMultiCoreReuseMultiCast1DProgramConfig(
      compute_with_storage_grid_size=CoreCoord(8, 4), in0_block_w=8,
      out_subblock_h=1, out_subblock_w=1, out_block_h=1, out_block_w=1,
      per_core_M=1, per_core_N=1, fuse_batch=True, mcast_in0=True,
      gather_in0=False, num_global_cb_receivers=0, untilize_out=False)

  That config re-derives cleanly: K = 2048 = 64 tiles and in0_block_w=8 divides 64; N = 1000
  tile-pads to 1024 = 32 tiles, and 8*4 cores x per_core_N=1 = 32 tiles. The test asserts both
  relations, so a stale table fails here rather than deep inside the matmul.

HOW THIS DIFFERS FROM ../ops/test_linear.py
  The sibling test feeds the fc a WIDTH_SHARDED activation on the matmul grid and derives its grid
  from the device. Forge does neither: it inserts a to_memory_config that gathers the activation to
  L1 INTERLEAVED first, then hands the 1D-mcast matmul an interleaved in0 on a fixed 8x4 grid. So
  this is a different in0 path through the same 1D-mcast kernel.

RANK-1 BIAS
  The Forge IR's fc bias is `tensor<1000xbf16>` -- rank 1, exactly as built here. If quasar.linear
  rejects it on rank, that is a genuine Forge-graph/Quasar mismatch worth reporting (Forge would
  have to emit a rank-lift), not a defect in this test; the failure message names the bias shape.

WHERE THE PROGRAM CONFIG CLASS LIVES
  quasar.linear only accepts a program config from ITS OWN overload set. The top-level
  ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig is a different C++ type and is rejected with a
  TypeError listing the accepted ones. The quasar classes are bound into the extension module
  (ttnn._ttnn.operations.experimental.quasar) but are NOT re-exported onto the ttnn.experimental
  .quasar python namespace, so they have to be reached through the binding module -- which is what
  _quasar_program_config does below.

NOTE ON THE MATMUL PATH
  The 2D-mcast matmul path has hung on Quasar (LLK dest-sync; those kernels are no-op'd). This fc
  uses the 1D-mcast config (mcast_in0=True), a different kernel path. If it hangs or mismatches,
  that is a DISTINCT bug from the 2D-mcast one and should be filed separately.

RUN
    pytest -s models/demos/vision/classification/resnet50/quasar/tests/ResNet50_Forge_Fe/test_linear_forge.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

PCC = 0.98  # bf16 + HiFi2 over a K=2048 reduction is noisy

# --- Forge's memory configs, verbatim from the TTNN IR --------------------------------------------
CR32 = (((0, 0), (7, 3)),)  # 8x4 = 32 cores

# (memory_layout, buffer_type, core_ranges, shard_shape, page_layout)
FC_IN = ("INTERLEAVED", "L1", None, None, "TILE")
FC_WEIGHT = ("INTERLEAVED", "DRAM", None, None, "TILE")
FC_BIAS = ("INTERLEAVED", "DRAM", None, None, "TILE")
FC_OUT = ("WIDTH_SHARDED", "L1", CR32, (32, 32), "TILE")

INPUT_SHAPE = (1, 2048)
WEIGHT_SHAPE = (2048, 1000)  # K x N, already transposed by Forge
BIAS_SHAPE = (1000,)  # rank 1, verbatim from the IR
OUTPUT_SHAPE = (1, 1000)

# MatmulMultiCoreReuseMultiCast1DProgramConfig fields, verbatim from the IR
GRID = (8, 4)
IN0_BLOCK_W = 8
OUT_SUBBLOCK_H = 1
OUT_SUBBLOCK_W = 1
OUT_BLOCK_H = 1
OUT_BLOCK_W = 1
PER_CORE_M = 1
PER_CORE_N = 1
FUSE_BATCH = True
MCAST_IN0 = True
GATHER_IN0 = False
NUM_GLOBAL_CB_RECEIVERS = 0
UNTILIZE_OUT = False


def _mem(spec):
    """Frozen Forge memory-config tuple -> a real ttnn.MemoryConfig."""
    memory_layout, buffer_type, core_ranges, shard_shape, _page_layout = spec
    layout = getattr(ttnn.TensorMemoryLayout, memory_layout)
    buffer = getattr(ttnn.BufferType, buffer_type)
    if core_ranges is None:
        return ttnn.MemoryConfig(layout, buffer, None)
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*lo), ttnn.CoreCoord(*hi)) for lo, hi in core_ranges])
    return ttnn.MemoryConfig(layout, buffer, ttnn.ShardSpec(ranges, list(shard_shape), ttnn.ShardOrientation.ROW_MAJOR))


def _page(spec):
    return ttnn.TILE_LAYOUT if spec[4] == "TILE" else ttnn.ROW_MAJOR_LAYOUT


def _to_device(x, spec, device, what):
    tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=_page(spec))
    try:
        return tt.to(device, _mem(spec))
    except Exception as e:
        raise AssertionError(
            "could not place the fc %s (%s tensor) in the Forge memory config (%s/%s/%s): %s"
            % (what, tuple(x.shape), spec[1], spec[0], spec[4], e)
        ) from e


def _quasar_program_config():
    """
    The quasar MatmulMultiCoreReuseMultiCast1DProgramConfig class.

    quasar.linear accepts only its own program-config types; the top-level ttnn one is a different
    C++ type and is rejected. The quasar classes are bound into the extension module but not
    re-exported onto ttnn.experimental.quasar, so reach them through the binding module.
    """
    name = "MatmulMultiCoreReuseMultiCast1DProgramConfig"
    for holder in (
        getattr(getattr(ttnn, "experimental", None), "quasar", None),
        getattr(getattr(getattr(ttnn._ttnn, "operations", None), "experimental", None), "quasar", None),
    ):
        cls = getattr(holder, name, None) if holder is not None else None
        if cls is not None:
            return cls
    pytest.fail(
        "NOT EXPOSED: no quasar %s could be found on ttnn.experimental.quasar or "
        "ttnn._ttnn.operations.experimental.quasar; the Forge fc matmul config cannot be built" % name
    )


def _require_grid(device, *specs):
    """Skip unless the device compute grid can hold every Forge core range in play."""
    grid = device.compute_with_storage_grid_size()
    needed = [(GRID[0], GRID[1])] + [
        (max(hi[0] for _lo, hi in s[2]) + 1, max(hi[1] for _lo, hi in s[2]) + 1) for s in specs if s[2] is not None
    ]
    for need_x, need_y in needed:
        if need_x > grid.x or need_y > grid.y:
            pytest.skip(
                "Forge pins a %dx%d core grid but this device grid is %dx%d. These configs need a "
                "full Quasar part; ../ops/ covers the same kernels with device-derived sharding."
                % (need_x, need_y, grid.x, grid.y)
            )


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_forge_linear(mesh_device):
    device = mesh_device
    torch.manual_seed(0)

    _require_grid(device, FC_OUT)

    k, n = WEIGHT_SHAPE
    # re-derive the program config's consistency from the shapes; catches a stale table cheaply
    assert (k // 32) % IN0_BLOCK_W == 0, "in0_block_w=%d does not divide K tiles=%d" % (IN0_BLOCK_W, k // 32)
    n_tiles = -(-n // 32)
    assert (
        GRID[0] * GRID[1] * PER_CORE_N == n_tiles
    ), "grid %dx%d x per_core_N=%d covers %d N tiles, but N=%d needs %d" % (
        GRID[0],
        GRID[1],
        PER_CORE_N,
        GRID[0] * GRID[1] * PER_CORE_N,
        n,
        n_tiles,
    )

    # ---- torch golden: act @ weight + bias (weight is already K x N) ----------------------------
    act_torch = torch.randn(INPUT_SHAPE, dtype=torch.bfloat16)
    w_torch = torch.randn(WEIGHT_SHAPE, dtype=torch.bfloat16)
    b_torch = torch.randn(BIAS_SHAPE, dtype=torch.bfloat16)
    golden = torch.matmul(act_torch.float(), w_torch.float()) + b_torch.float()
    assert tuple(golden.shape) == OUTPUT_SHAPE

    # ---- operands in Forge's exact placement ----------------------------------------------------
    act = _to_device(act_torch, FC_IN, device, "activation")
    weight = _to_device(w_torch, FC_WEIGHT, device, "weight")
    bias = _to_device(b_torch, FC_BIAS, device, "rank-1 bias")

    program_config = _quasar_program_config()(
        compute_with_storage_grid_size=ttnn.CoreCoord(*GRID),
        in0_block_w=IN0_BLOCK_W,
        out_subblock_h=OUT_SUBBLOCK_H,
        out_subblock_w=OUT_SUBBLOCK_W,
        out_block_h=OUT_BLOCK_H,
        out_block_w=OUT_BLOCK_W,
        per_core_M=PER_CORE_M,
        per_core_N=PER_CORE_N,
        fuse_batch=FUSE_BATCH,
        fused_activation=None,
        mcast_in0=MCAST_IN0,
        gather_in0=GATHER_IN0,
        num_global_cb_receivers=NUM_GLOBAL_CB_RECEIVERS,
        untilize_out=UNTILIZE_OUT,
    )

    out = ttnn.experimental.quasar.linear(
        act,
        weight,
        bias=bias,
        transpose_a=False,
        transpose_b=False,
        program_config=program_config,
        memory_config=_mem(FC_OUT),
        dtype=ttnn.bfloat16,
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2
        ),
    )
    ttnn.synchronize_device(device)

    assert tuple(out.shape) == OUTPUT_SHAPE, "fc output %s, Forge IR says %s" % (tuple(out.shape), OUTPUT_SHAPE)
    got_layout = out.memory_config().memory_layout
    assert got_layout == getattr(ttnn.TensorMemoryLayout, FC_OUT[0]), "fc landed in %s but Forge asked for %s" % (
        got_layout,
        FC_OUT[0],
    )

    got = ttnn.to_torch(ttnn.from_device(out)).float().reshape(OUTPUT_SHAPE)
    assert_with_pcc(golden, got, pcc=PCC)
