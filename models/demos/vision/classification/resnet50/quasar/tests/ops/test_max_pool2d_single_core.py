# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Quasar max_pool2d 3x3 CHANNEL sweep pinned to a SINGLE core (0,0).

Same geometry, channel counts, input modes and checks as test_max_pool2d_channel_sweep.py -- the only
difference is that the shard is placed on exactly one core instead of the largest core count that divides
the height. Helpers are imported from that test so the two cannot drift.

Why a single-core variant:

  * The sweep test picks `max(c for c in 1..max_cores if height_tiles % c == 0)`, and height_tiles is 32,
    so on a 1x3 emulator it still selects 2 cores -- the failure then reports as two sticks, [0, 128],
    one per core. Pinning to one core reduces that to a single stick, [0].
  * A waveform capture has to name one core's scope. With two cores the corruption appears on both
    (each on its own first stick), so a capture scoped to one core only tells half the story and the
    default scope in the wave tooling matches neither.
  * It removes the second reader/compute pair as a variable entirely.

L1 NOTE: with one core that core holds the whole input shard -- 1024 sticks x channels x 2B, i.e. 512 KB
at 256c versus 256 KB when split across two cores, plus the output shard and the CBs. If this fails to
allocate rather than failing the PCC/exact-match check, reduce IN_H (a 16x32 input halves the shard and
keeps out_w >= 2, which is what exercises the row-to-row L1 stride).

Run (craq-sim):
  ./qsr_sim_run models/demos/vision/classification/resnet50/quasar/tests/ops/test_max_pool2d_single_core.py

Env knobs are the same as the sweep test, plus input geometry:
  QSR_POOL_SWEEP_ONLY=256      -> run only channels=256
  QSR_POOL_SWEEP_SKIP=96,160   -> skip these
  QSR_POOL_INPUT_MODE=chramp   -> rand|chramp|stickramp|const
  QSR_POOL_CONST_VAL=-3        -> const-mode value; negative exposes identity/tail-fill bugs
  QSR_POOL_IN_H=4              -> input rows (default 32); shrinks the repro, see below
  QSR_POOL_IN_W=32             -> input cols (default 32); prefer leaving this alone

Smallest known-equivalent repro (8x fewer sticks, still 8 channel-tiles):
  QSR_POOL_IN_H=4 QSR_POOL_INPUT_MODE=chramp QSR_POOL_SWEEP_ONLY=256 pytest ... -k "256c"

Channel count is the axis the fault needs (>=5 channel-tiles); stick count is not, since the fault
is always on core-local output stick 0. Shrinking sticks does change timing, though, and this fault
has been masked by perturbation before -- so if the small config passes, that is a result about
pipelining, not proof the bug is gone. Re-confirm at IN_H=32 before concluding anything.
"""

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import is_quasar

# Reuse the sweep test's geometry constants.
from models.demos.vision.classification.resnet50.quasar.tests.ops.test_max_pool2d_channel_sweep import CHANNELS
from models.demos.vision.classification.resnet50.quasar.tests.ops.test_max_pool2d_channel_sweep import (
    IN_H as _SWEEP_IN_H,
)
from models.demos.vision.classification.resnet50.quasar.tests.ops.test_max_pool2d_channel_sweep import (
    IN_W as _SWEEP_IN_W,
)
from models.demos.vision.classification.resnet50.quasar.tests.ops.test_max_pool2d_channel_sweep import (
    KERNEL,
    PADDING,
    STRIDE,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


def _make_input(batch, channels, in_h, in_w, mode="rand"):
    """Build the NCHW input tensor according to the requested mode."""
    if mode == "chramp":
        ramp = torch.arange(1, channels + 1, dtype=torch.float32) / channels
        return ramp.view(1, channels, 1, 1).expand(batch, channels, in_h, in_w).to(torch.bfloat16)
    elif mode == "stickramp":
        n_sticks = batch * in_h * in_w
        ramp = torch.arange(1, n_sticks + 1, dtype=torch.float32) / n_sticks
        return ramp.view(batch, 1, in_h, in_w).expand(batch, channels, in_h, in_w).to(torch.bfloat16)
    elif mode == "const":
        val = float(os.environ.get("QSR_POOL_CONST_VAL", "1.0"))
        return torch.full((batch, channels, in_h, in_w), val, dtype=torch.bfloat16)
    else:
        torch.manual_seed(0)
        return torch.rand((batch, channels, in_h, in_w), dtype=torch.bfloat16)


def _describe_ramp_mismatch(golden, got, channels, mode):
    """Return a human-readable string describing mismatches, or None if exact match."""
    if torch.equal(golden, got):
        return None
    lines = []
    flat_g = golden.view(-1, channels)
    flat_o = got.view(-1, channels)
    n_sticks = flat_g.shape[0]
    mismatched_sticks = []
    for s in range(n_sticks):
        if not torch.equal(flat_g[s], flat_o[s]):
            mismatched_sticks.append(s)
    lines.append(f"mode={mode}: {len(mismatched_sticks)}/{n_sticks} output sticks differ: {mismatched_sticks[:16]}")
    for s in mismatched_sticks[:4]:
        diff_ch = (flat_g[s] != flat_o[s]).nonzero(as_tuple=True)[0].tolist()
        lines.append(f"  stick {s}: {len(diff_ch)} bad channels: {diff_ch[:16]}")
        for c in diff_ch[:4]:
            lines.append(f"    ch {c}: golden={flat_g[s, c].item():.6f}  got={flat_o[s, c].item():.6f}")
    return "\n".join(lines)


# Exactly one core, always the first in the range set -> worker core (0,0).
NUM_CORES = 1

# Input geometry, overridable to shrink the repro. Defaults are the sweep test's, so an unset
# environment reproduces the sweep exactly rather than silently diverging from it.
#
# The fault is always on core-local output stick 0, so cutting the stick count should not remove it,
# and a shorter run is what makes an emulator waveform capture affordable (conversion cost scales
# with how far into the timeline the compute sits). Prefer shrinking IN_H over IN_W: holding IN_W
# fixed keeps the reader's per-row L1 addressing byte-identical to the known-failing config and
# changes only the number of rows.
#
#   IN_H=4,  IN_W=32 -> 128 input sticks,  2x16 =  32 output sticks   (8x smaller, 32-aligned)
#   IN_H=8,  IN_W=32 -> 256 input sticks,  4x16 =  64 output sticks   (4x smaller)
#   IN_H=32, IN_W=32 -> 1024 input sticks, 16x16 = 256 output sticks  (default, as the sweep)
IN_H = int(os.environ.get("QSR_POOL_IN_H", _SWEEP_IN_H))
IN_W = int(os.environ.get("QSR_POOL_IN_W", _SWEEP_IN_W))


def _run_max_pool_single_core(mesh_device, channels):
    print(f"[single_core] START channels={channels} ({channels // 32} tiles)", flush=True)
    device = mesh_device
    torch.manual_seed(0)
    batch = 1
    out_h = (IN_H - KERNEL[0] + 2 * PADDING[0]) // STRIDE[0] + 1
    out_w = (IN_W - KERNEL[1] + 2 * PADDING[1]) // STRIDE[1] + 1

    input_mode = os.environ.get("QSR_POOL_INPUT_MODE", "rand").strip().lower()
    x_nchw = _make_input(batch, channels, IN_H, IN_W, input_mode)
    input_max = x_nchw.float().max().item()
    golden_nchw = torch.nn.functional.max_pool2d(
        x_nchw.float(), kernel_size=list(KERNEL), stride=list(STRIDE), padding=list(PADDING)
    )
    x_nhwc_flat = x_nchw.permute(0, 2, 3, 1).reshape(1, 1, batch * IN_H * IN_W, channels).contiguous()
    golden_flat = golden_nchw.permute(0, 2, 3, 1).reshape(1, 1, batch * out_h * out_w, channels).contiguous()

    tensor_height = batch * IN_H * IN_W
    out_sticks = batch * out_h * out_w
    assert tensor_height % 32 == 0, (
        f"input sticks {tensor_height} (IN_H={IN_H} x IN_W={IN_W}) must be a multiple of 32; "
        f"pick QSR_POOL_IN_H/QSR_POOL_IN_W so their product is 32-aligned"
    )
    # out_w == 1 collapses every output row onto one stick, which stops exercising the row-to-row
    # L1 stride -- the geometry the failure depends on. Shrink IN_H, not IN_W.
    assert out_w >= 2, f"out_w={out_w} (IN_W={IN_W}) is too small to exercise the row-to-row L1 stride"
    if out_sticks % 32 != 0:
        print(
            f"[single_core] WARNING: {out_sticks} output sticks is not 32-aligned "
            f"(out {out_h}x{out_w}); the output shard is tile-padded, so a mismatch report may "
            f"include padding rows. IN_H=4/IN_W=32 gives exactly 32.",
            flush=True,
        )
    grid = device.compute_with_storage_grid_size()
    assert grid.x * grid.y >= NUM_CORES, f"grid {grid.x}x{grid.y} cannot host {NUM_CORES} core(s)"
    # The whole height on one core, so every output stick is produced by core (0,0).
    shard_height = tensor_height
    core_grid = ttnn.num_cores_to_corerangeset(NUM_CORES, grid, True)
    mem_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, shard_height, channels),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    x = ttnn.from_torch(x_nhwc_flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT).to(device, mem_config)

    print(
        f"[single_core] launching max_pool2d channels={channels} ({channels // 32} tiles) "
        f"num_cores={NUM_CORES} in {IN_H}x{IN_W} ({tensor_height} sticks) -> out {out_h}x{out_w} "
        f"({out_sticks} sticks) shard_height={shard_height} grid={grid.x}x{grid.y}",
        flush=True,
    )
    max_pool2d = ttnn.experimental.quasar.max_pool2d if is_quasar() else ttnn.max_pool2d
    out = max_pool2d(
        input_tensor=x,
        batch_size=batch,
        input_h=IN_H,
        input_w=IN_W,
        channels=channels,
        kernel_size=list(KERNEL),
        stride=list(STRIDE),
        padding=list(PADDING),
        dilation=[1, 1],
    )
    ttnn.synchronize_device(device)
    print(f"[single_core] DONE device op channels={channels} ({channels // 32} tiles)", flush=True)

    got = ttnn.to_torch(out).float().reshape(1, 1, batch * out_h * out_w, channels)
    got_max = got.max().item()
    # Sharp only for mode=rand; see _make_input for why the ramp/const modes rely on exact-match instead.
    assert got_max <= input_max + 1e-2, (
        f"pool leaked stale L1: got.max={got_max:.4f} > input.max={input_max:.4f} "
        f"(mode={input_mode}, cores={NUM_CORES}, ch={channels}={channels // 32} tiles, "
        f"{IN_H}x{IN_W}, out {out_h}x{out_w})"
    )
    if input_mode == "rand":
        assert_with_pcc(golden_flat, got, pcc=0.99)
    else:
        report = _describe_ramp_mismatch(golden_flat, got, channels, input_mode)
        assert report is None, (
            f"mode={input_mode}, cores={NUM_CORES}, ch={channels}={channels // 32} tiles, "
            f"{IN_H}x{IN_W}, out {out_h}x{out_w}\n{report}"
        )


@pytest.mark.timeout(600000)
@pytest.mark.parametrize("channels", CHANNELS, ids=[f"{c}c_{c // 32}tiles" for c in CHANNELS])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_quasar_max_pool2d_single_core(mesh_device, channels):
    only = os.environ.get("QSR_POOL_SWEEP_ONLY", "").strip()
    if only and channels not in {int(c) for c in only.split(",") if c.strip()}:
        pytest.skip(f"channels={channels} not in QSR_POOL_SWEEP_ONLY={only}")
    skip = os.environ.get("QSR_POOL_SWEEP_SKIP", "").strip()
    if skip and channels in {int(c) for c in skip.split(",") if c.strip()}:
        pytest.skip(f"channels={channels} in QSR_POOL_SWEEP_SKIP={skip}")
    _run_max_pool_single_core(mesh_device, channels)
