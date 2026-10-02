# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0.5 on a restricted compute grid: check that no op runs on a Tensix core outside it.

tt-metal can cap the compute grid with TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE, but an op that
hardcodes its cores could still reach past the cap. This test runs the model in a child process with
the grid capped and the device profiler on (both must be set before tt-metal starts, hence the child
process): model build, one warmup sample_actions, then the trace-ready call (fixed noise + pre-staged
upstream artifacts, as in test_perf_ttnn_full_e2e_trace_2cq.py), all eager. The parent then asserts
that every core the device profiler saw execute a kernel lies inside the requested grid.

Usage (grid defaults to 11x10):
  source models/experimental/pi0_5/p150/common/pi05_production.env
  export PI05_CHECKPOINT_DIR=$PWD/models/experimental/pi0_5/p150/weights/pi05_base
  PI0_CORE_GRID=11x10 PI0_NUM_CAMERAS=3 pytest -sq models/experimental/pi0_5/p150/tests/perf/test_core_grid_bounds.py

Perf on the same grid: run test_perf_ttnn_full_e2e_trace_2cq.py with
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="10,9" (inclusive end core, i.e. 11x10).
"""

import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[6]
_DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "weights" / "pi05_base"
CHECKPOINT_DIR = Path(os.environ.get("PI05_CHECKPOINT_DIR", str(_DEFAULT_CHECKPOINT_DIR)))
GRID_X, GRID_Y = (int(v) for v in os.environ.get("PI0_CORE_GRID", "11x10").lower().split("x"))
PROFILER_PROGRAM_SUPPORT_COUNT = 20000  # headroom: model build + two eager sample_actions is ~4.7k programs

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"pi0.5 checkpoint not found at {CHECKPOINT_DIR}",
)


def _worker(out_dir: Path):
    """Child process: run the model on the capped grid with the device profiler on."""
    os.environ.setdefault("TT_VISIBLE_DEVICES", "0")
    from models.experimental.pi0_5.p150.common.prod_env import apply_production_env_defaults

    apply_production_env_defaults()

    import torch
    import ttnn

    from models.experimental.pi0_5.p150.common.checkpoint_meta import action_horizon_from_checkpoint
    from models.experimental.pi0_5.p150.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.p150.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.p150.tests.perf.test_perf_ttnn_full_e2e_trace_2cq import (
        LANG_SEQ_LEN,
        TRACE_REGION_SIZE,
        _build_inputs,
        _call_sample_actions,
    )
    from models.experimental.pi0_5.p150.tt.ttnn_pi0_5_model import Pi0_5ModelTTNN, use_upstream_masks

    device = ttnn.open_device(
        device_id=0, l1_small_size=24576, trace_region_size=TRACE_REGION_SIZE, num_command_queues=2
    )
    grid = device.compute_with_storage_grid_size()
    # Virtual (NoC) coordinates of the in-grid cores, plus a reverse map that names any stray core.
    allowed = []
    for x in range(grid.x):
        for y in range(grid.y):
            v = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            allowed.append([v.x, v.y])
    logical_of = {}
    for x in range(16):
        for y in range(12):
            try:
                v = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            except Exception:
                continue
            logical_of[f"{v.x},{v.y}"] = [x, y]

    cfg = Pi0_5ModelConfig(
        action_horizon=action_horizon_from_checkpoint(CHECKPOINT_DIR),
        num_denoising_steps=int(os.environ.get("PI05_NUM_DENOISE_STEPS", "10")),
    )
    model = Pi0_5ModelTTNN(cfg, Pi0_5WeightLoader(str(CHECKPOINT_DIR)), device)
    ttnn.ReadDeviceProfiler(device)
    images, img_masks, lang_tokens, lang_masks = _build_inputs(device)
    with torch.no_grad():
        _call_sample_actions(model, images, img_masks, lang_tokens, lang_masks)
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)
        model.resample_noise = False
        if use_upstream_masks():
            prefix_len = cfg.siglip_config.num_patches * len(img_masks) + LANG_SEQ_LEN
            model.prepare_upstream_artifacts(img_masks, lang_masks, prefix_len=prefix_len)
        _call_sample_actions(model, images, img_masks, lang_tokens, lang_masks)
        ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    ttnn.close_device(device)
    (out_dir / "grid.json").write_text(
        json.dumps({"grid": [grid.x, grid.y], "allowed": allowed, "logical_of": logical_of})
    )


def test_core_grid_bounds(tmp_path):
    env = dict(
        os.environ,
        TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=f"{GRID_X - 1},{GRID_Y - 1}",
        TT_METAL_DEVICE_PROFILER="1",
        TT_METAL_PROFILER_DIR=str(tmp_path),
        TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=str(PROFILER_PROGRAM_SUPPORT_COUNT),
    )
    module = "models.experimental.pi0_5.p150.tests.perf.test_core_grid_bounds"
    subprocess.run([sys.executable, "-m", module, str(tmp_path)], cwd=REPO_ROOT, env=env, check=True)

    meta = json.loads((tmp_path / "grid.json").read_text())
    assert meta["grid"] == [GRID_X, GRID_Y], f"device reports grid {meta['grid']}, wanted {GRID_X}x{GRID_Y}"
    allowed = {tuple(c) for c in meta["allowed"]}

    cores_per_op = defaultdict(set)  # run host ID -> virtual cores that executed a kernel
    with open(tmp_path / ".logs" / "profile_log_device.csv") as f:
        next(f)  # "ARCH: ..., CHIP_FREQ[MHz]: ..." line
        for row in csv.DictReader(f, skipinitialspace=True):
            cores_per_op[row["run host ID"]].add((int(row["core_x"]), int(row["core_y"])))
    assert cores_per_op, "device profiler recorded nothing"

    used = set().union(*cores_per_op.values())
    to_logical = lambda c: meta["logical_of"].get(f"{c[0]},{c[1]}", f"virtual {c}")
    logical_used = [meta["logical_of"][f"{c[0]},{c[1]}"] for c in used if f"{c[0]},{c[1]}" in meta["logical_of"]]
    print(
        f"\n{len(cores_per_op)} programs on grid {GRID_X}x{GRID_Y}: {len(used)} distinct cores, "
        f"logical x {min(c[0] for c in logical_used)}..{max(c[0] for c in logical_used)}, "
        f"y {min(c[1] for c in logical_used)}..{max(c[1] for c in logical_used)}; "
        f"max {max(len(c) for c in cores_per_op.values())} cores in one program"
    )
    stray = {
        op: [to_logical(c) for c in sorted(cores - allowed)] for op, cores in cores_per_op.items() if cores - allowed
    }
    assert not stray, f"{len(stray)} programs ran outside the {GRID_X}x{GRID_Y} grid (logical cores): {stray}"


if __name__ == "__main__":
    _worker(Path(sys.argv[1]))
