# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm — SHARDED perf harness at the perf group's EXACT config + geometry.

The `perf` loose-case group in `eval/golden_tests/rms_norm/feature_spec.py`
carries five *sharded* profiles, each pinned to its measured-fastest geometry
through `extras.shard_shape` + `extras.core_grid`:

    (1,1,32,1024)   WIDTH  [32,128]   (8,1)    4110 ns    8 cores
    (1,1,32,2304)   WIDTH  [32,256]   (9,1)    4617 ns    9 cores
    (1,1,32,5120)   WIDTH  [32,160]   (8,4)    5267 ns   32 cores
    (1,1,32,7168)   WIDTH  [32,256]   (7,4)    5481 ns   28 cores
    (1,1,8192,1024) BLOCK  [1024,128] (8,8)   25640 ns   64 cores

all at the group's fixed config: bf16 input, TILE layout, bf16 **TILE** gamma,
math_fidelity=HiFi2, fp32_dest_acc_en=False.  The shard geometry is reproduced
here exactly as `helpers.run_rms_norm` builds it (`eval.sharding.shard_config`),
so this file measures the same program the golden cell measures.

The input is already resident in L1, so DRAM traffic is gamma only and the whole
budget is kernel boot + the cross-core combine + (for BLOCK) the per-block
schedule.  These references sit at or below the ~3.5 us fixed dispatch/boot floor
measured for the interleaved path, so this is a FIXED-COST profile, not a
bandwidth one.

Run:

    scripts/run_safe_pytest.sh --profile --run-all \
        tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_perf_sharded.py

Each case runs ONCE — device kernel time has no warm-up transient.

Knob / ablation switches are FILES, not env vars: under `--profile` the measured
run lives in a `python -m tracy` child and an ad-hoc env var does not reach it.

    echo 32   > /tmp/rms_norm_dm_chunk       # DM_CHUNK_TILES
    echo 0    > /tmp/rms_norm_tree_min       # COMBINE_TREE_MIN_SLICES (0 disables the tree)
    echo no_gamma > /tmp/rms_norm_ablate     # gamma=None: costs out the gamma DRAM read + apply

The patch target is `create_program_descriptor.__globals__`, NOT the module
object obtained by importing the descriptor file (the package is reachable under
two names, so `monkeypatch.setattr(pd, KNOB, v)` patches a second import that
nobody runs).
"""

from __future__ import annotations

import pathlib

import pytest
import torch

import ttnn
from eval.sharding import shard_config
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm.rms_norm import create_program_descriptor as _create_program_descriptor

PLAN_GLOBALS = _create_program_descriptor.__globals__

_ML = ttnn.TensorMemoryLayout

TARGET_FIDELITY = ttnn.MathFidelity.HiFi2
TARGET_FP32_ACC = False

# (shape, memory_layout, shard_shape, core_grid, achievable_ns) — verbatim from
# feature_spec.LOOSE_CASES' sharded `perf` rows.
SHARDED_CASES = [
    pytest.param((1, 1, 32, 1024), _ML.WIDTH_SHARDED, [32, 128], (8, 1), 4110, id="wshard_w1024"),
    pytest.param((1, 1, 32, 2304), _ML.WIDTH_SHARDED, [32, 256], (9, 1), 4617, id="wshard_w2304"),
    pytest.param((1, 1, 32, 5120), _ML.WIDTH_SHARDED, [32, 160], (8, 4), 5267, id="wshard_w5120"),
    pytest.param((1, 1, 32, 7168), _ML.WIDTH_SHARDED, [32, 256], (7, 4), 5481, id="wshard_w7168"),
    pytest.param((1, 1, 8192, 1024), _ML.BLOCK_SHARDED, [1024, 128], (8, 8), 25640, id="bshard_prefill_w1024"),
]


def _read(path, cast, default=None):
    p = pathlib.Path(path)
    return cast(p.read_text().strip()) if p.exists() else default


DM_CHUNK = _read("/tmp/rms_norm_dm_chunk", int)
TREE_MIN = _read("/tmp/rms_norm_tree_min", int)
ABLATE = _read("/tmp/rms_norm_ablate", str, "none")
# Fidelity probe: a zero-code compute-cost classifier.  If the wall does not move
# between LoFi and HiFi4 the shape is not compute-bound.
_FIDELITY = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi3": ttnn.MathFidelity.HiFi3,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}
FIDELITY = _FIDELITY[_read("/tmp/rms_norm_fidelity", str, "HiFi2")]


@pytest.fixture(autouse=True)
def knobs(monkeypatch):
    if DM_CHUNK is not None:
        monkeypatch.setitem(PLAN_GLOBALS, "DM_CHUNK_TILES", DM_CHUNK)
    if TREE_MIN is not None:
        monkeypatch.setitem(PLAN_GLOBALS, "COMBINE_TREE_MIN_SLICES", TREE_MIN)
    return (DM_CHUNK, TREE_MIN, ABLATE)


def target_compute_config():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=FIDELITY,
        fp32_dest_acc_en=TARGET_FP32_ACC,
        math_approx_mode=False,
    )


@pytest.mark.parametrize("shape,memory_layout,shard_shape,core_grid,achievable_ns", SHARDED_CASES)
def test_rms_norm_perf_sharded(device, shape, memory_layout, shard_shape, core_grid, achievable_ns, knobs):
    torch.manual_seed(42)
    torch_x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    torch_gamma = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32).to(torch.bfloat16)

    memory_config = shard_config(
        shard_shape,
        core_grid,
        memory_layout,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )
    x = ttnn.from_torch(
        torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    # TILE-layout gamma in DRAM — the perf group's config.
    gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    use_gamma = None if ABLATE == "no_gamma" else gamma
    out = ttnn.to_torch(
        rms_norm(
            x,
            gamma=use_gamma,
            compute_kernel_config=target_compute_config(),
            memory_config=x.memory_config(),
        )
    ).to(torch.float32)

    if ABLATE != "none" or FIDELITY is not TARGET_FIDELITY:
        return  # payload stubbed / off-config: only the ns are meaningful

    xf = torch_x.to(torch.float32)
    expected = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    expected = expected * torch_gamma.to(torch.float32).reshape(-1)
    a, b = out.flatten(), expected.flatten()
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    # The perf group's soft precision gate.
    assert pcc > 0.9995, f"{shape}: PCC {pcc}"
