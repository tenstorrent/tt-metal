# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm — PREFILL perf harness at the perf target's EXACT config.

The `perf` loose-case group in `eval/golden_tests/rms_norm/feature_spec.py`
carries four interleaved *prefill* profiles — `(1, 1, 8192, W)` for
W in {1024, 2304, 5120, 7168}, `achievable_ns` 96744 / 211345 / 738307 /
1032281 on blackhole_p150b @ 1350 MHz — all at:

    bf16 input, TILE layout, INTERLEAVED,
    gamma: bf16, **TILE layout**,
    math_fidelity=HiFi2, fp32_dest_acc_en=False

This is the bandwidth-bound regime: Rt = 256 tile-rows, so rows over-fill the
grid and the plan lands at a large `core_row_tiles` with a small (often 1)
hidden split.

Run:

    scripts/run_safe_pytest.sh --profile --run-all \
        tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_perf_prefill.py

Each case runs ONCE — device kernel time has no warm-up transient.

Knob / ablation switches are FILES, not env vars: under `--profile` the measured
run lives in a `python -m tracy` child and an ad-hoc env var does not reach it.

    echo 32   > /tmp/rms_norm_dm_chunk      # DM_CHUNK_TILES (tiles per NoC barrier)
    echo 2    > /tmp/rms_norm_in_depth      # IN_CB_DEPTH    (input double buffer)
    echo 1.46 > /tmp/rms_norm_l1_mb         # L1_SIZE_PER_CORE_FALLBACK, in MB
    echo no_read   > /tmp/rms_norm_ablate   # ablation payload stubs (kernel-side)
    echo no_compute> /tmp/rms_norm_ablate
    echo no_write  > /tmp/rms_norm_ablate
    echo no_payload> /tmp/rms_norm_ablate

The patch target is `create_program_descriptor.__globals__`, NOT the module
object obtained by importing the descriptor file: the package is reachable under
two names, so `monkeypatch.setattr(pd, KNOB, v)` patches a second import that
nobody runs (this silently voided two A/B tables in `test_rms_norm_perf.py`).

Measured, Blackhole p150b (11x10 grid), TARGET CONFIG, one fresh run per point.
See the changelog for the full table.
"""

import pathlib

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm.rms_norm import create_program_descriptor as _create_program_descriptor

PLAN_GLOBALS = _create_program_descriptor.__globals__

# The perf group's fixed extras, spelled once.
TARGET_FIDELITY = ttnn.MathFidelity.HiFi2
TARGET_FP32_ACC = False

PREFILL_SHAPES = [
    pytest.param((1, 1, 8192, 1024), 96744, id="prefill_w1024"),
    pytest.param((1, 1, 8192, 2304), 211345, id="prefill_w2304"),
    pytest.param((1, 1, 8192, 5120), 738307, id="prefill_w5120"),
    pytest.param((1, 1, 8192, 7168), 1032281, id="prefill_w7168"),
]


def _read(path, cast, default=None):
    p = pathlib.Path(path)
    return cast(p.read_text().strip()) if p.exists() else default


DM_CHUNK = _read("/tmp/rms_norm_dm_chunk", int)
IN_DEPTH = _read("/tmp/rms_norm_in_depth", int)
L1_MB = _read("/tmp/rms_norm_l1_mb", float)
ABLATE = _read("/tmp/rms_norm_ablate", str, "none")


@pytest.fixture(autouse=True)
def knobs(monkeypatch):
    if DM_CHUNK is not None:
        monkeypatch.setitem(PLAN_GLOBALS, "DM_CHUNK_TILES", DM_CHUNK)
    if IN_DEPTH is not None:
        monkeypatch.setitem(PLAN_GLOBALS, "IN_CB_DEPTH", IN_DEPTH)
    if L1_MB is not None:
        monkeypatch.setitem(PLAN_GLOBALS, "L1_SIZE_PER_CORE_FALLBACK", int(L1_MB * 1024 * 1024))
    if ABLATE != "none":
        monkeypatch.setitem(PLAN_GLOBALS, "ABLATE", ABLATE)
    return (DM_CHUNK, IN_DEPTH, L1_MB, ABLATE)


def target_compute_config():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=TARGET_FIDELITY,
        fp32_dest_acc_en=TARGET_FP32_ACC,
        math_approx_mode=False,
    )


@pytest.mark.parametrize("shape,achievable_ns", PREFILL_SHAPES)
def test_rms_norm_perf_prefill(device, shape, achievable_ns, knobs):
    torch.manual_seed(42)
    torch_x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    torch_gamma = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32).to(torch.bfloat16)

    x = ttnn.from_torch(
        torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out = ttnn.to_torch(rms_norm(x, gamma=gamma, compute_kernel_config=target_compute_config())).to(torch.float32)

    if ABLATE != "none":
        return  # payload stubbed: the numbers are meaningless, only the ns matter

    xf = torch_x.to(torch.float32)
    expected = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    expected = expected * torch_gamma.to(torch.float32).reshape(-1)
    a, b = out.flatten(), expected.flatten()
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    # The perf group's soft precision gate.
    assert pcc > 0.9995, f"{shape}: PCC {pcc}"
