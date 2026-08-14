# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm — decode perf harness at the perf target's EXACT config.

`test_rms_norm_perf.py` measures with a ROW_MAJOR gamma and the Phase 0
compute-config default (HiFi4 / fp32_dest_acc_en=True).  The perf target in
`feature_spec.LOOSE_CASES` ("perf" group) is a DIFFERENT cell:

    bf16 input, TILE layout, INTERLEAVED,
    gamma: bf16, **TILE layout**,
    math_fidelity=HiFi2, fp32_dest_acc_en=False

That difference is not cosmetic — a TILE-layout gamma is a (1, 1, 1, W) tensor
padded to a full tile-row, so it occupies `Wt` whole 32x32 tiles of which only
row 0 carries data.  In the decode regime (Rt == 1) gamma is therefore the same
number of DRAM pages as x itself.

Run:

    scripts/run_safe_pytest.sh --profile --run-all \
        tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_perf_decode.py

Each case runs ONCE — device kernel time has no warm-up transient.

Ablation / knob switches are FILES, not env vars: under --profile the measured
run lives in a `python -m tracy` child, and an ad-hoc env var does not reach it.

    echo no_gamma > /tmp/rms_norm_ablate       # gamma=None: costs out gamma DRAM + the apply
    echo 8        > /tmp/rms_norm_hidden_floor # HIDDEN_TILES_PER_CORE_FLOOR (bounds S)
    echo 32       > /tmp/rms_norm_max_slices   # MAX_HIDDEN_SLICES        (bounds the fan-in s)

CAVEAT that invalidated the two knob A/Bs recorded in test_rms_norm_perf.py:
`monkeypatch.setattr(pd, KNOB, v)` on the module imported as
`ttnn.operations.rms_norm.rms_norm_program_descriptor` patches a SECOND import of
that file, not the dict the op executes in — so the "flat within 1-2%" floor and
rect sweeps in that file measured the shipped configuration four times over.
Patch `create_program_descriptor.__globals__` instead (see PLAN_GLOBALS below);
re-run honestly, the floor knob moves the decode wall by up to 25%.

Measured, Blackhole p150b (11x10 grid), TARGET CONFIG, ONE fresh run per point.
`s` = hidden slices (= cores here, since Rt == 1 gives one row-group):

    variant                        w1024  w2304  w5120  w7168  floor_1tile
    Refinement 2 baseline           5609   6615   9428  11340   3422
    + mcast pre-handshake off       5495   6552   9336  11359   3363
    + gamma row-0-only read         5525   6547   9316  11049   3343
    + MAX_HIDDEN_SLICES = 32        5551   6548   8445   9732   3355
    (shipped)                      -1.0%  -1.0% -10.4% -14.2%

The fan-in cap is the dominant lever and it is U-shaped in `s` (see
MAX_HIDDEN_SLICES in the program descriptor for the full sweep).
"""

import pathlib

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm.rms_norm import create_program_descriptor as _create_program_descriptor

# The module dict the descriptor ACTUALLY executes in.
#
# `import ttnn.operations.rms_norm.rms_norm_program_descriptor as pd` yields a
# module object whose `__file__` is the right file but whose `__dict__` is a
# SECOND import of it (the package is reachable under two names), so
# `monkeypatch.setattr(pd, "KNOB", v)` patches a copy nobody runs — an A/B that
# silently measures the same variant twice.  Patch the globals of the function
# the op really calls instead; that dict is the one `_plan` reads its knobs from.
PLAN_GLOBALS = _create_program_descriptor.__globals__

# The perf group's fixed extras, spelled once.
TARGET_FIDELITY = ttnn.MathFidelity.HiFi2
TARGET_FP32_ACC = False

DECODE_SHAPES = [
    pytest.param((1, 1, 32, 1024), id="decode_w1024"),
    pytest.param((1, 1, 32, 2304), id="decode_w2304"),
    pytest.param((1, 1, 32, 5120), id="decode_w5120"),
    pytest.param((1, 1, 32, 7168), id="decode_w7168"),
    pytest.param((32, 32), id="floor_1tile"),
]

_ABLATE = pathlib.Path("/tmp/rms_norm_ablate")
ABLATE = _ABLATE.read_text().strip() if _ABLATE.exists() else "none"

# Hidden-granularity floor sweep, same selector as test_rms_norm_perf.py:
#     echo 8 > /tmp/rms_norm_hidden_floor
# Higher floor => fewer, fatter hidden slices (bigger per-core NoC transfers, a
# smaller combine fan-in); lower floor => more cores, thinner transfers.
_FLOOR_SELECTOR = pathlib.Path("/tmp/rms_norm_hidden_floor")
HIDDEN_FLOOR = int(_FLOOR_SELECTOR.read_text().strip()) if _FLOOR_SELECTOR.exists() else None

# Combine fan-in cap sweep (the other end of the same tradeoff):
#     echo 20 > /tmp/rms_norm_max_slices
_MAXS_SELECTOR = pathlib.Path("/tmp/rms_norm_max_slices")
MAX_SLICES = int(_MAXS_SELECTOR.read_text().strip()) if _MAXS_SELECTOR.exists() else None


@pytest.fixture(autouse=True)
def hidden_floor(monkeypatch):
    if HIDDEN_FLOOR is not None:
        monkeypatch.setitem(PLAN_GLOBALS, "HIDDEN_TILES_PER_CORE_FLOOR", HIDDEN_FLOOR)
    if MAX_SLICES is not None:
        monkeypatch.setitem(PLAN_GLOBALS, "MAX_HIDDEN_SLICES", MAX_SLICES)
    return HIDDEN_FLOOR


def target_compute_config():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=TARGET_FIDELITY,
        fp32_dest_acc_en=TARGET_FP32_ACC,
        math_approx_mode=False,
    )


@pytest.mark.parametrize("shape", DECODE_SHAPES)
def test_rms_norm_perf_decode(device, shape):
    torch.manual_seed(42)
    torch_x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    torch_gamma = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32).to(torch.bfloat16)

    x = ttnn.from_torch(
        torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    # TILE-layout gamma — the perf target's config.
    gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    use_gamma = None if ABLATE == "no_gamma" else gamma
    out = ttnn.to_torch(rms_norm(x, gamma=use_gamma, compute_kernel_config=target_compute_config())).to(torch.float32)

    xf = torch_x.to(torch.float32)
    expected = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    if use_gamma is not None:
        expected = expected * torch_gamma.to(torch.float32).reshape(-1)
    a, b = out.flatten(), expected.flatten()
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    # The perf group's soft precision gate.
    assert pcc > 0.9995, f"{shape}: PCC {pcc}"
