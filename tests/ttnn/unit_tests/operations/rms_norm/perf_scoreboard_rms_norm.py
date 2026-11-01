# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""rms_norm perf scoreboard (verifier tool; not collected by a directory run — no `test_` file prefix).

Runs one op dispatch per cell through the golden `observed()` wrapper so the eval runner records the
grader's `device_kernel_ns`:

    eval/eval_test_runner.sh --no-precompile \
        tests/ttnn/unit_tests/operations/rms_norm/perf_scoreboard_rms_norm.py <out_dir>
    # then read <out_dir>/test_results.json -> device_kernel_ns, device_num_cores

Shapes = the LLM-derived interleaved perf profiles of eval/golden_tests/rms_norm/feature_spec.py
(`_perf_case`, TILE input, TILE gamma) plus the wide/few-row W-split shapes. CONFIG is the op's current
precision contract; once `fp32_dest_acc_en=False` is in SUPPORTED, set FP32_DEST_ACC_EN = False and
MATH_FIDELITY = HiFi2 to measure the perf profiles at their own config (never a proxy for acceptance).
"""

import pytest
import torch
import ttnn

from eval.golden_tests.rms_norm.axes import observed as rms_norm

FP32_DEST_ACC_EN = True
MATH_FIDELITY = ttnn.MathFidelity.HiFi4

SHAPES = [
    # decode (1 tile-row) — W split across <= 8 Tensix cores
    (1, 1, 32, 1024),
    (1, 1, 32, 2304),
    (1, 1, 32, 5120),
    (1, 1, 32, 7168),
    (1, 1, 32, 16384),
    (1, 1, 32, 32768),
    # mid (20 tile-rows) and prefill
    (1, 1, 640, 6144),
    (1, 1, 640, 7168),
    (1, 1, 1024, 1024),
    (1, 1, 7168, 1024),
    (1, 1, 7168, 2048),
    (1, 1, 8192, 1024),
    (1, 1, 8192, 2304),
    (1, 1, 8192, 5120),
    (1, 1, 8192, 7168),
]


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_scoreboard(device, shape):
    torch.manual_seed(0)
    x = torch.randn(shape).bfloat16()
    g = torch.randn(shape[-1]).bfloat16()
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tg = ttnn.from_torch(g.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = MATH_FIDELITY
    cfg.fp32_dest_acc_en = FP32_DEST_ACC_EN
    cfg.math_approx_mode = False
    y = ttnn.to_torch(rms_norm(tx, gamma=tg, compute_kernel_config=cfg)).float()
    xf = x.float()
    ref = xf / torch.sqrt((xf * xf).mean(-1, keepdim=True) + 1e-6) * g.float()
    pcc = torch.corrcoef(torch.stack([y.flatten().double(), ref.flatten().double()]))[0, 1].item()
    assert pcc >= 0.999, pcc
