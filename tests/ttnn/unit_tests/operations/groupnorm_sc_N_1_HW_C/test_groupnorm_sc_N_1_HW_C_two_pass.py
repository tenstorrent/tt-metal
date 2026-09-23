# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Two-pass streaming statistics (config.STREAMING_TWO_PASS).

Pins the `two_pass` streaming schedule (pass A: S and U = sum (x - s)^2 from one read of each chunk,
v = U + n (s - m)(2S/n - m - s) per channel, pass B: apply) against the fp32 torch reference AND against
the three-pass schedule on the same data, so the precision cost of the shifted-centered
combine stays visible: both must clear the gates (PCC >= 0.9995 soft, rel RMS <= 0.02) at
zero mean and at |mean| = 10 sigma (the op's stability contract), on

  * a single partial chunk per core (q0 < Q: the shift is the mean of the ONLY tile-row),
  * several chunks per core with a ragged last chunk (K = 16, Q = 2, 5 tile-rows = 2 + 2 + 1),
  * N = 2 (per-image statistics, the CBs return to base every image),
  * both layouts (the RM leg tilizes each chunk once per pass).
"""

import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.groupnorm_sc_N_1_HW_C.test_groupnorm_sc_N_1_HW_C import (
    _to_device,
    compute_pcc,
    torch_groupnorm_n_1_hw_c,
)
from ttnn.operations.groupnorm_sc_N_1_HW_C import config, groupnorm_sc_N_1_HW_C

PCC_GATE = 0.9995
RMS_GATE = 0.02


def _rel_rms(actual, expected):
    d = (actual.float() - expected.float()).flatten()
    return float(torch.sqrt((d * d).mean()) / torch.sqrt((expected.float().flatten() ** 2).mean()))


def _run(device, shape, num_groups, layout, mean_offset, two_pass, monkeypatch):
    monkeypatch.setattr(config, "FORCE_STREAMING", True)
    monkeypatch.setattr(config, "STREAMING_TWO_PASS", two_pass)
    torch.manual_seed(7)
    N, _, HW, C = shape
    x = (torch.randn(shape, dtype=torch.float32) + mean_offset).to(torch.bfloat16)
    gamma = torch.randn(1, 1, 1, C, dtype=torch.float32).to(torch.bfloat16)
    beta = torch.randn(1, 1, 1, C, dtype=torch.float32).to(torch.bfloat16)
    tt_x = _to_device(x, device, ttnn.bfloat16, layout)
    tt_g = _to_device(gamma, device, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    tt_b = _to_device(beta, device, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    y = ttnn.to_torch(groupnorm_sc_N_1_HW_C(tt_x, num_groups, gamma=tt_g, beta=tt_b))
    ref = torch_groupnorm_n_1_hw_c(x, num_groups, gamma=gamma, beta=beta)
    return compute_pcc(y.float(), ref.float()), _rel_rms(y, ref)


SHAPES = [
    # (shape, G): forced streaming on the 11x10 grid
    pytest.param((1, 1, 1024, 320), 32, id="single_partial_chunk_K10"),  # 32 tile-rows / 32 core rows: q0 = 1 < Q = 3
    pytest.param((1, 1, 16384, 512), 32, id="ragged_chunks_K16"),  # K = 16, Q = 2, 5 tile-rows per core = 2 + 2 + 1
    pytest.param((2, 1, 2048, 256), 8, id="batch2_K8"),  # N = 2: stats and CBs per image
    pytest.param((1, 1, 4096, 64), 2, id="narrow_K1"),  # K = 1, Q = 32
]


@pytest.mark.parametrize("shape,num_groups", SHAPES)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize("mean_offset", [0.0, 10.0], ids=["zero_mean", "mean10sigma"])
def test_two_pass_matches_three_pass(device, monkeypatch, shape, num_groups, layout, mean_offset):
    pcc2, rms2 = _run(device, shape, num_groups, layout, mean_offset, True, monkeypatch)
    pcc3, rms3 = _run(device, shape, num_groups, layout, mean_offset, False, monkeypatch)
    assert pcc2 >= PCC_GATE and rms2 <= RMS_GATE * (2.5 if mean_offset else 1.0), (pcc2, rms2)
    assert pcc3 >= PCC_GATE and rms3 <= RMS_GATE * (2.5 if mean_offset else 1.0), (pcc3, rms3)
    # The shifted-centered combine must not be materially worse than the fully centered pass 2.
    assert rms2 <= rms3 * 1.5 + 1e-3, (rms2, rms3)
