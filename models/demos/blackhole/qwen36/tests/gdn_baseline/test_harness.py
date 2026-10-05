# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks of the GDN baseline harness: accuracy gates and the reference-to-device conv-state layout.

Run with a mock cluster so importing ttnn touches no hardware:
    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_8xP150.yaml> pytest models/demos/blackhole/qwen36/tests/gdn_baseline/test_harness.py
"""

import pytest
import torch

from models.demos.blackhole.qwen36.tests.gdn_baseline.accuracy import measure
from models.demos.blackhole.qwen36.tests.gdn_baseline.cases import per_device_conv_columns
from models.demos.deepseek_v3_d_p.reference.gdn.layer import GdnShape

TINY = GdnShape(hidden=64, num_k_heads=2, num_v_heads=6, head_k_dim=16, head_v_dim=16, conv_kernel=4, eps=1e-6)


def test_conv_columns_match_tp_weight_grouping():
    """per_device_conv_columns reorders like tp_common.prepare_gdn_qkv reorders projection rows."""
    from models.demos.blackhole.qwen36.tt.tp_common import prepare_gdn_qkv

    tp = 2
    rows = torch.arange(TINY.conv_dim, dtype=torch.float32)[:, None].expand(-1, 3)
    grouped = prepare_gdn_qkv(
        rows,
        TINY.key_dim,
        TINY.value_dim,
        TINY.num_k_heads,
        TINY.head_k_dim,
        TINY.num_v_heads,
        TINY.head_v_dim,
        tp,
    )
    torch.testing.assert_close(per_device_conv_columns(rows.T, TINY, tp), grouped.T)


@pytest.mark.parametrize("scale", [1.05, 0.95])
def test_gates_reject_scale_error_that_pcc_accepts(scale):
    """Negative control: a pure scale error keeps PCC at 1 but fails the relative-RMSE and norm-ratio gates."""
    expected = torch.randn(64, 128, generator=torch.Generator().manual_seed(4))
    clean = measure(expected, expected + 1e-3 * torch.randn_like(expected), "clean")
    assert clean["passed"], clean
    scaled = measure(expected, expected * scale, "scaled")
    assert scaled["pcc"] > 0.99999
    assert not scaled["passed"]
    assert any("norm ratio" in f for f in scaled["failures"])
    assert any("relative RMSE" in f for f in scaled["failures"])


def test_gates_reject_non_finite():
    expected = torch.randn(8, 32)
    actual = expected.clone()
    actual[3, 4] = float("nan")
    assert not measure(expected, actual, "nan")["passed"]
