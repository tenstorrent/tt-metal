# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 64),  # Wt=2: one block
        (1, 1, 32, 256),  # Wt=8: several blocks, partial sums
        (1, 1, 96, 1024),  # 3 rows: uneven across 4 NEOs
        (2, 3, 64, 2048),  # many rows, Wt=64
        (1, 1, 32, 96),  # Wt=3: batch 3, block 2 + 1
    ],
)
@pytest.mark.parametrize("numeric_stable", [True, False])
def test_quasar_softmax(device, shape, numeric_stable):
    if device.arch() != ttnn.device.Arch.QUASAR:
        pytest.skip("Quasar-native kernels (DFB implicit sync, multi-hart DM / NEO threading)")
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    if not numeric_stable:
        x = x.clamp(-4, 4)  # keep exp(x) in range without the max subtraction
    ref = torch.softmax(x.float(), dim=-1)

    tx = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.experimental.quasar.softmax(tx, dim=-1, numeric_stable=numeric_stable)
    out = ttnn.to_torch(out).float()
    assert_with_pcc(ref, out, 0.999)


# ---- contract rejections (host-side validation, run on every arch) ----


def test_quasar_softmax_rejects_unaligned_width(device, expect_error):
    x = ttnn.from_torch(torch.randn((1, 1, 32, 100), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "must be a multiple of 32 with no padded columns"):
        ttnn.experimental.quasar.softmax(x, dim=-1)


def test_quasar_softmax_rejects_non_last_dim(device, expect_error):
    x = ttnn.from_torch(torch.randn((1, 1, 32, 64), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "only implements the last dimension"):
        ttnn.experimental.quasar.softmax(x, dim=2)


def test_quasar_softmax_rejects_fp32_input(device, expect_error):
    x = ttnn.from_torch(torch.randn((1, 1, 32, 64), dtype=torch.float32), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "input must be BFLOAT16"):
        ttnn.experimental.quasar.softmax(x, dim=-1)


def test_quasar_softmax_rejects_row_major_input(device, expect_error):
    x = ttnn.from_torch(torch.randn((1, 1, 32, 64), dtype=torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "must have TILE layout"):
        ttnn.experimental.quasar.softmax(x, dim=-1)


@pytest.mark.parametrize(
    "plan, message",
    [
        ("3,4,2", "must be a multiple of reader threads"),
        ("2,4,3", "must be a multiple of writer threads"),
        ("4,8,2", "exceed the 4 NEOs"),
        ("4,4,4", "exceed the 6 free DM harts"),
        ("-1,4,2", 'must be "R,C,W"'),
        ("4,4,2x", 'must be "R,C,W"'),
    ],
)
def test_quasar_softmax_rejects_bad_thread_plan(device, expect_error, monkeypatch, plan, message):
    monkeypatch.setenv("TTNN_QSR_SOFTMAX_THREADS", plan)
    x = ttnn.from_torch(torch.randn((1, 1, 32, 64), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, message):
        ttnn.experimental.quasar.softmax(x, dim=-1)


@pytest.mark.parametrize("plan", ["1,4,1", "2,4,4", "1,1,1"])
def test_quasar_softmax_thread_plan_override(device, monkeypatch, plan):
    if device.arch() != ttnn.device.Arch.QUASAR:
        pytest.skip("Quasar-native kernels (DFB implicit sync, multi-hart DM / NEO threading)")
    monkeypatch.setenv("TTNN_QSR_SOFTMAX_THREADS", plan)
    torch.manual_seed(0)
    x = torch.randn((1, 1, 96, 256), dtype=torch.bfloat16)
    ref = torch.softmax(x.float(), dim=-1)
    tx = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(ttnn.experimental.quasar.softmax(tx, dim=-1)).float()
    assert_with_pcc(ref, out, 0.999)
