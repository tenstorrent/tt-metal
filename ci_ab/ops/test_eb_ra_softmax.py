# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Regression audit: ttnn.softmax (the attention softmax kernel, softmax.cpp, opted in to the per-tile broadcast multiply)
at the shapes its records name (Whisper decoder, gpt-oss routing). Not for merge."""
import os as _os_guard, sys as _sys_guard
if not (_os_guard.environ.get("HWLOCK_HELD") or (_os_guard.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH") and _os_guard.path.isfile(_os_guard.environ["TT_METAL_MOCK_CLUSTER_DESC_PATH"]))):
    _sys_guard.exit("not under hwlock and no mock cluster descriptor")
import pytest
import torch
import ttnn


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=32768)
    yield dev
    ttnn.close_device(dev)


@pytest.mark.parametrize("shape", [(1, 20, 32, 32), (2, 20, 32, 32), (1, 1, 640, 32), (1, 1, 640, 256)], ids=["whisper_1x20", "whisper_2x20", "gptoss_640x4", "gptoss_640x256"])
def test_softmax_attention(device, shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.softmax(t, dim=-1)
    got = ttnn.to_torch(out).float()
    ref = torch.softmax(x.float(), dim=-1)
    assert torch.allclose(got, ref, atol=0.02, rtol=0.02)
