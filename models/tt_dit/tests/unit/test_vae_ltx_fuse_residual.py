# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""With LTX_VAE_FUSE_RESIDUAL=0 the resnet block runs conv2 and then ttnn.add as
before, and with it on conv2 takes the skip tensor only when conv3d's residual epilogue can read it. No
device."""

from types import SimpleNamespace

import pytest

import ttnn

from ...models.vae import vae_ltx


def _residual(dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mem=ttnn.DRAM_MEMORY_CONFIG):
    return SimpleNamespace(dtype=dtype, layout=layout, memory_config=lambda: mem)


def _conv(out_channels=128, unpadded_out_channels=128, dtype=ttnn.bfloat16):
    return SimpleNamespace(out_channels=out_channels, unpadded_out_channels=unpadded_out_channels, dtype=dtype)


@pytest.mark.parametrize(
    ("conv", "residual", "expected"),
    [
        (_conv(), _residual(), True),
        (_conv(dtype=ttnn.float32), _residual(), False),  # the epilogue is bf16 only
        (_conv(out_channels=128, unpadded_out_channels=96), _residual(), False),  # padded out_channels
        (_conv(out_channels=48, unpadded_out_channels=48), _residual(), False),  # not tile aligned
        (_conv(), _residual(dtype=ttnn.float32), False),
        (_conv(), _residual(layout=ttnn.TILE_LAYOUT), False),
        (_conv(), _residual(mem=ttnn.L1_MEMORY_CONFIG), False),
    ],
)
def test_can_fuse_residual(conv, residual, expected):
    assert vae_ltx.LTXCausalConv3d.can_fuse_residual(conv, residual) is expected


class _Recorder:
    def __init__(self, calls, name, can_fuse=True):
        self.calls, self.name, self.can_fuse = calls, name, can_fuse

    def __call__(self, x, **kwargs):
        self.calls.append((self.name, kwargs.get("residual")))
        return f"{self.name}_out"

    def can_fuse_residual(self, residual):
        return self.can_fuse


def _run_block(monkeypatch, fuse_residual, can_fuse=True):
    calls = []
    monkeypatch.setattr(vae_ltx.ttnn, "to_layout", lambda t, layout: t)
    monkeypatch.setattr(vae_ltx.ttnn, "add", lambda a, b: calls.append(("add", (a, b))) or "add_out")
    block = SimpleNamespace(
        norm1=lambda x, **kw: x,
        norm2=lambda x, **kw: x,
        conv1=_Recorder(calls, "conv1"),
        conv2=_Recorder(calls, "conv2", can_fuse),
        norm_compute_kernel_config=None,
        has_shortcut=False,
        fuse_residual=fuse_residual,
    )
    out = vae_ltx.LTXResnetBlock3D.forward(block, "x")
    return out, calls


def test_block_unfused_by_default(monkeypatch):
    out, calls = _run_block(monkeypatch, fuse_residual=False)
    assert calls == [("conv1", None), ("conv2", None), ("add", ("x", "conv2_out"))]
    assert out == "add_out"


def test_block_fused_passes_skip_to_conv2(monkeypatch):
    out, calls = _run_block(monkeypatch, fuse_residual=True)
    assert calls == [("conv1", None), ("conv2", "x")]
    assert out == "conv2_out"


def test_block_fused_falls_back_when_unsupported(monkeypatch):
    out, calls = _run_block(monkeypatch, fuse_residual=True, can_fuse=False)
    assert calls == [("conv1", None), ("conv2", None), ("add", ("x", "conv2_out"))]
    assert out == "add_out"
