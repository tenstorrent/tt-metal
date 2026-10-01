# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU check of the fused YUV output unpatch (``LTXVideoDecoder._unpatch_yuv_device``).

ttnn.reshape/permute are replaced by their torch equivalents, so this runs without a device. It pins
the element order against the (c,p,r,q) channel oracle and the op shapes that keep the RM tail cheap.
"""

from types import SimpleNamespace

import pytest
import torch

from models.tt_dit.models.vae import vae_ltx


class _RecordingOps:
    def __init__(self):
        self.calls = []

    def reshape(self, x, shape):
        out = x.reshape(shape)
        self.calls.append(("reshape", tuple(x.shape), tuple(out.shape)))
        return out

    def permute(self, x, dims):
        out = x.permute(dims).contiguous()
        self.calls.append(("permute", tuple(x.shape), tuple(out.shape)))
        return out


def _oracle_chwt(packed, patch):
    """(1, T, h, w, 3*patch*patch) -> (3, h*patch, w*patch, T); channel c*16 + r*4 + q, H uses q, W uses r."""
    _, t, h, w, _ = packed.shape
    out = torch.empty(3, h * patch, w * patch, t, dtype=packed.dtype)
    for c in range(3):
        for q in range(patch):
            for r in range(patch):
                out[c, q::patch, r::patch, :] = packed[0, :, :, :, c * patch * patch + r * patch + q].permute(1, 2, 0)
    return out


@pytest.mark.parametrize("t,h,w", [(1, 2, 3), (9, 4, 5), (145, 6, 4)])
def test_unpatch_yuv_device_matches_oracle(monkeypatch, t, h, w):
    patch = 4
    ops = _RecordingOps()
    monkeypatch.setattr(vae_ltx, "ttnn", ops)
    monkeypatch.setattr(vae_ltx, "rgb_chwt_to_yuv_device", lambda x: x)

    # Distinct bf16 values per element so any axis mix-up changes the result.
    packed = (torch.arange(t * h * w * 48, dtype=torch.float32) % 251 - 125).div(64).to(torch.bfloat16)
    packed = packed.reshape(1, t, h, w, 48)

    chwt = vae_ltx.LTXVideoDecoder._unpatch_yuv_device(SimpleNamespace(patch_size=patch), packed)

    assert chwt.shape == (3, h * patch, w * patch, t)
    assert torch.equal(chwt, _oracle_chwt(packed, patch))
    # The 8-D sequence that carried the p=1 axis through the permute.
    previous = packed.reshape(1, t, h, w, 3, 1, patch, patch).permute(0, 4, 2, 7, 3, 6, 1, 5)
    assert torch.equal(chwt, previous.reshape(3, h * patch, w * patch, t))

    # Row-major permute with a size-1 innermost output dim moves 1-element rows, and a reshape that
    # changes the innermost dim is a copy, not a view; both cost ~190 ms/chip at 1080p/145f.
    for op, _, out_shape in ops.calls:
        if op == "permute":
            assert out_shape[-1] == t
    final = ops.calls[-1]
    assert final[0] == "reshape" and final[1][-1] == final[2][-1]
