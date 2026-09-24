# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Zero-centred RMSNorm (``norm(x) * (1 + w)``), composed from primitives in fp32.

Structure from minimax_m3/tt/rms_norm.py (the ``+1`` folded into the gain at load, gain replicated),
with one measured deviation: the gain is kept **fp32** and the norm is composed as
``x * rsqrt(mean(x^2) + eps) * gain`` instead of calling ``ttnn.rms_norm``:

* the folded gain ``1 + w`` sits next to 1.0, where bf16's step is 2^-7, so ~87% of Qwen3.8's gain
  entries change when rounded to bf16 and the norm output picks up a systematic +0.1% per-token
  scale (measured on the real layer-31 residual: scale ratio 1.00096 -> 0.99998 with the fp32 gain).
  One such bias per norm, 129 norms deep, is the kind of coherent error this model amplifies.
* ``ttnn.rms_norm`` rejects an fp32 gain, and its single-pass row buffers for a 5120-wide fp32 row
  (the fp32 residual stream) exceed L1.

Output is the bf16 activation dtype.
"""

import torch

import ttnn
from models.demos.qwen_3_8_27b.tt.common import upload


def rms_norm_fp32(x, gain32, eps):
    """x (any dtype) ``[.., D]``, gain32 fp32 TILE ``[1, 1, 1, D]`` -> bf16 ``[.., D]``."""
    x32 = x if x.dtype == ttnn.float32 else ttnn.typecast(x, ttnn.float32)
    sq = ttnn.multiply(x32, x32)
    ms = ttnn.mean(sq, dim=-1, keepdim=True)
    ttnn.deallocate(sq)
    ms_eps = ttnn.add(ms, eps)
    ttnn.deallocate(ms)
    inv = ttnn.rsqrt(ms_eps)
    ttnn.deallocate(ms_eps)
    y = ttnn.multiply(x32, inv)
    ttnn.deallocate(inv)
    if x32 is not x:
        ttnn.deallocate(x32)
    out = ttnn.multiply(y, gain32, dtype=ttnn.bfloat16)
    ttnn.deallocate(y)
    return out


def gain_tensor(mesh_config, weight, *, unit_offset, cache=None, name=None):
    """fp32 TILE ``[1, 1, 1, D]`` gain (``1 + w`` when ``unit_offset``), replicated."""
    host = None
    if weight is not None:
        host = (weight.float() + (1.0 if unit_offset else 0.0)).reshape(1, 1, 1, -1)
    return upload(
        host,
        mesh_config.mesh_device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        mapper=mesh_config.replicate(),
        cache=cache,
        name=None if name is None else f"{name}_g32",
    )


class TtRMSNorm:
    def __init__(
        self, mesh_config, weight: torch.Tensor | None, eps: float, *, unit_offset=True, cache=None, name=None
    ):
        self.eps = eps
        self.gain = gain_tensor(mesh_config, weight, unit_offset=unit_offset, cache=cache, name=name)

    def __call__(self, x):
        return rms_norm_fp32(x, self.gain, self.eps)
