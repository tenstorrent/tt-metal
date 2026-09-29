# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 KV compressor (bead F2, graph node B6).

Ratio 1: ``norm(wkv(x))`` with the checkpoint's bf16 ``wkv``. Ratio 2: fp32 ``wkv`` and ``wgate`` projections,
a per-channel softmax over each non-overlapping pair of tokens, the weighted sum, cast to bf16, then norm
(``inference/model.py`` ``Compressor``). The output is the RoPE-free latent the index keys and the compressed KV
write consume. A trailing incomplete pair (odd valid length) yields no row; its projections are returned as the
carry (``kv_state`` / ``score_state``) the next chunk would complete.

Input ``[1, 1, S/sp, hidden/tp]`` bf16 (normed attention input); output ``[1, 1, S/(sp*r), head_dim]`` bf16,
replicated across TP. Pairs never straddle SP ranks because ``S/sp`` is even (cache geometry).
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.v41.ccl import V41Collectives


class TtV41Compressor(LightweightModule):
    def __init__(self, mesh_device, config, layer: int, weights: dict):
        """``weights``: ``wkv`` [head_dim, hidden], ``norm`` [head_dim], and ``wgate`` [head_dim, hidden] for ratio 2."""
        self.ratio = config.compress_ratio(layer)
        assert self.ratio in (1, 2), f"layer {layer} does not compress (ratio {self.ratio})"
        self.head_dim, self.eps = config.HEAD_DIM, config.RMS_NORM_EPS
        self.ccl = V41Collectives(mesh_device)
        shape = tuple(mesh_device.shape)
        # the ratio-2 pooling runs in fp32 in the reference; ratio 1 is a bf16 projection
        wdtype = ttnn.float32 if self.ratio > 1 else ttnn.bfloat16
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
        )

        def linear_weight(w):
            t = w.detach().float().transpose(-2, -1).contiguous()[None, None]
            return ttnn.from_torch(
                t,
                device=mesh_device,
                dtype=wdtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, shape, dims=(None, 2)),  # row-parallel over hidden
            )

        self.wkv = linear_weight(weights["wkv"])
        self.wgate = linear_weight(weights["wgate"]) if self.ratio > 1 else None
        self.norm = ttnn.from_torch(
            weights["norm"].detach().to(torch.bfloat16).reshape(1, 1, 1, -1),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def _project(self, x, weight, dtype):
        out = ttnn.linear(x, weight, dtype=dtype, compute_kernel_config=self.compute_kernel_config)
        return self.ccl.tp_all_reduce(out)

    def forward(self, x):
        """Returns (latent [1, 1, S/(sp*r), head_dim] bf16, carry) where carry is ``None`` for ratio 1 and
        otherwise the fp32 (kv, score) projections ``[1, 1, S/sp, head_dim]`` the caller slices at an odd
        valid length."""
        if self.ratio == 1:
            kv = self._project(x, self.wkv, ttnn.bfloat16)
            return ttnn.rms_norm(kv, weight=self.norm, epsilon=self.eps), None

        xf = ttnn.typecast(x, ttnn.float32)
        kv = self._project(xf, self.wkv, ttnn.float32)
        score = self._project(xf, self.wgate, ttnn.float32)
        rows = kv.shape[2]
        d = self.head_dim

        def pairs(t):
            # consecutive token rows (2i, 2i+1) become one row [even | odd]
            t = ttnn.reshape(ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT), [1, 1, rows // 2, 2 * d])
            t = ttnn.to_layout(t, ttnn.TILE_LAYOUT)
            return ttnn.slice(t, [0, 0, 0, 0], [1, 1, rows // 2, d]), ttnn.slice(
                t, [0, 0, 0, d], [1, 1, rows // 2, 2 * d]
            )

        k0, k1 = pairs(kv)
        s0, s1 = pairs(score)
        m = ttnn.maximum(s0, s1)
        e0, e1 = ttnn.exp(ttnn.subtract(s0, m)), ttnn.exp(ttnn.subtract(s1, m))
        pooled = ttnn.divide(ttnn.add(ttnn.multiply(k0, e0), ttnn.multiply(k1, e1)), ttnn.add(e0, e1))
        latent = ttnn.rms_norm(ttnn.typecast(pooled, ttnn.bfloat16), weight=self.norm, epsilon=self.eps)
        return latent, (kv, score)
