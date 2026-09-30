# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TTNN port of BEVFormer's detection decoder, from UniAD's ``TtDetectionTransformerDecoder``.

Tensors stay sequence-first ``(num_query, bs, embed_dims)`` between layers, as in the
reference. Parameters come from ``model_preprocessing_decoder.create_decoder_parameters``.
Forward runs on device only.

Reference points are float32, as is the sampling grid built from them (``GRID_DTYPE``).
In bfloat16 a point in (0.5, 1) moves in steps of 2^-8, 0.8 px on the 200x200 BEV grid,
and the error compounds across layers through the reference-point refinement.
"""

import math

import ttnn
from models.experimental.bevformer.config import DeformableAttentionConfig
from models.experimental.bevformer.reference.decoder import REG_XY, REG_Z
from models.experimental.bevformer.tt.tt_ms_deformable_attention import TTMSDeformableAttention

GRID_DTYPE = ttnn.float32


def _linear_rows(x, weight, bias):
    """``ttnn.linear`` with every leading dim folded into M.

    A sequence-first ``(nq, bs, C)`` tensor otherwise runs as ``nq`` matmuls of ``bs`` rows,
    each padded to a 32-row tile, and the matmul heuristic sizes its core grid from ``bs``.
    Skips the reshape when the second-to-last dim already holds every row.
    """
    shape = list(x.shape)
    rows = math.prod(shape[:-1])
    if len(shape) < 3 or rows == shape[-2]:
        return ttnn.linear(x, weight, bias=bias)
    y = ttnn.linear(ttnn.reshape(x, (1, 1, rows, shape[-1])), weight, bias=bias)
    return ttnn.reshape(y, tuple(shape[:-1]) + (y.shape[-1],))


def inverse_sigmoid(x, eps=1e-5):
    x = ttnn.clamp(x, min=0, max=1)
    x1 = ttnn.clamp(x, min=eps)
    x2 = ttnn.clamp(ttnn.rsub(x, 1.0), min=eps)
    return ttnn.log(ttnn.div(x1, x2))


class TtMultiheadAttention:
    """``params.qk_proj`` packs Q (pre-scaled by ``head_dim**-0.5``) and K into one Linear."""

    def __init__(self, params):
        self.params = params
        self.num_heads = params.num_heads

    def __call__(self, query, query_pos):
        p = self.params
        tgt_len, bsz, embed_dims = query.shape
        head_dim = embed_dims // self.num_heads

        qk = _linear_rows(ttnn.add(query, query_pos), p.qk_proj.weight, p.qk_proj.bias)
        q = qk[..., :embed_dims]
        k = qk[..., embed_dims:]
        v = _linear_rows(query, p.v_proj.weight, p.v_proj.bias)

        q = ttnn.permute(ttnn.reshape(q, (tgt_len, bsz * self.num_heads, head_dim)), (1, 0, 2))
        k = ttnn.permute(ttnn.reshape(k, (tgt_len, bsz * self.num_heads, head_dim)), (1, 2, 0))
        v = ttnn.permute(ttnn.reshape(v, (tgt_len, bsz * self.num_heads, head_dim)), (1, 0, 2))

        attn = ttnn.softmax(ttnn.matmul(q, k), dim=-1)
        out = ttnn.matmul(attn, v)

        out = ttnn.reshape(ttnn.permute(out, (1, 0, 2)), (tgt_len, bsz, embed_dims))
        out = _linear_rows(out, p.out_proj.weight, p.out_proj.bias)
        return ttnn.add(out, query)


class TtFFN:
    def __init__(self, params):
        self.params = params

    def __call__(self, x):
        p = self.params
        y = ttnn.relu(_linear_rows(x, p.linear1.weight, p.linear1.bias))
        y = _linear_rows(y, p.linear2.weight, p.linear2.bias)
        return ttnn.add(y, x)


class TtDetrTransformerDecoderLayer:
    def __init__(self, params, device, bev_shape, grid_sample_compute_config):
        self.params = params
        self.self_attn = TtMultiheadAttention(params.self_attn)
        cross = params.cross_attn
        self.cross_attn = TTMSDeformableAttention(
            DeformableAttentionConfig(
                embed_dims=cross.embed_dims,
                num_heads=cross.num_heads,
                num_levels=1,
                num_points=cross.num_points,
                batch_first=False,
            ),
            device,
            cross,
            spatial_shapes=[list(bev_shape)],
            grid_dtype=GRID_DTYPE,
            grid_sample_compute_config=grid_sample_compute_config,
        )
        self.ffn = TtFFN(params.ffn)

    def _norm(self, x, index):
        norm = self.params.norms[index]
        return ttnn.layer_norm(x, weight=norm.weight, bias=norm.bias, epsilon=norm.eps)

    def __call__(self, query, value, query_pos, reference_points):
        """``reference_points`` is ``(bs, nq, 1, 2)`` in [0, 1]."""
        query = self._norm(self.self_attn(query, query_pos), 0)
        query = self.cross_attn(query=query, value=value, query_pos=query_pos, reference_points=reference_points)
        query = self._norm(query, 1)
        return self._norm(self.ffn(query), 2)


class TtDetectionTransformerDecoder:
    """Decoder over a ``bev_shape`` ``(bev_h, bev_w)`` BEV map.

    The cross-attention folds ``bev_shape`` into its sampling-offset Linear, so the
    constructor consumes ``params.layers[*].cross_attn.sampling_offsets``; build a fresh
    ``params`` per instance.
    """

    def __init__(self, params, device, bev_shape):
        # fp32 accumulation of grid_sample's weighted 4-corner sum.
        grid_sample_compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
            math_approx_mode=False,
        )
        self.layers = [
            TtDetrTransformerDecoderLayer(p, device, bev_shape, grid_sample_compute_config) for p in params.layers
        ]

    @staticmethod
    def _reg_branch(x, branch):
        x = ttnn.relu(_linear_rows(x, branch[0].weight, branch[0].bias))
        x = ttnn.relu(_linear_rows(x, branch[1].weight, branch[1].bias))
        return _linear_rows(x, branch[2].weight, branch[2].bias)

    def __call__(self, query, value, query_pos, reference_points, reg_branches):
        """``reference_points`` is ``(bs, nq, 3)`` ``GRID_DTYPE`` in [0, 1].

        Returns every layer's output ``(L, nq, bs, C)`` and refined reference points ``(L, bs, nq, 3)``.
        """
        if reference_points.dtype != GRID_DTYPE:
            raise ValueError(f"reference_points must be {GRID_DTYPE}, got {reference_points.dtype}")
        output = query
        intermediate = []
        intermediate_reference_points = []
        for layer, branch in zip(self.layers, reg_branches, strict=True):
            output = layer(output, value, query_pos, ttnn.unsqueeze(reference_points[..., :2], 2))

            tmp = ttnn.typecast(self._reg_branch(ttnn.permute(output, (1, 0, 2)), branch), GRID_DTYPE)
            updated_xy = ttnn.add(tmp[..., REG_XY], inverse_sigmoid(reference_points[..., :2]))
            updated_z = ttnn.add(tmp[..., REG_Z], inverse_sigmoid(reference_points[..., 2:3]))
            reference_points = ttnn.sigmoid(ttnn.concat([updated_xy, updated_z], dim=-1))

            intermediate.append(output)
            intermediate_reference_points.append(reference_points)

        return ttnn.stack(intermediate, dim=0), ttnn.stack(intermediate_reference_points, dim=0)
