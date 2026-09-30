# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TTNN port of BEVFormer's detection decoder, from UniAD's ``TtDetectionTransformerDecoder``
(``models/experimental/uniad/tt/ttnn_decoder.py``).

Tensors stay sequence-first ``(num_query, bs, embed_dims)`` between layers, as in the
reference. Parameters come from ``model_preprocessing_decoder.create_decoder_parameters``.
Forward runs on device only.

Reference points are float32, as is the sampling grid built from them (``GRID_DTYPE``).
In bfloat16 a point in (0.5, 1) moves in steps of 2^-8, 0.8 px on the 200x200 BEV grid,
and the error compounds across layers through the reference-point refinement.
"""

import math

import ttnn
from models.experimental.bevformer.config.decoder_config import REG_XY, REG_Z
from models.experimental.bevformer.tt.tt_common import layer_norm
from models.experimental.bevformer.tt.tt_ms_deformable_attention import TTMSDeformableAttention, fp32_grid_sample_config

GRID_DTYPE = ttnn.float32


def _linear_rows(x, weight, bias, dtype=None):
    """``ttnn.linear`` with every leading dim folded into M.

    A sequence-first ``(nq, bs, C)`` tensor otherwise runs as ``nq`` matmuls of ``bs`` rows,
    each padded to a 32-row tile, and the matmul heuristic sizes its core grid from ``bs``.
    Skips the reshape when the second-to-last dim already holds every row.
    """
    shape = list(x.shape)
    rows = math.prod(shape[:-1])
    if len(shape) < 3 or rows == shape[-2]:
        return ttnn.linear(x, weight, bias=bias, dtype=dtype)
    y = ttnn.linear(ttnn.reshape(x, (1, 1, rows, shape[-1])), weight, bias=bias, dtype=dtype)
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
        self.cross_attn = TTMSDeformableAttention(
            params.cross_attn.config,
            device,
            params.cross_attn,
            spatial_shapes=[list(bev_shape)],
            grid_dtype=GRID_DTYPE,
            grid_sample_compute_config=grid_sample_compute_config,
        )
        self.ffn = TtFFN(params.ffn)

    def __call__(self, query, bev_value, query_pos, bev_query_pos, reference_points):
        """Sequence-first ``query``/``query_pos``; ``bev_value``/``bev_query_pos`` batch-first.

        The cross-attention runs batch-first, so the large BEV value is permuted once per
        decoder, not once per layer. ``reference_points`` is ``(bs, nq, 1, 2)`` in [0, 1].
        """
        norms = self.params.norms
        query = layer_norm(self.self_attn(query, query_pos), norms[0])
        query = self.cross_attn(
            query=ttnn.permute(query, (1, 0, 2)),
            value=bev_value,
            query_pos=bev_query_pos,
            reference_points=reference_points,
        )
        query = layer_norm(ttnn.permute(query, (1, 0, 2)), norms[1])
        return layer_norm(self.ffn(query), norms[2])


class TtDetectionTransformerDecoder:
    """Decoder over a ``bev_shape`` ``(bev_h, bev_w)`` BEV map.

    The cross-attention folds ``bev_shape`` into its sampling-offset Linear, which consumes
    ``params.layers[*].cross_attn.sampling_offsets``: each instance needs its own
    ``create_decoder_parameters``, and reusing them raises.
    """

    def __init__(self, params, device, bev_shape):
        grid_sample_compute_config = fp32_grid_sample_config(device)
        self.layers = [
            TtDetrTransformerDecoderLayer(p, device, bev_shape, grid_sample_compute_config) for p in params.layers
        ]

    @staticmethod
    def _reg_branch(x, branch):
        x = ttnn.relu(_linear_rows(x, branch[0].weight, branch[0].bias))
        x = ttnn.relu(_linear_rows(x, branch[1].weight, branch[1].bias))
        # Emitted in GRID_DTYPE: its output is added to the reference points' logits.
        return _linear_rows(x, branch[2].weight, branch[2].bias, dtype=GRID_DTYPE)

    def __call__(self, query, value, query_pos, reference_points, reg_branches):
        """``reference_points`` is ``(bs, nq, 3)`` ``GRID_DTYPE`` in [0, 1].

        Returns every layer's output ``(L, nq, bs, C)`` and refined reference points ``(L, bs, nq, 3)``.
        """
        if reference_points.dtype != GRID_DTYPE:
            raise ValueError(f"reference_points must be {GRID_DTYPE}, got {reference_points.dtype}")
        bev_value = ttnn.permute(value, (1, 0, 2))
        bev_query_pos = ttnn.permute(query_pos, (1, 0, 2))
        output = query
        intermediate = []
        intermediate_reference_points = []
        for layer, branch in zip(self.layers, reg_branches, strict=True):
            output = layer(output, bev_value, query_pos, bev_query_pos, ttnn.unsqueeze(reference_points[..., :2], 2))

            tmp = self._reg_branch(ttnn.permute(output, (1, 0, 2)), branch)
            updated_xy = ttnn.add(tmp[..., REG_XY], inverse_sigmoid(reference_points[..., :2]))
            updated_z = ttnn.add(tmp[..., REG_Z], inverse_sigmoid(reference_points[..., 2:3]))
            reference_points = ttnn.sigmoid(ttnn.concat([updated_xy, updated_z], dim=-1))

            intermediate.append(output)
            intermediate_reference_points.append(reference_points)

        return ttnn.stack(intermediate, dim=0), ttnn.stack(intermediate_reference_points, dim=0)
