# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TTNN port of BEVFormer's detection decoder, from UniAD's ``TtDetectionTransformerDecoder``
(``models/experimental/uniad/tt/ttnn_decoder.py``).

The layers run batch-first, the layout the cross-attention and reg branches need. By
default the decoder takes and returns the reference's sequence-first
``(num_query, bs, embed_dims)`` tensors, permuting ``query``, ``query_pos`` and ``value`` on
entry and the stacked layer outputs on exit; ``batch_first=True`` skips those permutes.
Parameters come from ``model_preprocessing_decoder.create_decoder_parameters``. Forward runs
on device only.

Reference points are float32, as is the sampling grid built from them (``GRID_DTYPE``).
In bfloat16 a point in (0.5, 1) moves in steps of 2^-8, 0.8 px on the 200x200 BEV grid,
and the error compounds across layers through the reference-point refinement.
"""

import dataclasses

import ttnn
from models.experimental.bevformer.config.decoder_config import REG_XY, REG_Z
from models.experimental.bevformer.tt.tt_common import layer_norm
from models.experimental.bevformer.tt.tt_ms_deformable_attention import TTMSDeformableAttention, fp32_grid_sample_config

GRID_DTYPE = ttnn.float32


def inverse_sigmoid(x, eps=1e-5):
    x = ttnn.clamp(x, min=0, max=1)
    x1 = ttnn.clamp(x, min=eps)
    x2 = ttnn.clamp(ttnn.rsub(x, 1.0), min=eps)
    return ttnn.log(ttnn.div(x1, x2))


class TtMultiheadAttention:
    """Batch-first self-attention over the object queries.

    ``params.qk_proj`` packs Q (pre-scaled by ``head_dim**-0.5``) and K into one Linear.
    """

    def __init__(self, params):
        self.params = params
        self.num_heads = params.num_heads

    def __call__(self, query, query_pos):
        """``query`` and ``query_pos`` are ``(bs, nq, C)``."""
        p = self.params
        bs, num_query, embed_dims = query.shape
        head_dim = embed_dims // self.num_heads

        qk = ttnn.linear(ttnn.add(query, query_pos), p.qk_proj.weight, bias=p.qk_proj.bias)
        v = ttnn.linear(query, p.v_proj.weight, bias=p.v_proj.bias)

        def heads(x, order=(0, 2, 1, 3)):
            return ttnn.permute(ttnn.reshape(x, (bs, num_query, self.num_heads, head_dim)), order)

        q = heads(qk[..., :embed_dims])
        k = heads(qk[..., embed_dims:], order=(0, 2, 3, 1))  # (bs, heads, head_dim, nq), transposed for q @ k
        v = heads(v)

        attn = ttnn.softmax(ttnn.matmul(q, k), dim=-1)
        out = ttnn.matmul(attn, v)

        out = ttnn.reshape(ttnn.permute(out, (0, 2, 1, 3)), (bs, num_query, embed_dims))
        out = ttnn.linear(out, p.out_proj.weight, bias=p.out_proj.bias)
        return ttnn.add(out, query)


class TtFFN:
    def __init__(self, params):
        self.params = params

    def __call__(self, x):
        p = self.params
        y = ttnn.relu(ttnn.linear(x, p.linear1.weight, bias=p.linear1.bias))
        y = ttnn.linear(y, p.linear2.weight, bias=p.linear2.bias)
        return ttnn.add(y, x)


class TtDetrTransformerDecoderLayer:
    def __init__(self, params, device, bev_shape, grid_sample_compute_config):
        self.params = params
        self.self_attn = TtMultiheadAttention(params.self_attn)
        self.cross_attn = TTMSDeformableAttention(
            dataclasses.replace(params.cross_attn.config, batch_first=True),
            device,
            params.cross_attn,
            spatial_shapes=[list(bev_shape)],
            grid_dtype=GRID_DTYPE,
            grid_sample_compute_config=grid_sample_compute_config,
        )
        self.ffn = TtFFN(params.ffn)

    def __call__(self, query, value, query_pos, reference_points):
        """All batch-first: ``query``/``query_pos`` ``(bs, nq, C)``, ``value`` ``(bs, bev_h * bev_w, C)``,
        ``reference_points`` ``(bs, nq, 1, 2)`` in [0, 1]."""
        norms = self.params.norms
        query = layer_norm(self.self_attn(query, query_pos), norms[0])
        query = self.cross_attn(query=query, value=value, query_pos=query_pos, reference_points=reference_points)
        query = layer_norm(query, norms[1])
        return layer_norm(self.ffn(query), norms[2])


class TtDetectionTransformerDecoder:
    """Decoder over a ``bev_shape`` ``(bev_h, bev_w)`` BEV map.

    The cross-attention folds ``bev_shape`` into its sampling-offset Linear, which consumes
    ``params.layers[*].cross_attn.sampling_offsets``: each instance needs its own
    ``create_decoder_parameters``, and reusing them raises.
    """

    def __init__(self, params, device, bev_shape, batch_first=False):
        self.batch_first = batch_first
        grid_sample_compute_config = fp32_grid_sample_config(device)
        self.layers = [
            TtDetrTransformerDecoderLayer(p, device, bev_shape, grid_sample_compute_config) for p in params.layers
        ]

    @staticmethod
    def _reg_branch(x, branch):
        x = ttnn.relu(ttnn.linear(x, branch[0].weight, bias=branch[0].bias))
        x = ttnn.relu(ttnn.linear(x, branch[1].weight, bias=branch[1].bias))
        # Emitted in GRID_DTYPE: its output is added to the reference points' logits.
        return ttnn.linear(x, branch[2].weight, bias=branch[2].bias, dtype=GRID_DTYPE)

    def __call__(self, query, value, query_pos, reference_points, reg_branches):
        """``query``/``query_pos`` ``(nq, bs, C)`` and ``value`` ``(bev_h * bev_w, bs, C)``, or
        ``(bs, nq, C)`` and ``(bs, bev_h * bev_w, C)`` with ``batch_first``; ``reference_points``
        ``(bs, nq, 3)`` ``GRID_DTYPE`` in [0, 1] either way.

        Returns every layer's output ``(L, nq, bs, C)`` (``(L, bs, nq, C)`` with ``batch_first``)
        and refined reference points ``(L, bs, nq, 3)``.
        """
        if reference_points.dtype != GRID_DTYPE:
            raise ValueError(f"reference_points must be {GRID_DTYPE}, got {reference_points.dtype}")
        output = query
        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))
            value = ttnn.permute(value, (1, 0, 2))
            query_pos = ttnn.permute(query_pos, (1, 0, 2))
        intermediate = []
        intermediate_reference_points = []
        for layer, branch in zip(self.layers, reg_branches, strict=True):
            # The cross-attention's single level is the new axis 2.
            output = layer(output, value, query_pos, ttnn.unsqueeze(reference_points[..., :2], 2))

            box_delta = self._reg_branch(output, branch)
            updated_xy = ttnn.add(box_delta[..., REG_XY], inverse_sigmoid(reference_points[..., :2]))
            updated_z = ttnn.add(box_delta[..., REG_Z], inverse_sigmoid(reference_points[..., 2:3]))
            reference_points = ttnn.sigmoid(ttnn.concat([updated_xy, updated_z], dim=-1))

            intermediate.append(output)
            intermediate_reference_points.append(reference_points)

        outputs = ttnn.stack(intermediate, dim=0)
        if not self.batch_first:
            outputs = ttnn.permute(outputs, (0, 2, 1, 3))
        return outputs, ttnn.stack(intermediate_reference_points, dim=0)
