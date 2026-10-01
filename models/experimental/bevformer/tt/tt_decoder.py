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
import math

import ttnn
from models.experimental.bevformer.tt.tt_common import layer_norm
from models.experimental.bevformer.tt.tt_ms_deformable_attention import TTMSDeformableAttention, fp32_grid_sample_config

GRID_DTYPE = ttnn.float32


INVERSE_SIGMOID_EPS = 1e-5
# inverse_sigmoid's range: its eps clamp bounds the logits to +-log(1 / eps).
LOGIT_BOUND = math.log(1.0 / INVERSE_SIGMOID_EPS)


class TtMultiheadAttention:
    """Batch-first self-attention over the object queries, without the residual (the layer
    adds it inside the following LayerNorm).

    ``params.qk_proj`` packs Q (pre-scaled by ``head_dim**-0.5``) and K into one Linear.
    """

    def __init__(self, params):
        self.params = params
        self.num_heads = params.num_heads

    def __call__(self, query, query_pos):
        """``query`` and ``query_pos`` are ``(bs, nq, C)``."""
        p = self.params
        qk = ttnn.linear(ttnn.add(query, query_pos), p.qk_proj.weight, bias=p.qk_proj.bias)
        v = ttnn.linear(query, p.v_proj.weight, bias=p.v_proj.bias)

        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            ttnn.concat([qk, v], dim=-1), num_heads=self.num_heads, transpose_key=False
        )
        # scale=1: the head_dim**-0.5 is folded into the Q weights.
        out = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False, scale=1.0)

        out = ttnn.transformer.concatenate_heads(out)
        return ttnn.linear(out, p.out_proj.weight, bias=p.out_proj.bias)


class TtFFN:
    """Linear-ReLU-Linear, without the residual (the layer adds it inside the following LayerNorm)."""

    def __init__(self, params):
        self.params = params

    def __call__(self, x):
        p = self.params
        y = ttnn.linear(x, p.linear1.weight, bias=p.linear1.bias, activation="relu")
        return ttnn.linear(y, p.linear2.weight, bias=p.linear2.bias)


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
        query = layer_norm(self.self_attn(query, query_pos), norms[0], residual=query)
        query = self.cross_attn(query=query, value=value, query_pos=query_pos, reference_points=reference_points)
        query = layer_norm(query, norms[1])
        return layer_norm(self.ffn(query), norms[2], residual=query)


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
        x = ttnn.linear(x, branch[0].weight, bias=branch[0].bias, activation="relu")
        x = ttnn.linear(x, branch[1].weight, bias=branch[1].bias, activation="relu")
        # (x, y, z) logit updates (see create_reg_branch_parameters), in GRID_DTYPE as they
        # are added to the reference points' logits.
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
        # The reference refines with sigmoid(delta + inverse_sigmoid(points)). Every layer's
        # points are the previous layer's sigmoid, and inverse_sigmoid(sigmoid(z)) is z clamped
        # to its eps bound, so the logits are carried instead of recomputed.
        logits = ttnn.logit(reference_points, eps=INVERSE_SIGMOID_EPS)
        intermediate = []
        intermediate_reference_points = []
        for index, (layer, branch) in enumerate(zip(self.layers, reg_branches, strict=True)):
            # The cross-attention's single level is the new axis 2.
            output = layer(output, value, query_pos, ttnn.unsqueeze(reference_points[..., :2], 2))

            if index:
                logits = ttnn.clamp(logits, min=-LOGIT_BOUND, max=LOGIT_BOUND)
            logits = ttnn.add(self._reg_branch(output, branch), logits)
            reference_points = ttnn.sigmoid(logits)

            intermediate.append(output)
            intermediate_reference_points.append(reference_points)

        outputs = ttnn.stack(intermediate, dim=0)
        if not self.batch_first:
            outputs = ttnn.permute(outputs, (0, 2, 1, 3))
        return outputs, ttnn.stack(intermediate_reference_points, dim=0)
