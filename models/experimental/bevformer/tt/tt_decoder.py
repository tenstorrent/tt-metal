# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TTNN port of BEVFormer's detection decoder (``reference/decoder.py``).

The layers run batch-first, the layout the cross-attention and reg branches need. By
default the decoder takes and returns the reference's sequence-first
``(num_query, bs, embed_dims)`` tensors, permuting ``query``, ``query_pos`` and ``value`` on
entry and the stacked layer outputs on exit; ``batch_first=True`` skips those permutes.
Unlike the reference, it also returns each layer's box codes, the reg branches' raw output,
so ``TTBEVFormerHead`` builds its box predictions without running the branches again.
Parameters come from ``model_preprocessing.create_decoder_parameters``. Forward runs
on device only.

The cross-attention is ``TTMSDeformableAttention``, one fused MSDA op. Reference points stay
float32 (``GRID_DTYPE``, the sampling positions' dtype) through the refinement and into the
op, which forms the sampling location from them in fp32; only the offsets, values and weights
are bfloat16. In bfloat16 a point in (0.5, 1) moves in steps of 2^-8, 0.8 px on the 200x200
BEV grid, and the error compounds across layers through the refinement.
"""

import dataclasses
import math

import ttnn
from models.experimental.bevformer.model_config import CENTERS_FIRST_CENTERS, GRID_DTYPE, INVERSE_SIGMOID_EPS
from models.experimental.bevformer.tt.tt_common import TTFFN, layer_norm, signpost
from models.experimental.bevformer.tt.tt_ms_deformable_attention import TTMSDeformableAttention

# inverse_sigmoid's range: its eps clamp bounds the logits to +-log(1 / eps).
LOGIT_BOUND = math.log(1.0 / INVERSE_SIGMOID_EPS)
CLAMP_LOGITS = [ttnn.UnaryWithParam(ttnn.UnaryOpType.HARDTANH, -LOGIT_BOUND, LOGIT_BOUND)]


class TTMultiheadAttention:
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


class TTDetrTransformerDecoderLayer:
    """One decoder layer, ``self_attn -> norm -> cross_attn -> norm -> ffn -> norm``.

    The self-attention and FFN residuals are added inside the LayerNorm that follows them;
    the cross-attention adds its own.
    """

    def __init__(self, params, device, bev_shape):
        """``params`` is one entry of ``create_decoder_parameters(...).layers``. The cross-attention
        samples a ``bev_shape`` ``(bev_h, bev_w)`` map."""
        self.params = params
        self.self_attn = TTMultiheadAttention(params.self_attn)
        # The module hands reference_points to the fused op unconverted, so they stay float32.
        self.cross_attn = TTMSDeformableAttention(
            dataclasses.replace(params.cross_attn.config, batch_first=True),
            device,
            params.cross_attn,
            spatial_shapes=[list(bev_shape)],
        )
        self.ffn = TTFFN(params.ffn)

    def __call__(self, query, value, query_pos, reference_points):
        """All batch-first: ``query``/``query_pos`` ``(bs, nq, C)``, ``value`` ``(bs, bev_h * bev_w, C)``,
        ``reference_points`` ``(bs, nq, 1, 2)`` ``GRID_DTYPE`` in [0, 1]."""
        norms = self.params.norms
        signpost(header="TTNN DecoderLayer Forward Start")
        signpost(header="DecoderLayer SelfAttn Start")
        self_attn_output = self.self_attn(query, query_pos)
        signpost(header="DecoderLayer SelfAttn End")
        query = layer_norm(self_attn_output, norms[0], residual=query)
        signpost(header="DecoderLayer CrossAttn Start")
        query = self.cross_attn(query=query, value=value, query_pos=query_pos, reference_points=reference_points)
        signpost(header="DecoderLayer CrossAttn End")
        query = layer_norm(query, norms[1])
        signpost(header="DecoderLayer FFN Start")
        ffn_output = self.ffn(query)
        signpost(header="DecoderLayer FFN End")
        query = layer_norm(ffn_output, norms[2], residual=query)
        signpost(header="TTNN DecoderLayer Forward End")
        return query


class TTDetectionTransformerDecoder:
    """Decoder over a ``bev_shape`` ``(bev_h, bev_w)`` BEV map."""

    def __init__(self, params, device, bev_shape, batch_first=False):
        """``params`` comes from ``create_decoder_parameters``; ``batch_first`` selects the
        layout of ``query``, ``query_pos``, ``value`` and the stacked outputs (see ``__call__``)."""
        self.batch_first = batch_first
        self.layers = [TTDetrTransformerDecoderLayer(p, device, bev_shape) for p in params.layers]

    @staticmethod
    def _reg_branch(x, branch):
        x = ttnn.linear(x, branch[0].weight, bias=branch[0].bias, activation="relu")
        x = ttnn.linear(x, branch[1].weight, bias=branch[1].bias, activation="relu")
        # GRID_DTYPE, as the center channels are added to the reference points' logits.
        return ttnn.linear(x, branch[2].weight, bias=branch[2].bias, dtype=GRID_DTYPE)

    def __call__(self, query, value, query_pos, reference_points, reg_branches):
        """``query``/``query_pos`` ``(nq, bs, C)`` and ``value`` ``(bev_h * bev_w, bs, C)``, or
        ``(bs, nq, C)`` and ``(bs, bev_h * bev_w, C)`` with ``batch_first``; ``reference_points``
        ``(bs, nq, 3)`` ``GRID_DTYPE`` in [0, 1] either way.

        Returns every layer's output ``(L, nq, bs, C)`` (``(L, bs, nq, C)`` with ``batch_first``),
        refined reference points ``(L, bs, nq, 3)`` and the reg branches' raw box codes
        ``(L, bs, nq, code_size)`` in ``model_config.py``'s ``CENTERS_FIRST`` order
        (batch-first either way), both ``GRID_DTYPE``. The codes' center channels are logit
        offsets; the refined points are their outcome.
        """
        if reference_points.dtype != GRID_DTYPE:
            raise ValueError(f"reference_points must be {GRID_DTYPE}, got {reference_points.dtype}")
        signpost(header="TTNN Decoder Forward Start")
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
        box_codes = []
        for layer, branch in zip(self.layers, reg_branches, strict=True):
            # Axis 2 is the fused op's pillar axis: R = 1, one point per query for every sample.
            output = layer(output, value, query_pos, ttnn.unsqueeze(reference_points[..., :2], 2))

            signpost(header="Decoder RegBranch Start")
            box_code = self._reg_branch(output, branch)
            signpost(header="Decoder RegBranch End")
            # The clamp (a no-op on the first layer's logit output) runs inside the add.
            logits = ttnn.add(box_code[..., CENTERS_FIRST_CENTERS], logits, input_tensor_b_activations=CLAMP_LOGITS)
            reference_points = ttnn.sigmoid(logits)

            intermediate.append(output)
            intermediate_reference_points.append(reference_points)
            box_codes.append(box_code)

        outputs = ttnn.stack(intermediate, dim=0)
        if not self.batch_first:
            outputs = ttnn.permute(outputs, (0, 2, 1, 3))
        reference_points_out = ttnn.stack(intermediate_reference_points, dim=0)
        box_codes_out = ttnn.stack(box_codes, dim=0)
        signpost(header="TTNN Decoder Forward End")
        return outputs, reference_points_out, box_codes_out
