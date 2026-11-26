# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Base of the TP4 Nemotron-H layer (multichip_decoder.MultichipDecoder).

Holds the layer-kind mapping, the per-slot state record, and the FP32 MoE norm and
router/top-k helpers MultichipDecoder inherits. The single-chip layer programs of
the bring-up, which MultichipDecoder overrides, are not carried.
"""

from dataclasses import dataclass

import ttnn
from models.common.lightweightmodule import LightweightModule


# transformers >= 5.13 (5.16 with vLLM 0.32) reports NemotronH layer types under the new names
# (mamba -> linear_attention, attention -> full_attention); this port keys on the checkpoint's names.
_LEGACY_LAYER_KIND = {"linear_attention": "mamba", "full_attention": "attention"}


def layer_kinds(hf_config):
    return [_LEGACY_LAYER_KIND.get(kind, kind) for kind in hf_config.layers_block_type]


@dataclass
class DecoderState:
    key: object = None
    value: object = None
    conv: object = None
    recurrent: object = None


class OptimizedDecoder(LightweightModule):
    def _norm_unsharded(self, x):
        rows = x.shape[2]
        if self.kind == "moe":
            # Keep the accurate FP32 reduction, avoiding its implicit-row fill.
            # Decode rows must share an addition order: width reduction can
            # differ by two FP32 ULP across rows, crossing the BF16 boundary and
            # changing MoE routing. Preserve the prepared prefill arithmetic.
            if rows == 1:
                x = ttnn.add(x, self.norm_expand_zero)
            xf = ttnn.typecast(x, ttnn.float32)
            square = ttnn.square(xf)
            uniform_decode_order = not self._inside_prefill and rows <= 32
            if uniform_decode_order:
                square = ttnn.transpose(square, -1, -2)
            variance = ttnn.mean(
                square,
                dim=2 if uniform_decode_order else 3,
                keepdim=True,
                compute_kernel_config=self.compute,
                fast_and_approximate_mode=False,
            )
            if uniform_decode_order:
                variance = ttnn.transpose(variance, -1, -2)
            normalized = ttnn.typecast(
                ttnn.multiply(
                    xf,
                    ttnn.add(variance, self.config.layer_norm_epsilon),
                    input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RSQRT, 0.0)],
                ),
                ttnn.bfloat16,
            )
        else:
            normalized = ttnn.rms_norm(x, epsilon=self.config.layer_norm_epsilon, compute_kernel_config=self.compute)
        result = ttnn.multiply(normalized, self.norm) if self.kind == "moe" else normalized
        return result[:, :, :rows, :] if self.kind == "moe" and rows == 1 else result

    def _router_logits(self, x):
        # Hard top-k membership needs the preserved accurate FP32 reduction.
        # Native HiFi4 matmul changes expert membership on the seed1000 gate.
        # Bound expanded products independently of the larger expert chunk.
        parts = []
        for start in range(0, x.shape[2], self.router_chunk):
            rows = (
                x if x.shape[2] <= self.router_chunk else x[:, :, start : min(start + self.router_chunk, x.shape[2]), :]
            )
            tokens = ttnn.permute(ttnn.typecast(rows, ttnn.float32), (0, 2, 1, 3))
            products = ttnn.multiply(tokens, self.router_rows)
            logits = ttnn.sum(
                products, dim=3, keepdim=True, compute_kernel_config=self.compute, fast_and_approximate_mode=False
            )
            parts.append(ttnn.permute(logits, (0, 3, 1, 2)))
            del rows, tokens, products, logits
        while len(parts) > 32:
            parts = [ttnn.concat(parts[i : i + 32], dim=2) for i in range(0, len(parts), 32)]
        return parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=2)

    def _route(self, x):
        scores = ttnn.sigmoid(self._router_logits(x))
        rows = x.shape[2]
        if rows in (1, 32):
            work_scores = ttnn.add(scores, self.route_expand_zero) if rows == 1 else scores
            choice = ttnn.add(work_scores, self.bias)
            _, indices = ttnn.topk(choice, k=32, dim=-1, sorted=True)
            selected = ttnn.gather(work_scores, dim=3, index=indices)
            indices = indices[:, :, :rows, :6]
            selected = ttnn.multiply(selected, self.route_score_mask)
        else:
            choice = ttnn.add(scores, self.bias)
            _, indices = ttnn.topk(choice, k=8, dim=-1, sorted=True)
            indices = indices[:, :, :, :6]
            selected = ttnn.gather(scores, dim=3, index=indices)
        if not self._inside_prefill and rows <= 32:
            # The FP32 width reduction associates additions differently across
            # tile row groups. Even a two-ULP denominator difference can change
            # the BF16 expert result and later sampled token for identical rows.
            # Reduce transposed columns with one association for every request.
            denominator = ttnn.transpose(
                ttnn.sum(
                    ttnn.transpose(selected, -1, -2),
                    dim=2,
                    keepdim=True,
                    compute_kernel_config=self.compute,
                    fast_and_approximate_mode=False,
                ),
                -1,
                -2,
            )
        else:
            denominator = ttnn.sum(selected, dim=3, keepdim=True)
        selected = ttnn.multiply(
            selected,
            denominator,
            input_tensor_b_activations=[
                ttnn.UnaryOpType.RECIP,
                ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, 2.5),
            ],
        )
        if rows in (1, 32):
            selected = selected[:, :, :rows, :6]
        return indices, selected, scores
