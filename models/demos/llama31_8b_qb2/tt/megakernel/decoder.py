# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Explicit integration of the experimental MLP phase into real decoder layers."""

from copy import copy

import ttnn

from models.demos.llama31_8b_qb2.tt.decoder import LlamaDecoder
from .mlp import FusedMLP
from .swiglu import fused_swiglu


class ExperimentalDecoder(LlamaDecoder):
    def _decode_finish(self, residual, attention):
        projected = self._output_projection(attention, "o")
        residual = ttnn.add(
            residual,
            ttnn.to_memory_config(projected, self.local_residual_memcfg),
            memory_config=self.local_residual_memcfg,
        )
        normalized = (
            self._ag(residual, decode=True, site="mlp")
            if self.fusion_mode == "norm_mlp_tail"
            else self._norm_input(residual, decode=True, site="mlp")
        )
        normalized = ttnn.to_memory_config(normalized, self.decode_inputs["gate_up"])
        if self.fusion_mode in ("mlp", "mlp_reduce", "mlp_tail", "norm_mlp_tail"):
            local_down = self.fused_body(
                normalized,
                self.fused_layer_index,
                residual=residual if self.fusion_mode in ("mlp_tail", "norm_mlp_tail") else None,
            )
            if self.fusion_mode in ("mlp_tail", "norm_mlp_tail"):
                return local_down
            down = local_down if self.fusion_mode == "mlp_reduce" else self._rs(local_down, decode=True, site="down")
        else:
            packed = self._decode_linear(normalized, "gate_up")
            product = fused_swiglu(packed, output_memory_config=self.decode_inputs["down"])
            down = self._output_projection(product, "down")
        result = ttnn.add(
            residual,
            ttnn.to_memory_config(down, self.local_residual_memcfg),
            memory_config=self.local_residual_memcfg,
        )
        return ttnn.typecast(result, self.residual_dtype) if result.dtype != self.residual_dtype else result


def experimental_layers(layers, *, mode="mlp", reuse_scratch=False, gu_workers=8):
    """Share original weights, KV ownership and workspace, with explicit opt-in.

    Construct before warming or capturing any trace. The caller owns the
    resulting layers and must retain them until all referencing traces release.
    Prefill is inherited unchanged. This does not fuse the complete decoder.
    """
    if mode not in ("swiglu", "mlp", "mlp_reduce", "mlp_tail", "norm_mlp_tail"):
        raise ValueError("mode must be swiglu, mlp, mlp_reduce or mlp_tail")
    if not layers or any(layer.decode_workspace.batch != 1 for layer in layers):
        raise ValueError("Experimental decode supports only prepared batch-one layers")
    body = (
        FusedMLP(
            layers,
            reuse_scratch=reuse_scratch,
            fuse_reduce=mode in ("mlp_reduce", "mlp_tail", "norm_mlp_tail"),
            fuse_norm=mode == "norm_mlp_tail",
            gu_workers=gu_workers,
        )
        if mode != "swiglu"
        else None
    )
    result = []
    for index, layer in enumerate(layers):
        adapted = copy(layer)
        adapted.__class__ = ExperimentalDecoder
        adapted.fusion_mode = mode
        adapted.fused_body = body
        adapted.fused_layer_index = index
        result.append(adapted)
    return result


def enable_experimental_decode(model, *, mode="mlp", reuse_scratch=False, gu_workers=8):
    """Install the same body across all 32 layers before generator trace setup.

    The embedding, final norm, head and sampler remain the existing traced
    boundary, so end-to-end tests exercise actual token feedback and positions.
    Callers must release all prior traces before switching implementations.
    """
    if model.max_batch_size != 1 or set(model.decode_families) != {1}:
        raise ValueError("Only a batch-one model without additional families is supported")
    model.layers = experimental_layers(model.layers, mode=mode, reuse_scratch=reuse_scratch, gu_workers=gu_workers)
    model.decode_families[1] = model.layers
