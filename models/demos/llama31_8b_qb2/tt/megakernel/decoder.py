# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in partial-layer, device-layer-loop and token-to-logits prototypes."""

from copy import copy

import ttnn

from models.demos.llama31_8b_qb2.tt.decoder import LlamaDecoder
from models.demos.llama31_8b_qb2.tt.model import LlamaModel
from .mlp import FusedMLP
from .swiglu import fused_swiglu


class ExperimentalDecoder(LlamaDecoder):
    def decode_forward(self, x, *, current_pos, page_table, kv_cache, rotary_pos=None):
        if self.fusion_mode == "decoder":
            self._validate_cache(page_table, kv_cache)
            if tuple(x.shape) != (1, 1, 1, self.hidden):
                raise ValueError("Complete fused layer supports batch one")
            if tuple(current_pos.shape) != (1,) or current_pos.dtype != ttnn.int32 or page_table.shape[0] != 1:
                raise ValueError("Expected one device position and page-table row")
            residual = ttnn.to_memory_config(x, self.local_residual_memcfg)
            pages_per_chunk = max(1, 256 // self.page_size)
            tail_pages = (-page_table.shape[1]) % pages_per_chunk
            table = ttnn.pad(page_table, ((0, 0), (0, tail_pages)), value=0) if tail_pages else page_table
            return self.fused_body(
                residual,
                self.fused_layer_index,
                residual=residual,
                cache_inputs=(*kv_cache, current_pos, table, current_pos if rotary_pos is None else rotary_pos),
            )
        if self.fusion_mode != "attention_tail":
            return super().decode_forward(
                x, current_pos=current_pos, page_table=page_table, kv_cache=kv_cache, rotary_pos=rotary_pos
            )
        self._validate_cache(page_table, kv_cache)
        if tuple(x.shape) != (1, 1, 1, self.hidden):
            raise ValueError("Fused attention supports only batch-one decode")
        if tuple(current_pos.shape) != (1,) or current_pos.dtype != ttnn.int32 or page_table.shape[0] != 1:
            raise ValueError("Expected one device position and page-table row")
        residual = ttnn.to_memory_config(x, self.local_residual_memcfg)
        normalized = self._norm_input(residual, decode=True, site="attn")
        q, k, v = self._create_decode_heads(self._decode_linear(normalized, "qkv"), 1)
        cos, sin = self._rope_tables(current_pos if rotary_pos is None else rotary_pos, 1, q.memory_config())
        q = self._decode_rope(q, cos, sin)
        k = self._decode_rope(k, cos, sin)
        self._update_cache(k, v, current_pos, page_table, kv_cache)
        # Preserve the native cache contract for chunks extending past the
        # logical page-table row; masked reads still need a valid physical page.
        pages_per_chunk = max(1, 256 // self.page_size)
        tail_pages = (-page_table.shape[1]) % pages_per_chunk
        attention_table = ttnn.pad(page_table, ((0, 0), (0, tail_pages)), value=0) if tail_pages else page_table
        return self.fused_body(
            q,
            self.fused_layer_index,
            residual=residual,
            attention_inputs=(q, kv_cache[0], kv_cache[1], current_pos, attention_table),
        )

    def _decode_finish(self, residual, attention):
        if self.fusion_mode == "post_attention":
            return self.fused_body(attention, self.fused_layer_index, residual=residual)
        projected = self._output_projection(attention, "o")
        residual = ttnn.add(
            residual,
            ttnn.to_memory_config(projected, self.local_residual_memcfg),
            memory_config=self.local_residual_memcfg,
        )
        normalized = (
            residual
            if self.fusion_mode == "gather_norm_mlp_tail"
            else (
                self._ag(residual, decode=True, site="mlp")
                if self.fusion_mode == "norm_mlp_tail"
                else self._norm_input(residual, decode=True, site="mlp")
            )
        )
        if self.fusion_mode != "gather_norm_mlp_tail":
            normalized = ttnn.to_memory_config(normalized, self.decode_inputs["gate_up"])
        if self.fusion_mode in (
            "mlp",
            "mlp_reduce",
            "mlp_tail",
            "norm_mlp_tail",
            "gather_norm_mlp_tail",
            "post_attention",
            "attention_tail",
            "decoder",
        ):
            local_down = self.fused_body(
                normalized,
                self.fused_layer_index,
                residual=(
                    residual
                    if self.fusion_mode
                    in (
                        "mlp_tail",
                        "norm_mlp_tail",
                        "gather_norm_mlp_tail",
                        "post_attention",
                        "attention_tail",
                        "decoder",
                    )
                    else None
                ),
            )
            if self.fusion_mode in (
                "mlp_tail",
                "norm_mlp_tail",
                "gather_norm_mlp_tail",
                "post_attention",
                "attention_tail",
                "decoder",
            ):
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


def experimental_layers(layers, *, mode="mlp", reuse_scratch=False, gu_workers=8, tuning=None):
    """Share original weights, KV ownership and workspace, with explicit opt-in.

    Construct before warming or capturing any trace. The caller owns the
    resulting layers and must retain them until all referencing traces release.
    Prefill is inherited unchanged. Only mode="decoder" composes the complete layer.
    """
    if mode not in (
        "swiglu",
        "mlp",
        "mlp_reduce",
        "mlp_tail",
        "norm_mlp_tail",
        "gather_norm_mlp_tail",
        "post_attention",
        "attention_tail",
        "decoder",
    ):
        raise ValueError(f"Unknown experimental decode mode: {mode}")
    if not layers or any(layer.decode_workspace.batch != 1 for layer in layers):
        raise ValueError("Experimental decode supports only prepared batch-one layers")
    body = (
        FusedMLP(
            layers,
            reuse_scratch=reuse_scratch,
            fuse_reduce=mode
            in (
                "mlp_reduce",
                "mlp_tail",
                "norm_mlp_tail",
                "gather_norm_mlp_tail",
                "post_attention",
                "attention_tail",
                "decoder",
            ),
            fuse_norm=mode in ("norm_mlp_tail", "gather_norm_mlp_tail", "post_attention", "attention_tail", "decoder"),
            fuse_gather=mode in ("gather_norm_mlp_tail", "post_attention", "attention_tail", "decoder"),
            fuse_output=mode in ("post_attention", "attention_tail", "decoder"),
            fuse_attention=mode in ("attention_tail", "decoder"),
            fuse_prepare=mode == "decoder",
            gu_workers=gu_workers,
            tuning=tuning,
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


class ExperimentalModel(LlamaModel):
    def decode(self, tokens, *, current_pos, rotary_pos, page_table, kv_cache, execution_batch=None):
        if execution_batch not in (None, 1):
            raise ValueError("The device layer loop supports batch one")
        if (
            any(
                a.buffer_address() != b.buffer_address()
                for pair, bound in zip(kv_cache, self.fused_decode_loop.caches)
                for a, b in zip(pair, bound)
            )
            or len(kv_cache) != self.num_layers
        ):
            raise ValueError("Release traces and rebuild the loop before rebinding KV allocations")
        layer = self.layers[0]
        layer._validate_cache(page_table, kv_cache[0])
        if self.fused_decode_loop.embedding_weight is not None:
            x = self.fused_decode_loop.body.reduction.output
        else:
            indices = ttnn.reshape(tokens, (1, 32))
            x = ttnn.embedding(
                indices, self.embedding_weight, layout=ttnn.TILE_LAYOUT, memory_config=layer.local_residual_memcfg
            )
            x = ttnn.reshape(x, ttnn.Shape((1, 1, 1, 1024)), ttnn.Shape((1, 1, 32, 1024)), skip_padding_fill=True)
        pages_per_chunk = max(1, 256 // layer.page_size)
        tail_pages = (-page_table.shape[1]) % pages_per_chunk
        table = ttnn.pad(page_table, ((0, 0), (0, tail_pages)), value=0) if tail_pages else page_table
        x = self.fused_decode_loop(
            x,
            current_pos,
            table,
            rotary_pos,
            tokens=tokens if self.fused_decode_loop.embedding_weight is not None else None,
        )
        if self.fused_decode_loop.head is not None:
            logits = x
        elif self.fused_head is not None:
            logits = self.fused_head(layer._ag(x, decode=True, site="attn"))
        else:
            x = layer._norm_input(x, decode=True, site="attn")
            x = ttnn.to_memory_config(x, self.lm_head.config.input_memcfg)
            logits = self.lm_head(x)
        shape = ttnn.Shape((1, 1, 32, self.padded_vocab_size // 4))
        return ttnn.reshape(logits, shape, shape, skip_padding_fill=True)


def enable_experimental_decode(model, *, mode="mlp", reuse_scratch=False, gu_workers=8, kv_cache=None, tuning=None):
    """Install the same body across all 32 layers before generator trace setup.

    Mode "decode_token" composes embedding, all layers and terminal logits in
    one program. Native sampling still owns token feedback and position updates.
    Callers must release all prior traces before switching implementations.
    """
    if model.max_batch_size != 1 or set(model.decode_families) != {1}:
        raise ValueError("Only a batch-one model without additional families is supported")
    is_loop = mode in ("decoder_loop", "decoder_loop_embedding", "decoder_loop_head", "decode_token")
    if tuning is not None and (tuning.prefetch_gu_blocks or tuning.prefetch_down_blocks) and not is_loop:
        raise ValueError("Weight staging requires a device layer loop")
    if tuning is not None and tuning.alias_projection_cbs and not is_loop:
        raise ValueError("Static projection aliasing requires the device layer loop")
    if is_loop and kv_cache is None:
        raise ValueError("The loop must bind all KV allocations before warmup and capture")
    model.layers = experimental_layers(
        model.layers,
        mode="decoder" if is_loop else mode,
        reuse_scratch=reuse_scratch,
        gu_workers=gu_workers,
        tuning=tuning,
    )
    if is_loop:
        from .loop import DecoderLoop
        from .head import FusedHead

        body = model.layers[0].fused_body
        integrated_head = None
        if mode == "decode_token":
            occupied = {
                (c.x, c.y)
                for c in (
                    body.projection_cores
                    + body.sfpu_cores
                    + body.communication_cores
                    + body.norm_cores
                    + body.attention_stage.cores
                    + body.preparation.cores
                )
            }
            if body.tuning.share_qkv_workers:
                # Keep terminal placement fixed while measuring QKV reuse.
                # The freed workers still own the original packed tensor.
                occupied.update((c.x, c.y) for c in body.preparation.packed_cores)
            grid = model.mesh_device.compute_with_storage_grid_size()
            free = [
                ttnn.CoreCoord(x, y)
                for y in range(min(10, grid.y))
                for x in range(min(11, grid.x))
                if (x, y) not in occupied
            ]
            if len(free) < 24:
                raise ValueError("The integrated terminal boundary requires24 free workers on the QB2 grid")
            if body.tuning.head_placement != "row":
                from .placement import terminal_head_placement
                free = terminal_head_placement(model.mesh_device, free, body.tuning.head_placement)
            integrated_head = FusedHead(
                model, cores=free[:16], norm_cores=free[16:24], norm_output=body.normalizer.output, tuning=tuning
            )
        model.fused_decode_loop = DecoderLoop(
            body,
            kv_cache,
            embedding_weight=model.embedding_weight if mode != "decoder_loop" else None,
            head=integrated_head,
        )
        model.fused_head = FusedHead(model, tuning=tuning) if mode == "decoder_loop_head" else None
        model.__class__ = ExperimentalModel
    model.decode_families[1] = model.layers
