# SPDX-License-Identifier: Apache-2.0
"""Internal compact decode graphs with fixed32 public state and sampling.

Only a contiguous active row prefix can use these views. Device token feedback,
positions, request slots and complete page tables are sliced inside each trace;
the original owner's sampler and output buffer remain the only token producer.
"""

import copy

import torch

import ttnn

from .prefill_view import _clone, _shared_zero
from .state_ops import advance_positions

DECODE_BUCKETS = (4, 8, 16)


def select_decode_batch(active, *, enabled=True):
    """Use compact geometry only for a nonempty front-packed fixed32 batch."""
    if len(active) != 32:
        return len(active)
    count = sum(bool(value) for value in active)
    if not enabled or not count or list(active) != [True] * count + [False] * (32 - count):
        return 32
    return next((batch for batch in DECODE_BUCKETS if count <= batch), 32)


def _ops(owner):
    # Ordinary MeshOps retains every B>1 precision branch. In particular do
    # not use the singleton prefill norm/output-projection override classes.
    result = copy.copy(owner)
    result.gather_outputs = {}
    result.reduce_outputs = {}
    result.large_prefill = False
    result.l1_prefill = False
    return result


def _decoder(owner, batch, pool):
    ops = _ops(owner.ops)
    result = _clone(owner, ops)
    result.batch_size = batch
    result.lengths = {}
    result.attn_hyper = _clone(owner.attn_hyper, ops)
    result.mlp_hyper = _clone(owner.mlp_hyper, ops)
    result.mlp = _clone(owner.mlp, ops)
    # Packed multirow MoE does not use capacity; retain the owner allocation.
    attention = _clone(owner.attention, ops)
    result.attention = attention
    attention.batch_size = batch
    attention.output_projection = _clone(owner.attention.output_projection, ops)
    if owner.kind == "linear_attention":
        for name in ("recurrent_state", "zero_recurrent", "conv_state", "zero_conv"):
            reference = getattr(owner.attention, name)
            setattr(
                attention,
                name,
                _shared_zero(pool, name, reference, (batch, *tuple(reference.shape)[1:]), owner.mesh),
            )
    else:
        if not attention.paged_pooled:
            raise ValueError("Compact decode requires the serving paged QSA cache")
        attention._decode_phase = False
        if hasattr(attention, "_pooled_gather_partition"):
            del attention._pooled_gather_partition
        attention.indexer_prefix = ops.upload(torch.zeros(batch, 4, 31, 128))
        attention.indexer_weights = ops.upload(torch.ones(batch, 1, attention.chunk_size + 32, 4))
        attention.pooled_keys = _shared_zero(
            pool, "pooled_keys", owner.attention.pooled_keys, owner.attention.pooled_keys.shape, owner.mesh
        )
    if owner.ple is not None:
        result.ple = _clone(owner.ple, ops)
        result.ple.workspaces = {}
        result.ple.flat_norm_weights = {}
    result.prepare([1])
    return result


def _model(owner, batch, pool):
    if not owner.serving_pool or owner.batch_size != 32 or batch not in DECODE_BUCKETS:
        raise ValueError("Compact decode requires a fixed32 serving owner and B4/B8/B16")
    result = _clone(owner, _ops(owner.ops))
    result.batch_size = batch
    result.terminal = _clone(owner.terminal, result.ops)
    result.layers = [_decoder(layer, batch, pool) for layer in owner.layers]
    result.cache = result.export_cache()
    result._bound_cache = result.cache
    for original, view in zip(owner.layers, result.layers):
        public = (
            ("recurrent_bank", "conv_bank")
            if original.kind == "linear_attention"
            else ("packed_cache", "raw_key_cache", "pooled_cache")
        )
        for name in public:
            if getattr(original.attention, name) is not getattr(view.attention, name):
                raise RuntimeError(f"Compact decode lost public ownership of {name}")
        if original.ple is not None and original.ple_cache is not view.ple_cache:
            raise RuntimeError("Compact decode lost public PLE ownership")
    if result.precision_policy is not owner.precision_policy or result.lm_head is not owner.lm_head:
        raise RuntimeError("Compact decode changed immutable owner policy or weights")
    return result


class CompactDecodeView:
    """One internal batch graph; owns neither a sampler nor public state."""

    def __init__(self, generator, batch, pool):
        self.generator = generator
        self.batch_size = batch
        self.model = _model(generator.model, batch, pool)
        # All views execute serially on CQ0. Each invocation refreshes every
        # row from current owner logits before overwriting its computed prefix.
        if "logit_rows" not in pool:
            pool["logit_rows"] = ttnn.zeros(
                tuple(generator.tt_logits.shape),
                dtype=generator.tt_logits.dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=generator.mesh,
            )
        self.rows = pool["logit_rows"]

    def forward(self):
        g, batch = self.generator, self.batch_size
        # These slices must remain inside the captured graph: reload_inputs
        # false consumes fresh device sampler feedback, never stale host data.
        tokens = ttnn.reshape(g.tt_token, (32, 1))[:batch, :]
        positions = g.tt_pos[:batch, :]
        slots = g.tt_slots[:batch, :]
        pages = g.tt_page[:batch, :]
        logits = self.model.forward(
            tokens, positions=positions, slots=slots, page_table=pages, kv_cache=self.model.cache, decode=True
        )
        ttnn.copy(ttnn.to_layout(g.tt_logits, ttnn.ROW_MAJOR_LAYOUT), self.rows)
        ttnn.experimental.slice_write(
            ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT),
            self.rows,
            [0, 0, 0, 0],
            [1, 1, batch, self.rows.shape[-1]],
            [1, 1, 1, 1],
        )
        # Computed padding rows have private state, but must not replace the
        # current owner's inactive logits. Sampling still runs once at B32.
        merged = ttnn.where(g.tt_logit_mask, ttnn.to_layout(self.rows, ttnn.TILE_LAYOUT), g.tt_logits)
        ttnn.copy(merged, g.tt_logits)
        advance_positions(g.tt_pos, g.tt_active)

    def runtime_summary(self):
        owner = self.generator.model
        layers = []
        for index, original, view in zip(owner.layer_indices, owner.layers, self.model.layers):
            names = (
                ("recurrent_bank", "conv_bank")
                if original.kind == "linear_attention"
                else ("packed_cache", "raw_key_cache", "pooled_cache")
            )
            layers.append(
                {
                    "layer": index,
                    "public_state": {
                        name: {
                            "shared": getattr(original.attention, name) is getattr(view.attention, name),
                            "buffer_address": getattr(view.attention, name).buffer_address(),
                        }
                        for name in names
                    },
                    "ple_cache_shared": original.ple is None or original.ple_cache is view.ple_cache,
                }
            )
        return {
            "execution_batch": self.batch_size,
            "layers": layers,
            "layer_indices": self.model.layer_indices,
            "context": self.model.context,
            "public_pool_tokens": self.model.public_pool_tokens,
            "precision": self.model.precision_policy,
            "owner_public_state_shared": all(
                row["ple_cache_shared"] and all(state["shared"] for state in row["public_state"].values())
                for row in layers
            ),
            "owner_precision_policy_shared": self.model.precision_policy is owner.precision_policy,
            "owner_lm_head_shared": self.model.lm_head is owner.lm_head,
            "input_bridge": "device_prefix_slice_tokens_positions_slots_full_page_rows",
            "logits_bridge": "owner_tile_to_row_major_prefix_slice_write_tile_active_mask_merge_copy",
            "input_shapes": {
                "tokens": [self.batch_size, 1],
                "positions": [self.batch_size, 32],
                "slots": [self.batch_size, 32],
                "pages": [self.batch_size, self.model.context // 32],
            },
            "logit_shapes": {
                "view": [1, 1, self.batch_size, self.model.cfg.vocab_size // 4],
                "owner": list(self.generator.tt_logits.shape),
                "row_major_work": list(self.rows.shape),
            },
            "gdn_recurrence_broadcast": [
                {"layer": index, "enabled": layer.attention.recurrence_broadcast}
                for index, layer in zip(self.model.layer_indices, self.model.layers)
                if layer.kind == "linear_attention"
            ],
            "indexer_prefix_cap": [
                {"layer": index, "enabled": layer.attention.indexer_prefix_cap}
                for index, layer in zip(self.model.layer_indices, self.model.layers)
                if layer.kind != "linear_attention"
            ],
            "qsa_decode_fold_vectorized": [
                {"layer": index, "enabled": getattr(layer.attention, "_decode_fold_vectorized", False)}
                for index, layer in zip(self.model.layer_indices, self.model.layers)
                if layer.kind != "linear_attention"
            ],
            "pooled_gather_partitions": [
                {"layer": index, **getattr(layer.attention, "_pooled_gather_partition", {})}
                for index, layer in zip(self.model.layer_indices, self.model.layers)
                if layer.kind != "linear_attention"
            ],
        }
