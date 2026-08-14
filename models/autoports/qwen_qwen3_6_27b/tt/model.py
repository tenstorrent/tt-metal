# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Full TP4 text-only Qwen3.6-27B autoregressive model.

This wrapper deliberately stacks :class:`MultichipDecoder` without changing
its replicated BF16 residual boundary.  Embedding and terminal projection are
the only full-model boundaries; no decoder-layer output is gathered to host.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

import torch
from safetensors import safe_open
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import TARGET_MESH_SHAPE, MultichipDecoder
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import _dram_weight_memory_config, _l1_width_memory_config
from models.autoports.qwen_qwen3_6_27b.tt.precision_config import load_precision_config


def _replicate(value, mesh, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        value.contiguous(),
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        dtype=dtype,
        layout=layout,
        memory_config=memory_config,
    )


def _shard(value, mesh, dim, *, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        value.contiguous(),
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
    )


class SnapshotReader(Mapping[str, torch.Tensor]):
    """Lazy, bounded host reader for an indexed safetensors checkpoint."""

    def __init__(self, snapshot: Path):
        self.snapshot = snapshot
        with (snapshot / "model.safetensors.index.json").open() as handle:
            self.weight_map = json.load(handle)["weight_map"]
        self._cached_shard = None
        self._cached_tensors = {}

    def __getitem__(self, key):
        shard = self.weight_map[key]
        if shard != self._cached_shard:
            # Construction walks layers in checkpoint order. Cache one shard
            # at a time so dozens of tensors do not repeatedly reopen/remap a
            # multi-GB safetensors file; old shards are released promptly.
            with safe_open(self.snapshot / shard, framework="pt", device="cpu") as handle:
                self._cached_tensors = {name: handle.get_tensor(name) for name in handle.keys()}
            self._cached_shard = shard
        return self._cached_tensors[key]

    def __iter__(self):
        return iter(self.weight_map)

    def __len__(self):
        return len(self.weight_map)


class Qwen36Model:
    PREFILL_STACK_CHUNK_SIZE = 32768
    """Text-only HF causal-LM path over the optimized TP4 decoder stack."""

    def __init__(
        self,
        *,
        mesh_device,
        config,
        state_dict: Mapping[str, torch.Tensor],
        batch: int = 1,
        max_context: int = 262144,
        page_size: int = 64,
        attention_cache_blocks: int | None = None,
        num_layers: int | None = None,
        layer_indices: list[int] | None = None,
        precision_config=None,
    ):
        if tuple(mesh_device.shape) != TARGET_MESH_SHAPE:
            raise ValueError(f"Qwen3.6 full model preserves the decoder's TP4 mesh {TARGET_MESH_SHAPE}")
        self.mesh_device, self.config = mesh_device, config
        self.batch, self.max_context, self.page_size = batch, max_context, page_size
        self.precision_config = load_precision_config(precision_config)
        if layer_indices is not None and num_layers is not None:
            raise ValueError("specify num_layers or layer_indices, not both")
        selected_layers = (
            list(range(int(config.num_hidden_layers if num_layers is None else num_layers)))
            if layer_indices is None
            else list(layer_indices)
        )
        self.num_layers = len(selected_layers)
        if not selected_layers or min(selected_layers) < 0 or max(selected_layers) >= int(config.num_hidden_layers):
            raise ValueError("num_layers must select a non-empty prefix of the HF stack")

        prefix = "model.language_model."
        embedding = state_dict[prefix + "embed_tokens.weight"].bfloat16()
        self.embedding = _replicate(embedding, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT)
        norm = (state_dict[prefix + "norm.weight"].bfloat16() + 1).reshape(1, 1, 160, 32)
        self.final_norm = _replicate(norm, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT)

        # Column-parallel vocab ownership.  Pad globally so every local shard
        # is tile aligned; padded IDs are masked before sampling/readback.
        output = state_dict["lm_head.weight"].bfloat16().transpose(0, 1).contiguous()
        self.vocab_size = int(config.vocab_size)
        self.padded_vocab_size = math.ceil(self.vocab_size / (4 * 32)) * (4 * 32)
        output = torch.nn.functional.pad(output, (0, self.padded_vocab_size - self.vocab_size))
        local_vocab = self.padded_vocab_size // 4
        # A single 62,080-column local projection exceeds worker L1 even with
        # the smallest DRAM-sharded K block. Split *within* every TP shard,
        # retaining contiguous per-device vocab ownership, then concatenate
        # the two device-local results below. The split is chosen to minimize
        # DRAM-bank padding across both weights.
        first_vocab = (local_vocab // (2 * 256)) * 256
        self.lm_head_chunk_sizes = (first_vocab, local_vocab - first_vocab)
        device_weights = list(output.split(local_vocab, dim=-1))
        chunk_weights = [
            torch.cat([device_weight[:, offset : offset + width] for device_weight in device_weights], dim=-1)
            for offset, width in ((0, first_vocab), (first_vocab, local_vocab - first_vocab))
        ]
        self.lm_head_weights = [
            _shard(
                chunk,
                mesh_device,
                -1,
                dtype=self.precision_config.lm_head_weight_dtype,
                memory_config=_dram_weight_memory_config(mesh_device, k=int(config.hidden_size), n=width),
            )
            for chunk, width in zip(chunk_weights, self.lm_head_chunk_sizes)
        ]
        self.lm_head_dtype = self.precision_config.lm_head_output_dtype
        self.lm_head_local_vocab = local_vocab
        self.lm_head_cores = mesh_device.dram_grid_size().x
        self.lm_head_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=self.precision_config.lm_head_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        self.layers = [
            MultichipDecoder.from_state_dict(
                state_dict,
                hf_config=config,
                layer_idx=i,
                mesh_device=mesh_device,
                batch=batch,
                max_context=max_context,
                page_size=page_size,
                attention_cache_blocks=attention_cache_blocks,
                candidate="default",
                policy_override=self.precision_config.policy_for(i, config.layer_types[i]),
                ccl_token_mixer_dtype=self.precision_config.ccl_dtype("token_mixer"),
                ccl_mlp_dtype=self.precision_config.ccl_dtype("mlp"),
            )
            for i in selected_layers
        ]
        self.kv_cache = [
            [
                layer.caches[name]
                for name in (("key", "value") if layer.layer_kind == "full_attention" else ("conv", "recurrent"))
            ]
            for layer in self.layers
        ]
        print("PRECISION_CONFIG", json.dumps(self.precision_summary(), sort_keys=True, default=str))

    def precision_summary(self):
        """Return the exact policy consumed by this constructed runtime."""
        layers = []
        for layer in self.layers:
            p = layer.policy
            layers.append(
                {
                    "layer_idx": layer.layer_idx,
                    "layer_kind": layer.layer_kind,
                    "attention_weight_dtype": str(p.attention_weight_dtype),
                    "mlp_gate_up_dtype": str(p.mlp_gate_up_dtype),
                    "mlp_down_dtype": str(p.mlp_down_dtype),
                    "cache_dtype": str(p.cache_dtype),
                    "attention_fidelity": str(p.attention_fidelity),
                    "qkv_fidelity": str(p.qkv_fidelity or p.attention_fidelity),
                    "o_fidelity": str(p.o_fidelity or p.attention_fidelity),
                    "mlp_fidelity": str(p.mlp_fidelity),
                    "linear_input_weight_dtype": str(p.linear_input_weight_dtype),
                    "linear_input_fidelity": str(p.linear_input_fidelity),
                    "linear_output_weight_dtype": str(p.linear_output_weight_dtype),
                    "linear_output_fidelity": str(p.linear_output_fidelity),
                    "linear_recurrent_state_dtype": str(p.linear_recurrent_state_dtype),
                    "linear_recurrent_fidelity": str(p.linear_recurrent_fidelity),
                    "ccl_token_mixer_dtype": str(layer.ccl_token_mixer_dtype),
                    "ccl_mlp_dtype": str(layer.ccl_mlp_dtype),
                }
            )
        return {
            "precision_config": self.precision_config.summary(),
            "activation_residual_dtype": str(self.precision_config.activation_residual_dtype),
            "lm_head_weight_dtype": str(self.precision_config.lm_head_weight_dtype),
            "lm_head_output_dtype": str(self.lm_head_dtype),
            "lm_head_fidelity": str(self.precision_config.lm_head_fidelity),
            "sampling_logits_dtype": str(self.lm_head_dtype),
            "sampled_token_dtype": str(ttnn.uint32),
            "layers": layers,
        }

    def bind_kv_cache(self, kv_cache):
        """Bind caller-owned persistent cache/state tensors for low-level serving."""
        if len(kv_cache) != len(self.layers):
            raise ValueError(f"expected {len(self.layers)} layer cache pairs, got {len(kv_cache)}")
        for layer, pair in zip(self.layers, kv_cache):
            if len(pair) != 2:
                raise ValueError("each layer cache entry must contain exactly two tensors")
            names = ("key", "value") if layer.layer_kind == "full_attention" else ("conv", "recurrent")
            for name, tensor in zip(names, pair):
                layer.caches[name] = tensor
        self.kv_cache = kv_cache

    @classmethod
    def from_pretrained(cls, *, mesh_device, snapshot: str | Path, **kwargs):
        snapshot = Path(snapshot)
        outer = AutoConfig.from_pretrained(snapshot, local_files_only=True)
        return cls(mesh_device=mesh_device, config=outer.text_config, state_dict=SnapshotReader(snapshot), **kwargs)

    def allocate_page_table(self, batch: int | None = None):
        batch = self.batch if batch is None else batch
        blocks_per_user = math.ceil(self.max_context / self.page_size)
        return torch.arange(batch * blocks_per_user, dtype=torch.int32).reshape(batch, blocks_per_user)

    def reset_cache(self):
        for layer in self.layers:
            for name in ("key", "value", "conv", "recurrent"):
                if name in layer.caches:
                    ttnn.multiply(layer.caches[name], 0.0, output_tensor=layer.caches[name])
        # reset_cache is a public request boundary: cache invalidation must be
        # complete, not merely queued, when the method returns.
        ttnn.synchronize_device(self.mesh_device)

    def reset_slots(self, slots):
        """Reset request-local linear state for selected fixed slots.

        Paged K/V is logically invalidated: the required subsequent prefill
        overwrites every prefix position that decode can observe.  Avoiding a
        whole-slot KV clear preserves the established cache capacity and is
        safe because the generator forbids decode of a reset slot until that
        prefill completes.
        """
        slots = sorted({int(slot) for slot in slots})
        if any(slot < 0 or slot >= self.batch for slot in slots):
            raise ValueError("slot index is outside the fixed batch")
        if not slots:
            return
        active = torch.ones(self.batch, dtype=torch.bfloat16)
        active[slots] = 0
        active = active.contiguous()
        mask = ttnn.from_torch(
            active,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        conv_mask = ttnn.reshape(mask, (1, self.batch, 1, 1))
        recurrent_mask = ttnn.reshape(mask, (self.batch, 1, 1, 1))
        for layer in self.layers:
            if layer.layer_kind != "linear_attention":
                continue
            ttnn.multiply(layer.caches["conv"], conv_mask, output_tensor=layer.caches["conv"])
            ttnn.multiply(layer.caches["recurrent"], recurrent_mask, output_tensor=layer.caches["recurrent"])
        ttnn.synchronize_device(self.mesh_device)
        ttnn.deallocate(mask)

    def remap_slots(self, remap):
        """Gather model-owned linear state into vLLM's new decode rows.

        ``remap[new_slot] = old_slot``.  Full-attention K/V is intentionally
        excluded: vLLM owns it and selects it through the page table.
        """
        remap = [int(slot) for slot in torch.as_tensor(remap).reshape(-1)[: self.batch]]
        if sorted(remap) != list(range(self.batch)):
            raise ValueError("linear-state slot remap must be a full permutation")
        if all(new == old for new, old in enumerate(remap)):
            return
        for layer in self.layers:
            if layer.layer_kind != "linear_attention":
                continue
            for name, dim in (("conv", 1), ("recurrent", 0)):
                cache = layer.caches[name]
                rows = [cache[:, old : old + 1] if dim == 1 else cache[old : old + 1] for old in remap]
                reordered = ttnn.concat(rows, dim=dim, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.copy(reordered, cache)
                for row in rows:
                    ttnn.deallocate(row)
                ttnn.deallocate(reordered)
        ttnn.synchronize_device(self.mesh_device)

    def embed_tokens(self, token_ids):
        return ttnn.embedding(
            token_ids,
            self.embedding,
            layout=ttnn.TILE_LAYOUT,
            dtype=self.precision_config.activation_residual_dtype,
        )

    def terminal_forward(self, hidden_states, *, pad_decode_rows: bool = False):
        logical_rows = hidden_states.shape[-2]
        tiled_rows = math.ceil(logical_rows / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        if logical_rows < tiled_rows:
            hidden_states = ttnn.pad(
                hidden_states,
                [(0, 0), (0, 0), (0, tiled_rows - logical_rows), (0, 0)],
                value=0.0,
            )
        hidden_states = ttnn.rms_norm(hidden_states, epsilon=float(self.config.rms_norm_eps), weight=self.final_norm)
        rows = hidden_states.shape[-2]
        if rows > ttnn.TILE_SIZE:
            projected = []
            for start in range(0, rows, ttnn.TILE_SIZE):
                tile = ttnn.slice(
                    hidden_states,
                    (0, 0, start, 0),
                    (1, self.batch, start + ttnn.TILE_SIZE, int(self.config.hidden_size)),
                )
                projected.append(self._project_lm_head_tile(tile))
            output = ttnn.concat(projected, dim=-2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            output = self._project_lm_head_tile(hidden_states)
        # Decode intentionally retains the 32-row common-sampler contract.
        # Prefill exposes only the caller's logical sequence extent.
        if not pad_decode_rows and logical_rows != tiled_rows:
            output = ttnn.slice(output, (0, 0, 0, 0), (1, self.batch, logical_rows, self.lm_head_local_vocab))
        return output

    def select_prefill_terminal_rows(self, hidden_states, prompt_lens):
        """Select each fixed slot's final logical prompt row on device.

        Public prefill normally needs one next-token distribution per slot.  Do
        this selection before the vocabulary projection so long-context calls
        do not materialize ``sequence * vocab`` logits.  A zero-length inactive
        slot contributes a zero hidden row and is ignored by its active mask.
        """
        if len(prompt_lens) != self.batch:
            raise ValueError("prompt_lens must match the model's fixed slot count")
        rows = []
        for slot, length in enumerate(prompt_lens):
            if length:
                row = ttnn.slice(
                    hidden_states,
                    (0, slot, int(length) - 1, 0),
                    (1, slot + 1, int(length), int(self.config.hidden_size)),
                )
            else:
                row = ttnn.multiply(
                    ttnn.slice(hidden_states, (0, slot, 0, 0), (1, slot + 1, 1, int(self.config.hidden_size))),
                    0.0,
                )
            rows.append(row)
        return ttnn.concat(rows, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _project_lm_head_tile(self, hidden_states):
        """Project exactly one sequence tile without an oversized L1 output."""
        # The retained DRAM-sharded LM-head program supports exactly one
        # 32-row M tile. Preserve it for B>1 by projecting each fixed slot as
        # an independent device slice, then concatenate along the batch axis.
        # This changes neither weights nor sharding and adds no host boundary.
        if hidden_states.shape[1] > 1:
            slot_outputs = []
            for slot in range(hidden_states.shape[1]):
                slot_hidden = ttnn.slice(
                    hidden_states,
                    (0, slot, 0, 0),
                    (1, slot + 1, hidden_states.shape[-2], hidden_states.shape[-1]),
                )
                slot_outputs.append(self._project_lm_head_tile(slot_hidden))
            return ttnn.concat(slot_outputs, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        sequence_rows = hidden_states.shape[-2]
        shape = tuple(hidden_states.shape)
        rows = math.prod(shape[:-1])
        assert rows == sequence_rows == ttnn.TILE_SIZE
        hidden_states = ttnn.to_memory_config(
            hidden_states,
            _l1_width_memory_config(rows=rows, width=int(self.config.hidden_size), cores=self.lm_head_cores),
        )
        outputs = []
        for weight, width in zip(self.lm_head_weights, self.lm_head_chunk_sizes):
            chunk = ttnn.linear(
                hidden_states,
                weight,
                dtype=self.lm_head_dtype,
                program_config=ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=1,
                    per_core_M=math.ceil(rows / ttnn.TILE_SIZE),
                    per_core_N=math.ceil(width / (ttnn.TILE_SIZE * self.lm_head_cores)),
                ),
                compute_kernel_config=self.lm_head_compute,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            outputs.append(
                ttnn.sharded_to_interleaved(chunk, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                if chunk.is_sharded()
                else chunk
            )
        output = ttnn.concat(outputs, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return output

    def prefill_forward(
        self,
        *,
        token_ids,
        page_table,
        current_positions,
        sequence_mask=None,
        conv_state_selectors=None,
        return_hidden=False,
        logit_positions=None,
        cache_page_table=None,
    ):
        sequence = token_ids.shape[-1]
        if sequence > self.PREFILL_STACK_CHUNK_SIZE and not return_hidden:
            return self._prefill_forward_streaming(
                token_ids=token_ids,
                page_table=page_table,
                current_positions=current_positions,
                sequence_mask=sequence_mask,
                conv_state_selectors=conv_state_selectors,
                logit_positions=logit_positions,
                cache_page_table=cache_page_table,
            )
        hidden = self.embed_tokens(token_ids)
        # Embedding is [B,S,H]; the inherited decoder contract is [1,B,S,H].
        hidden = ttnn.reshape(hidden, (1, self.batch, token_ids.shape[-1], int(self.config.hidden_size)))
        for layer in self.layers:
            hidden = layer.prefill_forward(
                hidden_states=hidden,
                page_table=page_table,
                current_positions=current_positions,
                sequence_mask=sequence_mask,
                conv_state_selectors=conv_state_selectors,
                cache_page_table=cache_page_table,
            )
        if return_hidden:
            return hidden
        if logit_positions is not None:
            hidden = self.select_prefill_terminal_rows(hidden, logit_positions)
        return self.terminal_forward(hidden)

    def _prefill_forward_streaming(
        self,
        *,
        token_ids,
        page_table,
        current_positions,
        sequence_mask,
        conv_state_selectors,
        logit_positions,
        cache_page_table,
    ):
        """Prefill long prompts without materializing a sequence-sized layer output.

        Causal full attention and gated-delta recurrence both admit an ordered
        chunk traversal.  Every chunk crosses the complete TP4 layer stack
        before its residual is released; full-attention layers read prior K/V
        from their paged caches and linear-attention layers carry their normal
        conv/recurrent state.  The normal public path retains only each slot's
        terminal prompt row.
        """
        if logit_positions is None:
            raise ValueError("long streaming prefill returns terminal prompt logits only")
        sequence = token_ids.shape[-1]
        terminal_rows = [None] * self.batch
        for start in range(0, sequence, self.PREFILL_STACK_CHUNK_SIZE):
            end = min(start + self.PREFILL_STACK_CHUNK_SIZE, sequence)
            token_chunk = ttnn.slice(token_ids, (0, start), (self.batch, end))
            position_chunk = ttnn.slice(current_positions, (0, start), (self.batch, end))
            hidden = self.embed_tokens(token_chunk)
            hidden = ttnn.reshape(hidden, (1, self.batch, end - start, int(self.config.hidden_size)))
            metadata_start = start // 64
            metadata_end = math.ceil(end / 64)
            masks = sequence_mask[metadata_start:metadata_end] if sequence_mask is not None else None
            selectors = conv_state_selectors[metadata_start:metadata_end] if conv_state_selectors is not None else None
            for layer in self.layers:
                layer._prefill_chunk_start = start
                previous = hidden
                hidden = layer.prefill_forward(
                    hidden_states=hidden,
                    page_table=page_table,
                    current_positions=position_chunk,
                    sequence_mask=masks,
                    conv_state_selectors=selectors,
                    cache_page_table=cache_page_table,
                )
                if previous is not hidden:
                    ttnn.deallocate(previous)
            for slot, length in enumerate(logit_positions):
                if terminal_rows[slot] is None and start < int(length) <= end:
                    terminal_rows[slot] = ttnn.clone(
                        ttnn.slice(
                            hidden,
                            (0, slot, int(length) - start - 1, 0),
                            (1, slot + 1, int(length) - start, int(self.config.hidden_size)),
                        ),
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
            ttnn.deallocate(hidden)
            ttnn.deallocate(token_chunk)
            ttnn.deallocate(position_chunk)
        if any(row is None for row in terminal_rows):
            # Zero-length fixed slots are inactive.  Use a device zero row so
            # the terminal keeps its ordinary fixed-batch contract.
            exemplar = next((row for row in terminal_rows if row is not None), None)
            if exemplar is None:
                raise ValueError("streaming prefill requires at least one active prompt")
            for slot, row in enumerate(terminal_rows):
                if row is None:
                    terminal_rows[slot] = ttnn.multiply(exemplar, 0.0)
        hidden = ttnn.concat(terminal_rows, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return self.terminal_forward(hidden)

    def clear_prefill_request_state(self):
        """Drop decoder references to generator-owned prefill metadata.

        Call only after queued prefill consumers have completed.  The tensors
        themselves remain owned by the generator, which deallocates them after
        clearing these aliases.
        """
        for layer in self.layers:
            layer._sequence_masks = None
            layer._conv_state_selector_chunks = None
            layer._sequence_mask = None
            layer._conv_state_selectors = None
            layer._cache_page_table = None
            layer._prefill_chunk_start = None

    def decode_forward(self, *, token_ids, page_table, current_positions, active_mask=None, return_hidden=False):
        # The common sampler writes token feedback as [1,1,1,32].  Keep that
        # persistent tensor as the trace input and select only the model's
        # active fixed slots on device; no host token reconstruction is needed.
        if len(token_ids.shape) == 4:
            token_ids = ttnn.slice(token_ids, (0, 0, 0, 0), (1, 1, 1, self.batch))
            token_ids = ttnn.reshape(token_ids, (self.batch, 1))
        if len(current_positions.shape) == 4:
            current_positions = ttnn.to_layout(current_positions, ttnn.ROW_MAJOR_LAYOUT)
            current_positions = ttnn.slice(current_positions, (0, 0, 0, 0), (1, 1, 1, self.batch))
            current_positions = ttnn.reshape(current_positions, (self.batch,))
        hidden = self.embed_tokens(token_ids)
        hidden = ttnn.reshape(hidden, (1, 1, self.batch, int(self.config.hidden_size)))
        for layer in self.layers:
            hidden = layer.decode_forward(
                hidden_states=hidden,
                page_table=page_table,
                current_positions=current_positions,
                active_mask=active_mask,
            )
        return hidden if return_hidden else self.terminal_forward(hidden, pad_decode_rows=True)


__all__ = ["Qwen36Model", "SnapshotReader"]
