# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6 text model over the accepted TP4/EP4 decoder.

Weight conversion and request allocation are setup operations. Device forwards
preserve the decoder's replicated BF16 residual between every layer.
"""

import json
import math
import os
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig

import ttnn

from .multichip_decoder import CollectiveWorkspace, MultichipDecoder
from .precision import actual_compute, assert_compute, device_compute, device_dtype, layer_precision, resolve_precision
from .slot_state import advance_position, copy_if, position_inputs

HF_MODEL = "Qwen/Qwen3.6-35B-A3B"
HF_REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"


class Checkpoint:
    def __init__(self, path=None):
        self.path = (
            Path(path)
            if path
            else Path(os.environ["HF_HOME"]) / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots" / HF_REVISION
        )
        self.index = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]
        self.config = AutoConfig.from_pretrained(self.path, local_files_only=True).text_config

    def select(self, prefix):
        selected = {k: v for k, v in self.index.items() if k.startswith(prefix)}
        result = {}
        for shard in sorted(set(selected.values())):
            with safe_open(self.path / shard, framework="pt", device="cpu") as handle:
                for key, filename in selected.items():
                    if filename == shard:
                        result[key[len(prefix) :]] = handle.get_tensor(key)
        if not result:
            raise KeyError(prefix)
        return result

    def tensor(self, name):
        with safe_open(self.path / self.index[name], framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)


class QwenModel:
    def __init__(
        self,
        mesh_device,
        *,
        checkpoint=None,
        layer_indices=None,
        max_seq_len=None,
        lm_head_mode="dram_sharded",
        fuse_hidden_row_mask=False,
        precision_config=None,
        suppress_inactive_moe_routes=False,
        fuse_moe_route_plan=False,
        specialize_solo_slot0_attention=False,
        fuse_recurrent_state_commit=False,
        native_qk_reader=False,
        packed_postreduce_weights=False,
        prune_postreduce_padding=False,
    ):
        if tuple(mesh_device.shape) != (1, 4):
            raise ValueError("The accepted decoder requires a 1x4 mesh")
        if lm_head_mode not in ("dram_sharded", "interleaved"):
            raise ValueError("Unknown LM-head mode")
        if type(suppress_inactive_moe_routes) is not bool:
            raise TypeError("Inactive MoE route suppression must be a boolean")
        self.suppress_inactive_moe_routes = suppress_inactive_moe_routes
        if type(fuse_moe_route_plan) is not bool:
            raise TypeError("MoE route-plan fusion must be a boolean")
        if fuse_moe_route_plan and not suppress_inactive_moe_routes:
            raise ValueError("MoE route-plan fusion requires inactive MoE route suppression")
        self.fuse_moe_route_plan = fuse_moe_route_plan
        if type(specialize_solo_slot0_attention) is not bool:
            raise TypeError("Solo attention specialization must be a boolean")
        if specialize_solo_slot0_attention and not suppress_inactive_moe_routes:
            raise ValueError("Solo attention specialization requires inactive MoE route suppression")
        self.specialize_solo_slot0_attention = specialize_solo_slot0_attention
        if type(fuse_recurrent_state_commit) is not bool:
            raise TypeError("Recurrent state-commit fusion must be a boolean")
        self.fuse_recurrent_state_commit = fuse_recurrent_state_commit
        if type(native_qk_reader) is not bool:
            raise TypeError("Native Q/K reader must be a boolean")
        if native_qk_reader and not fuse_recurrent_state_commit:
            raise ValueError("Native Q/K reader requires the conditional state-commit path")
        self.native_qk_reader = native_qk_reader
        if type(packed_postreduce_weights) is not bool:
            raise TypeError("Packed post-reduce weights must be a boolean")
        self.packed_postreduce_weights = packed_postreduce_weights
        if type(prune_postreduce_padding) is not bool:
            raise TypeError("Postreduce padding specialization must be a boolean")
        if prune_postreduce_padding and not suppress_inactive_moe_routes:
            raise ValueError("Postreduce padding specialization requires inactive route suppression")
        self.prune_postreduce_padding = prune_postreduce_padding
        self.solo_linear_zero = self.solo_full_zero = None
        self.precision_config = resolve_precision(precision_config)
        self.boundary_dtypes = {
            field: device_dtype(self.precision_config[field])
            for field in ("activation_dtype", "residual_dtype", "logits_dtype", "sampling_dtype")
        }
        self.boundary_observations = {}
        if type(fuse_hidden_row_mask) is not bool:
            raise TypeError("Hidden row extraction fusion must be a boolean")
        self.fuse_hidden_row_mask = fuse_hidden_row_mask
        self.lm_head_mode = lm_head_mode
        self.mesh = mesh_device
        self.checkpoint = Checkpoint(checkpoint)
        self.config = self.checkpoint.config
        self.max_seq_len = self.config.max_position_embeddings if max_seq_len is None else max_seq_len
        if (
            isinstance(self.max_seq_len, bool)
            or not isinstance(self.max_seq_len, int)
            or not 1 <= self.max_seq_len <= self.config.max_position_embeddings
        ):
            raise ValueError("Invalid configured context")
        self.layer_indices = (
            tuple(range(self.config.num_hidden_layers)) if layer_indices is None else tuple(layer_indices)
        )
        self.workspace = CollectiveWorkspace(mesh_device)
        self.memory_views = []
        self.census("before_weights")
        self.layers = []
        self.immutable_tensors = {}
        for index in self.layer_indices:
            sd = self.checkpoint.select(f"model.language_model.layers.{index}.")
            layer = MultichipDecoder.from_state_dict(
                sd,
                hf_config=self.config,
                layer_idx=index,
                mesh_device=mesh_device,
                workspace=self.workspace,
                full_model_optimized=True,
                precision_config=layer_precision(self.precision_config, index),
            )
            # The selected TP4 router uses MeshFP32TopK/native topk exclusively.
            # Its parent constructors also allocate legacy single-chip router
            # inputs/outputs and an InlineSelector that this override never
            # reads. Releasing these unreachable buffers leaves SDPA's static
            # circular-buffer region available at full model depth.
            assert layer._moe.__func__ is MultichipDecoder._moe
            for name in ("router_bias", "router_indices", "router_out", "router_out_ids", "inline_selector"):
                if hasattr(layer, name):
                    delattr(layer, name)
            if layer.kind == "linear_attention":
                # The selected helper's exact SiLU lookup is immutable and
                # independent of layer weights. Adopt immediately: retaining
                # 30 copies exhausts L1 before the full stack can be loaded.
                # Helper instances, norm weights and mutable state stay local.
                table = layer.decode_local_silu.table
                key = type(layer.decode_local_silu).__name__
                if key in self.immutable_tensors:
                    shared = self.immutable_tensors[key]
                    assert table.shape == shared.shape and table.padded_shape == shared.padded_shape
                    assert table.dtype == shared.dtype and table.layout == shared.layout
                    assert table.memory_config() == shared.memory_config()
                    layer.decode_local_silu.table = shared
                else:
                    self.immutable_tensors[key] = table
                del table
            layer.fuse_recurrent_state_commit = self.fuse_recurrent_state_commit
            layer.native_qk_reader = self.native_qk_reader
            layer.packed_postreduce_weights = self.packed_postreduce_weights
            layer.prune_postreduce_padding = self.prune_postreduce_padding
            self.layers.append(layer)
            del sd
            self.census(f"layer_{index}")
            print(f"LOADED_LAYER {index} DRAM {self.memory_views[-1]}", flush=True)
        self.vocab_size = self.config.vocab_size
        self.padded_vocab_size = ((self.vocab_size + 127) // 128) * 128
        embed = self.checkpoint.tensor("model.language_model.embed_tokens.weight")
        self.embedding_weight = self.upload(
            embed, device_dtype(self.precision_config["embedding_dtype"]), layout=ttnn.ROW_MAJOR_LAYOUT, shard_dim=1
        )
        head = embed if self.config.tie_word_embeddings else self.checkpoint.tensor("lm_head.weight")
        head = torch.nn.functional.pad(head.T, (0, self.padded_vocab_size - self.vocab_size))
        self.head_weight = self.upload(
            head, device_dtype(self.precision_config["weight_groups"]["lm_head"]), shard_dim=1
        )
        norm = self.checkpoint.tensor("model.language_model.norm.weight").float() + 1
        self.norm_weight = self.upload(norm.reshape(1, 1, -1), ttnn.float32)
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, math_approx_mode=False
        )
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.lm_head import DramHead

        self.head_compute = device_compute(mesh_device, self.precision_config["compute_fidelities"]["lm_head"])
        self.dram_head = (
            DramHead(self.head_weight, self.mesh, self.head_compute, output_dtype=self.boundary_dtypes["logits_dtype"])
            if lm_head_mode == "dram_sharded"
            else None
        )
        inv = 1.0 / (self.config.rope_parameters["rope_theta"] ** (torch.arange(0, 64, 2).float() / 64))
        angles = torch.arange(self.max_seq_len).float()[:, None] * inv[None, :]
        angles = torch.cat((angles, angles), dim=-1)
        self.rope_cos = self.upload(angles.cos().bfloat16(), ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.rope_sin = self.upload(angles.sin().bfloat16(), ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        if self.specialize_solo_slot0_attention:
            if any(
                not layer.batch_attention_reduce
                or not layer.batch_full_projections
                or not layer.local_expert_dispatch
                or layer.residual_layout != "replicated"
                or layer.attention_collective != "separate"
                for layer in self.layers
            ):
                raise ValueError("Solo attention requires the selected replicated grouped-decode topology")
            # Immutable shared partials are allocated before generator setup or
            # capture. They are never destinations, and all traces may read them.
            self.solo_linear_zero = self.upload(torch.zeros(1, 1, 2048), ttnn.float32)
            self.solo_full_zero = self.upload(torch.zeros(1, 1, 1024), ttnn.bfloat16)
        self.census("all_weights_and_rope")

    def observe_boundary(self, tensor, field, site):
        """Check actual tensor metadata; no device operation or host readback.

        These are fixed BF16 public boundaries, not controls over internal
        FP32 recurrence or native grouped-MoE BFP8 intermediates.
        Observations are Python dispatch/capture evidence, not replay counters.
        """
        expected = self.boundary_dtypes[field]
        if tensor.dtype != expected:
            raise ValueError(f"{site}: expected {field}={expected}, observed {tensor.dtype}")
        self.boundary_observations[site] = dict(
            field=field, selected=self.precision_config[field], actual_dtype=str(tensor.dtype)
        )
        return tensor

    def precision_summary(self):
        """Assert selected fields against the actual measured construction."""
        assert self.embedding_weight.dtype == device_dtype(self.precision_config["embedding_dtype"])
        assert self.head_weight.dtype == device_dtype(self.precision_config["weight_groups"]["lm_head"])
        assert_compute(self.head_compute, self.precision_config["compute_fidelities"]["lm_head"])
        if self.dram_head is not None:
            assert self.dram_head.weight.dtype == self.head_weight.dtype
            assert self.dram_head.compute is self.head_compute
            assert self.dram_head.output_dtype == self.boundary_dtypes["logits_dtype"]
        assert self.norm_weight.dtype == ttnn.float32
        assert self.compute.math_fidelity == ttnn.MathFidelity.HiFi2
        return dict(
            config_id=self.precision_config["config_id"],
            selected_policy=self.precision_config,
            layers=[
                dict(layer_index=index, **layer.precision_summary())
                for index, layer in zip(self.layer_indices, self.layers)
            ],
            lm_head=dict(
                dtype=str(self.head_weight.dtype), compute=actual_compute(self.head_compute), mode=self.lm_head_mode
            ),
            embedding_dtype=str(self.embedding_weight.dtype),
            terminal_norm=dict(dtype=str(self.norm_weight.dtype), compute=actual_compute(self.compute)),
            fixed_boundaries={
                key: value
                for key, value in self.precision_config.items()
                if key not in ("weight_groups", "compute_fidelities", "layer_exceptions")
            },
            observed_boundaries=dict(self.boundary_observations),
            boundary_policy_support={
                field: dict(selected=self.precision_config[field], supported=["bf16"], mode="fixed_model_contract")
                for field in self.boundary_dtypes
            },
            cache=self.cache_spec(),
        )

    def upload(self, value, dtype, *, layout=ttnn.TILE_LAYOUT, shard_dim=None):
        mapper = (
            ttnn.ReplicateTensorToMesh(self.mesh)
            if shard_dim is None
            else ttnn.ShardTensorToMesh(self.mesh, dim=shard_dim)
        )
        return ttnn.from_torch(
            value.contiguous(),
            device=self.mesh,
            dtype=dtype,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    def census(self, stage):
        view = ttnn.get_memory_view(self.mesh, ttnn.BufferType.DRAM)
        row = {"stage": stage, "banks": int(view.num_banks)}
        for name in ("total_bytes_per_bank", "total_bytes_allocated_per_bank", "total_bytes_free_per_bank"):
            row[name] = int(getattr(view, name))
        l1 = ttnn.get_memory_view(self.mesh, ttnn.BufferType.L1)
        row["l1"] = {
            name: int(getattr(l1, name))
            for name in (
                "num_banks",
                "total_bytes_per_bank",
                "total_bytes_allocated_per_bank",
                "total_bytes_free_per_bank",
            )
        }
        self.memory_views.append(row)
        return row

    def embed(self, tokens):
        local = ttnn.embedding(tokens, self.embedding_weight, layout=ttnn.TILE_LAYOUT)
        # Only the embedding boundary gathers hidden shards. Every decoder then
        # consumes and produces the accepted replicated hidden-width contract.
        hidden = ttnn.experimental.all_gather_async(
            local,
            dim=-1,
            cluster_axis=1,
            mesh_device=self.mesh,
            topology=ttnn.Topology.Linear,
            num_links=2,
            multi_device_global_semaphore=self.workspace.ccl.get_ag_ping_pong_semaphore(),
            barrier_semaphore=self.workspace.ccl.get_barrier_semaphore(),
        )

        self.observe_boundary(hidden, "activation_dtype", "embedding.output")
        return self.observe_boundary(hidden, "residual_dtype", "embedding.residual")

    def terminal(self, hidden):
        self.observe_boundary(hidden, "residual_dtype", "terminal.input")
        # A decoder singleton can be L1-sharded, while bounded prefill and
        # batched rows are interleaved DRAM. Use one terminal norm contract;
        # implicit sharded/unsharded dispatch otherwise changes accumulation.
        hidden = ttnn.to_memory_config(hidden, ttnn.DRAM_MEMORY_CONFIG)
        norm = ttnn.rms_norm(
            ttnn.typecast(hidden, ttnn.float32),
            weight=self.norm_weight,
            epsilon=self.config.rms_norm_eps,
            compute_kernel_config=self.compute,
        )
        norm = ttnn.typecast(norm, self.boundary_dtypes["activation_dtype"])
        self.observe_boundary(norm, "activation_dtype", "terminal.normalized_activation")
        # Bounded prefill emits <=32 logits rows, as does fixed-slot decode.
        # Preserve the legacy many-row terminal API with its native geometry.
        if self.dram_head is not None and math.prod(tuple(norm.padded_shape)[:-1]) == 32:
            return self.observe_boundary(self.dram_head(norm), "logits_dtype", "terminal.dram_logits")
        logits = ttnn.linear(
            norm,
            self.head_weight,
            compute_kernel_config=self.head_compute,
            dtype=self.boundary_dtypes["logits_dtype"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return self.observe_boundary(logits, "logits_dtype", "terminal.interleaved_logits")

    def allocate_cache(self, batch=1, *, context_length=None):
        context_length = self.max_seq_len if context_length is None else context_length
        if (
            isinstance(context_length, bool)
            or not isinstance(context_length, int)
            or not 1 <= context_length <= self.max_seq_len
        ):
            raise ValueError("Cache context exceeds the configured model context")
        if not 1 <= batch <= 32:
            raise ValueError("Supported request slots: 1..32")
        # One aggregate physical pool per full-attention layer. Scheduler page
        # tables, not duplicated pools, select each request's owned pages.
        result = []
        for layer in self.layers:
            if layer.kind == "linear_attention":
                result.append([layer.new_recurrent_state() for _ in range(batch)])
                continue
            first = layer.new_kv_state(context_length=context_length)
            states = [first]
            for _ in range(1, batch):
                states.append(layer.new_kv_state(context_length=context_length, kv_cache=(first["k"], first["v"])))
            result.append(states)
        return result

    def cache_spec(self):
        from .cache_contract import describe_cache

        return describe_cache(self)

    def get_max_tokens_all_users(self):
        """Configured aggregate KV budget, shared by all request slots.

        This is the allocation contract, not a free-memory estimator. The
        prepared full-stack memory census validates the advertised ceiling.
        """
        return self.cache_spec()["aggregate_token_capacity"]

    def allocate_kv_cache_per_layer(self, per_layer_specs, *, batch=1):
        from .cache_contract import allocate_kv_cache_per_layer

        return allocate_kv_cache_per_layer(self, per_layer_specs, batch=batch)

    def allocate_kv_cache(self, shape, dtype, num_layers, *, batch=1):
        from .cache_contract import allocate_kv_cache

        return allocate_kv_cache(self, shape, dtype, num_layers, batch=batch)

    def prefill(self, tokens, *, cache, slot=0, start=0, plans=None, return_all_logits=True):
        length = tokens.shape[-1]
        plans = plans or [layer.prepare_prefill(seq_len=length, start_pos=start) for layer in self.layers]
        hidden = ttnn.reshape(self.embed(tokens), (1, length, self.config.hidden_size))
        for layer, states, plan in zip(self.layers, cache, plans):
            self.observe_boundary(hidden, "residual_dtype", f"legacy_prefill.layer{layer.layer_idx}.input")
            hidden = layer.prefill_forward(hidden, state=states[slot], plan=plan)
            self.observe_boundary(hidden, "residual_dtype", f"legacy_prefill.layer{layer.layer_idx}.output")
        return self.terminal(hidden if return_all_logits else hidden[:, -1:, :])

    def device_positions(self, position, *, activity=None):
        # Runtime lookup consumes the same device index as paged attention.
        rope_position, current = (
            (position, position) if activity is None else position_inputs(position, activity, self.mesh)
        )
        index = ttnn.typecast(ttnn.reshape(rope_position, (1, 1)), ttnn.uint32)
        cos = ttnn.embedding(index, self.rope_cos, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(index, self.rope_sin, layout=ttnn.TILE_LAYOUT)
        result = {
            "cos": ttnn.reshape(cos, (1, 1, 1, 64)),
            "sin": ttnn.reshape(sin, (1, 1, 1, 64)),
            "current_pos": current,
        }
        if activity is not None:
            result["activity"] = activity
        return result

    def decode(self, tokens, *, cache, positions, slots, activity=None, active_mask=None, solo_slot0_attention=False):
        # Masked fixed-slot decode captures canonical physical rows once. The
        # optional mask is device data, never a Python branch on live activity.
        # The existing unmasked solo path keeps its exact operation sequence.
        if activity is not None and len(activity) != len(positions):
            raise ValueError("Activity must provide one persistent scalar per physical slot")
        route_mask = None
        if self.suppress_inactive_moe_routes and activity is not None and len(slots) > 1:
            if active_mask is None:
                raise ValueError("Masked grouped decode requires the persistent generator activity vector")
            route_mask = active_mask
        fused_hidden_mask = None
        # Other accepted serving capacities retain generic extraction/masking;
        # the fused helper has kernels only for these prepared row counts.
        if self.fuse_hidden_row_mask and activity is not None and len(slots) in (2, 4, 32):
            if active_mask is None:
                raise ValueError("Fused hidden row extraction requires the persistent activity vector")
            fused_hidden_mask = active_mask
        if solo_slot0_attention and (
            not self.specialize_solo_slot0_attention or tuple(slots) != (0, 1) or route_mask is None
        ):
            raise ValueError("Solo attention requires its prepared opt-in B2 graph and persistent activity mask")
        hidden = [
            ttnn.reshape(self.embed(tokens[..., slot : slot + 1]), (1, 1, self.config.hidden_size)) for slot in slots
        ]
        positional = [
            self.device_positions(positions[slot], activity=None if activity is None else activity[slot])
            for slot in slots
        ]
        if activity is not None:
            hidden = self._mask_hidden_rows(hidden, slots, activity)
        for layer, states in zip(self.layers, cache):
            for value in hidden:
                self.observe_boundary(value, "residual_dtype", f"decode.layer{layer.layer_idx}.input")
            hidden = layer.decode_batch(
                hidden,
                states=[states[s] for s in slots],
                positions=positional,
                hidden_row_mask=fused_hidden_mask,
                hidden_row_slots=slots if fused_hidden_mask is not None else None,
                route_activity_mask=route_mask,
                fuse_moe_route_plan=self.fuse_moe_route_plan and route_mask is not None,
                route_slots=slots if route_mask is not None else None,
                solo_slot0_attention=solo_slot0_attention,
                solo_zero_partial=(self.solo_linear_zero if layer.kind == "linear_attention" else self.solo_full_zero)
                if solo_slot0_attention
                else None,
            )
            for value in hidden:
                self.observe_boundary(value, "residual_dtype", f"decode.layer{layer.layer_idx}.output")
            if activity is not None and fused_hidden_mask is None:
                hidden = self._mask_hidden_rows(hidden, slots, activity)
        logits = self.terminal(
            ttnn.concat([ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG) for x in hidden], dim=1)
            if len(hidden) > 1
            else hidden[0]
        )
        for slot in slots:
            if activity is None:
                ttnn.plus_one(positions[slot])
            else:
                advance_position(positions[slot], activity[slot], self.mesh)
        return logits

    def _mask_hidden_rows(self, hidden, slots, activity):
        result = []
        for value, slot in zip(hidden, slots):
            value = ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG)
            # Decoder row slices can share backing storage. Give the masked
            # row its own output instead of zeroing another row's tile alias.
            masked = ttnn.empty_like(value)
            result.append(copy_if(value, masked, activity[slot], self.mesh, zero_if_inactive=True))
        return result

    def host_logits(self, logits):
        self.observe_boundary(logits, "logits_dtype", "public_logits.device_input")
        return ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=-1))[
            ..., : self.vocab_size
        ].float()
