# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 Decoder Layer.

Each layer has 7 RMSNorms + layer_scalar:
  - input_layernorm: before attention
  - post_attention_layernorm: after attention, before residual add
  - pre_feedforward_layernorm: before shared MLP
  - post_feedforward_layernorm: after combined MLP+MoE, before final residual add
  - post_feedforward_layernorm_1: after shared MLP output (MoE path only)
  - pre_feedforward_layernorm_2: before expert input (MoE path only)
  - post_feedforward_layernorm_2: after expert output (MoE path only)
  - layer_scalar: learned per-layer scalar

Forward flow (matching HF exactly):
  residual = x
  x = input_layernorm(x)
  x = self_attn(x)
  x = post_attention_layernorm(x)
  x = residual + x

  residual = x
  x = pre_feedforward_layernorm(x)
  x = mlp(x)

  if enable_moe_block:
    x_1 = post_feedforward_layernorm_1(x)
    x_flat = residual.reshape(-1, H)     # router input = pre-norm residual
    _, top_k_w, top_k_idx = router(x_flat)
    x_2 = pre_feedforward_layernorm_2(x_flat)
    x_2 = experts(x_2, top_k_idx, top_k_w)
    x_2 = post_feedforward_layernorm_2(x_2)
    x = x_1 + x_2

  x = post_feedforward_layernorm(x)
  x = residual + x
  x *= layer_scalar
"""

import torch

import ttnn
from models.demos.gemma4.tt.attention import Gemma4Attention, Gemma4AttentionConfig
from models.demos.gemma4.tt.dram_sharded import decode_tuning_enabled
from models.demos.gemma4.tt.gemma4_attention_config import get_attention_program_config
from models.demos.gemma4.tt.moe import MoEBlock
from models.demos.gemma4.tt.precision import resolve_single_tile_dest_acc
from models.demos.gemma4.tt.rms_norm import RMSNorm, decode_width_shard_memcfg
from models.demos.gemma4.tt.shared_mlp import SharedMLP
from models.demos.gemma4.utils.general_utils import get_cache_file_name
from models.demos.gemma4.utils.substate import substate


class Gemma4DecoderLayer:
    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        layer_idx,
        ccl_manager,
        dtype,
        tensor_cache_path,
        mesh_config,
        max_seq_len,
        max_local_batch_size,
        shared_mlp_dtype=None,
        attention_dtype=None,
        experts_dtype=None,
        router_dtype=None,
        single_tile_dest_acc=None,
        bounded_sliding_kv_cache: bool = False,
        transformation_mats=None,  # Legacy — ignored (HF-style RoPE needs no transformation mats)
    ):
        # Per-module dtype overrides default to the model-wide ``dtype`` so
        # callers that don't care about precision config see no change.
        if shared_mlp_dtype is None:
            shared_mlp_dtype = dtype
        if attention_dtype is None:
            attention_dtype = dtype
        if experts_dtype is None:
            experts_dtype = dtype
        if router_dtype is None:
            router_dtype = dtype
        self.mesh_device = mesh_device
        self.layer_idx = layer_idx
        self.hidden_size = hf_config.hidden_size
        self.layer_type = hf_config.layer_types[layer_idx]
        self.enable_moe_block = hf_config.enable_moe_block
        self.hidden_size_per_layer_input = getattr(hf_config, "hidden_size_per_layer_input", 0) or 0

        # Try both key formats (HF uses "model.language_model.layers", tests use "model.layers")
        layer_state = {}
        if state_dict:
            for prefix in [f"model.language_model.layers.{layer_idx}", f"model.layers.{layer_idx}"]:
                layer_state = substate(state_dict, prefix)
                if layer_state:
                    break

        def _norm(name, with_scale=True):
            return RMSNorm(
                mesh_device=mesh_device,
                hf_config=hf_config,
                state_dict=substate(layer_state, name) if layer_state else {},
                tensor_cache_path=f"{tensor_cache_path}/layer_{layer_idx}/{name}" if tensor_cache_path else None,
                mesh_config=mesh_config,
                with_scale=with_scale,
            )

        # 4 norms present on every layer
        self.input_layernorm = _norm("input_layernorm")
        self.post_attention_layernorm = _norm("post_attention_layernorm")
        self.pre_feedforward_layernorm = _norm("pre_feedforward_layernorm")
        self.post_feedforward_layernorm = _norm("post_feedforward_layernorm")

        # 3 additional norms for MoE layers
        if self.enable_moe_block:
            self.post_feedforward_layernorm_1 = _norm("post_feedforward_layernorm_1")
            self.pre_feedforward_layernorm_2 = _norm("pre_feedforward_layernorm_2")
            self.post_feedforward_layernorm_2 = _norm("post_feedforward_layernorm_2")

        # Layer scalar
        if layer_state and "layer_scalar" in layer_state:
            self.layer_scalar = layer_state["layer_scalar"].item()
        else:
            self.layer_scalar = 1.0

        # Dense 12B/31B on a full Wormhole T3K: keep the decode residual stream
        # in the width-sharded L1 layout RMSNorm and the TP all-reduce share,
        # instead of round-tripping it through DRAM between every op. Same gate
        # as the tuned matmul path, and off for multi-user decode; see
        # dram_sharded.decode_tuning_enabled.
        self._tuned_decode = decode_tuning_enabled(mesh_device, hf_config)
        # On that same target the layer scalar can ride the final residual add
        # as an output activation instead of costing its own device op.
        #
        # Restricted to models that keep the m<=32 fp32 dest-accumulation on.
        # ``single_tile_dest_acc=False`` marks a model already known to be
        # fragile about where rounding happens in that path (31B and E2B set it;
        # see precision_overrides.json), and 31B -- the one of those two that
        # reaches here at all, E2B being held off by its per-layer inputs --
        # proves the point: with the scalar fused, its long-context-128k answer
        # collapses into the same 40x repetition loop that flag exists to
        # prevent (532 chars, 27% unique words, reproduced twice), and
        # test_full_model_decode PCC drops 0.99807 -> 0.99617. Gemma4-12B, which
        # leaves the accumulation on, goes the other way: PCC 0.98840 -> 0.99121
        # and 128k stays clean.
        self._fuse_layer_scalar = (
            self.layer_scalar != 1.0 and bool(resolve_single_tile_dest_acc(single_tile_dest_acc)) and self._tuned_decode
        )

        # Attention
        attn_config = Gemma4AttentionConfig(hf_config, layer_idx)
        attn_program_config = get_attention_program_config(attn_config, mesh_config, is_decode=True)
        self.self_attn = Gemma4Attention(
            mesh_device=mesh_device,
            config=attn_config,
            state_dict=substate(layer_state, "self_attn") if layer_state else {},
            ccl_manager=ccl_manager,
            mesh_config=mesh_config,
            program_config=attn_program_config,
            layer_idx=layer_idx,
            tensor_cache_path=f"{tensor_cache_path}/layer_{layer_idx}/self_attn" if tensor_cache_path else None,
            weight_dtype=attention_dtype,
            single_tile_dest_acc=single_tile_dest_acc,
            bounded_sliding_kv_cache=bounded_sliding_kv_cache,
        )

        # Shared/dense MLP (HF key: "mlp")
        self.shared_mlp = SharedMLP(
            mesh_device=mesh_device,
            hf_config=hf_config,
            state_dict=substate(layer_state, "mlp") if layer_state else {},
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
            dtype=shared_mlp_dtype,
            tensor_cache_path=f"{tensor_cache_path}/layer_{layer_idx}/mlp" if tensor_cache_path else None,
            layer_idx=layer_idx,
            single_tile_dest_acc=single_tile_dest_acc,
        )

        # MoE block (router + routed experts) — split dtypes between the two
        if self.enable_moe_block:
            self.moe = MoEBlock(
                mesh_device=mesh_device,
                hf_config=hf_config,
                state_dict=layer_state,  # MoE expects "router.*" and "experts.*" keys
                ccl_manager=ccl_manager,
                mesh_config=mesh_config,
                dtype=experts_dtype,
                router_dtype=router_dtype,
                tensor_cache_path=f"{tensor_cache_path}/layer_{layer_idx}/moe" if tensor_cache_path else None,
            )

        # Per-layer input embeddings (E2B/E4B feature)
        if self.hidden_size_per_layer_input:
            pli_prefix = f"{tensor_cache_path}/layer_{layer_idx}" if tensor_cache_path else None

            if layer_state and "per_layer_input_gate.weight" in layer_state:
                gate_w = layer_state["per_layer_input_gate.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
                proj_w = layer_state["per_layer_projection.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
            else:
                gate_w = None
                proj_w = None

            self.per_layer_input_gate = ttnn.as_tensor(
                gate_w,
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=get_cache_file_name(pli_prefix, "per_layer_input_gate"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.per_layer_projection = ttnn.as_tensor(
                proj_w,
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=get_cache_file_name(pli_prefix, "per_layer_projection"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.post_per_layer_input_norm = _norm("post_per_layer_input_norm")

    def __call__(
        self,
        hidden_states,
        rope_mats,
        position_idx,
        page_table,
        kv_cache,
        is_decode,
        token_index=None,
        per_layer_input=None,
        shared_kv=None,
        keep_kv=False,
        is_kv_shared=False,
        position_idx_cache=None,
        batch_size=1,
        user_id=0,
        valid_seq_len=None,
        sequential_kv_write=False,
        rope_presliced=False,
        packed=None,
        chunk_start_idx=None,
        chunk_page_table=None,
    ):
        """
        Decoder layer forward pass.

        Args:
            hidden_states: [1, 1, seq_len, hidden_size] on device
            rope_mats: precomputed RoPE matrices
            position_idx: current position index
            page_table: paged attention page table
            kv_cache: KV cache for this layer
            is_decode: True for decode mode
            shared_kv: optional (tt_k, tt_v) from source layer for KV sharing (prefill only)
            keep_kv: if True, keep K/V alive for sharing with later layers (prefill only)
            is_kv_shared: if True, this layer shares KV from source (skip K/V proj + cache update)

        Returns:
            hidden_states: [1, 1, seq_len, hidden_size] on device
        """
        # Dense decode without per-layer inputs keeps the residual stream in the
        # exact width-sharded L1 layout shared by RMSNorm and the TP all-reduce.
        # ``stream_memcfg`` stays None everywhere else, and every placement below
        # then resolves to the op default it uses today.
        stream_memcfg = None
        # Short-circuits on the fusion gate so this shape read is unreachable on
        # every mesh and model the fusion does not apply to.
        decode_users = int(hidden_states.shape[-2]) if (is_decode and self._fuse_layer_scalar) else 0
        if is_decode and self._tuned_decode and not self.enable_moe_block and not self.hidden_size_per_layer_input:
            stream_memcfg = decode_width_shard_memcfg(self.mesh_device, hidden_states.shape[-1])
            if stream_memcfg is not None and not hidden_states.is_sharded():
                sharded_hidden_states = ttnn.to_memory_config(hidden_states, stream_memcfg)
                hidden_states.deallocate(True)
                hidden_states = sharded_hidden_states
        shard_stream = (
            stream_memcfg is not None and hidden_states.is_sharded() and hidden_states.memory_config() == stream_memcfg
        )

        # 1. Attention block: norm -> attn -> post_attn_norm -> residual add
        residual = hidden_states
        normed = self.input_layernorm.forward(
            hidden_states,
            interleaved_memory_config=ttnn.L1_MEMORY_CONFIG if shard_stream else None,
        )
        if not is_decode and batch_size > 1:
            attn_in = ttnn.reshape(normed, [batch_size, 1, normed.shape[-2] // batch_size, -1])
        else:
            attn_in = normed
        attn_output = self.self_attn(
            attn_in,
            rope_mats=rope_mats,
            position_idx=position_idx,
            page_table=page_table,
            kv_cache=kv_cache,
            is_decode=is_decode,
            token_index=token_index,
            shared_kv=shared_kv,
            keep_kv=keep_kv,
            is_kv_shared=is_kv_shared,
            position_idx_cache=position_idx_cache,
            batch_size=batch_size,
            user_id=user_id,
            valid_seq_len=valid_seq_len,
            sequential_kv_write=sequential_kv_write,
            rope_presliced=rope_presliced,
            packed=packed,
            chunk_start_idx=chunk_start_idx,
            chunk_page_table=chunk_page_table,
        )

        if isinstance(attn_output, torch.Tensor):
            hidden_states = residual
        else:
            attn_output = self.post_attention_layernorm.forward(attn_output, keep_sharded=shard_stream)
            if not is_decode and batch_size > 1:
                residual = ttnn.reshape(
                    residual, [1, 1, residual.shape[-2] * residual.shape[-3] * residual.shape[0], -1]
                )
            hidden_states = ttnn.add(
                residual,
                attn_output,
                memory_config=stream_memcfg if shard_stream else None,
            )
            residual.deallocate(True)
            attn_output.deallocate(True)

        # 2. MLP + MoE block
        residual = hidden_states
        normed = self.pre_feedforward_layernorm.forward(
            hidden_states,
            keep_sharded=shard_stream,
            interleaved_memory_config=ttnn.L1_MEMORY_CONFIG if shard_stream else None,
        )
        mlp_output = self.shared_mlp(normed)
        normed.deallocate(True)

        if self.enable_moe_block:
            # post_feedforward_layernorm_1 on MLP output
            mlp_normed = self.post_feedforward_layernorm_1.forward(mlp_output)
            mlp_output.deallocate(True)

            # Router input = pre-MLP residual, expert input = normed residual
            # All on device — no CPU round-trip
            residual_for_router = residual
            expert_input = self.pre_feedforward_layernorm_2.forward(residual_for_router)

            # MoE: router(residual) → dense_routing → experts(normed_input, routing)
            expert_output = self.moe(residual_for_router, expert_input)
            expert_input.deallocate(True)

            # post_feedforward_layernorm_2 on expert output
            expert_normed = self.post_feedforward_layernorm_2.forward(expert_output)
            expert_output.deallocate(True)

            # Combine: mlp_normed + expert_normed
            hidden_states = ttnn.add(mlp_normed, expert_normed)
            mlp_normed.deallocate(True)
            expert_normed.deallocate(True)
        else:
            hidden_states = mlp_output

        # post_feedforward_layernorm -> residual add
        hidden_states = self.post_feedforward_layernorm.forward(hidden_states, keep_sharded=shard_stream)
        # ``shard_stream`` is only set for dense decode with no per-layer inputs,
        # which is exactly the case where nothing runs between this add and the
        # layer_scalar multiply below, so the scalar can ride the add and save a
        # device op per layer. Not bit-identical -- the fused form scales in the
        # fp32 destination register where the two-op form packs the sum to bf16
        # first -- which is why ``_fuse_layer_scalar`` is gated on the model's
        # dest-accumulation policy.
        #
        # Restricted to one decode user, and the bound is measured, not derived.
        # At batch-8 the fused form is not reproducible run to run: 13 repeats of
        # the 12B batch-8 demo produced two distinct whole-batch outputs (10x and
        # 3x), where 20 repeats of the two-op form produced one. Both outputs are
        # coherent, so this is a token flip at an argmax near-tie rather than
        # corruption, but a shipped bucket must not vary between runs. Batch-1 is
        # stable over 15 repeats, and batch-32 does not reach the multi-user L1
        # activation path at all. The root cause is not understood -- the
        # arithmetic here is deterministic, so the suspicion is that being ~0.8%
        # faster shifts timing in the CCL path -- so do not widen this without
        # re-running the repeat test at the batch you are widening to.
        fuse_scalar = self._fuse_layer_scalar and shard_stream and decode_users == 1
        scalar_activation = (
            {"activations": [ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, self.layer_scalar)]}
            if fuse_scalar
            else {}
        )
        combined = ttnn.add(
            residual,
            hidden_states,
            memory_config=stream_memcfg if shard_stream else None,
            **scalar_activation,
        )
        residual.deallocate(True)
        hidden_states.deallocate(True)

        hidden_states = combined

        # Per-layer input embeddings (E2B/E4B) — BEFORE layer_scalar (matching HF order)
        if self.hidden_size_per_layer_input and per_layer_input is not None and hasattr(self, "per_layer_input_gate"):
            residual_pli = hidden_states
            from models.demos.gemma4.tt.compute_config import gelu_variant

            gated = ttnn.linear(hidden_states, self.per_layer_input_gate)
            gated = ttnn.gelu(gated, variant=gelu_variant())
            gated = ttnn.mul(gated, per_layer_input)
            projected = ttnn.linear(gated, self.per_layer_projection)
            normed_pli = self.post_per_layer_input_norm.forward(projected)
            hidden_states = ttnn.add(residual_pli, normed_pli)
            if len(hidden_states.shape) > 4:
                hidden_states = ttnn.reshape(hidden_states, (1, 1, hidden_states.shape[-2], self.hidden_size))

        # Layer scalar — AFTER PLI (matching HF order)
        if self.layer_scalar != 1.0 and not fuse_scalar:
            hidden_states = ttnn.mul(
                hidden_states,
                self.layer_scalar,
                memory_config=hidden_states.memory_config() if hidden_states.is_sharded() else None,
            )

        return hidden_states
