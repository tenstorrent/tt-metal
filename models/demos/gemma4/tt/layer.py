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

GEMMA4_LAYER_NORM_FUSED=1 (decode rows, with GEMMA4_NORM_FUSED=2 and GEMMA4_MOE_FUSED=1): the same arithmetic
as five ``custom_ops/fused_norm_decode`` programs per layer instead of 8 norms + 3 adds + 1 multiply:
``post_attention_layernorm + residual``, ``[pre_feedforward_layernorm, pre_feedforward_layernorm_2, the router's
norm * scale * hidden**-0.5]`` from one row-sum exchange, ``post_feedforward_layernorm_1(mlp) +
post_feedforward_layernorm_2(experts)``, ``post_feedforward_layernorm + residual, * layer_scalar``, and the next
layer's ``input_layernorm``.

On the dense variants (31B / 12B, no MoE block) the same flags fuse the layer tail into three programs instead of
4 norms (one program each under GEMMA4_NORM_FUSED=2) + 2 residual adds + 1 scalar multiply:
``post_attention_layernorm + residual``, ``pre_feedforward_layernorm``, ``post_feedforward_layernorm + residual,
* layer_scalar`` (plus the ``input_layernorm`` program at the top of the layer).
"""


import torch

import ttnn
from models.demos.gemma4.custom_ops.fused_norm_decode import FusedNormDecode
from models.demos.gemma4.tt.attention import Gemma4Attention, Gemma4AttentionConfig
from models.demos.gemma4.tt.decode_matmul import flag, is_decode_rows
from models.demos.gemma4.tt.dram_sharded import DramShardedLinear
from models.demos.gemma4.tt.gemma4_attention_config import get_attention_program_config
from models.demos.gemma4.tt.mesh_orientation import decode_flag
from models.demos.gemma4.tt.moe import MoEBlock
from models.demos.gemma4.tt.rms_norm import RMSNorm
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
                layer_idx=layer_idx,
            )

        # GEMMA4_NORM_FUSED=2: the router's scale as the row-replicated gamma of the layer's fused norm program
        self._router_scale_full = (
            FusedNormDecode.full_gamma(self.moe.router.scale)
            if self.enable_moe_block and decode_flag("GEMMA4_NORM_FUSED") == "2" and self.moe.router.scale is not None
            else None
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

    def _fused_layer_norms_ok(self, rows):
        """GEMMA4_LAYER_NORM_FUSED=1 on tp-only decode rows: the fused norm op (GEMMA4_NORM_FUSED=2) and the fused
        MoE path (GEMMA4_MOE_FUSED=1, whole experts, no lanes, no decode kernel) are both in use, so the layer can
        hand the router its pre-normed input."""
        if not (flag("GEMMA4_LAYER_NORM_FUSED") and decode_flag("GEMMA4_NORM_FUSED") == "2"):
            return False
        if not self.enable_moe_block:
            return False
        moe = self.moe
        return (
            moe.fused_decode_ok()
            and self.post_attention_layernorm.tt_weight_full is not None
            and self._router_scale_full is not None
            and not self.hidden_size_per_layer_input
            and int(rows.shape[-1]) == self.hidden_size
        )

    def _fused_dense_layer_norms_ok(self, rows):
        """GEMMA4_LAYER_NORM_FUSED=1 on the decode rows of a dense layer (no MoE block): the fused norm op
        (GEMMA4_NORM_FUSED=2) carries the residual adds and the layer scalar, so the tail is three programs."""
        if not (flag("GEMMA4_LAYER_NORM_FUSED") and decode_flag("GEMMA4_NORM_FUSED") == "2"):
            return False
        return (
            not self.enable_moe_block
            and not self.hidden_size_per_layer_input
            and self.post_attention_layernorm.tt_weight_full is not None
            and self.pre_feedforward_layernorm.tt_weight_full is not None
            and self.post_feedforward_layernorm.tt_weight_full is not None
            and int(rows.shape[-1]) == self.hidden_size
            and not rows.is_sharded()
        )

    def _sharded_act_layout(self, proj, rows, is_decode):
        """The L1 width-sharded activation layout of a DRAM-sharded decode matmul for the norm before it to write
        (GEMMA4_NORM_SHARDED_OUT=1 on dense decode rows: the fused norm with GEMMA4_NORM_FUSED=2, the native
        width-sharded LayerNorm on the matmul's grid otherwise; a path that cannot honour it returns interleaved
        rows and the matmul reshards as before), else None."""
        if not (is_decode and not self.enable_moe_block and decode_flag("GEMMA4_NORM_SHARDED_OUT") == "1"):
            return None
        if not is_decode_rows(rows) or rows.is_sharded():
            return None
        if not isinstance(proj, DramShardedLinear) or not proj.accepts_sharded_rows():
            return None
        return proj.act_memory_config

    def _forward_tail_fused_dense(self, residual, attn_output):
        """The dense layer after attention as three fused norm programs (see the module docstring)."""
        eps = self.post_attention_layernorm.eps
        gate_up_act = self._sharded_act_layout(self.shared_mlp.gate_up_proj, attn_output, True)
        # post_attention_layernorm(attn) + residual
        hidden = FusedNormDecode.op(
            attn_output, [self.post_attention_layernorm.tt_weight_full], eps, residual=residual
        )[0]
        residual.deallocate(True)
        attn_output.deallocate(True)
        normed = FusedNormDecode.op(
            hidden, [self.pre_feedforward_layernorm.tt_weight_full], eps, out_memory_configs=[gate_up_act]
        )[0]
        mlp_output = self.shared_mlp(normed)
        normed.deallocate(True)
        # post_feedforward_layernorm + residual, times the layer scalar
        out = FusedNormDecode.op(
            mlp_output,
            [self.post_feedforward_layernorm.tt_weight_full],
            eps,
            residual=hidden,
            scalars=[self.layer_scalar],
        )[0]
        mlp_output.deallocate(True)
        hidden.deallocate(True)
        return out

    def _forward_tail_fused(self, residual, attn_output):
        """The layer after attention as five fused norm programs (see the module docstring)."""
        eps = self.post_attention_layernorm.eps
        router = self.moe.router
        # post_attention_layernorm(attn) + residual
        hidden = FusedNormDecode.op(
            attn_output, [self.post_attention_layernorm.tt_weight_full], eps, residual=residual
        )[0]
        residual.deallocate(True)
        attn_output.deallocate(True)
        # pre_feedforward_layernorm / pre_feedforward_layernorm_2 / the router norm from one exchange
        normed, expert_input, router_in = FusedNormDecode.op(
            hidden,
            [
                self.pre_feedforward_layernorm.tt_weight_full,
                self.pre_feedforward_layernorm_2.tt_weight_full,
                self._router_scale_full,
            ],
            eps,
            scalars=[1.0, 1.0, router.scalar_root_size],
        )
        mlp_output = self.shared_mlp(normed)
        normed.deallocate(True)
        routing = router.fused_routing(router_in, self.moe.experts.weights, self.moe.mesh_config, prenormed=True)
        router_in.deallocate(True)
        expert_output = self.moe.experts(expert_input, routing)
        expert_input.deallocate(True)
        # post_feedforward_layernorm_1(mlp) + post_feedforward_layernorm_2(experts)
        summed = FusedNormDecode.op(
            mlp_output,
            [self.post_feedforward_layernorm_1.tt_weight_full],
            eps,
            x2=expert_output,
            gamma2=self.post_feedforward_layernorm_2.tt_weight_full,
        )[0]
        mlp_output.deallocate(True)
        expert_output.deallocate(True)
        # post_feedforward_layernorm + residual, times the layer scalar
        out = FusedNormDecode.op(
            summed, [self.post_feedforward_layernorm.tt_weight_full], eps, residual=hidden, scalars=[self.layer_scalar]
        )[0]
        summed.deallocate(True)
        hidden.deallocate(True)
        return out

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
        rope_prepared=None,
        packed=None,
        chunk_start_idx=None,
        chunk_page_table=None,
        mm_attn_mask=None,
        traced_short_bucket=False,
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
        # 1. Attention block: norm -> attn -> post_attn_norm -> residual add
        residual = hidden_states
        # GEMMA4_NORM_SHARDED_OUT=1 (dense decode rows, fused norm): the normed rows land in the DRAM-sharded wqkv
        # matmul's L1 activation layout, so that matmul skips its interleaved -> sharded copy.
        wqkv_act = self._sharded_act_layout(self.self_attn.weights.wqkv, hidden_states, is_decode)
        normed = (
            self.input_layernorm.forward(hidden_states, out_memory_config=wqkv_act)
            if wqkv_act is not None
            else self.input_layernorm.forward(hidden_states)
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
            rope_prepared=rope_prepared,
            packed=packed,
            chunk_start_idx=chunk_start_idx,
            chunk_page_table=chunk_page_table,
            mm_attn_mask=mm_attn_mask,
            traced_short_bucket=traced_short_bucket,
        )

        if (
            not isinstance(attn_output, torch.Tensor)
            and is_decode
            and self.enable_moe_block
            and is_decode_rows(attn_output)
            and self._fused_layer_norms_ok(attn_output)
        ):
            return self._forward_tail_fused(residual, attn_output)
        if (
            not isinstance(attn_output, torch.Tensor)
            and is_decode
            and not self.enable_moe_block
            and is_decode_rows(attn_output)
            and self._fused_dense_layer_norms_ok(attn_output)
        ):
            return self._forward_tail_fused_dense(residual, attn_output)

        if isinstance(attn_output, torch.Tensor):
            hidden_states = residual
        else:
            attn_output = self.post_attention_layernorm.forward(attn_output)
            if not is_decode and batch_size > 1:
                residual = ttnn.reshape(
                    residual, [1, 1, residual.shape[-2] * residual.shape[-3] * residual.shape[0], -1]
                )
            hidden_states = ttnn.add(residual, attn_output)
            residual.deallocate(True)
            attn_output.deallocate(True)

        # 2. MLP + MoE block
        residual = hidden_states
        # GEMMA4_NORM_SHARDED_OUT=1 (dense decode rows): the normed rows land in the DRAM-sharded gate_up matmul's
        # L1 activation layout (the fused tail above does the same for its pre-FF norm)
        gate_up_act = self._sharded_act_layout(self.shared_mlp.gate_up_proj, hidden_states, is_decode)
        normed = (
            self.pre_feedforward_layernorm.forward(hidden_states, out_memory_config=gate_up_act)
            if gate_up_act is not None
            else self.pre_feedforward_layernorm.forward(hidden_states)
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
        hidden_states = self.post_feedforward_layernorm.forward(hidden_states)
        combined = ttnn.add(residual, hidden_states)
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
        if self.layer_scalar != 1.0:
            hidden_states = ttnn.mul(hidden_states, self.layer_scalar)

        return hidden_states
