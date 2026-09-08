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
  x = input_layernorm(x)                  # S2I out — QKV wants interleaved
  x = self_attn(x)                        # TP all-reduce may gather width-sharded
  x = post_attention_layernorm(x)         # keep width-sharded (decode + short prefill)
  x = residual + x                        # residual I2S onto LN layout when needed

  residual = x
  x = pre_feedforward_layernorm(x)        # already_sharded in → keep out
  x = mlp(x)                              # WH prefill: S2I→L1 before gate_up (CB clash)

  if enable_moe_block:
    x_1 = post_feedforward_layernorm_1(x)
    x_flat = residual.reshape(-1, H)     # router input = pre-norm residual
    _, top_k_w, top_k_idx = router(x_flat)
    x_2 = pre_feedforward_layernorm_2(x_flat)
    x_2 = experts(x_2, top_k_idx, top_k_w)
    x_2 = post_feedforward_layernorm_2(x_2)
    x = x_1 + x_2

  x = post_feedforward_layernorm(x)       # keep width-sharded after MLP AR
  x = (residual + x) * layer_scalar       # fused BinaryNg when no PLI path
"""

import torch

import ttnn
from models.demos.gemma4.tt.attention import Gemma4Attention, Gemma4AttentionConfig
from models.demos.gemma4.tt.attention.operations import prefill_matmul_in0_memcfg
from models.demos.gemma4.tt.gemma4_attention_config import get_attention_program_config
from models.demos.gemma4.tt.moe import MoEBlock
from models.demos.gemma4.tt.rms_norm import (
    RMSNorm,
    align_to_memcfg,
    align_to_sharded,
    maybe_interleave,
    norm_keep_sharded_enabled,
    prefill_mlp_island_enabled,
    sharded_norm_enabled,
    width_shard_input_memcfg,
)
from models.demos.gemma4.tt.shared_mlp import SharedMLP
from models.demos.gemma4.utils.general_utils import get_cache_file_name
from models.demos.gemma4.utils.substate import substate


def _residual_add(residual, other, *, scale=None):
    """Residual add that preserves a width-sharded L1 island when possible.

    If ``other`` (typically a keep-sharded LN output) is width-sharded, bring
    ``residual`` onto the same layout so the sum stays sharded for the next LN /
    matmul. Mirrors the AR→LN→MLP layout island: one I2S of the residual beats
    S2I→DRAM on every LN output. Callers still deallocate the original
    ``residual`` / ``other``; any layout temps are freed here.

    Optional ``scale`` fuses ``(a + b) * scale`` via BinaryNg post-activations
    (drops a separate layer_scalar mul when there is no PLI path after the add).
    """
    a, a_owned = residual, False
    b, b_owned = other, False
    if other.is_sharded() and (not residual.is_sharded() or residual.memory_config() != other.memory_config()):
        a, a_owned = align_to_sharded(residual, other)
    elif residual.is_sharded() and not other.is_sharded():
        b, b_owned = align_to_sharded(other, residual)
    kwargs = {}
    if scale is not None and scale != 1.0:
        kwargs["activations"] = [ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, float(scale))]
    if a.is_sharded():
        kwargs["memory_config"] = a.memory_config()
    out = ttnn.add(a, b, **kwargs)
    if a_owned:
        a.deallocate(True)
    if b_owned:
        b.deallocate(True)
    return out


def _prefill_park_l1_interleaved_only(tensor):
    """Move *interleaved* L1 activations to DRAM before input_layernorm.

    Width-sharded L1 from the prior layer's post-MLP island is consumed
    in-place by input LN (``already_sharded`` → S2I to matmul L1). Parking
    sharded L1 to DRAM first added S2I→DRAM then I2S→sharded then S2I again.

    L1 *interleaved* (e.g. post-embed Tilize) still parks once so sharded-norm
    CBs are not competing with a second L1 buffer before QKV.
    """
    if tensor is None:
        return tensor
    try:
        buf = tensor.memory_config().buffer_type
    except (AttributeError, RuntimeError):
        # Not a device tensor (or ttnn declined to report a layout) -- leave it
        # where it is. Narrow on purpose: a real failure inside
        # ``to_memory_config`` below must surface, not read as "keep in L1".
        return tensor
    if buf != ttnn.BufferType.L1 or tensor.is_sharded():
        return tensor
    parked = ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG)
    tensor.deallocate(True)
    return parked


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
        # 1. Attention block: norm -> attn -> post_attn_norm -> residual add
        # Width-sharded AR→LN island (decode, and short dense prefill): after the
        # attn all-reduce the gather may already be LN's width-shard layout. Keep
        # post-attn LN, residual, pre-MLP LN, and post-MLP LN in that domain —
        # drops the post-LN S2I→DRAM that otherwise runs on every sharded norm.
        # input_layernorm still S2I's out: QKV needs interleaved. Prefill gate_up
        # also S2I's (WH 1D CBs clash with sharded in0); decode gate_up on BH
        # consumes the shard via DramShardedLinear. MoE is not sharded-safe.
        padded_h = ((int(hidden_states.shape[-2]) + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        keep_island_decode = norm_keep_sharded_enabled() and not self.enable_moe_block and is_decode
        keep_island_prefill = (not is_decode) and prefill_mlp_island_enabled(
            padded_h, batch_size=batch_size, enable_moe=self.enable_moe_block
        )
        keep_island = keep_island_decode or keep_island_prefill
        residual = hidden_states
        # Prefetch the residual onto the LN width-shard layout *before* input_LN so
        # one I2S serves both the norm (already_sharded) and the post-attn residual
        # add. Batch>1 reshape of the residual is interleaved-only; skip it there.
        if keep_island and sharded_norm_enabled() and batch_size <= 1:
            ln_memcfg = width_shard_input_memcfg(self.mesh_device, int(hidden_states.shape[-1]), padded_h)
            if ln_memcfg is not None:
                aligned, owned = align_to_memcfg(hidden_states, ln_memcfg)
                if owned:
                    hidden_states.deallocate(True)
                residual = aligned
                hidden_states = aligned
        # Prefill: park L1 interleaved only (embed Tilize). Width-sharded L1 from
        # the prior post-MLP island flows into input LN without a DRAM bounce.
        if not is_decode:
            parked = _prefill_park_l1_interleaved_only(hidden_states)
            if parked is not hidden_states:
                residual = parked
                hidden_states = parked
        # Decode: land the input-LN S2I straight in L1 interleaved, which is where
        # the QKV matmul reads in0 from anyway. The default (DRAM) sent the norm
        # output on a full DRAM round-trip, because apply_qkv_projection always
        # passes a program config, so hoist_prefill_matmul_in0_if_needed copied it
        # back to L1 — one CopyDeviceOperation per layer. The
        # matmul now sees the same bits in the same buffer type under the same
        # program config, so the change is bit-exact by construction.
        if is_decode:
            ln_out_memcfg = ttnn.L1_MEMORY_CONFIG
        else:
            ln_out_memcfg = prefill_matmul_in0_memcfg(padded_h, self.hidden_size)
        normed = self.input_layernorm.forward(
            hidden_states,
            keep_sharded=False,
            interleaved_memory_config=ln_out_memcfg,
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
            attn_output = self.post_attention_layernorm.forward(attn_output, keep_sharded=keep_island)
            if not is_decode and batch_size > 1:
                residual = ttnn.reshape(
                    residual, [1, 1, residual.shape[-2] * residual.shape[-3] * residual.shape[0], -1]
                )
            hidden_states = _residual_add(residual, attn_output)
            residual.deallocate(True)
            attn_output.deallocate(True)

        # 2. MLP + MoE block
        residual = hidden_states
        normed = self.pre_feedforward_layernorm.forward(hidden_states, keep_sharded=keep_island)
        # Prefill island is capped at M<=128. Raising `_PREFILL_ISLAND_MAX_HEIGHT`
        # without parking residual+LN to DRAM before gate_up reopens the WH 1D
        # CB clash at demo warmup M>=512 (input LN, then gate_up).
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
        # MLP all-reduce may gather into the LN width-shard layout; keep it
        # through the residual so the next layer's input_LN can skip I2S.
        # Fuse layer_scalar into this add when there is no PLI path after
        # (HF applies scalar after PLI; 31B dense has no PLI).
        hidden_states = self.post_feedforward_layernorm.forward(hidden_states, keep_sharded=keep_island)
        has_pli = (
            self.hidden_size_per_layer_input and per_layer_input is not None and hasattr(self, "per_layer_input_gate")
        )
        fuse_layer_scalar = self.layer_scalar != 1.0 and not has_pli
        combined = _residual_add(
            residual,
            hidden_states,
            scale=self.layer_scalar if fuse_layer_scalar else None,
        )
        residual.deallocate(True)
        hidden_states.deallocate(True)

        hidden_states = combined

        # Per-layer input embeddings (E2B/E4B) — BEFORE layer_scalar (matching HF order)
        # PLI matmuls need interleaved activations.
        if has_pli:
            hidden_states = maybe_interleave(hidden_states)
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

        # Layer scalar — AFTER PLI (matching HF order). Dense/no-PLI already fused
        # into the residual add above. Keep width-sharded when possible so the
        # next layer's input_LN can skip I2S (island across layers).
        if self.layer_scalar != 1.0 and not fuse_layer_scalar:
            if hidden_states.is_sharded():
                hidden_states = ttnn.mul(
                    hidden_states,
                    self.layer_scalar,
                    memory_config=hidden_states.memory_config(),
                )
            else:
                hidden_states = ttnn.mul(hidden_states, self.layer_scalar)

        return hidden_states
