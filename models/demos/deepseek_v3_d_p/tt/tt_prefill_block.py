# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Callable, Optional

from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import compute_constants, extract_mesh_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe import TtMoe
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
from models.demos.deepseek_v3_d_p.tt.tt_ffn import TtFfn


class TtPrefillBlock(LightweightModule):
    """
    Single transformer block for DeepSeek V3 prefill.

    Constructs and composes: attn_norm → MLA → residual → ffn_norm → FFN/MoE → residual

    Canonical tensor format: [1, 1, seq_len_local, emb_dim_per_tp], TILE_LAYOUT, TP-sharded on last dim.

    State dict keys:
        attn_norm_weight:       torch.Tensor [emb_dim]
        mla_weights:            dict for ttMLA (q_a_proj.weight, q_b_proj.weight, ...)
        ffn_norm_weight:        torch.Tensor [emb_dim]
        For MoE layers (layer_idx >= n_dense_layers):
            gate_weights:           dict with "weight" and "e_score_correction_bias"
            routed_expert_weights:  list[dict] with "gate_proj", "up_proj", "down_proj" per expert
            shared_expert_weights:  dict with "gate_proj", "up_proj", "down_proj"
        For dense layers (layer_idx < n_dense_layers):
            ffn_weights:            dict with "gate_proj", "up_proj", "down_proj"
    """

    @staticmethod
    def check_cache_complete(cache_path: Path, layer_idx: int, is_dense: bool, experts_per_chip: int = 8) -> bool:
        """Check if block cache is complete (norms + MLA + FFN/MoE)."""
        prefix = f"layer_{layer_idx}"

        if not TtDistributedRmsNorm.check_cache_complete(cache_path, f"{prefix}.attn_norm"):
            return False
        if not ttMLA.check_cache_complete(cache_path, f"{prefix}.mla"):
            return False
        if not TtDistributedRmsNorm.check_cache_complete(cache_path, f"{prefix}.ffn_norm"):
            return False

        if is_dense:
            if not TtFfn.check_cache_complete(cache_path, f"{prefix}.ffn"):
                return False
        else:
            if not TtMoe.check_cache_complete(cache_path, layer_idx, experts_per_chip):
                return False

        return True

    @staticmethod
    def build_ttnn_cache(
        state_dict: dict,
        layer_idx: int,
        cache_path: Path,
        mesh_device: ttnn.MeshDevice,
        config: PretrainedConfig,
        model_cfg: type,
        seq_len: int = 1024,
        dispatch_buffer_capacity_factor: int = 2,
        num_links: int = 2,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        sp_axis: int = 0,
        tp_axis: int = 1,
        gate_fallback_mode: GateComputeMode = GateComputeMode.HOST_ALL,
        routed_expert_activations_dtype=ttnn.bfloat8_b,
        routed_expert_weights_dtype=ttnn.bfloat4_b,
        shared_expert_activations_dtype=ttnn.bfloat16,
        shared_expert_weights_dtype=ttnn.bfloat8_b,
        kv_only: bool = False,
    ):
        """
        Build TTNN cache for this block (norms + MLA + FFN/MoE) without device copy.

        Args:
            state_dict: Layer weights dict
            layer_idx: Layer index
            cache_path: Cache directory
            mesh_device: Mesh device reference
            config: Model config
            model_cfg: Variant static-constants class (DeepSeekV3Config | KimiK26Config)
            ... other args for sub-components
        """
        is_moe = layer_idx >= model_cfg.NUM_DENSE_LAYERS
        emb_dim = config.hidden_size

        logger.info(f"Building TTNN cache for TtPrefillBlock layer {layer_idx} ({'MoE' if is_moe else 'dense'})")

        # Build attn_norm cache
        TtDistributedRmsNorm.build_ttnn_cache(
            torch_weight=state_dict["attn_norm_weight"],
            emb_dim=emb_dim,
            mesh_device=mesh_device,
            cache_path=cache_path,
            cache_name_prefix=f"layer_{layer_idx}.attn_norm",
        )

        # Build MLA cache
        ttMLA.build_ttnn_cache(
            state_dict=state_dict.get("mla_weights", {}),
            cache_path=cache_path,
            mesh_device=mesh_device,
            config=config,
            layer_idx=layer_idx,
            seq_len=seq_len,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            kv_only=kv_only,
        )

        if kv_only:
            # The kv-only last layer has no ffn_norm / FFN / MoE.
            logger.info(f"Cache built for layer {layer_idx} (kv_only)")
            return

        # Build ffn_norm cache
        TtDistributedRmsNorm.build_ttnn_cache(
            torch_weight=state_dict["ffn_norm_weight"],
            emb_dim=emb_dim,
            mesh_device=mesh_device,
            cache_path=cache_path,
            cache_name_prefix=f"layer_{layer_idx}.ffn_norm",
        )

        # Build FFN or MoE cache
        if is_moe:
            # Use static method (no device copy!)
            mesh_config = extract_mesh_config(mesh_device)
            sp_factor = mesh_device.shape[sp_axis]
            seq_len_per_chip = seq_len // sp_factor
            experts_per_chip, _, _, _ = compute_constants(
                seq_len_per_chip,
                model_cfg.NUM_ROUTED_EXPERTS,
                model_cfg.NUM_EXPERTS_PER_TOKEN,
                mesh_device.get_num_devices(),
                mesh_config.dispatch_group_size,
                dispatch_buffer_capacity_factor,
            )

            TtMoe.build_ttnn_cache(
                gate_weights=state_dict.get("gate_weights"),
                routed_expert_weights=state_dict.get("routed_expert_weights"),
                shared_expert_weights=state_dict.get("shared_expert_weights"),
                experts_per_chip=experts_per_chip,
                emb_dim=emb_dim,
                hidden_dim=model_cfg.MOE_INTERMEDIATE_SIZE,
                mesh_device=mesh_device,
                routed_expert_weights_dtype=routed_expert_weights_dtype,
                shared_expert_weights_dtype=shared_expert_weights_dtype,
                cache_path=cache_path,
                layer_idx=layer_idx,
            )
        else:
            # Use static method (no device copy!)
            TtFfn.build_ttnn_cache(
                torch_weights=state_dict.get("ffn_weights"),
                mesh_device=mesh_device,
                cache_path=cache_path,
                cache_name_prefix=f"layer_{layer_idx}.ffn",
            )

        logger.info(f"Cache built for layer {layer_idx}")

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        config: PretrainedConfig,
        model_cfg: type,
        state_dict: dict,
        layer_idx: int,
        seq_len: int,
        dispatch_buffer_capacity_factor: int = 2,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        sp_axis: int = 0,
        tp_axis: int = 1,
        is_balanced: bool = False,
        gate_fallback_mode: GateComputeMode = GateComputeMode.HOST_ALL,
        routed_expert_activations_dtype=ttnn.bfloat8_b,
        routed_expert_weights_dtype=ttnn.bfloat4_b,
        shared_expert_activations_dtype=ttnn.bfloat16,
        shared_expert_weights_dtype=ttnn.bfloat8_b,
        weight_cache_path: Optional[Path] = None,
        is_chunked: bool = False,
        slot_num: int = 1,
        layer_num: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        kv_only: bool = False,
        routing_use_l1_small_for_semaphores: bool = False,
    ):
        super().__init__()
        self.routing_use_l1_small_for_semaphores = routing_use_l1_small_for_semaphores
        # In chunked prefill the flat KV-cache slot is cache_user_id * layer_num + cache_layer_idx, so
        # layer_num must be the model's actual layer count — there is no safe default to fall back to.
        assert not is_chunked or layer_num is not None, "chunked prefill requires layer_num (model layer count)"
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.topology = topology
        self.kv_only = kv_only
        self.is_moe = layer_idx >= model_cfg.NUM_DENSE_LAYERS

        emb_dim = config.hidden_size

        logger.info(
            f"Building TtPrefillBlock layer_idx={layer_idx} "
            f"({'MoE' if self.is_moe else 'dense'}, kv_only={kv_only})"
        )

        # --- Attention norm ---
        self.attn_norm = TtDistributedRmsNorm(
            mesh_device=mesh_device,
            emb_dim=emb_dim,
            torch_weight=state_dict.get("attn_norm_weight"),  # None if cache exists
            epsilon=config.rms_norm_eps,
            cluster_axis=tp_axis,
            num_links=num_links,
            topology=topology,
            weight_cache_path=weight_cache_path,
            cache_name_prefix=f"layer_{layer_idx}.attn_norm",
        )

        # --- MLA ---
        # In chunked prefill the MLA's seq_len sizes the gathered-KV ring buffer (= full per-user
        # cache length), while the block's seq_len is the per-chunk size used by the MoE/FFN dispatch
        # buffers. They are equal in the single-shot path (max_seq_len is None).
        self.mla = ttMLA(
            config,
            state_dict.get("mla_weights", {}),  # Empty dict if cache exists
            mesh_device,
            layer_idx=layer_idx,
            seq_len=max_seq_len if max_seq_len is not None else seq_len,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            is_balanced=is_balanced,
            weight_cache_path=weight_cache_path,
            is_chunked=is_chunked,
            slot_num=slot_num,
            layer_num=layer_num,
            kv_only=kv_only,
        )

        if kv_only:
            # Last layer: no FFN norm, no FFN/MoE. Forward returns after MLA.
            return

        # --- FFN norm ---
        self.ffn_norm = TtDistributedRmsNorm(
            mesh_device=mesh_device,
            emb_dim=emb_dim,
            torch_weight=state_dict.get("ffn_norm_weight"),  # None if cache exists
            epsilon=config.rms_norm_eps,
            cluster_axis=tp_axis,
            num_links=num_links,
            topology=topology,
            weight_cache_path=weight_cache_path,
            cache_name_prefix=f"layer_{layer_idx}.ffn_norm",
        )

        # --- FFN (MoE or dense) ---
        if self.is_moe:
            self.ffn = self._build_moe(
                mesh_device=mesh_device,
                model_cfg=model_cfg,
                state_dict=state_dict,
                seq_len=seq_len,
                sp_axis=sp_axis,
                emb_dim=emb_dim,
                num_links=num_links,
                topology=topology,
                gate_fallback_mode=gate_fallback_mode,
                routed_expert_activations_dtype=routed_expert_activations_dtype,
                routed_expert_weights_dtype=routed_expert_weights_dtype,
                shared_expert_activations_dtype=shared_expert_activations_dtype,
                shared_expert_weights_dtype=shared_expert_weights_dtype,
                weight_cache_path=weight_cache_path,
                layer_idx=layer_idx,
                dispatch_buffer_capacity_factor=dispatch_buffer_capacity_factor,
                routing_use_l1_small_for_semaphores=routing_use_l1_small_for_semaphores,
                is_balanced=is_balanced,
            )
        else:
            # emb_dim/hidden_dim default to DSv3/Kimi's 7168/18432 in TtFfn; pass the variant's real dims
            # so GLM-5.1 (hidden 6144, dense intermediate 12288) doesn't inherit the 7168 default. emb_dim
            # is always safe (== default for 7168-dim models); hidden_dim only overrides when the config
            # exposes intermediate_size (GLM does; DSv3/Kimi fall back to the TtFfn default).
            _dense_ffn_kwargs = {}
            if getattr(config, "intermediate_size", None):
                _dense_ffn_kwargs["hidden_dim"] = config.intermediate_size
            self.ffn = TtFfn(
                mesh_device=mesh_device,
                torch_weights=state_dict.get("ffn_weights"),  # None if cache exists
                emb_dim=emb_dim,
                num_links=num_links,
                topology=topology,
                weight_cache_path=weight_cache_path,
                cache_name_prefix=f"layer_{layer_idx}.ffn",
                **_dense_ffn_kwargs,
            )

    @staticmethod
    def _build_moe(
        mesh_device,
        model_cfg,
        state_dict,
        seq_len,
        sp_axis,
        emb_dim,
        num_links,
        topology,
        gate_fallback_mode,
        routed_expert_activations_dtype,
        routed_expert_weights_dtype,
        shared_expert_activations_dtype,
        shared_expert_weights_dtype,
        dispatch_buffer_capacity_factor,
        weight_cache_path=None,
        layer_idx=0,
        routing_use_l1_small_for_semaphores=False,
        is_balanced=False,
    ):
        mesh_config = extract_mesh_config(mesh_device)
        sp_factor = mesh_device.shape[sp_axis]
        seq_len_per_chip = seq_len // sp_factor

        (
            experts_per_chip,
            metadata_len,
            max_dispatch_buffer_token_size,
            max_dispatched_tokens_per_expert,
        ) = compute_constants(
            seq_len_per_chip,
            model_cfg.NUM_ROUTED_EXPERTS,
            model_cfg.NUM_EXPERTS_PER_TOKEN,
            mesh_device.get_num_devices(),
            mesh_config.dispatch_group_size,
            dispatch_buffer_capacity_factor,
        )

        return TtMoe(
            mesh_device=mesh_device,
            dispatch_group_size=mesh_config.dispatch_group_size,
            num_dispatch_groups=mesh_config.num_dispatch_groups,
            experts_per_chip=experts_per_chip,
            num_routed_experts=model_cfg.NUM_ROUTED_EXPERTS,
            num_experts_per_tok=model_cfg.NUM_EXPERTS_PER_TOKEN,
            metadata_len=metadata_len,
            max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
            max_dispatch_buffer_token_size=max_dispatch_buffer_token_size,
            seq_len_per_chip=seq_len_per_chip,
            emb_dim=emb_dim,
            hidden_dim=model_cfg.MOE_INTERMEDIATE_SIZE,
            num_links=num_links,
            topology=topology,
            routed_expert_weights=state_dict.get("routed_expert_weights"),  # None if cache exists
            shared_expert_weights=state_dict.get("shared_expert_weights"),  # None if cache exists
            routed_expert_activations_dtype=routed_expert_activations_dtype,
            routed_expert_weights_dtype=routed_expert_weights_dtype,
            shared_expert_activations_dtype=shared_expert_activations_dtype,
            shared_expert_weights_dtype=shared_expert_weights_dtype,
            gate_weights=state_dict.get("gate_weights"),  # None if cache exists
            gate_fallback_mode=gate_fallback_mode,
            n_expert_groups=model_cfg.NUM_EXPERT_GROUPS,
            n_limited_groups=model_cfg.NUM_LIMITED_GROUPS,
            route_scale=model_cfg.ROUTE_SCALE,
            weight_cache_path=weight_cache_path,
            layer_idx=layer_idx,
            overlap_shared_expert_with_dispatch=True,
            routing_use_l1_small_for_semaphores=routing_use_l1_small_for_semaphores,
            is_balanced=is_balanced,
        )

    def forward(
        self,
        x: ttnn.Tensor,
        rope_tensors: dict,
        kvpe_cache: ttnn.Tensor,
        cache_layer_idx: int = 0,
        return_kv_cache: bool = False,
        return_intermediates: bool = False,
        on_layer_complete: Optional[Callable[[int], None]] = None,
        actual_start: Optional[int] = None,
        actual_end: Optional[int] = None,
        cache_user_id: int = 0,
        return_kv_intermediates: bool = False,
        actual_isl: Optional[int] = None,
        padding_side: str = "right",
    ):
        """
        Args:
            x: [1, 1, seq_len_local, emb_dim_per_tp] TILE_LAYOUT, TP-sharded
            rope_tensors: dict with keys "cos_matrix", "sin_matrix", "trans_matrix"
            return_kv_cache: if True, also return KVPE cache from MLA
            return_intermediates: if True, forward to TtMoe so it runs its
                intermediates-gated checks (per-chip dispatch buffer overflow,
                region-offset bounds). Has no effect on dense layers.
            on_layer_complete: optional per-layer migration ack. In chunked prefill, after MLA writes
                the chunk this block zeros the pad window past actual_end, flushes, then fires this.
            actual_start: chunked-prefill absolute KV pos of this chunk's first real token (the cache
                write offset = cumulative valid-KV count before it; None for single-shot). Selects
                MLA's chunked path; requires the block to have been built with is_chunked=True.
            actual_end: absolute KV pos past this chunk's last real token (the migration pad-zero boundary).
            cache_user_id: chunked-prefill cache slot index (user-major batch).
            return_kv_intermediates: if True, MLA surfaces its 4 KV stages (tt_kv, tt_kv_nope,
                tt_kv_rope, tt_kvpe) and this returns (output_tensor, kv_intermediates_dict) — also
                carrying post_mla_residual + post_attn_norm — instead of (output_tensor, kv_cache).
            actual_isl: actual (unpadded) count of real tokens; threaded to the MoE FFN for
                padding-aware routing.
            padding_side: "right" or "left"; threaded to the MoE FFN for padding-aware routing.

        Returns:
            (output_tensor, kv_cache) where kv_cache is a host tensor or None, or
            (output_tensor, kv_intermediates_dict) when return_kv_intermediates=True.
        """
        # --- Attention ---
        attn_norm_out = self.attn_norm(x)
        seq_len_local = attn_norm_out.shape[2]
        mla_out = self.mla.forward(
            attn_norm_out,
            rope_tensors,
            kvpe_cache,
            cache_layer_idx=cache_layer_idx,
            actual_start=actual_start,
            cache_user_id=cache_user_id,
            return_kv_intermediates=return_kv_intermediates,
        )
        kv_intermediates = None
        if return_kv_intermediates:
            mla_out, kv_intermediates = mla_out
        ttnn.deallocate(attn_norm_out)

        # Chunked-prefill migration handoff. MLA's update_padded_kv_cache wrote this chunk as full
        # 32-row tiles, leaving stale data between the last real token (actual_end) and the next
        # 128-boundary; zero that pad window so the decode side reads clean zeros. The synchronize
        # flushes the (async) zero to device before on_layer_complete hands this layer's KV to the
        # migration worker, which reads the cache over NoC out-of-band from the ttnn command queue —
        # without the flush it could copy pre-zero data. layer_idx is GLOBAL (the scheduler orders acks
        # across pipeline ranks); cache_layer_idx is the LOCAL per-rank cache slot.
        if on_layer_complete is not None:
            assert actual_end is not None, "actual_end required when on_layer_complete is set"
            ttnn.experimental.deepseek_prefill.zero_padded_kv_cache(
                kvpe_cache,
                cache_user_id,
                cache_layer_idx,
                self.mla.layer_num,
                actual_end,
                seq_len_local * self.mla.sp_factor,
                self.mla.sp_axis,
            )
            ttnn.synchronize_device(self.mesh_device)
            on_layer_complete(self.mla.layer_idx)

        if self.kv_only:
            # KV cache filled (by MLA), migration callback fired. The block
            # output is unused (no FFN, no further layers). Return (None, None)
            # so the transformer can short-circuit.
            return None, None

        x = ttnn.add(x, mla_out)
        ttnn.deallocate(mla_out)
        if return_kv_intermediates:
            # post-MLA residual (x + mla_out), TP-sharded on hidden.
            kv_intermediates["post_mla_residual"] = ttnn.clone(x)

        # --- FFN ---
        ffn_norm_out = self.ffn_norm(x)
        if return_kv_intermediates:
            # post_attention_layernorm output (the FFN norm), TP-sharded on hidden.
            kv_intermediates["post_attn_norm"] = ttnn.clone(ffn_norm_out)

        if self.is_moe:
            ffn_out = self._moe_path(
                ffn_norm_out,
                return_intermediates=return_intermediates,
                actual_isl=actual_isl,
                padding_side=padding_side,
            )
        else:
            ffn_out = self._dense_ffn_path(ffn_norm_out)

        ttnn.deallocate(ffn_norm_out)
        x = ttnn.add(x, ffn_out)
        ttnn.deallocate(ffn_out)

        if return_kv_intermediates:
            return x, kv_intermediates

        kv_cache = ttMLA.kv_cache_to_host(kvpe_cache, self.mesh_device) if return_kv_cache else None
        return x, kv_cache

    def _moe_path(
        self,
        ffn_norm_out: ttnn.Tensor,
        return_intermediates: bool = False,
        actual_isl: Optional[int] = None,
        padding_side: str = "right",
    ) -> ttnn.Tensor:
        """MoE FFN path: 4D TILE → 3D ROW_MAJOR → MoE → 3D TILE → 4D TILE."""
        moe_input = ttnn.squeeze(ffn_norm_out, dim=0)

        moe_out, _ = self.ffn(
            moe_input,
            return_intermediates=return_intermediates,
            actual_isl=actual_isl,
            padding_side=padding_side,
        )

        moe_out = ttnn.unsqueeze(moe_out, dim=0)
        return moe_out

    def _dense_ffn_path(self, ffn_norm_out: ttnn.Tensor) -> ttnn.Tensor:
        """Dense FFN path: all_gather to get full emb_dim → TtFfn (does reduce_scatter internally)."""
        if self.mesh_device.shape[1] > 1:
            gathered = ttnn.all_gather(
                ffn_norm_out,
                dim=-1,
                cluster_axis=1,
                num_links=self.num_links,
                topology=self.topology,
            )
            ffn_out = self.ffn(gathered)
            ttnn.deallocate(gathered)
        else:
            ffn_out = self.ffn(ffn_norm_out)
        return ffn_out
