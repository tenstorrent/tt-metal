# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One Kimi-K3 decoder layer.

Not a `TtPrefillBlock` subclass, for three reasons that are structural rather than stylistic:
`TtPrefillBlock.__init__` builds `ttMLA` unconditionally, before its own `kv_only` early return, so
a KDA layer cannot decline it; its `forward` reads `self.mla.*` in six places; and it returns the
residual, which under AttnRes belongs to the walk. Overriding around all three is a parallel stack
wearing inheritance, so this is the parallel stack without it — the same arrangement
`models/demos/gpt_oss_d_p/tt/layer.py` and `models/demos/minimax_m3/tt/layer.py` already use.

What it does NOT re-implement: the MoE is built by `TtPrefillBlock._build_moe`, already a fully
parameterised staticmethod, and the KV pad-zero and migration ack come from `tt/kv_ack.py`, lifted
verbatim out of `TtPrefillBlock.forward` for exactly this. The norms, the FFN and the attention
modules are the shared ones. What is K3's is the shape of the layer: an injected attention that
answers `writes_kv`, and a residual it talks to rather than carries.
"""

from __future__ import annotations

from typing import Optional

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.kimi_k3.attention import K3AttnContext
from models.demos.deepseek_v3_d_p.tt.kv_ack import zero_pad_and_ack
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.moe.tt_shared_expert import ACTIVATION_SILU
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
from models.demos.deepseek_v3_d_p.tt.tt_ffn import TtFfn
from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TtPrefillBlock


class TtKimiK3Block(LightweightModule):
    """A layer that knows its own attention only through `writes_kv`.

    `attention` is built by the caller (`build_attention` in `attention.py`) because whether this
    layer is MLA or KDA is a property of the *schedule*, not of the block, and the schedule lives one
    level up. That is the whole of the hybrid handling: no `is_kda` anywhere below this line.
    """

    def __init__(
        self,
        mesh_device,
        config,
        model_cfg: type,
        state_dict: dict,
        layer_idx: int,
        local_idx: int,
        attention,
        seq_len: int,
        *,
        num_links: int = 1,
        topology=ttnn.Topology.Linear,
        sp_axis: int = 0,
        tp_axis: int = 1,
        is_balanced: bool = False,
        gate_fallback_mode=None,  # defaults to K3's own mode below
        weight_cache_path=None,
        kv_only: bool = False,
        dispatch_buffer_capacity_factor: int = 2,
        routed_expert_activations_dtype=ttnn.bfloat8_b,
        routed_expert_weights_dtype=ttnn.bfloat4_b,
        shared_expert_activations_dtype=ttnn.bfloat16,
        shared_expert_weights_dtype=ttnn.bfloat8_b,
        routing_use_l1_small_for_semaphores: bool = False,
        overlap_shared_expert_with_dispatch: bool = True,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.layer_idx = layer_idx
        self.local_idx = local_idx
        self.attention = attention
        self.kv_only = kv_only
        self.num_links = num_links
        self.is_moe = layer_idx >= model_cfg.NUM_DENSE_LAYERS

        emb_dim = config.hidden_size
        # Norms and the dense FFN reduce on the tensor axis only; MLA and MoE want both elements.
        # `ttnn.all_gather` takes ONE topology, so handing it the per-axis pair is a TypeError deep
        # inside the binding rather than anything that reads as a wiring mistake -- which is why
        # `TtPrefillBlock` stores the TP element under `self.topology` and this does the same.
        tp_topology = topology[tp_axis] if isinstance(topology, (tuple, list)) else topology
        self.topology = tp_topology

        self.attn_norm = TtDistributedRmsNorm(
            mesh_device=mesh_device,
            emb_dim=emb_dim,
            torch_weight=state_dict.get("attn_norm_weight"),
            epsilon=config.rms_norm_eps,
            cluster_axis=tp_axis,
            num_links=num_links,
            topology=tp_topology,
            weight_cache_path=weight_cache_path,
            cache_name_prefix=f"layer_{layer_idx}.attn_norm",
        )

        if kv_only:
            # Last layer of a chunk whose output nobody reads: attention and the KV write, nothing
            # after. The schedule guarantees this layer is MLA (a KDA layer writes no KV, so a
            # kv_only KDA layer would be a full recurrence thrown away) — see
            # KimiK3LayerSchedule.validate_kv_only_last_layer.
            return

        self.ffn_norm = TtDistributedRmsNorm(
            mesh_device=mesh_device,
            emb_dim=emb_dim,
            torch_weight=state_dict.get("ffn_norm_weight"),
            epsilon=config.rms_norm_eps,
            cluster_axis=tp_axis,
            num_links=num_links,
            topology=tp_topology,
            weight_cache_path=weight_cache_path,
            cache_name_prefix=f"layer_{layer_idx}.ffn_norm",
        )

        if self.is_moe:
            # Kimi-K3 runs the device FP32 gate: one expert group, so there is no grouped-topk
            # fallback to prefer, and `KimiK3Adapter.default_gate_mode` names it. Defaulted here
            # rather than left None, which reaches TtMoEGatePrefill as a mode it cannot read.
            gate_mode = gate_fallback_mode if gate_fallback_mode is not None else GateComputeMode.DEVICE_FP32
            self.ffn = TtPrefillBlock._build_moe(
                mesh_device=mesh_device,
                model_cfg=model_cfg,
                config=config,
                state_dict=state_dict,
                seq_len=seq_len,
                sp_axis=sp_axis,
                emb_dim=emb_dim,
                num_links=num_links,
                topology=topology,
                gate_fallback_mode=gate_mode,
                routed_expert_activations_dtype=routed_expert_activations_dtype,
                routed_expert_weights_dtype=routed_expert_weights_dtype,
                shared_expert_activations_dtype=shared_expert_activations_dtype,
                shared_expert_weights_dtype=shared_expert_weights_dtype,
                dispatch_buffer_capacity_factor=dispatch_buffer_capacity_factor,
                weight_cache_path=weight_cache_path,
                layer_idx=layer_idx,
                routing_use_l1_small_for_semaphores=routing_use_l1_small_for_semaphores,
                is_balanced=is_balanced,
                overlap_shared_expert_with_dispatch=overlap_shared_expert_with_dispatch,
            )
        else:
            # Layer 0 only, under first_k_dense_replace=1. K3 runs SiTU here, not SiLU (#53625).
            self.ffn = TtFfn(
                mesh_device=mesh_device,
                emb_dim=emb_dim,
                torch_weights=state_dict.get("ffn_weights"),
                num_links=num_links,
                topology=tp_topology,
                activations_dtype=shared_expert_activations_dtype,
                weights_dtype=shared_expert_weights_dtype,
                weight_cache_path=weight_cache_path,
                cache_name_prefix=f"layer_{layer_idx}.ffn",
                activation=getattr(model_cfg, "DENSE_FFN_ACTIVATION", ACTIVATION_SILU),
                situ_beta=getattr(model_cfg, "ACTIVATION_SITU_BETA", None),
                situ_linear_beta=getattr(model_cfg, "ACTIVATION_SITU_LINEAR_BETA", None),
                hidden_dim=config.intermediate_size,
            )

    def set_trace_controller(self, controller):
        """Attach (or clear with None) a SubDeviceTraceController, so a ttnn trace captured over
        forward() is split at this block's shared-expert/dispatch sub-device boundaries.

        `forward` already reads `self._trace_controller` for the migration-ack site; without this
        setter it stayed unset and `TtPrefillRuntime._prepare_trace` failed outright on the model.

        No indexer guard here, unlike `TtPrefillBlock`: Kimi-K3's attention is KDA or dense MLA, and
        neither carries an indexer. See the note in kimi_k3/attention.py.
        """
        self._trace_controller = controller
        ffn = getattr(self, "ffn", None)
        if ffn is not None and hasattr(ffn, "set_trace_controller"):
            ffn.set_trace_controller(controller)

    def release_sub_device_managers(self):
        """Drop this block's MoE overlap sub-device manager before mesh close (no-op for a dense FFN)."""
        ffn = getattr(self, "ffn", None)
        if ffn is not None and hasattr(ffn, "release_sub_device_manager"):
            ffn.release_sub_device_manager()

    def forward(
        self,
        residual,
        ctx: K3AttnContext,
        *,
        d2h_service=None,
        metadata_msg=None,
        on_layer_complete=None,
        actual_end: Optional[int] = None,
        actual_isl: Optional[int] = None,
        padding_side: str = "right",
    ) -> None:
        """One layer, folded into `residual`. Returns nothing — the stream holds the state.

        Order matches `TtPrefillBlock.forward` where it matters: the KV pad-zero and ack fire
        immediately after attention and *before* the residual write, so migration is told the layer
        is done as early as it was before.
        """
        hidden = residual.open(self.local_idx)
        normed = self.attn_norm(hidden)
        seq_len_local = normed.shape[2]

        attn_out = self.attention.forward(normed, ctx)
        ttnn.deallocate(normed)

        if self.attention.writes_kv:
            zero_pad_and_ack(
                kvpe_cache=ctx.kvpe_cache,
                mesh_device=self.mesh_device,
                cache_layer_idx=ctx.cache_layer_idx,
                cache_user_id=ctx.cache_user_id,
                layer_num=self.attention.mla.layer_num,
                sp_factor=self.attention.sp_factor,
                sp_axis=self.attention.sp_axis,
                global_layer_idx=self.layer_idx,
                seq_len_local=seq_len_local,
                actual_end=actual_end,
                metadata=ctx.metadata,
                d2h_service=d2h_service,
                metadata_msg=metadata_msg,
                on_layer_complete=on_layer_complete,
                trace_controller=getattr(self, "_trace_controller", None),
            )

        if self.kv_only:
            # No FFN and no output. The walk's remaining sites simply go unconsumed; the caller
            # discards the stream rather than taking its model-level read.
            #
            # `attn_out` is None here whenever the attention is MLA: a kv_only ttMLA writes its KV
            # slab and returns nothing, because nothing downstream reads its output. That is the
            # normal case — the schedule guarantees a kv_only layer is MLA, since a kv_only KDA layer
            # would run a full recurrence and throw it away.
            if attn_out is not None:
                ttnn.deallocate(attn_out)
            residual.release(hidden)
            return

        residual.write(attn_out)
        residual.release(hidden)

        pre_ffn = residual.read()
        ffn_norm_out = self.ffn_norm(pre_ffn)
        ffn_out = self._ffn_path(ffn_norm_out, actual_isl=actual_isl, padding_side=padding_side, ctx=ctx)
        ttnn.deallocate(ffn_norm_out)
        residual.write(ffn_out)
        residual.release(pre_ffn)

    def _ffn_path(self, ffn_norm_out, *, actual_isl, padding_side, ctx) -> ttnn.Tensor:
        if self.is_moe:
            # 4D TILE -> 3D -> MoE -> 3D -> 4D, matching TtPrefillBlock._moe_path exactly. `TtMoe`
            # returns `(out, intermediates)`, and the squeezed input is NOT freed here: upstream
            # does not free it either, and it is a view the op may still reference.
            moe_input = ttnn.squeeze(ffn_norm_out, dim=0)
            moe_out, _ = self.ffn(
                moe_input,
                return_intermediates=False,
                actual_isl=actual_isl,
                padding_side=padding_side,
                actual_start=ctx.actual_start,
                metadata=ctx.metadata,
            )
            return ttnn.unsqueeze(moe_out, dim=0)

        # Dense: gather to the full hidden dim, then TtFfn reduce-scatters internally — the same
        # collective pair the KDA attention above already uses.
        if self.mesh_device.shape[1] > 1:
            gathered = ttnn.all_gather(
                ffn_norm_out, dim=-1, cluster_axis=1, num_links=self.num_links, topology=self.topology
            )
            out = self.ffn(gathered)
            ttnn.deallocate(gathered)
            return out
        return self.ffn(ffn_norm_out)
