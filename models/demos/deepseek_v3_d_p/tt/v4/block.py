# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""One DeepSeek-V4 decoder block for prefill."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.reference.mhc.mhc_reference import MHCConfig
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import TtMHCWrap
from models.demos.deepseek_v3_d_p.tt.mla.heavily_compressed_attention import TtHCA
from models.demos.deepseek_v3_d_p.tt.mla.sliding_window_attention import TtSWA
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TopologyArg, TtPrefillBlock

_ATTENTION = {
    "sliding_attention": TtSWA,
    "heavily_compressed_attention": TtHCA,
}

_SUBLAYER_DTYPE = ttnn.bfloat16


class TtV4Block(LightweightModule):
    """A norm, the layer's attention, a norm, the layer's MoE, and two hyper-connection residuals.

    Two properties shape the class:

      * attention is per layer. ``config.layer_types[layer_idx]`` names it, and V4's attention modules
        own their KV state and build their own rope, so the block supplies no cache, no rope tensors
        and no indexer, and holds no cache-migration handoff.
      * the residual is hyper-connections, so the block's own input and output carry ``hc_mult``
        parallel streams in fp32. Every sublayer between them still sees one hidden state.

    Shapes, taking a single device first: input and output are ``[1, 1, seq, hc_mult * hidden]`` fp32,
    and each sublayer sees ``[1, 1, seq, hidden]``. On a mesh the sequence is sharded on the SP axis
    and hidden on the TP axis, giving ``[1, 1, seq/sp, hc_mult * hidden/tp]`` and
    ``[1, 1, seq/sp, hidden/tp]``. Hidden is never gathered across the block. Within a chip the columns
    hold its own hidden slice of every stream, which is what ``mhc_expand`` and this block's own
    output already produce -- chaining layers needs no repacking. Only a host upload has to permute
    into that order, since a mesh mapper splits the last dim globally (``_pack_streams`` in the test).

    The MoE comes from ``TtPrefillBlock._build_moe``: the same TtMoe off the same model config,
    reading ``state_dict["hash_table"]`` for V4's hash-routed layers.

    ``weight_cache_path`` lets the norms and the MoE write their caches, but there is no
    check-complete / build-cache pre-pass yet -- that belongs with the transformer that drives a
    whole model's worth of these.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        config: PretrainedConfig,
        model_cfg: type,
        state_dict: dict,
        layer_idx: int,
        seq_len: int,
        attn_reference,
        mhc_weights: dict,
        num_links: int = 1,
        topology: TopologyArg = ttnn.Topology.Linear,
        sp_axis: int = 0,
        tp_axis: int = 1,
        max_seq_len: Optional[int] = None,
        dispatch_buffer_capacity_factor: int = 2,
        gate_fallback_mode: GateComputeMode = GateComputeMode.HOST_ALL,
        routed_expert_activations_dtype=ttnn.bfloat8_b,
        routed_expert_weights_dtype=DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE,
        shared_expert_activations_dtype=ttnn.bfloat16,
        shared_expert_weights_dtype=ttnn.bfloat8_b,
        weight_cache_path: Optional[Path] = None,
        is_balanced: bool = False,
        routing_use_l1_small_for_semaphores: bool = False,
        overlap_shared_expert_with_dispatch: bool = True,
    ):
        """``seq_len`` is the padded per-chunk length; ``max_seq_len`` the full per-user length the
        attention state must span, defaulting to one chunk. ``attn_reference`` is the torch attention
        module the weights are read off, and ``mhc_weights`` the ``(fn, base, scale)`` triple per site.
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.num_links = num_links
        # A (SP-axis, TP-axis) tuple configures each mesh axis separately; a scalar applies to both.
        # A torus config wraps only the SP axis, so TP-axis collectives must stay Linear: Ring on an
        # unwrapped axis waits forever on a wrap link with no fabric edge behind it.
        assert (
            not isinstance(topology, tuple) or len(topology) == 2
        ), f"per-axis topology must be a 2-tuple (sp_axis, tp_axis), got {topology!r}"
        tp_topology = topology[1] if isinstance(topology, tuple) else topology
        self.layer_idx = layer_idx

        attn_kind = config.layer_types[layer_idx]
        assert attn_kind in _ATTENTION, f"layer {layer_idx} wants attention {attn_kind!r}, have {tuple(_ATTENTION)}"
        logger.info(f"Building TtV4Block layer_idx={layer_idx} ({attn_kind}, {config.mlp_layer_types[layer_idx]})")

        self.attn_norm = TtDistributedRmsNorm(
            mesh_device=mesh_device,
            emb_dim=config.hidden_size,
            torch_weight=state_dict.get("attn_norm_weight"),
            epsilon=config.rms_norm_eps,
            cluster_axis=tp_axis,
            num_links=num_links,
            topology=tp_topology,
            weight_cache_path=weight_cache_path,
            cache_name_prefix=f"layer_{layer_idx}.attn_norm",
        )
        self.ffn_norm = TtDistributedRmsNorm(
            mesh_device=mesh_device,
            emb_dim=config.hidden_size,
            torch_weight=state_dict.get("ffn_norm_weight"),
            epsilon=config.rms_norm_eps,
            cluster_axis=tp_axis,
            num_links=num_links,
            topology=tp_topology,
            weight_cache_path=weight_cache_path,
            cache_name_prefix=f"layer_{layer_idx}.ffn_norm",
        )

        # V4 attention keeps ONE topology for every collective, its SP-axis gather included, so it
        # takes the TP element. Its weights come off the torch module, not a state dict, so there is
        # nothing for a weight cache to hold for this half of the block.
        self.attn = _ATTENTION[attn_kind].from_reference(
            mesh_device, attn_reference, config, sp_axis=sp_axis, tp_axis=tp_axis, topology=tp_topology
        )
        # Allocated once: the state's shape is the same for every chunk, only its contents advance.
        self.attn_state = self.attn.alloc_state(max_seq_len or seq_len, chunk_tokens=seq_len)

        self.ffn = TtPrefillBlock._build_moe(
            mesh_device=mesh_device,
            model_cfg=model_cfg,
            config=config,
            state_dict=state_dict,
            seq_len=seq_len,
            sp_axis=sp_axis,
            emb_dim=config.hidden_size,
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
            overlap_shared_expert_with_dispatch=overlap_shared_expert_with_dispatch,
        )

        assert mhc_weights is not None and set(mhc_weights) >= {
            "attn",
            "ffn",
        }, f"both sites need an (fn, base, scale) triple; got keys {sorted(mhc_weights or ())}"
        mhc_cfg = MHCConfig(
            dim=config.hidden_size,
            n=config.hc_mult,
            sinkhorn_iters=config.hc_sinkhorn_iters,
            eps=config.hc_eps,
            norm_eps=config.rms_norm_eps,
        )
        self.attn_res = TtMHCWrap(
            mesh_device, mhc_cfg, *mhc_weights["attn"], tp_axis=tp_axis, num_links=num_links, topology=tp_topology
        )
        self.ffn_res = TtMHCWrap(
            mesh_device, mhc_cfg, *mhc_weights["ffn"], tp_axis=tp_axis, num_links=num_links, topology=tp_topology
        )

    def forward(
        self,
        x: ttnn.Tensor,
        actual_isl: Optional[int] = None,
        input_ids: Optional[torch.Tensor] = None,
        actual_start: Optional[int] = None,
        padding_side: str = "right",
    ) -> ttnn.Tensor:
        """One chunk of packed residual streams in, the same shape out.

        ``actual_isl`` is the chunk's real pre-pad length, ``actual_start`` where it begins in the
        sequence, and ``input_ids`` the chunk's token ids, which a hash-routed layer indexes its
        expert table with and a top-k layer ignores.

        Each norm sits inside its site's sublayer, not before it: the hyper-connection collapses its
        streams first, and the norm belongs on what comes out of that collapse.
        """

        def _attn(h):
            normed = self.attn_norm(h)
            out = self.attn(normed, seq_len_actual=actual_isl, state=self.attn_state)
            ttnn.deallocate(normed)
            return out

        def _ffn(h):
            normed = self.ffn_norm(h)
            # TtMoe works in 3D; the block's own tensors are 4D.
            out, _ = self.ffn(
                ttnn.squeeze(normed, dim=0),
                actual_isl=actual_isl,
                padding_side=padding_side,
                actual_start=actual_start or 0,
                input_ids=input_ids,
            )
            ttnn.deallocate(normed)
            return ttnn.unsqueeze(out, dim=0)

        x = self.attn_res(x, _attn, sublayer_dtype=_SUBLAYER_DTYPE)
        return self.ffn_res(x, _ffn, sublayer_dtype=_SUBLAYER_DTYPE)
