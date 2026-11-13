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
from models.demos.deepseek_v3_d_p.tt.mla.compressed_sparse_attention import TtCSA
from models.demos.deepseek_v3_d_p.tt.mla.heavily_compressed_attention import TtHCA
from models.demos.deepseek_v3_d_p.tt.mla.sliding_window_attention import TtSWA
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe import TtMoe
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode, gate_bias_centered
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TopologyArg, TtPrefillBlock
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker

_ATTENTION = {
    "sliding_attention": TtSWA,
    "heavily_compressed_attention": TtHCA,
    "compressed_sparse_attention": TtCSA,
}

# The only gate modes that read the hash table; the rest route by top-k and ignore it, which on a
# hash layer means picking other experts entirely.
_HASH_GATE_MODES = (GateComputeMode.HASH_HOST, GateComputeMode.HASH_DEVICE)

_SUBLAYER_DTYPE = ttnn.bfloat16

# V4's routed experts are MXFP4 in the checkpoint, scaled per 32 inputs; transposed tiles put bfloat4_b's shared
# exponents along the same axis (see TtRoutedExpert). Shared by the constructor and ``moe_cache_complete``, which
# must ask for exactly what the constructor will request.
DEFAULT_ROUTED_EXPERT_WEIGHTS_TILE_TRANSPOSED = True


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
    into that order, since a mesh mapper splits the last dim globally (``pack_streams`` in tests/v4/helpers.py).

    The MoE comes from ``TtPrefillBlock._build_moe``: the same TtMoe off the same model config,
    reading ``state_dict["hash_table"]`` for V4's hash-routed layers.

    ``weight_cache_path`` lets the norms and the MoE write their caches, but there is no
    check-complete / build-cache pre-pass yet -- that belongs with the transformer that drives a
    whole model's worth of these.
    """

    @staticmethod
    def _mhc_config(config) -> MHCConfig:
        return MHCConfig(
            dim=config.hidden_size,
            n=config.hc_mult,
            sinkhorn_iters=config.hc_sinkhorn_iters,
            eps=config.hc_eps,
            norm_eps=config.rms_norm_eps,
        )

    @staticmethod
    def check_cache_complete(
        cache_path: Optional[Path],
        layer_idx: int,
        config: PretrainedConfig,
        model_cfg: type,
        experts_per_chip: int,
        *,
        routed_expert_weights_dtype=DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE,
        routed_expert_weights_tile_transposed: bool = DEFAULT_ROUTED_EXPERT_WEIGHTS_TILE_TRANSPOSED,
    ) -> bool:
        """True iff everything this layer caches is there: both norms, the attention (with its compressor and, for
        CSA, its indexer), both mHC sites and the MoE (gate, routed and shared experts).

        The dtype and layout arguments are the ones the constructor will request: they are part of the file names,
        and a cache that does not match them must not count as complete, or a build from ``None`` would write the
        empty placeholder out as the weights. Takes a fresh directory listing, which is all the checkers read."""
        if cache_path is None:
            return False
        cache_path = Path(cache_path)
        init_checker(cache_path)
        prefix = f"layer_{layer_idx}"
        attn_kind = config.layer_types[layer_idx]
        attn_cls = _ATTENTION[attn_kind]
        if attn_kind == "compressed_sparse_attention":
            attn_complete = attn_cls.check_cache_complete(cache_path, f"{prefix}.attn", layer_idx)
        else:
            attn_complete = attn_cls.check_cache_complete(cache_path, f"{prefix}.attn")
        return (
            TtDistributedRmsNorm.check_cache_complete(cache_path, f"{prefix}.attn_norm")
            and attn_complete
            and TtDistributedRmsNorm.check_cache_complete(cache_path, f"{prefix}.ffn_norm")
            and TtMHCWrap.check_cache_complete(cache_path, f"{prefix}.mhc_attn")
            and TtMHCWrap.check_cache_complete(cache_path, f"{prefix}.mhc_ffn")
            and TtMoe.check_cache_complete(
                cache_path,
                layer_idx,
                experts_per_chip,
                routed_expert_weights_dtype=routed_expert_weights_dtype,
                gate_bias_centered=gate_bias_centered(model_cfg),
                routed_expert_weights_tile_transposed=routed_expert_weights_tile_transposed,
            )
        )

    @staticmethod
    def build_ttnn_cache(
        state_dict: dict,
        attn_reference,
        mhc_weights: dict,
        layer_idx: int,
        cache_path: Path,
        mesh_device: ttnn.MeshDevice,
        config: PretrainedConfig,
        model_cfg: type,
        *,
        sp_axis: int = 0,
        tp_axis: int = 1,
        routed_expert_weights_dtype=DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE,
        shared_expert_weights_dtype=ttnn.bfloat8_b,
        routed_expert_weights_tile_transposed: bool = DEFAULT_ROUTED_EXPERT_WEIGHTS_TILE_TRANSPOSED,
    ) -> None:
        """Write this layer's whole weight cache without putting any of it on the device (the V4 counterpart of
        ``TtPrefillBlock.build_ttnn_cache``): both norms, the attention, both mHC sites and the MoE.

        The dtype and layout arguments must be the ones the layer will be built with."""
        cache_path = Path(cache_path)
        prefix = f"layer_{layer_idx}"
        attn_kind = config.layer_types[layer_idx]
        emb_dim = config.hidden_size
        logger.info(f"Building TTNN cache for TtV4Block layer {layer_idx} ({attn_kind})")

        TtDistributedRmsNorm.build_ttnn_cache(
            torch_weight=state_dict["attn_norm_weight"],
            emb_dim=emb_dim,
            mesh_device=mesh_device,
            cache_path=cache_path,
            cache_name_prefix=f"{prefix}.attn_norm",
        )
        attn_extra = {"layer_idx": layer_idx} if attn_kind == "compressed_sparse_attention" else {}
        _ATTENTION[attn_kind].build_ttnn_cache(
            attn_reference,
            config,
            mesh_device,
            cache_path,
            f"{prefix}.attn",
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            **attn_extra,
        )
        TtDistributedRmsNorm.build_ttnn_cache(
            torch_weight=state_dict["ffn_norm_weight"],
            emb_dim=emb_dim,
            mesh_device=mesh_device,
            cache_path=cache_path,
            cache_name_prefix=f"{prefix}.ffn_norm",
        )
        mhc_cfg = TtV4Block._mhc_config(config)
        for site in ("attn", "ffn"):
            TtMHCWrap.build_ttnn_cache(
                *mhc_weights[site], mhc_cfg, mesh_device, cache_path, f"{prefix}.mhc_{site}", tp_axis=tp_axis
            )
        n_devices = mesh_device.get_num_devices()
        assert model_cfg.NUM_ROUTED_EXPERTS % n_devices == 0, (model_cfg.NUM_ROUTED_EXPERTS, n_devices)
        TtMoe.build_ttnn_cache(
            gate_weights=state_dict.get("gate_weights"),
            routed_expert_weights=state_dict.get("routed_expert_weights"),
            shared_expert_weights=state_dict.get("shared_expert_weights"),
            experts_per_chip=model_cfg.NUM_ROUTED_EXPERTS // n_devices,
            emb_dim=emb_dim,
            hidden_dim=model_cfg.MOE_INTERMEDIATE_SIZE,
            mesh_device=mesh_device,
            routed_expert_weights_dtype=routed_expert_weights_dtype,
            shared_expert_weights_dtype=shared_expert_weights_dtype,
            cache_path=cache_path,
            layer_idx=layer_idx,
            # Must match TtPrefillBlock._build_moe's reads: a shared expert cached at the wrong intermediate size
            # regenerates as a plausible-but-wrong cache rather than an error.
            shared_hidden_dim=getattr(model_cfg, "SHARED_EXPERT_INTERMEDIATE_SIZE", None),
            routed_emb_dim=getattr(model_cfg, "ROUTED_EXPERT_HIDDEN_SIZE", None),
            gate_bias_centered=gate_bias_centered(model_cfg),
            routed_expert_weights_tile_transposed=routed_expert_weights_tile_transposed,
            gate_router_logits_dtype=getattr(model_cfg, "GATE_ROUTER_LOGITS_DTYPE", None),
        )

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
        gate_fallback_mode: Optional[GateComputeMode] = None,
        routed_expert_activations_dtype=ttnn.bfloat8_b,
        routed_expert_weights_dtype=DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE,
        shared_expert_activations_dtype=ttnn.bfloat16,
        shared_expert_weights_dtype=ttnn.bfloat8_b,
        weight_cache_path: Optional[Path] = None,
        is_balanced: bool = False,
        routing_use_l1_small_for_semaphores: bool = False,
        overlap_shared_expert_with_dispatch: bool = True,
        slot_num: int = 1,
        routed_expert_weights_tile_transposed: bool = DEFAULT_ROUTED_EXPERT_WEIGHTS_TILE_TRANSPOSED,
    ):
        """``seq_len`` is the padded per-chunk length; ``max_seq_len`` the full per-user length the
        attention state must span, defaulting to one chunk. ``attn_reference`` is the torch attention
        module the weights are read off, and ``mhc_weights`` the ``(fn, base, scale)`` triple per site.
        ``slot_num`` is how many users' attention state the block holds at once.
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.num_links = num_links
        # A (SP-axis, TP-axis) tuple configures each mesh axis separately; a scalar applies to both.
        assert (
            not isinstance(topology, tuple) or len(topology) == 2
        ), f"per-axis topology must be a 2-tuple (sp_axis, tp_axis), got {topology!r}"
        tp_topology = topology[1] if isinstance(topology, tuple) else topology
        self.layer_idx = layer_idx

        attn_kind = config.layer_types[layer_idx]
        assert attn_kind in _ATTENTION, f"layer {layer_idx} wants attention {attn_kind!r}, have {tuple(_ATTENTION)}"

        mlp_kind = config.mlp_layer_types[layer_idx]
        hash_layer = mlp_kind == "hash_moe"
        if gate_fallback_mode is None:
            gate_fallback_mode = GateComputeMode.HASH_DEVICE if hash_layer else GateComputeMode.DEVICE_FP32
        hash_mode = gate_fallback_mode in _HASH_GATE_MODES
        assert hash_layer == hash_mode, f"layer {layer_idx} is {mlp_kind}, which {gate_fallback_mode} cannot route"

        logger.info(f"Building TtV4Block layer_idx={layer_idx} ({attn_kind}, {mlp_kind})")

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

        assert slot_num >= 1, f"slot_num must be positive, got {slot_num}"
        is_csa = attn_kind == "compressed_sparse_attention"
        # The CSA indexer keeps per-slot overlap state and a user-major key cache, so it is told the
        # slot count; SWA and HCA state is one tensor set per slot, held below.
        attn_kwargs = {"slot_num": slot_num} if is_csa else {}
        # The full per-axis topology: the attention's SP-axis gather needs the SP element and its
        # other collectives the TP one. Its weights come off the torch module, ``attn_reference``, or, with
        # none, off the weight cache (which then has to be complete: a missing file raises).
        if weight_cache_path is not None:
            attn_kwargs.update(weight_cache_path=weight_cache_path, cache_name_prefix=f"layer_{layer_idx}.attn")
            if is_csa:
                attn_kwargs["layer_idx"] = layer_idx  # names the indexer's files, which are keyed by GLOBAL layer
        if attn_reference is None:
            assert (
                weight_cache_path is not None
            ), "no attn_reference and no weight cache: nothing to build the attention from"
            self.attn = _ATTENTION[attn_kind].from_cache(
                mesh_device, config, sp_axis=sp_axis, tp_axis=tp_axis, topology=topology, **attn_kwargs
            )
        else:
            self.attn = _ATTENTION[attn_kind].from_reference(
                mesh_device, attn_reference, config, sp_axis=sp_axis, tp_axis=tp_axis, topology=topology, **attn_kwargs
            )
        # Tables once, then one state per user slot. Each state's shape is the same for every chunk,
        # only its contents advance.
        # TODO once a V4 transformer owns this state instead of the block, forward() needs the chunk
        # offset back the way TtPrefillBlock takes actual_start: a caller-owned cache does not
        # advance itself, so the position has to come in with the chunk.
        self.attn.build_tables(max_seq_len or seq_len, chunk_tokens=seq_len)
        self.attn_states = [self.attn.new_state(cache_user_id=slot) for slot in range(slot_num)]

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
            # V4's routed experts are MXFP4 in the checkpoint, scaled per 32 inputs. Transposed tiles
            # put bfloat4_b's shared exponents along the same axis (see TtRoutedExpert).
            routed_expert_weights_tile_transposed=routed_expert_weights_tile_transposed,
        )

        assert mhc_weights is not None or weight_cache_path is not None, "no mhc_weights and no weight cache"
        assert mhc_weights is None or set(mhc_weights) >= {
            "attn",
            "ffn",
        }, f"both sites need an (fn, base, scale) triple; got keys {sorted(mhc_weights or ())}"
        mhc_cfg = self._mhc_config(config)

        def _mhc(site: str) -> TtMHCWrap:
            cache = {}
            if weight_cache_path is not None:
                cache = {"weight_cache_path": weight_cache_path, "cache_name_prefix": f"layer_{layer_idx}.mhc_{site}"}
            kwargs = dict(tp_axis=tp_axis, num_links=num_links, topology=tp_topology, **cache)
            if mhc_weights is None:
                return TtMHCWrap.from_cache(mesh_device, mhc_cfg, **kwargs)
            return TtMHCWrap(mesh_device, mhc_cfg, *mhc_weights[site], **kwargs)

        self.attn_res = _mhc("attn")
        self.ffn_res = _mhc("ffn")

    def reset_state(self, slot: int) -> None:
        """Start a new request in user slot ``slot``: its attention state goes back to an empty cache at
        position 0. The tables, and every other slot's state, stay as they are.

        The fresh state is built before the old one is freed, so a failed allocation leaves the old one
        intact. That includes the CSA indexer's overlap state for the slot, which the indexer holds and
        ``new_state`` replaces only after the rest has been allocated. The shared index key cache is not
        cleared: each chunk writes its rows before scoring and scores only up to its own position, so
        nothing reads what the slot's previous request left."""
        old = self.attn_states[slot]
        self.attn_states[slot] = self.attn.new_state(cache_user_id=slot)
        old.deallocate()

    def forward(
        self,
        x: ttnn.Tensor,
        actual_isl: Optional[int] = None,
        input_ids: Optional[torch.Tensor] = None,
        padding_side: str = "right",
        cache_user_id: int = 0,
    ) -> ttnn.Tensor:
        """One chunk of packed residual streams in, the same shape out.

        ``actual_isl`` is the chunk's real pre-pad length and ``input_ids`` the chunk's token ids,
        which a hash-routed layer indexes its expert table with and a top-k layer ignores.
        ``cache_user_id`` is the user slot whose attention state the chunk continues. Where the chunk
        begins is not an argument: that slot's state tracks it and advances itself.

        Each norm sits inside its site's sublayer, not before it: the hyper-connection collapses its
        streams first, and the norm belongs on what comes out of that collapse.
        """
        assert padding_side == "right", f"V4 attention is right-padded by construction, got {padding_side!r}"
        assert (
            0 <= cache_user_id < len(self.attn_states)
        ), f"cache_user_id {cache_user_id} is outside [0, {len(self.attn_states)})"
        state = self.attn_states[cache_user_id]

        def _attn(h):
            normed = self.attn_norm(h)
            out = self.attn(normed, seq_len_actual=actual_isl, state=state)
            ttnn.deallocate(normed)
            return out

        def _ffn(h):
            normed = self.ffn_norm(h)
            # TtMoe works in 3D; the block's own tensors are 4D.
            out, _ = self.ffn(
                ttnn.squeeze(normed, dim=0),
                actual_isl=actual_isl,
                padding_side=padding_side,
                # The MoE reads a non-zero actual_start as "your rows are rotated across the SP
                # axis", which is ttMLA's layout, not ours: this block uploads the sequence as a
                # plain contiguous shard, so the sequential row counts are the correct ones.
                actual_start=0,
                input_ids=input_ids,
            )
            ttnn.deallocate(normed)
            return ttnn.unsqueeze(out, dim=0)

        x = self.attn_res(x, _attn, sublayer_dtype=_SUBLAYER_DTYPE)
        return self.ffn_res(x, _ffn, sublayer_dtype=_SUBLAYER_DTYPE)
