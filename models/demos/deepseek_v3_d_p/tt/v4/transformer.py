# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The DeepSeek-V4 layer stack for prefill: embed, expand to hyper-connection streams, the decoder
blocks, then the head that collapses the streams and the final norm.

Not a ``TtPrefillTransformer``: that stack is MLA with a caller-owned KV cache, where every V4 attention
owns its own state (``TtV4Block.attn_state``) and advances it per chunk. There is no LM head; the
output on the last rank is ``norm(hc_head(h))``, ``DeepseekV4Model``'s ``last_hidden_state``.

Activations between layers, and across a pipeline boundary, are the packed residual streams
``[1, 1, S/sp, hc_mult * hidden/tp]`` fp32 (see ``TtV4Block``).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.reference.mhc.mhc_reference import MHCConfig
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import TtMHCHead, mhc_expand
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
from models.demos.deepseek_v3_d_p.tt.tt_parallel_embedding import TtParallelEmbedding
from models.demos.deepseek_v3_d_p.tt.v4.block import TtV4Block

_HASH_GATE_MODES = (GateComputeMode.HASH_HOST, GateComputeMode.HASH_DEVICE)


class TtV4Transformer(LightweightModule):
    """A slice of DeepSeek-V4's layers, plus the embedding on the first rank and the tail on the last."""

    @staticmethod
    def check_cache_complete(
        cache_path,
        num_layers: int,
        experts_per_chip: int = 8,
        first_k_dense: int = 0,
        first_layer_idx: int = 0,
        is_first_rank: bool = True,
        is_last_rank: bool = True,
        kv_only_last_layer: bool = False,
        model_cfg: type | None = None,
        routed_expert_weights_dtype=None,
        mtp_levels: int = 0,
    ) -> bool:
        """Always False: V4 attention reads its weights off torch modules and writes no ttnn cache, so a
        model can only be built from source weights."""
        return False

    def __init__(
        self,
        mesh_device,
        config,
        model_cfg: type,
        state_dict: dict,
        num_layers: int,
        seq_len: int,
        *,
        first_layer_idx: int = 0,
        is_first_rank: bool = True,
        is_last_rank: bool = True,
        max_seq_len: Optional[int] = None,
        num_links: int = 1,
        topology=None,
        sp_axis: int = 0,
        tp_axis: int = 1,
        gate_fallback_mode: Optional[GateComputeMode] = None,
        weight_cache_path=None,
        is_balanced: bool = False,
        is_chunked: bool = True,
        slot_num: int = 1,
        kv_only_last_layer: bool = False,
        padding_side: str = "right",
        sparse_kv_cache_format=None,
        mtp_predictor=None,
        **block_kwargs,
    ):
        """``state_dict`` is ``{"embed_weight", "norm_weight", "hc_head": (fn, base, scale), "layers"}``,
        where ``layers[i]`` holds local layer i's ``{"block", "attn_reference", "mhc_weights"}``
        (``reference.deepseek_v4.model.v4_layer_weights``). ``layers`` may be a lazy sequence; each
        entry is read once, while its block is built.

        ``gate_fallback_mode`` applies to the top-k MoE layers. Hash layers route by token id and always
        take the hash gate, ``HASH_DEVICE`` unless a hash mode is passed.
        """
        super().__init__()
        if mtp_predictor is not None:
            raise ValueError("DeepSeek-V4 prefill has no MTP predictor; got a non-None mtp_predictor")
        if sparse_kv_cache_format is not None:
            raise ValueError(
                f"DeepSeek-V4 attention owns its own state and takes no KV cache format; got {sparse_kv_cache_format!r}"
            )
        if kv_only_last_layer:
            raise ValueError("DeepSeek-V4 has no KV-only layer: the attention state is the whole sublayer's output")
        # Each attention state is one user's (TtHCA / TtSWA assert batch 1).
        if slot_num != 1:
            raise ValueError(f"DeepSeek-V4 attention state is single-user; got slot_num={slot_num}")
        if padding_side != "right":
            raise ValueError(f"V4 attention is right-padded by construction, got {padding_side!r}")

        topology = topology if topology is not None else per_axis_topology()
        tp_topology = topology[1] if isinstance(topology, tuple) else topology
        self.mesh_device = mesh_device
        self.config = config
        self.first_layer_idx = first_layer_idx
        self.num_layers = num_layers
        self.is_first_rank = is_first_rank
        self.is_last_rank = is_last_rank
        self.sp_axis = sp_axis
        self.hc_mult = config.hc_mult
        self.position = 0

        layer_ids = range(first_layer_idx, first_layer_idx + num_layers)
        assert (
            layer_ids.stop <= config.num_hidden_layers
        ), f"layers [{layer_ids.start}, {layer_ids.stop}) exceed the config's {config.num_hidden_layers}"
        self.has_hash_layers = any(config.mlp_layer_types[i] == "hash_moe" for i in layer_ids)

        self.embed = None
        if is_first_rank:
            self.embed = TtParallelEmbedding(
                mesh_device=mesh_device,
                vocab_size=config.vocab_size,
                emb_dim=config.hidden_size,
                torch_weight=state_dict.get("embed_weight"),
                sp_axis=sp_axis,
                tp_axis=tp_axis,
                weight_cache_path=weight_cache_path,
            )

        layers = state_dict["layers"]
        assert len(layers) == num_layers, f"state_dict holds {len(layers)} layers, the slice is {num_layers}"
        self.layers = []
        for local_idx, layer_idx in enumerate(layer_ids):
            hash_layer = config.mlp_layer_types[layer_idx] == "hash_moe"
            # None lets the block pick the gate its layer kind routes with.
            layer_gate = gate_fallback_mode if (gate_fallback_mode in _HASH_GATE_MODES) == hash_layer else None
            weights = layers[local_idx]
            self.layers.append(
                TtV4Block(
                    mesh_device=mesh_device,
                    config=config,
                    model_cfg=model_cfg,
                    state_dict=weights["block"],
                    layer_idx=layer_idx,
                    seq_len=seq_len,
                    attn_reference=weights["attn_reference"],
                    mhc_weights=weights["mhc_weights"],
                    num_links=num_links,
                    topology=topology,
                    sp_axis=sp_axis,
                    tp_axis=tp_axis,
                    max_seq_len=max_seq_len,
                    gate_fallback_mode=layer_gate,
                    weight_cache_path=weight_cache_path,
                    is_balanced=is_balanced,
                    **block_kwargs,
                )
            )
            del weights

        self.hc_head = None
        self.norm = None
        if is_last_rank:
            mhc_cfg = MHCConfig(
                dim=config.hidden_size,
                n=config.hc_mult,
                sinkhorn_iters=config.hc_sinkhorn_iters,
                eps=config.hc_eps,
                norm_eps=config.rms_norm_eps,
            )
            self.hc_head = TtMHCHead(
                mesh_device, mhc_cfg, *state_dict["hc_head"], tp_axis=tp_axis, num_links=num_links, topology=tp_topology
            )
            self.norm = TtDistributedRmsNorm(
                mesh_device=mesh_device,
                emb_dim=config.hidden_size,
                torch_weight=state_dict.get("norm_weight"),
                epsilon=config.rms_norm_eps,
                cluster_axis=tp_axis,
                num_links=num_links,
                topology=tp_topology,
                weight_cache_path=weight_cache_path,
                cache_name_prefix="norm",
            )
        logger.info(
            f"Built TtV4Transformer layers [{layer_ids.start}, {layer_ids.stop}), "
            f"first_rank={is_first_rank}, last_rank={is_last_rank}"
        )

    def reset_streams(self) -> None:
        """Start a new request: every layer's attention state goes back to empty at position 0."""
        for layer in self.layers:
            layer.reset_state()
        self.position = 0

    def release_sub_device_managers(self):
        """Remove the MoE overlap sub-device managers before the mesh device closes. Idempotent."""
        self.mesh_device.clear_loaded_sub_device_manager()
        for layer in self.layers:
            layer.ffn.release_sub_device_manager()

    def _host_token_ids(self, token_ids: ttnn.Tensor) -> torch.Tensor:
        """The chunk's ids in sequence order, ``[1, S]``, for the hash layers' tid2eid lookup.

        ``token_ids`` is ``[1, 1, S/sp]`` per chip, sharded over SP and replicated over TP.
        """
        mesh_shape = tuple(self.mesh_device.shape)
        dims = (0, 1) if self.sp_axis == 0 else (1, 0)
        ids = ttnn.to_torch(
            token_ids, mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, mesh_shape=mesh_shape, dims=dims)
        )
        return ids[:, 0, :].reshape(1, -1).to(torch.int64)

    def forward(
        self,
        token_ids: ttnn.Tensor,
        kvpe_cache=None,
        actual_isl: Optional[int] = None,
        return_intermediates: bool = False,
        read_profiler: bool = False,
        d2h_service=None,
        metadata_msg=None,
        on_layer_complete: Optional[Callable[[int], None]] = None,
        on_layer_hidden: Optional[Callable] = None,
        actual_start: Optional[int] = None,
        actual_end: Optional[int] = None,
        cache_user_id: int = 0,
        index_kv_cache=None,
        metadata=None,
        *,
        input_ids: Optional[torch.Tensor] = None,
        layer_tap: Optional[Callable[[int, ttnn.Tensor], None]] = None,
        mtp_union=None,
        on_mtp_complete=None,
        input_is_embedded: bool = False,
        provided_levels: int = 0,
    ):
        """Run one chunk through this rank's layers.

        ``token_ids`` is the SP-sharded id tensor on the first rank and the packed residual streams on
        any other. ``input_ids`` are the chunk's host ids in sequence order; a first rank reads them back
        from ``token_ids`` when a hash layer needs them, a later rank holding a hash layer must be given
        them. ``actual_start``, when given, must be where the attention state already is: the state
        advances itself and cannot seek.

        ``layer_tap(global_layer_idx, h)`` sees each layer's output streams; ``on_layer_complete`` is
        called with the global layer index once that layer's attention state has advanced.

        Returns ``norm(hc_head(h))`` ``[1, 1, S/sp, hidden/tp]`` bf16 on the last rank, else the packed
        streams for the next rank.
        """
        if mtp_union is not None or on_mtp_complete is not None or input_is_embedded or provided_levels:
            raise ValueError(
                "DeepSeek-V4 prefill has no MTP predictor; MTP forward arguments must be left at their defaults"
            )
        if kvpe_cache is not None or index_kv_cache is not None:
            raise ValueError("DeepSeek-V4 attention owns its state; kvpe_cache and index_kv_cache must be None")
        if metadata is not None:
            raise NotImplementedError("DeepSeek-V4 has no traced path: the attention state advances on host counters")
        if d2h_service is not None or on_layer_hidden is not None or return_intermediates:
            raise NotImplementedError("DeepSeek-V4 supports host layer acks (on_layer_complete) only")
        if cache_user_id not in (None, 0):
            raise ValueError(f"DeepSeek-V4 attention state is single-user; got cache_user_id={cache_user_id}")
        if actual_start is not None and actual_start != self.position:
            raise ValueError(
                f"chunk starts at {actual_start}, but the attention state is at {self.position}; "
                f"call reset_streams() to start a new request"
            )

        if self.is_first_rank:
            if input_ids is None and self.has_hash_layers:
                input_ids = self._host_token_ids(token_ids)
            embedded = ttnn.unsqueeze_to_4D(self.embed(token_ids))
            streams = ttnn.typecast(embedded, ttnn.float32)
            ttnn.deallocate(embedded)
            h = mhc_expand(streams, self.hc_mult)
            ttnn.deallocate(streams)
            owns_input = True
        else:
            assert (
                input_ids is not None or not self.has_hash_layers
            ), f"layers from {self.first_layer_idx} include a hash-routed layer, which needs the chunk's host input_ids"
            h = token_ids
            owns_input = False

        chunk_tokens = h.shape[-2] * self.mesh_device.shape[self.sp_axis]
        actual_isl = chunk_tokens if actual_isl is None else actual_isl
        for local_idx, layer in enumerate(self.layers):
            layer_idx = self.first_layer_idx + local_idx
            out = layer(h, actual_isl=actual_isl, input_ids=input_ids)
            if owns_input:
                ttnn.deallocate(h)
            h, owns_input = out, True
            if on_layer_complete is not None:
                on_layer_complete(layer_idx)
            if layer_tap is not None:
                layer_tap(layer_idx, h)
        self.position += actual_isl

        if not self.is_last_rank:
            return h
        collapsed = self.hc_head(h)
        if owns_input:
            ttnn.deallocate(h)
        normed_in = ttnn.typecast(collapsed, ttnn.bfloat16)
        ttnn.deallocate(collapsed)
        out = self.norm(normed_in)
        ttnn.deallocate(normed_in)
        return out
