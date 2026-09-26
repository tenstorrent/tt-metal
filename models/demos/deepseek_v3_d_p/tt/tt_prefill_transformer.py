# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
TtPrefillTransformer — multi-layer prefill model for DeepSeek V3.

Composes: embed -> [block x N]. The populated KV cache is the output: production prefill hands the
KV cache to decode, which owns the LM head, so there is no norm / LM-head / sampling tail here.

Equivalent to the reference Transformer class (models/demos/deepseek_v3/reference/deepseek/model.py:419)
but targeting the TT prefill path with SP+TP parallelism.
"""

from pathlib import Path
from typing import Callable, Optional

import torch
from loguru import logger
from tracy import signpost
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.mla.indexer import resolve_has_indexer
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.mla.utils import (
    create_balanced_chunk_order,
    global_to_local_token_id,
    reverse_reorder_tensor_chunks,
    rotated_row_of_position,
    rotated_rows_are_contiguous,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.device_windows import MTPDeviceEmbedSource, MTPDeviceGeneration
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import (
    build_mtp_generation_keep_mask,
    build_mtp_generation_select,
)
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
from models.demos.deepseek_v3_d_p.tt.tt_lm_head import TtLMHead
from models.demos.deepseek_v3_d_p.tt.tt_parallel_embedding import TtParallelEmbedding
from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TopologyArg, TtPrefillBlock
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCache, MlaKvCacheFormat


def rank_loads_embedding(is_first_rank: bool, is_last_rank: bool, mtp_levels: int) -> bool:
    """Does this rank load the token-embedding table?

    The first rank embeds the prompt; a last rank running MTP needs the same table for the final
    chunk's generated tokens.
    """
    return bool(is_first_rank or (mtp_levels and is_last_rank))


class TtPrefillTransformer(LightweightModule):
    """
    Multi-layer prefill transformer for DeepSeek V3.

    Architecture: embed -> [TtPrefillBlock x num_layers]. No norm / LM-head / sampling tail: the
    populated KV cache is the output (decode owns that processing).

    State dict keys:
        embed_weight:   torch.Tensor [vocab_size, emb_dim]
        layers:         list[dict] — per-layer state dicts for TtPrefillBlock
    """

    @staticmethod
    def check_cache_complete(
        cache_path: Path | None,
        num_layers: int,
        experts_per_chip: int = 8,
        first_k_dense: int = 3,
        first_layer_idx: int = 0,
        is_first_rank: bool = True,
        is_last_rank: bool = True,
        kv_only_last_layer: bool = False,
        model_cfg: type | None = None,
        routed_expert_weights_dtype: ttnn.DataType = DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE,
        mtp_levels: int = 0,
    ) -> bool:
        """
        Top-level cache completeness check for the full transformer.

        Checks the embedding and all blocks (norms + MLA + FFN/MoE). There is no final norm / LM head
        to check: the transformer has no tail.
        Replaces the monolithic check_ttnn_cache_complete from cache_utils.py.

        Args:
            cache_path: Path to TTNN weight cache directory
            num_layers: Number of transformer layers built by this instance
            experts_per_chip: Number of routed experts per chip (default: 8)
            first_k_dense: Number of initial dense (non-MoE) layers (default: 3)
            first_layer_idx: Global index of this instance's first layer. Non-zero
                for a pipeline-parallel rank owning a layer slice; block cache keys
                are global, so dense/MoE selection must use the global index.
            routed_expert_weights_dtype: dtype the routed experts were/will be BUILT at.
                as_tensor stamps it into the tensorbin filename, so the completeness check must
                pin the same value it will later request -- otherwise a stale cache at another
                dtype reports complete and the empty placeholder is loaded as the weights.
            is_first_rank: a pipeline-parallel rank builds the embedding only on the
                first rank, so check it only there. True for single-rank.
            is_last_rank / kv_only_last_layer: the rank's position and last-layer mode, passed by
                the runtime alongside the model's own construction arguments. There is no final
                norm / LM-head cache to gate on them any more, so they do not change what is checked.
            model_cfg: Variant static-constants class, forwarded to the per-block check. Optional
                so existing callers are unaffected, but MUST be passed for a LatentMoE model
                (Kimi-K3): without it the block check cannot know to look for the
                latent-projection cache files and reports a cache missing them as complete.
            mtp_levels: K, the MTP levels this model runs (0 = none). A last rank running MTP loads
                the embedding table too, so pass the config value and let rank_loads_embedding() decide.

        Returns:
            True if all expected cache files exist, False otherwise
        """
        if not cache_path or not cache_path.exists():
            logger.debug(f"TTNN cache path does not exist: {cache_path}")
            return False

        # Initialize fast cache checker for this directory
        init_checker(cache_path)

        if rank_loads_embedding(
            is_first_rank, is_last_rank, mtp_levels
        ) and not TtParallelEmbedding.check_cache_complete(cache_path):
            return False

        # Per-layer blocks — cache keys are global, so index globally.
        for local_idx in range(num_layers):
            layer_idx = first_layer_idx + local_idx
            is_dense = layer_idx < first_k_dense
            if not TtPrefillBlock.check_cache_complete(
                cache_path,
                layer_idx,
                is_dense,
                experts_per_chip,
                model_cfg=model_cfg,
                routed_expert_weights_dtype=routed_expert_weights_dtype,
            ):
                return False

        if mtp_levels and is_last_rank and not kv_only_last_layer:
            if not TtDistributedRmsNorm.check_cache_complete(cache_path, "norm"):
                return False
            if not TtLMHead.check_cache_complete(cache_path):
                return False

        logger.info(f"TTNN cache complete at {cache_path} ({num_layers} layers)")
        return True

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        config: PretrainedConfig,
        model_cfg: type,
        state_dict: dict,
        num_layers: int,
        seq_len: int,
        dispatch_buffer_capacity_factor: int = 2,
        num_links: int = 1,
        topology: TopologyArg = ttnn.Topology.Linear,
        sp_axis: int = 0,
        tp_axis: int = 1,
        is_balanced: bool = False,
        padding_side: str = "right",
        gate_fallback_mode: GateComputeMode = GateComputeMode.HOST_ALL,
        routed_expert_activations_dtype=ttnn.bfloat8_b,
        routed_expert_weights_dtype=DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE,
        shared_expert_activations_dtype=ttnn.bfloat16,
        shared_expert_weights_dtype=ttnn.bfloat8_b,
        weight_cache_path: Optional[Path] = None,
        is_chunked: bool = False,
        slot_num: int = 1,
        max_seq_len: Optional[int] = None,
        kv_only_last_layer: bool = False,
        routing_use_l1_small_for_semaphores: bool = False,
        first_layer_idx: int = 0,
        is_first_rank: bool = True,
        is_last_rank: bool = True,
        sparse_kv_cache_format: MlaKvCacheFormat = MlaKvCacheFormat.BF16_RM,
        overlap_shared_expert_with_dispatch: bool = True,
        lm_head_is_column_parallel: bool = True,
        mtp_predictor=None,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.seq_len = seq_len
        self.padding_side = padding_side
        self.is_chunked = is_chunked
        self.num_layers = num_layers
        self.kv_only_last_layer = kv_only_last_layer
        # Pipeline-parallel slicing. A rank owns layers [first_layer_idx, first_layer_idx+num_layers) and
        # builds the embedding only on the first rank. There is no norm / LM-head / sampling tail: on the
        # last rank the populated KV cache is the output. All default so a single-rank instance builds the
        # whole model unchanged.
        self.is_first_rank = is_first_rank
        self.is_last_rank = is_last_rank
        # A kv-only last layer produces no hidden state, so it only makes sense on the rank with no
        # downstream consumer of the activation (the runner sets kv_only_last_layer = is_last_rank and ...).
        assert not (kv_only_last_layer and not is_last_rank), (
            "kv_only_last_layer requires is_last_rank: a non-last pipeline rank must hand its hidden state "
            "to the next rank, which a kv-only last layer does not produce"
        )
        # GLM-5.2 indexer reuse: global per-layer full/shared map (None on models without it -> every
        # layer computes its own indexer, i.e. current behavior). first_layer_idx maps this rank's
        # local layer slice onto the global map.
        self.first_layer_idx = first_layer_idx
        self.indexer_types = getattr(config, "indexer_types", None)

        tp_topology = topology[1] if isinstance(topology, tuple) else topology
        sp_topology = topology[0] if isinstance(topology, tuple) else topology

        if not state_dict and not (weight_cache_path and weight_cache_path.exists()):
            raise ValueError(
                "TtPrefillTransformer requires weights: pass a non-empty state_dict "
                f"or a weight_cache_path to an existing cache (got {weight_cache_path=})."
            )

        logger.info(f"Building TtPrefillTransformer with {num_layers} layers, seq_len={seq_len}")

        num_mtp_levels = 0 if mtp_predictor is None else int(mtp_predictor.num_levels)

        self.embed = (
            TtParallelEmbedding(
                mesh_device=mesh_device,
                vocab_size=config.vocab_size,
                emb_dim=config.hidden_size,
                torch_weight=state_dict.get("embed_weight"),  # None if cache exists
                sp_axis=sp_axis,
                tp_axis=tp_axis,
                weight_cache_path=weight_cache_path,
            )
            if rank_loads_embedding(is_first_rank, is_last_rank, num_mtp_levels)
            else None
        )

        self.num_kvpe_cache_layers = num_layers + num_mtp_levels

        # --- Transformer layers ---
        # layer_idx is the GLOBAL index (drives weight cache keys + dense/MoE selection);
        # sparse indexer which stage it is, so its (separately numbered) key cache is rank-local too.
        # With kv_only_last_layer, the last block is built kv_only=True (only attn_norm + the KV
        # branch of MLA).
        self.layers = []
        # One llama4 query-scale cache for every layer: its contents depend only on the chunk offset
        # and mesh/config geometry, all layer-invariant (see ttMLA._llama4_scale). Per-layer dicts held
        # 36 byte-identical copies of each offset's tensor.
        self._llama4_scale_cache: dict = {}
        for local_idx in range(num_layers):
            layer_idx = first_layer_idx + local_idx
            is_last = local_idx == num_layers - 1
            logger.info(f"Building layer {local_idx}/{num_layers} (global idx {layer_idx})...")
            # Get layer weights or empty dict if loading from cache. state_dict, when
            # provided, holds this instance's slice (local indexing).
            layer_state = state_dict["layers"][local_idx] if state_dict.get("layers") else {}
            layer = TtPrefillBlock(
                mesh_device=mesh_device,
                config=config,
                model_cfg=model_cfg,
                state_dict=layer_state,
                layer_idx=layer_idx,
                seq_len=seq_len,
                dispatch_buffer_capacity_factor=dispatch_buffer_capacity_factor,
                num_links=num_links,
                topology=topology,
                sp_axis=sp_axis,
                tp_axis=tp_axis,
                is_balanced=is_balanced,
                gate_fallback_mode=gate_fallback_mode,
                routed_expert_activations_dtype=routed_expert_activations_dtype,
                routed_expert_weights_dtype=routed_expert_weights_dtype,
                shared_expert_activations_dtype=shared_expert_activations_dtype,
                shared_expert_weights_dtype=shared_expert_weights_dtype,
                weight_cache_path=weight_cache_path,
                is_chunked=is_chunked,
                slot_num=slot_num,
                layer_num=self.num_kvpe_cache_layers,
                max_seq_len=max_seq_len,
                kv_only=kv_only_last_layer and is_last,
                routing_use_l1_small_for_semaphores=routing_use_l1_small_for_semaphores,
                sparse_kv_cache_format=sparse_kv_cache_format,
                overlap_shared_expert_with_dispatch=overlap_shared_expert_with_dispatch,
                first_layer_idx=first_layer_idx,
                llama4_scale_cache=self._llama4_scale_cache,
            )
            self.layers.append(layer)

        build_tail = mtp_predictor is not None and is_last_rank and not kv_only_last_layer
        self.norm = (
            TtDistributedRmsNorm(
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
            if build_tail
            else None
        )
        self.lm_head = (
            TtLMHead(
                mesh_device=mesh_device,
                emb_dim=config.hidden_size,
                vocab_size=config.vocab_size,
                torch_weight=state_dict.get("lm_head_weight"),
                num_links=num_links,
                topology=tp_topology,
                is_balanced=is_balanced,
                weight_cache_path=weight_cache_path,
                is_column_parallel=lm_head_is_column_parallel,
            )
            if build_tail
            else None
        )

        # --- RoPE (computed once, reused across all layers) ---
        self.rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=is_balanced)

        # Chunked prefill uses the KV-pad-aware indexed rotated path: whole-cache cos/sin/trans built
        # once here and reused for every chunk (only the runtime kv_actual offset varies). seq_len is
        # the per-chunk size and max_seq_len the full per-user cache length.
        #
        # SPARSE (DSA) layers ALWAYS use the indexed rotated path — single-shot is folded onto the
        # block-cyclic path as one full-seq chunk (chunk_size_global == seq_len), so build the indexed
        # tables whenever the model is sparse too, not only when chunked. Dense single-shot keeps None
        # (rotary_embedding_llama via get_rope_tensors).
        self._has_indexer = resolve_has_indexer(config)
        self.indexed_rope = (
            self.rope_setup.get_rope_tensors_indexed(
                cache_seq_len_global=max_seq_len if max_seq_len is not None else seq_len,
                chunk_size_global=seq_len,
                tail_slack=is_chunked,
            )
            if (is_chunked or self._has_indexer)
            else None
        )

        self.is_balanced = is_balanced
        self.chunk_order = create_balanced_chunk_order(mesh_device.shape[sp_axis]) if is_balanced else None

        self.sp_axis = sp_axis
        self.tp_axis = tp_axis
        self.mesh_shape = tuple(mesh_device.shape)
        self.sp_factor = mesh_device.shape[sp_axis]
        self.tp_factor = mesh_device.shape[tp_axis]
        self.emb_dim_per_chip = config.hidden_size // self.tp_factor
        self.num_links = num_links
        self.sp_topology = sp_topology

        self.mtp_predictor = mtp_predictor
        self.num_mtp_levels = num_mtp_levels
        if mtp_predictor is not None:
            assert is_last_rank and not kv_only_last_layer, (
                "MTP is seeded by h^0 = model.norm(trunk output) and needs the LM head for the last "
                "chunk's generated tokens; both live only on a last rank that builds the tail"
            )
            assert padding_side == "right", (
                f"MTP assumes right padding, got padding_side={padding_side!r}. Under left padding "
                "the last chunk's generated ids are written at positions actual_end + k, which land "
                "in the middle of the padding instead of after the last real token. It does not "
                "raise -- it just produces the wrong embedding window."
            )
            assert mtp_predictor.first_cache_slot == num_layers, (
                f"MTP writes KV slots [first_cache_slot, first_cache_slot + K); it must start where "
                f"the trunk's slots end, at {num_layers}, not {mtp_predictor.first_cache_slot} -- "
                "otherwise the levels overwrite trunk layers or leave a hole"
            )
            if self.indexer_types is not None:
                assert len(self.indexer_types) > mtp_predictor.layer_idx, (
                    f"config.indexer_types has {len(self.indexer_types)} entries and does not cover "
                    f"MTP layer {mtp_predictor.layer_idx}. indexer_layer_is_reused() then falls "
                    "through its out-of-range guard, so the level gets a real indexer by accident "
                    "rather than by declaration, and full_indexer_rank() sizes the index cache one "
                    "slot short. Call enable_mtp_indexer_slot(config, layer_idx) before building the "
                    "predictor -- on a COPY, since config_only is lru_cached and this mutates."
                )
            mtp_stride = getattr(mtp_predictor.module.layer.mla, "layer_num", None)
            assert mtp_stride in (None, self.num_kvpe_cache_layers), (
                f"the MTP block strides users by {mtp_stride} but the trunk blocks stride by "
                f"{self.num_kvpe_cache_layers}; build the predictor with "
                f"layer_num={self.num_kvpe_cache_layers} (it reaches TtPrefillBlock through "
                "TtMTPModule's **block_kwargs)"
            )
            assert self.embed is not None, "MTP needs the embedding table on this rank (see --- Embedding ---)"

        logger.info(f"TtPrefillTransformer construction complete ({num_layers} layers)")

    def set_trace_controller(self, controller):
        """Attach (or clear with None) a SubDeviceTraceController on every layer's MoE, so a ttnn
        trace captured over forward() is split at the shared-expert/dispatch sub-device boundaries
        (see utils/sub_device_trace.py). Pass None to restore plain eager load/clear.

        Both dense-MLA and sparse/DSA (indexer) models are traceable: the indexer ops
        (ring_indexer_score_dsa, topk_large_indices) read their per-chunk scalars on-device from the
        metadata tensors, so a replay derives each chunk's causal window instead of reusing the
        captured one."""
        for layer in self.layers:
            layer.set_trace_controller(controller)

    def release_sub_device_managers(self):
        """Remove every MoE-created overlap sub-device manager before closing the mesh device.
        Ensures none is loaded first (clear is idempotent). Leaving managers registered at mesh close
        has been observed to segfault the teardown. Safe/idempotent — call once at end of a run."""
        self.mesh_device.clear_loaded_sub_device_manager()
        for layer in self.layers:
            layer.release_sub_device_managers()

    def _to_host(self, tt_tensor):
        """Bring SP+TP sharded tensor to host as [1, seq, emb] bfloat16."""
        host = ttnn.to_torch(
            tt_tensor,
            mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(-2, -1), mesh_shape=self.mesh_device.shape),
        ).to(torch.bfloat16)
        if len(tt_tensor.shape) == 4:
            host = host.squeeze(0)
        return host

    def forward(
        self,
        token_ids: ttnn.Tensor,
        kvpe_cache: MlaKvCache,
        actual_isl: int,
        return_intermediates: bool = False,
        read_profiler: bool = False,
        d2h_service=None,
        metadata_msg: Optional[ttnn.Tensor] = None,
        on_layer_complete: Optional[Callable[[int], None]] = None,
        on_layer_hidden: Optional[Callable[[int, ttnn.Tensor], None]] = None,
        actual_start: Optional[int] = None,
        actual_end: Optional[int] = None,
        cache_user_id: int = 0,
        index_kv_cache: Optional[ttnn.Tensor] = None,
        metadata: Optional[ttnn.Tensor] = None,
        mtp_union=None,
        on_mtp_complete: Optional[Callable] = None,
        input_is_embedded: bool = False,
        provided_levels: int = 0,
    ):
        """
        Forward pass: [embed] -> [block x N]. The populated KV cache is the output.

        Pipeline-parallel ranks run a slice of this: the embedding runs only on the
        first rank and only the last rank ends the forward (there is no norm / LM-head /
        sampling tail: decode owns the processing), so the input and output are dual-mode
        (see Args/Returns).

        Args:
            token_ids: on the first rank, [1, 1, seq_len_per_chip] uint32 SP-sharded
                token IDs to embed; on a non-first rank, the [1, 1, seq_per_chip,
                emb_dim/tp] hidden-state activation handed over from the previous rank.
                With `input_is_embedded` it is that activation on the first rank too.
            kvpe_cache: externally created KVPE cache [num_layers, 1, seq_len_local, head_dim];
                        each layer writes to its own slot via cache_layer_idx
            index_kv_cache: sparse-DSA (v3.2 / GLM) — the caller-owned, layer-stacked block-cyclic indexer
                        key cache [num_users * num_layers, 1, T, D_idx] (SP-sharded on the seq axis), same
                        ownership as kvpe_cache. Required for EVERY sparse forward — chunked AND single-shot
                        (folded onto the block-cyclic path); the indexer never self-allocates it. None only
                        for dense (non-sparse) variants.
            return_intermediates: if True, sync + snapshot to host after each stage
            read_profiler: if True, read TTNN profiler after each layer to avoid profiler buffer overflows
            d2h_service: optional service used to send a layer-ack completion signal back to host once
                        each layer's KV cache has been populated on device. When set, each block zeros the
                        cache pad window and enqueues the ack via the outbound_socket_service_sync device op
                        on the same CQ (no host sync). When None, no ack or zeroing.
            metadata_msg: the chunk's PrefillMetadata device tensor sent as each ack record; required when
                        d2h_service is set.
            on_layer_complete: the HOST-callback alternative to d2h_service (used by pipelined prefill's
                        layer-completion router). Called as on_layer_complete(layer_idx) after the same
                        pad-zero, but with a device sync first. Wire one or the other, never both.
            on_layer_hidden: optional tap fired at the END of each block with (GLOBAL layer index, block
                        output activation). Read-only — see tt_prefill_block.forward.
            mtp_union: an `MTPUnionEmbedding` holding this chunk's embedded trunk and lookahead rows
                        per chip, out of which each MTP level slices its own window. None disables MTP.
            provided_levels: how many leading MTP levels already have their lookahead token in the ids
                        that arrived; the levels above that generate one on device. Set by the runner.
            on_mtp_complete: tap fired once with (MTPPredictorOutput, generated_tokens), so the trunk's
                        return arity is the same whether or not MTP ran.
            input_is_embedded: the first rank's `token_ids` is ALREADY the embedding, so skip the
                        gather. Set by the device MTP path, which embeds the trunk rows itself.

        Returns:
            On a non-last rank: the hidden-state activation tensor to hand to the next rank.

            On the last rank (and single-rank): the intermediates dict when
            return_intermediates=True ("embed" on the first rank, then "layer_i" for every
            layer that produced a hidden state; a kv-only last layer adds none, and MTP adds
            "norm" for h^0), otherwise None. No token is produced: the populated KV cache is
            the output.
        """
        # The two ack transports are mutually exclusive: the block takes the d2h_service branch and would
        # silently drop on_layer_complete, so a caller wiring both would get half the acks it asked for
        # with no diagnostic. The runner's single-rank and pipeline branches are disjoint today; keep it so.
        assert d2h_service is None or on_layer_complete is None, (
            "d2h_service and on_layer_complete are mutually exclusive ack transports; the block takes "
            "d2h_service and would silently drop on_layer_complete"
        )

        if mtp_union is not None:
            assert self.mtp_predictor is not None, "MTP input passed but this transformer has no mtp_predictor"
            assert mtp_union.num_levels == self.num_mtp_levels, (
                f"union carries {mtp_union.num_levels} levels, predictor runs {self.num_mtp_levels}; "
                "the runner and the runtime disagree on PREFILL_MTP_LEVELS"
            )

        # Chunked prefill ([actual_start, actual_end) set) uses the prebuilt whole-cache indexed rope
        # and writes this chunk at the actual_start offset of user cache_user_id's slot; the single-shot
        # path builds per-call rope for this seq_len.
        if actual_start is not None or metadata is not None:
            # metadata path: per-chunk actual_start/actual_end live on-device in the metadata tensor
            # (read by the trace-safe MLA ops), so actual_start is None here -- still chunked prefill,
            # still the prebuilt whole-cache indexed rope.
            assert self.is_chunked, "chunked prefill (actual_start or metadata) requires is_chunked=True"
            rope_tensors = self.indexed_rope
        elif self._has_indexer:
            # Sparse single-shot is folded onto the block-cyclic path (one full-seq chunk at offset 0),
            # so it uses the indexed rope tables just like the chunked path.
            rope_tensors = self.indexed_rope
        else:
            rope_tensors = self.rope_setup.get_rope_tensors(self.seq_len)
        intermediates = {} if return_intermediates else None

        if self.is_first_rank and not input_is_embedded:
            h = self.embed(token_ids)  # [1, seq_per_chip, emb_dim/tp]
            h = ttnn.unsqueeze_to_4D(h)  # [1, 1, seq_per_chip, emb_dim/tp]
            if return_intermediates:
                ttnn.synchronize_device(self.mesh_device)
                intermediates["embed"] = self._to_host(h)
        else:
            h = token_ids

        # GLM-5.2 reuse: hold the most recent "full" layer's top-k indices and inject them into the
        # following "shared" layers. reuse=False (no indexer_types) leaves the call + 2-tuple return
        # exactly as before.
        reuse = self.indexer_types is not None
        # reuse seeds from the first "full" layer within this forward; a stack starting on a "shared"
        # layer has no prior indices (pipeline-parallel would need them threaded in from the prior rank).
        if reuse:
            assert (
                self.indexer_types[self.first_layer_idx] == "full"
            ), f"first layer {self.first_layer_idx} must be 'full' to seed indexer reuse, got '{self.indexer_types[self.first_layer_idx]}'"
        indexer_indices = None
        for i, layer in enumerate(self.layers):
            signpost(f"forward_layer_{i}_start")
            mode = self.indexer_types[self.first_layer_idx + i] if reuse else "full"
            inject = indexer_indices if (reuse and mode == "shared") else None
            ret = layer(
                h,
                rope_tensors,
                kvpe_cache,
                cache_layer_idx=i,
                return_intermediates=return_intermediates,
                d2h_service=d2h_service,
                metadata_msg=metadata_msg,
                on_layer_complete=on_layer_complete,
                on_layer_hidden=on_layer_hidden,
                actual_start=actual_start,
                actual_end=actual_end,
                cache_user_id=cache_user_id,
                actual_isl=actual_isl,
                padding_side=self.padding_side,
                indexer_indices=inject,
                return_indexer_indices=reuse,
                index_kv_cache=index_kv_cache,
                metadata=metadata,
            )
            if reuse:
                h, _, new_idx = ret
                if mode == "full":
                    # Keep the full layer's indices alive through every shared consumer. TP sequence
                    # shards own their top-k allocation; reference replacement releases it when no
                    # consumer holds it. Gathered outputs can alias persistent scratch, so do not
                    # explicitly deallocate the old tensor (it may back new_idx as well).
                    indexer_indices = new_idx
            else:
                h, _ = ret
            signpost(f"forward_layer_{i}_end")
            if self.kv_only_last_layer and i == len(self.layers) - 1:
                # Last layer was kv-only: KV cache filled, migration callback fired, no hidden state
                # produced. Nothing more to run or snapshot; the populated cache is the output.
                return intermediates
            if return_intermediates:
                ttnn.synchronize_device(self.mesh_device)
                intermediates[f"layer_{i}"] = self._to_host(h)
            if read_profiler:
                ttnn.ReadDeviceProfiler(self.mesh_device)
        # Drop the held reference. Python reference counting releases owned top-k allocations after
        # their last consumer; any TP gather scratch remains owned by TT_CCL.
        indexer_indices = None

        # Non-last pipeline ranks stop here: the layer slice's output activation is
        # handed to the next rank, which continues from this hidden state.
        if not self.is_last_rank:
            return h

        if self.norm is not None:
            h = self.norm(h)

            if return_intermediates:
                ttnn.synchronize_device(self.mesh_device)
                intermediates["norm"] = self._to_host(h)

        if return_intermediates and self.is_balanced:
            # Balanced (zigzag) SP shards the sequence in a permuted chunk order; restore the natural
            # order for every host snapshot ("embed" and "layer_i" are all sequence tensors).
            for key, tensor in intermediates.items():
                if isinstance(tensor, torch.Tensor):
                    logger.debug(f"Reordering intermediate {key} with shape {tensor.shape}")
                    intermediates[key] = reverse_reorder_tensor_chunks(tensor, self.chunk_order, seq_dim=-2)

        if mtp_union is not None:
            assert actual_start is not None, (
                "MTP needs actual_start on the host to place the rows it generates; the on-device "
                "metadata path keeps actual_start on device and cannot answer that here"
            )
            mtp_out, mtp_generated = self.run_mtp(
                h,
                kvpe_cache,
                rope_tensors,
                actual_isl,
                union=mtp_union,
                provided_levels=provided_levels,
                cache_user_id=cache_user_id,
                actual_start=actual_start,
                actual_end=actual_end,
                padding_side=self.padding_side,
                index_kv_cache=index_kv_cache,
                metadata=metadata,
                d2h_service=d2h_service,
                metadata_msg=metadata_msg,
                on_layer_complete=on_layer_complete,
                layer_ack_base=self.first_layer_idx + self.mtp_predictor.first_cache_slot,
            )
            if on_mtp_complete is not None:
                on_mtp_complete(mtp_out, mtp_generated)

        return intermediates

    def mtp_embed_ids(self, tt_ids: ttnn.Tensor) -> ttnn.Tensor:
        """Gather ``[sp, 1, N]`` uint32 ids into ``[1, 1, N, H/tp]`` bf16 TILE. Does not consume ``tt_ids``."""
        return ttnn.unsqueeze_to_4D(self.embed(tt_ids))

    def mtp_generate_embedding(self, h_normed: ttnn.Tensor, last_row: int) -> ttnn.Tensor:
        """``H^k -> [1, 1, 32*sp, H/tp]``: the greedy next token at ``last_row``, embedded and
        SP-broadcast so every chip can read it. ``last_row`` is the chip-major flat row carrying the
        chunk's last real position, which is ``actual_isl - 1`` only on a slab-aligned chunk.
        """
        assert self.lm_head is not None, "MTP generation needs the LM head (last rank, build_tail)"
        logits, _ = self.lm_head(h_normed, last_row)
        if self.lm_head.is_column_parallel and self.tp_factor > 1:
            full = ttnn.all_gather(
                logits,
                dim=-1,
                cluster_axis=self.tp_axis,
                num_links=self.lm_head.num_links,
                topology=self.lm_head.topology,
            )
            ttnn.deallocate(logits)
            logits = full
        ids = ttnn.argmax(logits, dim=-1, keepdim=False)
        ttnn.deallocate(logits)
        emb = self.mtp_embed_ids(ids)
        ttnn.deallocate(ids)
        if self.sp_factor == 1:
            return emb
        gathered = ttnn.all_gather(
            emb, dim=-2, cluster_axis=self.sp_axis, num_links=self.num_links, topology=self.sp_topology
        )
        ttnn.deallocate(emb)
        return gathered

    def _mtp_build_generation(
        self, union, actual_isl: int, actual_start: int, actual_end: int, *, provided_levels: int = 0
    ):
        """The :class:`MTPDeviceGeneration` for levels ``[provided_levels, K)``: one keep mask and one
        one-hot selector per generated level. A level below ``provided_levels`` gets no selector.
        """
        assert not self.is_balanced, (
            "MTP device generation is block-cyclic only: under is_balanced a chip's union is not a "
            "contiguous position range, so 'the rows holding position actual_end + k' is not one row "
            "per chip. Chunked prefill passes is_balanced=False throughout."
        )
        assert (
            actual_end - actual_start == actual_isl
        ), f"actual_end - actual_start = {actual_end - actual_start} != actual_isl {actual_isl}"
        last_row = rotated_row_of_position(actual_start, self.sp_factor, self.seq_len // self.sp_factor, actual_end - 1)
        assert (
            last_row is not None
        ), f"the chunk at {actual_start} does not carry its own last real position {actual_end - 1}"
        device_id, local_token_id = global_to_local_token_id(
            last_row, self.sp_factor, self.seq_len, is_balanced=self.is_balanced
        )
        source_row = device_id * ttnn.TILE_SIZE + local_token_id % ttnn.TILE_SIZE
        geom = dict(
            mesh_device=self.mesh_device,
            sp_factor=self.sp_factor,
            chunk_size=self.seq_len,
            mesh_shape=self.mesh_shape,
            sp_axis=self.sp_axis,
            num_mtp_tokens=union.num_mtp_tokens,
            chunk_start=actual_start,
            actual_end=actual_end,
        )
        generated = range(provided_levels, self.num_mtp_levels)
        keep_mask = build_mtp_generation_keep_mask(**geom, emb_dim_per_chip=self.emb_dim_per_chip, levels=generated)
        selects = [
            build_mtp_generation_select(**geom, level=k, source_row=source_row) if k in generated else None
            for k in range(self.num_mtp_levels)
        ]
        return MTPDeviceGeneration(keep_mask, selects, embed_fn=lambda h: self.mtp_generate_embedding(h, last_row))

    def run_mtp(
        self,
        h_normed: ttnn.Tensor,
        kvpe_cache: MlaKvCache,
        rope_tensors: dict,
        actual_isl: int,
        *,
        union,
        provided_levels: int = 0,
        **fwd_kwargs,
    ):
        """Run the K MTP levels off ``h^0``. Returns ``(MTPPredictorOutput, generated_tokens)``.

        Only levels ``[provided_levels, K)`` generate their lookahead token on device; the rest slice
        the union. ``fwd_kwargs`` reaches every level's block, minus the predictor-owned cache slot.
        """
        assert self.mtp_predictor is not None, "run_mtp called on a transformer built without an mtp_predictor"
        assert union is not None, "run_mtp needs this chunk's MTPUnionEmbedding"
        assert (
            0 <= provided_levels <= self.num_mtp_levels
        ), f"provided_levels {provided_levels} outside [0, {self.num_mtp_levels}]"
        isl_per_chip = self.seq_len // self.sp_factor
        assert rotated_rows_are_contiguous(fwd_kwargs["actual_start"], isl_per_chip), (
            f"MTP needs a chunk start that is a multiple of the per-chip shard {isl_per_chip}; got "
            f"{fwd_kwargs['actual_start']}. Off that boundary the rotated chunk leaves the boundary chip's "
            "rows position-discontiguous, and an MTP window is a ROW shift, so level k would read the wrong "
            "position on that chip. Resume on a multiple of chunk_size // sp_factor."
        )
        generation = None
        if provided_levels < self.num_mtp_levels:
            generation = self._mtp_build_generation(
                union,
                actual_isl,
                fwd_kwargs["actual_start"],
                fwd_kwargs["actual_end"],
                provided_levels=provided_levels,
            )
        source = MTPDeviceEmbedSource(
            union,
            generation=generation,
            provided_levels=provided_levels,
        )
        fwd_kwargs["actual_isl"] = actual_isl
        try:
            out = self.mtp_predictor.forward(source, h_normed, rope_tensors, kvpe_cache, **fwd_kwargs)
        finally:
            if generation is not None:
                generation.deallocate()
        return out, source.generated_tokens
