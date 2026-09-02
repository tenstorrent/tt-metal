# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
TtPrefillTransformer — multi-layer prefill model for DeepSeek V3.

Composes: embed -> [block x N] -> norm -> lm_head -> sample

Equivalent to the reference Transformer class (models/demos/deepseek_v3/reference/deepseek/model.py:419)
but targeting the TT prefill path with SP+TP parallelism.
"""

from pathlib import Path
from typing import Callable, Optional, Union

import torch
from loguru import logger
from tracy import signpost
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.mla.indexer import resolve_has_indexer
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.mla.utils import create_balanced_chunk_order, reverse_reorder_tensor_chunks
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.device_windows import MTPDeviceEmbedSource
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.token_windows import MTPEmbedSource
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import build_position_zero_mask, prepare_prefill_mtp_window
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
from models.demos.deepseek_v3_d_p.tt.tt_lm_head import TtLMHead
from models.demos.deepseek_v3_d_p.tt.tt_parallel_embedding import TtParallelEmbedding
from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TopologyArg, TtPrefillBlock
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCache, MlaKvCacheFormat


def rank_loads_embedding(is_first_rank: bool, is_last_rank: bool, mtp_levels: int) -> bool:
    """Does this rank load the token-embedding table?

    The first rank embeds the trunk's prompt. A last rank running MTP needs the SAME table for the
    final chunk's generated tokens: the LM head hands it a token id that exists nowhere until the
    level before it has run, so no upstream rank could have embedded it and no socket could have
    carried it (see ``mtp_generate_embedding``). Its shifted WINDOWS arrive pre-gathered in the D2D
    union and cost nothing here -- it is generation alone that needs the table, 453.75 MiB/chip at
    GLM-5.2's 154880-entry vocab.

    Single source of truth, because the question is asked twice from different vantage points: the
    constructor asks it with the predictor OBJECT in hand, while ``check_cache_complete`` is a
    staticmethod that runs BEFORE the object exists and has to predict the same answer from raw
    config. When those two drifted, ``check_cache_complete`` stopped looking for the embedding at
    all on a non-first tail rank -- latent today, since the builder writes the table unconditionally
    into a rank-shared cache dir, but it would call a cache that lacks it complete.
    """
    return bool(is_first_rank or (mtp_levels and is_last_rank))


class TtPrefillTransformer(LightweightModule):
    """
    Multi-layer prefill transformer for DeepSeek V3.

    Architecture: embed -> [TtPrefillBlock x num_layers] -> norm -> lm_head -> sample

    State dict keys:
        embed_weight:   torch.Tensor [vocab_size, emb_dim]
        norm_weight:    torch.Tensor [emb_dim]
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
        mtp_levels: int = 0,
    ) -> bool:
        """
        Top-level cache completeness check for the full transformer.

        Checks embedding, all blocks (norms + MLA + FFN/MoE), and final norm.
        Replaces the monolithic check_ttnn_cache_complete from cache_utils.py.

        Args:
            cache_path: Path to TTNN weight cache directory
            num_layers: Number of transformer layers built by this instance
            experts_per_chip: Number of routed experts per chip (default: 8)
            first_k_dense: Number of initial dense (non-MoE) layers (default: 3)
            first_layer_idx: Global index of this instance's first layer. Non-zero
                for a pipeline-parallel rank owning a layer slice; block cache keys
                are global, so dense/MoE selection must use the global index.
            is_first_rank / is_last_rank: a pipeline-parallel rank builds the
                embedding only on the first rank and the final norm + LM head only
                on the last, so check only the weights it actually loads. Both True
                for single-rank.
            model_cfg: Variant static-constants class, forwarded to the per-block check. Optional
                so existing callers are unaffected, but MUST be passed for a LatentMoE model
                (Kimi-K3): without it the block check cannot know to look for the
                latent-projection cache files and reports a cache missing them as complete.
            mtp_levels: K, the MTP levels this model runs (0 = none). A last rank running MTP
                loads the embedding table too, so the check has to look for it there -- pass the
                config value and let rank_loads_embedding() decide; the rank flags are already args.

        Returns:
            True if all expected cache files exist, False otherwise
        """
        if not cache_path or not cache_path.exists():
            logger.debug(f"TTNN cache path does not exist: {cache_path}")
            return False

        # Initialize fast cache checker for this directory
        init_checker(cache_path)

        # Embedding: the first rank, plus any rank that embeds MTP windows.
        if rank_loads_embedding(
            is_first_rank, is_last_rank, mtp_levels
        ) and not TtParallelEmbedding.check_cache_complete(cache_path):
            return False

        # Per-layer blocks — cache keys are global, so index globally.
        for local_idx in range(num_layers):
            layer_idx = first_layer_idx + local_idx
            is_dense = layer_idx < first_k_dense
            if not TtPrefillBlock.check_cache_complete(
                cache_path, layer_idx, is_dense, experts_per_chip, model_cfg=model_cfg
            ):
                return False

        # Final norm + LM head: only the last rank that emits a token loads these
        # (skipped for a kv_only last layer and for non-last pipeline ranks).
        if is_last_rank and not kv_only_last_layer:
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
        routed_expert_weights_dtype=ttnn.bfloat4_b,
        shared_expert_activations_dtype=ttnn.bfloat16,
        shared_expert_weights_dtype=ttnn.bfloat8_b,
        weight_cache_path: Optional[Path] = None,
        lm_head_is_column_parallel: bool = False,
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
        mtp_predictor=None,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.seq_len = seq_len
        self.padding_side = padding_side
        self.is_chunked = is_chunked
        self.num_layers = num_layers
        self.kv_only_last_layer = kv_only_last_layer
        # Pipeline-parallel slicing. A rank owns layers [first_layer_idx, first_layer_idx+num_layers),
        # builds the embedding only on the first rank, and the norm + LM head only on the last rank that
        # also emits a token (is_last_rank and not kv_only_last_layer). All default so a single-rank
        # instance builds the whole model unchanged.
        self.is_first_rank = is_first_rank
        self.is_last_rank = is_last_rank
        # GLM-5.2 indexer reuse: global per-layer full/shared map (None on models without it -> every
        # layer computes its own indexer, i.e. current behavior). first_layer_idx maps this rank's
        # local layer slice onto the global map.
        self.first_layer_idx = first_layer_idx
        self.indexer_types = getattr(config, "indexer_types", None)

        # The blocks take the full per-axis topology (they split SP/TP internally for the MoE).
        # The final norm and LM head are pure TP-axis (cluster_axis=tp_axis) collectives, so they
        # take the scalar TP element.
        tp_topology = topology[1] if isinstance(topology, tuple) else topology

        if not state_dict and not (weight_cache_path and weight_cache_path.exists()):
            raise ValueError(
                "TtPrefillTransformer requires weights: pass a non-empty state_dict "
                f"or a weight_cache_path to an existing cache (got {weight_cache_path=})."
            )

        logger.info(f"Building TtPrefillTransformer with {num_layers} layers, seq_len={seq_len}")

        # Needed before the embedding: whether this rank loads the table depends on it, and so does
        # the KV-cache stride below.
        num_mtp_levels = 0 if mtp_predictor is None else int(mtp_predictor.num_levels)

        # --- Embedding ---
        # rank_loads_embedding() owns the rule; check_cache_complete() asks it the same question
        # before this object exists, so the cache check and this build cannot disagree.
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

        # The KV-cache slot stride, in slots per user. Normally the rank's layer count, but MTP
        # writes K MORE slots per user -- level k lands at num_layers + k -- so the cache is
        # num_layers + K deep and EVERY block, trunk and MTP alike, must stride by that same
        # number. Striding the trunk by num_layers while the cache is deeper puts user 1's layer 0
        # on top of user 0's MTP slots: silent corruption, invisible at num_users == 1, which is
        # exactly the configuration MTP is being brought up in.
        self.num_kvpe_cache_layers = num_layers + num_mtp_levels

        # --- Transformer layers ---
        # layer_idx is the GLOBAL index (drives weight cache keys + dense/MoE selection);
        # cache_layer_idx in forward is the LOCAL slot. layer_num is the per-user slot stride
        # (the block's flat KV slot is cache_user_id * layer_num + cache_layer_idx), so it matches
        # the cache's actual depth, not this rank's layer count. first_layer_idx additionally tells the
        # sparse indexer which stage it is, so its (separately numbered) key cache is rank-local too.
        # With kv_only_last_layer, the last block is built kv_only=True (only attn_norm + the KV
        # branch of MLA).
        self.layers = []
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
            )
            self.layers.append(layer)

        # --- Final norm (last token-emitting rank only) ---
        # Built iff is_last_rank and not kv_only_last_layer: a kv_only last layer (chunked prefill)
        # emits no token, and non-last pipeline ranks forward the hidden state — both skip the tail.
        build_tail = is_last_rank and not kv_only_last_layer
        self.norm = (
            TtDistributedRmsNorm(
                mesh_device=mesh_device,
                emb_dim=config.hidden_size,
                torch_weight=state_dict.get("norm_weight"),  # None if cache exists
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
            )
            if (is_chunked or self._has_indexer)
            else None
        )

        # --- LM Head (last token-emitting rank only) ---
        self.lm_head = (
            TtLMHead(
                mesh_device=mesh_device,
                emb_dim=config.hidden_size,
                vocab_size=config.vocab_size,
                torch_weight=state_dict.get("lm_head_weight"),  # None if cache exists
                num_links=num_links,
                topology=tp_topology,
                is_balanced=is_balanced,
                weight_cache_path=weight_cache_path,
                is_column_parallel=lm_head_is_column_parallel,
            )
            if build_tail
            else None
        )

        self.is_balanced = is_balanced
        self.chunk_order = create_balanced_chunk_order(mesh_device.shape[sp_axis]) if is_balanced else None

        # Sharding parameters, kept because MTP uploads its own shift-windows through the same
        # path the trunk input took (see _mtp_embed_window).
        self.sp_axis = sp_axis
        self.tp_axis = tp_axis
        self.mesh_shape = tuple(mesh_device.shape)
        self.sp_factor = mesh_device.shape[sp_axis]
        self.tp_factor = mesh_device.shape[tp_axis]
        self.emb_dim_per_chip = config.hidden_size // self.tp_factor

        # --- MTP (GLM-5.2, issue #53533) -----------------------------------------------------
        # Injected rather than built here: the predictor owns MTP weights and a full GLM decoder
        # block, and the transformer has no business loading either. What the transformer
        # contributes is the three things only it holds at once -- the embedding table, model.norm,
        # and the LM head -- which is exactly what the token->embedding boundary needs.
        self.mtp_predictor = mtp_predictor
        self.num_mtp_levels = num_mtp_levels
        self._mtp_pos0_mask = None
        if mtp_predictor is not None:
            assert is_last_rank and not kv_only_last_layer, (
                "MTP is seeded by h^0 = model.norm(trunk output) and needs the LM head for the last "
                "chunk's generated tokens; both live only on a last rank that builds the tail"
            )
            assert padding_side == "right", (
                f"MTP assumes right padding, got padding_side={padding_side!r}. Two things break "
                "under left padding, both silently: mtp_extended_stream lays the last chunk out as "
                "[ prompt | K generation slots | pad ], which puts the slots in the middle of the "
                "padding instead of after the last real token; and the position-0 mask zeroes "
                "chunk-local row 0, which is absolute position 0 only when the real tokens start "
                "there. Neither raises -- they just produce the wrong embedding window."
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
            # Built above precisely because mtp_predictor is not None; assert the wiring rather
            # than trust it, since a None table fails deep inside the first level's embed call.
            assert self.embed is not None, "MTP needs the embedding table on this rank (see --- Embedding ---)"

        logger.info(f"TtPrefillTransformer construction complete ({num_layers} layers)")

    def set_trace_controller(self, controller):
        """Attach (or clear with None) a SubDeviceTraceController on every layer's MoE, so a ttnn
        trace captured over forward() is split at the shared-expert/dispatch sub-device boundaries
        (see utils/sub_device_trace.py). Pass None to restore plain eager load/clear.

        DENSE-MLA ONLY. Tracing a sparse/DSA (indexer) model is rejected: the traced forward advances
        its per-chunk scalars through the metadata ops, and the indexer path has no metadata overload
        yet — the captured forward also never threads index_kv_cache, so a sparse model would replay
        silently WITHOUT its indexer cache and produce wrong KV rather than failing. Porting the
        indexer ops is out of scope here."""
        if controller is not None:
            assert not self._has_indexer, (
                "trace capture is not supported for sparse/DSA (indexer) attention. Supported today: "
                "the dense-MLA models (deepseek_v3, kimi_k2_6, kimi_k2_7). GLM (glm_5_1 / glm_5_2) and "
                "any other indexer/sparse-attention variant need their indexer ops ported to the "
                "per-element-tensor metadata form first — until then run them untraced (use_trace=False "
                "/ PREFILL_USE_TRACE=0)."
            )
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
        temperature: Union[float, list[float]] = 0.0,
        d2h_service=None,
        record_dev: Optional[ttnn.Tensor] = None,
        on_layer_complete: Optional[Callable[[int], None]] = None,
        on_layer_hidden: Optional[Callable[[int, ttnn.Tensor], None]] = None,
        actual_start: Optional[int] = None,
        actual_end: Optional[int] = None,
        cache_user_id: int = 0,
        index_kv_cache: Optional[ttnn.Tensor] = None,
        metadata: Optional[ttnn.Tensor] = None,
        mtp_tokens: Optional[list[int]] = None,
        mtp_union=None,
        mtp_seed_token: Optional[int] = None,
        on_mtp_complete: Optional[Callable] = None,
        input_is_embedded: bool = False,
        is_last_chunk: bool = False,
    ):
        """
        Forward pass: [embed] -> [block x N] -> [norm -> lm_head -> sample].

        Pipeline-parallel ranks run a slice of this: the embedding runs only on the
        first rank and the norm/LM-head/sample tail only on the last, so the input
        and output are dual-mode (see Args/Returns).

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
            temperature: Temperature for sampling. Can be a single float or list of floats.
                        If list, returns first temperature result but stores all in intermediates.
            d2h_service: optional service used to send a layer-ack completion signal back to host once
                        each layer's KV cache has been populated on device. When set, each block zeros the
                        cache pad window and enqueues the ack via the outbound_socket_service_sync device op
                        on the same CQ (no host sync). When None, no ack or zeroing.
            record_dev: the chunk's PrefillMetadata device tensor sent as each ack record; required when
                        d2h_service is set.
            on_layer_complete: the HOST-callback alternative to d2h_service (used by pipelined prefill's
                        layer-completion router). Called as on_layer_complete(layer_idx) after the same
                        pad-zero, but with a device sync first. Wire one or the other, never both.
            on_layer_hidden: optional tap fired at the END of each block with (GLOBAL layer index, block
                        output activation). Read-only — see tt_prefill_block.forward.
            mtp_tokens: GLM-5.2 MTP (#53533) — this chunk's seq_len + K extended token stream from
                        mtp_extended_stream(), indexed by chunk-local position, so level k's window is
                        the slice [k : k+seq_len]. None disables MTP for this chunk. Requires an
                        mtp_predictor; the K levels run after the trunk tail, off model.norm's output.
            mtp_union: the DEVICE alternative to `mtp_tokens` — an `MTPUnionEmbedding` holding this
                        chunk's `L + overhang` embedded rows per chip, from which each level slices
                        its own window on device. This is what the prefill runner passes: its ids
                        come off the H2D socket, are embedded on the first rank and reach this rank
                        inside the activation. Mutually exclusive with `mtp_tokens`.
            mtp_seed_token: optional t_P for the last chunk, saving one 32-row LM head call. Only pass
                        it when it was produced greedily — see _mtp_next_token_fn. Host path only:
                        the device path never runs the LM head (see `MTPDeviceEmbedSource`).
            is_last_chunk: this chunk ends the request, so the K positions its MTP windows read
                        past `actual_end` have no ids in the prompt. Only the producer knows that
                        boundary; see CHUNK_METADATA_SIZE_BYTES.
            on_mtp_complete: tap fired once with (MTPPredictorOutput, generated_tokens). A tap rather
                        than an extra return value, so the trunk's return arity is unchanged whether or
                        not MTP ran — the same reason on_layer_complete is a callback.
            input_is_embedded: the first rank's `token_ids` is ALREADY the embedding, so skip the
                        gather. The device MTP path sets it: that rank embeds the chunk's ids and the
                        lookahead ids into one union, and the union's trunk block is bit-identical to
                        what this embed would produce — so gathering again would re-read the same
                        rows. An explicit flag, not inferred from `mtp_union`, because a first rank
                        with no predictor is passed `mtp_union=None` (tt_prefill_runtime).

        Returns:
            On a non-last rank: the hidden-state activation tensor to hand to the next
            rank (no token — the tail did not run).

            On the last rank (and single-rank): a tuple of
            (first_token_id, first_token_prob, intermediates_dict or None)
            - first_token_id: sampled token ID (for first temperature if list provided)
            - first_token_prob: probability of sampled token (for first temperature if list provided)
            - intermediates: dict with keys like "embed", "layer_0", "norm", "lm_head", "first_token"
                            where "first_token" is a list of results for each temperature
                            (None if return_intermediates=False)
        """
        # The two ack transports are mutually exclusive: the block takes the d2h_service branch and would
        # silently drop on_layer_complete, so a caller wiring both would get half the acks it asked for
        # with no diagnostic. The runner's single-rank and pipeline branches are disjoint today; keep it so.
        assert d2h_service is None or on_layer_complete is None, (
            "d2h_service and on_layer_complete are mutually exclusive ack transports; the block takes "
            "d2h_service and would silently drop on_layer_complete"
        )

        # Check the MTP token contract HERE, before the trunk runs. The stream is not consumed until
        # the very end of this forward (run_mtp -> MTPEmbedSource, which asserts the same length), so
        # a wrong-length list otherwise costs a whole trunk chunk -- one that has already written KV --
        # before it raises. Pure host arithmetic; it costs nothing.
        assert mtp_tokens is None or mtp_union is None, (
            "mtp_tokens (host ids) and mtp_union (device embedding) are two spellings of the same "
            "input; pass exactly one"
        )
        if mtp_union is not None:
            assert self.mtp_predictor is not None, "mtp_union passed but this transformer has no mtp_predictor"
            assert mtp_union.num_levels == self.num_mtp_levels, (
                f"union carries {mtp_union.num_levels} levels, predictor runs {self.num_mtp_levels}; "
                "the runner and the runtime disagree on PREFILL_MTP_LEVELS"
            )
        if mtp_tokens is not None:
            assert self.mtp_predictor is not None, "mtp_tokens passed but this transformer has no mtp_predictor"
            assert len(mtp_tokens) == self.seq_len + self.num_mtp_levels, (
                f"mtp_tokens is {len(mtp_tokens)} ids, expected seq_len + K = {self.seq_len} + "
                f"{self.num_mtp_levels}. It is the EXTENDED stream from mtp_extended_stream(), not the "
                "chunk's own tokens -- it carries K positions past the chunk's right edge. Slice the "
                "trunk input off the same list with TtPrefillRuntime.make_mtp_chunk_input()."
            )

        # Chunked prefill ([actual_start, actual_end) set) uses the prebuilt whole-cache indexed rope
        # and writes this chunk at the actual_start offset of user cache_user_id's slot; the single-shot
        # path builds per-call rope for this seq_len. The norm/lm_head/sample tail still runs and a token
        # is returned, but the chunked caller ignores it (the populated cache is the output).
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
            # Already [1, 1, seq_per_chip, emb_dim/tp]: the upstream rank's hidden-state activation,
            # or -- on a first rank with input_is_embedded -- this chunk's own embedding. Either way
            # there is nothing to gather.
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
                record_dev=record_dev,
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
                    if indexer_indices is not None:
                        ttnn.deallocate(indexer_indices)
                    indexer_indices = new_idx
            else:
                h, _ = ret
            signpost(f"forward_layer_{i}_end")
            if self.kv_only_last_layer and i == len(self.layers) - 1:
                # Last layer was kv-only — KV cache filled, migration callback
                # fired, no hidden state flowing forward. Skip norm + lm_head +
                # sample; no first_token to produce.
                return None, None, intermediates
            if return_intermediates:
                ttnn.synchronize_device(self.mesh_device)
                intermediates[f"layer_{i}"] = self._to_host(h)
            if read_profiler:
                ttnn.ReadDeviceProfiler(self.mesh_device)
        # GLM-5.2 reuse: free the last full layer's held top-k indices after the final layer.
        if reuse and indexer_indices is not None:
            ttnn.deallocate(indexer_indices)

        # Non-last pipeline ranks stop here: the layer slice's output activation is
        # handed to the next rank, which continues from this hidden state. The norm /
        # LM-head / sample tail (and its weights) live only on the last rank.
        if not self.is_last_rank:
            return h

        h = self.norm(h)

        if return_intermediates:
            ttnn.synchronize_device(self.mesh_device)
            intermediates["norm"] = self._to_host(h)

        # LM Head: extract logits for last real token
        logits_host, first_token_logits = self._lm_head_and_extract(h, actual_isl)

        if return_intermediates:
            intermediates["lm_head"] = logits_host
            intermediates["logits"] = first_token_logits

        # Reorder intermediates if balanced. Skip reordering for logits and lm_head in zigzag mode.
        no_reorder_keys = {"logits", "lm_head"}
        if return_intermediates and self.is_balanced:
            for key, tensor in intermediates.items():
                if key in no_reorder_keys:
                    logger.debug(f"Skipping reordering for non-sequence intermediate {key}")
                    continue
                if isinstance(tensor, torch.Tensor):
                    logger.debug(f"Reordering intermediate {key} with shape {tensor.shape}")
                    intermediates[key] = reverse_reorder_tensor_chunks(tensor, self.chunk_order, seq_dim=-2)
                else:
                    logger.debug(f"Skipping reordering for intermediate {key} of type {type(tensor)}")

        # Sample token(s) from logits
        first_token_id, first_token_prob, sweep_results = self._sample(first_token_logits, actual_isl, temperature)

        if return_intermediates:
            intermediates["first_token"] = sweep_results

        # --- MTP levels (GLM-5.2, #53533) ---------------------------------------------------
        # After the trunk tail, so the trunk path is byte-identical when MTP is off. `h` is h^0
        # (post-model.norm) and is still live: neither the LM head nor _sample frees it.
        if mtp_tokens is not None or mtp_union is not None:
            assert self.mtp_predictor is not None, "MTP tokens passed but this transformer has no mtp_predictor"
            assert actual_start is not None, (
                "MTP needs actual_start on the host to know whether this chunk contains absolute "
                "position 0, where vLLM zeroes the embedding on every level; the on-device metadata "
                "path keeps actual_start on device and cannot answer that here"
            )
            # d2h_service / record_dev / on_layer_complete / on_layer_hidden are deliberately NOT
            # forwarded: the layer-ack protocol counts TRUNK layers, so K extra acks would be K
            # records the producer never asked for, and on_layer_hidden's index would collide with
            # the trunk's. MTP's product is the KV it writes, not an ack.
            mtp_out, mtp_generated = self.run_mtp(
                h,
                kvpe_cache,
                rope_tensors,
                mtp_tokens,
                actual_isl,
                zero_position_0=(actual_start == 0),
                seed_token=mtp_seed_token,
                union=mtp_union,
                is_last_chunk=is_last_chunk,
                cache_user_id=cache_user_id,
                actual_start=actual_start,
                actual_end=actual_end,
                padding_side=self.padding_side,
                index_kv_cache=index_kv_cache,
                metadata=metadata,
            )
            if on_mtp_complete is not None:
                on_mtp_complete(mtp_out, mtp_generated)

        return first_token_id, first_token_prob, intermediates

    def _lm_head_and_extract(
        self,
        h: ttnn.Tensor,
        actual_isl: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run LM head and extract last-token logits. Topology-aware.

        Args:
            h: Hidden states after final norm
            actual_isl: Count of real tokens in the sequence

        Returns:
            Tuple of (logits_host, first_token_logits)
        """
        if self.padding_side == "right":
            global_token_id = actual_isl - 1
        else:  # "left"
            global_token_id = self.seq_len - 1

        logits, (device_id, token_offset) = self.lm_head(h, global_token_id)

        logits_host = self.lm_head.logit_to_host(logits, device_id)
        assert (
            logits_host.shape[-1] == self.lm_head.vocab_size
        ), f"Expected full vocab {self.lm_head.vocab_size}, got {logits_host.shape[-1]} — TP concat may be broken"
        first_token_logits = self.lm_head.select_first_token(logits_host, token_offset)

        logger.debug(f"[TtPrefillTransformer._extract] {logits.shape}")
        logger.debug(f"[TtPrefillTransformer._extract] {logits_host.shape}")
        logger.debug(f"[TtPrefillTransformer._extract] {first_token_logits.shape}")

        return logits_host, first_token_logits

    # ----------------------------------------------------------------------------------------------
    # MTP (GLM-5.2, issue #53533)
    # ----------------------------------------------------------------------------------------------

    def _mtp_position_zero_mask(self) -> ttnn.Tensor:
        """Cached per-chip ``[1, 1, L, H/tp]`` mask zeroing ABSOLUTE position 0. Built once, reused."""
        if self._mtp_pos0_mask is None:
            self._mtp_pos0_mask = build_position_zero_mask(
                self.mesh_device,
                self.sp_factor,
                self.seq_len,
                self.is_balanced,
                self.mesh_shape,
                self.sp_axis,
                emb_dim_per_chip=self.emb_dim_per_chip,
            )
        return self._mtp_pos0_mask

    def _mtp_embed_window(self, window_ids: list[int], zero_position_0: bool) -> ttnn.Tensor:
        """Shard, upload and embed ONE MTP shift-window. Returns ``[1, 1, L, H/tp]`` TILE_LAYOUT.

        The window is sharded with THIS chunk's ``is_balanced``, and that is what makes the shift
        correct: sharding is a fixed row -> position permutation applied to the window's *contents*,
        so applying the trunk's permutation to a window shifted by ``k`` lands ``t_{p+k}`` on the row
        whose hidden sits at ``p``. Shifting tokens rather than embeddings is also what keeps it
        free -- the token tensor is ROW_MAJOR uint32, so a row offset costs nothing, while the
        embedding is TILE_LAYOUT and ``k`` in 1..4 is never 32-row aligned.

        The sequence axis does not grow: ``+K`` is a token-window overhang, not extra rows of
        compute. Every window is ``L`` rows per chip, exactly like the trunk input.
        """
        assert (
            len(window_ids) == self.seq_len
        ), f"MTP window is {len(window_ids)} ids, expected the padded chunk length {self.seq_len}"
        tt_ids = prepare_prefill_mtp_window(
            window_ids, self.mesh_device, self.sp_factor, self.is_balanced, self.mesh_shape, self.sp_axis
        )
        return self._mtp_embed_window_dev(tt_ids, zero_position_0)

    def mtp_embed_ids(self, tt_ids: ttnn.Tensor) -> ttnn.Tensor:
        """Gather ``[sp, 1, N]`` uint32 ids into ``[1, 1, N, H/tp]`` bf16 TILE. Does NOT consume
        ``tt_ids``, and does NOT mask position 0 -- masking is per WINDOW, and the runner path gathers
        the union's blocks (``MTPUnionEmbedding.from_ids``), which span more rows than any window.
        Public because that is the one thing the runner needs off this object: with the same weights
        and the same ids it produces exactly what ``forward``'s first-rank embed does, which is what
        lets that gather double as the model's input (``input_is_embedded``)."""
        return ttnn.unsqueeze_to_4D(self.embed(tt_ids))

    def _mtp_mask_position_zero(self, emb: ttnn.Tensor, zero_position_0: bool) -> ttnn.Tensor:
        """Zero the row at ABSOLUTE position 0, on a ``[1, 1, L, H/tp]`` window. Consumes ``emb``.

        Applied to every level, per vLLM. No-op unless this chunk contains position 0.
        """
        if not zero_position_0:
            return emb
        masked = ttnn.multiply(emb, self._mtp_position_zero_mask())
        ttnn.deallocate(emb)
        return masked

    def _mtp_embed_window_dev(self, tt_ids: ttnn.Tensor, zero_position_0: bool) -> ttnn.Tensor:
        """Embed ONE already-on-device MTP window: ``[sp, 1, L]`` uint32 -> ``[1, 1, L, H/tp]`` TILE,
        masked. The device half of :meth:`_mtp_embed_window`; consumes ``tt_ids``."""
        emb = self.mtp_embed_ids(tt_ids)
        ttnn.deallocate(tt_ids)
        return self._mtp_mask_position_zero(emb, zero_position_0)

    def _mtp_next_token_fn(self, actual_isl: int):
        """``H^k -> int``: the greedy token at the last real row, through the trunk's own LM head.

        Greedy (argmax), not :meth:`_sample`: the MTP chain is a draft, the reference is argmax, and
        sampling here would make level ``k+1``'s *input* depend on the sampling temperature. It is
        the same head and the same row the trunk already used, so at temperature 0 the two agree by
        construction -- which is why ``seed_token`` is an optimisation and not the definition.

        Cost is one extra 32-row LM head call per level: ``TtLMHead.forward`` narrows to the single
        tile containing the target row before the vocab matmul.
        """

        def next_token(h_normed):
            _, logits = self._lm_head_and_extract(h_normed, actual_isl)
            flat = logits.reshape(-1)
            assert (
                flat.numel() == self.lm_head.vocab_size
            ), f"expected full-vocab logits, got {flat.numel()} of {self.lm_head.vocab_size}"
            return int(torch.argmax(flat).item())

        return next_token

    def run_mtp(
        self,
        h_normed: ttnn.Tensor,
        kvpe_cache: MlaKvCache,
        rope_tensors: dict,
        mtp_tokens: Optional[list[int]],
        actual_isl: int,
        *,
        zero_position_0: bool,
        seed_token: Optional[int] = None,
        union=None,
        is_last_chunk: bool = False,
        **fwd_kwargs,
    ):
        """Run the K MTP levels off ``h^0``. Returns ``(MTPPredictorOutput, generated_tokens)``.

        Args:
            h_normed: ``h^0`` -- the trunk output AFTER ``model.norm``.
            mtp_tokens: this chunk's ``C + K`` extended token stream from
                ``mtp_extended_stream``. Interior chunks carry the next chunk's first ``K`` ids;
                the last chunk carries ``K`` generation slots instead, filled level by level from
                each level's own LM head. None when ``union`` is given.
            union: ``MTPUnionEmbedding`` — the on-device alternative to ``mtp_tokens``, where each
                level's embedding is a row slice of one already-gathered tensor rather than a host
                list uploaded and gathered per level. It never runs the LM head, so the last chunk's
                ``K`` trailing positions carry whatever the producer padded them with and
                ``generated_tokens`` comes back empty.
            actual_isl: this chunk's real-token count -- both the LM-head row and where the last
                chunk's generation slots start.
            zero_position_0: True only on the chunk containing absolute position 0.
            is_last_chunk: True only on the chunk that ends the request, where the prompt has no ids
                past ``actual_end``. Carried here for the levels to use; see #53533.
            seed_token: ``t_P`` if the caller already has it. Optional; omitting it costs one 32-row
                LM head call and removes any coupling to the trunk's sampling temperature.
            fwd_kwargs: passed to every level's block. The KV-cache slot is NOT among them -- the
                predictor owns ``cache_layer_idx``, writing level ``k`` (0-based) to
                ``first_cache_slot + k``, so the caller's cache must have ``num_layers + K`` slots
                per user and every block in the model must have been built with the same
                ``layer_num``, since the flat slot is ``cache_user_id * layer_num + cache_layer_idx``.
        """
        assert self.mtp_predictor is not None, "run_mtp called on a transformer built without an mtp_predictor"
        assert (mtp_tokens is None) != (union is None), "run_mtp takes exactly one of mtp_tokens/union"
        if union is not None:
            source = MTPDeviceEmbedSource(union, mask_fn=lambda emb: self._mtp_mask_position_zero(emb, zero_position_0))
        else:
            source = MTPEmbedSource(
                mtp_tokens,
                self.seq_len,
                self.num_mtp_levels,
                embed_fn=lambda window: self._mtp_embed_window(window, zero_position_0),
                next_token_fn=self._mtp_next_token_fn(actual_isl),
                real_len=actual_isl,
                seed_token=seed_token,
            )
        # Forwarded here rather than by the caller: `actual_isl` is a named parameter of this
        # method AND something every level's block needs, so a caller that passed both would hit
        # "got multiple values for argument 'actual_isl'". It cannot already be in fwd_kwargs --
        # a keyword of that name binds to the parameter above, never to **fwd_kwargs.
        fwd_kwargs["actual_isl"] = actual_isl
        out = self.mtp_predictor.forward(source, h_normed, rope_tensors, kvpe_cache, **fwd_kwargs)
        return out, source.generated_tokens

    def _sample(
        self,
        first_token_logits: torch.Tensor,
        actual_isl: int,
        temperature: Union[float, list[float]],
    ) -> tuple[int, float, list[dict]]:
        """Sample token(s) from extracted logits with temperature sweep.

        Args:
            first_token_logits: Logits for the last real token position
            actual_isl: Count of real tokens (stored in results)
            temperature: Temperature for sampling (single float or list for sweep)

        Returns:
            Tuple of (first_token_id, first_token_prob, sweep_results)
        """
        temperatures = temperature if isinstance(temperature, list) else [temperature]

        sweep_results = []
        for temp in temperatures:
            token_id, token_prob, top5 = self._sample_token(first_token_logits.clone(), temp)
            sweep_results.append(
                {
                    "actual_isl": actual_isl,
                    "token_id": token_id,
                    "probability": token_prob,
                    "temperature": temp,
                    "top5": top5,
                }
            )

        first_token_id = sweep_results[0]["token_id"]
        first_token_prob = sweep_results[0]["probability"]

        logger.debug(f"[TtPrefillTransformer._sample] {first_token_id=}, {first_token_prob=:.4f}")

        return first_token_id, first_token_prob, sweep_results

    def _sample_token(self, logits: torch.Tensor, temperature: float = 1.0) -> tuple[int, float, list]:
        """
        Sample token from logits with temperature scaling.

        Uses Gumbel-softmax trick for sampling (same as DeepSeek reference).
        https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/generate.py

        Args:
            logits: Logits tensor for a single token position
            temperature: Temperature for scaling (0.0 = argmax)

        Returns:
            Tuple of (sampled_token_id, probability, top5_list)
            where top5_list is [{token_id, probability}, ...]
        """
        probs = torch.softmax(logits.float(), dim=-1)

        # Get top-5 tokens (unscaled)
        top5_probs, top5_ids = torch.topk(probs.flatten(), k=5)
        top5 = [{"token_id": tid.item(), "probability": tprob.item()} for tid, tprob in zip(top5_ids, top5_probs)]

        if temperature <= 0:
            # Deterministic argmax — no Gumbel noise
            sampled_id = probs.argmax(dim=-1)
            prob = probs.flatten()[sampled_id.item()].item()
            return sampled_id.item(), prob, top5

        logits = logits / temperature
        probs = torch.softmax(logits.float(), dim=-1)

        # Recompute top-5 with temperature-scaled probs
        top5_probs, top5_ids = torch.topk(probs.flatten(), k=5)
        top5 = [{"token_id": tid.item(), "probability": tprob.item()} for tid, tprob in zip(top5_ids, top5_probs)]

        # Gumbel-softmax trick for sampling (use non-in-place to preserve probs)
        gumbel = probs / torch.empty_like(probs).exponential_(1)
        sampled_id = gumbel.argmax(dim=-1)
        prob = probs.flatten()[sampled_id.item()].item()
        return sampled_id.item(), prob, top5
