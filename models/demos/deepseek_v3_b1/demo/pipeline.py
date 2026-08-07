# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Pipeline orchestration: configuration, factory functions, Pipeline.
Stage kinds (Embedding, LMHead, Passthrough) live in stage.py; MoE/dense decoder in decoder_stage.py.
"""

from __future__ import annotations

from typing import Any, Callable

from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.demo.decoder_stage import DenseDecoderStage, MoEDecoderStage
from models.demos.deepseek_v3_b1.demo.pipeline_routing import LocalStageSocketPlan, StageRouting
from models.demos.deepseek_v3_b1.demo.stage import (
    DEFAULT_ACTIVATION_FIFO_PAGES,
    BaseLMHeadStage,
    EmbeddingStage,
    PassthroughPayload,
    PassthroughStage,
    SpecLMHeadWithEmbeddingStage,
    StageContext,
    StageKind,
)
from models.demos.deepseek_v3_b1.demo.stage_family import StageFamily
from models.demos.deepseek_v3_b1.demo.weight_provider import WeightProvider
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import PipelineBlockKind, StageMetadata


def create_fabric_router_config(max_payload_size: int) -> Any:
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def create_passthrough_pipeline_configuration(
    weight_provider: WeightProvider,
    num_procs: int,
    *,
    host_loopback: bool = False,
) -> PipelineConfiguration:
    """N-stage pipeline: stage 0 is :class:`EmbeddingStage`; stages 1..N-1 are :class:`PassthroughStage`.

    The embedding forwards its row plus the DeepseekMetadata struct (the real-model config), so the
    ring carries ``ACTIVATION_W_TOKEN_META`` end to end.

    ``host_loopback`` selects the return path: ``False`` uses fabric loopback (D2H on stage 0);
    ``True`` uses MPI loopback (D2H on the last stage). With ``False``, the synthetic-weights smoke
    hits a D2H pinned-buffer address anomaly under investigation (see
    ``docs/d2h_socket_loopback_pinned_address_bug.md``), so the smoke prefers ``host_loopback=True``.

    Stage indices 0 .. num_procs-1 must match the distributed mesh count (e.g. 4, 16, 64).
    """
    if num_procs < 1:
        raise ValueError(f"num_procs must be >= 1, got {num_procs}")

    def stage_0(device: ttnn.MeshDevice) -> StageKind:
        return EmbeddingStage(
            weight_provider.load_embedding(device),
            loopback_payload=PassthroughPayload.ACTIVATION_W_TOKEN_META,
            host_loopback=host_loopback,
        )

    stage_factories: dict[int, Callable[[ttnn.MeshDevice], StageKind]] = {0: stage_0}
    for i in range(1, num_procs):
        stage_factories[i] = lambda _d: PassthroughStage(
            PassthroughPayload.ACTIVATION_W_TOKEN_META, host_loopback=host_loopback
        )
    return PipelineConfiguration(stage_factories)


def create_single_galaxy_deepseek_pipeline_configuration(
    weight_provider: WeightProvider,
    *,
    lm_head_fp32_dest_acc_en: bool = True,
    lm_head_persistent_mode: bool = True,
    host_loopback: bool = True,
) -> PipelineConfiguration:
    """4-stage single-galaxy: Embed -> LMHead -> Passthrough(TOKEN) -> Passthrough(TOKEN).

    When ``host_loopback=True``, the last-stage token is returned to rank 0 via host MPI
    (``send_bytes``/``recv_bytes``) instead of a fabric loopback kernel.
    """

    def stage_0(device: ttnn.MeshDevice) -> StageKind:
        return EmbeddingStage(
            weight_provider.load_embedding(device),
            host_loopback=host_loopback,
        )

    def stage_3(device: ttnn.MeshDevice) -> StageKind:
        return BaseLMHeadStage(
            weights=weight_provider.load_lm_head(device),
            fp32_dest_acc_en=lm_head_fp32_dest_acc_en,
            persistent_mode=lm_head_persistent_mode,
        )

    return PipelineConfiguration(
        {
            0: stage_0,
            1: stage_3,
            2: lambda d: PassthroughStage(PassthroughPayload.TOKEN, host_loopback=host_loopback),
            3: lambda d: PassthroughStage(PassthroughPayload.TOKEN, host_loopback=host_loopback),
        }
    )


def create_spec_decode_pipeline_configuration(
    weight_provider: WeightProvider,
    *,
    fp32_dest_acc_en: bool = True,
    persistent_mode: bool = True,
    num_mtp_levels: int = 1,
) -> PipelineConfiguration:
    """Unified spec-decode pipeline that works for any process count.

    Auto-detects the number of live processes and places stages accordingly:

      P0:                      SpecLMHead+Embed  (terminal verify, mtp_level=N)
      P1  .. P_N:              BaseLMHead+MTP    (mtp_level=0 .. N-1)
      P_{N+1} .. P_{procs-1}:  Passthrough       (decoder stand-ins)

    Limits:
      single galaxy  (4 procs) → up to 3 MTP levels
      single pod    (16 procs) → up to 4 MTP levels

    ``generate_blitz_decode_pipeline`` assigns fabric entry/exit nodes by **MPI rank index**
    (``pipeline_config[rank]``). Base LMHead stages must occupy consecutive ranks
    ``1 .. num_mtp_levels``; inserting passthrough before the first base (e.g. base only at
    rank 2) is a different topology and is not supported by this factory.
    """
    num_procs = int(ttnn.distributed_context_get_size())
    max_mtp = min(num_procs - 1, 4)
    assert 1 <= num_mtp_levels <= max_mtp, (
        f"num_mtp_levels={num_mtp_levels} out of range [1, {max_mtp}] " f"for {num_procs} processes"
    )

    def stage_0(device: ttnn.MeshDevice) -> StageKind:
        return SpecLMHeadWithEmbeddingStage(
            embedding_weights=weight_provider.load_embedding(device),
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
            spec_weights=weight_provider.load_spec(device),
            mtp_level=num_mtp_levels,
        )

    def base_lm_head_factory(level: int):
        def factory(device: ttnn.MeshDevice) -> StageKind:
            if level == 0:
                lm_head_weights = weight_provider.load_lm_head(device)
            else:
                lm_head_weights = weight_provider.load_spec(device).as_lm_head_weights()
            return BaseLMHeadStage(
                weights=lm_head_weights,
                fp32_dest_acc_en=fp32_dest_acc_en,
                persistent_mode=persistent_mode,
                mtp_weights=weight_provider.load_mtp(device),
                embedding_weights=weight_provider.load_embedding(device),
                mtp_level=level,
            )

        return factory

    # Slot k (1 <= k <= num_mtp_levels) runs the BaseLMHead at MTP level k-1.
    # Remaining slots [num_mtp_levels + 1 .. num_procs - 1] are passthroughs.
    stage_factories: dict[int, Callable[[ttnn.MeshDevice], StageKind]] = {0: stage_0}
    for slot in range(1, num_mtp_levels + 1):
        stage_factories[slot] = base_lm_head_factory(slot - 1)
    for slot in range(num_mtp_levels + 1, num_procs):
        stage_factories[slot] = lambda _d: PassthroughStage(PassthroughPayload.ACTIVATION_W_TOKEN_META)
    return PipelineConfiguration(stage_factories)


def create_single_pod_pipeline_configuration(
    weight_provider: WeightProvider,
    *,
    fp32_dest_acc_en: bool = True,
    persistent_mode: bool = True,
    enable_mtp: bool = False,
    dense_layer_id_override: int | None = None,
    moe_layer_id_override: int | None = None,
) -> PipelineConfiguration:
    """16-stage single-pod: Embed -> Dense(0,1,2) -> Decoder(3..12) -> LMHead -> Token fwd.

    If dense_layer_id_override is set (e.g. 0), all dense stages use that layer id.
    If moe_layer_id_override is set (e.g. 3), all decoder stages use that layer id.
    """
    fwd_payload = PassthroughPayload.ACTIVATION_W_TOKEN_META if enable_mtp else PassthroughPayload.TOKEN

    def stage_0(device: ttnn.MeshDevice) -> StageKind:
        return EmbeddingStage(weight_provider.load_embedding(device))

    def stage_14(device: ttnn.MeshDevice) -> StageKind:
        mtp_weights = weight_provider.load_mtp(device) if enable_mtp else None
        return BaseLMHeadStage(
            weights=weight_provider.load_lm_head(device),
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
            mtp_weights=mtp_weights,
            embedding_weights=weight_provider.load_embedding(device) if enable_mtp else None,
        )

    def _dense_stage(layer_id: int):
        return lambda d: DenseDecoderStage(
            weights=weight_provider.load_dense_layer(layer_id=layer_id, device=d),
            layer_idx=layer_id,
        )

    def _decoder_stage(
        layer_id: int,
        *,
        upstream_fifo_pages: int = DEFAULT_ACTIVATION_FIFO_PAGES,
        downstream_fifo_pages: int = DEFAULT_ACTIVATION_FIFO_PAGES,
    ):
        return lambda d: MoEDecoderStage(
            weights=weight_provider.load_moe_layer(layer_id=layer_id, device=d),
            layer_idx=layer_id,
            upstream_fifo_pages=upstream_fifo_pages,
            downstream_fifo_pages=downstream_fifo_pages,
        )

    dense_ids = (dense_layer_id_override,) * 3 if dense_layer_id_override is not None else (0, 1, 2)
    moe_layer_id = moe_layer_id_override

    stage_factories: dict[int, Callable[[ttnn.MeshDevice], StageKind]] = {
        0: stage_0,
        1: _dense_stage(dense_ids[0]),
        2: _dense_stage(dense_ids[1]),
        3: _dense_stage(dense_ids[2]),
        **{i: _decoder_stage(moe_layer_id if moe_layer_id is not None else i - 1) for i in range(4, 14)},
        14: stage_14,
        15: lambda d: PassthroughStage(fwd_payload),
    }
    return PipelineConfiguration(stage_factories)


def create_single_pod_spec_decode_pipeline_configuration(
    weight_provider: WeightProvider,
    *,
    fp32_dest_acc_en: bool = True,
    persistent_mode: bool = True,
    enable_mtp: bool = False,
    dense_layer_id_override: int | None = None,
    moe_layer_id_override: int | None = None,
    num_slots: int = 64,
    enable_sram_bspm: bool = False,
    num_mtp_levels: int = 1,
) -> PipelineConfiguration:
    """16-stage single-pod spec-decode pipeline.

    Layout (N = num_mtp_levels):
      P0:                        SpecLMHead+Embed  (terminal verify, mtp_level=N)
      P1  .. P3:                 Dense decoders
      P4  .. P_{16-2N-1}:        MoE decoders
      [BaseLMHead(k) + MTP decoder] × N   for k = 0 .. N-1

    Each MTP level occupies a pair of consecutive stages: a BaseLMHead that
    runs LM-head sampling + MTP EH matmul, followed by the MTP decoder that
    produces the hidden state for the next level.
    """
    _MTP_DECODER_LAYER_IDX = 61
    assert 0 <= num_mtp_levels <= 4, f"num_mtp_levels={num_mtp_levels} out of range [0, 4]"

    def stage_0(device: ttnn.MeshDevice) -> StageKind:
        return SpecLMHeadWithEmbeddingStage(
            embedding_weights=weight_provider.load_embedding(device),
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
            spec_weights=weight_provider.load_spec(device)
            if num_mtp_levels > 0
            else weight_provider.load_lm_head(device),
        )

    def base_lm_head_factory(level: int):
        def factory(device: ttnn.MeshDevice) -> StageKind:
            if level == 0:
                lm_head_weights = weight_provider.load_lm_head(device)
            else:
                lm_head_weights = weight_provider.load_spec(device).as_lm_head_weights()
            return BaseLMHeadStage(
                weights=lm_head_weights,
                fp32_dest_acc_en=fp32_dest_acc_en,
                persistent_mode=persistent_mode,
                mtp_weights=weight_provider.load_mtp(device),
                embedding_weights=weight_provider.load_embedding(device),
                mtp_level=level,
            )

        return factory

    def passthrough_stage(device: ttnn.MeshDevice) -> StageKind:
        return PassthroughStage(PassthroughPayload.ACTIVATION_W_TOKEN_META)

    def _dense_stage(layer_id: int):
        return lambda d: DenseDecoderStage(
            weights=weight_provider.load_dense_layer(layer_id=layer_id, device=d),
            layer_idx=layer_id,
            num_slots=num_slots,
        )

    def _decoder_stage(
        layer_id: int,
        *,
        upstream_fifo_pages: int = DEFAULT_ACTIVATION_FIFO_PAGES,
        downstream_fifo_pages: int = DEFAULT_ACTIVATION_FIFO_PAGES,
    ):
        return lambda d: MoEDecoderStage(
            weights=weight_provider.load_moe_layer(layer_id=layer_id, device=d),
            layer_idx=layer_id,
            num_slots=num_slots,
            upstream_fifo_pages=upstream_fifo_pages,
            downstream_fifo_pages=downstream_fifo_pages,
            enable_sram_bspm=enable_sram_bspm,
        )

    def mtp_stages_factory(level: int) -> dict[int, Callable[[ttnn.MeshDevice], StageKind]]:
        first_mtp_stage = 16 - 2 * num_mtp_levels
        base_idx = first_mtp_stage + 2 * level
        decoder_idx = base_idx + 1
        return {
            base_idx: base_lm_head_factory(level),
            decoder_idx: _decoder_stage(_MTP_DECODER_LAYER_IDX),
        }

    dense_ids = (dense_layer_id_override,) * 3 if dense_layer_id_override is not None else (0, 1, 2)
    moe_layer_id = moe_layer_id_override
    moe_end = 16 - 2 * num_mtp_levels

    stage_factories: dict[int, Callable[[ttnn.MeshDevice], StageKind]] = {
        0: stage_0,
        1: _dense_stage(dense_ids[0]),
        2: _dense_stage(dense_ids[1]),
        3: _dense_stage(dense_ids[2]),
        **{i: _decoder_stage(moe_layer_id if moe_layer_id is not None else i - 1) for i in range(4, moe_end)},
    }
    for level in range(num_mtp_levels):
        stage_factories.update(mtp_stages_factory(level))
    return PipelineConfiguration(stage_factories)


def create_single_galaxy_spec_decode_pipeline_configuration(
    weight_provider: WeightProvider,
    *,
    fp32_dest_acc_en: bool = True,
    persistent_mode: bool = True,
    num_mtp_levels: int = 1,
) -> PipelineConfiguration:
    """Deprecated: use :func:`create_spec_decode_pipeline_configuration` instead."""
    return create_spec_decode_pipeline_configuration(
        weight_provider,
        fp32_dest_acc_en=fp32_dest_acc_en,
        persistent_mode=persistent_mode,
        num_mtp_levels=num_mtp_levels,
    )


def create_sp4_pipeline_configuration(
    weight_provider: WeightProvider,
    *,
    fp32_dest_acc_en: bool = True,
    persistent_mode: bool = True,
    enable_mtp: bool = False,
    num_mtp_levels: int = 1,
    dense_layer_id_override: int | None = None,
    moe_layer_id_override: int | None = None,
    num_slots: int = 64,
    enable_sram_bspm: bool = False,
) -> PipelineConfiguration:
    """64-stage super-pod: Embed -> Dense(0,1,2) -> Decoder(3..60) -> LMHead -> Token fwd.

    If dense_layer_id_override is set (e.g. 0), all dense stages use that layer id.
    If moe_layer_id_override is set (e.g. 3), all decoder stages use that layer id.
    """
    fwd_payload = PassthroughPayload.ACTIVATION_W_TOKEN_META if enable_mtp else PassthroughPayload.TOKEN
    assert 0 <= num_mtp_levels <= 1, f"num_mtp_levels={num_mtp_levels} out of range [0, 1], SP4 only supports MTP-1"

    def stage_0(device: ttnn.MeshDevice) -> StageKind:
        return SpecLMHeadWithEmbeddingStage(
            embedding_weights=weight_provider.load_embedding(device),
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
            spec_weights=weight_provider.load_spec(device)
            if num_mtp_levels > 0
            else weight_provider.load_lm_head(device),
        )

    def stage_62(device: ttnn.MeshDevice) -> StageKind:
        mtp_weights = weight_provider.load_mtp(device) if enable_mtp else None
        return BaseLMHeadStage(
            weights=weight_provider.load_lm_head(device),
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
            mtp_weights=mtp_weights,
            embedding_weights=weight_provider.load_embedding(device),
        )

    def passthrough_stage(device: ttnn.MeshDevice) -> StageKind:
        return PassthroughStage(PassthroughPayload.ACTIVATION_W_TOKEN_META)

    def _dense_stage(layer_id: int):
        return lambda d: DenseDecoderStage(
            weights=weight_provider.load_dense_layer(layer_id=layer_id, device=d),
            layer_idx=layer_id,
            num_slots=num_slots,
        )

    def _decoder_stage(
        layer_id: int,
        *,
        upstream_fifo_pages: int = DEFAULT_ACTIVATION_FIFO_PAGES,
        downstream_fifo_pages: int = DEFAULT_ACTIVATION_FIFO_PAGES,
    ):
        return lambda d: MoEDecoderStage(
            weights=weight_provider.load_moe_layer(layer_id=layer_id, device=d),
            layer_idx=layer_id,
            num_slots=num_slots,
            upstream_fifo_pages=upstream_fifo_pages,
            downstream_fifo_pages=downstream_fifo_pages,
            enable_sram_bspm=enable_sram_bspm,
        )

    dense_ids = (dense_layer_id_override,) * 3 if dense_layer_id_override is not None else (0, 1, 2)
    moe_layer_id = moe_layer_id_override

    stage_factories: dict[int, Callable[[ttnn.MeshDevice], StageKind]] = {
        0: stage_0,
        1: _dense_stage(dense_ids[0]),
        2: _dense_stage(dense_ids[1]),
        3: _dense_stage(dense_ids[2]),
        **{i: _decoder_stage(moe_layer_id if moe_layer_id is not None else i - 1) for i in range(4, 62)},
    }
    if enable_mtp:
        stage_factories[61] = _decoder_stage(60)
    if num_mtp_levels == 1:
        stage_factories[62] = stage_62
        stage_factories[63] = _decoder_stage(61)
    else:
        stage_factories[62] = passthrough_stage
        stage_factories[63] = passthrough_stage
    return PipelineConfiguration(stage_factories)


def create_sp5_pipeline_configuration(
    weight_provider: WeightProvider,
    *,
    fp32_dest_acc_en: bool = True,
    persistent_mode: bool = True,
    enable_mtp: bool = False,
    num_mtp_levels: int = 1,
    dense_layer_id_override: int | None = None,
    moe_layer_id_override: int | None = None,
    num_slots: int = 64,
    num_procs: int = 80,
    passthrough_payload: PassthroughPayload = PassthroughPayload.ACTIVATION_W_TOKEN_META,
    enable_sram_bspm: bool = False,
) -> PipelineConfiguration:
    """Super-pod spec-decode pipeline parameterized by ``num_mtp_levels`` (N)."""
    _MTP_DECODER_LAYER_IDX = 61
    assert 0 <= num_mtp_levels <= 4, f"SP5 only supports num_mtp_levels in [0..4]; got num_mtp_levels={num_mtp_levels}"
    active_stages = 62 + 2 * num_mtp_levels
    assert num_procs >= active_stages, (
        f"num_procs={num_procs} is smaller than the active SP5 stage count "
        f"{active_stages} for num_mtp_levels={num_mtp_levels}"
    )

    def stage_0(device: ttnn.MeshDevice) -> StageKind:
        return SpecLMHeadWithEmbeddingStage(
            embedding_weights=weight_provider.load_embedding(device),
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
            spec_weights=weight_provider.load_spec(device),
            mtp_level=num_mtp_levels,
        )

    def base_lm_head_factory(level: int):
        def factory(device: ttnn.MeshDevice) -> StageKind:
            if level == 0:
                lm_head_weights = weight_provider.load_lm_head(device)
            else:
                lm_head_weights = weight_provider.load_spec(device).as_lm_head_weights()
            mtp_weights = weight_provider.load_mtp(device) if enable_mtp else None
            return BaseLMHeadStage(
                weights=lm_head_weights,
                fp32_dest_acc_en=fp32_dest_acc_en,
                persistent_mode=persistent_mode,
                mtp_weights=mtp_weights,
                embedding_weights=weight_provider.load_embedding(device),
                mtp_level=level,
            )

        return factory

    def _dense_stage(layer_id: int):
        return lambda d: DenseDecoderStage(
            weights=weight_provider.load_dense_layer(layer_id=layer_id, device=d),
            layer_idx=layer_id,
            num_slots=num_slots,
        )

    def _decoder_stage(
        layer_id: int,
        *,
        upstream_fifo_pages: int = DEFAULT_ACTIVATION_FIFO_PAGES,
        downstream_fifo_pages: int = DEFAULT_ACTIVATION_FIFO_PAGES,
    ):
        return lambda d: MoEDecoderStage(
            weights=weight_provider.load_moe_layer(layer_id=layer_id, device=d),
            layer_idx=layer_id,
            num_slots=num_slots,
            upstream_fifo_pages=upstream_fifo_pages,
            downstream_fifo_pages=downstream_fifo_pages,
            enable_sram_bspm=enable_sram_bspm,
        )

    def mtp_stages_factory(level: int) -> dict[int, Callable[[ttnn.MeshDevice], StageKind]]:
        base_idx = 62 + 2 * level
        decoder_idx = base_idx + 1
        return {
            base_idx: base_lm_head_factory(level),
            decoder_idx: _decoder_stage(_MTP_DECODER_LAYER_IDX),
        }

    dense_ids = (dense_layer_id_override,) * 3 if dense_layer_id_override is not None else (0, 1, 2)
    moe_layer_id = moe_layer_id_override

    stage_factories: dict[int, Callable[[ttnn.MeshDevice], StageKind]] = {
        0: stage_0,  # Embed + terminal verify LMHead (mtp_level=N)
        1: _dense_stage(dense_ids[0]),
        2: _dense_stage(dense_ids[1]),
        3: _dense_stage(dense_ids[2]),
        **{i: _decoder_stage(moe_layer_id if moe_layer_id is not None else i - 1) for i in range(4, 62)},
    }
    for level in range(num_mtp_levels):
        stage_factories.update(mtp_stages_factory(level))

    for slot in range(active_stages, num_procs):
        stage_factories[slot] = lambda _d, p=passthrough_payload: PassthroughStage(p)
    return PipelineConfiguration(stage_factories)


def create_pipeline_configuration_from_num_procs(
    num_procs: int,
    weight_provider: WeightProvider,
    *,
    fp32_dest_acc_en: bool = True,
    persistent_mode: bool = True,
    enable_mtp: bool = False,
    num_mtp_levels: int = 1,
    dense_layer_id_override: int | None = None,
    moe_layer_id_override: int | None = None,
    num_slots: int = 64,
    enable_sram_bspm: bool = False,
) -> PipelineConfiguration:
    """Pick topology from process count (4 -> single_galaxy, 16 -> single_pod, 64 -> sp4, 80 -> sp5).

    ``num_mtp_levels`` is consumed by the SP5 builder (80-proc, MTP in [2, 4]; trailing
    procs past ``62 + 2 * num_mtp_levels`` are filled with passthrough stages so the same
    80-proc allocation can run any supported ``num_mtp_levels``). Other topologies are
    MTP-1 (or non-MTP) and currently ignore this argument; the assertion below catches
    accidental misuse.
    """
    if num_procs == 4:
        if num_mtp_levels > 0:
            return create_single_galaxy_spec_decode_pipeline_configuration(
                weight_provider,
                fp32_dest_acc_en=fp32_dest_acc_en,
                persistent_mode=persistent_mode,
                num_mtp_levels=num_mtp_levels,
            )
        return create_single_galaxy_deepseek_pipeline_configuration(
            weight_provider,
            lm_head_fp32_dest_acc_en=fp32_dest_acc_en,
            lm_head_persistent_mode=persistent_mode,
        )
    if num_procs == 16:
        if num_mtp_levels > 0:
            assert enable_mtp, "16-proc pipeline currently requires enable_mtp=True and uses the spec decode topology"
        return create_single_pod_spec_decode_pipeline_configuration(
            weight_provider,
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
            enable_mtp=enable_mtp,
            dense_layer_id_override=dense_layer_id_override,
            moe_layer_id_override=moe_layer_id_override,
            num_slots=num_slots,
            enable_sram_bspm=enable_sram_bspm,
            num_mtp_levels=num_mtp_levels,
        )
    if num_procs == 64:
        assert num_mtp_levels == 1, "64-proc pipeline (SP4) is the MTP-1 topology and requires num_mtp_levels=1"
        return create_sp4_pipeline_configuration(
            weight_provider,
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
            enable_mtp=enable_mtp,
            num_mtp_levels=num_mtp_levels,
            dense_layer_id_override=dense_layer_id_override,
            moe_layer_id_override=moe_layer_id_override,
            num_slots=num_slots,
        )
    if num_procs == 80:
        return create_sp5_pipeline_configuration(
            weight_provider,
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
            enable_mtp=enable_mtp,
            num_mtp_levels=num_mtp_levels,
            dense_layer_id_override=dense_layer_id_override,
            moe_layer_id_override=moe_layer_id_override,
            num_slots=num_slots,
            enable_sram_bspm=enable_sram_bspm,
        )
    raise ValueError(f"Unsupported num_procs: {num_procs}")


def create_pipeline_configuration_from_stage_count(
    num_stages: int,
    stage_family: StageFamily,
    weight_provider: WeightProvider,
    *,
    fp32_dest_acc_en: bool = True,
    persistent_mode: bool = True,
    enable_mtp: bool = False,
    dense_layer_id_override: int | None = None,
    moe_layer_id_override: int | None = None,
    num_slots: int = 64,
    enable_sram_bspm: bool = False,
    num_mtp_levels: int = 1,
) -> PipelineConfiguration:
    """Pick the model role map from logical stage count and derived stage family."""

    if stage_family == StageFamily.STAGE_4X2 and num_stages in (4, 16, 64):
        return create_pipeline_configuration_from_num_procs(
            num_stages,
            weight_provider,
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
            enable_mtp=enable_mtp,
            dense_layer_id_override=dense_layer_id_override,
            moe_layer_id_override=moe_layer_id_override,
            num_slots=num_slots,
            enable_sram_bspm=enable_sram_bspm,
            num_mtp_levels=num_mtp_levels,
        )

    return create_passthrough_pipeline_configuration(weight_provider, num_stages)


class PipelineConfiguration:
    """Maps stage IDs to stage factories. Each host builds only its stage (lazy weights)."""

    def __init__(
        self,
        stage_factories: dict[int, Callable[[ttnn.MeshDevice], StageKind]],
    ) -> None:
        self._stage_factories = stage_factories

    @property
    def num_stages(self) -> int:
        return len(self._stage_factories)

    def build_pipeline(
        self,
        mesh_device: ttnn.MeshDevice,
        my_stage_idx: int | None = None,
        stages_metadata: dict[int, StageMetadata | StageRouting] | None = None,
        pipeline_config: list | None = None,
        stage_plan: LocalStageSocketPlan | None = None,
        host_loopback: bool = False,
    ) -> Pipeline:
        """Create a Pipeline for this process's stage.

        Args:
            mesh_device: The MeshDevice (or submesh) for this stage.
            my_stage_idx: Which stage this process runs. Defaults to
                ``mesh_device.get_system_mesh_id()`` for backwards compatibility.
            stages_metadata: Per-stage rank/mesh_id routing info.
            pipeline_config: List of PipelineConfigEntry (entry/exit coords per stage).
                Required when stages_metadata is provided.
            host_loopback: If True, generate a host-loopback pipeline config (no fabric loopback).
        """
        if my_stage_idx is None:
            my_stage_idx = mesh_device.get_system_mesh_id()
        stage = self._stage_factories[my_stage_idx](mesh_device)

        if pipeline_config is not None and 0 <= my_stage_idx < len(pipeline_config):
            entry = pipeline_config[my_stage_idx].entry_node_coord
            exit_ = pipeline_config[my_stage_idx].exit_node_coord
            entry_t = tuple(int(entry[j]) for j in range(entry.dims()))
            exit_t = tuple(int(exit_[j]) for j in range(exit_.dims()))
            logger.info(
                "[TOPO P{}] stage_kind={} entry={} exit={}",
                my_stage_idx,
                type(stage).__name__,
                entry_t,
                exit_t,
            )
        return Pipeline(
            mesh_device,
            stage,
            my_stage_idx,
            stages_metadata=stages_metadata,
            pipeline_config=pipeline_config,
            stage_plan=stage_plan,
            host_loopback=host_loopback,
        )


class Pipeline:
    """Orchestrator for one pipeline stage with explicit 4-phase setup."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        stage_kind: StageKind,
        my_stage_idx: int,
        stages_metadata: dict[int, StageMetadata | StageRouting] | None = None,
        pipeline_config: list | None = None,
        stage_plan: LocalStageSocketPlan | None = None,
        host_loopback: bool = False,
    ) -> None:
        self._mesh_device = mesh_device
        self._stage_kind = stage_kind
        self._my_stage_idx = my_stage_idx
        if stages_metadata is not None:
            assert pipeline_config is not None, "pipeline_config required when stages_metadata is provided"
            self._pipeline_config = pipeline_config
        else:
            # Wait for all ranks to finish weight loading before entering distributed topology generation.
            ttnn.distributed_context_barrier()
            self._pipeline_config = ttnn._ttnn.multi_device.experimental.generate_blitz_decode_pipeline(
                not host_loopback
            )
        self._ctx = StageContext(
            mesh_device=mesh_device,
            pipeline_config=self._pipeline_config,
            my_stage_idx=self._my_stage_idx,
            stages_metadata=stages_metadata,
            stage_plan=stage_plan,
        )
        self._pipeline_block: PipelineBlockKind | None = None

    @property
    def my_mesh_id(self) -> int:
        """Backwards-compatible alias for my_stage_idx."""
        return self._my_stage_idx

    @property
    def my_stage_idx(self) -> int:
        return self._my_stage_idx

    @property
    def _block(self) -> PipelineBlockKind:
        """Non-null view of ``_pipeline_block``; raises if :meth:`configure_block` hasn't run."""
        if self._pipeline_block is None:
            raise RuntimeError("Pipeline.configure_block() must be called first")
        return self._pipeline_block

    def configure_block(self) -> None:
        """Phase 1: Create the PipelineBlock (socket wiring)."""
        self._pipeline_block = self._stage_kind.create_pipeline_block(self._ctx)

    def setup(self) -> None:
        """Phase 2: Allocate tensors, weights, semaphores on device.

        Decoder/dense stages also build :meth:`DecoderBlock.get_program_context` here so
        program construction finishes before :meth:`start_pipeline`.
        """
        self._stage_kind.setup(self._ctx, self._block)

    def start_pipeline(self) -> None:
        """Phase 3: Start pipeline block kernels (socket interfaces + auxiliary bypass sockets)."""
        self._block.run()
        self._stage_kind.run_auxiliary_sockets()

    def start_compute(self) -> None:
        """Phase 4: Launch stage compute (e.g. ``LMHeadSampling.op``, ``DecoderBlock.execute``)."""
        self._stage_kind.launch_compute(self._ctx, self._block)

    def setup_and_run(self) -> None:
        """Run all four phases in order."""

        self.barrier()  # Synchronize before socket creation — stage 0 may be slow due to weight loading
        logger.info("Configuring block")
        self.configure_block()

        logger.info("Setting up")
        self.setup()
        logger.info("Pipeline setup complete, waiting for all stages to complete...")
        self.barrier()

        logger.info("Starting pipeline")
        self.start_pipeline()
        logger.info("Pipeline started, waiting for all stages to complete...")
        self.barrier()

        logger.info("Starting compute")
        self.start_compute()
        logger.info("Compute started, waiting for all stages to complete...")
        self.barrier()

    def write_token(self, token_tensor: ttnn.Tensor) -> None:
        self._block.write_token(token_tensor)

    def read_output(self, output_tensor: ttnn.Tensor) -> object | None:
        # Returns the block's result. The fabric/D2H path writes into output_tensor and returns
        # None; the host-loopback path returns the MPI-received tensor on the first stage.
        return self._block.read_output(output_tensor)

    def export_host_socket_descriptors(self, prefix: str) -> None:
        self._block.export_host_socket_descriptors(prefix)

    def barrier(self) -> None:
        ttnn.distributed_context_barrier()

    def terminate(self) -> None:
        """Terminate the pipeline block.

        Compute kernels and the d2d_exchange / host_io kernels are independent
        persistent programs that communicate over sockets; each has its own
        termination semaphore.

        The shutdown sequence is:

          1. Barrier — all ranks start teardown together.
          2. Set the compute-kernel termination semaphore.  The flag is written
             to L1 but no core observes it yet: they are all blocked at the
             iteration gate (``persistent_next_iter_sem``) or mid-iteration.
          3. Barrier — all ranks have written their termination flags.
          4. Stage 0 pushes a dummy token; the D2H-owning stage drains the
             round-trip result (stage 0 for fabric loopback, the last stage for
             host/no loopback). The token naturally triggers
             ``persistent_next_iter_sem`` via the pipeline's socket/d2d flow,
             providing both the gate release AND the data payload.  All cores
             complete this final iteration together, loop back to the
             top-of-loop termination check, see the flag, and break.
          5. Barrier — non-stage-0 ranks wait for the dummy round-trip.
             (No ``synchronize_device`` here: d2d/host_io kernels are still
             running and would block.)
          6. Tear down d2d_exchange / host_io via ``PipelineBlock.terminate``
             (sets socket termination semaphores + ``synchronize_device``).
          7. Final ``synchronize_device``.
        """
        if self._pipeline_block is None:
            return

        ttnn.distributed_context_barrier()

        self._stage_kind.terminate(self._ctx, self._pipeline_block)
        self._stage_kind.terminate_auxiliary()

        ttnn.distributed_context_barrier()

        # Push the wake-up dummy token only from the rank that owns host ingress (the H2D socket).
        # is_first_pipeline_stage() is keyed on logical stage 0, which on a split stage 0 is true for
        # both halves — but only the H2D-owning half can push (push_dummy_token asserts on the socket).
        # h2d_socket is None on every other rank, so this targets the ingress owner precisely.
        if self._pipeline_block.h2d_socket is not None:
            self._pipeline_block.push_dummy_token()
        # Drain the dummy round-trip on whichever stage owns the D2H socket: stage 0 for fabric
        # loopback, the last stage for host/no loopback. drain_dummy_output no-ops without a D2H
        # socket, so calling it on every rank is safe. Without draining on the last stage, host
        # loopback leaves the dummy page unread, the D2H FIFO stays full, and the persistent D2H
        # kernel blocks in socket_reserve_pages (no termination check) — hanging teardown.
        self._pipeline_block.drain_dummy_output()

        ttnn.distributed_context_barrier()

        self._pipeline_block.terminate()
        ttnn.synchronize_device(self._mesh_device)
