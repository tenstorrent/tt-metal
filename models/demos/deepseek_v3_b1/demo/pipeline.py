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
from models.demos.deepseek_v3_b1.demo.stage import (
    ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES,
    DEFAULT_ACTIVATION_FIFO_PAGES,
    SINGLE_BUFFER_FIFO_PAGES,
    TOKEN_META_PAGE_SIZE_BYTES,
    BaseLMHeadStage,
    EmbeddingStage,
    PassthroughPayload,
    PassthroughStage,
    SpecLMHeadStage,
    SpecLMHeadWithEmbeddingStage,
    StageContext,
    StageKind,
)
from models.demos.deepseek_v3_b1.demo.weight_provider import WeightProvider
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import PipelineBlockKind, StageMetadata


def create_fabric_router_config(max_payload_size: int) -> Any:
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


LMHEAD_SPECIAL_FIFO_PAGES = SINGLE_BUFFER_FIFO_PAGES


def create_passthrough_pipeline_configuration(
    weight_provider: WeightProvider,
    num_procs: int,
    *,
    payload: PassthroughPayload = PassthroughPayload.ACTIVATION,
) -> PipelineConfiguration:
    """N-stage pipeline: stage 0 is :class:`EmbeddingStage`; stages 1..N-1 are :class:`PassthroughStage`.

    Default ``payload`` is :attr:`~PassthroughPayload.ACTIVATION` so D2D FIFO/page sizes match
    :class:`EmbeddingStage` downstream (embedding rows). Use :attr:`~PassthroughPayload.TOKEN` only
    when every stage before the passthrough chain emits token-sized pages (not this factory's
    embedding-first layout).

    Stage indices 0 .. num_procs-1 must match the distributed mesh count (e.g. 4, 16, 64).
    """
    if num_procs < 1:
        raise ValueError(f"num_procs must be >= 1, got {num_procs}")

    def stage_0(device: ttnn.MeshDevice) -> StageKind:
        return EmbeddingStage(
            weight_provider.load_embedding(device),
            loopback_payload=PassthroughPayload.ACTIVATION,
        )

    stage_factories: dict[int, Callable[[ttnn.MeshDevice], StageKind]] = {0: stage_0}
    for i in range(1, num_procs):
        stage_factories[i] = lambda _d, p=payload: PassthroughStage(p)
    return PipelineConfiguration(stage_factories)


def create_single_galaxy_pipeline_configuration(
    weight_provider: WeightProvider,
    *,
    fp32_dest_acc_en: bool = True,
    persistent_mode: bool = True,
    enable_mtp: bool = False,
) -> PipelineConfiguration:
    """4-stage single-galaxy: Embed -> LMHead -> Token fwd -> Token fwd."""
    fwd_payload = PassthroughPayload.ACTIVATION_W_TOKEN_META if enable_mtp else PassthroughPayload.TOKEN

    def stage_0(device: ttnn.MeshDevice) -> StageKind:
        return EmbeddingStage(
            weight_provider.load_embedding(device),
            d2h_page_size=ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES if enable_mtp else None,
            forward_metadata=True,  # BaseLMHeadStage always expects ACTIVATION_W_TOKEN_META upstream
        )

    def stage_1(device: ttnn.MeshDevice) -> StageKind:
        mtp_weights = weight_provider.load_mtp(device) if enable_mtp else None
        return BaseLMHeadStage(
            weights=weight_provider.load_lm_head(device),
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
            mtp_weights=mtp_weights,
            send_mtp_output_downstream=enable_mtp,
            embedding_weights=weight_provider.load_embedding(device) if enable_mtp else None,
            upstream_fifo_pages=LMHEAD_SPECIAL_FIFO_PAGES if enable_mtp else DEFAULT_ACTIVATION_FIFO_PAGES,
            downstream_fifo_pages=LMHEAD_SPECIAL_FIFO_PAGES if enable_mtp else DEFAULT_ACTIVATION_FIFO_PAGES,
        )

    return PipelineConfiguration(
        {
            0: stage_0,
            1: stage_1,
            2: lambda d: PassthroughStage(fwd_payload),
            3: lambda d: PassthroughStage(fwd_payload),
        }
    )


def create_single_galaxy_spec_decode_pipeline_configuration(
    weight_provider: WeightProvider,
    *,
    fp32_dest_acc_en: bool = True,
    persistent_mode: bool = True,
) -> PipelineConfiguration:
    """4-stage single-galaxy MTP speculative decoding pipeline:
    Embed -> LMHead+MTP -> Passthrough(ACTIVATION_W_TOKEN_META) -> Verify -> loopback(TOKEN_META)."""

    def stage_0(device: ttnn.MeshDevice) -> StageKind:
        return EmbeddingStage(
            weight_provider.load_embedding(device),
            d2h_page_size=TOKEN_META_PAGE_SIZE_BYTES,
            forward_metadata=True,
        )

    def stage_1(device: ttnn.MeshDevice) -> StageKind:
        return BaseLMHeadStage(
            weights=weight_provider.load_lm_head(device),
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
            mtp_weights=weight_provider.load_mtp(device),
            send_mtp_output_downstream=True,
            embedding_weights=weight_provider.load_embedding(device),
            upstream_fifo_pages=LMHEAD_SPECIAL_FIFO_PAGES,
            downstream_fifo_pages=LMHEAD_SPECIAL_FIFO_PAGES,
        )

    def stage_2(device: ttnn.MeshDevice) -> StageKind:
        return PassthroughStage(PassthroughPayload.ACTIVATION_W_TOKEN_META)

    def stage_3(device: ttnn.MeshDevice) -> StageKind:
        return SpecLMHeadStage(
            weights=weight_provider.load_lm_head(device),
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
            spec_weights=weight_provider.load_spec(device),
        )

    return PipelineConfiguration(
        {
            0: stage_0,
            1: stage_1,
            2: stage_2,
            3: stage_3,
        }
    )


def create_single_galaxy_combined_spec_decode_pipeline_configuration(
    weight_provider: WeightProvider,
    *,
    fp32_dest_acc_en: bool = True,
    persistent_mode: bool = True,
) -> PipelineConfiguration:
    """4-stage single-galaxy pipeline with SpecLMHead + Embedding fused on P0:
    P0(SpecLMHead+Embed) -> P1(BaseLMHead+MTP) -> P2(Passthrough) -> P3(Passthrough) -> back to P0."""

    def stage_0(device: ttnn.MeshDevice) -> StageKind:
        return SpecLMHeadWithEmbeddingStage(
            weights=weight_provider.load_lm_head(device),
            embedding_weights=weight_provider.load_embedding(device),
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
            spec_weights=weight_provider.load_spec(device),
            loopback_input_fifo_pages=LMHEAD_SPECIAL_FIFO_PAGES,
        )

    def stage_1(device: ttnn.MeshDevice) -> StageKind:
        return PassthroughStage(PassthroughPayload.ACTIVATION_W_TOKEN_META)

    def stage_2(device: ttnn.MeshDevice) -> StageKind:
        return PassthroughStage(
            PassthroughPayload.ACTIVATION_W_TOKEN_META,
            downstream_fifo_pages=LMHEAD_SPECIAL_FIFO_PAGES,
        )

    def stage_3(device: ttnn.MeshDevice) -> StageKind:
        return BaseLMHeadStage(
            weights=weight_provider.load_lm_head(device),
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
            mtp_weights=weight_provider.load_mtp(device),
            send_mtp_output_downstream=True,
            embedding_weights=weight_provider.load_embedding(device),
            upstream_fifo_pages=LMHEAD_SPECIAL_FIFO_PAGES,
            downstream_fifo_pages=LMHEAD_SPECIAL_FIFO_PAGES,
        )

    return PipelineConfiguration(
        {
            0: stage_0,
            1: stage_1,
            2: stage_2,
            3: stage_3,
        }
    )


def create_single_galaxy_deepseek_pipeline_configuration(
    weight_provider: WeightProvider,
    *,
    lm_head_fp32_dest_acc_en: bool = True,
    lm_head_persistent_mode: bool = True,
    dense_layer_id: int = 0,
    moe_layer_id: int = 0,
    host_loopback: bool = True,
) -> PipelineConfiguration:
    """4-stage single-galaxy: Embed -> LMHead -> Passthrough(TOKEN) -> Passthrough(TOKEN).

    When ``host_loopback=True``, the last-stage token is returned to rank 0 via host MPI
    (``send_bytes``/``recv_bytes``) instead of a fabric loopback kernel.
    """

    def stage_0(device: ttnn.MeshDevice) -> StageKind:
        return EmbeddingStage(
            weight_provider.load_embedding(device),
            forward_metadata=True,
            host_loopback=host_loopback,
        )

    def stage_1(device: ttnn.MeshDevice) -> StageKind:
        return DenseDecoderStage(
            weights=weight_provider.load_dense_layer(layer_id=dense_layer_id, device=device),
            layer_idx=dense_layer_id,
            forward_metadata=True,
            host_loopback=host_loopback,
        )

    def stage_2(device: ttnn.MeshDevice) -> StageKind:
        return MoEDecoderStage(
            weights=weight_provider.load_moe_layer(layer_id=moe_layer_id, device=device),
            layer_idx=moe_layer_id,
            forward_metadata=True,
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
        return EmbeddingStage(weight_provider.load_embedding(device), forward_metadata=True)

    def stage_14(device: ttnn.MeshDevice) -> StageKind:
        mtp_weights = weight_provider.load_mtp(device) if enable_mtp else None
        return BaseLMHeadStage(
            weights=weight_provider.load_lm_head(device),
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
            mtp_weights=mtp_weights,
            send_mtp_output_downstream=enable_mtp,
            embedding_weights=weight_provider.load_embedding(device) if enable_mtp else None,
            upstream_fifo_pages=LMHEAD_SPECIAL_FIFO_PAGES if enable_mtp else DEFAULT_ACTIVATION_FIFO_PAGES,
            downstream_fifo_pages=LMHEAD_SPECIAL_FIFO_PAGES if enable_mtp else DEFAULT_ACTIVATION_FIFO_PAGES,
        )

    def _dense_stage(layer_id: int):
        return lambda d: DenseDecoderStage(
            weights=weight_provider.load_dense_layer(layer_id=layer_id, device=d),
            layer_idx=layer_id,
            forward_metadata=True,
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
            forward_metadata=True,
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
    if enable_mtp:
        stage_factories[13] = _decoder_stage(
            moe_layer_id if moe_layer_id is not None else 12,
            downstream_fifo_pages=LMHEAD_SPECIAL_FIFO_PAGES,
        )
    return PipelineConfiguration(stage_factories)


def create_sp4_pipeline_configuration(
    weight_provider: WeightProvider,
    *,
    fp32_dest_acc_en: bool = True,
    persistent_mode: bool = True,
    enable_mtp: bool = False,
    dense_layer_id_override: int | None = None,
    moe_layer_id_override: int | None = None,
    num_slots: int = 64,
) -> PipelineConfiguration:
    """64-stage super-pod: Embed -> Dense(0,1,2) -> Decoder(3..60) -> LMHead -> Token fwd.

    If dense_layer_id_override is set (e.g. 0), all dense stages use that layer id.
    If moe_layer_id_override is set (e.g. 3), all decoder stages use that layer id.
    """
    fwd_payload = PassthroughPayload.ACTIVATION_W_TOKEN_META if enable_mtp else PassthroughPayload.TOKEN

    def stage_0(device: ttnn.MeshDevice) -> StageKind:
        return SpecLMHeadWithEmbeddingStage(
            weights=weight_provider.load_lm_head(device),
            embedding_weights=weight_provider.load_embedding(device),
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
            spec_weights=weight_provider.load_spec(device) if enable_mtp else None,
        )

    def stage_62(device: ttnn.MeshDevice) -> StageKind:
        mtp_weights = weight_provider.load_mtp(device) if enable_mtp else None
        return BaseLMHeadStage(
            weights=weight_provider.load_lm_head(device),
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
            mtp_weights=mtp_weights,
            send_mtp_output_downstream=enable_mtp,
            embedding_weights=weight_provider.load_embedding(device),
            upstream_fifo_pages=LMHEAD_SPECIAL_FIFO_PAGES if enable_mtp else DEFAULT_ACTIVATION_FIFO_PAGES,
            downstream_fifo_pages=LMHEAD_SPECIAL_FIFO_PAGES if enable_mtp else DEFAULT_ACTIVATION_FIFO_PAGES,
        )

    def _dense_stage(layer_id: int):
        return lambda d: DenseDecoderStage(
            weights=weight_provider.load_dense_layer(layer_id=layer_id, device=d),
            layer_idx=layer_id,
            num_slots=num_slots,
            forward_metadata=True,
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
            forward_metadata=True,
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
        **{i: _decoder_stage(moe_layer_id if moe_layer_id is not None else i - 1) for i in range(4, 62)},
        62: stage_62,
        63: _decoder_stage(
            61,
            upstream_fifo_pages=LMHEAD_SPECIAL_FIFO_PAGES if enable_mtp else DEFAULT_ACTIVATION_FIFO_PAGES,
        ),
    }
    if enable_mtp:
        stage_factories[61] = _decoder_stage(60, downstream_fifo_pages=LMHEAD_SPECIAL_FIFO_PAGES)
    return PipelineConfiguration(stage_factories)


def create_pipeline_configuration_from_num_procs(
    num_procs: int,
    weight_provider: WeightProvider,
    *,
    fp32_dest_acc_en: bool = True,
    persistent_mode: bool = True,
    enable_mtp: bool = False,
    dense_layer_id_override: int | None = None,
    moe_layer_id_override: int | None = None,
    num_slots: int = 64,
) -> PipelineConfiguration:
    """Pick topology from process count (4 -> single_galaxy, 16 -> single_pod, 64 -> sp4)."""
    if num_procs == 4:
        return create_single_galaxy_pipeline_configuration(
            weight_provider,
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
            enable_mtp=enable_mtp,
        )
    if num_procs == 16:
        assert enable_mtp, "16-proc pipeline currently requires enable_mtp=True and uses the spec decode topology"
        return create_single_pod_spec_decode_pipeline_configuration(
            weight_provider,
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
            enable_mtp=enable_mtp,
            dense_layer_id_override=dense_layer_id_override,
            moe_layer_id_override=moe_layer_id_override,
            num_slots=num_slots,
        )
    if num_procs == 64:
        return create_sp4_pipeline_configuration(
            weight_provider,
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
            enable_mtp=enable_mtp,
            dense_layer_id_override=dense_layer_id_override,
            moe_layer_id_override=moe_layer_id_override,
            num_slots=num_slots,
        )
    raise ValueError(f"Unsupported num_procs: {num_procs}")


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
        stages_metadata: dict[int, StageMetadata] | None = None,
        pipeline_config: list | None = None,
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
        return Pipeline(
            mesh_device,
            stage,
            my_stage_idx,
            stages_metadata=stages_metadata,
            pipeline_config=pipeline_config,
            host_loopback=host_loopback,
        )


class Pipeline:
    """Orchestrator for one pipeline stage with explicit 4-phase setup."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        stage_kind: StageKind,
        my_stage_idx: int,
        stages_metadata: dict[int, StageMetadata] | None = None,
        pipeline_config: list | None = None,
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

    def read_output(self, output_tensor: ttnn.Tensor) -> None:
        self._block.read_output(output_tensor)

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
          4. Stage 0 pushes a dummy token and drains the round-trip result.
             The token naturally triggers ``persistent_next_iter_sem`` via the
             pipeline's socket/d2d flow, providing both the gate release AND
             the data payload.  All cores complete this final iteration
             together, loop back to the top-of-loop termination check, see the
             flag, and break.
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

        if self._pipeline_block.is_first_pipeline_stage():
            self._pipeline_block.push_dummy_token()
            self._pipeline_block.drain_dummy_output()

        ttnn.distributed_context_barrier()

        self._pipeline_block.terminate()
        ttnn.synchronize_device(self._mesh_device)


def create_single_pod_spec_decode_pipeline_configuration(
    weight_provider: WeightProvider,
    *,
    fp32_dest_acc_en: bool = True,
    persistent_mode: bool = True,
    enable_mtp: bool = False,
    dense_layer_id_override: int | None = None,
    moe_layer_id_override: int | None = None,
    num_slots: int = 64,
) -> PipelineConfiguration:
    """16-stage single-pod: Embed -> Dense(0,1,2) -> Decoder(3..12) -> LMHead -> Token fwd.

    If dense_layer_id_override is set (e.g. 0), all dense stages use that layer id.
    If moe_layer_id_override is set (e.g. 3), all decoder stages use that layer id.
    """
    fwd_payload = PassthroughPayload.ACTIVATION_W_TOKEN_META if enable_mtp else PassthroughPayload.TOKEN

    def stage_0(device: ttnn.MeshDevice) -> StageKind:
        return SpecLMHeadWithEmbeddingStage(
            weights=weight_provider.load_lm_head(device),
            embedding_weights=weight_provider.load_embedding(device),
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
            spec_weights=weight_provider.load_spec(device) if enable_mtp else None,
        )

    def stage_14(device: ttnn.MeshDevice) -> StageKind:
        mtp_weights = weight_provider.load_mtp(device) if enable_mtp else None
        return BaseLMHeadStage(
            weights=weight_provider.load_lm_head(device),
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
            mtp_weights=mtp_weights,
            send_mtp_output_downstream=enable_mtp,
            embedding_weights=weight_provider.load_embedding(device),
            upstream_fifo_pages=LMHEAD_SPECIAL_FIFO_PAGES if enable_mtp else DEFAULT_ACTIVATION_FIFO_PAGES,
            downstream_fifo_pages=LMHEAD_SPECIAL_FIFO_PAGES if enable_mtp else DEFAULT_ACTIVATION_FIFO_PAGES,
        )

    def _dense_stage(layer_id: int):
        return lambda d: DenseDecoderStage(
            weights=weight_provider.load_dense_layer(layer_id=layer_id, device=d),
            layer_idx=layer_id,
            num_slots=num_slots,
            forward_metadata=True,
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
            forward_metadata=True,
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
        15: _decoder_stage(
            61,
            upstream_fifo_pages=LMHEAD_SPECIAL_FIFO_PAGES if enable_mtp else DEFAULT_ACTIVATION_FIFO_PAGES,
        ),
    }
    if enable_mtp:
        stage_factories[13] = _decoder_stage(
            moe_layer_id if moe_layer_id is not None else 12,
            downstream_fifo_pages=LMHEAD_SPECIAL_FIFO_PAGES,
        )
    return PipelineConfiguration(stage_factories)


def create_single_pod_combined_spec_decode_pipeline_configuration(
    weight_provider: WeightProvider,
    *,
    fp32_dest_acc_en: bool = True,
    persistent_mode: bool = True,
) -> PipelineConfiguration:
    """4-stage single-galaxy pipeline with SpecLMHead + Embedding fused on P0:
    P0(SpecLMHead+Embed) -> P1(BaseLMHead+MTP) -> P2(Passthrough) -> P3(Passthrough) -> back to P0."""

    def stage_0(device: ttnn.MeshDevice) -> StageKind:
        return SpecLMHeadWithEmbeddingStage(
            weights=weight_provider.load_lm_head(device),
            embedding_weights=weight_provider.load_embedding(device),
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
            spec_weights=weight_provider.load_spec(device),
        )

    def passthrough_stage(device: ttnn.MeshDevice) -> StageKind:
        return PassthroughStage(PassthroughPayload.ACTIVATION_W_TOKEN_META)

    def stage_14(device: ttnn.MeshDevice) -> StageKind:
        return BaseLMHeadStage(
            weights=weight_provider.load_lm_head(device),
            fp32_dest_acc_en=fp32_dest_acc_en,
            persistent_mode=persistent_mode,
            mtp_weights=weight_provider.load_mtp(device),
            send_mtp_output_downstream=True,
            embedding_weights=weight_provider.load_embedding(device),
        )

    return PipelineConfiguration(
        {
            0: stage_0,
            **{i: passthrough_stage for i in range(1, 14)},
            14: stage_14,
            15: passthrough_stage,
        }
    )
