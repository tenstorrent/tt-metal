# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from loguru import logger

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.demo.mesh_device_context import _worker_l1_size_for_rank
from models.demos.deepseek_v3_b1.demo.pipeline import create_pipeline_configuration_from_num_procs
from models.demos.deepseek_v3_b1.demo.weight_provider import (
    CacheWeightProvider,
    StateDictWeightProvider,
    SyntheticWeightProvider,
    WeightProvider,
)
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import StageMetadata
from models.demos.deepseek_v3_b1.model import (
    TOKEN_ID_BYTES,
    DecodeResult,
    DeepSeekV3,
    TokenType,
    create_output_buffer,
    page_size_bytes,
    to_spec_input,
)

_HOST_LOOPBACK_CONTROL_TAG = 73101
_HOST_LOOPBACK_READ_CMD = b"R"
_HOST_LOOPBACK_STOP_CMD = b"S"


class ModelPipeline:
    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        *,
        weights_mode: Literal["synthetic", "real", "state_dict"] = "real",
        cache_path: Path | None = None,
        model_path: Path | None = None,
        lm_head_fp32_dest_acc_en: bool = True,
        lm_head_persistent_mode: bool = True,
        seed: int = 520,
        dense_layer_id_override: int | None = None,
        moe_layer_id_override: int | None = None,
        io_socket_descriptor_prefix: str | None = None,
        num_slots: int = 64,
        relaxed_acceptance_delta: float = 0.6,
        top_k: int = 1,
        top_p: float = 1.0,
        temperature: float = 0.6,
        enable_speculative_decode: bool = True,
        enable_sram_hot_experts: bool = False,
        sram_hot_experts_ceiling: int = 64,
        bspm_dir: Path | None = None,
        bspm_variant: str = "B",
        bspm_budget: float = 3.5,
    ):
        logger.info(
            "Initializing DeepSeek V3 B1 pod pipeline (weights={}, seed={}, lm_head_fp32={}, lm_head_persistent_mode={}, speculative_decode={})",
            weights_mode,
            seed,
            lm_head_fp32_dest_acc_en,
            lm_head_persistent_mode,
            enable_speculative_decode,
        )
        if not is_slow_dispatch():
            raise RuntimeError(
                "DeepSeek V3 B1 pod pipeline requires slow dispatch mode. Set TT_METAL_SLOW_DISPATCH_MODE=1 and rerun."
            )
        self.mesh_device = mesh_device
        self.relaxed_acceptance_delta = relaxed_acceptance_delta
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        num_procs = int(ttnn.distributed_context_get_size())
        if num_procs not in (4, 16, 64):
            raise RuntimeError(f"Pod pipeline requires 4, 16, or 64 distributed processes; got {num_procs}")
        if not enable_speculative_decode and num_procs not in (16, 64):
            raise RuntimeError("Base decode is currently supported only for the 16- and 64-process pipelines")
        ttnn.enable_asynchronous_slow_dispatch(self.mesh_device)
        self.enable_speculative_decode = enable_speculative_decode

        # Each host loads/creates only the weights for its stage via the provider.
        if weights_mode == "real":
            if cache_path is None:
                raise ValueError("weights_mode='real' requires cache_path")
            if model_path is None:
                raise ValueError("weights_mode='real' requires model_path")
            if enable_sram_hot_experts:
                from models.demos.deepseek_v3_b1.compressed_tensor.assigner import CompressedTensorAssigner
                from models.demos.deepseek_v3_b1.weights.transforms.sram_experts import (
                    SramExpertCoreGrids,
                    _load_routing_frequencies,
                    build_sram_hot_expert_config,
                )

                # Layers 3..60 are routed-expert MoE layers; layer 61 (MTP) is intentionally
                # skipped (no calibration data) and runs DRAM-only.
                moe_layer_indices = list(range(3, 61))
                freqs = _load_routing_frequencies()
                ranked = build_sram_hot_expert_config(moe_layer_indices, freqs)
                sram_hot_experts = {k: v[:sram_hot_experts_ceiling] for k, v in ranked.items()}
                logger.info(
                    "SRAM hot experts enabled (ceiling={}) on layers {}",
                    sram_hot_experts_ceiling,
                    sorted(sram_hot_experts.keys()),
                )
                provider: WeightProvider = CacheWeightProvider(
                    cache_path,
                    model_path,
                    sram_hot_experts=sram_hot_experts,
                    sram_core_grids=SramExpertCoreGrids.shared_expert_mirror(),
                    sram_assigner=CompressedTensorAssigner(formats=["bfp4"]),
                    worker_l1_size=_worker_l1_size_for_rank(
                        num_procs,
                        enable_speculative_decode=enable_speculative_decode,
                    ),
                    bspm_dir=bspm_dir,
                    bspm_variant=bspm_variant,
                    bspm_budget=bspm_budget,
                )
            else:
                provider = CacheWeightProvider(
                    cache_path,
                    model_path,
                    bspm_dir=bspm_dir,
                    bspm_variant=bspm_variant,
                    bspm_budget=bspm_budget,
                )
        elif weights_mode == "state_dict":
            if model_path is None:
                raise ValueError("weights_mode='state_dict' requires model_path")
            provider = StateDictWeightProvider(
                model_path,
                bspm_dir=bspm_dir,
                bspm_variant=bspm_variant,
                bspm_budget=bspm_budget,
            )
        elif weights_mode == "synthetic":
            provider = SyntheticWeightProvider(
                seed=seed,
                bspm_dir=bspm_dir,
                bspm_variant=bspm_variant,
                bspm_budget=bspm_budget,
            )
        else:
            raise ValueError(f"Unknown weights_mode: {weights_mode!r}")
        config = create_pipeline_configuration_from_num_procs(
            num_procs,
            provider,
            fp32_dest_acc_en=lm_head_fp32_dest_acc_en,
            persistent_mode=lm_head_persistent_mode,
            dense_layer_id_override=dense_layer_id_override,
            moe_layer_id_override=moe_layer_id_override,
            enable_mtp=enable_speculative_decode,
            enable_speculative_decode=enable_speculative_decode,
            num_slots=num_slots,
        )
        if config.num_stages != num_procs:
            raise RuntimeError(f"Pipeline configuration has {config.num_stages} stages but {num_procs} processes")

        logger.info("Building pipeline")
        # Propagate an identical, explicit pipeline_config and stages_metadata
        # to every rank. Without this, each rank independently calls
        # generate_blitz_decode_pipeline() and the submesh-aware code paths
        # added in #42002 can produce inconsistent entry/exit MeshCoordinates
        # across ranks, causing the inter-mesh MeshSocket handshake in
        # SpecEmbeddingPipelineBlock.__init__'s h2d_host_io to time out.
        pipeline_config = ttnn._ttnn.multi_device.experimental.generate_blitz_decode_pipeline()
        stages_metadata = {i: StageMetadata(rank=i, mesh_id=i) for i in range(num_procs)}
        self.pipeline = config.build_pipeline(
            self.mesh_device,
            stages_metadata=stages_metadata,
            pipeline_config=pipeline_config,
        )

        logger.info("Setting up and running pipeline")
        self.pipeline.setup_and_run()

        self._page_size_datums = page_size_bytes(1) // TOKEN_ID_BYTES
        self._output_buffer = create_output_buffer(self._page_size_datums)
        self.position_id: int | None = None
        self._in_thinking_phase = False
        self.model: DeepSeekV3 | None = None
        if self.pipeline.my_stage_idx == 0:
            self.model = DeepSeekV3(
                write_fn=self.pipeline.write_token,
                read_fn=self._read_pipeline_output,
                batch_size=1,
                pipeline_depth=config.num_stages,
            )

            if io_socket_descriptor_prefix is not None:
                self.pipeline.export_host_socket_descriptors(io_socket_descriptor_prefix)

        logger.info(f"Created ModelPipeline for stage {self.pipeline.my_stage_idx}.")

    def _pipeline_block(self):
        return getattr(self.pipeline, "_pipeline_block", None)

    def _uses_host_loopback(self) -> bool:
        block = self._pipeline_block()
        return block is not None and getattr(block, "_loopback_mode", None) == "host"

    def _host_loopback_rank(self, stage_idx: int) -> int:
        block = self._pipeline_block()
        assert block is not None
        return block._stages[stage_idx].rank

    def _request_host_loopback_output(self) -> None:
        if not self._uses_host_loopback() or self.pipeline.my_stage_idx != 0:
            return
        dest = self._host_loopback_rank(self.pipeline._pipeline_block.num_procs - 1)
        if dest == int(ttnn.distributed_context_get_rank()):
            return
        from ttnn._ttnn.multi_device import send_bytes

        send_bytes(_HOST_LOOPBACK_READ_CMD, dest, _HOST_LOOPBACK_CONTROL_TAG)

    def _read_pipeline_output(self, output_tensor: ttnn.Tensor):
        self._request_host_loopback_output()
        return self.pipeline.read_output(output_tensor)

    def run_host_loopback_output_forwarder(self) -> None:
        """Serve rank-0 host-loopback read requests on the last pipeline stage."""
        block = self._pipeline_block()
        if not self._uses_host_loopback() or block is None or not block.is_last_stage or block.is_pipeline_start:
            return

        from ttnn._ttnn.multi_device import recv_bytes

        source = self._host_loopback_rank(0)
        logger.info("Serving host-loopback D2H output requests")
        while True:
            cmd = recv_bytes(1, source, _HOST_LOOPBACK_CONTROL_TAG)
            if cmd == _HOST_LOOPBACK_READ_CMD:
                self.pipeline.read_output(self._output_buffer)
            elif cmd == _HOST_LOOPBACK_STOP_CMD:
                logger.info("Host-loopback output forwarder stopped")
                return
            else:
                raise RuntimeError(f"Unknown host-loopback output command: {cmd!r}")

    def stop_host_loopback_output_forwarder(self) -> None:
        if not self._uses_host_loopback() or self.pipeline.my_stage_idx != 0:
            return
        dest = self._host_loopback_rank(self.pipeline._pipeline_block.num_procs - 1)
        if dest == int(ttnn.distributed_context_get_rank()):
            return
        from ttnn._ttnn.multi_device import send_bytes

        send_bytes(_HOST_LOOPBACK_STOP_CMD, dest, _HOST_LOOPBACK_CONTROL_TAG)

    def prefill_forward(self, tokens: list[int]) -> list[DecodeResult]:
        """Prefill prompt tokens and return the DecodeResults from the last prompt token's outputs."""
        if self.pipeline.my_stage_idx != 0:
            raise RuntimeError("prefill_forward() should only be called on mesh id 0")
        assert self.model is not None
        logger.debug(f"Prefilling with {len(tokens)} tokens...")
        prompt_token_tensors = [
            to_spec_input(
                tokens[i],
                tokens[i + 1] if i < len(tokens) - 1 else -1,
                user_id=0,
                position_id=i,
                page_size_datums=self._page_size_datums,
                token_type=TokenType.BASE,
                temperature=self.temperature,
                top_k=self.top_k,
                probability_mass_threshold=self.top_p,
            )
            for i in range(len(tokens))
        ]
        results = self.model.prefill(prompt_token_tensors)
        self.position_id = len(tokens)
        logger.debug(f"Done prefilling with {len(tokens)} tokens.")
        return results

    def decode_forward(self, input_token: int) -> int:
        """Run 1 decode step and return the next token id."""
        if self.pipeline.my_stage_idx != 0:
            raise RuntimeError("decode_forward() should only be called on mesh id 0")
        assert self.model is not None
        if self.position_id is None:
            raise RuntimeError("decode_forward() requires prefill_forward() to be called first")

        self.model.write_input(
            input_token,
            -1,
            user_id=0,
            position_id=self.position_id,
            token_type=TokenType.BASE,
            temperature=self.temperature,
            top_k=self.top_k,
            probability_mass_threshold=self.top_p,
        )
        result = self.model.read_result()
        self.position_id += 1
        return result.token_0

    def check_acceptance(self, prev_spec_token_id: int, result: DecodeResult) -> bool:
        """Check whether the speculative token should be accepted.

        Outside the thinking phase: strict exact-match.
        Inside the thinking phase: relaxed probability-delta threshold.
        """
        if result.p_indices is None or result.p_scores is None:
            return False
        if prev_spec_token_id not in result.p_indices:
            return False
        prev_spec_token_index = result.p_indices.index(prev_spec_token_id)
        p_max_prob = result.p_scores[0]
        p_draft_prob = result.p_scores[prev_spec_token_index]
        return (p_max_prob - p_draft_prob) <= self.relaxed_acceptance_delta

    def _write_spec_pair(
        self,
        token_0: int,
        pos_0: int,
        token_1: int,
        pos_1: int,
        user_id: int = 0,
        temperature: float = 0.6,
        top_k: int = 1,
        probability_mass_threshold: float = 1.0,
    ) -> None:
        """Write two tokens (base + speculation) into the pipeline."""
        assert self.model is not None
        self.model.write_input(
            token_0,
            -1,
            user_id,
            pos_0,
            token_type=TokenType.BASE,
            temperature=temperature,
            top_k=top_k,
            probability_mass_threshold=probability_mass_threshold,
        )
        self.model.write_input(
            token_1,
            -1,
            user_id,
            pos_1,
            token_type=TokenType.SPEC,
            temperature=temperature,
            top_k=top_k,
            probability_mass_threshold=probability_mass_threshold,
        )

    def run_inference_base_decode(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        on_token: Callable[[int], None] | None = None,
        eos_token_id: int | None = None,
        return_generated_tokens: bool = False,
    ) -> list[int] | None:
        """Run base decode without the speculative acceptance/rejection state machine."""
        if self.pipeline.my_stage_idx != 0:
            raise RuntimeError("run_inference_base_decode() should only be called on stage 0")
        assert max_new_tokens >= 1, f"max_new_tokens must be >= 1, got {max_new_tokens}"
        assert self.model is not None

        generated_tokens: list[int] = []

        def is_eos(token_id: int) -> bool:
            return eos_token_id is not None and token_id == eos_token_id

        def emit(token_id: int) -> None:
            if on_token is not None:
                on_token(token_id)
            generated_tokens.append(token_id)

        pending: deque[DecodeResult] = deque(self.prefill_forward(prompt_token_ids))

        start_time = time.time()
        num_reads = 0
        num_writes = 0
        while len(generated_tokens) < max_new_tokens:
            if pending:
                result = pending.popleft()
            else:
                result = self.model.read_result()
                num_reads += 1

            if result.token_1 is None or result.token_1_pos is None:
                raise RuntimeError("Base decode requires token_1 output from the spec LM head")

            next_token = result.token_1
            next_pos = result.token_1_pos - 1
            if next_pos != result.token_0_pos:
                raise RuntimeError(
                    f"Base decode position mismatch: token_1_pos - 1 = {next_pos}, "
                    f"but token_0_pos = {result.token_0_pos}"
                )
            emit(next_token)
            if is_eos(next_token) or len(generated_tokens) >= max_new_tokens:
                break

            self.model.write_input(
                next_token,
                -1,
                0,
                next_pos,
                token_type=TokenType.BASE,
                temperature=self.temperature,
                top_k=self.top_k,
                probability_mass_threshold=self.top_p,
            )
            num_writes += 1

        while num_reads < num_writes:
            self.model.read_result()
            num_reads += 1

        end_time = time.time()
        elapsed = end_time - start_time
        logger.debug(f"Time taken: {elapsed} seconds")
        logger.debug(f"Tokens per second: {len(generated_tokens) / max(elapsed, 1e-9)}")
        logger.debug("Base decode generation complete ({} tokens generated)", len(generated_tokens))
        return generated_tokens if return_generated_tokens else None

    def run_inference(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        on_token: Callable[[int], None] | None = None,
        eos_token_id: int | None = None,
        think_token_ids: list[int] | None = None,
        return_generated_tokens: bool = False,
    ) -> list[int] | None:
        """Run speculative-decode inference: prefill then decode with multi-token prediction.

        The pipeline produces structured 2-token output pages (base + speculation).
        Each output page carries position IDs that the host uses directly for
        write-back, so no manual position tracking is needed.

        The host drives the ACCEPT / REJECT / STALE / CONTINUE state machine:
          - ACCEPT:   emit confirmed token, then read CONTINUE.
          - CONTINUE: emit bonus token, write tokens at device-supplied positions.
          - REJECT:   emit base output, write tokens at device-supplied positions.
          - STALE:    discard and re-read.
        """
        if not self.enable_speculative_decode:
            return self.run_inference_base_decode(
                prompt_token_ids=prompt_token_ids,
                max_new_tokens=max_new_tokens,
                on_token=on_token,
                eos_token_id=eos_token_id,
                return_generated_tokens=return_generated_tokens,
            )

        if self.pipeline.my_stage_idx != 0:
            raise RuntimeError("run_inference() should only be called on stage 0")
        assert max_new_tokens >= 1, f"max_new_tokens must be >= 1, got {max_new_tokens}"

        self._in_thinking_phase = False
        generated_tokens: list[int] = []
        verified_spec_tokens: list[int] = []
        unverified_spec_tokens: list[int] = []
        if think_token_ids is not None:
            think_open_id, think_close_id = think_token_ids
        else:
            think_open_id, think_close_id = None, None

        def is_eos(token_id: int) -> bool:
            """Returns True if a token is the EOS token"""
            return eos_token_id is not None and token_id == eos_token_id

        def emit(token_id: int) -> None:
            """Emit a token to the caller and update thinking-phase state."""
            if on_token is not None:
                on_token(token_id)
            generated_tokens.append(token_id)
            if token_id == think_open_id:
                self._in_thinking_phase = True
            elif token_id == think_close_id:
                self._in_thinking_phase = False

        # --- Prefill --------------------------------------------------------
        prefill_results = self.prefill_forward(prompt_token_ids)

        # Seed the state machine with both pages from the last prefill write,
        # then read from the pipeline for all subsequent results.
        pending: deque[DecodeResult] = deque(prefill_results)
        base_accept = 0
        spec_accept = 0
        base_reject = 0
        spec_reject = 0
        # --- Speculative decode state machine --------------------------------
        iteration = 0
        start_time = time.time()
        num_emits = 0
        num_writes = 0
        num_reads = 0
        signal_to_exit = False
        while len(generated_tokens) < max_new_tokens or signal_to_exit:
            iteration += 1
            if pending:
                result = pending.popleft()
            else:
                result = self.model.read_result()
                num_reads += 1

            if not unverified_spec_tokens and not verified_spec_tokens:
                unverified_spec_tokens.append(result.token_1)
                emit(result.token_0)
                num_emits += 1
            else:
                if result.token_0_type == TokenType.BASE:
                    if self.check_acceptance(unverified_spec_tokens[-1], result):
                        verified_spec_tokens.append(unverified_spec_tokens.pop())
                        emit(result.token_0)
                        base_accept += 1
                        num_emits += 1
                        signal_to_exit = is_eos(result.token_0) or len(generated_tokens) >= max_new_tokens
                        continue
                    else:
                        unverified_spec_tokens.pop()
                        unverified_spec_tokens.append(result.token_1)
                        emit(result.token_0)
                        base_reject += 1
                        num_emits += 1
                        signal_to_exit = is_eos(result.token_0) or len(generated_tokens) >= max_new_tokens

                if result.token_0_type == TokenType.SPEC:
                    if verified_spec_tokens:
                        verified_spec_tokens.pop()
                        unverified_spec_tokens.append(result.token_1)

                        if signal_to_exit:
                            break

                        emit(result.token_0)
                        spec_accept += 1
                        num_emits += 1
                    else:
                        if signal_to_exit:
                            break
                        spec_reject += 1
                        continue

            if is_eos(result.token_0) or len(generated_tokens) >= max_new_tokens:
                break

            self._write_spec_pair(
                result.token_0,
                result.token_0_pos,
                result.token_1,
                result.token_1_pos,
                temperature=self.temperature,
                top_k=self.top_k if self._in_thinking_phase else 1,
                probability_mass_threshold=self.top_p,
            )
            num_writes += 2

        while num_reads < num_writes:
            self.model.read_result()
            num_reads += 1

        end_time = time.time()
        logger.debug(f"Time taken: {end_time - start_time} seconds")
        logger.debug(f"Tokens per second: {num_emits / (end_time - start_time)}")
        logger.debug(
            f"Base Accept: {base_accept}, Base Reject: {base_reject}, Spec Accept: {spec_accept}, Spec Reject: {spec_reject}, Base Accept Rate: {base_accept / (base_accept + base_reject + 1e-5)}, Spec Accept Rate: {spec_accept / (spec_accept + spec_reject + 1e-5)}"
        )
        logger.debug("Generation complete ({} tokens generated)", len(generated_tokens))
        return generated_tokens if return_generated_tokens else None

    def barrier(self) -> None:
        self.pipeline.barrier()

    def terminate(self) -> None:
        self.pipeline.terminate()
