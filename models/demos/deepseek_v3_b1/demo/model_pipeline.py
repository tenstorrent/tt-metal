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
from models.demos.deepseek_v3_b1.demo.pipeline import create_pipeline_configuration_from_stage_count
from models.demos.deepseek_v3_b1.demo.pipeline_routing import build_local_stage_socket_plans, build_stage_routing
from models.demos.deepseek_v3_b1.demo.stage_family import stage_family_from_shape
from models.demos.deepseek_v3_b1.demo.weight_provider import (
    CacheWeightProvider,
    StateDictWeightProvider,
    SyntheticWeightProvider,
    WeightProvider,
)
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import PipelineConfigEntry
from models.demos.deepseek_v3_b1.model import (
    TOKEN_ID_BYTES,
    DecodeResult,
    DeepSeekV3,
    TokenType,
    page_size_bytes,
    to_spec_input,
)


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
        enable_sram_bspm: bool = False,
    ):
        logger.info(
            "Initializing DeepSeek V3 B1 pod pipeline (weights={}, lm_head_fp32={}, lm_head_persistent_mode={}, speculative_decode={})",
            weights_mode,
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
        stage_family = stage_family_from_shape(self.mesh_device.shape)
        allocation = ttnn._ttnn.multi_device.experimental.resolve_blitz_decode_pipeline_allocation()
        num_stages = len(allocation.stages)
        my_rank = int(ttnn.distributed_context_get_rank())
        local_stage_plans = build_local_stage_socket_plans(allocation, my_rank)
        if len(local_stage_plans) != 1:
            raise RuntimeError(
                f"Expected exactly one local stage plan for rank {my_rank}, got {len(local_stage_plans)}"
            )
        local_stage_plan = local_stage_plans[0]
        if not enable_speculative_decode and stage_family.value == "4x2" and num_stages not in (16, 64):
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
                    worker_l1_size=_worker_l1_size_for_rank(
                        num_procs,
                        enable_speculative_decode=enable_speculative_decode,
                    ),
                    bspm_dir=bspm_dir,
                    bspm_variant=bspm_variant,
                    bspm_budget=bspm_budget,
                    enable_sram_bspm=enable_sram_bspm,
                )
            else:
                provider = CacheWeightProvider(
                    cache_path,
                    model_path,
                    bspm_dir=bspm_dir,
                    bspm_variant=bspm_variant,
                    bspm_budget=bspm_budget,
                    enable_sram_bspm=enable_sram_bspm,
                )
        elif weights_mode == "state_dict":
            if model_path is None:
                raise ValueError("weights_mode='state_dict' requires model_path")
            provider = StateDictWeightProvider(
                model_path,
                bspm_dir=bspm_dir,
                bspm_variant=bspm_variant,
                bspm_budget=bspm_budget,
                enable_sram_bspm=enable_sram_bspm,
            )
        elif weights_mode == "synthetic":
            provider = SyntheticWeightProvider(
                bspm_dir=bspm_dir,
                bspm_variant=bspm_variant,
                bspm_budget=bspm_budget,
                enable_sram_bspm=enable_sram_bspm,
            )
        else:
            raise ValueError(f"Unknown weights_mode: {weights_mode!r}")
        config = create_pipeline_configuration_from_stage_count(
            num_stages,
            stage_family,
            provider,
            fp32_dest_acc_en=lm_head_fp32_dest_acc_en,
            persistent_mode=lm_head_persistent_mode,
            dense_layer_id_override=dense_layer_id_override,
            moe_layer_id_override=moe_layer_id_override,
            enable_mtp=enable_speculative_decode,
            enable_speculative_decode=enable_speculative_decode,
            num_slots=num_slots,
            enable_sram_bspm=enable_sram_bspm,
        )
        if config.num_stages != num_stages:
            raise RuntimeError(f"Pipeline configuration has {config.num_stages} stages but {num_stages} stages")

        logger.info("Building pipeline")
        pipeline_config = [
            PipelineConfigEntry(
                stage.entry_endpoint.mesh_coord,
                (stage.exit_endpoint if stage.exit_endpoint is not None else stage.entry_endpoint).mesh_coord,
            )
            for stage in allocation.stages
        ]
        if allocation.initialize_loopback:
            pipeline_config.append(
                PipelineConfigEntry(
                    allocation.loopback_entry_endpoint.mesh_coord,
                    allocation.host_egress_endpoint.mesh_coord,
                )
            )
        stages_metadata = build_stage_routing(allocation)
        self.pipeline = config.build_pipeline(
            self.mesh_device,
            my_stage_idx=local_stage_plan.logical_stage_index,
            stages_metadata=stages_metadata,
            pipeline_config=pipeline_config,
            stage_plan=local_stage_plan,
        )

        logger.info("Setting up and running pipeline")
        self.pipeline.setup_and_run()

        self._page_size_datums = page_size_bytes(1) // TOKEN_ID_BYTES
        self.position_id: int | None = None
        self._in_thinking_phase = False
        self.model: DeepSeekV3 | None = None
        if self.pipeline.my_stage_idx == 0:
            self.model = DeepSeekV3(
                write_fn=self.pipeline.write_token,
                read_fn=self.pipeline.read_output,
                batch_size=1,
                pipeline_depth=config.num_stages,
            )

            if io_socket_descriptor_prefix is not None:
                self.pipeline.export_host_socket_descriptors(io_socket_descriptor_prefix)

        logger.info(f"Created ModelPipeline for stage {self.pipeline.my_stage_idx}.")

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
        first_emit_time: float | None = None
        last_emit_time: float | None = None

        def is_eos(token_id: int) -> bool:
            return eos_token_id is not None and token_id == eos_token_id

        def emit(token_id: int) -> None:
            nonlocal first_emit_time, last_emit_time
            if on_token is not None:
                on_token(token_id)
            generated_tokens.append(token_id)
            now = time.time()
            if first_emit_time is None:
                first_emit_time = now
            last_emit_time = now

        prefill_start = time.time()
        pending: deque[DecodeResult] = deque(self.prefill_forward(prompt_token_ids))
        prefill_end = time.time()

        decode_start = time.time()
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

        decode_end = time.time()

        while num_reads < num_writes:
            self.model.read_result()
            num_reads += 1

        n_emitted = len(generated_tokens)
        decode_elapsed = decode_end - decode_start
        prefill_elapsed = prefill_end - prefill_start
        ttft = (first_emit_time - prefill_start) if first_emit_time is not None else float("nan")
        if first_emit_time is not None and last_emit_time is not None and n_emitted > 1:
            tpot_elapsed = last_emit_time - first_emit_time
            tps_steady = (n_emitted - 1) / max(tpot_elapsed, 1e-9)
        else:
            tps_steady = float("nan")
        tps_avg = n_emitted / max(decode_elapsed, 1e-9)
        logger.debug(
            "Prefill: {:.2f}s ({} tokens, {:.1f} tok/s)",
            prefill_elapsed,
            len(prompt_token_ids),
            len(prompt_token_ids) / max(prefill_elapsed, 1e-9),
        )
        logger.debug(f"TTFT (prefill + first decode token): {ttft:.3f}s")
        logger.debug(f"Decode wall-time (excl. drain): {decode_elapsed:.2f}s")
        logger.debug(f"Tokens per second (steady-state, inter-token): {tps_steady:.1f}")
        logger.debug(f"Tokens per second (avg over decode loop): {tps_avg:.1f}")
        logger.debug("Base decode generation complete ({} tokens generated)", n_emitted)
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

        first_emit_time: float | None = None
        last_emit_time: float | None = None

        def is_eos(token_id: int) -> bool:
            """Returns True if a token is the EOS token"""
            return eos_token_id is not None and token_id == eos_token_id

        def emit(token_id: int) -> None:
            """Emit a token to the caller and update thinking-phase state."""
            nonlocal first_emit_time, last_emit_time
            if on_token is not None:
                on_token(token_id)
            generated_tokens.append(token_id)
            now = time.time()
            if first_emit_time is None:
                first_emit_time = now
            last_emit_time = now
            if token_id == think_open_id:
                self._in_thinking_phase = True
            elif token_id == think_close_id:
                self._in_thinking_phase = False

        # --- Prefill --------------------------------------------------------
        prefill_start = time.time()
        prefill_results = self.prefill_forward(prompt_token_ids)
        prefill_end = time.time()

        # Seed the state machine with both pages from the last prefill write,
        # then read from the pipeline for all subsequent results.
        pending: deque[DecodeResult] = deque(prefill_results)
        base_accept = 0
        spec_accept = 0
        base_reject = 0
        spec_reject = 0
        # --- Speculative decode state machine --------------------------------
        decode_start = time.time()
        num_writes = 0
        num_reads = 0
        signal_to_exit = False
        while len(generated_tokens) < max_new_tokens or signal_to_exit:
            if pending:
                result = pending.popleft()
            else:
                result = self.model.read_result()
                num_reads += 1

            if not unverified_spec_tokens and not verified_spec_tokens:
                unverified_spec_tokens.append(result.token_1)
                emit(result.token_0)
            else:
                if result.token_0_type == TokenType.BASE:
                    if self.check_acceptance(unverified_spec_tokens[-1], result):
                        verified_spec_tokens.append(unverified_spec_tokens.pop())
                        emit(result.token_0)
                        base_accept += 1
                        signal_to_exit = is_eos(result.token_0) or len(generated_tokens) >= max_new_tokens
                        continue
                    else:
                        unverified_spec_tokens.pop()
                        unverified_spec_tokens.append(result.token_1)
                        emit(result.token_0)
                        base_reject += 1
                        signal_to_exit = is_eos(result.token_0) or len(generated_tokens) >= max_new_tokens

                if result.token_0_type == TokenType.SPEC:
                    if verified_spec_tokens:
                        verified_spec_tokens.pop()
                        unverified_spec_tokens.append(result.token_1)

                        if signal_to_exit:
                            break

                        emit(result.token_0)
                        spec_accept += 1
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

        decode_end = time.time()

        while num_reads < num_writes:
            self.model.read_result()
            num_reads += 1

        n_emitted = len(generated_tokens)
        decode_elapsed = decode_end - decode_start
        # TTFT = prefill + time to first emitted decode token.
        ttft = (first_emit_time - prefill_start) if first_emit_time is not None else float("nan")
        # Steady-state inter-token throughput excludes the first emission.
        if first_emit_time is not None and last_emit_time is not None and n_emitted > 1:
            tpot_elapsed = last_emit_time - first_emit_time
            tps_steady = (n_emitted - 1) / max(tpot_elapsed, 1e-9)
        else:
            tps_steady = float("nan")
        tps_avg = n_emitted / max(decode_elapsed, 1e-9)
        prefill_elapsed = prefill_end - prefill_start
        logger.debug(
            "Prefill: {:.2f}s ({} tokens, {:.1f} tok/s)",
            prefill_elapsed,
            len(prompt_token_ids),
            len(prompt_token_ids) / max(prefill_elapsed, 1e-9),
        )
        logger.debug(f"TTFT (prefill + first decode token): {ttft:.3f}s")
        logger.debug(f"Decode wall-time (excl. drain): {decode_elapsed:.2f}s")
        logger.debug(f"Tokens per second (steady-state, inter-token): {tps_steady:.1f}")
        logger.debug(f"Tokens per second (avg over decode loop): {tps_avg:.1f}")
        logger.debug(
            f"Base Accept: {base_accept}, Base Reject: {base_reject}, Spec Accept: {spec_accept}, Spec Reject: {spec_reject}, Base Accept Rate: {base_accept / (base_accept + base_reject + 1e-5)}, Spec Accept Rate: {spec_accept / (spec_accept + spec_reject + 1e-5)}"
        )
        logger.debug("Generation complete ({} tokens generated)", n_emitted)
        return generated_tokens if return_generated_tokens else None

    def barrier(self) -> None:
        self.pipeline.barrier()

    def terminate(self) -> None:
        self.pipeline.terminate()
