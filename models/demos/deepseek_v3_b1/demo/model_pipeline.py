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
    ):
        logger.info(
            "Initializing DeepSeek V3 B1 pod pipeline (weights={}, lm_head_fp32={}, lm_head_persistent_mode={})",
            weights_mode,
            lm_head_fp32_dest_acc_en,
            lm_head_persistent_mode,
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
        ttnn.enable_asynchronous_slow_dispatch(self.mesh_device)

        # Each host loads/creates only the weights for its stage via the provider.
        if weights_mode == "real":
            if cache_path is None:
                raise ValueError("weights_mode='real' requires cache_path")
            if model_path is None:
                raise ValueError("weights_mode='real' requires model_path")
            provider: WeightProvider = CacheWeightProvider(cache_path, model_path)
        elif weights_mode == "state_dict":
            if model_path is None:
                raise ValueError("weights_mode='state_dict' requires model_path")
            provider = StateDictWeightProvider(model_path)
        elif weights_mode == "synthetic":
            provider = SyntheticWeightProvider()
        else:
            raise ValueError(f"Unknown weights_mode: {weights_mode!r}")
        config = create_pipeline_configuration_from_num_procs(
            num_procs,
            provider,
            fp32_dest_acc_en=lm_head_fp32_dest_acc_en,
            persistent_mode=lm_head_persistent_mode,
            dense_layer_id_override=dense_layer_id_override,
            moe_layer_id_override=moe_layer_id_override,
            enable_mtp=True,
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
