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
        num_mtp_levels: int = 1,
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
        self._num_mtp_levels = num_mtp_levels
        self.relaxed_acceptance_delta = relaxed_acceptance_delta
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        num_procs = int(ttnn.distributed_context_get_size())
        if num_procs not in (4, 16, 64, 66):
            raise RuntimeError(f"Pod pipeline requires 4, 16, 64, or 66 distributed processes; got {num_procs}")
        ttnn.enable_asynchronous_slow_dispatch(self.mesh_device)

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
            num_mtp_levels=self._num_mtp_levels,
            num_slots=num_slots,
        )
        if config.num_stages != num_procs:
            raise RuntimeError(f"Pipeline configuration has {config.num_stages} stages but {num_procs} processes")

        logger.info("Building pipeline")
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

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def prefill_forward(self, tokens: list[int]) -> list[DecodeResult]:
        """Prefill prompt tokens and return the DecodeResults from the last prompt token's outputs."""
        if self.pipeline.my_stage_idx != 0:
            raise RuntimeError("prefill_forward() should only be called on mesh id 0")
        assert self.model is not None
        logger.debug(f"Prefilling with {len(tokens)} tokens...")
        n = len(tokens)
        prompt_token_tensors = [
            to_spec_input(
                tokens[i],
                slot_id=0,
                position_id=i,
                page_size_datums=self._page_size_datums,
                lane_id=0,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                prefill_token_ids=[tokens[i + 1 + j] if i + 1 + j < n else -1 for j in range(self._num_mtp_levels)],
            )
            for i in range(n)
        ]
        results = self.model.prefill(prompt_token_tensors)
        self.position_id = len(tokens)
        logger.debug(f"Done prefilling with {len(tokens)} tokens.")
        return results

    # ------------------------------------------------------------------
    # Single-step decode (non-speculative)
    # ------------------------------------------------------------------

    def decode_forward(self, input_token: int) -> int:
        """Run 1 decode step and return the next token id."""
        if self.pipeline.my_stage_idx != 0:
            raise RuntimeError("decode_forward() should only be called on mesh id 0")
        assert self.model is not None
        if self.position_id is None:
            raise RuntimeError("decode_forward() requires prefill_forward() to be called first")

        self.model.write_input(
            input_token,
            slot_id=0,
            position_id=self.position_id,
            lane_id=0,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )
        result = self.model.read_result()
        self.position_id += 1
        return result.base_token

    # ------------------------------------------------------------------
    # Speculative-decode helpers
    # ------------------------------------------------------------------

    def _sampling_params(self) -> dict:
        return dict(
            temperature=self.temperature,
            top_k=self.top_k if self._in_thinking_phase else 1,
            top_p=self.top_p,
        )

    def _spec_chain(self, result: DecodeResult) -> list[tuple[int, int]]:
        """Extract (token_id, position) for each speculative level from a DecodeResult."""
        return [(result.spec_token(i), result.spec_position(i)) for i in range(self._num_mtp_levels)]

    def _write_spec_tokens(self, result: DecodeResult) -> int:
        """Write base + N spec tokens from a DecodeResult into the pipeline. Returns write count."""
        assert self.model is not None
        params = self._sampling_params()
        self.model.write_input(result.base_token, slot_id=0, position_id=result.position_id, lane_id=0, **params)
        for level, (token_id, pos) in enumerate(self._spec_chain(result)):
            self.model.write_input(token_id, slot_id=0, position_id=pos, lane_id=level + 1, **params)
        return 1 + self._num_mtp_levels

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

    # ------------------------------------------------------------------
    # Speculative-decode inference loop
    # ------------------------------------------------------------------

    def run_inference(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        on_token: Callable[[int], None] | None = None,
        eos_token_id: int | None = None,
        think_token_ids: list[int] | None = None,
        return_generated_tokens: bool = False,
    ) -> list[int] | None:
        """Run speculative-decode inference: prefill then decode with MTP-N.

        Each cycle writes 1+N tokens (base + N specs) and reads 1+N result pages.
        Acceptance walks the chain; the pivot result seeds the next cycle.
        """
        if self.pipeline.my_stage_idx != 0:
            raise RuntimeError("run_inference() should only be called on stage 0")
        assert max_new_tokens >= 1, f"max_new_tokens must be >= 1, got {max_new_tokens}"

        self._in_thinking_phase = False
        generated_tokens: list[int] = []
        if think_token_ids is not None:
            think_open_id, think_close_id = think_token_ids
        else:
            think_open_id, think_close_id = None, None

        def is_eos(token_id: int) -> bool:
            return eos_token_id is not None and token_id == eos_token_id

        def emit(token_id: int) -> None:
            if on_token is not None:
                on_token(token_id)
            generated_tokens.append(token_id)
            if token_id == think_open_id:
                self._in_thinking_phase = True
            elif token_id == think_close_id:
                self._in_thinking_phase = False

        def finished() -> bool:
            return bool(generated_tokens) and (is_eos(generated_tokens[-1]) or len(generated_tokens) >= max_new_tokens)

        # --- Prefill --------------------------------------------------------
        prefill_results = self.prefill_forward(prompt_token_ids)
        pending: deque[DecodeResult] = deque(prefill_results)

        tokens_per_cycle = 1 + self._num_mtp_levels
        depth_counts = [0] * tokens_per_cycle
        num_writes = 0
        num_reads = 0
        start_time = time.time()

        def read_one() -> DecodeResult:
            nonlocal num_reads
            if pending:
                return pending.popleft()
            num_reads += 1
            return self.model.read_result()

        # --- Seed: first result from prefill, no verification yet -----------
        seed = read_one()
        emit(seed.base_token)
        prev_chain = self._spec_chain(seed)
        pivot = seed

        # --- Speculative decode loop ----------------------------------------
        while not finished():
            
            num_writes += self._write_spec_tokens(pivot)

            cycle_results = [read_one() for _ in range(tokens_per_cycle)]

            depth = 0
            for i in range(self._num_mtp_levels):
                if self.check_acceptance(prev_chain[i][0], cycle_results[i]):
                    depth += 1
                else:
                    break
            depth_counts[depth] += 1

            for i in range(depth):
                emit(cycle_results[i].base_token)
                if finished():
                    break
            if finished():
                break

            pivot = cycle_results[depth]
            emit(pivot.base_token)
            prev_chain = self._spec_chain(pivot)

        # Drain remaining in-flight results
        while num_reads < num_writes:
            self.model.read_result()
            num_reads += 1

        elapsed = time.time() - start_time
        total_cycles = sum(depth_counts)
        logger.debug(f"Time taken: {elapsed:.2f}s")
        logger.debug(f"Tokens per second: {len(generated_tokens) / elapsed:.1f}")
        logger.debug(
            "Acceptance depth: {} ({} cycles)",
            ", ".join(f"d{i}={c}" for i, c in enumerate(depth_counts)),
            total_cycles,
        )
        logger.debug("Generation complete ({} tokens generated)", len(generated_tokens))
        return generated_tokens if return_generated_tokens else None

    def barrier(self) -> None:
        self.pipeline.barrier()

    def terminate(self) -> None:
        self.pipeline.terminate()
