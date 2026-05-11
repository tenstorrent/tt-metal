# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import torch
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
from models.demos.deepseek_v3_b1.model import MAX_SPECULATIVE_TOKENS
from models.demos.deepseek_v3_b1.model import RELAXED_ACCEPT_TOPN as MAX_RELAXED_ACCEPT_TOPN
from models.demos.deepseek_v3_b1.model import (
    TOKEN_ID_BYTES,
    DecodeResult,
    DeepSeekV3,
    TokenType,
    page_size_bytes,
    to_spec_input,
)

DEFAULT_RELAXED_ACCEPT_TOPN = MAX_RELAXED_ACCEPT_TOPN
DEFAULT_RELAXED_ACCEPT_DELTA = 0.6


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
        num_speculative_tokens: int = 1,
        relaxed_accept_topn: int = DEFAULT_RELAXED_ACCEPT_TOPN,
        relaxed_accept_delta: float = DEFAULT_RELAXED_ACCEPT_DELTA,
        top_k: int = 32,
        top_p: float = 1.0,
        temperature: float = 0.6,
        enable_sram_hot_experts: bool = False,
        sram_hot_experts_ceiling: int = 64,
    ):
        if not 1 <= num_speculative_tokens <= MAX_SPECULATIVE_TOKENS:
            raise ValueError(
                f"num_speculative_tokens must be between 1 and {MAX_SPECULATIVE_TOKENS}, "
                f"got {num_speculative_tokens}"
            )
        if not 1 <= int(top_k) <= 32:
            raise ValueError(f"top_k must be between 1 and 32, got {top_k}")
        if not 0.0 <= float(top_p) <= 1.0:
            raise ValueError(f"top_p must be between 0 and 1, got {top_p}")
        if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
            raise ValueError(f"temperature must be a finite positive value, got {temperature}")
        self._set_relaxed_acceptance_params(relaxed_accept_topn, relaxed_accept_delta)
        logger.info(
            "Initializing DeepSeek V3 B1 pod pipeline "
            "(weights={}, lm_head_fp32={}, lm_head_persistent_mode={}, num_speculative_tokens={}, "
            "relaxed_accept_topn={}, relaxed_accept_delta={})",
            weights_mode,
            lm_head_fp32_dest_acc_en,
            lm_head_persistent_mode,
            num_speculative_tokens,
            self.relaxed_accept_topn,
            self.relaxed_accept_delta,
        )
        if not is_slow_dispatch():
            raise RuntimeError(
                "DeepSeek V3 B1 pod pipeline requires slow dispatch mode. Set TT_METAL_SLOW_DISPATCH_MODE=1 and rerun."
            )
        self.mesh_device = mesh_device
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.temperature = float(temperature)
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
                    worker_l1_size=_worker_l1_size_for_rank(num_procs),
                )
            else:
                provider = CacheWeightProvider(cache_path, model_path)
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
            num_speculative_tokens=num_speculative_tokens,
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
        self.num_speculative_tokens = num_speculative_tokens
        self.position_id: int | None = None
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

    def _set_relaxed_acceptance_params(self, relaxed_accept_topn: int, relaxed_accept_delta: float) -> None:
        topn = int(relaxed_accept_topn)
        if not 1 <= topn <= MAX_RELAXED_ACCEPT_TOPN:
            raise ValueError(
                f"relaxed_accept_topn must be between 1 and {MAX_RELAXED_ACCEPT_TOPN}, got {relaxed_accept_topn}"
            )

        delta = float(relaxed_accept_delta)
        if not math.isfinite(delta) or delta < 0.0:
            raise ValueError(f"relaxed_accept_delta must be a finite non-negative value, got {relaxed_accept_delta}")

        self.relaxed_accept_topn = topn
        self.relaxed_accept_delta = delta

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
                request_id=0,
                position_id=i,
                page_size_datums=self._page_size_datums,
                token_type=TokenType.PREFILL,
                lane_idx=0,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
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

        output = self.model.decode_step(
            to_spec_input(
                input_token,
                -1,
                request_id=0,
                position_id=self.position_id,
                page_size_datums=self._page_size_datums,
                token_type=TokenType.BASE,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )
        )
        self.position_id += 1

        next_token_id = int(ttnn.to_torch(output).to(torch.int32)[0, 0].item())
        return next_token_id

    def _relaxed_accepts_speculation(
        self,
        expected_token: int,
        result: DecodeResult,
    ) -> bool:
        """Return whether the speculative token is accepted by the base lane output."""
        if result.target_topn_tokens and result.target_topn_probs:
            active_topn = min(
                getattr(self, "relaxed_accept_topn", DEFAULT_RELAXED_ACCEPT_TOPN),
                len(result.target_topn_tokens),
                len(result.target_topn_probs),
            )
            target_topn_tokens = result.target_topn_tokens[:active_topn]
            target_topn_probs = result.target_topn_probs[:active_topn]
            if expected_token not in target_topn_tokens:
                return False
            top1_prob = target_topn_probs[0]
            spec_idx = target_topn_tokens.index(expected_token)
            return target_topn_probs[spec_idx] >= top1_prob - getattr(
                self, "relaxed_accept_delta", DEFAULT_RELAXED_ACCEPT_DELTA
            )

        return bool(result.token_ids) and int(result.token_ids[0]) == int(expected_token)

    def _record_speculations(
        self,
        result: DecodeResult,
        speculations_by_pos: dict[int, int],
        num_speculative_tokens: int,
        window_start_pos: int | None = None,
    ) -> None:
        """Record speculative candidate slots from a result for future acceptance checks."""
        required_slots = num_speculative_tokens + 1
        token_ids = result.token_ids
        positions = result.positions
        if len(token_ids) < required_slots or len(positions) < required_slots:
            raise RuntimeError(f"Spec-decode result is missing speculative candidate metadata: {result}")

        window_start_pos = int(positions[0]) if window_start_pos is None else int(window_start_pos)
        for candidate_idx in range(1, required_slots):
            expected_position = window_start_pos + candidate_idx
            if int(positions[candidate_idx]) != expected_position:
                raise RuntimeError(
                    "Speculative candidate positions must be contiguous from window_start_pos: "
                    f"window_start_pos={window_start_pos}, candidate_idx={candidate_idx}, "
                    f"candidate_position={positions[candidate_idx]}, result={result}"
                )
            speculations_by_pos[expected_position] = int(token_ids[candidate_idx])

    def _write_speculation_window(
        self,
        result: DecodeResult,
        *,
        request_id: int = 0,
        num_speculative_tokens: int | None = None,
        window_start_pos: int | None = None,
    ) -> int:
        """Write base lane plus available speculative lanes into the pipeline."""
        assert self.model is not None
        depth = self.num_speculative_tokens if num_speculative_tokens is None else num_speculative_tokens
        required_slots = depth + 1
        token_ids = result.token_ids
        positions = result.positions
        if len(token_ids) < required_slots or len(positions) < required_slots:
            raise RuntimeError(f"Spec-decode result has too few candidate slots to write MTP{depth}: {result}")

        window_start_pos = int(positions[0]) if window_start_pos is None else int(window_start_pos)
        for lane_idx in range(required_slots):
            expected_position = window_start_pos + lane_idx
            if int(positions[lane_idx]) != expected_position:
                raise RuntimeError(
                    "Spec-decode candidate positions must be contiguous from window_start_pos: "
                    f"window_start_pos={window_start_pos}, lane_idx={lane_idx}, "
                    f"candidate_position={positions[lane_idx]}, result={result}"
                )
            self.model.write_input(
                int(token_ids[lane_idx]),
                -1,
                request_id,
                expected_position,
                token_type=TokenType.BASE if lane_idx == 0 else TokenType.SPEC,
                lane_idx=lane_idx,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )
        return required_slots

    def verify_packet_window(
        self,
        packets: list[DecodeResult],
        *,
        num_speculative_tokens: int,
        allow_prefill: bool = False,
    ) -> dict[int, DecodeResult]:
        """Validate one complete device-to-host packet window and return packets by lane."""
        required_slots = num_speculative_tokens + 1
        if not packets:
            raise RuntimeError("Speculative packet window is empty")
        if allow_prefill and len(packets) != 1:
            raise RuntimeError(f"Bootstrap expects exactly one PREFILL packet, got {len(packets)}")
        if not allow_prefill and len(packets) != required_slots:
            raise RuntimeError(
                f"MTP{num_speculative_tokens} requires {required_slots} packets per window, got {len(packets)}"
            )

        results_by_lane: dict[int, DecodeResult] = {}
        for result in packets:
            if allow_prefill:
                if result.token_type != TokenType.PREFILL:
                    raise RuntimeError(f"Bootstrap packet must be PREFILL: {result}")
                if result.lane_idx != 0:
                    raise RuntimeError(f"Bootstrap PREFILL packet must use lane 0: {result}")
            elif result.token_type == TokenType.PREFILL:
                raise RuntimeError(f"PREFILL packet is only valid during bootstrap: {result}")

            if result.lane_idx in results_by_lane:
                raise RuntimeError(f"Duplicate speculative lane {result.lane_idx} in one round")
            results_by_lane[result.lane_idx] = result

            positions = result.positions
            if len(result.token_ids) < required_slots or len(positions) < required_slots:
                raise RuntimeError(
                    f"MTP{num_speculative_tokens} requires {required_slots} candidate slots, got {result}"
                )

            lane_start_pos = int(result.window_start_pos) + int(result.lane_idx)
            for candidate_idx in range(required_slots):
                expected_position = lane_start_pos + candidate_idx
                if int(positions[candidate_idx]) != expected_position:
                    raise RuntimeError(
                        "MTP candidate positions must be contiguous from window_start_pos + lane_idx: "
                        f"window_start_pos={result.window_start_pos}, lane_idx={result.lane_idx}, "
                        f"candidate_idx={candidate_idx}, candidate_position={positions[candidate_idx]}, "
                        f"result={result}"
                    )

        if allow_prefill:
            return results_by_lane

        missing_lanes = [lane_idx for lane_idx in range(required_slots) if lane_idx not in results_by_lane]
        if missing_lanes:
            raise RuntimeError(f"Speculative round is missing lane(s): {missing_lanes}")

        for lane_idx in range(required_slots):
            result = results_by_lane[lane_idx]
            expected_type = TokenType.BASE if lane_idx == 0 else TokenType.SPEC
            if result.token_type != expected_type:
                expected_name = "BASE" if expected_type == TokenType.BASE else "SPEC"
                raise RuntimeError(f"Lane {lane_idx} must be a {expected_name} packet: {result}")

        window_start_pos = packets[0].window_start_pos
        if any(result.window_start_pos != window_start_pos for result in packets):
            raise RuntimeError(f"Speculative round has inconsistent window_start_pos: {packets}")

        base_lane = results_by_lane[0]
        if int(base_lane.positions[0]) != int(window_start_pos):
            raise RuntimeError(
                "window_start_pos must equal tokens[0].pos of the base lane: "
                f"window_start_pos={window_start_pos}, base_lane={base_lane}"
            )

        return results_by_lane

    def run_inference(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        on_token: Callable[[int], None] | None = None,
        eos_token_id: int | None = None,
        return_generated_tokens: bool = False,
    ) -> list[int] | None:
        """Run dynamic-depth speculative decode with relaxed acceptance.

        Each device round returns one BASE lane and N SPEC lanes for the same
        `window_start_pos`, where N is `num_speculative_tokens`. The host commits
        the longest accepted speculative prefix and then reseeds the next window
        from the lane that owns the first uncommitted token.
        """
        if self.pipeline.my_stage_idx != 0:
            raise RuntimeError("run_inference() should only be called on stage 0")
        assert max_new_tokens >= 1, f"max_new_tokens must be >= 1, got {max_new_tokens}"
        assert self.model is not None
        num_speculative_tokens = getattr(self, "num_speculative_tokens", 1)
        if not 1 <= num_speculative_tokens <= MAX_SPECULATIVE_TOKENS:
            raise ValueError(
                f"num_speculative_tokens must be between 1 and {MAX_SPECULATIVE_TOKENS}, "
                f"got {num_speculative_tokens}"
            )

        generated_tokens: list[int] = []
        speculations_by_pos: dict[int, int] = {}
        num_accepts = 0
        num_rejects = 0
        num_reads = 0
        num_writes = 0

        def is_eos(token_id: int) -> bool:
            return eos_token_id is not None and token_id == eos_token_id

        def emit(token_id: int) -> None:
            if on_token is not None:
                on_token(token_id)
            generated_tokens.append(token_id)

        def emit_committed_prefix(
            results_by_lane: dict[int, DecodeResult],
            accepted_tokens_by_lane: dict[int, int],
            accepts: int,
        ) -> tuple[bool, int, bool]:
            committed_accepts = 0
            emitted_owner_token = False
            for lane_idx in range(accepts + 1):
                if lane_idx < accepts:
                    token_id = int(accepted_tokens_by_lane[lane_idx])
                    committed_accepts += 1
                else:
                    token_id = int(results_by_lane[lane_idx].token_ids[0])
                    emitted_owner_token = True
                emit(token_id)
                if is_eos(token_id) or len(generated_tokens) >= max_new_tokens:
                    return True, committed_accepts, emitted_owner_token
            return False, committed_accepts, emitted_owner_token

        start_time = time.time()
        prefill_results = self.prefill_forward(prompt_token_ids)
        if not prefill_results:
            raise RuntimeError("prefill_forward() returned no results")

        bootstrap = self.verify_packet_window(
            [prefill_results[-1]],
            num_speculative_tokens=num_speculative_tokens,
            allow_prefill=True,
        )[0]
        emit(int(bootstrap.token_ids[0]))
        if not is_eos(int(bootstrap.token_ids[0])) and len(generated_tokens) < max_new_tokens:
            self._record_speculations(bootstrap, speculations_by_pos, num_speculative_tokens)
            num_writes += self._write_speculation_window(bootstrap, num_speculative_tokens=num_speculative_tokens)

        while len(generated_tokens) < max_new_tokens:
            # Wait for all packets for current speculative round to arrive
            round_results: list[DecodeResult] = []
            for _ in range(num_speculative_tokens + 1):
                result = self.model.read_result()
                num_reads += 1
                round_results.append(result)

            # Verify that all packets are valid
            results_by_lane = self.verify_packet_window(
                round_results,
                num_speculative_tokens=num_speculative_tokens,
            )

            # Check which speculative lanes are accepted
            round_accepts = 0
            rejected_expected_token = False
            accepted_tokens_by_lane: dict[int, int] = {}
            for lane_idx in range(num_speculative_tokens):
                result = results_by_lane[lane_idx]
                result_pos = int(result.positions[0])
                expected_token = speculations_by_pos.get(result_pos)
                if expected_token is None:
                    raise RuntimeError(
                        f"{'BASE' if lane_idx == 0 else 'SPEC'} result at position {result_pos} "
                        f"has no pending speculation: {result}"
                    )
                if not self._relaxed_accepts_speculation(expected_token, result):
                    rejected_expected_token = True
                    break
                accepted_tokens_by_lane[lane_idx] = int(expected_token)
                round_accepts += 1

            for result in round_results:
                speculations_by_pos.pop(int(result.positions[0]), None)

            # Owner lane is the lane with lane_id=round_accepts
            owner = results_by_lane.get(round_accepts)

            # Emit the committed tokens and check if the generation should stop
            should_stop, committed_accepts, emitted_owner_token = emit_committed_prefix(
                results_by_lane,
                accepted_tokens_by_lane,
                round_accepts,
            )
            num_accepts += committed_accepts
            if rejected_expected_token and emitted_owner_token:
                num_rejects += 1
            if should_stop:
                break

            # Record next speculation window and write to the pipeline
            next_window_start_pos = int(owner.positions[0])
            self._record_speculations(
                owner,
                speculations_by_pos,
                num_speculative_tokens,
                window_start_pos=next_window_start_pos,
            )
            num_writes += self._write_speculation_window(
                owner,
                num_speculative_tokens=num_speculative_tokens,
                window_start_pos=next_window_start_pos,
            )

        # Drain remaining reads from the pipeline
        while num_reads < num_writes:
            self.model.read_result()
            num_reads += 1

        self.last_inference_stats = {
            "num_accepts": num_accepts,
            "num_rejects": num_rejects,
        }

        end_time = time.time()
        logger.debug(f"Time taken: {end_time - start_time} seconds")
        logger.debug(f"Tokens per second: {len(generated_tokens) / (end_time - start_time)}")
        logger.debug(
            f"Accept: {num_accepts}, Reject: {num_rejects}, Accept Rate: {num_accepts / (num_accepts + num_rejects + 1e-5)}"
        )
        logger.debug("Generation complete ({} tokens generated)", len(generated_tokens))
        return generated_tokens if return_generated_tokens else None

    def barrier(self) -> None:
        self.pipeline.barrier()

    def terminate(self) -> None:
        self.pipeline.terminate()
