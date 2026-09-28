# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import os
import pickle
import queue
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, TypeVar, cast

import torch
import torch.distributed as dist

from vllm.config import ParallelConfig, VllmConfig
from vllm.utils.network_utils import get_tcp_uri
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.engine import (
    EngineCoreOutputs,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
)
from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.request import Request
from vllm_tt_plugin.config import get_tt_config, get_tt_per_lane_max_num_seqs
from vllm_tt_plugin.logger import init_tt_logger
from vllm_tt_plugin.scheduler import TTSchedulingMode

logger = init_tt_logger(__name__)
_T = TypeVar("_T")


def _normal_init_dp_group(parallel_config: ParallelConfig) -> dist.ProcessGroup:
    """Create the TT engine DP group with rooted collectives enabled."""
    from torch.distributed import DistNetworkError

    if dist.is_initialized():
        raise RuntimeError(
            "TT DP gather requires a fresh default torch.distributed process "
            "group in the engine process."
        )

    max_retries = 5
    last_exc: Exception | None = None
    for _ in range(max_retries):
        init_method = get_tcp_uri(
            parallel_config.data_parallel_master_ip,
            parallel_config.get_next_dp_init_port(),
        )
        try:
            dist.init_process_group(
                backend="gloo",
                init_method=init_method,
                rank=parallel_config.data_parallel_rank,
                world_size=parallel_config.data_parallel_size,
            )
            return dist.group.WORLD
        except DistNetworkError as e:
            if "EADDRINUSE" in str(e):
                logger.warning("Address already in use. Retrying with a new port.")
                last_exc = e
                continue
            raise

    assert last_exc is not None
    raise last_exc


@dataclass
class DPGatherHandle:
    future: Future[tuple[torch.Tensor, list]]
    scheduler_output: SchedulerOutput | None
    local_has_requests: bool
    is_decode: bool
    overlap_ok: bool
    any_needs_logprobs: bool
    intermediate_prefill_mask: torch.Tensor | None
    req_ids: list[str]
    req_id_to_index: dict[str, int]


class TTDPEngineCoreProc(DPEngineCoreProc):
    """TT data-parallel engine core with gathered-batch orchestration."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        DEBUG_DPG = os.environ.get("DP_GATHER_DEBUG") == "1"

        def dlog_logger(msg: str, *a: object) -> None:
            if DEBUG_DPG:
                formatted = (msg % a) if a else msg
                logger.info("dp_gather r%d: %s", self.dp_rank, formatted)

        self.dlog = dlog_logger
        super().__init__(vllm_config, *args, **kwargs)
        self._dp_in_flight: DPGatherHandle | None = None
        if self.batch_queue is not None:
            self.step_fn = self.step_dp_with_batch_queue

    def _process_input_queue(self) -> None:
        waited = False
        while (
            not self.engines_running
            and not self.scheduler.has_requests()
            and not self.batch_queue
            and not self._dp_in_flight
            and not self._scheduler_paused
        ):
            # Idle TT ranks must keep progressing collectives, so do not block
            # indefinitely waiting for client input.
            try:
                req = self.input_queue.get_nowait()
                self._handle_client_request(*req)
                waited = True
            except queue.Empty:
                break

        if waited:
            logger.debug("EngineCore loop active.")

        delay = get_tt_config(self.vllm_config).get("input_queue_batching_delay", 0.002)

        def _should_add_queue_delay() -> bool:
            if delay <= 0:
                return False
            num_running, num_waiting = self.scheduler.get_request_counts()
            has_running = num_running > 0
            max_batch_waiting = (
                num_waiting >= self.vllm_config.scheduler_config.max_num_seqs
            )
            return not has_running and not max_batch_waiting

        if _should_add_queue_delay():
            import time

            time.sleep(delay)

        while not self.input_queue.empty():
            req = self.input_queue.get_nowait()
            self._handle_client_request(*req)
            if self.input_queue.empty() and _should_add_queue_delay():
                import time

                time.sleep(delay)

    def _init_tt_dp_group(self, parallel_config: ParallelConfig) -> None:
        self.dp_group = _normal_init_dp_group(parallel_config)

        local_dp_rank = parallel_config.data_parallel_rank_local
        dp_size = parallel_config.data_parallel_size
        local_dp_rank_tensor = torch.tensor(
            [local_dp_rank], dtype=torch.int32, device="cpu"
        )
        gathered_local_ranks = [
            torch.zeros(1, dtype=torch.int32) for _ in range(dp_size)
        ]
        dist.all_gather(gathered_local_ranks, local_dp_rank_tensor, group=self.dp_group)
        self.dp_device_ranks = [
            i
            for i, rank_tensor in enumerate(gathered_local_ranks)
            if rank_tensor.item() == 0
        ]
        logger.info("DP device ranks: %s", self.dp_device_ranks)

    def _init_data_parallel(self, vllm_config: VllmConfig) -> None:
        parallel_config = vllm_config.parallel_config
        dp_rank = parallel_config.data_parallel_rank
        dp_size = parallel_config.data_parallel_size
        local_dp_rank = parallel_config.data_parallel_rank_local

        assert dp_size > 1
        assert local_dp_rank is not None
        assert 0 <= local_dp_rank <= dp_rank < dp_size

        self.dp_rank = dp_rank
        self._init_tt_dp_group(parallel_config)

    def shutdown(self) -> None:
        EngineCoreProc.shutdown(self)
        if dp_group := getattr(self, "dp_group", None):
            dist.destroy_process_group(dp_group)

    def add_request(self, request: Request, request_wave: int = 0) -> None:
        start_wave = False
        if self.has_coordinator and request_wave != self.current_wave:
            if request_wave > self.current_wave:
                self.current_wave = request_wave
            elif not self.engines_running:
                # Request received for an already-completed wave, notify
                # front-end that we need to start the next one.
                start_wave = True

        if self.has_coordinator and not self.engines_running:
            # The front-end normally notifies the coordinator before sending
            # the first request in a new wave. If that notification races with
            # wave completion state, this rank must still wake its peers before
            # entering TT gathered-DP collectives.
            self.engines_running = True
            start_wave = True

        if start_wave:
            self.output_queue.put_nowait(
                (-1, EngineCoreOutputs(start_wave=self.current_wave))
            )

        super().add_request(request, request_wave)

    def run_busy_loop(self) -> None:
        while True:
            # Rendezvous all DP ranks at iteration start to prevent
            # FIFO-collective skew accumulation across iterations.
            # gloo collectives are matched in call order per group, so once
            # ranks drift by one iteration, every subsequent collective can
            # deadlock waiting for a future peer call.
            try:
                dist.barrier(group=self.dp_group)
            except RuntimeError as e:
                if "Connection closed by peer" in str(e):
                    raise SystemExit() from e
                raise

            self._process_input_queue()
            self._process_engine_step()
            self._maybe_publish_request_counts()

            local_unfinished_reqs = self.scheduler.has_unfinished_requests()

            # TT does not call execute_dummy_batch() on idle steps because
            # _dp_any_rank_has_scheduler_requests() already synchronises all
            # ranks before any execution is attempted.  Rank alignment for the
            # wave-finish all-reduce happens inside _has_global_unfinished_reqs.
            self.engines_running = self._has_global_unfinished_reqs(
                local_unfinished_reqs
            )

            if not self.engines_running:
                if self.dp_rank == 0 or not self.has_coordinator:
                    logger.debug(
                        "Wave %d finished, pausing engine loop.", self.current_wave
                    )
                    client_index = -1 if self.has_coordinator else 0
                    self.output_queue.put_nowait(
                        (
                            client_index,
                            EngineCoreOutputs(wave_complete=self.current_wave),
                        )
                    )
                    if self.has_coordinator and client_index == -1:
                        self.output_queue.put_nowait(
                            (
                                0,
                                EngineCoreOutputs(
                                    wave_complete=self.current_wave,
                                    scheduler_stats=self.scheduler.make_stats(),
                                ),
                            )
                        )
                self.current_wave += 1
                self.step_counter = 0

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        dist.destroy_process_group(self.dp_group)
        self.shutdown()

        parallel_config = self.vllm_config.parallel_config
        old_dp_size = parallel_config.data_parallel_size
        parallel_config.data_parallel_size = reconfig_request.new_data_parallel_size
        if reconfig_request.new_data_parallel_rank != -1:
            parallel_config.data_parallel_rank = reconfig_request.new_data_parallel_rank
        assert (
            reconfig_request.new_data_parallel_rank_local
            == ReconfigureRankType.KEEP_CURRENT_RANK
        )
        parallel_config.data_parallel_master_ip = (
            reconfig_request.new_data_parallel_master_ip
        )
        parallel_config.data_parallel_master_port = (
            reconfig_request.new_data_parallel_master_port
        )
        if reconfig_request.new_data_parallel_rank != -2:
            self.dp_rank = parallel_config.data_parallel_rank
            self._init_tt_dp_group(parallel_config)
        reconfig_request.new_data_parallel_master_port = (
            parallel_config.data_parallel_master_port
        )

        self.model_executor.reinitialize_distributed(reconfig_request)
        if reconfig_request.new_data_parallel_size > old_dp_size:
            assert self.available_gpu_memory_for_kv_cache > 0
            ParallelConfig.sync_kv_cache_memory_size(
                self.dp_group, self.available_gpu_memory_for_kv_cache
            )
            self.model_executor.collective_rpc("compile_or_warm_up_model")
        if (
            reconfig_request.new_data_parallel_rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
        ):
            self.shutdown()
            logger.info("TTDPEngineCoreProc %s shutdown", self.dp_rank)
        else:
            logger.info(
                "Distributed environment reinitialized for DP rank %s", self.dp_rank
            )

    def _dp_any_rank_has_scheduler_requests(self) -> bool:
        local_has_requests = 1 if self.scheduler.has_requests() else 0
        has_requests_t = torch.tensor([local_has_requests], dtype=torch.int32)
        try:
            dist.all_reduce(has_requests_t, op=dist.ReduceOp.SUM, group=self.dp_group)
        except RuntimeError as e:
            # During shutdown, peers may close connections mid-collective.
            if "Connection closed by peer" in str(e):
                logger.debug("Collective failed during shutdown, exiting gracefully")
                raise SystemExit() from e
            raise
        return int(has_requests_t.item()) > 0

    def _dp_negotiate_forced_mode(self) -> TTSchedulingMode:
        running = getattr(self.scheduler, "running", [])
        has_running = bool(running)
        has_waiting = bool(getattr(self.scheduler, "waiting", False))
        max_running = getattr(self.scheduler, "max_num_running_reqs", 0)
        has_partial_prefill = any(request.is_prefill_chunk for request in running)
        has_capacity = len(running) < max_running
        local_prefill_intent = (
            1
            if (
                has_partial_prefill
                or (has_waiting and ((not has_running) or has_capacity))
            )
            else 0
        )
        intent_tensor = torch.tensor([local_prefill_intent], dtype=torch.int32)
        self.dlog("before_intent_allreduce intent_tensor=%s", intent_tensor)
        dist.all_reduce(intent_tensor, op=dist.ReduceOp.MAX, group=self.dp_group)
        forced_mode = TTSchedulingMode.from_prefill_intent(int(intent_tensor.item()))
        self.dlog("after_intent_allreduce forced_mode=%s", forced_mode)
        self._dp_gather_forced_mode = forced_mode
        return forced_mode

    def _dp_apply_forced_mode(self, forced_mode: TTSchedulingMode) -> None:
        set_mode = getattr(self.scheduler, "set_forced_mode", None)
        if callable(set_mode):
            set_mode(forced_mode)

    def _dp_schedule_with_zero_prefill_fallback(
        self,
        forced_mode: TTSchedulingMode,
        scheduler_output: SchedulerOutput | None,
    ) -> SchedulerOutput | None:
        if forced_mode != TTSchedulingMode.PREFILL_ONLY:
            return scheduler_output

        local_tokens = (
            scheduler_output.total_num_scheduled_tokens if scheduler_output else 0
        )
        local_has_decode = any(
            not request.is_prefill_chunk
            for request in getattr(self.scheduler, "running", [])
        )
        # Both quantities are non-negative, so a single SUM answers both:
        # zero total tokens, and zero iff no rank has a running decode.
        probe_t = torch.tensor([local_tokens, int(local_has_decode)], dtype=torch.int32)
        dist.all_reduce(probe_t, op=dist.ReduceOp.SUM, group=self.dp_group)
        global_tokens, ranks_with_decode = (int(v) for v in probe_t.tolist())

        if global_tokens != 0 or ranks_with_decode == 0:
            return scheduler_output

        self._dp_gather_forced_mode = TTSchedulingMode.DECODE_ONLY
        self._dp_apply_forced_mode(TTSchedulingMode.DECODE_ONLY)

        decode_output = (
            self.scheduler.schedule()
            if self.scheduler.has_unfinished_requests()
            else None
        )
        if decode_output is None:
            return scheduler_output

        if scheduler_output is not None:
            decode_output.finished_req_ids |= scheduler_output.finished_req_ids
            decode_output.free_encoder_mm_hashes = (
                scheduler_output.free_encoder_mm_hashes
                + decode_output.free_encoder_mm_hashes
            )

        return decode_output

    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        if self._scheduler_paused:
            return {}, False

        if not self._dp_any_rank_has_scheduler_requests():
            return {}, False

        forced_mode = self._dp_negotiate_forced_mode()
        has_local_requests = self.scheduler.has_requests()
        self._dp_apply_forced_mode(forced_mode)
        scheduler_output = self.scheduler.schedule() if has_local_requests else None
        scheduler_output = self._dp_schedule_with_zero_prefill_fallback(
            forced_mode, scheduler_output
        )
        self._dp_apply_forced_mode(TTSchedulingMode.DEFAULT)
        if not has_local_requests:
            _ = self._execute_model_dp_gather(None, None)
            return {}, False

        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        model_output = self._execute_model_dp_gather(scheduler_output, grammar_output)
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0

    def step_dp_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        assert self.batch_queue is not None

        global_has_requests = self._dp_any_rank_has_scheduler_requests()
        prev_handle = self._dp_in_flight
        if not global_has_requests and prev_handle is None:
            return {}, False

        forced_mode = TTSchedulingMode.DEFAULT
        scheduler_output: SchedulerOutput | None = None
        grammar_output: GrammarOutput | None = None
        model_executed = False
        current_overlap_ok = False
        if global_has_requests:
            forced_mode = self._dp_negotiate_forced_mode()
            if self.scheduler.has_requests():
                self._dp_apply_forced_mode(forced_mode)
                scheduler_output = self.scheduler.schedule()
            scheduler_output = self._dp_schedule_with_zero_prefill_fallback(
                forced_mode, scheduler_output
            )
            forced_mode = self._dp_gather_forced_mode
            self._dp_apply_forced_mode(TTSchedulingMode.DEFAULT)
            if scheduler_output is not None:
                if not self.is_ec_producer:
                    model_executed = scheduler_output.total_num_scheduled_tokens > 0
                if not scheduler_output.pending_structured_output_tokens:
                    grammar_output = self.scheduler.get_grammar_bitmask(
                        scheduler_output
                    )
            if forced_mode == TTSchedulingMode.DECODE_ONLY:
                current_overlap_ok = self._dp_can_attempt_steady_decode_from_scheduler(
                    scheduler_output, grammar_output
                )

        def _finalize_previous(
            handle: DPGatherHandle,
        ) -> dict[int, EngineCoreOutputs]:
            model_output = self.dp_gather_finalize(handle)
            if handle.scheduler_output is None:
                return {}
            return self.scheduler.update_from_output(
                handle.scheduler_output, model_output
            )

        # Always finalize the previous step before submitting the next one.
        #
        # The submit reads ``input_batch.token_ids_cpu`` to build the decode
        # input for the next step; that table is only updated once
        # ``apply_dp_execution_result`` runs inside ``_finalize_previous``. The
        # original overlap path (submit-next then finalize-prev) therefore
        # built the next step's input from stale token state, so the device
        # re-sampled the previous step's near-deterministic position — most
        # visibly as doubled ``<|end|>`` and ``<|start|>assistant`` tokens,
        # which break harmony parsing and silently null out chat responses.
        finalize_before_submit = prev_handle is not None

        engine_core_outputs: dict[int, EngineCoreOutputs] | None = {}
        if finalize_before_submit:
            assert prev_handle is not None
            engine_core_outputs = _finalize_previous(prev_handle)
            prev_handle = None

        if (
            scheduler_output is not None
            and grammar_output is None
            and scheduler_output.pending_structured_output_tokens
        ):
            grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)

        next_handle: DPGatherHandle | None = None
        if global_has_requests:
            next_handle = self.dp_gather_submit(
                scheduler_output,
                grammar_output,
                overlap_ok=current_overlap_ok,
            )

        if not finalize_before_submit and prev_handle is not None:
            engine_core_outputs = _finalize_previous(prev_handle)

        self._dp_in_flight = next_handle

        if not global_has_requests:
            return engine_core_outputs, False

        return engine_core_outputs, model_executed

    def _dp_can_attempt_steady_decode_from_scheduler(
        self,
        scheduler_output: SchedulerOutput | None,
        grammar_output: GrammarOutput | None,
    ) -> bool:
        local_overlap_ok = int(
            self.model_executor.collective_rpc(
                "can_attempt_steady_dp_decode_from_scheduler",
                args=(scheduler_output, grammar_output),
            )[0]
        )
        overlap_ok_t = torch.tensor([local_overlap_ok], dtype=torch.int32)
        dist.all_reduce(overlap_ok_t, op=dist.ReduceOp.MIN, group=self.dp_group)
        overlap_ok = bool(overlap_ok_t.item())
        self.dlog("steady_decode_overlap_ok=%s", overlap_ok)
        return overlap_ok

    def dp_gather_submit(
        self,
        scheduler_output: SchedulerOutput | None,
        grammar_output: GrammarOutput | None,
        *,
        overlap_ok: bool = False,
    ) -> DPGatherHandle:
        parallel_config = self.vllm_config.parallel_config
        group = self.dp_group
        rank = self.dp_rank
        local_rank = parallel_config.data_parallel_rank_local
        world = parallel_config.data_parallel_size

        local_has_requests = scheduler_output is not None
        if scheduler_output is not None:
            self.dlog(
                "enter_gather tokens=%d",
                scheduler_output.total_num_scheduled_tokens,
            )

        assert hasattr(self, "_dp_gather_forced_mode"), "forced_mode not set"
        is_decode = self._dp_gather_forced_mode == TTSchedulingMode.DECODE_ONLY

        all_local_inputs = self.model_executor.collective_rpc(
            "build_dp_model_input", args=(scheduler_output, grammar_output)
        )[0]
        (
            local_input,
            local_max_blocks,
            local_has_structured,
            local_has_penalties,
            local_reset_batch,
            local_can_sample_device,
            local_needs_logprobs,
            intermediate_prefill_mask,
            req_ids,
            req_id_to_index,
        ) = all_local_inputs
        max_blocks_decode = None
        any_structured_inputs = False
        any_needs_logprobs = False

        gathered_inputs: Any = None
        if is_decode:
            input_info_t = torch.tensor(
                [
                    local_max_blocks,
                    local_has_structured,
                    local_has_penalties,
                    local_reset_batch,
                    1 - local_can_sample_device,
                    local_needs_logprobs,
                ],
                dtype=torch.int32,
            )
            dist.all_reduce(input_info_t, op=dist.ReduceOp.MAX, group=group)
            max_blocks_decode = int(input_info_t[0].item())
            any_structured_inputs = input_info_t[1].item() > 0
            any_penalties_inputs = input_info_t[2].item() > 0
            any_reset_batch = input_info_t[3].item() > 0
            all_sample_device = input_info_t[4].item() == 0
            any_needs_logprobs = input_info_t[5].item() > 0

            decode_inputs: dict[str, Any] = self.model_executor.collective_rpc(
                "build_dp_decode_gather_input",
                args=(
                    local_input,
                    max_blocks_decode,
                    any_structured_inputs,
                    any_penalties_inputs,
                ),
            )[0]

            int_local = decode_inputs["int_inputs"]
            float_local = decode_inputs["float_inputs"]

            stacked_int = None
            stacked_float = None
            gather_list_int = None
            gather_list_float = None
            if rank == 0:
                stacked_int = torch.empty(
                    (world, *int_local.shape), dtype=int_local.dtype
                )
                stacked_float = torch.empty(
                    (world, *float_local.shape), dtype=float_local.dtype
                )
                gather_list_int = [stacked_int[i] for i in range(world)]
                gather_list_float = [stacked_float[i] for i in range(world)]

            dist.gather(int_local, gather_list_int, dst=0, group=group)
            dist.gather(float_local, gather_list_float, dst=0, group=group)
            if len(self.dp_device_ranks) > 1:
                if rank == 0:
                    for dst in self.dp_device_ranks[1:]:
                        dist.send(stacked_int, dst=dst, group=group)
                        dist.send(stacked_float, dst=dst, group=group)
                elif local_rank == 0:
                    stacked_int = torch.empty(
                        (world, *int_local.shape), dtype=int_local.dtype
                    )
                    stacked_float = torch.empty(
                        (world, *float_local.shape), dtype=float_local.dtype
                    )
                    dist.recv(stacked_int, src=0, group=group)
                    dist.recv(stacked_float, src=0, group=group)

            gathered_tokens_inputs = None
            if any_penalties_inputs and (not all_sample_device or any_reset_batch):
                if rank == 0:
                    gathered_tokens_inputs = [None for _ in range(world)]
                local_tokens_inputs = decode_inputs["sampling_tokens_inputs"]
                dist.gather_object(
                    local_tokens_inputs, gathered_tokens_inputs, dst=0, group=group
                )

                if len(self.dp_device_ranks) > 1:
                    if rank == 0:
                        pickled_tokens = pickle.dumps(gathered_tokens_inputs)
                        tokens_tensor = torch.frombuffer(
                            pickled_tokens, dtype=torch.uint8
                        )
                        tokens_size = torch.tensor(
                            [tokens_tensor.numel()], dtype=torch.long
                        )
                        for dst in self.dp_device_ranks[1:]:
                            dist.send(tokens_size, dst=dst, group=group)
                            dist.send(tokens_tensor, dst=dst, group=group)
                    elif local_rank == 0:
                        tokens_size = torch.zeros(1, dtype=torch.long)
                        dist.recv(tokens_size, src=0, group=group)
                        tokens_tensor = torch.empty(
                            tokens_size.item(), dtype=torch.uint8
                        )
                        dist.recv(tokens_tensor, src=0, group=group)
                        gathered_tokens_inputs = pickle.loads(
                            tokens_tensor.numpy().tobytes()
                        )

            gathered_host_only_sample_params = None
            if not all_sample_device:
                if rank == 0:
                    gathered_host_only_sample_params = [None for _ in range(world)]
                local_host_only_sample_params = decode_inputs.get(
                    "host_only_sample_params"
                )
                dist.gather_object(
                    local_host_only_sample_params,
                    gathered_host_only_sample_params,
                    dst=0,
                    group=group,
                )

                if len(self.dp_device_ranks) > 1:
                    if rank == 0:
                        pickled_host_only = pickle.dumps(
                            gathered_host_only_sample_params
                        )
                        host_only_tensor = torch.frombuffer(
                            pickled_host_only, dtype=torch.uint8
                        )
                        host_only_size = torch.tensor(
                            [host_only_tensor.numel()], dtype=torch.long
                        )
                        for dst in self.dp_device_ranks[1:]:
                            dist.send(host_only_size, dst=dst, group=group)
                            dist.send(host_only_tensor, dst=dst, group=group)
                    elif local_rank == 0:
                        host_only_size = torch.zeros(1, dtype=torch.long)
                        dist.recv(host_only_size, src=0, group=group)
                        host_only_tensor = torch.empty(
                            host_only_size.item(), dtype=torch.uint8
                        )
                        dist.recv(host_only_tensor, src=0, group=group)
                        gathered_host_only_sample_params = pickle.loads(
                            host_only_tensor.numpy().tobytes()
                        )

            if local_rank == 0:
                gathered_inputs = {
                    "int_inputs": stacked_int,
                    "float_inputs": stacked_float,
                    "sampling_tokens_inputs": gathered_tokens_inputs,
                    "host_only_sample_params": gathered_host_only_sample_params,
                    "reset_batch": any_reset_batch,
                    "all_sample_device": all_sample_device,
                }

        else:
            gathered_inputs = None
            if rank == 0:
                gathered_inputs = [None for _ in range(world)]

            logprobs_flag_t = torch.tensor([local_needs_logprobs], dtype=torch.int32)
            dist.all_reduce(logprobs_flag_t, op=dist.ReduceOp.MAX, group=group)
            any_needs_logprobs = logprobs_flag_t[0].item() > 0

            dist.gather_object(local_input, gathered_inputs, dst=0, group=group)
            if len(self.dp_device_ranks) > 1:
                if rank == 0:
                    pickled_data = pickle.dumps(gathered_inputs)
                    object_tensor = torch.frombuffer(pickled_data, dtype=torch.uint8)
                    size_tensor = torch.tensor(
                        [object_tensor.numel()], dtype=torch.long
                    )
                    for dst in self.dp_device_ranks[1:]:
                        dist.send(size_tensor, dst=dst, group=group)
                        dist.send(object_tensor, dst=dst, group=group)
                elif local_rank == 0:
                    size_tensor = torch.zeros(1, dtype=torch.long)
                    dist.recv(size_tensor, src=0, group=group)
                    object_tensor = torch.empty(size_tensor.item(), dtype=torch.uint8)
                    dist.recv(object_tensor, src=0, group=group)
                    gathered_inputs = pickle.loads(object_tensor.numpy().tobytes())
        self.dlog("after_inputs_gather")

        should_submit = is_decode or (
            isinstance(gathered_inputs, list)
            and any(x is not None for x in gathered_inputs)
        )
        if should_submit:
            collective_future = cast(
                Future[list[tuple[torch.Tensor, list]]],
                self.model_executor.collective_rpc(
                    "concat_and_execute_dp",
                    args=(
                        gathered_inputs,
                        is_decode,
                        max_blocks_decode,
                        any_structured_inputs,
                    ),
                    kwargs={"non_block": True},
                    non_block=True,
                ),
            )
            future = _unwrap_single_worker_future(collective_future)
        else:
            future = self._completed_dp_gather_future()

        return DPGatherHandle(
            future=future,
            scheduler_output=scheduler_output,
            local_has_requests=local_has_requests,
            is_decode=is_decode,
            overlap_ok=overlap_ok,
            any_needs_logprobs=any_needs_logprobs,
            intermediate_prefill_mask=intermediate_prefill_mask,
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
        )

    def dp_gather_finalize(self, handle: DPGatherHandle) -> ModelRunnerOutput:
        parallel_config = self.vllm_config.parallel_config
        group = self.dp_group
        rank = self.dp_rank
        world = parallel_config.data_parallel_size
        logprobs_per_dp: list = [None] * world

        result = handle.future.result()
        assert isinstance(result, tuple) and len(result) == 2
        send_tensor, logprobs_per_dp = result
        assert isinstance(send_tensor, torch.Tensor)

        my_ids = torch.empty_like(send_tensor[0])
        scatter_list = None
        if rank == 0:
            scatter_list = [send_tensor[i] for i in range(world)]
        dist.scatter(my_ids, scatter_list, src=0, group=group)
        self.dlog("after_results_gather my_ids_shape=%s", tuple(my_ids.shape))

        my_logprobs_val = None
        if handle.any_needs_logprobs:
            my_logprobs: list = [None]
            logprobs_scatter_list = logprobs_per_dp if rank == 0 else None
            dist.scatter_object_list(
                my_logprobs, logprobs_scatter_list, src=0, group=group
            )
            my_logprobs_val = my_logprobs[0]

        if handle.local_has_requests:
            output: ModelRunnerOutput = self.model_executor.collective_rpc(
                "apply_dp_execution_result",
                args=(
                    my_ids,
                    my_logprobs_val,
                    handle.req_ids,
                    handle.req_id_to_index,
                    handle.intermediate_prefill_mask,
                ),
            )[0]
            return output
        return EMPTY_MODEL_RUNNER_OUTPUT

    def _execute_model_dp_gather(
        self,
        scheduler_output: SchedulerOutput | None,
        grammar_output: GrammarOutput | None,
    ) -> ModelRunnerOutput:
        handle = self.dp_gather_submit(
            scheduler_output, grammar_output, overlap_ok=False
        )
        return self.dp_gather_finalize(handle)

    def _completed_dp_gather_future(self) -> Future[tuple[torch.Tensor, list]]:
        parallel_config = self.vllm_config.parallel_config
        world = parallel_config.data_parallel_size
        batch_size = get_tt_per_lane_max_num_seqs(self.vllm_config)
        return _as_future(
            (torch.zeros((world, batch_size, 1), dtype=torch.int32), [None] * world)
        )


def _as_future(value: _T) -> Future[_T]:
    future: Future[_T] = Future()
    future.set_result(value)
    return future


def _unwrap_single_worker_future(future: Future[list[_T]]) -> Future[_T]:
    single_future: Future[_T] = Future()

    def _set_single_result(done_future: Future[list[_T]]) -> None:
        try:
            results = done_future.result()
            assert len(results) == 1
            single_future.set_result(results[0])
        except Exception as exc:
            single_future.set_exception(exc)

    future.add_done_callback(_set_single_result)
    return single_future
