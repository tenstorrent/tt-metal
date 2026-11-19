# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Explicit diagnostic factory: record real runner inputs without extra waits.

Set QWEN_VLLM_HISTORY_DIR to a fresh directory and register this factory instead
of the production class for a diagnostic launch only. The returned model object
is exactly QwenFlashNextForCausalLM. Pickles are local replay inputs produced by
this test; never load recordings from an untrusted source.
"""

import copy
import functools
import os
import pickle
import threading
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import MethodType

from models.autoports.qwen_qwen3_8_flash_next.tt.generator_vllm import QwenFlashNextForCausalLM


def plain(value):
    """Serialize CPU metadata only; never pull a device tensor to the host."""
    import torch

    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise TypeError("Recorder refuses device tensor readback")
        return value.tolist()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(plain(item) for item in value)
    if is_dataclass(value):
        return {field.name: plain(getattr(value, field.name)) for field in fields(value)}
    if type(value).__module__.startswith("numpy"):
        return value.tolist()
    raise TypeError(f"Unsupported metadata type {type(value)}")


def output_ids(output):
    if output is None:
        return None
    return dict(req_ids=list(output.req_ids), sampled_token_ids=plain(output.sampled_token_ids))


def input_contract(model_input):
    if model_input is None:
        return None
    names = (
        "input_tokens",
        "input_positions",
        "prompt_lens",
        "block_tables",
        "block_tables_per_group",
        "block_tables_per_layer",
        "unpadded_batch_size",
        "perform_device_sampling",
        "decode_layout_changed",
        "slot_remap",
        "prefill_empty_slots",
        "intermediate_prefill_mask",
        "row_req_ids",
        "live_rows",
        "grammar_row_req_ids",
        "tt_sampling_params",
    )
    return {name: plain(getattr(model_input, name)) for name in names}


def decode_contract(model_input, plan, resident_positions):
    positions = plain(model_input.input_positions)
    return dict(
        reload_plan=plain(plan),
        resident_positions=resident_positions,
        authoritative_tokens=plain(model_input.input_tokens) if plan.reload_inputs else None,
        authoritative_positions=positions if plan.reload_inputs else None,
        active_mask=[position >= 0 for position in positions],
        slot_remap=plain(model_input.slot_remap),
        page_tables=plain(model_input.block_tables),
    )


class Recorder:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.stream = (self.directory / "events.pkl").open("xb")
        self.lock = threading.Lock()
        self.event_count = 0
        self.step_count = 0
        self.step_by_scheduler = {}
        self.scheduler_refs = []  # Prevent id reuse until recording ends.
        self.pending_samples = []
        self.current_step = None
        self.setup_written = False
        self.write("start", pid=os.getpid(), scope="diagnostic_no_performance_claim")

    def write(self, kind, **payload):
        with self.lock:
            item = dict(event=self.event_count, kind=kind, time_ns=time.monotonic_ns(), **payload)
            pickle.dump(item, self.stream, protocol=pickle.HIGHEST_PROTOCOL)
            self.stream.flush()
            self.event_count += 1

    def save_setup(self, runner, cache_config):
        if self.setup_written:
            raise RuntimeError("History recorder supports exactly one runner/cache initialization")
        config = copy.copy(runner.vllm_config)
        config.device_config = copy.copy(config.device_config)
        config.device_config.device = None  # Live mesh objects are never serialized.
        metadata = dict(
            vllm_config=config,
            kv_cache_config=copy.deepcopy(cache_config),
            trace_mode=runner.trace_mode,
            enable_model_warmup=runner.enable_model_warmup,
            num_devices=runner.num_devices,
            sample_on_device_mode=runner.sample_on_device_mode,
            environment={
                name: os.environ.get(name)
                for name in (
                    "TT_VISIBLE_DEVICES",
                    "TT_MESH_GRAPH_DESC_PATH",
                    "MESH_DEVICE",
                    "QWEN_VLLM_LAYER_INDICES",
                    "TT_METAL_TRACE_ALLOC_TRACKING",
                    "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
                )
            },
        )
        with (self.directory / "setup.pkl").open("xb") as handle:
            pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.setup_written = True
        self.write(
            "setup",
            summary=dict(
                max_model_len=config.model_config.max_model_len,
                max_num_seqs=config.scheduler_config.max_num_seqs,
                async_scheduling=config.scheduler_config.async_scheduling,
                trace_mode=runner.trace_mode,
                adapter=type(runner.model).__module__ + ":" + type(runner.model).__name__,
                num_devices=runner.num_devices,
                environment=metadata["environment"],
            ),
        )

    def install(self):
        from vllm.v1.outputs import AsyncModelRunnerOutput
        from vllm_tt_plugin.async_decode import TTAsyncDecodeController
        from vllm_tt_plugin.model_runner import TTModelRunner
        from vllm_tt_plugin.scheduler import TTScheduler

        recorder = self
        original_initialize = TTModelRunner.initialize_kv_cache
        original_execute = TTModelRunner.execute_model
        original_sample = TTModelRunner.sample_tokens
        original_build = TTModelRunner.build_model_input
        original_commit = TTScheduler.update_from_output
        original_submit = TTAsyncDecodeController.submit_decode

        @functools.wraps(original_submit)
        def submit(controller, model_input, **kwargs):
            positions = [cache.position for cache in controller.runner.model.generator.caches]
            result = original_submit(controller, model_input, **kwargs)
            recorder.write(
                "decode_contract",
                step=recorder.current_step,
                contract=decode_contract(model_input, result.reload_plan, positions),
            )
            return result

        @functools.wraps(original_initialize)
        def initialize(runner, cache_config):
            recorder.save_setup(runner, cache_config)
            return original_initialize(runner, cache_config)

        @functools.wraps(original_execute)
        def execute(runner, scheduler_output):
            step = recorder.step_count
            recorder.step_count += 1
            recorder.step_by_scheduler[id(scheduler_output)] = step
            recorder.scheduler_refs.append(scheduler_output)
            recorder.current_step = step
            recorder.write("execute", step=step, scheduler_output=copy.deepcopy(scheduler_output))
            result = original_execute(runner, scheduler_output)
            if result is None:
                recorder.pending_samples.append(step)
            recorder.write("execute_return", step=step, output=output_ids(result))
            return result

        @functools.wraps(original_build)
        def build(runner, scheduler_output, grammar_output):
            result = original_build(runner, scheduler_output, grammar_output)
            recorder.write("model_input", step=recorder.current_step, contract=input_contract(result))
            return result

        @functools.wraps(original_sample)
        def sample(runner, grammar_output):
            if not recorder.pending_samples:
                raise RuntimeError("Unpaired sample_tokens call in recorder")
            step = recorder.pending_samples.pop(0)
            recorder.current_step = step
            recorder.write("sample", step=step, grammar_output=copy.deepcopy(grammar_output))
            result = original_sample(runner, grammar_output)
            deferred = isinstance(result, AsyncModelRunnerOutput)
            recorder.write(
                "sample_return", step=step, deferred=deferred, output=None if deferred else output_ids(result)
            )
            if deferred:
                original_get = result.get_output

                def get_output(_self):
                    recorder.write("resolve", step=step)
                    output = original_get()  # Exactly the original caller's wait.
                    recorder.write("resolved", step=step, output=output_ids(output))
                    return output

                result.get_output = MethodType(get_output, result)
            return result

        @functools.wraps(original_commit)
        def commit(scheduler, scheduler_output, model_runner_output):
            key = id(scheduler_output)
            if key not in recorder.step_by_scheduler:
                raise RuntimeError("Recorder requires UniProc scheduler/runner identity in the same process")
            step = recorder.step_by_scheduler[key]
            before = output_ids(model_runner_output)
            result = original_commit(scheduler, scheduler_output, model_runner_output)
            emitted = []
            for engine_output in result.values():
                for request_output in engine_output.outputs:
                    emitted.append(
                        dict(req_id=request_output.request_id, new_token_ids=list(request_output.new_token_ids))
                    )
            recorder.write("commit", step=step, output=before, emitted=emitted)
            return result

        TTModelRunner.initialize_kv_cache = initialize
        TTModelRunner.execute_model = execute
        TTModelRunner.sample_tokens = sample
        TTModelRunner.build_model_input = build
        TTScheduler.update_from_output = commit
        TTAsyncDecodeController.submit_decode = submit


_RECORDER = None


class RecordedQwenFlashNextForCausalLM(QwenFlashNextForCausalLM):
    @classmethod
    def initialize_vllm_model(cls, *args, **kwargs):
        global _RECORDER
        directory = os.environ.get("QWEN_VLLM_HISTORY_DIR")
        if not directory or _RECORDER is not None:
            raise RuntimeError("Diagnostic factory requires a fresh QWEN_VLLM_HISTORY_DIR and one model")
        _RECORDER = Recorder(directory)
        _RECORDER.install()
        return QwenFlashNextForCausalLM.initialize_vllm_model(*args, **kwargs)
