# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Replay this test's trusted local recording through the real TT worker/runner.

Run each read mode in a fresh process under a bounded timeout, serialized with
all device jobs. The scheduler inputs are fixed; model outputs are never fed
back from the recording. Both modes retain async configuration and the real
runner's lifecycle/reload logic. This is a correctness diagnostic, not a server.
"""

import argparse
import copy
import json
import os
import pickle
import time
import traceback
from pathlib import Path


def load_events(path):
    result = []
    with path.open("rb") as handle:
        while handle.peek(1):
            result.append(pickle.load(handle))
    if [event["event"] for event in result] != list(range(len(result))):
        raise ValueError("Incomplete or noncontiguous recording")
    return result


def history_projection(contract):
    if contract is None:
        return None
    result = dict(contract)
    if result["prompt_lens"] is None:
        # Host tokens/positions may be stale in the queued column. The separate
        # decode contract compares authoritative values whenever reload=True,
        # active masks and effective resident positions on every submission.
        result.pop("input_tokens")
        result.pop("input_positions")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--read-mode", choices=("immediate", "recorded_queue"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output artifact")
    client = json.loads((args.recording / "client.json").read_text())
    if not client.get("passed"):
        parser.error("Recording client did not complete successfully")
    os.environ.pop("QWEN_VLLM_HISTORY_DIR", None)
    os.environ["TT_MODEL_CLASS_OVERRIDES"] = (
        "TTQwen4ExpForConditionalGeneration="
        "models.autoports.qwen_qwen3_8_flash_next.tt.generator_vllm:QwenFlashNextForCausalLM"
    )
    import torch
    import vllm_tt_plugin  # noqa: F401
    from vllm.config import set_current_vllm_config
    from vllm.v1.executor.uniproc_executor import UniProcExecutor
    from vllm.v1.outputs import AsyncModelRunnerOutput
    from vllm_tt_plugin.async_decode import TTAsyncDecodeController
    from vllm_tt_plugin.model_runner import TTModelRunner

    from models.autoports.qwen_qwen3_8_flash_next.tests.vllm_history_record import (
        decode_contract,
        input_contract,
        output_ids,
    )

    with (args.recording / "setup.pkl").open("rb") as handle:
        setup = pickle.load(handle)
    events = load_events(args.recording / "events.pkl")
    config = setup["vllm_config"]
    assert config.parallel_config.pipeline_parallel_size == 1
    assert config.scheduler_config.async_scheduling, "Both columns must preserve the recorded async configuration"
    assert setup["sample_on_device_mode"] == "all" and setup["trace_mode"] == "all"
    for name, expected in setup["environment"].items():
        if expected is not None and os.environ.get(name) != expected:
            raise RuntimeError(f"Restore recorded environment {name}={expected!r} before replay")
    commits = [event for event in events if event["kind"] == "commit"]
    if not commits:
        raise RuntimeError("No scheduler commits recorded; recording must use UniProc in the scheduler process")
    emitted = {}
    for event in commits:
        for row in event["emitted"]:
            emitted.setdefault(row["req_id"], []).extend(row["new_token_ids"])
    for row in client["requests"]:
        # Upstream appends a random suffix to each internal engine request ID.
        # Require one recorded identity for this API choice, then compare all
        # of its emitted tokens exactly; never match requests by token content.
        prefix = row["response_id"] + "-0-"
        matches = [request_id for request_id in emitted if request_id.startswith(prefix)]
        assert len(matches) == 1, f"Ambiguous or missing scheduler identity for {prefix}"
        expected_id = matches[0]
        assert emitted.get(expected_id) == row["token_ids"], f"Scheduler/API token IDs differ for {expected_id}"

    report = dict(
        passed=False,
        read_mode=args.read_mode,
        recording=str(args.recording),
        scope="fixed_recorded_scheduler_history_real_plugin_runner",
        history_checks=0,
        output_checks=0,
        committed_steps=0,
        emitted_matches_http=True,
        comparisons=[],
        execution_signatures=[
            dict(
                step=event["step"],
                prefill=event["contract"]["prompt_lens"] is not None,
                positions=event["contract"]["input_positions"],
                prompt_lens=event["contract"]["prompt_lens"],
                rows=event["contract"]["row_req_ids"],
                slot_remap=event["contract"]["slot_remap"],
                intermediate_mask=event["contract"]["intermediate_prefill_mask"],
            )
            for event in events
            if event["kind"] == "model_input" and event["contract"] is not None
        ],
    )
    executor = None
    current = dict(step=None)
    observed_inputs, observed_decodes = {}, {}
    original_build = TTModelRunner.build_model_input
    original_submit = TTAsyncDecodeController.submit_decode

    def build(runner, scheduler_output, grammar_output):
        result = original_build(runner, scheduler_output, grammar_output)
        observed_inputs[current["step"]] = input_contract(result)
        return result

    def submit(controller, model_input, **kwargs):
        positions = [cache.position for cache in controller.runner.model.generator.caches]
        result = original_submit(controller, model_input, **kwargs)
        observed_decodes[current["step"]] = decode_contract(model_input, result.reload_plan, positions)
        return result

    TTModelRunner.build_model_input = build
    TTAsyncDecodeController.submit_decode = submit
    outputs = {}
    started = time.monotonic()
    try:
        torch.set_num_threads(8)
        with set_current_vllm_config(config):
            # Use the real worker's normal mesh/distributed/model setup, not a
            # hand-built runner fixture or a substituted state/input builder.
            executor = UniProcExecutor(config)
            executor.initialize_from_config([setup["kv_cache_config"]])
            executor.compile_or_warm_up_model()
            runner = executor.driver_worker.worker.model_runner
            assert type(runner.model).__name__ == "QwenFlashNextForCausalLM"
            for event in events:
                kind, step = event["kind"], event.get("step")
                current["step"] = step
                if kind == "execute":
                    outputs[step] = runner.execute_model(copy.deepcopy(event["scheduler_output"]))
                elif kind == "sample":
                    outputs[step] = runner.sample_tokens(copy.deepcopy(event["grammar_output"]))
                    if args.read_mode == "immediate" and isinstance(outputs[step], AsyncModelRunnerOutput):
                        outputs[step] = outputs[step].get_output()
                elif kind == "resolve":
                    if isinstance(outputs[step], AsyncModelRunnerOutput):
                        outputs[step] = outputs[step].get_output()
                elif kind in ("execute_return", "sample_return", "resolved", "commit"):
                    if kind == "sample_return" and event["deferred"]:
                        continue
                    actual = output_ids(outputs[step])
                    expected = event["output"]
                    equal = actual == expected
                    report["comparisons"].append(
                        dict(step=step, boundary=kind, equal=equal, expected=expected, actual=actual)
                    )
                    assert equal, f"Step{step} {kind}: exact request/token outputs differ"
                    report["output_checks"] += 1
                    report["committed_steps"] += int(kind == "commit")
                elif kind == "model_input":
                    actual = history_projection(observed_inputs[step])
                    expected = history_projection(event["contract"])
                    if actual != expected:
                        report["input_contract_mismatch"] = dict(step=step, expected=expected, actual=actual)
                    assert actual == expected, f"Step{step}: execution history differs"
                    report["history_checks"] += 1
                elif kind == "decode_contract":
                    actual = observed_decodes[step]
                    if actual != event["contract"]:
                        report["decode_contract_mismatch"] = dict(step=step, expected=event["contract"], actual=actual)
                    assert actual == event["contract"], f"Step{step}: effective decode inputs/reload plan differ"
                    report["history_checks"] += 1
            runner.async_decode.wait_for_all_pending_async_steps()
            report["overlapped_submissions"] = runner.async_decode._overlapped_submissions
            report["unsafe_overlaps"] = runner.async_decode._overlapped_unsafe_submissions
            assert report["unsafe_overlaps"] == 0
            if args.read_mode == "recorded_queue":
                assert report["overlapped_submissions"] > 0, "Recorded replay did not exercise pending overlap"
            report["final_request_outputs"] = {
                request_id: list(state.output_token_ids) for request_id, state in runner.requests.items()
            }
            report["passed"] = True
    except Exception:
        report["failed_event"] = current["step"]
        report["error"] = traceback.format_exc()
        raise
    finally:
        try:
            if executor is not None:
                executor.shutdown()
        except Exception:
            report["passed"] = False
            report["shutdown_error"] = traceback.format_exc()
            raise
        finally:
            TTModelRunner.build_model_input = original_build
            TTAsyncDecodeController.submit_decode = original_submit
            report["seconds"] = time.monotonic() - started
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("passed", "read_mode", "history_checks", "output_checks")}))


if __name__ == "__main__":
    main()
