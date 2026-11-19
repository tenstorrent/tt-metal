# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real layer-0 continuation/trace diagnostics on one loaded TTNN instance.

Two independent request states are exercised serially. This does not establish
public batched API support. Optional signposts identify warmed measured windows;
host wall measurements are not represented as device kernel performance.
"""

import argparse
import contextlib
import json
import time
from pathlib import Path
from unittest import mock

import torch
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_components import load_weights
from models.autoports.qwen_qwen3_8_flash_next.tt.functional_decoder import FunctionalDecoder


class NoTorchCalls(torch.overrides.TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        raise RuntimeError(f"Runtime torch operation forbidden: {func}")


@contextlib.contextmanager
def runtime_audit():
    def forbidden(*args, **kwargs):
        raise RuntimeError("Runtime TTNN/torch conversion forbidden")

    with (
        mock.patch.object(ttnn, "from_torch", forbidden),
        mock.patch.object(ttnn, "to_torch", forbidden),
        ttnn.manage_config("throw_exception_on_fallback", True),
        NoTorchCalls(),
    ):
        yield


def row_pcc(left, right):
    left, right = left.float().reshape(-1, left.shape[-1]), right.float().reshape(-1, right.shape[-1])
    left = left - left.mean(-1, keepdim=True)
    right = right - right.mean(-1, keepdim=True)
    return (left * right).sum(-1) / (left.square().sum(-1) * right.square().sum(-1)).sqrt()


def measure(expected, actual, residual):
    actual = actual.reshape(expected.shape)
    contributions = row_pcc(expected.float() - residual.float(), actual.float() - residual.float())
    whole = row_pcc(expected, actual)
    return {
        "whole_output_min_row_pcc": whole.min().item(),
        "contribution_min_row_pcc": contributions.min().item(),
        "contribution_per_row_pcc": contributions.tolist(),
        "finite": bool(torch.isfinite(actual).all()),
        "passed": bool(
            torch.isfinite(actual).all()
            and torch.isfinite(contributions).all()
            and (whole >= 0.995).all()
            and (contributions >= 0.995).all()
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", default="1,31,32,33,65")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    lengths = [int(value) for value in args.lengths.split(",")]
    if not lengths or min(lengths) < 1:
        parser.error("--lengths requires positive lengths")
    torch.set_num_threads(16)
    torch.manual_seed(932)
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
    config._attn_implementation = "eager"
    weights = load_weights(args.snapshot, 0, "")
    with torch.device("meta"):
        reference = Qwen4ExpTextDecoderLayer(config, 0)
    reference.load_state_dict(weights, assign=True)
    reference.eval()
    full_length = max(65, *lengths)
    x = (torch.randn(1, full_length, 10240) * 0.02).bfloat16()
    x_other = (torch.randn(1, 7, 10240) * 0.02).bfloat16()
    decode_x = [(torch.randn(1, 1, 10240) * 0.02).bfloat16() for _ in range(3)]

    def hf(value, cache=None):
        return reference(
            value,
            position_embeddings=(None, None),
            past_key_values=cache if cache is not None else DynamicCache(config=config),
        )

    with torch.inference_mode():
        expected = {length: hf(x[:, :length]) for length in set([full_length, *lengths])}
        other_expected = hf(x_other)
        cache = DynamicCache(config=config)
        hf(x, cache)
        expected_decode = [hf(value, cache) for value in decode_x]
        hf_continuation = {}
        for start in (1, 31, 32, 33):
            cache = DynamicCache(config=config)
            hf(x[:, :start], cache)
            resumed = hf(x[:, start : start + 8], cache)
            hf_continuation[str(start)] = measure(
                expected[full_length][:, start : start + 8], resumed, x[:, start : start + 8]
            )
    print("LINEAR_CONTRACT_HF_READY", flush=True)
    report = {
        "layer_idx": 0,
        "weights": "real",
        "snapshot": str(args.snapshot),
        "threshold": 0.995,
        "sweep": [],
        "continuation": [],
        "decode": [],
        "hf_continuation_controls": hf_continuation,
        "batch_scope": "two independent request states executed serially; public batched API is untested",
        "runtime_audit": {
            "torch_function_mode": True,
            "forbid_from_torch_to_torch": True,
            "throw_exception_on_fallback": True,
            "completed_calls": 0,
        },
    }
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace_id = None

    def signpost(name):
        if args.profile:
            from tracy import signpost as mark

            mark(name)

    try:
        model = FunctionalDecoder.from_state_dict(weights, hf_config=config, layer_idx=0, mesh_device=mesh)
        state, other_state = model.create_state(), model.create_state()

        def host(value, dtype=ttnn.bfloat16):
            return ttnn.from_torch(value.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT)

        def device(value):
            return ttnn.to_device(host(value.unsqueeze(0)), mesh)

        def snapshot(target):
            return {
                name: host(ttnn.to_torch(getattr(target, name)), getattr(target, name).dtype)
                for name in ("conv", "recurrent")
            }

        def restore(target, saved):
            for name, tensor in saved.items():
                ttnn.copy_host_to_device_tensor(tensor, getattr(target, name))

        initial = snapshot(state)
        prepared = {length: device(x[:, :length]) for length in set([full_length, *lengths])}
        continuation_inputs = {
            start: (device(x[:, :start]), device(x[:, start : start + 8])) for start in (1, 31, 32, 33)
        }
        other_input = device(x_other)
        decode_input = device(decode_x[0])
        decode_refresh = [host(value.unsqueeze(0)) for value in decode_x]

        def prefill(value, target=state):
            with runtime_audit():
                result = model.prefill_forward(value, state=target)
            report["runtime_audit"]["completed_calls"] += 1
            return result

        def decode(value):
            with runtime_audit():
                result = model.decode_forward(value, state=state)
            report["runtime_audit"]["completed_calls"] += 1
            return result

        first_results = {}
        for length in [*lengths, lengths[0]]:
            restore(state, initial)
            result = prefill(prepared[length])
            actual = ttnn.to_torch(result)
            ttnn.deallocate(result)
            entry = dict(length=length, **measure(expected[length], actual, x[:, :length]))
            if length in first_results:
                entry["reset_equal"] = torch.equal(first_results[length], actual)
            else:
                first_results[length] = actual.clone()
            report["sweep"].append(entry)
        # Prefix continuation is judged against a full logical run, not against
        # another resumed run. The dropped-history control uses the SAME gate.
        for start, (prefix, resume) in continuation_inputs.items():
            restore(state, initial)
            intermediate = prefill(prefix)
            ttnn.deallocate(intermediate)
            correct = prefill(resume)
            correct_host = ttnn.to_torch(correct)
            ttnn.deallocate(correct)
            restore(state, initial)
            dropped = prefill(resume)
            dropped_host = ttnn.to_torch(dropped)
            ttnn.deallocate(dropped)
            expected_window = expected[full_length][:, start : start + 8]
            residual = x[:, start : start + 8]
            correct_metrics = measure(expected_window, correct_host, residual)
            dropped_metrics = measure(expected_window, dropped_host, residual)
            report["continuation"].append(
                {
                    "start": start,
                    "rows": 8,
                    "correct": correct_metrics,
                    "dropped_history_negative_control": dropped_metrics,
                    "negative_control_detected": dropped_metrics["finite"]
                    and dropped_metrics["contribution_min_row_pcc"] < 0.995,
                }
            )
        # Distinct row ownership and content: keep another request's state alive
        # while running the primary request. This is serial request isolation.
        other_result = prefill(other_input, other_state)
        other_actual = ttnn.to_torch(other_result)
        ttnn.deallocate(other_result)
        other_before = snapshot(other_state)
        restore(state, initial)
        primary_result = prefill(prepared[full_length])
        ttnn.deallocate(primary_result)
        report["independent_requests"] = {
            "other_request": measure(other_expected, other_actual, x_other),
            "other_state_unchanged": all(
                torch.equal(ttnn.to_torch(tensor), ttnn.to_torch(getattr(other_state, name)))
                for name, tensor in other_before.items()
            ),
        }
        if args.profile:
            restore(state, initial)
            ttnn.synchronize_device(mesh)
            signpost("PERF_PREFILL")
            started = time.perf_counter()
            measured = prefill(prepared[full_length])
            ttnn.synchronize_device(mesh)
            report["warmed_prefill_host_ms"] = (time.perf_counter() - started) * 1000
            signpost("PERF_PREFILL_END")
            ttnn.deallocate(measured)
        # State now holds exactly this request's full prefix. Prepare decode and
        # restore before capture AND after capture, which also mutates state.
        prefix_snapshot = snapshot(state)
        warmed = decode(decode_input)
        ttnn.synchronize_device(mesh)
        ttnn.deallocate(warmed)
        restore(state, prefix_snapshot)
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        output = decode(decode_input)
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        trace_allocation_tracker.acknowledge_corruptible(output)
        # Warm the actual trace before its signposted measured replay. Restore
        # the prefix afterwards, so warmup never advances the tested request.
        restore(state, prefix_snapshot)
        with runtime_audit():
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
        report["runtime_audit"]["completed_calls"] += 1
        warm_trace_output = ttnn.to_torch(output)
        del warm_trace_output
        saved_outputs = []
        for repetition in range(2):
            restore(state, prefix_snapshot)
            outputs = []
            for step in range(3):
                ttnn.copy_host_to_device_tensor(decode_refresh[step], decode_input)
                ttnn.synchronize_device(mesh)
                if repetition == 0:
                    signpost("PERF_DECODE" if step == 0 else f"PERF_DECODE_STEP_{step}")
                started = time.perf_counter()
                with runtime_audit():
                    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                report["runtime_audit"]["completed_calls"] += 1
                elapsed = (time.perf_counter() - started) * 1000
                if repetition == 0:
                    signpost("PERF_DECODE_END" if step == 0 else f"PERF_DECODE_STEP_{step}_END")
                actual = ttnn.to_torch(output)
                outputs.append(actual.clone())
                if repetition == 0:
                    report["decode"].append(
                        dict(step=step, host_ms=elapsed, **measure(expected_decode[step], actual, decode_x[step]))
                    )
            saved_outputs.append(outputs)
        report["decode_reset_equal"] = all(torch.equal(a, b) for a, b in zip(*saved_outputs))
        report["interleaved_prefill"] = []
        for length in lengths:
            restore(state, initial)
            result = prefill(prepared[length])
            actual_prefill = ttnn.to_torch(result)
            ttnn.deallocate(result)
            restore(state, prefix_snapshot)
            ttnn.copy_host_to_device_tensor(decode_refresh[0], decode_input)
            with runtime_audit():
                ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
            report["runtime_audit"]["completed_calls"] += 1
            actual_decode = ttnn.to_torch(output)
            report["interleaved_prefill"].append(
                {
                    "length": length,
                    "prefill": measure(expected[length], actual_prefill, x[:, :length]),
                    "same_decode_trace_equal": torch.equal(actual_decode, saved_outputs[0][0]),
                }
            )
        report["trace"] = {
            "captures": 1,
            "replays": 7 + len(lengths),
            "unmeasured_warmup_replays": 1,
            "trace_id": str(trace_id),
            "acknowledgment": "exact output only; read back before subsequent replay",
        }
        report["passed"] = (
            all(row["passed"] and row.get("reset_equal", True) for row in report["sweep"])
            and all(row["correct"]["passed"] and row["negative_control_detected"] for row in report["continuation"])
            and all(row["passed"] for row in report["decode"])
            and all(row["passed"] for row in hf_continuation.values())
            and report["decode_reset_equal"]
            and all(
                row["prefill"]["passed"] and row["same_decode_trace_equal"] for row in report["interleaved_prefill"]
            )
            and report["independent_requests"]["other_request"]["passed"]
            and report["independent_requests"]["other_state_unchanged"]
        )
    except Exception as exc:
        report["error"] = str(exc).split("backtrace:")[0]
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if trace_id is not None:
            ttnn.release_trace(mesh, trace_id)
        ttnn.close_mesh_device(mesh)
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
