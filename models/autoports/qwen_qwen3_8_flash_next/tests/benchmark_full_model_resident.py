# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""All-layer stage-7 benchmark with caller-read and resident token-out windows.

Run serially in the configured lane environment, under a bounded timeout::

    timeout 5400 python -m models.autoports.qwen_qwen3_8_flash_next.tests.benchmark_full_model_resident \
      --output models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/baseline.json

Optional ``--ttft-lengths 31,32,33,128,1024`` adds individually warmed public
generate TTFT rows. The primary comparison always remains B1 128 in / 128 out.
Optional ``--public-resident`` adds a separate public decode_forward resident
window with explicit cache/slot/reload state; the original resident metric is unchanged.
The acceptance driver owns mesh setup, provenance, diagnostics exclusions and
failure recording. This wrapper never launches a profiler or changes policy.
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import run_full_model_acceptance as acceptance


def source_snapshot():
    """Capture executable inputs outside timing, including native source and shared sampling."""
    model_dir = Path(__file__).resolve().parents[1]
    repo = model_dir.parents[2]
    paths = [
        *sorted(
            path
            for path in (model_dir / "tt").rglob("*")
            if path.suffix in (".py", ".cpp", ".h", ".hpp", ".cc", ".cxx")
        ),
        Path(__file__).resolve(),
        Path(acceptance.__file__).resolve(),
        repo / "models/common/sampling/generator.py",
        repo / "models/common/sampling/tt_sampling.py",
        repo / "models/common/tt_ccl.py",
    ]
    return {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def run_resident(generator, model_dir, reference_tokens):
    """Replay the public generator's exact decode/sample sequence without reads."""
    assert generator.max_batch_size == 1 and len(generator.model.layers) == 48
    prompt = acceptance.natural_prompt(generator, model_dir, 128)
    generator.reset()
    generator.prefill_forward(
        torch.tensor([prompt]),
        page_table=None,
        kv_cache=[generator.caches[0]],
        prompt_lens=[128],
        slots=(0,),
        return_device=True,
    )
    generator._sample((0,))
    initial_tokens = generator._read_tokens()
    assert initial_tokens[0] == reference_tokens[0], "Prefill sample differs from caller-read control"
    ids_before = acceptance.trace_ids(generator)
    token_input = generator.decode_tokens
    sampler_input = generator.sample_logits
    before = dict(generator.counters)
    acceptance.marker("PERF_RESIDENT_TOKEN_OUT_START", prompt=128, generated=128, decode_steps=127)
    ttnn.synchronize_device(generator.mesh)
    started = time.perf_counter()
    for _ in range(127):
        # These helpers enqueue nonblocking model/sampling traces on the same CQ.
        # Sampling writes tt_out_tok directly into the next decode input. Device
        # position/RoPE updates belong to the model trace; only scheduler metadata
        # is mirrored here, exactly as in public generate().
        generator._replay_decode((0,))
        generator.caches[0].position += 1
        generator._sample((0,))
    ttnn.synchronize_device(generator.mesh)
    elapsed = time.perf_counter() - started
    steady = acceptance.counter_delta(generator.counters, before)
    assert steady["model_replays"] == 127 and steady["sampling_replays"] == 127
    for name in (
        "token_refreshes",
        "position_refreshes",
        "page_table_refreshes",
        "cache_resets",
        "captures",
        "releases",
        "readbacks",
        "synchronizations",
    ):
        assert steady.get(name, 0) == 0, f"Resident window performed {name}: {steady}"
    assert acceptance.trace_ids(generator) == ids_before, "Resident window changed trace identities"
    assert generator.decode_tokens is token_input and generator.sample_logits is sampler_input
    final_tokens = generator._read_tokens()
    assert final_tokens[0] == reference_tokens[-1], "Resident final token differs from caller-read control"
    assert final_tokens[1:] == initial_tokens[1:], "Resident decode changed inactive token slots"
    assert 0 <= final_tokens[0] < generator.model.config.vocab_size
    result = {
        "workload": {"batch": 1, "prompt_tokens": 128, "generated_tokens": 128, "decode_steps": 127},
        "host_seconds": elapsed,
        "ms_per_step": elapsed * 1000 / 127,
        "t_s_u": 127 / elapsed,
        "blocking_replay": False,
        "scope": "Model plus split sampling, device token feedback and position advance; no per-token caller read",
        "window_boundary_synchronizations": 2,
        "timed_boundary_synchronizations": 1,
        "counter_note": "Generator counters exclude the two explicit benchmark window boundary synchronizations",
        "steady_decode_counter_delta": steady,
        "final_readback_outside_timing": True,
        "first_token": int(initial_tokens[0]),
        "final_token": int(final_tokens[0]),
        "caller_read_control_first_and_final_match": True,
        "inactive_token_slots_unchanged": True,
        "persistent_token_and_sampler_input_identity_unchanged": True,
        "trace_ids": ids_before,
        "qsa_positions": acceptance.qsa_positions(generator, 255),
        "validation_scope": "First/final token and positions; per-step output correctness is a separate accuracy/feedback gate",
    }
    acceptance.marker("PERF_RESIDENT_TOKEN_OUT_DONE", t_s_u=result["t_s_u"], ms_per_step=result["ms_per_step"])
    return result


def run_public_resident(generator, model_dir, reference_tokens):
    """Measure the serving-ready public decode contract with persistent inputs."""
    assert generator.max_batch_size == 1 and len(generator.model.layers) == 48
    prompt = acceptance.natural_prompt(generator, model_dir, 128)
    generator.reset()
    generator.prefill_forward(
        torch.tensor([prompt]),
        page_table=None,
        kv_cache=[generator.caches[0]],
        prompt_lens=[128],
        slots=(0,),
        return_device=True,
    )
    generator._sample((0,))
    initial_tokens = generator._read_tokens()
    assert initial_tokens[0] == reference_tokens[0], "Prefill sample differs from caller-read control"
    ids_before = acceptance.trace_ids(generator)
    token_input = generator.decode_tokens
    sampler_input = generator.sample_logits
    # Shape-valid CPU tokens are ignored when reload_inputs=False. Allocate the
    # placeholder and cache container before timing; no token reconstruction or
    # upload occurs in the loop. Positions are scheduler metadata only.
    placeholder_tokens = torch.zeros(1, 1, dtype=torch.int64)
    caches = [generator.caches[0]]
    public_kwargs = dict(
        page_table=None,
        kv_cache=caches,
        slots=(0,),
        sample_on_device=True,
        reload_inputs=False,
        reload_page_table=False,
        read_from_device=False,
    )
    before = dict(generator.counters)
    acceptance.marker("PERF_PUBLIC_RESIDENT_TOKEN_OUT_START", prompt=128, generated=128, decode_steps=127)
    ttnn.synchronize_device(generator.mesh)
    started = time.perf_counter()
    for _ in range(127):
        submission = generator.decode_forward(
            placeholder_tokens,
            [caches[0].position],
            **public_kwargs,
        )
    ttnn.synchronize_device(generator.mesh)
    elapsed = time.perf_counter() - started
    assert submission.tensor is token_input and submission.is_tokens
    assert submission.slots == (0,) and submission.active_rows == (0,)
    steady = acceptance.counter_delta(generator.counters, before)
    assert steady["model_replays"] == 127 and steady["sampling_replays"] == 127
    for name in (
        "token_refreshes",
        "position_refreshes",
        "page_table_refreshes",
        "cache_resets",
        "captures",
        "releases",
        "readbacks",
        "synchronizations",
    ):
        assert steady.get(name, 0) == 0, f"Resident window performed {name}: {steady}"
    assert acceptance.trace_ids(generator) == ids_before, "Resident window changed trace identities"
    assert generator.decode_tokens is token_input and generator.sample_logits is sampler_input
    final_tokens = generator._read_tokens()
    assert final_tokens[0] == reference_tokens[-1], "Resident final token differs from caller-read control"
    assert final_tokens[1:] == initial_tokens[1:], "Resident decode changed inactive token slots"
    assert 0 <= final_tokens[0] < generator.model.config.vocab_size
    result = {
        "workload": {"batch": 1, "prompt_tokens": 128, "generated_tokens": 128, "decode_steps": 127},
        "host_seconds": elapsed,
        "ms_per_step": elapsed * 1000 / 127,
        "t_s_u": 127 / elapsed,
        "blocking_replay": False,
        "scope": "Public decode_forward validation/orchestration plus model and split sampling; no per-token upload/readback",
        "public_boundary": {
            "method": "QwenGenerator.decode_forward",
            "tokens": "Persistent CPU int64 tensor [1,1]; ignored with reload_inputs=False",
            "start_pos": "One CPU metadata position from cache.position per step; never uploaded",
            "kwargs": {
                "page_table": None,
                "kv_cache": "[generator.caches[0]]; fixed caller-owned cache binding",
                "slots": [0],
                "sample_on_device": True,
                "reload_inputs": False,
                "reload_page_table": False,
                "read_from_device": False,
            },
            "unprovided_defaults": {
                "reload_sampling_params": False,
                "reset_sampling_state": False,
                "sampling_params": None,
                "slot_remap": None,
                "page_tables_per_layer": None,
            },
            "result": "DecodeSubmission referencing persistent device token tensor; no process/read call",
        },
        "window_boundary_synchronizations": 2,
        "timed_boundary_synchronizations": 1,
        "counter_note": "Generator counters exclude the two explicit benchmark window boundary synchronizations",
        "steady_decode_counter_delta": steady,
        "final_readback_outside_timing": True,
        "first_token": int(initial_tokens[0]),
        "final_token": int(final_tokens[0]),
        "caller_read_control_first_and_final_match": True,
        "inactive_token_slots_unchanged": True,
        "persistent_token_and_sampler_input_identity_unchanged": True,
        "trace_ids": ids_before,
        "qsa_positions": acceptance.qsa_positions(generator, 255),
        "validation_scope": "First/final token and positions; per-step output correctness is a separate accuracy/feedback gate",
    }
    acceptance.marker("PERF_PUBLIC_RESIDENT_TOKEN_OUT_DONE", t_s_u=result["t_s_u"], ms_per_step=result["ms_per_step"])
    return result


def warmed_ttft(generator, model_dir, length):
    prompt = acceptance.natural_prompt(generator, model_dir, length)
    generator.generate(prompt, 2, enable_trace=True)
    ids_before = acceptance.trace_ids(generator)
    ttnn.synchronize_device(generator.mesh)
    reads = []
    original_read = generator._read_tokens

    def observed_read():
        value = original_read()
        reads.append(time.perf_counter())
        return value

    started = time.perf_counter()
    with patch.object(generator, "_read_tokens", side_effect=observed_read):
        tokens = generator.generate(prompt, 2, enable_trace=True)
    assert len(tokens) == len(reads) == 2
    assert acceptance.trace_ids(generator) == ids_before
    return {
        "prompt_tokens": length,
        "generated_tokens": 2,
        "batch": 1,
        "ttft_ms": (reads[0] - started) * 1000,
        "scope": "Warmed public generate: reset/setup, complete prefill, first device sample and caller read",
    }


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ttft-lengths", default="128", help="Comma-separated extra warmed public TTFT prompt lengths")
    parser.add_argument(
        "--public-resident",
        action="store_true",
        help="Also measure the public decode_forward contract with input/page reload and caller reads disabled",
    )
    options, forwarded = parser.parse_known_args()
    try:
        lengths = list(dict.fromkeys(int(value) for value in options.ttft_lengths.split(",")))
    except ValueError:
        parser.error("--ttft-lengths must be comma-separated positive integers")
    if not lengths or min(lengths) < 1:
        parser.error("--ttft-lengths must contain positive lengths")
    original_perf = acceptance.run_perf
    sources_before_setup = source_snapshot()

    def run_perf(generator, model_dir):
        result = original_perf(generator, model_dir)
        if hasattr(generator.model, "precision_summary"):
            result["runtime_precision"] = generator.model.precision_summary()
        if (Path(model_dir) / "doc/datatype_sweep/selected_precision_config.json").exists():
            from models.autoports.qwen_qwen3_8_flash_next.tests.selected_precision_snapshot import snapshot

            result["selected_default_snapshot"] = snapshot(generator, model_dir)
        result["resident_token_out"] = run_resident(generator, model_dir, result["generated_token_ids"])
        result["resident_token_out"]["layer_stack_overhead_ms"] = (
            result["resident_token_out"]["ms_per_step"] - result["decoder_stack_comparison"]["summed_layer_stack_ms"]
        )
        if options.public_resident:
            result["serving_contract_resident_token_out"] = run_public_resident(
                generator, model_dir, result["generated_token_ids"]
            )
            result["serving_contract_resident_token_out"]["layer_stack_overhead_ms"] = (
                result["serving_contract_resident_token_out"]["ms_per_step"]
                - result["decoder_stack_comparison"]["summed_layer_stack_ms"]
            )
        result["warmed_ttft_by_length"] = [
            (
                {
                    "prompt_tokens": 128,
                    "generated_tokens": 128,
                    "batch": 1,
                    "ttft_ms": result["ttft_ms"],
                    "scope": result["ttft_scope"],
                }
                if length == 128
                else warmed_ttft(generator, model_dir, length)
            )
            for length in lengths
        ]
        result["resident_benchmark_driver"] = str(Path(__file__).resolve())
        result["resident_benchmark_driver_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        sources_after_performance = source_snapshot()
        assert sources_before_setup == sources_after_performance, "Executable sources changed during resident benchmark"
        result["source_provenance"] = {
            "scope": "All model Python/native sources, shared sampling/CCL and both benchmark drivers",
            "before_setup_sha256": sources_before_setup,
            "after_performance_sha256": sources_after_performance,
            "unchanged": True,
            "snapshot_reads_outside_timed_windows": True,
        }
        return result

    if "--perf" not in forwarded:
        forwarded.append("--perf")
    with (
        patch.object(acceptance, "run_perf", side_effect=run_perf),
        patch.object(sys, "argv", [sys.argv[0], *forwarded]),
    ):
        acceptance.main()


if __name__ == "__main__":
    main()
