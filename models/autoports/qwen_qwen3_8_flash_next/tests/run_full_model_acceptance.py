# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Selected acceptance phases over one resident all-48-layer generator.

No profiler is permitted. Run serially after lane-env, with the accepted mesh
descriptor restored. Without selectors, runs accuracy and clean B1 performance;
max context always requires its explicit flag. Examples::

    timeout 5400 python -m models.autoports.qwen_qwen3_8_flash_next.tests.run_full_model_acceptance \
      --accuracy --perf --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/acceptance_accuracy_perf.json
    timeout 7200 python -m models.autoports.qwen_qwen3_8_flash_next.tests.run_full_model_acceptance \
      --autoregressive --qualitative --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/acceptance_text.json
    timeout 43200 python -m models.autoports.qwen_qwen3_8_flash_next.tests.run_full_model_acceptance \
      --max-context --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/acceptance_max_context.json

Text phases require their exact HF cache/control artifacts before devices open.
The one max-context attempt is recorded in doc/full_model/max_context_attempt.json;
this harness never silently retries or replaces that record. The 262143-token
public prompt plus two generated tokens consumes exactly 262144 cache positions.
Execution success does not replace human/blind qualitative review or stage review.
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoTokenizer, GenerationConfig
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.run_full_model_accuracy import memory
from models.autoports.qwen_qwen3_8_flash_next.tt.generator import build_generator
from models.common.sampling.generator import SamplingParams

MODEL_ID = "Qwen/Qwen3.8-Flash-Next"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def marker(phase, **values):
    print(json.dumps({"phase": phase, **values}), flush=True)


def counter_delta(after, before):
    return {key: value - before.get(key, 0) for key, value in after.items()}


def trace_ids(generator):
    return {
        name: {str(key): str(value) for key, value in getattr(generator, name).items()}
        for name in ("traces", "prefill_traces", "sampling_traces")
    }


def qsa_positions(generator, expected):
    results = []
    for layer_index, state in zip(generator.model.layer_indices, generator.caches[0].layers, strict=True):
        if state.attention is None:
            continue
        for rank, tensor in enumerate(ttnn.get_device_tensors(state.attention.position)):
            value = ttnn.to_torch(tensor).flatten().tolist()
            assert value == [expected], f"Layer {layer_index} rank {rank}: position {value}, expected {expected}"
            results.append({"layer": layer_index, "rank": rank, "position": value[0]})
    assert results, "All-layer acceptance must contain QSA state"
    assert generator.caches[0].position == expected, "Scheduler cache position differs from device position"
    return results


def natural_prompt(generator, model_dir, length):
    runner = importlib.import_module("readiness_check.run_prefill_resume_check")
    text = runner.load_natural_text(None, model_dir / "doc/full_model/hf_reference/prompt.txt")
    return runner.natural_tokens(generator.tokenizer, text, length)


def decoder_lower_bound(generator, model_dir):
    counts = Counter()
    for layer in generator.model.layers:
        if layer.kind == "linear_attention":
            representative = 1 if layer.has_ple else 0
        elif layer.kind in ("indexed_attention", "full_attention") and not layer.has_ple:
            representative = 3
        else:
            raise ValueError(f"No stage5 representative for layer kind {layer.kind}, PLE={layer.has_ple}")
        counts[representative] += 1
    terms = []
    root = model_dir / "doc/optimized_multichip_decoder"
    for representative, count in sorted(counts.items()):
        path = root / f"real_mesh_final_l{representative}.json"
        artifact = json.loads(path.read_text())
        assert artifact["passed"] and artifact["regime"] == "serial" and artifact["trace_samples"]
        ms = float(artifact["decode_ms"])
        terms.append(
            {
                "representative_layer": representative,
                "count": count,
                "ms_per_layer": ms,
                "stack_ms": count * ms,
                "evidence": str(path),
            }
        )
    bound = sum(row["stack_ms"] for row in terms)
    return {
        "terms": terms,
        "summed_layer_stack_ms": bound,
        "layer_stack_t_s_u": 1000 / bound,
        "comparison_scope": "Stage5 serial B1 warmed blocking layer traces; estimate, not an end-to-end bound proof",
    }


def require_cached_autoregressive(model_dir, prompt_file):
    """Validate the exact installed runner key; never substitute refpt tokens."""
    from readiness_check.reference_cache import ReferenceCache, model_revision, prompt_hash, tokenizer_fingerprint

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, local_files_only=True)
    tokens = tokenizer.encode(prompt_file.read_text().strip(), add_special_tokens=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = ReferenceCache.make_key(
        kind="hf_greedy",
        hf_model_id=MODEL_ID,
        revision=model_revision(MODEL_ID),
        tokenizer=tokenizer_fingerprint(MODEL_ID, tokenizer),
        prompt=prompt_hash([int(t) for t in tokens]),
        settings={"max_new_tokens": 128, "do_sample": False, "num_beams": 1, "device": device},
    )
    cache = ReferenceCache.for_model_dir(model_dir)
    cached = cache.get_json(key)
    if not cache.enabled or not isinstance(cached, list) or not cached:
        raise RuntimeError(
            "Exact run_autoregressive HF greedy cache is not ready; complete the existing HF control first"
        )
    return {"directory": str(cache.directory), "key": key, "tokens": len(cached), "prompt_file": str(prompt_file)}


def pinned_stopping_config(model_dir):
    root = model_dir / "doc/full_model/qualitative"
    metadata = json.loads((root / "qualitative_prompt_format.json").read_text())
    stopping = json.loads((root / "stopping_config.json").read_text())
    assert metadata["hf_model"] == MODEL_ID and stopping["max_new_tokens"] == 128
    revision = metadata["revision"]
    assert stopping["source"] == f"{MODEL_ID}@{revision}/generation_config.json::eos_token_id"
    config = GenerationConfig.from_pretrained(MODEL_ID, revision=revision, local_files_only=True)
    eos = config.eos_token_id
    expected = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
    assert stopping["eos_token_ids"] == expected, "Stopping metadata differs from the pinned HF generation config"
    return stopping


def normalize_autoregressive_artifacts(paths, tokenizer, stopping):
    """Keep raw fixed-step evidence; expose matched EOS semantics to the checker."""
    meta_path = Path(paths["meta"])
    meta = json.loads(meta_path.read_text())
    assert meta["hf_model_id"] == MODEL_ID
    assert meta["max_new_tokens"] == stopping["max_new_tokens"]
    raw_path = meta_path.with_name("autoregressive_raw.json")
    save(raw_path, meta)
    summary = {"raw_meta": str(raw_path), "stopping": stopping, "backends": {}}
    for backend in ("hf", "tt"):
        item = meta[backend]
        raw = [int(token) for token in item.get("raw_token_ids", item["token_ids"])]
        first = next((i for i, token in enumerate(raw) if token in stopping["eos_token_ids"]), None)
        logical = raw if first is None else raw[: first + 1]
        text_path = Path(paths[f"{backend}_completion"])
        raw_text_path = text_path.with_name(f"{backend}_completion_raw.txt")
        raw_text_path.write_text(tokenizer.decode(raw, skip_special_tokens=False), encoding="utf-8")
        text_path.write_text(tokenizer.decode(logical, skip_special_tokens=False), encoding="utf-8")
        item.update(
            raw_token_ids=raw,
            raw_num_tokens=len(raw),
            token_ids=logical,
            num_tokens=len(logical),
            first_eos_index=first,
            post_eos_tokens_discarded=len(raw) - len(logical),
        )
        summary["backends"][backend] = {
            "raw_tokens": len(raw),
            "logical_tokens": len(logical),
            "first_eos_index": first,
            "raw_completion": str(raw_text_path),
        }
    meta["stopping"] = stopping
    meta["raw_artifact"] = str(raw_path)
    save(meta_path, meta)
    return summary


def run_perf(generator, model_dir):
    mesh = generator.mesh
    prompt = natural_prompt(generator, model_dir, 128)
    generator.set_sampling_params(SamplingParams(temperature=0.0, top_k=1, top_p=1.0))
    marker("PERF_WARMUP_START", prompt=128, generated=2)
    generator.generate(prompt, 2, enable_trace=True)
    ids_before = trace_ids(generator)
    ttnn.synchronize_device(mesh)
    baseline = dict(generator.counters)
    reads = []
    first_counters = None
    original_read = generator._read_tokens

    def observed_read():
        nonlocal first_counters
        value = original_read()
        reads.append(time.perf_counter())
        if first_counters is None:
            first_counters = dict(generator.counters)
        return value

    marker("PERF_TOKEN_OUT_START", prompt=128, generated=128)
    started = time.perf_counter()
    with patch.object(generator, "_read_tokens", side_effect=observed_read):
        tokens = generator.generate(prompt, 128, enable_trace=True)
    finished = time.perf_counter()
    assert len(tokens) == 128 and len(reads) == 128, "Expected one caller token read per output token"
    after = dict(generator.counters)
    steady = counter_delta(after, first_counters)
    for name in ("token_refreshes", "position_refreshes", "page_table_refreshes", "cache_resets", "captures"):
        assert steady.get(name, 0) == 0, f"Steady-state token-out performed {name}"
    assert steady["model_replays"] == 127 and steady["sampling_replays"] == 127
    assert trace_ids(generator) == ids_before, "Warmed performance request changed trace identities"
    decode_seconds = reads[-1] - reads[0]
    result = {
        "workload": {"batch": 1, "prompt_tokens": 128, "generated_tokens": 128, "decode_steps": 127},
        "ttft_ms": (reads[0] - started) * 1000,
        "ttft_scope": "Public generate: reset/setup, complete prefill, first on-device sample and caller token read",
        "token_out_decode_ms": decode_seconds * 1000 / 127,
        "token_out_decode_t_s_u": 127 / decode_seconds,
        "total_host_seconds": finished - started,
        "total_counter_delta": counter_delta(after, baseline),
        "steady_decode_counter_delta": steady,
        "generated_token_ids": [int(t) for t in tokens],
        "trace_ids": ids_before,
        "qsa_positions": qsa_positions(generator, 255),
        "timing_observer": "perf_counter after each normal token read; no extra device work/readback",
    }
    marker("PERF_LOGITS_ONLY_START", prompt=128, decode_steps=127)
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
    trace = generator._capture((0,))
    ttnn.synchronize_device(mesh)
    started = time.perf_counter()
    for _ in range(127):
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        generator.counters["model_replays"] += 1
        generator.caches[0].position += 1
    ttnn.synchronize_device(mesh)
    elapsed = time.perf_counter() - started
    result["traced_logits_only"] = {
        "decode_steps": 127,
        "host_seconds": elapsed,
        "ms_per_step": elapsed * 1000 / 127,
        "t_s_u": 127 / elapsed,
        "trace_id": str(trace),
        "scope": "Model trace only: fixed resident token repeated, device positions advance; no sampling/readback. Not teacher-forcing accuracy.",
        "blocking_replay": False,
        "synchronization": "One mesh synchronization at each window boundary",
    }
    result["traced_logits_only"]["qsa_positions"] = qsa_positions(generator, 255)
    bound = decoder_lower_bound(generator, model_dir)
    bound["token_out_overhead_ms"] = result["token_out_decode_ms"] - bound["summed_layer_stack_ms"]
    bound["logits_only_overhead_ms"] = result["traced_logits_only"]["ms_per_step"] - bound["summed_layer_stack_ms"]
    result["decoder_stack_comparison"] = bound
    result[
        "teacher_forcing_metric_note"
    ] = "Installed readiness teacher forcing includes sampling/token read; see accuracy phase separately"
    marker("PERF_DONE", ttft_ms=result["ttft_ms"], token_out_t_s_u=result["token_out_decode_t_s_u"])
    return result


def terminal_logits_finite(generator):
    """Post-run diagnostic: read one finiteness bit per vocabulary shard.

    Call only after the final replay. These temporary reductions are released
    before any later device work; they never supply tokens or affect sampling.
    Qwen's vocabulary is exactly divisible by four, with no masked padding.
    """
    assert generator.model.config.vocab_size == generator.sample_logits.shape[-1] * 4
    finite = ttnn.min(ttnn.isfinite(generator.sample_logits[:, :, :1, :]), dim=-1, keepdim=True)
    values = [ttnn.to_torch(rank).reshape(-1).tolist() for rank in ttnn.get_device_tensors(finite)]
    assert len(values) == 4 and all(len(value) == 1 for value in values), values
    return {
        "all_finite": all(value[0] == 1 for value in values),
        "per_rank_scalar": values,
        "scope": "final active-row terminal logits; no full-logit readback or quality claim",
    }


def run_max_context(generator, model_dir, output):
    attempt_path = model_dir / "doc/full_model/max_context_attempt.json"
    assert generator.capacity == 262144 and generator.model.config.max_position_embeddings == 262144
    record = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(output),
        "prompt_length": 262143,
        "generated_tokens": 2,
        "consumed_context": 262144,
        "quality_claim": False,
    }
    # Exclusive creation prevents accidental repeated expensive acceptance runs.
    with attempt_path.open("x") as handle:
        json.dump(record, handle, indent=2)
    stop = threading.Event()
    started = time.monotonic()
    baseline = dict(generator.counters)

    def progress():
        while not stop.wait(30):
            marker(
                "MAX_CONTEXT_PROGRESS",
                wall_seconds=time.monotonic() - started,
                submitted_prefill_replays=generator.counters["prefill_replays"] - baseline["prefill_replays"],
                note="Host submission count, not a device-completion claim",
            )

    worker = threading.Thread(target=progress, daemon=True)
    try:
        record["memory_before"] = memory(generator.mesh)
        prompt = natural_prompt(generator, model_dir, 262143)
        generator.set_sampling_params(SamplingParams(temperature=0.0, top_k=1, top_p=1.0))
        marker("MAX_CONTEXT_START", logical_prompt=262143, new_tokens=2)
        worker.start()
        tokens = generator.generate(prompt, 2, enable_trace=True)
        assert len(tokens) == 2 and all(
            isinstance(t, int) and 0 <= t < generator.model.config.vocab_size for t in tokens
        )
        record["qsa_positions"] = qsa_positions(generator, 262144)
        record["sampled_token_ids"] = tokens
        record["memory_after"] = memory(generator.mesh)
        record["counter_delta"] = counter_delta(generator.counters, baseline)
        assert record["counter_delta"]["model_replays"] == 1
        record["terminal_finiteness"] = terminal_logits_finite(generator)
        assert record["terminal_finiteness"]["all_finite"], "Nonfinite final max-context logits"
        record["status"] = "pass"
        return record
    except BaseException as exc:
        record["status"] = "fail"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        stop.set()
        if worker.is_alive():
            worker.join(timeout=1)
        record["wall_seconds"] = time.monotonic() - started
        save(attempt_path, record)
        marker("MAX_CONTEXT_DONE", status=record["status"], wall_seconds=record["wall_seconds"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    for flag in ("accuracy", "autoregressive", "qualitative", "perf", "max-context"):
        parser.add_argument("--" + flag, action="store_true")
    parser.add_argument("--resume", action="store_true", help="Add the installed resumed-prefill check to --accuracy")
    parser.add_argument("--capacity", type=int, default=262144)
    parser.add_argument("--trace-region-size", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path; prior acceptance evidence is immutable")
    if not any((args.accuracy, args.autoregressive, args.qualitative, args.perf, args.max_context)):
        args.accuracy = args.perf = True
    if args.resume and not args.accuracy:
        parser.error("--resume requires --accuracy")
    if args.max_context and args.capacity != 262144:
        parser.error("Max-context acceptance requires --capacity 262144")
    model_dir = Path(__file__).resolve().parents[1]
    prompt_file = model_dir / "doc/full_model/hf_reference/prompt.txt"
    selected = [
        name for name in ("accuracy", "autoregressive", "qualitative", "perf", "max_context") if getattr(args, name)
    ]
    report = {
        "passed": False,
        "scope": "all_48_layers",
        "regime": "serial",
        "capacity": args.capacity,
        "requested_phases": selected,
        "stage_acceptance_claim": False,
        "source_hashes": {
            str(path.relative_to(model_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((model_dir / "tt").glob("*.py"))
        },
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "TT_VISIBLE_DEVICES",
                "TT_MESH_GRAPH_DESC_PATH",
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_TRACE_ALLOC_TRACEBACKS",
                "TT_METAL_DEVICE_PROFILER",
                "TT_METAL_PROFILE_PERF_COUNTERS",
                "TT_METAL_WATCHER",
            )
        },
    }
    started = time.monotonic()
    generator = mesh = None
    phase = "preflight"
    try:
        assert os.environ.get("TT_MESH_GRAPH_DESC_PATH"), "Restore accepted descriptor after lane-env"
        for name in ("TT_METAL_DEVICE_PROFILER", "TT_METAL_TRACE_PROFILER", "TT_METAL_STREAMING_PROFILER"):
            assert os.environ.get(name) in (None, "", "0"), f"No all-layer profiler allowed: {name}"
        assert (
            int(os.environ.get("TT_METAL_PROFILE_PERF_COUNTERS") or "0") == 0
        ), "TT_METAL_PROFILE_PERF_COUNTERS independently enables the device profiler"
        if args.perf:
            assert (
                not trace_allocation_tracker.TRACE_ALLOC_TRACKING
                and not trace_allocation_tracker.TRACE_ALLOC_DIAGNOSTICS
            )
            assert (
                "TT_METAL_WATCHER" not in os.environ
            ), "Clean performance requires TT_METAL_WATCHER unset; even an empty value or0 enables watcher"
        if args.max_context and (model_dir / "doc/full_model/max_context_attempt.json").exists():
            raise RuntimeError(
                "A max-context attempt already exists; this driver will not repeat acceptance automatically"
            )
        if args.autoregressive:
            report["autoregressive_exact_cache"] = require_cached_autoregressive(model_dir, prompt_file)
            report["autoregressive_stopping"] = pinned_stopping_config(model_dir)
        if args.qualitative:
            hf_status = json.loads((model_dir / "doc/full_model/qualitative/hf_status.json").read_text())
            assert hf_status["status"] == "complete", "Finish the matched HF suite before final paired TT evidence"
        torch.set_num_threads(8)
        marker("MODEL_LOAD_START", layers=48, capacity=args.capacity, requested=selected)
        phase = "setup"
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(1, 4), trace_region_size=args.trace_region_size, l1_small_size=32768
        )
        generator = build_generator(model_dir, mesh, max_seq_len=args.capacity, max_batch_size=1)
        assert len(generator.model.layers) == 48 and generator.model.layer_indices == tuple(range(48))
        report["memory_loaded"] = memory(mesh)
        marker("MODEL_PREPARE_START")
        generator.prepare()
        report["memory_prepared"] = memory(mesh)
        report["setup_seconds"] = time.monotonic() - started
        save(args.output, report)
        marker("MODEL_READY", setup_seconds=report["setup_seconds"])
        if args.accuracy:
            for name in ("run_prefill_check", "run_teacher_forcing"):
                phase = name
                marker(name.upper() + "_START")
                module = importlib.import_module("readiness_check." + name)
                with patch.object(module, "_import_build_generator", return_value=lambda **kwargs: generator):
                    rows = getattr(module, name)(
                        model_dir=model_dir, reference_path=model_dir / "readiness_aime24_chat.refpt", mesh_device=mesh
                    )
                report[name] = rows
                save(args.output, report)
                assert all(row["top5"] >= 0.98 and row["top100"] == 1.0 for row in rows), rows
                marker(name.upper() + "_DONE", rows=rows)
            if args.resume:
                phase = "resume"
                marker("PREFILL_RESUME_START")
                module = importlib.import_module("readiness_check.run_prefill_resume_check")
                with patch.object(module, "_import_build_generator", return_value=lambda **kwargs: generator):
                    result = module.run_prefill_resume_check(
                        model_dir=model_dir,
                        mesh_device=mesh,
                        lengths=[257],
                        block_sizes=[32],
                        sliding_window=None,
                        tile=32,
                        window=128,
                        controls=module.CONTROL_NAMES,
                        thresholds=dict(module.DEFAULT_THRESHOLDS),
                        vocab_size=generator.model.config.vocab_size,
                        seed=0,
                    )
                resume_path = model_dir / "readiness/prefill_resume_check.json"
                save(resume_path, result)
                report["resume"] = {"status": result["status"], "artifact": str(resume_path)}
                assert result["status"] == "pass", result["reasons"]
                marker("PREFILL_RESUME_DONE", status=result["status"])
        if args.perf:
            phase = "perf"
            report["performance"] = run_perf(generator, model_dir)
            save(args.output, report)
        if args.autoregressive:
            phase = "autoregressive"
            marker("AUTOREGRESSIVE_START", generation_tokens=128)
            module = importlib.import_module("readiness_check.run_autoregressive")
            with patch.object(module, "_import_build_generator", return_value=lambda **kwargs: generator):
                paths = module.run_autoregressive(
                    model_dir=model_dir,
                    hf_model_id=MODEL_ID,
                    prompt_file=prompt_file,
                    mesh_device=mesh,
                    output_dir=model_dir / "readiness_autoregressive",
                    max_new_tokens=128,
                )
            report["autoregressive"] = {key: str(value) for key, value in paths.items()}
            report["autoregressive"]["normalization"] = normalize_autoregressive_artifacts(
                paths, generator.tokenizer, report["autoregressive_stopping"]
            )
            check = Path(module.__file__).with_name("check_degenerate_output.py")
            command = [
                sys.executable,
                str(check),
                "--hf-model",
                MODEL_ID,
                "--missing-artifacts",
                "critical",
                "--scope",
                "autoregressive",
            ]
            log = model_dir / "doc/full_model/autoregressive_degenerate_check.log"
            with log.open("w") as handle:
                completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, timeout=120)
            report["autoregressive"]["degenerate_check"] = {
                "command": command,
                "returncode": completed.returncode,
                "log": str(log),
            }
            report["autoregressive"][
                "content_review"
            ] = "pending direct HF/TT reading and matched controls where needed"
            assert completed.returncode == 0, f"Degenerate output check failed; inspect {log}"
            marker("AUTOREGRESSIVE_DONE", artifacts=report["autoregressive"])
            save(args.output, report)
        if args.qualitative:
            phase = "qualitative"
            marker("QUALITATIVE_START", sampling="on-device", matched_seeds="0..9")
            path = model_dir / "doc/full_model/qualitative/run_controls.py"
            spec = importlib.util.spec_from_file_location("full_model_acceptance_controls", path)
            controls = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(controls)
            output_path = path.parent / "tt_outputs.json"
            original_generate = generator.generate

            def observed_generate(*a, **kw):
                marker("QUALITATIVE_REQUEST_START", prompt_tokens=len(a[0]), requested_tokens=a[1])
                output = original_generate(*a, **kw)
                marker("QUALITATIVE_REQUEST_DONE", output_tokens=len(output))
                return output

            with patch.object(generator, "generate", side_effect=observed_generate):
                outputs = controls.run_tt(generator, host_sampling=False, output_path=output_path)
            report["qualitative"] = {
                "artifact": str(output_path),
                "outputs": len(outputs),
                "host_sampling": False,
                "blind_verdict": "pending separate blind scoring",
            }
            marker("QUALITATIVE_DONE", outputs=len(outputs))
            save(args.output, report)
        if args.max_context:
            phase = "max_context"
            report["max_context"] = run_max_context(generator, model_dir, args.output)
        report["memory_final"] = memory(mesh)
        report["counters"] = dict(generator.counters)
        report["passed"] = True
    except BaseException as exc:
        report["failed_phase"] = phase
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        save(args.output, report)
        if generator is not None:
            generator.close()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        marker("ACCEPTANCE_DRIVER_DONE", passed=report["passed"], completed_phase=phase, output=str(args.output))


if __name__ == "__main__":
    main()
