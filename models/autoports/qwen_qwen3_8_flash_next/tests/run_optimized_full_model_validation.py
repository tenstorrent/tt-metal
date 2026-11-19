# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage7 validation with one resident full48/C262144 generator.

Run serially under an external timeout after restoring the accepted lane env.
Selectors are independent; no selector means --accuracy. No profiler or maximum
context acceptance is run. Existing stage6 evidence is read, never overwritten.
Text execution success still requires separate content/blind review.
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch


@contextmanager
def audit_generator_bodies(generator, record):
    """Guard complete device bodies, leaving request/setup/scoring on the host.

    B1 preparation executes every physical prefill bucket, complete decode and
    common sampling variants. Later captures execute those bodies again. Guard
    the callable passed into _record_trace as well, so sampler seed progression,
    saved-token copies and inactive-row restoration are covered along with the
    common sampler. Trace begin/end and scheduler input uploads remain outside.
    """
    import torch

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit

    assert generator.max_batch_size == 1 and generator.factored_decode is None
    assert generator.model.layer_indices == tuple(range(48))
    assert not generator.prepared
    source = Path(sys.modules[runtime_audit.__module__].__file__)
    record.update(
        passed=False,
        scope="Full48 B1 prefill/decode bodies and complete recorded trace bodies including split sampling",
        guard_source=str(source),
        guard_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        blocked=["torch operations via NoTorchCalls", "ttnn.from_torch", "ttnn.to_torch", "TTNN fallback"],
        excluded="Weight/setup/request input preparation, reference scoring and caller/diagnostic reads",
        replay_note="Replays execute the audited captured command streams; Python body counts are not replay counts",
        layer_indices=list(generator.model.layer_indices),
        calls={},
        positive_controls={},
        bindings_restored=False,
    )
    # These controls cannot open a device: the existing guard intercepts the
    # conversion entrypoints before their arguments are interpreted.
    for name, operation in (
        ("torch", lambda: torch.zeros(1)),
        ("from_torch", lambda: ttnn.from_torch(None)),
        ("to_torch", lambda: ttnn.to_torch(None)),
    ):
        try:
            with runtime_audit():
                assert ttnn.CONFIG.throw_exception_on_fallback is True
                operation()
        except RuntimeError as error:
            assert "Runtime torch operation forbidden" in str(
                error
            ) or "Runtime TTNN/torch conversion forbidden" in str(error)
            record["positive_controls"][name] = {"rejected": True, "error": str(error)}
        else:
            raise AssertionError(f"Runtime audit positive control was not rejected: {name}")

    state = {"guard_depth": 0, "capture_depth": 0}

    def run(name, function, signature, *args, **kwargs):
        signature = {
            "phase": "runtime" if generator.prepared else "preparation",
            "inside_capture": bool(state["capture_depth"]),
            **signature,
        }
        key = json.dumps(signature, sort_keys=True)
        group = record["calls"].setdefault(name, {"calls": 0, "succeeded": 0, "failed": 0, "signatures": {}})
        group["calls"] += 1
        group["signatures"][key] = group["signatures"].get(key, 0) + 1
        try:
            if state["guard_depth"]:
                result = function(*args, **kwargs)
            else:
                with runtime_audit():
                    state["guard_depth"] += 1
                    try:
                        assert ttnn.CONFIG.throw_exception_on_fallback is True
                        result = function(*args, **kwargs)
                    finally:
                        state["guard_depth"] -= 1
        except BaseException as error:
            group["failed"] += 1
            group["last_error"] = f"{type(error).__name__}: {error}"
            raise
        group["succeeded"] += 1
        return result

    original_prefill = generator._prefill_body
    original_decode = generator._decode_body
    original_sampling = generator.sampler._run_sampling
    original_record = generator._record_trace

    def prefill(count, slot=None):
        return run("prefill_body", original_prefill, {"physical_bucket": int(count), "slot": slot}, count, slot)

    def decode(slots):
        return run("decode_body", original_decode, {"slots": list(slots)}, slots)

    def sampling(logits, **kwargs):
        signature = {
            "logits_shape": list(logits.shape),
            "logits_dtype": str(logits.dtype),
            "penalties_on": bool(kwargs["penalties_on"]),
            "grammar_on": bool(kwargs["grammar_on"]),
            "force_argmax": bool(generator.sampler.tt_sampling.force_argmax_sampling),
            "persistent_token_output": kwargs.get("tt_out_tok") is generator.decode_tokens,
            "count_tokens": bool(kwargs.get("count_tokens", True)),
        }
        return run("common_sampling_body", original_sampling, signature, logits, **kwargs)

    def record_trace(body):
        signature = {"callable": body.__qualname__}

        def audited_body():
            state["capture_depth"] += 1
            try:
                return run("complete_capture_body", body, signature)
            finally:
                state["capture_depth"] -= 1

        return original_record(audited_body)

    try:
        with ExitStack() as bindings:
            bindings.enter_context(patch.object(generator, "_prefill_body", new=prefill))
            bindings.enter_context(patch.object(generator, "_decode_body", new=decode))
            bindings.enter_context(patch.object(generator.sampler, "_run_sampling", new=sampling))
            bindings.enter_context(patch.object(generator, "_record_trace", new=record_trace))
            yield
    finally:
        record["bindings_restored"] = True
        record["passed"] = all(
            record["calls"].get(name, {}).get("succeeded", 0) > 0
            for name in ("prefill_body", "decode_body", "common_sampling_body", "complete_capture_body")
        ) and all(group["failed"] == 0 for group in record["calls"].values())


def prepare_controls(model_dir, destination):
    """Reuse byte-identical pinned HF controls without rewriting source evidence."""
    source = model_dir / "doc/full_model/qualitative"
    names = (
        "prompts.json",
        "qualitative_prompt_format.json",
        "stopping_config.json",
        "rubric.json",
        "hf_outputs.json",
        "hf_status.json",
        "hf_validation.json",
        "prompt_validation.json",
        "aime_autoregressive_hf.json",
    )
    assert json.loads((source / "hf_status.json").read_text())["status"] == "complete"
    if (destination / "tt_outputs.json").exists():
        raise FileExistsError("Preserve existing stage7 qualitative output before a new run")
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in names:
        original, target = source / name, destination / name
        if target.exists():
            if original.read_bytes() != target.read_bytes():
                raise ValueError(f"Existing control differs from pinned stage6 evidence: {target}")
        else:
            shutil.copyfile(original, target)
        copied.append({"source": str(original), "copy": str(target), "byte_identical": True})
    path = source / "run_controls.py"
    spec = importlib.util.spec_from_file_location("optimized_full_model_controls", path)
    controls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controls)
    controls.ROOT = destination
    return controls, copied


def run_context_smoke(generator, model_dir, acceptance):
    """Non-aligned 32K smoke; retains full capacity and does not touch max latch."""
    from models.common.sampling.generator import SamplingParams

    length = 32769
    record = {
        "logical_prompt_tokens": length,
        "generated_tokens": 2,
        "capacity": generator.capacity,
        "quality_claim": False,
        "memory_before": acceptance.memory(generator.mesh),
    }
    before = dict(generator.counters)
    started = time.monotonic()
    prompt = acceptance.natural_prompt(generator, model_dir, length)
    assert len(prompt) == length and generator.capacity == 262144
    generator.set_sampling_params(SamplingParams(temperature=0.0, top_k=1, top_p=1.0))
    acceptance.marker("CONTEXT_SMOKE_START", logical_prompt=length, generated_tokens=2)
    tokens = generator.generate(prompt, 2, enable_trace=True)
    assert len(tokens) == 2 and all(isinstance(t, int) and 0 <= t < generator.model.config.vocab_size for t in tokens)
    record.update(
        sampled_token_ids=tokens,
        qsa_positions=acceptance.qsa_positions(generator, length + 1),
        counter_delta=acceptance.counter_delta(generator.counters, before),
        memory_after=acceptance.memory(generator.mesh),
        terminal_finiteness=acceptance.terminal_logits_finite(generator),
        wall_seconds=time.monotonic() - started,
    )
    assert record["counter_delta"]["model_replays"] == 1
    assert record["terminal_finiteness"]["all_finite"], "Nonfinite smoke logits"
    # The tracker validates on replay, not when diagnostic helpers allocate.
    # Consume one more real device-feedback token after the finiteness helpers,
    # so a late helper/program-cache allocation cannot hide at workload end.
    # Keep this validation step separate from the requested two-token smoke.
    validation_before = dict(generator.counters)
    generator._replay_decode((0,))
    generator.caches[0].position += 1
    generator._sample((0,))
    record["final_validation_replay"] = {
        "scope": "One extra model+sampling replay after finiteness diagnostics; no host token refresh",
        "counter_delta": acceptance.counter_delta(generator.counters, validation_before),
        "qsa_positions": acceptance.qsa_positions(generator, length + 2),
    }
    delta = record["final_validation_replay"]["counter_delta"]
    assert delta["model_replays"] == delta["sampling_replays"] == 1
    assert all(
        delta[name] == 0 for name in ("token_refreshes", "position_refreshes", "page_table_refreshes", "readbacks")
    )
    record["wall_seconds_including_final_validation"] = time.monotonic() - started
    record["passed"] = True
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", choices=("optimized_full_model", "datatype_sweep"), default="optimized_full_model")
    for name in ("accuracy", "resume", "autoregressive", "qualitative", "context-smoke"):
        parser.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path; existing evidence is immutable")
    phases = ("accuracy", "resume", "autoregressive", "qualitative", "context_smoke")
    if not any(getattr(args, name) for name in phases):
        args.accuracy = True
    model_dir = Path(__file__).resolve().parents[1]
    stage_dir = model_dir / "doc" / args.stage
    if not args.output.resolve().is_relative_to(stage_dir):
        parser.error(f"--output must be under {stage_dir}")
    run_dir = args.output.with_suffix("")
    if run_dir.exists():
        parser.error("Use a new output stem; its artifact directory already exists")
    run_dir.mkdir(parents=True)
    report = {
        "passed": False,
        "stage": args.stage,
        "stage_acceptance_claim": False,
        "scope": "all_48_layers",
        "regime": "serial",
        "mesh": [1, 4],
        "capacity": 262144,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "requested_phases": [name for name in phases if getattr(args, name)],
        "command": [sys.executable, *sys.argv],
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_hashes": {
            str(path.relative_to(model_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((model_dir / "tt").rglob("*"))
            if path.is_file() and path.suffix in (".py", ".cpp", ".h", ".hpp")
        },
        "environment": {name: value for name, value in os.environ.items() if name.startswith(("TT_", "READINESS_"))},
        "artifacts": str(run_dir),
    }
    started = time.monotonic()
    generator = mesh = None
    audit_scope = ExitStack()
    phase = "preflight"
    acceptance = None
    try:
        expected = Path("tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto").resolve()
        assert (
            Path(os.environ.get("TT_MESH_GRAPH_DESC_PATH", "")).resolve() == expected
        ), "Restore accepted mesh descriptor"
        assert os.environ.get("TT_VISIBLE_DEVICES") == "0,1,2,3", "This serial run owns the accepted four chips"
        for name in ("TT_METAL_DEVICE_PROFILER", "TT_METAL_TRACE_PROFILER", "TT_METAL_STREAMING_PROFILER"):
            assert os.environ.get(name) in (None, "", "0"), f"No all-layer profiler allowed: {name}"
        assert int(os.environ.get("TT_METAL_PROFILE_PERF_COUNTERS") or "0") == 0
        report["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, timeout=10).strip()
        report["git_branch"] = subprocess.check_output(
            ["git", "branch", "--show-current"], text=True, timeout=10
        ).strip()

        # Imports occur after CLI/preflight so --help remains completely host-only.
        import torch

        import ttnn
        from models.autoports.qwen_qwen3_8_flash_next.tests import run_full_model_acceptance as acceptance

        report["runner_sources"] = {
            str(Path(acceptance.__file__)): hashlib.sha256(Path(acceptance.__file__).read_bytes()).hexdigest()
        }
        runner_names = []
        if args.accuracy:
            runner_names.extend(("run_prefill_check", "run_teacher_forcing"))
            report["reference"] = {
                "path": str(model_dir / "readiness_aime24_chat.refpt"),
                "provenance": str(model_dir / "doc/full_model/hf_reference"),
                "prompt_mode": "AIME24 chat template",
                "thresholds": {"top1_min": 0.90, "top5_min": 0.98, "top100": 1.0},
            }
        if args.resume:
            runner_names.append("run_prefill_resume_check")
        if args.autoregressive:
            runner_names.append("run_autoregressive")
        for name in runner_names:
            path = Path(importlib.import_module("readiness_check." + name).__file__)
            report["runner_sources"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        prompt_file = model_dir / "doc/full_model/hf_reference/prompt.txt"
        if args.autoregressive:
            report["autoregressive_exact_cache"] = acceptance.require_cached_autoregressive(model_dir, prompt_file)
            report["autoregressive_stopping"] = acceptance.pinned_stopping_config(model_dir)
        if args.qualitative:
            controls, copied = prepare_controls(model_dir, run_dir / "qualitative")
            report["qualitative_controls"] = copied
            path = Path(controls.__file__)
            report["runner_sources"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        torch.set_num_threads(8)
        phase = "setup"
        acceptance.marker("MODEL_LOAD_START", layers=48, capacity=262144)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0, l1_small_size=32768)
        generator = acceptance.build_generator(model_dir, mesh, max_seq_len=262144, max_batch_size=1)
        # Installed readiness runners normally own their generator. This driver
        # reuses one across phases and closes it once in the outer finally.
        audit_scope.enter_context(patch.object(generator, "teardown", new=lambda: None))
        if hasattr(generator.model, "precision_summary"):
            report["runtime_precision"] = generator.model.precision_summary()
        if args.stage == "datatype_sweep":
            from models.autoports.qwen_qwen3_8_flash_next.tests.selected_precision_snapshot import snapshot

            report["selected_default_snapshot"] = snapshot(generator, model_dir)
        assert len(generator.model.layers) == 48 and generator.model.layer_indices == tuple(range(48))
        report["memory_loaded"] = acceptance.memory(mesh)
        report["runtime_audit"] = {}
        audit_scope.enter_context(audit_generator_bodies(generator, report["runtime_audit"]))
        generator.prepare()
        if args.context_smoke:
            # This B1 generator prepares eagerly and defers all captures to
            # first use. Compile the exact final diagnostic signature now;
            # unlike model setup, diagnostics are not in generator.prepare().
            assert generator.factored_decode is None
            assert all(not values for values in acceptance.trace_ids(generator).values())
            report["terminal_finiteness_warmup"] = {
                "before_first_capture": True,
                "sample_logits_shape": list(generator.sample_logits.shape),
                "result": acceptance.terminal_logits_finite(generator),
                "scope": "Read-only diagnostic helper warmup; no request state mutation",
            }
        report["memory_prepared"] = acceptance.memory(mesh)
        report["setup_seconds"] = time.monotonic() - started
        acceptance.save(args.output, report)

        if args.accuracy:
            for name in ("run_prefill_check", "run_teacher_forcing"):
                phase = name
                acceptance.marker(name.upper() + "_START")
                module = importlib.import_module("readiness_check." + name)
                # The installed runner/scoring remains unchanged; only construction reuses resident weights.
                with patch.object(module, "_import_build_generator", return_value=lambda **kwargs: generator):
                    rows = getattr(module, name)(
                        model_dir=model_dir, reference_path=model_dir / "readiness_aime24_chat.refpt", mesh_device=mesh
                    )
                report[name] = rows
                acceptance.save(run_dir / (name + ".json"), rows)
                acceptance.save(args.output, report)
                assert rows and all(
                    row["top1"] >= 0.90 and row["top5"] >= 0.98 and row["top100"] == 1.0 for row in rows
                ), rows
                acceptance.marker(name.upper() + "_DONE", rows=rows)
        if args.resume:
            phase = "resume"
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
            path = run_dir / "prefill_resume_check.json"
            acceptance.save(path, result)
            report["resume"] = {"status": result["status"], "artifact": str(path)}
            acceptance.save(args.output, report)
            assert result["status"] == "pass", result["reasons"]
        if args.autoregressive:
            phase = "autoregressive"
            module = importlib.import_module("readiness_check.run_autoregressive")
            output_dir = run_dir / "autoregressive"
            with patch.object(module, "_import_build_generator", return_value=lambda **kwargs: generator):
                paths = module.run_autoregressive(
                    model_dir=model_dir,
                    hf_model_id=acceptance.MODEL_ID,
                    prompt_file=prompt_file,
                    mesh_device=mesh,
                    output_dir=output_dir,
                    max_new_tokens=128,
                )
            report["autoregressive"] = {key: str(value) for key, value in paths.items()}
            report["autoregressive"]["normalization"] = acceptance.normalize_autoregressive_artifacts(
                paths, generator.tokenizer, report["autoregressive_stopping"]
            )
            command = [
                sys.executable,
                str(Path(module.__file__).with_name("check_degenerate_output.py")),
                str(output_dir),
                "--missing-artifacts",
                "critical",
                "--scope",
                "autoregressive",
                "--json",
                str(run_dir / "autoregressive_degenerate_check.json"),
            ]
            log = run_dir / "autoregressive_degenerate_check.log"
            with log.open("w") as handle:
                completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, timeout=120)
            report["autoregressive"]["degenerate_check"] = {
                "command": command,
                "returncode": completed.returncode,
                "log": str(log),
            }
            report["autoregressive"]["content_review"] = "pending direct matched HF/TT review"
            acceptance.save(args.output, report)
            assert completed.returncode == 0, f"Degenerate output check failed: {log}"
        if args.qualitative:
            phase = "qualitative"
            acceptance.marker("QUALITATIVE_START", host_sampling=False, matched_seeds="0..9")
            path = controls.ROOT / "tt_outputs.json"
            outputs = controls.run_tt(generator, host_sampling=False, output_path=path)
            report["qualitative"] = {
                "artifact": str(path),
                "outputs": len(outputs),
                "host_sampling": False,
                "blind_verdict": "pending separate blind scoring",
            }
            acceptance.save(args.output, report)
        if args.context_smoke:
            phase = "context_smoke"
            report["context_smoke"] = run_context_smoke(generator, model_dir, acceptance)
            acceptance.save(run_dir / "context_smoke.json", report["context_smoke"])
        report["memory_final"] = acceptance.memory(mesh)
        report["counters"] = dict(generator.counters)
        report["trace_ids"] = acceptance.trace_ids(generator)
        audit_scope.close()
        assert report["runtime_audit"][
            "passed"
        ], "Runtime audit missed a required body or recorded a forbidden operation"
        report["passed"] = True
    except BaseException as exc:
        report.update(failed_phase=phase, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        audit_scope.close()
        report["wall_seconds"] = time.monotonic() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        try:
            if generator is not None:
                generator.close()
        finally:
            if mesh is not None:
                ttnn.close_mesh_device(mesh)
        print(
            json.dumps({"phase": "VALIDATION_DONE", "passed": report["passed"], "output": str(args.output)}), flush=True
        )


if __name__ == "__main__":
    main()
