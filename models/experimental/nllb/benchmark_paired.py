# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""Whole-generation full/last paired benchmark. External FP32 qualification required.
Run one public case per invocation; raw receipts remain in the requested output directory.
Provisional export: timing/body preserved; no historical scores or qualification inherited.
"""

import argparse, hashlib, json, sys, time, traceback
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import torch

if __package__:
    from .nllb_validation import (
        integer,
        validate_config,
        validate_inputs,
        validate_checkpoint_config,
        _unique_checkpoint_object,
    )
else:
    from nllb_validation import (
        integer,
        validate_config,
        validate_inputs,
        validate_checkpoint_config,
        _unique_checkpoint_object,
    )


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_identity(weights_path, config_path):
    """Mirror existing load_checkpoint routing; never load learned tensors here."""
    path = Path(weights_path)
    root = path if path.is_dir() else path.parent
    files = []
    index = root / "pytorch_model.bin.index.json"
    if path.is_file():
        files.append(path)
        layout = "single_file"
    elif path.is_dir() and index.is_file():
        document = json.loads(index.read_text(), object_pairs_hook=_unique_checkpoint_object)
        mapping = document.get("weight_map") if isinstance(document, dict) else None
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Checkpoint index requires a nonempty weight_map")
        shards = set()
        for name, shard in mapping.items():
            if not isinstance(name, str) or not name or not isinstance(shard, str) or not shard:
                raise ValueError("Checkpoint index names must be nonempty strings")
            resolved = (root / shard).resolve()
            if not resolved.is_relative_to(root.resolve()):
                raise ValueError("Checkpoint shard must remain inside checkpoint directory")
            shards.add(resolved)
        files = [index] + sorted(shards)
        layout = "sharded"
    elif path.is_dir():
        files.append(root / "pytorch_model.bin")
        layout = "single_file_directory"
    else:
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    metadata = [Path(config_path)]
    if (root / "config.json").is_file():
        metadata.append(root / "config.json")

    def identities(paths):
        return {str(p): {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(set(paths))}

    return {
        "loader_path": str(path),
        "layout": layout,
        "consumed_checkpoint_files": identities(files),
        "configuration_files": identities(metadata),
    }


CASES = ("short_b1", "short_b2", "long_b1", "long_b2", "source256_b1", "source256_b4")


def workload(name, arrays, suite):
    key = "single0" if name.startswith("short") else "length-33"
    if name == "short_b2":
        key = "batch"
    ids = arrays[key + "__input_ids"].copy()
    mask = arrays[key + "__attention_mask"].copy()
    target = next(c["target_id"] for c in suite["cases"] if c["name"] == key)
    cap = 8 if name.startswith(("short", "source")) else 40
    note = "unchanged public bringup request"
    if name == "long_b2":
        ids = np.concatenate([ids, np.pad(arrays["length-32__input_ids"], ((0, 0), (0, 1)), constant_values=1)])
        mask = np.concatenate([mask, np.pad(arrays["length-32__attention_mask"], ((0, 0), (0, 1)))])
        note = "public length33 plus padded public length32"
    if name.startswith("source256"):
        # Explicit synthetic envelope workload from public token IDs, not text quality.
        row = np.resize(ids[0], 256)
        row[-2:] = ids[0, -2:]
        count = 4 if name.endswith("b4") else 1
        ids = np.stack([np.roll(row, i) for i in range(count)])
        mask = np.ones_like(ids)
        note = "synthetic source256: repeated public length33 tokens, distinct rolled rows"
    return ids, mask, int(target), cap, note


def canonical(tokens, *, config, batch, target_id, max_new_tokens):
    """Validate the complete raw result before removing only legal EOS padding."""
    vocab = integer(config["vocab_size"], "vocab_size", 1, 2**31 - 1)
    start, eos, pad = [
        integer(config[key], key, 0, vocab - 1) for key in ("decoder_start_token_id", "eos_token_id", "pad_token_id")
    ]
    target = integer(target_id, "target_id", 0, vocab - 1)
    if target in {
        config.get(key)
        for key in ("decoder_start_token_id", "eos_token_id", "pad_token_id", "bos_token_id", "unk_token_id")
    }:
        raise ValueError("forced target must not be a reserved token")
    cap = integer(max_new_tokens, "max_new_tokens", 1, 256)
    batch = integer(batch, "batch", 1, 4)
    if not isinstance(tokens, np.ndarray) or tokens.ndim != 2:
        raise ValueError("output must be a rank-two NumPy array")
    if tokens.shape[0] != batch:
        raise ValueError("output batch does not match request")
    if not 2 <= tokens.shape[1] <= cap + 1:
        raise ValueError("output length must include start and target and respect cap")
    if tokens.dtype.kind not in "iu":
        raise ValueError("output tokens must have integer dtype")
    if np.any(tokens < 0) or np.any(tokens >= vocab):
        raise ValueError("output token outside vocabulary")
    rows = []
    for row in tokens.tolist():
        if row[0] != start:
            raise ValueError("wrong decoder start")
        if row[1] != target:
            raise ValueError("wrong forced target or invalid EOS placement")
        # Position zero may itself be EOS: it is the decoder start, not termination.
        ends = [i for i in range(1, len(row)) if row[i] == eos]
        if ends:
            end = ends[0]
            if end < 2:
                raise ValueError("EOS before forced target")
            if any(token != pad for token in row[end + 1 :]):
                raise ValueError("non-PAD token after first EOS")
            rows.append(row[: end + 1])
        else:
            if len(row) != cap + 1:
                raise ValueError("output ended before cap without EOS")
            rows.append(row)
    return rows


PACKAGE = Path(__file__).resolve().parent
SOURCE_FILES = (
    "__init__.py",
    "benchmark_paired.py",
    "backend.py",
    "nllb_validation.py",
    "generation_projection.txt",
    "trace_decode.py",
    "runtime_setup.py",
)


def source_identity():
    files = {name: sha(PACKAGE / name) for name in SOURCE_FILES}
    return {
        "files": files,
        "sha256": hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }


def snapshot(device):
    result = {"scope": "endpoint snapshots, not peaks"}
    result["rss_bytes"] = (
        int(next(x.split()[1] for x in Path("/proc/self/status").read_text().splitlines() if x.startswith("VmRSS:")))
        * 1024
    )
    for name in ("DRAM", "L1"):
        try:
            view = ttnn.device.get_memory_view(device, getattr(ttnn.BufferType, name))
            result[name] = {
                k: int(getattr(view, k))
                for k in (
                    "num_banks",
                    "total_bytes_per_bank",
                    "total_bytes_allocated_per_bank",
                    "total_bytes_free_per_bank",
                )
                if hasattr(view, k)
            }
        except Exception as e:
            result[name] = {"unavailable": repr(e)}
    return result


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--input-case", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--precision", choices=("bf16", "bfp8_b"), required=True)
    p.add_argument("--pairs", type=int, default=6)
    p.add_argument("--validation", action="store_true", help="cap4 mechanics diagnostic; not a speedup claim")
    a = p.parse_args(argv)
    if a.pairs < 6:
        p.error("at least six pairs required")
    if a.device < 0:
        p.error("device must be nonnegative")
    return a


def load_case(path, config_path, config):
    data = json.loads(Path(path).read_text(), object_pairs_hook=_unique_checkpoint_object)
    required = {"name", "input_ids", "attention_mask", "target_id", "max_new_tokens", "workload_note", "config_sha256"}
    if not isinstance(data, dict) or set(data) != required:
        raise ValueError("case requires exactly the documented public fields")
    if data["config_sha256"] != sha(config_path):
        raise ValueError("case configuration hash mismatch")
    if not isinstance(data["name"], str) or not data["name"] or not isinstance(data["workload_note"], str):
        raise ValueError("case name and workload note must be strings")
    ids, mask = np.asarray(data["input_ids"]), np.asarray(data["attention_mask"])
    validate_inputs(ids, mask, config)
    target = integer(data["target_id"], "target_id", 0, config["vocab_size"] - 1)
    cap = integer(data["max_new_tokens"], "max_new_tokens", 1, 256)
    canonical(
        np.array([[config["decoder_start_token_id"], target]], dtype=np.int64),
        config=config,
        batch=1,
        target_id=target,
        max_new_tokens=1,
    )
    return data["name"], ids, mask, target, cap, data["workload_note"]


def emit(x):
    print("PAIRED " + json.dumps(x, sort_keys=True), flush=True)


def run(a):
    global ttnn, backend
    cfg = validate_config(json.loads(a.config.read_text(), object_pairs_hook=_unique_checkpoint_object))
    validate_checkpoint_config(a.checkpoint, cfg)
    weights_path = a.checkpoint
    checkpoint = checkpoint_identity(weights_path, a.config)
    name, ids, mask, target, cap, note = load_case(a.input_case, a.config, cfg)
    if a.validation:
        cap = 4
    if __package__:
        from . import runtime_setup
    else:
        import runtime_setup
    runtime_setup.configure_tracking()
    import ttnn

    if __package__:
        from . import backend
    else:
        import backend
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    saved_ids, saved_mask = ids.copy(), mask.copy()
    report = {
        "case": name,
        "precision": a.precision,
        "validation_only": a.validation,
        "qualification": "external unchanged evaluator receipts required for BOTH paths; this run cannot qualify itself",
        "source": source_identity(),
        "backend_sha256": sha(PACKAGE / "backend.py"),
        "selection": (PACKAGE / "generation_projection.txt").read_text(),
        "selector_sha256": sha(PACKAGE / "generation_projection.txt"),
        "control": "same loaded backend, full vocabulary projection versus last row",
        "checkpoint": checkpoint,
        "config_sha256": sha(a.config),
        "input_case_sha256": sha(a.input_case),
        "config": cfg,
        "diagnostic_only": True,
        "accepted": False,
        "runtime_file": ttnn.__file__,
        "ttnn_version": str(getattr(ttnn, "__version__", "unavailable")),
        "input_ids": ids.tolist(),
        "attention_mask": mask.tolist(),
        "target_id": target,
        "cap": cap,
        "workload_note": note,
        "cold": [],
        "warmups": [],
        "samples": [],
        "pairs": [],
        "mismatches": [],
        "memory_scope": "process RSS and device allocation endpoint snapshots, not peaks",
        "bandwidth": "not measured",
        "cold_scope": "one shared weight load; mode first calls share runtime/program caches",
    }
    emit({"identity": report})
    owner = runtime_setup.RuntimeOwner()
    primary = None
    d = None
    model = None
    report["success"] = False
    try:
        start = time.perf_counter()
        d = owner.open(a.device)
        ttnn.synchronize_device(d)
        report["device_open_seconds"] = time.perf_counter() - start
        report["before_load"] = snapshot(d)
        ttnn.synchronize_device(d)
        start = time.perf_counter()
        model = owner.bind(backend.create_backend(str(weights_path), cfg, d, precision=a.precision))
        ttnn.synchronize_device(d)
        report["cold_model_load_seconds"] = time.perf_counter() - start
        report["after_load"] = snapshot(d)
        report["precision_policy"] = model.precision_policy
        report["factory_selection"] = model.generation_projection
        if model.precision_policy.get("mode") != a.precision:
            raise ValueError("effective precision differs from requested precision")
        if model.generation_projection not in ("full", "last"):
            raise ValueError("backend did not declare a supported projection selector")

        def checked(tokens):
            return canonical(tokens, config=cfg, batch=ids.shape[0], target_id=target, max_new_tokens=cap)

        def call(mode, kind, index):
            model.generation_projection = mode
            ttnn.synchronize_device(d)
            start = time.perf_counter()
            tokens = model.generate(ids, mask, target, cap)
            ttnn.synchronize_device(d)
            elapsed = time.perf_counter() - start
            # All canonicalization, verification, memory probes and writes are OUTSIDE timing.
            try:
                rows = checked(tokens)
            except ValueError as error:
                report["mismatches"].append(
                    {
                        "phase": kind,
                        "mode": mode,
                        "index": index,
                        "error": str(error),
                        "raw_type": type(tokens).__name__,
                        "raw_dtype": str(getattr(tokens, "dtype", None)),
                        "raw_shape": list(getattr(tokens, "shape", ())),
                        "raw_tokens": tokens.tolist() if isinstance(tokens, np.ndarray) else repr(tokens),
                    }
                )
                raise
            completed = sum(len(r) - 1 for r in rows)  # includes forced target and EOS
            sample = {
                "mode": mode,
                "kind": kind,
                "index": index,
                "seconds": elapsed,
                "tokens": tokens.tolist(),
                "canonical_tokens": rows,
                "completed_new_tokens": completed,
                "tokens_per_second": completed / elapsed,
                "memory": snapshot(d),
                "input_preserved": bool(np.array_equal(ids, saved_ids) and np.array_equal(mask, saved_mask)),
            }
            if not sample["input_preserved"]:
                raise AssertionError("input mutation")
            if "cross_kv" in model.__dict__:
                raise AssertionError("persistent cross KV")
            emit({"sample": sample})
            return sample

        for mode in ("full", "last"):
            report["cold"].append(call(mode, "mode_first_call", 0))
        # Untimed state audit: real decode fills cross-KV before normal return
        # and before an injected projection error. Restore all probes before warming.
        report["cleanup_checks"] = []
        for mode in ("full", "last"):
            model.generation_projection = mode
            captured = []
            original_decode = model.decode
            original_linear = ttnn.linear
            armed = [False]

            def auditing_decode(*args, **kw):
                cache = kw.get("cross_kv")
                if cache is not None and all(cache is not c for c in captured):
                    captured.append(cache)
                return original_decode(*args, **kw)

            def injected_linear(*args, **kw):
                if armed[0] and len(args) > 1 and args[1] is model.lm_weight:
                    armed[0] = False
                    assert captured and any(captured)
                    raise RuntimeError("paired benchmark untimed cleanup injection")
                return original_linear(*args, **kw)

            model.decode = auditing_decode
            ttnn.linear = injected_linear
            try:
                normal = model.generate(ids, mask, target, cap)
                ttnn.synchronize_device(d)
                assert captured and all(not c for c in captured)
                captured.clear()
                armed[0] = True
                try:
                    model.generate(ids, mask, target, cap)
                except RuntimeError as e:
                    assert str(e) == "paired benchmark untimed cleanup injection"
                else:
                    raise AssertionError("injection did not fire")
                assert captured and all(not c for c in captured)
                captured.clear()
                recovery = model.generate(ids, mask, target, cap)
                ttnn.synchronize_device(d)
                assert captured and all(not c for c in captured)
                assert checked(normal) == checked(recovery) == report["cold"][0]["canonical_tokens"]
                assert np.array_equal(ids, saved_ids) and np.array_equal(mask, saved_mask)
                report["cleanup_checks"].append({"mode": mode, "normal_exception_recovery": True})
            finally:
                model.decode = original_decode
                ttnn.linear = original_linear
        # Both modes explicitly warm before measured pairs.
        for mode in ("last", "full"):
            report["warmups"].append(call(mode, "warmup", 0))
        expected = report["cold"][0]["canonical_tokens"]
        for s in report["cold"] + report["warmups"]:
            if s["canonical_tokens"] != expected:
                report["mismatches"].append({"phase": s["kind"], "mode": s["mode"], "index": s["index"]})
        for i in range(a.pairs):
            order = ("full", "last") if i % 2 == 0 else ("last", "full")
            pair = {}
            for mode in order:
                s = call(mode, "measured", i)
                report["samples"].append(s)
                pair[mode] = s
                if s["canonical_tokens"] != expected:
                    report["mismatches"].append({"phase": "measured_vs_first", "mode": mode, "index": i})
            equal = pair["full"]["canonical_tokens"] == pair["last"]["canonical_tokens"]
            report["pairs"].append(
                {
                    "index": i,
                    "order": list(order),
                    "exact_canonical_tokens": equal,
                    "full_seconds": pair["full"]["seconds"],
                    "last_seconds": pair["last"]["seconds"],
                }
            )
            if not equal:
                report["mismatches"].append({"phase": "pair", "index": i})
        report["summary"] = {}
        for mode in ("full", "last"):
            samples = [s for s in report["samples"] if s["mode"] == mode]
            seconds = [s["seconds"] for s in samples]
            report["summary"][mode] = {
                "raw_seconds": seconds,
                "median_seconds": float(np.median(seconds)),
                "p95_seconds": float(np.percentile(seconds, 95)),
                "p95_note": "linear interpolation; small sample, not a tail SLA",
                "completed_new_tokens": sum(s["completed_new_tokens"] for s in samples),
                "aggregate_tokens_per_second": sum(s["completed_new_tokens"] for s in samples) / sum(seconds),
            }
        report["after_request"] = snapshot(d)
        report["source_after"] = source_identity()
        report["source_unchanged"] = report["source_after"] == report["source"]
        report["success"] = not report["mismatches"] and report["source_unchanged"]
    except BaseException as e:
        primary = e
        report["success"] = False
        try:
            report["error"] = repr(e)
            traceback.print_exc()
        except BaseException:
            pass
        if not isinstance(e, Exception):
            raise
    finally:
        if model is not None and "factory_selection" in report:
            try:
                model.generation_projection = report["factory_selection"]
            except BaseException as error:
                if primary is None:
                    primary = error
                report["success"] = False
                try:
                    report["restore_error"] = repr(error)
                except BaseException:
                    pass
        try:
            owner.finish(primary)
        except BaseException as error:
            if primary is None:
                primary = error
            report["success"] = False
        report["device_closed"] = owner.closed and owner.device is not None
        report["owner_retained"] = owner in runtime_setup.retained_owners()
        if owner.cleanup_errors:
            report["success"] = False
            try:
                report["close_error"] = repr(owner.cleanup_errors[0])
            except BaseException:
                pass
        try:
            (a.output / "paired.json").write_text(json.dumps(report, indent=2))
            emit({"final": report})
        except BaseException:
            if primary is not None:
                raise primary
            raise
    return 0 if report.get("success") else 1


def main(argv=None):
    a = arguments(argv)
    # New output only: never overwrite an earlier failed or successful report.
    a.output.mkdir(parents=True, exist_ok=False)
    try:
        return run(a)
    except Exception as error:
        failed = a.output / "paired.json"
        # run() closes its device before publication; preflight errors open none.
        try:
            if not failed.exists():
                failed.write_text(
                    json.dumps(
                        {
                            "success": False,
                            "accepted": False,
                            "diagnostic_only": True,
                            "error": repr(error),
                            "phase": "preflight_or_publication",
                        },
                        indent=2,
                    )
                )
        except BaseException:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
