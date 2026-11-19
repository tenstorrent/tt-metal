# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Recover QSA boundary/state evidence without rerunning earlier attention.

All TT state starts empty. A test-local sentinel stops the unchanged runtime
forward_chunk immediately before selection, after every mutable-state write.
The unchanged advance updates position. No HF cache/intermediate enters TT.
This is reduced state evidence: the original full-prefill failure stays failed.
Recovery exports deliberately use a distinct origin/version from passing-prefix
exports and cannot be loaded by the existing acceptance export helper.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa import ReferenceCache, load_reference
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa_prefill import output_metrics
from models.autoports.qwen_qwen3_8_flash_next.tests.qsa_state_fixture import STATE_TENSORS, runtime_source_hashes

CAPACITY = 262144
CHECKPOINT = CAPACITY - 32
PREFIX = CAPACITY - 1
MUTABLE = ("k_cache", "v_cache", "raw_history", "compressed_keys", "position")
ORIGIN = "TT exact runtime state-only prefix from empty state; boundary recovery, not all-output acceptance"


def file_identity(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


class _StateUpdated(Exception):
    """Private control-flow sentinel, never a device or numerical exception."""


def state_only_chunk(model, hidden, state):
    calls = 0

    def stop(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise _StateUpdated

    with patch.object(model, "_selected_ids", stop):
        try:
            model.forward_chunk(hidden, state=state)
        except _StateUpdated:
            pass
        else:
            raise AssertionError("Runtime no longer reaches the state-update boundary")
    assert calls == 1
    model.advance(state, hidden.shape[-2])


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return file_identity(path)


def bind_inputs(args):
    root = Path(__file__).resolve().parents[1]
    record = json.loads(args.full_provenance.read_text())
    full = json.loads(args.full_report.read_text())
    assert record["status"] == "finished" and record["exit_code"] == 1
    for flag, expected in (
        ("--output", args.full_report),
        ("--device-output-cache", args.full_output),
        ("--fixture-cache", args.source_cache),
    ):
        recorded = Path(record["argv"][record["argv"].index(flag) + 1])
        assert (Path(record["cwd"]) / recorded).resolve() == expected.resolve(), flag
    log = Path(record["cwd"]) / record["log"]
    assert file_identity(log)["sha256"] == record["log_sha256"]
    inventory = runtime_source_hashes()
    assert {
        key.removeprefix("tt/"): value for key, value in record["source_sha256"].items() if key.startswith("tt/")
    } == inventory
    # These original harness dependencies define saved output and state semantics.
    for relative in ("tests/probe_qsa_prefill.py", "tests/probe_qsa.py"):
        assert file_identity(root / relative)["sha256"] == record["source_sha256"][relative]
    assert full["prefill"]["rows_compared"] == CAPACITY and not full["prefill"]["passed"]
    assert full["error"] == "AssertionError: Full prefill all-row gate failed before boundary continuation"
    assert full["device_output_cache"] == str(args.full_output)
    source = torch.load(args.source_cache, weights_only=True, mmap=True, map_location="cpu")
    saved = torch.load(args.full_output, weights_only=True, mmap=True, map_location="cpu")
    key = source["key"]
    assert key["length"] == key["capacity"] == CAPACITY and key["decode_steps"] == 0 and key["seed"] == 1979
    assert key["snapshot"] == str(args.snapshot.resolve())
    assert saved["length"] == CAPACITY
    cache = saved["reference_fixture"]
    assert cache.get("source_key", cache["key"]) == key
    assert full["fixture_cache"] == cache
    for tensor in (source["hidden"], source["expected"], saved["actual"]):
        assert tuple(tensor.shape) == (1, CAPACITY, 2560) and tensor.dtype == torch.bfloat16
    permutation = torch.randperm(8192, generator=torch.Generator().manual_seed(1980)).reshape(1, -1).int()
    if permutation[0, 0] == 0:
        permutation = permutation.roll(1, dims=1)
    assert torch.equal(saved["page_table"], permutation)
    bindings = {
        name: file_identity(path)
        for name, path in {
            "full_report": args.full_report,
            "full_provenance": args.full_provenance,
            "full_log": log,
            "full_output": args.full_output,
            "source_fixture": args.source_cache,
            "recovery_harness": Path(__file__),
        }.items()
    }
    return source, saved, full, inventory, bindings


@torch.inference_mode()
def reference_final_inputs(args, model, rope, source, provenance, bindings):
    """Independent literal HF prefixes stay on CPU and are never uploaded."""
    key = {
        "prefix": PREFIX,
        "source_fixture": bindings["source_fixture"],
        "provenance": provenance,
        "harness_sha256": bindings["recovery_harness"]["sha256"],
    }
    if args.reference_cache.exists():
        payload = torch.load(args.reference_cache, weights_only=True, map_location="cpu")
        assert payload["key"] == key, "Choose a fresh reference-cache path after source changes"
        return payload
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    keys = torch.empty(1, 2, PREFIX, 256, dtype=torch.bfloat16)
    values = torch.empty_like(keys)
    raw = torch.empty(1, PREFIX, 128, dtype=torch.bfloat16)
    for start in range(0, PREFIX, 32):
        end = min(start + 32, PREFIX)
        hidden = source["hidden"][:, start:end]
        cos, sin = rope(hidden, torch.arange(start, end).reshape(1, -1))
        k = model.k_norm(model.k_proj(hidden).reshape(1, end - start, 2, 256)).transpose(1, 2)
        keys[:, :, start:end] = apply_rotary_pos_emb(k, cos=cos, sin=sin)
        values[:, :, start:end] = model.v_proj(hidden).reshape(1, end - start, 2, 256).transpose(1, 2)
        raw[:, start:end] = model.indexer.index_qk_proj(hidden)[..., 512:]
    inputs = [source["hidden"][:, PREFIX:].clone(), -source["hidden"][:, PREFIX:]]
    assert not torch.equal(*inputs)
    embeddings = rope(inputs[0], torch.arange(CAPACITY).reshape(1, -1))
    expected = []
    for hidden in inputs:
        output, _ = model(
            hidden,
            embeddings,
            torch.zeros(1, 1, 1, CAPACITY, dtype=torch.bfloat16),
            ReferenceCache(raw=raw, k=keys, v=values),
        )
        expected.append(output)
    original_control = output_metrics(source["expected"][:, PREFIX:], expected[0])
    assert original_control["minimum_row_pcc"] >= 0.99999
    payload = {"key": key, "inputs": inputs, "expected": expected, "original_final_reference_control": original_control}
    atomic_save(payload, args.reference_cache)
    return payload


def run_device(args, config, weights, source, saved, reference, report):
    assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1", "trace allocation tracking must be enabled"
    assert (
        os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "0"
    ), "program cache tracking must be enabled"
    from ttnn.tools import trace_allocation_tracker

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit
    from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import WeightLoader
    from models.autoports.qwen_qwen3_8_flash_next.tt.qsa import QSA
    from models.autoports.qwen_qwen3_8_flash_next.tt.qsa_chunked import ChunkedQSA

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    try:
        loader = WeightLoader(weights, mesh)
        model = ChunkedQSA(QSA(loader, config))
        permutation = saved["page_table"]
        hidden = loader.tensor(source["hidden"].unsqueeze(0))

        def host(value, target=None):
            return ttnn.from_torch(
                value.contiguous(),
                dtype=target.dtype if target is not None else ttnn.bfloat16,
                layout=target.layout if target is not None else ttnn.TILE_LAYOUT,
            )

        def snapshot(state):
            return {name: ttnn.to_torch(getattr(state, name)).clone() for name in MUTABLE}

        def restore(state, values):
            for name, value in values.items():
                target = getattr(state, name)
                ttnn.copy_host_to_device_tensor(host(value, target), target)

        # Same capacity/page permutation and actual runtime writes; compare all bytes.
        ordinary = model.attention.make_state(capacity=CAPACITY, page_table=permutation)
        reduced = model.attention.make_state(capacity=CAPACITY, page_table=permutation)
        start = 0
        report["state_stop_control"] = []
        for count in (32, 31, 1):
            x = hidden[:, :, start : start + count]
            with runtime_audit():
                output = model.forward_chunk(x, state=ordinary)
                model.advance(ordinary, count)
                state_only_chunk(model, x, reduced)
            ttnn.synchronize_device(mesh)
            lhs, rhs = snapshot(ordinary), snapshot(reduced)
            equal = {name: torch.equal(lhs[name], rhs[name]) for name in MUTABLE}
            assert all(equal.values()), equal
            report["state_stop_control"].append({"start": start, "count": count, "all_mutable_state_equal": equal})
            ttnn.deallocate(output)
            start += count
        for control in (ordinary, reduced):
            for name in STATE_TENSORS:
                ttnn.deallocate(getattr(control, name))
        del ordinary, reduced, lhs, rhs
        state = model.attention.make_state(capacity=CAPACITY, page_table=permutation)
        began = time.monotonic()
        for start in range(0, CHECKPOINT, 32):
            with runtime_audit():
                state_only_chunk(model, hidden[:, :, start : start + 32], state)
            if (start + 32) % 16384 == 0:
                ttnn.synchronize_device(mesh)
                print(
                    json.dumps({"state_only_prefix_rows": start + 32, "seconds": time.monotonic() - began}), flush=True
                )
        ttnn.synchronize_device(mesh)
        checkpoint = snapshot(state)
        assert int(checkpoint["position"].reshape(-1)[0]) == CHECKPOINT
        report["prefix_state_build"] = {
            "rows": CHECKPOINT,
            "seconds": time.monotonic() - began,
            "origin": ORIGIN,
            "skipped": "selection, SDPA, gate/output and output assembly",
        }
        with runtime_audit():
            output32 = model.prefill_forward(hidden[:, :, CHECKPOINT:], state=state)
        actual32 = ttnn.to_torch(output32).reshape(1, 32, 2560)
        report["final32_saved_TT_bit_exact"] = torch.equal(actual32, saved["actual"][:, CHECKPOINT:])
        report["final32_HF"] = output_metrics(source["expected"][:, CHECKPOINT:], actual32)
        assert report["final32_saved_TT_bit_exact"] and report["final32_HF"]["passed"]
        ttnn.deallocate(output32)
        restore(state, checkpoint)
        with runtime_audit():
            output31 = model.prefill_forward(hidden[:, :, CHECKPOINT:PREFIX], state=state)
        actual31 = ttnn.to_torch(output31).reshape(1, 31, 2560)
        report["nonaligned_tail_HF"] = output_metrics(source["expected"][:, CHECKPOINT:PREFIX], actual31)
        assert report["nonaligned_tail_HF"]["passed"]
        ttnn.deallocate(output31)
        prefix = snapshot(state)
        assert int(prefix["position"].reshape(-1)[0]) == PREFIX
        export_tensors = {name: ttnn.to_torch(getattr(state, name)).clone() for name in STATE_TENSORS}
        decode_input = loader.tensor(reference["inputs"][0].unsqueeze(0))
        decode_host = [host(value.unsqueeze(0)) for value in reference["inputs"]]
        # Warm the changed-input copy signature before capture.
        for value in decode_host:
            ttnn.copy_host_to_device_tensor(value, decode_input)
        ttnn.copy_host_to_device_tensor(decode_host[0], decode_input)
        with runtime_audit():
            warm = model.forward_chunk(decode_input, state=state)
            model.advance(state, 1)
        ttnn.synchronize_device(mesh)
        ttnn.deallocate(warm)
        restore(state, prefix)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        with runtime_audit():
            decoded = model.forward_chunk(decode_input, state=state)
            model.advance(state, 1)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        trace_allocation_tracker.acknowledge_corruptible(decoded)
        first = {}
        report["decode"] = []
        for branch in (0, 1, 0, 1):
            restore(state, prefix)
            ttnn.copy_host_to_device_tensor(decode_host[branch], decode_input)
            with runtime_audit():
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            actual = ttnn.to_torch(decoded).reshape(1, 1, 2560)
            after = snapshot(state)
            measured = output_metrics(reference["expected"][branch], actual)
            assert measured["passed"] and int(after["position"].reshape(-1)[0]) == CAPACITY
            if branch in first:
                assert torch.equal(actual, first[branch][0])
                assert all(torch.equal(after[name], first[branch][1][name]) for name in MUTABLE)
            else:
                first[branch] = (actual.clone(), after)
            report["decode"].append({"branch": branch, **measured, "position": CAPACITY})
        report.update(
            capture_count=1,
            replay_count=4,
            repeated_output_and_state_exact=True,
            runtime_fallback_audit="clean",
            allocation_tracking=True,
            boundary_state_passed=True,
        )
        payload = {
            "version": 2,
            "origin": ORIGIN,
            "prefix": PREFIX,
            "capacity": CAPACITY,
            "page_size": 32,
            "source_fixture_key": source["key"],
            "source_fixture_path": str(args.source_cache),
            "runtime_source_sha256": report["runtime_source_sha256"],
            "bindings": report["bindings"],
            "strict_full_prefill_passed": False,
            "original_full_prefill": report["original_full_prefill"],
            "boundary_evidence": {
                key: report[key]
                for key in (
                    "state_stop_control",
                    "final32_saved_TT_bit_exact",
                    "final32_HF",
                    "nonaligned_tail_HF",
                    "decode",
                    "repeated_output_and_state_exact",
                )
            },
            "tensors": export_tensors,
        }
        report["recovery_state_export"] = atomic_save(payload, args.prefix_state_cache)
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--full-report", type=Path, required=True)
    parser.add_argument("--full-provenance", type=Path, required=True)
    parser.add_argument("--full-output", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--prefix-state-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    source, saved, full, inventory, bindings = bind_inputs(args)
    config, model, rope, weights, provenance = load_reference(args.snapshot)
    for name in ("reference_sha256", "config_sha256", "transformers"):
        assert source["key"][name] == provenance[name]
    report = {
        "scope": __doc__,
        "bindings": bindings,
        "runtime_source_sha256": inventory,
        "provenance": provenance,
        "original_full_prefill": full["prefill"],
        "strict_full_prefill_passed": False,
        "stage_acceptance": "not_evaluated",
        "boundary_state_passed": False,
        "device_executed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        reference = reference_final_inputs(args, model, rope, source, provenance, bindings)
        report["reference_cache"] = file_identity(args.reference_cache)
        report["original_final_reference_control"] = reference["original_final_reference_control"]
        if not args.cpu_only:
            report["device_executed"] = True
            run_device(args, config, weights, source, saved, reference, report)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {str(exc).split('backtrace:')[0]}"
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
