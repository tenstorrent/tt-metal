# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Compare tile-peel prefill with the pinned 512-peel public-call baseline.

Reduced GDN -> full-attention -> GDN diagnostic, not serving performance or a
context-capability test. Run under an external bounded timeout. No traces are
prepared/captured and this program never resets devices. Default slot capacity
is four; a single request uses slot three while populated peers must survive.
Use --batch 3 --uniform for the optional three-row uniform path.
"""

import argparse
import json
import subprocess
import tempfile
import time
import types
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.probe_bounded_prefill import (
    addresses,
    logit_metrics,
    read_states,
    tensor_metrics,
)
from models.autoports.qwen_qwen3_6_27b.tt.generator import QwenGenerator
from models.autoports.qwen_qwen3_6_27b.tt.model import Model
from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import RecurrentState

BASE_COMMIT = "d67807951e5411a4c1a79144da81b7fbc7852a43"
SOURCE_PATH = "models/autoports/qwen_qwen3_6_27b/tt/bounded_prefill.py"


def greedy_membership(reference, actual):
    """Accept any exact reference maximum, as the device greedy contract does."""
    chosen = actual.argmax(dim=-1, keepdim=True)
    return reference.gather(-1, chosen).squeeze(-1) == reference.max(dim=-1).values


CASES = {
    "start1": [0, 1, 545],
    "start31": [0, 31, 545],
    "start32": [0, 32, 545],
    "start33": [0, 33, 545],
    "start481": [0, 481, 1025],
    "start962": [0, 962, 1025],
    "start2015": [0, 2015, 2048],
    "start2016": [0, 2016, 2048],
    "scheduler481": [0, 481, 962, 1443, 1924],
}


def old_bounded_class(repo):
    source = subprocess.check_output(["git", "show", f"{BASE_COMMIT}:{SOURCE_PATH}"], cwd=repo, text=True)
    module = types.ModuleType("models.autoports.qwen_qwen3_6_27b.tt._probe_old_bounded_prefill")
    module.__package__ = "models.autoports.qwen_qwen3_6_27b.tt"
    module.__file__ = f"git:{BASE_COMMIT}:{SOURCE_PATH}"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module.BoundedPrefill


def cache_snapshot(generator):
    """Read all mesh ranks, retaining the physical page axis for ownership checks."""
    result = {}
    for index, state in zip(generator.model.layer_indices, generator.cache):
        if not isinstance(state, RecurrentState):
            result[str(index)] = {
                name: ttnn.to_torch(
                    getattr(state, name), mesh_composer=ttnn.ConcatMeshToTensor(generator.mesh, dim=1)
                ).clone()
                for name in ("key", "value")
            }
    return result


def public_call(generator, tokens, slots, start, end, *, all_logits=True):
    return generator.prefill_forward(
        tokens,
        page_table=generator.page_host[slots].clone(),
        kv_cache=generator.cache,
        prompt_lens=[end] * len(slots),
        start_pos=[start] * len(slots),
        empty_slots=slots,
        return_all_logits=all_logits,
        enable_trace=False,
    )


def cache_guards(before, after, states_before, states_after, page_table, slots, start, end):
    """Bit-exact ownership checks permit only the active trailing padded tile."""
    result = {"prefix_equal": True, "outside_write_footprint_equal": True, "peer_state_equal": True}
    details = {}
    active = set(slots)
    for layer, tensors in before.items():
        details[layer] = {}
        for name, old in tensors.items():
            new = after[layer][name]
            # Cache layout [physical_page, local_heads_across_mesh, 32, head_dim].
            changed_allowed = torch.zeros(old.shape[0], old.shape[2], dtype=torch.bool)
            for slot in slots:
                positions = torch.arange(start, ((end + 31) // 32) * 32)
                changed_allowed[page_table[slot, positions // 32].long(), positions % 32] = True
            unchanged = ~changed_allowed[:, None, :, None].expand_as(old)
            outside_equal = torch.equal(old[unchanged], new[unchanged])
            prefix_equal = True
            for slot in slots:
                positions = torch.arange(start)
                if positions.numel():
                    pages = page_table[slot, positions // 32].long()
                    prefix_equal &= torch.equal(old[pages, :, positions % 32, :], new[pages, :, positions % 32, :])
            details[layer][name] = {"prefix_equal": prefix_equal, "outside_write_footprint_equal": outside_equal}
            result["prefix_equal"] &= prefix_equal
            result["outside_write_footprint_equal"] &= outside_equal
    peers = [slot for slot in range(page_table.shape[0]) if slot not in active]
    peer_details = {}
    for layer, tensors in states_before.items():
        peer_details[layer] = {}
        for name, old in tensors.items():
            equal = torch.equal(old[peers], states_after[layer][name][peers])
            peer_details[layer][name] = equal
            result["peer_state_equal"] &= equal
    result.update(kv_details=details, peer_state_details=peer_details, populated_peer_slots=peers)
    result["passed"] = all(result[key] for key in ("prefix_equal", "outside_write_footprint_equal", "peer_state_equal"))
    return result


def run_side(generator, engines, tokens, slots, boundaries, directory, side):
    generator.prefill_single, generator.prefill_uniform = engines
    generator.reset()
    # Nonzero peer caches/state make cross-request preservation a real control.
    for peer in sorted(set(range(generator.batch_size)) - set(slots)):
        peer_tokens = torch.arange(37, dtype=torch.int32)[None] + 9000 + peer * 997
        public_call(generator, peer_tokens, [peer], 0, 37, all_logits=False)
    stable_addresses = addresses(generator)
    results = []
    for call, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        before_cache, before_states = cache_snapshot(generator), read_states(generator)
        begin = time.perf_counter()
        logits = public_call(generator, tokens, slots, start, end)
        # Public all-logit readback plus state/cache readbacks make this timing
        # completion-inclusive. It is deliberately not serving latency.
        after_states, after_cache = read_states(generator), cache_snapshot(generator)
        elapsed = time.perf_counter() - begin
        guards = cache_guards(
            before_cache, after_cache, before_states, after_states, generator.page_host, slots, start, end
        )
        entry = {
            "start": start,
            "end": end,
            "logical_length": end - start,
            "completion_inclusive_diagnostic_seconds": elapsed,
            "guards": guards,
            "stable_cache_addresses": addresses(generator) == stable_addresses,
        }
        reference_file = directory / f"call_{call}.pt"
        if side == "old":
            torch.save({"logits": logits, "states": after_states}, reference_file)
        else:
            reference = torch.load(reference_file, map_location="cpu", weights_only=True)
            metrics = logit_metrics(reference["logits"], logits)
            metrics["strict_lowest_index_top1_agreement"] = metrics["top1_agreement"]
            metrics["top1_agreement"] = float(greedy_membership(reference["logits"], logits).float().mean())
            metrics["top1_definition"] = "candidate argmax belongs to the exact reference maximum set; no epsilon"
            expected = reference["logits"].argmax(dim=-1)
            actual_ids = logits.argmax(dim=-1)
            misses = (expected != actual_ids).nonzero()
            metrics["top1_misses"] = [
                dict(
                    row=int(row),
                    token=int(token),
                    old_id=int(expected[row, token]),
                    new_id=int(actual_ids[row, token]),
                    old_at_old=float(reference["logits"][row, token, expected[row, token]]),
                    old_at_new=float(reference["logits"][row, token, actual_ids[row, token]]),
                    new_at_old=float(logits[row, token, expected[row, token]]),
                    new_at_new=float(logits[row, token, actual_ids[row, token]]),
                )
                for row, token in misses[:64].tolist()
            ]
            # The requested gate is PCC/top1/top5. Keep top100 diagnostic-only.
            metrics["passed"] = (
                metrics.get("finite", False)
                and metrics.get("pcc", 0) >= 0.995
                and metrics.get("top1_agreement", 0) >= 0.98
                and metrics.get("top5_agreement", 0) >= 0.98
            )
            state_metrics = {
                layer: {name: tensor_metrics(value, after_states[layer][name]) for name, value in tensors.items()}
                for layer, tensors in reference["states"].items()
            }
            entry.update(logits=metrics, gdn_states=state_metrics)
            del reference
        entry["passed"] = guards["passed"] and entry["stable_cache_addresses"]
        if side == "candidate":
            entry["passed"] &= entry["logits"]["passed"] and all(
                metric["passed"] for tensors in entry["gdn_states"].values() for metric in tensors.values()
            )
        results.append(entry)
        print(f"CONTINUATION_CALL side={side} start={start} end={end} passed={entry['passed']}", flush=True)
        del logits, before_cache, before_states, after_states, after_cache
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, choices=range(1, 33), default=4)
    parser.add_argument("--uniform", action="store_true", help="Exercise full-capacity uniform rows instead of slot3")
    parser.add_argument("--cases", default=",".join(CASES), help="Comma-separated case names")
    args = parser.parse_args()
    selected = args.cases.split(",")
    if not selected or any(case not in CASES for case in selected):
        parser.error(f"--cases must select from {list(CASES)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[4]
    old_class = old_bounded_class(repo)
    precision_file = Path(__file__).resolve().parents[1] / "doc/datatype_sweep/selected_precision_config.json"
    torch.set_num_threads(8)
    result = {
        "passed": False,
        "old_source_commit": BASE_COMMIT,
        "candidate_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "candidate_diff": subprocess.check_output(["git", "diff", "HEAD", "--", SOURCE_PATH], cwd=repo, text=True),
        "layers": [0, 3, 4],
        "batch_capacity": args.batch,
        "uniform": args.uniform,
        "context_allocation": 2048,
        "context_scope": "Reduced diagnostic allocation only; advertised serving context is unchanged",
        "trace_capture": False,
        "thresholds": {"pcc": 0.995, "top1_agreement": 0.98, "top5_agreement": 0.98},
        "timing_scope": "Public all-logit prefill plus cache/state readbacks; not serving performance",
        "precision_path": str(precision_file),
        "cases": [],
    }
    generator = mesh = None
    started = time.perf_counter()
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
        model = Model(mesh, layer_indices=[0, 3, 4], precision_config=precision_file)
        generator = QwenGenerator(
            model, batch_size=args.batch, max_seq_len=2048, trace_prefill=False, batched_prefill=args.uniform
        )
        candidate_engines = (generator.prefill_single, generator.prefill_uniform)
        old_single = old_class(model, generator.cache, generator.page_host)
        old_uniform = (
            old_class(model, generator.cache, generator.page_host, prefill_batch_size=args.batch)
            if args.uniform and args.batch > 1
            else old_single
        )
        slots = list(range(args.batch)) if args.uniform else [min(3, args.batch - 1)]
        result["active_slots"] = slots
        tokens = torch.arange(2048, dtype=torch.int32)[None] + 100 + torch.tensor(slots)[:, None] * 997
        for case in selected:
            boundaries = CASES[case]
            print(f"CONTINUATION_CASE name={case} boundaries={boundaries} slots={slots}", flush=True)
            # Disk spill bounds host RAM to one public call's reference/actual.
            with tempfile.TemporaryDirectory(prefix="tile_probe_", dir=args.output.parent) as temporary:
                directory = Path(temporary)
                old = run_side(generator, (old_single, old_uniform), tokens, slots, boundaries, directory, "old")
                actual = run_side(generator, candidate_engines, tokens, slots, boundaries, directory, "candidate")
            result["cases"].append(
                {
                    "name": case,
                    "boundaries": boundaries,
                    "old": old,
                    "candidate": actual,
                    "passed": all(row["passed"] for row in old + actual),
                }
            )
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        result["precision_runtime"] = model.precision_summary()
        result["cache_runtime"] = [
            {
                "layer": layer,
                "tensors": {
                    name: {"shape": list(getattr(state, name).shape), "dtype": str(getattr(state, name).dtype)}
                    for name in (("conv", "recurrent") if isinstance(state, RecurrentState) else ("key", "value"))
                },
            }
            for layer, state in zip(model.layer_indices, generator.cache)
        ]
        result["generator_captures"] = generator.counters["captures"]
        result["helper_counters"] = {
            "old_single": old_single.counters,
            "old_uniform": old_uniform.counters,
            "candidate_single": candidate_engines[0].counters,
            "candidate_uniform": candidate_engines[1].counters,
        }
        result["passed"] = all(case["passed"] for case in result["cases"]) and not result["generator_captures"]
        if generator.traces or result["generator_captures"]:
            raise AssertionError("Diagnostic unexpectedly captured traces")
    except BaseException as error:
        result.update(passed=False, exception=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        try:
            if generator is not None:
                generator.close()
        finally:
            if mesh is not None:
                ttnn.close_mesh_device(mesh)
    print(f"TILE_CONTINUATION_PROBE passed={result['passed']} output={args.output}", flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
