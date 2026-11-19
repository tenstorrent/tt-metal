# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Separate HF accuracy evidence; NEVER supersedes the frozen B1/B2 diagnostic.

One correlated AIME24 sequence, not 32 independent prompts. Row r starts at
prompt + HF generated[:r], then causally consumes the remaining HF tokens.
Every prediction comes from the actual traced device greedy sampler. EOS does
not truncate teacher forcing. CPU logits/state reads are diagnostic, not perf.
--diagnostic-layers 0 1 3 retains the same32-slot history/migration control but
does not score HF accuracy and is explicitly ineligible for stage acceptance.

Device owner only, after restoring the accepted mesh descriptor:
  TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  timeout --signal=INT --kill-after=60 7200 python_env/bin/python -m \
  models.autoports.qwen_qwen3_8_flash_next.tests.probe_full_model_batch_hf \
  --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/batch32_hf_v1.json
"""

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import torch


def digest_tensor(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def load_reference(path, batch):
    ref = torch.load(path, weights_only=True, map_location="cpu")
    assert ref["hf_model_id"] == "Qwen/Qwen3.8-Flash-Next"
    assert ref["k"] == 100 and len(ref["entries"]) == 1
    entry = ref["entries"][0]
    prompt = entry["prompt_tokens"].flatten().long()
    generated = entry["generated_tokens"].flatten().long()
    topk = entry["topk_tokens"].long()
    assert generated.shape == (100,) and topk.shape == (100, 100)
    # HF cached generation and its full-forward scoring can differ numerically.
    # Force the stored generated history exactly, as TokenAccuracy does; never
    # replace it with topk[:, 0] (this reference differs at index53).
    assert 2 <= batch <= 32 and len(prompt) == entry["tf_prompt_len"]
    full = torch.cat((prompt, generated))
    prompts = [full[: len(prompt) + row].clone() for row in range(batch)]
    return prompt, generated, topk, prompts


def accuracy(records):
    count = len(records)
    assert count > 0
    hits = {k: sum(row[f"top{k}"] for row in records) for k in (1, 5, 100)}
    result = {"total": count, **{f"matches_top{k}": value for k, value in hits.items()}}
    result.update({f"top{k}": value / count for k, value in hits.items()})
    result["passed"] = result["top5"] >= 0.98 and result["top100"] == 1
    return result


def prediction_record(prediction, hf_topk, request, target, phase):
    prediction = int(prediction)
    return {
        "request": request,
        "reference_index": target,
        "phase": phase,
        "tt_device_greedy": prediction,
        **{f"top{k}": prediction in hf_topk[:k].tolist() for k in (1, 5, 100)},
    }


def greedy_params_for_order(params_type, request_seeds, order):
    """Describe every request explicitly; a scalar common seed targets lane0 only."""
    count = len(order)
    assert sorted(order) == list(range(len(request_seeds)))
    return params_type(
        temperature=[0.0] * count,
        top_k=[1] * count,
        top_p=[1.0] * count,
        seed=[request_seeds[request] for request in order],
    )


def state_field_differences(expected, actual):
    """Localize final-state failures without dropping the combined digest gate."""
    return {
        request: {
            field: {"expected": expected.get(request, {}).get(field), "actual": actual.get(request, {}).get(field)}
            for field in sorted(set(expected.get(request, {})) | set(actual.get(request, {})))
            if expected.get(request, {}).get(field) != actual.get(request, {}).get(field)
        }
        for request in sorted(set(expected) | set(actual))
        if expected.get(request) != actual.get(request)
    }


class FactoredTraceEvidence:
    """Host-only trace/metadata observations; never reads model tensors."""

    def __init__(self, generator, memory_view, program_count, synchronize, result):
        self.generator = generator
        self.owner = generator.factored_decode
        assert self.owner is not None
        self.memory_view, self.program_count, self.synchronize = memory_view, program_count, synchronize
        self.result = result
        self.ids = None
        self.programs = None
        self.result.update(
            capture_windows=[],
            request_metadata_uploads=0,
            unchanged_membership_replays=0,
            unchanged_plan_identity_reuses=0,
            native_id_stability_checks=0,
            program_cache_stability_checks=0,
            membership_events=[],
        )
        original_capture = self.owner.capture

        def observed_capture():
            self.synchronize()
            before = self.memory_view()
            programs_before = self.program_count()
            metadata_before = self.owner.metadata_uploads
            ids_before = self.native_ids()
            window = {
                "memory_before": before,
                "program_cache_entries_before": programs_before,
                "metadata_uploads_before": metadata_before,
                "native_ids_before": ids_before,
            }
            self.result["capture_windows"].append(window)
            try:
                original_capture()
            finally:
                self.synchronize()
                after = self.memory_view()
                ids_after = self.native_ids()
                window.update(
                    memory_after=after,
                    retained_capture_bytes_per_rank=self.allocated(after) - self.allocated(before),
                    program_cache_entries_after=self.program_count(),
                    metadata_uploads_after=self.owner.metadata_uploads,
                    native_ids_after=ids_after,
                    native_capture_count=len(ids_after) - len(ids_before),
                    role_keys=self.role_keys(),
                )
            assert (
                window["program_cache_entries_after"] == programs_before
            ), "New binary compiled during factored capture"

        self.owner.capture = observed_capture

    @staticmethod
    def allocated(view):
        return view["num_banks"] * view["total_bytes_allocated_per_bank"]

    def native_ids(self):
        return {repr(key): str(trace) for key, trace in self.generator.traces.items()}

    def role_keys(self):
        result = {}
        for key in self.generator.traces:
            result.setdefault(key[0], []).append(list(key))
        return result

    def prepared(self):
        self.ids = self.native_ids()
        assert len(set(self.ids.values())) == len(self.ids), "Native IDs must be owned exactly once"
        layers, slots, buckets = len(self.generator.model.layers), self.owner.slot_count, len(self.owner.buckets)
        expected = 1 + layers * slots + layers * buckets + slots
        assert len(self.ids) == expected
        assert len(self.result["capture_windows"]) == 1, "Preparation should capture the bounded set once"
        self.programs = self.program_count()
        window = self.result["capture_windows"][0]
        self.result.update(
            native_capture_count=len(self.ids),
            native_ids_after_prepare=self.ids,
            role_capture_counts={role: len(keys) for role, keys in self.role_keys().items()},
            role_keys=self.role_keys(),
            metadata_warmup_uploads=window["metadata_uploads_before"],
            metadata_capture_uploads=window["metadata_uploads_after"] - window["metadata_uploads_before"],
            metadata_uploads_after_prepare=self.owner.metadata_uploads,
            program_cache_entries_after_prepare=self.programs,
            scope="Allocator delta around capture only, including bank alignment; metadata observations contain no tensor readbacks",
        )

    def check_programs(self):
        assert self.program_count() == self.programs, "New program compiled after prepare"
        self.result["program_cache_stability_checks"] += 1

    def auxiliary_ids(self):
        return {
            name: {repr(key): str(trace) for key, trace in getattr(self.generator, name).items()}
            for name in ("sampling_traces", "prefill_traces")
        }

    def before(self):
        return (
            self.owner.active_slots,
            self.owner.metadata_uploads,
            self.owner._plan,
            self.owner._plan_slots,
            self.generator.counters["captures"],
            self.auxiliary_ids(),
        )

    def after(self, before, slots, *, repeat, step, migration=False):
        old_slots, uploads, plan, plan_slots, captures, auxiliary_ids = before
        slots = tuple(sorted(slots))
        assert self.native_ids() == self.ids, "Native model IDs changed across shrink/migration/replay"
        assert self.auxiliary_ids() == auxiliary_ids, "Sampling/prefill native IDs changed during decode"
        self.result["auxiliary_native_ids_last"] = auxiliary_ids
        self.result["native_id_stability_checks"] += 1
        assert self.generator.counters["captures"] == captures, "Decode captured after bounded preparation/prefill"
        self.check_programs()
        delta = self.owner.metadata_uploads - uploads
        expected = int(bool(slots) and slots != old_slots)
        assert delta == expected, (slots, old_slots, delta, expected)
        self.result["request_metadata_uploads"] += delta
        if slots and slots == old_slots:
            self.result["unchanged_membership_replays"] += 1
        if slots and plan_slots == slots:
            assert self.owner._plan is plan, "Unchanged membership rebuilt its dispatch plan"
            self.result["unchanged_plan_identity_reuses"] += 1
        if not slots:
            assert self.owner._plan is plan and self.owner.active_slots == old_slots
        if delta or migration or not slots:
            self.result["membership_events"].append(
                {
                    "repeat": repeat,
                    "step": step,
                    "migration": migration,
                    "active_slots": list(slots),
                    "metadata_upload_delta": delta,
                    "native_ids_unchanged": True,
                }
            )
        self.result["metadata_uploads_current"] = self.owner.metadata_uploads


def run_resident_feedback(generator, pool, evidence, read_feedback, read_positions):
    """Four real sampled feedback steps; small diagnostic reads, no host forcing."""
    slots = tuple(range(generator.max_batch_size))
    odd = tuple(slot for slot in slots if slot % 2)
    assert odd
    schedule = (slots, slots, odd, odd)
    tokens_before = list(read_feedback())
    positions_before = read_positions()
    result = {
        "passed": False,
        "steps": [],
        "scope": "Resident common-sampled feedback; no HF accuracy/performance claim",
        "position_read_scope": "Every QSA layer/all fixed slots on rank0; scalar positions only",
    }
    for step, active in enumerate(schedule):
        before = dict(generator.counters)
        trace_before = evidence.before()
        host_positions = [cache.position for cache in generator.caches]
        # These stale sentinels are deliberately not request tokens/positions.
        # Only the negative markers determine which fixed slots are inactive.
        predicted = generator.decode_forward(
            torch.full((len(slots),), -12345, dtype=torch.long),
            [0 if slot in active else -1 for slot in slots],
            slots=slots,
            kv_cache=pool,
            page_table=None,
            sample_on_device=True,
            reload_inputs=False,
            reload_page_table=False,
        )
        evidence.after(trace_before, active, repeat="resident_feedback", step=step)
        tokens_after = list(read_feedback())
        positions_after = read_positions()
        delta = {key: generator.counters[key] - value for key, value in before.items()}
        for key in ("token_refreshes", "position_refreshes", "page_table_refreshes", "captures", "releases"):
            assert delta[key] == 0, (step, key, delta[key])
        assert delta["model_replays"] == 1 and delta["sampling_replays"] == 1
        assert positions_before.keys() == positions_after.keys()
        for key, value in positions_before.items():
            slot = key[0]
            assert value == host_positions[slot], (step, key, "initial device position disagrees")
            assert positions_after[key] == generator.caches[slot].position
            assert positions_after[key] == value + int(slot in active), (step, key, value, positions_after[key])
        for slot in slots:
            assert generator.caches[slot].position == host_positions[slot] + int(slot in active)
            if slot in active:
                assert 0 <= tokens_after[slot] < generator.model.config.vocab_size
                assert int(predicted[slot]) == tokens_after[slot]
            else:
                assert int(predicted[slot]) == -1
                assert tokens_after[slot] == tokens_before[slot], (step, slot, "inactive feedback changed")
        result["steps"].append(
            {
                "step": step,
                "active_slots": list(active),
                "counter_delta": delta,
                "resident_input_tokens": tokens_before,
                "sampled_output_tokens": tokens_after,
                "host_positions_before": host_positions,
                "host_positions_after": [cache.position for cache in generator.caches],
                "device_positions_coherent": True,
                "inactive_feedback_unchanged": True,
            }
        )
        tokens_before, positions_before = tokens_after, positions_after
    result["passed"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--capacity", type=int, default=512)
    parser.add_argument(
        "--diagnostic-layers", type=int, nargs="+", help="CPU-planned real0/1/3 state repro; never HF acceptance"
    )
    args = parser.parse_args()
    if args.diagnostic_layers is not None and args.diagnostic_layers != [0, 1, 3]:
        parser.error("The reduced diagnostic must use real layers 0 1 3")
    diagnostic = args.diagnostic_layers is not None
    if args.output.exists():
        parser.error("Use a new output path; prior evidence is immutable")
    model_dir = Path(__file__).resolve().parents[1]
    refpath = model_dir / "readiness_aime24_chat.refpt"
    prompt, generated, topk, prompts = load_reference(refpath, args.batch_size)
    if args.capacity < len(prompt) + len(generated):
        parser.error("Capacity must cover the complete reference history")
    assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "0"
    assert os.environ.get("TT_MESH_GRAPH_DESC_PATH"), "Restore the accepted mesh descriptor"
    for flag in ("TT_METAL_DEVICE_PROFILER", "TT_METAL_TRACE_PROFILER", "TT_METAL_STREAMING_PROFILER"):
        assert os.environ.get(flag) in (None, "", "0"), "No all-layer profiler"
    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tests.run_full_model_accuracy import memory
    from models.autoports.qwen_qwen3_8_flash_next.tests.shared_pool_layout import shared_pool_layout
    from models.autoports.qwen_qwen3_8_flash_next.tt.generator import DecodeSubmission, build_generator
    from models.common.sampling.generator import SamplingParams

    torch.set_num_threads(8)
    result = {
        "passed": False,
        "scope": (
            "reduced_real_0_1_3_matched_history_state_diagnostic"
            if diagnostic
            else "all48_HF_batched_accuracy_and_matched_history_controls"
        ),
        "acceptance_eligible": not diagnostic,
        "hf_accuracy_evaluated": not diagnostic,
        "layer_indices": args.diagnostic_layers or list(range(48)),
        "reference": str(refpath),
        "reference_sha256": hashlib.sha256(refpath.read_bytes()).hexdigest(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "batch_size": args.batch_size,
        "capacity": args.capacity,
        "prompt_length": len(prompt),
        "source_prompt_count": 1,
        "correlated_reference": True,
        "reference_generated_vs_scored_top1_misses": (generated != topk[:, 0]).nonzero().flatten().tolist(),
        "metric": (
            "Exact TT repeat/migration state and outputs only; HF history supplies inputs, not an accuracy target"
            if diagnostic
            else "TT traced greedy token belongs to HF reference top-k; not reverse membership"
        ),
        "thresholds": {"top5_min": 0.98, "top100": 1.0},
        "diagnostic_only_readbacks": True,
        "performance_claim": False,
        "existing_cross_geometry_gate": {
            "unchanged": True,
            "status": "not_adjudicated_by_this_test",
            "artifact": "doc/full_model/shared_pool_generator_v5.json",
            "sha256": hashlib.sha256(
                (model_dir / "doc/full_model/shared_pool_generator_v5.json").read_bytes()
            ).hexdigest(),
        },
        "source_hashes": {
            str(p.relative_to(model_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((model_dir / "tt").rglob("*"))
            if p.is_file() and p.suffix in (".py", ".cpp", ".h", ".hpp", ".cc", ".cxx")
        },
        "common_sampling_generator_sha256": hashlib.sha256(
            (model_dir.parents[1] / "common/sampling/generator.py").read_bytes()
        ).hexdigest(),
        "runs": [],
    }
    mesh = generator = None
    started = time.monotonic()
    phase = "setup"

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    def mark(value, **fields):
        nonlocal phase
        phase = value
        print(json.dumps({"phase": value, "seconds": time.monotonic() - started, **fields}), flush=True)

    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0, l1_small_size=32768)
        mark("LOAD")
        generator = build_generator(
            model_dir,
            mesh,
            max_seq_len=args.capacity,
            max_batch_size=args.batch_size,
            defer_kv_cache=True,
            layer_indices=args.diagnostic_layers,
        )
        assert list(generator.model.layer_indices) == result["layer_indices"]
        attention_count = sum(layer.attention is not None for layer in generator.caches[0].layers)
        assert attention_count == (1 if diagnostic else 12)
        result["memory_loaded"] = memory(mesh)
        specs, tables = shared_pool_layout(args.batch_size, args.capacity, attention_count)
        pool = generator.allocate_kv_cache_per_layer(specs)
        assert len(pool.pairs) == attention_count
        for cache in [*generator.caches, generator.slot_mover.scratch]:
            assert cache.shared_kv and cache.kv_bound
            attention = [layer.attention for layer in cache.layers if layer.attention is not None]
            assert len(attention) == attention_count
            for index, state in enumerate(attention):
                pair = pool.pair_for_attention_layer(index)
                assert state.k_cache is pair[0] and state.v_cache is pair[1]
                assert all(t.dtype == ttnn.bfloat16 and tuple(t.shape) == specs[index][0] for t in pair)
            assert all(
                layer.recurrent.dtype == ttnn.float32
                for layer in cache.layers
                if getattr(layer, "recurrent", None) is not None
            )
        addresses = [
            part.buffer_address()
            for pair in pool.pairs.values()
            for tensor in pair
            for part in ttnn.get_device_tensors(tensor)
        ]
        result["shared_pool"] = {
            "pairs": len(pool.pairs),
            "bytes_per_rank": pool.bytes_per_rank,
            "specs": [list(s[0]) for s in specs],
            "addresses": addresses,
            "page_tables": [table.tolist() for table in tables],
            "cache_spec": generator.cache_spec(),
        }
        result["factored_trace_evidence"] = {}
        evidence = FactoredTraceEvidence(
            generator,
            lambda: memory(mesh),
            mesh.num_program_cache_entries,
            lambda: ttnn.synchronize_device(mesh),
            result["factored_trace_evidence"],
        )
        mark("PREPARE")
        generator.warmup_model_prefill(kv_cache=pool)
        generator.warmup_model_decode(kv_cache=pool)
        assert pool.frozen
        evidence.prepared()
        result["memory_prepared"] = memory(mesh)
        result["program_cache_entries_after_prepare"] = mesh.num_program_cache_entries()
        result["setup_seconds"] = time.monotonic() - started
        request_seeds = [0] * args.batch_size
        result["sampling_request_seeds"] = request_seeds
        result[
            "sampling_seed_contract"
        ] = "Explicit per-request seed lists in current submission order; no scalar broadcast"
        slots = tuple(range(args.batch_size))
        baseline = {}
        baseline_states = None
        baseline_fields = None

        def inputs(order):
            selected = [table[list(order)] for table in tables]
            return {"kv_cache": pool, "page_table": selected[0], "page_tables_per_layer": selected}

        def logit_hashes(active_rows):
            submission = DecodeSubmission(generator.sample_logits, slots, tuple(active_rows), False)
            values = generator.process_decode_output_host(generator.read_decode_output(submission))[:, 0]
            assert bool(torch.isfinite(values).all()), "Nonfinite diagnostic logits"
            return {row: digest_tensor(values[row]) for row in active_rows}

        def state_hashes(order, selected_requests=None):
            # Read a shared tensor once per rank, then select logical owned KV
            # tokens; padding/future bytes are not request-owned history.
            chosen = set(order if selected_requests is None else selected_requests)
            hashes = {request: hashlib.sha256() for request in chosen}
            field_hashes = {str(request): {} for request in chosen}

            def add(request, name, tensor):
                digest = digest_tensor(tensor)
                hashes[request].update(name.encode())
                hashes[request].update(digest.encode())
                assert name not in field_hashes[str(request)], (request, name)
                field_hashes[str(request)][name] = digest

            for slot, request in enumerate(order):
                if request not in chosen:
                    continue
                cache = generator.caches[slot]
                hashes[request].update(str(cache.position).encode())
                field_hashes[str(request)]["host_position"] = hashlib.sha256(str(cache.position).encode()).hexdigest()
                for layer_index, layer in zip(generator.model.layer_indices, cache.layers, strict=True):
                    objects = [("layer", layer)]
                    if layer.attention is not None:
                        objects.append(("attention", layer.attention))
                    for owner, obj in objects:
                        for field in (
                            "conv",
                            "recurrent",
                            "ple_conv",
                            "token_history",
                            "position",
                            "page_table",
                            "raw_history",
                            "compressed_keys",
                        ):
                            tensor = getattr(obj, field, None)
                            if tensor is not None:
                                for rank, part in enumerate(ttnn.get_device_tensors(tensor)):
                                    host = ttnn.to_torch(part)
                                    if owner == "attention" and field == "position":
                                        assert host.flatten().tolist() == [cache.position], (
                                            request,
                                            layer_index,
                                            rank,
                                            cache.position,
                                            host.tolist(),
                                        )
                                    add(request, f"{layer_index}.{owner}.{field}.{rank}", host)
            for index in range(attention_count):
                for field, tensor in zip(("K", "V"), pool.pair_for_attention_layer(index), strict=True):
                    for rank, part in enumerate(ttnn.get_device_tensors(tensor)):
                        storage = ttnn.to_torch(part)
                        for slot, request in enumerate(order):
                            if request not in chosen:
                                continue
                            pos = generator.caches[slot].position
                            ids = tables[index][request, : (pos + 31) // 32].long()
                            logical = storage[ids].permute(1, 0, 2, 3).reshape(1, -1, 256)[:, :pos]
                            add(request, f"pool{index}.{field}.{rank}", logical)
            for field, tensor in (("feedback", generator.decode_tokens), ("seed", generator.seed_state.state)):
                for rank, part in enumerate(ttnn.get_device_tensors(tensor)):
                    values = ttnn.to_torch(part)
                    values = values.flatten() if field == "feedback" else values
                    for slot, request in enumerate(order):
                        if request in chosen:
                            add(request, f"{field}.{rank}", values[slot : slot + 1])
            return {str(request): value.hexdigest() for request, value in hashes.items()}, field_hashes

        for repeat in range(2):
            order = list(slots)
            generator.reset()
            greedy = greedy_params_for_order(SamplingParams, request_seeds, order)
            generator.set_sampling_params(greedy)
            run = {
                "name": "baseline" if repeat == 0 else "repeat_with_slot_migration",
                "migration_step": None if repeat == 0 else 32,
                "migration_destination_to_source": None if repeat == 0 else list(reversed(slots)),
                "records": [],
                "exact_same_history_logits": True,
                "exact_same_history_tokens": True,
            }
            result["runs"].append(run)
            packed = torch.nn.utils.rnn.pad_sequence(prompts, batch_first=True)
            mark("PREFILL", repeat=repeat)
            predicted = generator.prefill_forward(
                packed,
                prompt_lens=[len(p) for p in prompts],
                slots=slots,
                sampling_params=greedy,
                return_device=True,
                **inputs(order),
            ).flatten()
            run["sampling_seeds_after_prefill"] = generator.request_seeds[: args.batch_size]
            assert run["sampling_seeds_after_prefill"] == request_seeds
            evidence.check_programs()
            hashes = logit_hashes(slots)

            def record(row, request, target, kind, token, value_hash):
                item = (
                    {"request": request, "reference_index": target, "phase": kind, "tt_device_greedy": int(token)}
                    if diagnostic
                    else prediction_record(token, topk[target], request, target, kind)
                )
                item["logits_sha256"] = value_hash
                key = (request, target)
                if repeat == 0:
                    baseline[key] = (int(token), value_hash)
                else:
                    expected_token, expected_hash = baseline[key]
                    if int(token) != expected_token:
                        run["exact_same_history_tokens"] = False
                    if value_hash != expected_hash:
                        run["exact_same_history_logits"] = False
                run["records"].append(item)

            for row, request in enumerate(order):
                record(row, request, request, "prefill", predicted[row], hashes[row])
            dormant_before = None
            dormant_request = args.batch_size - 1
            dormant_step = 99 - dormant_request
            for step in range(99):
                remap = None
                if repeat and step == 32:
                    remap = tuple(reversed(slots))
                    order = [order[source] for source in remap]
                if step == dormant_step:
                    dormant_before = state_hashes(order, {dormant_request})
                active = [row for row, request in enumerate(order) if request + step + 1 < 100]
                positions = [len(prompt) + request + step if row in active else -1 for row, request in enumerate(order)]
                forced = torch.tensor(
                    [int(generated[request + step]) if row in active else 0 for row, request in enumerate(order)]
                )
                trace_before = evidence.before()
                predicted = generator.decode_forward(
                    forced,
                    positions,
                    slots=slots,
                    sample_on_device=True,
                    **inputs(order),
                    **(
                        {
                            "slot_remap": remap,
                            "reload_sampling_params": True,
                            "reset_sampling_state": True,
                            "sampling_params": greedy_params_for_order(SamplingParams, request_seeds, order),
                        }
                        if remap
                        else {}
                    ),
                )
                if remap:
                    run["sampling_seeds_after_migration"] = generator.request_seeds[: args.batch_size]
                    assert run["sampling_seeds_after_migration"] == [request_seeds[request] for request in order]
                evidence.after(
                    trace_before,
                    tuple(slots[row] for row in active),
                    repeat=repeat,
                    step=step,
                    migration=remap is not None,
                )
                hashes = logit_hashes(active)
                assert all(int(predicted[row]) == -1 for row in slots if row not in active)
                for row in active:
                    request = order[row]
                    record(row, request, request + step + 1, "decode", predicted[row], hashes[row])
                for slot, request in enumerate(order):
                    assert generator.caches[slot].position == min(len(prompt) + request + step + 1, len(prompt) + 99)
                if step % 8 == 0 or step == 98:
                    mark("DECODE", repeat=repeat, step=step, active=len(active))
                    save()
            assert dormant_before is not None
            run["dormant_state_before"], run["dormant_state_fields_before"] = dormant_before
            run["dormant_state_after"], run["dormant_state_fields_after"] = state_hashes(order, {dormant_request})
            run["dormant_state_exact"] = run["dormant_state_before"] == run["dormant_state_after"]
            run["dormant_state_mismatches"] = state_field_differences(
                run["dormant_state_fields_before"], run["dormant_state_fields_after"]
            )
            run["final_request_state_hashes"], run["final_request_state_fields"] = state_hashes(order)
            if repeat == 0:
                baseline_states = run["final_request_state_hashes"]
                baseline_fields = run["final_request_state_fields"]
            else:
                run["exact_same_history_final_state"] = run["final_request_state_hashes"] == baseline_states
                run["final_state_mismatches"] = state_field_differences(
                    baseline_fields, run["final_request_state_fields"]
                )
            before = dict(generator.counters)
            trace_before = evidence.before()
            empty = generator.decode_forward(
                torch.zeros(args.batch_size, dtype=torch.long),
                [-1] * args.batch_size,
                slots=slots,
                kv_cache=pool,
                page_table=None,
                sample_on_device=True,
                reload_inputs=False,
            )
            assert empty.tolist() == [-1] * args.batch_size and generator.counters == before
            evidence.after(trace_before, (), repeat=repeat, step="all_inactive")
            run["all_inactive_noop"] = True
            if not diagnostic:
                run["aggregate"] = accuracy(run["records"])
                run["per_phase"] = {
                    kind: accuracy([r for r in run["records"] if r["phase"] == kind]) for kind in ("prefill", "decode")
                }
                run["per_request"] = {
                    str(request): {
                        "aggregate": accuracy([r for r in run["records"] if r["request"] == request]),
                        **{
                            kind: accuracy(
                                [r for r in run["records"] if r["request"] == request and r["phase"] == kind]
                            )
                            for kind in ("prefill", "decode")
                        },
                    }
                    for request in slots
                }
                run["hf_accuracy_passed"] = (
                    run["aggregate"]["passed"]
                    and all(x["passed"] for x in run["per_phase"].values())
                    and all(x["passed"] for row in run["per_request"].values() for x in row.values())
                )
            run["state_control_passed"] = (
                run["exact_same_history_logits"]
                and run["exact_same_history_tokens"]
                and run["dormant_state_exact"]
                and run.get("exact_same_history_final_state", True)
            )
            run["passed"] = run["state_control_passed"] and run.get("hf_accuracy_passed", True)
            save()
        mark("RESIDENT_SAMPLED_FEEDBACK")

        def read_small_positions():
            return {
                (slot, index): int(ttnn.to_torch(ttnn.get_device_tensors(layer.attention.position)[0]).item())
                for slot, cache in enumerate(generator.caches)
                for index, layer in enumerate(cache.layers)
                if layer.attention is not None
            }

        result["resident_sampled_feedback"] = run_resident_feedback(
            generator, pool, evidence, generator._read_tokens, read_small_positions
        )
        save()
        assert addresses == [
            part.buffer_address()
            for pair in pool.pairs.values()
            for tensor in pair
            for part in ttnn.get_device_tensors(tensor)
        ]
        result["memory_final"] = memory(mesh)
        result["program_cache_entries_final"] = mesh.num_program_cache_entries()
        evidence.check_programs()
        assert evidence.native_ids() == evidence.ids
        result["factored_trace_evidence"]["passed"] = True
        result["counters"] = dict(generator.counters)
        result["passed"] = (
            all(run["passed"] for run in result["runs"]) and result["resident_sampled_feedback"]["passed"]
        )
        assert result["passed"], "Batched accuracy or matched-history control failed; inspect per-row/field evidence"
    except BaseException as exc:
        result.update(failed_phase=phase, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        # Preserve capture cardinality on early failure (e.g. an OOM halfway
        # through the first multi-row prefill, before any prediction is read).
        if generator is not None:
            result["counters_at_exit"] = dict(generator.counters)
            result["trace_cache_keys_at_exit"] = {
                name: [repr(key) for key in getattr(generator, name)]
                for name in ("prefill_traces", "traces", "sampling_traces")
            }
        if mesh is not None:
            try:
                result["memory_at_exit"] = memory(mesh)
            except Exception as memory_error:
                result["memory_at_exit_error"] = str(memory_error)
        save()
        if generator is not None:
            generator.close()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        print(json.dumps({"passed": result["passed"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
