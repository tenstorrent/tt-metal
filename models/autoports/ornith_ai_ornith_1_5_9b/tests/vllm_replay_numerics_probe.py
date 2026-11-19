# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Separate replay numerical drift from RNG/penalty state on the server stimulus.

Parent runs this device probe serially. Raw decode logits must be read BEFORE
sampling: the common penalty operation modifies trace_logits in place. The
readback hooks below are diagnostic only and never feed tokens to the model.
"""

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.vllm_adapter_decode_probe import build_adapter
from models.common.sampling.generator import SamplingParams, _hash_request_seed_to_device_seed


def compare_logits(left, right):
    left, right = left.double(), right.double()
    delta = left - right
    a, b = left - left.mean(), right - right.mean()
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    return {
        "exact": bool(torch.equal(left, right)),
        "finite": bool(torch.isfinite(left).all() and torch.isfinite(right).all()),
        "pcc": float(torch.dot(a, b) / denominator) if denominator else None,
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "top32_order_equal": bool(torch.equal(left.topk(32).indices, right.topk(32).indices)),
        "top64_overlap": len(set(left.topk(64).indices.tolist()) & set(right.topk(64).indices.tolist())),
    }


def raw_candidates(logits):
    # Full-vocabulary softmax of raw logits at temperature 1; these are NOT
    # the actual temperature/penalty/top-k/top-p sampling probabilities.
    probabilities = logits.double().softmax(-1)
    values, indices = logits.topk(64)
    return [
        {"token": int(token), "logit": float(value), "raw_softmax_probability": float(probabilities[token])}
        for token, value in zip(indices, values)
    ]


def snapshot_signature(snapshot):
    """Compact exact-table evidence; each table is also checked against history."""
    return {
        "seed": snapshot["seed"],
        "tables_sha256": {
            field: [hashlib.sha256(shard.contiguous().numpy().tobytes()).hexdigest() for shard in shards]
            for field, shards in snapshot["tables"].items()
        },
    }


def token_comparison(left, right):
    different = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), None)
    return {
        "exact": left == right,
        "lengths": [len(left), len(right)],
        "first_difference": (
            {"index_zero_based": different, "left": left[different], "right": right[different]}
            if different is not None
            else None
        ),
    }


def compact_linear_state(value, name, state_row):
    """CPU indexing matching MultichipDecoder.read_linear_state's pooled layout."""
    if state_row is None:
        return value.clone()
    if name == "conv":
        return value[:, :, state_row : state_row + 1, :].permute(2, 1, 0, 3).contiguous()
    if name == "recurrent":
        return value[state_row : state_row + 1].clone()
    raise ValueError(f"Unknown linear state: {name}")


def live_physical_pages(mapping, length, *, global_pages, page_offset, physical_page_count):
    """Use the decoder's effective page mapping, not the pooled row's nominal offset."""
    needed = (length + 31) // 32
    if length <= 0 or mapping.ndim != 1 or mapping.numel() < needed:
        raise ValueError("Invalid live-prefix page geometry")
    local = mapping[:needed].tolist()
    if any(isinstance(page, bool) or not isinstance(page, int) or page < 0 for page in local):
        raise ValueError("Live pages must be allocated integer IDs")
    physical = [page + (0 if global_pages else page_offset) for page in local]
    if len(set(physical)) != len(physical) or any(page >= physical_page_count for page in physical):
        raise ValueError("Live pages alias or exceed their physical pool")
    return physical


def logical_kv_prefix(page_values, length):
    """Selected physical pages are already in logical order; trim the unused tail."""
    if not page_values or any(value.ndim != 4 or value.shape[0] != 1 or value.shape[2] != 32 for value in page_values):
        raise ValueError("Expected individual [1,heads,32,dim] KV pages")
    # Concatenating on the token axis preserves heads when generalizing beyond GQA1.
    value = torch.cat(page_values, dim=2)[0]
    if not 0 < length <= value.shape[1]:
        raise ValueError("KV pages do not cover the logical prefix")
    return value[:, :length].clone()


def compare_model_tensor(left, right):
    same_shape = left.shape == right.shape
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    result = {
        "shapes": [list(left.shape), list(right.shape)],
        "same_shape": same_shape,
        "finite": finite,
        "exact": same_shape and bool(torch.equal(left, right)),
    }
    if same_shape and finite:
        delta = left.double() - right.double()
        norm = torch.linalg.vector_norm(left.double())
        worst_flat = int(delta.abs().reshape(-1).argmax())
        result.update(
            max_abs=float(delta.abs().max()),
            mean_abs=float(delta.abs().mean()),
            rmse=float(delta.square().mean().sqrt()),
            relative_l2=float(torch.linalg.vector_norm(delta) / norm) if norm else None,
            worst_index=list(torch.unravel_index(torch.tensor(worst_flat), delta.shape)),
        )
        result["worst_index"] = [int(index) for index in result["worst_index"]]
    return result


def collect_model_state(gen, decode_state, prefix, *, phase, layer_ids):
    """Diagnostic reads only; no tensor, token, or state is written back to the model."""
    length = len(prefix)
    if phase == "decode":
        ends = [
            int(ttnn.to_torch(shard).reshape(-1)[0]) for shard in ttnn.get_device_tensors(decode_state.positions[0])
        ]
        mapping = decode_state.host_tables
    else:
        ends = [gen.prefill_runtime.bindings[(id(gen.cache), 0)]["end"]]
        mapping = gen.page_table
    if any(end != length for end in ends):
        raise AssertionError(f"Model state is not aligned with consumed prefix: {ends} != {length}")
    snapshot = snapshot_logical_model_state(gen, mapping, length, layer_ids=layer_ids)
    return {"prefix": list(prefix), "logical_end": length, "observed_ends": ends, **snapshot}


def snapshot_logical_model_state(gen, mapping, length, *, layer_ids):
    """Shared warmup/read path: identical cache tensors, shards and slice offsets."""
    tensors, pages, storage = {}, {}, {}
    for layer_id, layer, states in zip(layer_ids, gen.model.layers, gen.cache, strict=True):
        for slot, cached in enumerate(states):
            if layer.kind == "linear_attention":
                for name in ("conv", "recurrent"):
                    source = getattr(cached, name)
                    if id(source) not in storage:
                        storage[id(source)] = [
                            ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(source)
                        ]
                    for rank, host in enumerate(storage[id(source)]):
                        tensors[f"layer{layer_id}.slot{slot}.{name}.rank{rank}"] = compact_linear_state(
                            host, name, cached.state_row
                        )
            else:
                # Active causal prefix, plus one mapped sentinel page per inactive
                # slot. Never transfer the entire 262K-context physical KV pool.
                live_length = length if slot == 0 else 32
                physical = live_physical_pages(
                    mapping[slot],
                    live_length,
                    global_pages=cached.global_pages,
                    page_offset=cached.page_offset,
                    physical_page_count=cached.key.shape[0],
                )
                pages[f"layer{layer_id}.slot{slot}"] = physical
                for name in ("key", "value"):
                    for rank, shard in enumerate(ttnn.get_device_tensors(getattr(cached, name))):
                        host_pages = [ttnn.to_torch(shard[page : page + 1]).clone() for page in physical]
                        tensors[f"layer{layer_id}.slot{slot}.{name}.rank{rank}"] = logical_kv_prefix(
                            host_pages, live_length
                        )
        if layer.kind == "full_attention":
            owned = [page for key, values in pages.items() if key.startswith(f"layer{layer_id}.") for page in values]
            if len(owned) != len(set(owned)):
                raise AssertionError("Active/inactive sampled page regions alias")
    return {"pages": pages, "tensors": tensors}


def warm_model_state_reads(gen, decode_state, *, length, layer_ids):
    """Prepare diagnostic program-cache allocations before any live trace exists."""
    if any(state.model_trace is not None for state in gen.states.values()):
        raise RuntimeError("Model-state diagnostic reads must be warmed before capture")
    if not torch.equal(decode_state.host_tables, gen.page_table):
        raise AssertionError("Diagnostic decode/prefill mappings differ before capture")
    before = gen.device.num_program_cache_entries()
    # Use the actual reader, not shape-only slices: integer page offsets and
    # rank-local shard placement participate in the TTNN program-cache key.
    # CPU copies are discarded; no model state is mutated or fed back.
    snapshot_logical_model_state(gen, gen.page_table, length, layer_ids=layer_ids)
    after_first = gen.device.num_program_cache_entries()
    snapshot_logical_model_state(gen, gen.page_table, length, layer_ids=layer_ids)
    after_second = gen.device.num_program_cache_entries()
    if after_second != after_first:
        raise AssertionError("Diagnostic read program-cache keys did not stabilize before capture")
    return {
        "logical_prefix": length,
        "programs_before": before,
        "programs_after_first": after_first,
        "programs_after_second": after_second,
        "scope": "Actual snapshot reads, including all selected page offsets and TP shards, before capture; no model-state writes.",
    }


def summarize_model_states(snapshots):
    comparisons = {}
    for name, left_name, right_name in (
        ("decode_vs_host_replay", "decode", "host_replay"),
        ("decode_vs_device_replay", "decode", "device_replay"),
        ("host_vs_device_replay", "host_replay", "device_replay"),
    ):
        left, right = snapshots[left_name], snapshots[right_name]
        if left["tensors"].keys() != right["tensors"].keys():
            raise AssertionError("Model-state tensor inventories differ")
        comparisons[name] = {
            key: compare_model_tensor(value, right["tensors"][key]) for key, value in left["tensors"].items()
        }
    base = snapshots["decode"]
    checks = {
        "same_prefix151": len(base["prefix"]) == 151 and all(s["prefix"] == base["prefix"] for s in snapshots.values()),
        "same_logical_end": all(s["logical_end"] == 151 for s in snapshots.values()),
        "same_page_mapping": all(s["pages"] == base["pages"] for s in snapshots.values()),
        "finite_and_shape_matched": all(
            m["finite"] and m["same_shape"] for rows in comparisons.values() for m in rows.values()
        ),
        "host_device_replay_exact": all(m["exact"] for m in comparisons["host_vs_device_replay"].values()),
        "inactive_preserved": all(
            m["exact"] for rows in comparisons.values() for key, m in rows.items() if ".slot0." not in key
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "comparisons": comparisons,
        "metadata": {name: {k: v for k, v in snap.items() if k != "tensors"} for name, snap in snapshots.items()},
        "scope": "Layers0/3, B3; full active conv/recurrent and logical KV prefix151; inactive conv/recurrent and first mapped KV page only. Readback never feeds generation.",
        "interpretation": "Decode/prefill error magnitudes have no acceptance tolerance here. Matching replay states do not rule out a systematic prefill/decode numerical or kernel defect.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--greedy", action="store_true", help="Use temperature zero with the same penalties.")
    parser.add_argument(
        "--compare-model-state",
        action="store_true",
        help="Read aligned prefix151 model states and save raw tensors; diagnostic only.",
    )
    parser.add_argument(
        "--follow-outputs", type=int, default=23, help="Total outputs per branch; default keeps output23 probe."
    )
    parser.add_argument(
        "--served-reference", type=Path, help="Optional diagnostic preemption.json to compare prefixes."
    )
    args = parser.parse_args()
    if args.follow_outputs < 23:
        parser.error("--follow-outputs must be at least 23")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    prompt = [9707 + i % 13 for i in range(129)]
    report = {
        "status": "running",
        "layers": [0, 3],
        "batch": 3,
        "active_slots": [0],
        "context": args.context,
        "prompt": prompt,
        "resume_after_outputs": 22,
        "sampling": {
            "temperature": 0.0 if args.greedy else 0.8,
            "top_k": 32,
            "top_p": 0.9,
            "seed": 42,
            "presence_penalty": 0.75,
            "frequency_penalty": 0.5,
            "repetition_penalty": 1.1,
        },
        "state_checks": [],
    }

    def save():
        path.write_text(json.dumps(report, indent=2) + "\n")

    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    gen = None
    try:
        gen = build_adapter(None, mesh, max_batch_size=3, max_seq_len=args.context)
        state = gen.prepare_decode(kv_cache=gen.cache, page_table=gen.page_table, slots=(0, 1, 2))
        if args.compare_model_state:
            report["model_state_read_warmup"] = warm_model_state_reads(
                gen, state, length=151, layer_ids=report["layers"]
            )
            save()
        gen.capture(state)
        controls = {
            "top_k": 32,
            "top_p": 0.9,
            "presence_penalty": 0.75,
            "frequency_penalty": 0.5,
            "repetition_penalty": 1.1,
        }
        temperature = report["sampling"]["temperature"]
        prefill_params = SamplingParams(temperature=[temperature], seed=[42], **controls)
        decode_params = SamplingParams(temperature=[temperature, 0.0, 0.0], seed=[42, None, None], **controls)
        base = (_hash_request_seed_to_device_seed(42, 0) % 500_000) + 1
        vocab = gen.sampler.tt_penalties.vocab_size
        fields = ("prompt_mask", "output_mask", "output_counts", "output_counts_gathered")

        def sampler_snapshot():
            return {
                "seed": [
                    int(ttnn.to_torch(shard).reshape(-1)[0])
                    for shard in ttnn.get_device_tensors(gen.sampler.tt_sampling.seeds_tt_tensor)
                ],
                "tables": {
                    field: [
                        ttnn.to_torch(shard).reshape(32, -1)[0].long().clone()
                        for shard in ttnn.get_device_tensors(getattr(gen.sampler.tt_penalties, field))
                    ]
                    for field in fields
                },
            }

        def check_state(name, snapshot, outputs, *, record=True):
            counts = torch.bincount(torch.tensor(outputs, dtype=torch.int64), minlength=vocab)
            prompt_mask = (torch.bincount(torch.tensor(prompt), minlength=vocab) > 0).long()
            expected = {
                "prompt_mask": prompt_mask,
                "output_mask": (counts > 0).long(),
                "output_counts": counts,
                "output_counts_gathered": counts,
            }
            checked = {}
            for field in fields:
                checked[field] = []
                for rank, actual in enumerate(snapshot["tables"][field]):
                    wanted = expected[field] if field == "output_counts_gathered" else expected[field].chunk(4)[rank]
                    exact = torch.equal(actual, wanted)
                    checked[field].append(
                        {
                            "rank": rank,
                            "exact": exact,
                            "actual_shape": list(actual.shape),
                            "expected_shape": list(wanted.shape),
                            "different_elements": (
                                int((actual != wanted).sum()) if actual.shape == wanted.shape else None
                            ),
                        }
                    )
            expected_seeds = [base + len(outputs)] * 4
            passed = snapshot["seed"] == expected_seeds and all(
                item["exact"] for rows in checked.values() for item in rows
            )
            support = torch.nonzero(counts).reshape(-1).tolist()
            checked_state = {
                "name": name,
                "passed": passed,
                "actual_seeds": snapshot["seed"],
                "expected_seeds": expected_seeds,
                "tables": checked,
                "expected_output_counts": [[token, int(counts[token])] for token in support],
            }
            if record:
                report["state_checks"].append(checked_state)
                save()
            return checked_state

        def prefill(ids, *, device):
            result = gen.prefill_forward(
                torch.tensor([ids]),
                page_table=gen.page_table[:1],
                kv_cache=gen.cache,
                prompt_lens=[len(ids)],
                original_prompt_lens=[129],
                empty_slots=[0],
                sampling_params=prefill_params if device else None,
            )
            return result[0] if isinstance(result, tuple) else result

        def decode(outputs, *, reset=False):
            handle = gen.decode_forward(
                torch.tensor([[outputs[-1]], [0], [0]]) if reset else None,
                [129 + len(outputs) - 1, -1, -1] if reset else None,
                page_table=gen.page_table if reset else None,
                kv_cache=gen.cache,
                sampling_params=decode_params,
                reload_inputs=reset,
                reload_sampling_params=reset,
                reset_sampling_state=reset,
                prompt_tokens=[prompt, [], []] if reset else None,
                output_tokens=[outputs, [], []] if reset else None,
                read_from_device=False,
            )
            read, events = gen.read_decode_output(handle, async_read=True)
            for event in events:
                ttnn.event_synchronize(event)
            return int(gen.process_decode_output_host(read, is_tokens=True)[0, 0])

        generated = [int(prefill(prompt, device=True)[0, 0])]
        while len(generated) < 22:
            generated.append(decode(generated, reset=len(generated) == 1))
        checkpoint = sampler_snapshot()
        check_state("uninterrupted_after_22_before_output23", checkpoint, generated)
        raw = {}
        model_states = {}
        original_execute = ttnn.execute_trace

        def before_decode_sample(device, trace, **kwargs):
            if trace == state.sample_traces[(state.masked, gen.sampler._penalties_active)]:
                # Synchronizing diagnostic read sits between model and sampler;
                # reading after the sampler would observe penalized logits.
                raw["uninterrupted"] = (
                    gen.model.host_logits(state.trace_logits[state.masked]).reshape(32, -1)[0].clone()
                )
                if args.compare_model_state:
                    model_states["decode"] = collect_model_state(
                        gen, state, prompt + generated, phase="decode", layer_ids=report["layers"]
                    )
            return original_execute(device, trace, **kwargs)

        with patch.object(ttnn, "execute_trace", before_decode_sample):
            uninterrupted_token = decode(generated)
        if "uninterrupted" not in raw:
            raise AssertionError("Raw-logit hook did not run before the sampling trace")
        check_state("uninterrupted_after_output23", sampler_snapshot(), generated + [uninterrupted_token])
        report["generated_22"] = generated
        report["uninterrupted_output23"] = uninterrupted_token
        save()

        # Keep raw logits and compact state oracles from each decision. These
        # diagnostics never supply logits/tokens back to generation. The only
        # host token feedback below is the existing explicit post-prefill reload.
        baseline_outputs = generated + [uninterrupted_token]
        baseline_raw = {22: raw["uninterrupted"]}
        baseline_states = {
            22: {
                "signature": snapshot_signature(checkpoint),
                "oracle": report["state_checks"][0],
            }
        }

        def observed_decode(outputs, *, reset=False):
            observed = {}

            def before_sample(device, trace, **kwargs):
                if trace == state.sample_traces[(state.masked, gen.sampler._penalties_active)]:
                    logits = gen.model.host_logits(state.trace_logits[state.masked]).reshape(32, -1)[0].clone()
                    snapshot = sampler_snapshot()
                    observed.update(
                        logits=logits,
                        signature=snapshot_signature(snapshot),
                        oracle=check_state("before_output" + str(len(outputs) + 1), snapshot, outputs, record=False),
                    )
                return original_execute(device, trace, **kwargs)

            with patch.object(ttnn, "execute_trace", before_sample):
                token = decode(outputs, reset=reset)
            if not observed:
                raise AssertionError("Follow-up hook did not run before the sampling trace")
            return token, observed

        while len(baseline_outputs) < args.follow_outputs:
            index = len(baseline_outputs)
            token, observed = observed_decode(baseline_outputs)
            baseline_raw[index] = observed.pop("logits")
            baseline_states[index] = observed
            baseline_outputs.append(token)

        replay_prefix = prompt + generated
        raw["replay_host_prefill"] = prefill(replay_prefix, device=False).reshape(-1).float().clone()
        if args.compare_model_state:
            model_states["host_replay"] = collect_model_state(
                gen, state, replay_prefix, phase="prefill", layer_ids=report["layers"]
            )
        original_sample = gen._sample_logits
        replay_pre_sample = {}

        def before_prefill_sample(logits, tokens):
            raw["replay_device_prefill"] = gen.model.host_logits(logits).reshape(32, -1)[0].clone()
            replay_pre_sample.update(sampler_snapshot())
            if args.compare_model_state:
                model_states["device_replay"] = collect_model_state(
                    gen, state, replay_prefix, phase="prefill", layer_ids=report["layers"]
                )
            return original_sample(logits, tokens)

        with patch.object(gen, "_sample_logits", before_prefill_sample):
            replay_token = int(prefill(replay_prefix, device=True)[0, 0])
        check_state("replay_before_output23", replay_pre_sample, generated)
        check_state("replay_after_output23", sampler_snapshot(), generated + [replay_token])
        state_identical = checkpoint["seed"] == replay_pre_sample["seed"] and all(
            torch.equal(left, right)
            for field in fields
            for left, right in zip(checkpoint["tables"][field], replay_pre_sample["tables"][field])
        )
        report["pre_sample_state_identical"] = state_identical
        if args.compare_model_state:
            model_state_path = path.with_suffix(".model_state.pt")
            torch.save(model_states, model_state_path)
            report["model_state_comparison"] = summarize_model_states(model_states)
            report["model_state_comparison"]["raw_tensor_file"] = str(model_state_path)
            save()
            if not report["model_state_comparison"]["passed"]:
                raise AssertionError("Aligned model-state structural/replay checks failed; see saved comparisons")
        report["replay_output23"] = replay_token
        report["token_equal"] = uninterrupted_token == replay_token
        report["logits"] = {
            "decode_vs_host_prefill": compare_logits(raw["uninterrupted"], raw["replay_host_prefill"]),
            "decode_vs_device_prefill": compare_logits(raw["uninterrupted"], raw["replay_device_prefill"]),
            "host_vs_device_prefill": compare_logits(raw["replay_host_prefill"], raw["replay_device_prefill"]),
        }
        report["raw_top64"] = {name: raw_candidates(logits) for name, logits in raw.items()}
        replay_outputs = generated + [replay_token]
        replay_states = {
            22: {
                "signature": snapshot_signature(replay_pre_sample),
                "oracle": report["state_checks"][2],
            }
        }
        first_divergence = None

        def compare_decision(index, logits):
            nonlocal first_divergence
            if first_divergence is None and baseline_outputs[index] != replay_outputs[index]:
                first_divergence = {
                    "index_zero_based": index,
                    "baseline_token": baseline_outputs[index],
                    "replay_token": replay_outputs[index],
                    "preceding_tokens_identical": baseline_outputs[:index] == replay_outputs[:index],
                    "pre_sample_state_identical": (
                        baseline_states[index]["signature"] == replay_states[index]["signature"]
                    ),
                    "baseline_state": baseline_states[index],
                    "replay_state": replay_states[index],
                    "logits": compare_logits(baseline_raw[index], logits),
                    "raw_top64": {
                        "baseline": raw_candidates(baseline_raw[index]),
                        "replay": raw_candidates(logits),
                    },
                }
                if args.follow_outputs > 23 or args.greedy:
                    logits_path = path.with_suffix(".first_divergence.pt")
                    torch.save({"baseline": baseline_raw[index], "replay": logits}, logits_path)
                    first_divergence["raw_logits_file"] = str(logits_path)

        compare_decision(22, raw["replay_device_prefill"])
        while len(replay_outputs) < args.follow_outputs:
            index = len(replay_outputs)
            token, observed = observed_decode(replay_outputs, reset=index == 23)
            logits = observed.pop("logits")
            replay_states[index] = observed
            replay_outputs.append(token)
            compare_decision(index, logits)

        follow_states_valid = all(
            item["oracle"]["passed"] for states in (baseline_states, replay_states) for item in states.values()
        )
        if args.follow_outputs > 23 or args.greedy or args.served_reference:
            report["followup"] = {
                "output_count": args.follow_outputs,
                "baseline_tokens": baseline_outputs,
                "replay_tokens": replay_outputs,
                "strict_token_equal": baseline_outputs == replay_outputs,
                "all_pre_sample_state_oracles_passed": follow_states_valid,
                "first_divergence": first_divergence,
                "scope": "Raw-logit and sampler-state diagnostics; no recurrent/convolution/KV-state equivalence claim.",
                "greedy_control_note": (
                    "Seed42 and top-k/top-p are retained for the state oracle; temperature0 uses argmax. "
                    "The served greedy reference may omit those controls."
                    if args.greedy
                    else None
                ),
            }
            if args.served_reference:
                reference = json.loads(args.served_reference.read_text())
                mode = "device_greedy" if args.greedy else "device_seeded"
                cases = {case["forced"]: case for case in reference["cases"] if case["name"] == mode}
                report["followup"]["served_reference"] = {
                    "path": str(args.served_reference),
                    "mode": mode,
                    "baseline": token_comparison(baseline_outputs, cases[False]["tokens"][: args.follow_outputs]),
                    "forced": token_comparison(replay_outputs, cases[True]["tokens"][: args.follow_outputs]),
                    "requests": {str(forced): case["request"] for forced, case in cases.items()},
                }
        report[
            "probability_semantics"
        ] = "Full-vocabulary raw-logit softmax at temperature 1, before penalties/filtering."
        exact_logits = report["logits"]["decode_vs_device_prefill"]["exact"]
        valid_state = (
            state_identical and follow_states_valid and all(check["passed"] for check in report["state_checks"])
        )
        finite_logits = all(metrics["finite"] for metrics in report["logits"].values())
        if first_divergence is not None:
            valid_state = valid_state and first_divergence["pre_sample_state_identical"]
            finite_logits = finite_logits and first_divergence["logits"]["finite"]
        decision_logits_exact = first_divergence["logits"]["exact"] if first_divergence else exact_logits
        report["diagnosis"] = (
            "sampler_state_mismatch"
            if not valid_state
            else (
                "nonfinite_logits"
                if not finite_logits
                else (
                    "replay_numerical_difference_with_matching_sampler_state"
                    if not decision_logits_exact
                    else (
                        "sampling_path_difference_with_identical_logits_and_state"
                        if first_divergence is not None
                        else "identical_logits_state_and_token"
                    )
                )
            )
        )
        report["status"] = "complete" if valid_state and finite_logits else "failed"
        report["counters"] = dict(gen.counters)
        save()
        if not valid_state or not finite_logits:
            raise AssertionError(report["diagnosis"])
        if exact_logits and uninterrupted_token != replay_token:
            raise AssertionError("Identical raw logits and sampler state produced different output23")
        if first_divergence is not None and first_divergence["logits"]["exact"]:
            raise AssertionError("Identical raw logits and sampler state produced different follow-up tokens")
        print("REPLAY_NUMERICS_PROBE_COMPLETE " + report["diagnosis"], flush=True)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
