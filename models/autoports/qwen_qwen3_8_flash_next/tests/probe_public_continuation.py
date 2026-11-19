# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real public decoder continuation, heterogeneous starts, and live trace isolation.

Two requests first populate independent31/33-token prefixes, then one public
tuple prefill consumes8/7 resumed rows. Every resumed contribution is compared
with the same rows of an independent full HF prefill. Dropping all prefix
history must fail that same per-row contribution gate. No anomaly exception is
available here. One tuple decode trace consumes two changed token inputs per
request, interleaved with a third request's independent65/97/129-token prefills.
Those logical lengths are never run on device before capture; their32+1 math
buckets are warmed using the existing33-token prefix and single-token decode.
For QSA, the
same trace is finally replayed after in-place page-table/cache remapping.

The three contexts are distinct: fresh prefix, nonzero-history continuation,
and independent fresh prefill while a continuation decode trace remains live.
This short test does not establish long-context or maximum-batch capacity.
Default comparisons are strict literal HF. Explicit --ple-backend-control may
classify only fresh independent-prefill contribution discrepancies reproduced
by one independent packed-lookup/six-projection CPU control. Every literal whole
and every matched whole/contribution gate remains .995; resumed/decode gates
stay strict literal HF. Any classification remains subject to stage review.
"""

import argparse
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")

import torch
from safetensors import safe_open
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer, Qwen4ExpTextRotaryEmbedding

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_quantization import SelectedRows
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa import ReferenceCache
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa_decoder import metric_passes, metrics

STARTS = (31, 33)
COUNTS = (8, 7)
ENDS = tuple(start + count for start, count in zip(STARTS, COUNTS))
LIVE_LENGTHS = (65, 97, 129)


def load_reference(snapshot, layer, config):
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{layer}."
    keys = [key for key in index if key.startswith(prefix) and ".ngram_embedding.shard_" not in key]
    weights = {}
    for filename in sorted({index[key] for key in keys}):
        with safe_open(snapshot / filename, framework="pt") as stream:
            for key in keys:
                if index[key] == filename:
                    weights[key.removeprefix(prefix)] = stream.get_tensor(key)
    with torch.device("meta"):
        reference = Qwen4ExpTextDecoderLayer(config, layer)
    result = reference.load_state_dict(weights, strict=False, assign=True)
    assert not result.unexpected_keys
    assert result.missing_keys == (["ple.ple_embedding.ngram_embedding.weight"] if layer == 1 else [])
    if layer == 1:
        reference.ple.ple_embedding.ngram_embedding = SelectedRows(
            snapshot, index, prefix + "ple.ple_embedding.ngram_embedding.", "absmax"
        )
    reference.eval()
    return reference, weights


@torch.inference_mode()
def make_fixture(reference, config, layer):
    # Reuse the established short-contract activation distributions/seeds.
    generator = torch.Generator().manual_seed({0: 932, 1: 1753, 3: 1798}[layer])
    prompt = (torch.randn(1, 65, 10240, generator=generator) * 0.02).bfloat16()
    for _ in range(3):
        torch.randn(1, 1, 10240, generator=generator)
    ids = torch.randint(0, config.vocab_size, (1, 68), generator=generator)
    rope = Qwen4ExpTextRotaryEmbedding(config)

    def cache():
        return ReferenceCache() if layer == 3 else DynamicCache(config=config)

    def forward(start, count, state):
        value = prompt[:, start : start + count]
        if layer == 3:
            end = start + count
            embeddings = rope(value, torch.arange(end).reshape(1, -1))
            mask = torch.zeros(1, 1, count, end, dtype=torch.bfloat16)
            future = torch.arange(end)[None, :] > torch.arange(start, end)[:, None]
            mask.masked_fill_(future, torch.finfo(torch.bfloat16).min)
            return reference(value, position_embeddings=embeddings, attention_mask=mask, past_key_values=state)
        kwargs = {"ple_input_ids": ids[:, start : start + count]} if layer == 1 else {}
        return reference(value, position_embeddings=(None, None), past_key_values=state, **kwargs)

    expected = forward(0, 65, cache())
    controls = []
    for start, count in zip(STARTS, COUNTS):
        state = cache()
        forward(0, start, state)
        resumed = forward(start, count, state)
        measured = metrics(expected[:, start : start + count], resumed, prompt[:, start : start + count])
        assert metric_passes(measured), "HF continuation must match its full-prefill oracle"
        controls.append(measured)
    return prompt, ids, expected, controls


@torch.inference_mode()
def make_live_fixtures(reference, config, layer):
    """Independent HF full-prefill oracles without changing the original RNG."""
    rope = Qwen4ExpTextRotaryEmbedding(config)
    fixtures = []
    for length in LIVE_LENGTHS:
        seed = {0: 932, 1: 1753, 3: 1798}[layer] + 100000 + length
        generator = torch.Generator().manual_seed(seed)
        prompt = (torch.randn(1, length, 10240, generator=generator) * 0.02).bfloat16()
        tokens = torch.randint(0, config.vocab_size, (1, length), generator=generator)
        if layer == 3:
            positions = torch.arange(length)
            mask = torch.zeros(1, 1, length, length, dtype=torch.bfloat16)
            mask.masked_fill_(positions[None, :] > positions[:, None], torch.finfo(torch.bfloat16).min)
            expected = reference(
                prompt,
                position_embeddings=rope(prompt, positions.reshape(1, -1)),
                attention_mask=mask,
                past_key_values=ReferenceCache(),
            )
        else:
            kwargs = {"ple_input_ids": tokens} if layer == 1 else {}
            expected = reference(
                prompt, position_embeddings=(None, None), past_key_values=DynamicCache(config=config), **kwargs
            )
        fixtures.append({"length": length, "seed": seed, "prompt": prompt, "tokens": tokens, "expected": expected})
    return fixtures


def device_run(args, config, weights, fixture, live_fixtures, report):
    from ttnn.tools import trace_allocation_tracker

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit
    from models.autoports.qwen_qwen3_8_flash_next.tt.functional_decoder import FunctionalDecoder
    from models.autoports.qwen_qwen3_8_flash_next.tt.packed_embedding import PackedEmbedding

    device_readbacks = {
        "identity": {
            "layer": args.layer,
            "expert_dtype": args.expert_dtype,
            "snapshot": str(args.snapshot),
            "config_sha256": hashlib.sha256((args.snapshot / "config.json").read_bytes()).hexdigest(),
            "hf_source_sha256": hashlib.sha256(
                Path(inspect.getfile(Qwen4ExpTextDecoderLayer)).read_bytes()
            ).hexdigest(),
            "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "Existing fresh live-prefill CPU readbacks only; no acceptance change or device injection",
        },
        "live_other_prefill": [],
    }
    assert trace_allocation_tracker.TRACE_ALLOC_TRACKING, "Live trace allocation tracking is required"
    prompt, tokens, expected, _ = fixture
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    checks = []
    fresh_literal_checks = []
    fresh_acceptance_checks = []
    try:
        packed = (
            PackedEmbedding.from_checkpoint(args.snapshot, cache_dir=args.cache_dir, mesh_device=mesh)
            if args.layer == 1
            else None
        )
        model = FunctionalDecoder.from_state_dict(
            weights,
            hf_config=config,
            layer_idx=args.layer,
            mesh_device=mesh,
            packed_embedding=packed,
            expert_dtype=getattr(ttnn, args.expert_dtype),
        )
        permutations = [
            torch.tensor([[2, 0, 3, 1, 6, 4, 7, 5]]),
            torch.tensor([[3, 1, 0, 2, 7, 5, 4, 6]]),
            torch.tensor([[1, 3, 2, 0, 5, 7, 6, 4]]),
        ]
        states = tuple(
            model.create_state(capacity=256, page_table=page.int())
            if args.layer == 3
            else model.create_state(hash_lengths=(1, 32) if index == 2 else (1, 7, 8, 31, 32))
            for index, page in enumerate(permutations)
        )
        names = (
            ("k_cache", "v_cache", "raw_history", "compressed_keys", "position", "page_table")
            if args.layer == 3
            else (("conv", "recurrent", "ple_conv", "token_history") if args.layer == 1 else ("conv", "recurrent"))
        )

        def buffers(state):
            owner = state.attention if args.layer == 3 else state
            return {name: getattr(owner, name) for name in names}

        def host(value, target=None, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                value.contiguous(),
                dtype=target.dtype if target is not None else dtype,
                layout=target.layout if target is not None else layout,
            )

        def snapshot(state):
            return {name: ttnn.to_torch(tensor).clone() for name, tensor in buffers(state).items()}

        def restore(state, saved):
            for name, tensor in buffers(state).items():
                ttnn.copy_host_to_device_tensor(host(saved[name], tensor), tensor)

        def equal(left, right):
            return all(torch.equal(left[name], right[name]) for name in names)

        def prepare(start, count, input_prompt=prompt, input_tokens=tokens):
            hidden = ttnn.to_device(host(input_prompt[:, start : start + count].unsqueeze(0)), mesh)
            words = None
            if args.layer == 1:
                value = torch.zeros(1, count, 16, dtype=torch.int64)
                value[..., 0], value[..., 1] = input_tokens[:, start : start + count], 1
                words = ttnn.to_device(host(value, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), mesh)
            return hidden, words

        prefixes = [prepare(0, start) for start in STARTS]
        resumed = [prepare(start, count) for start, count in zip(STARTS, COUNTS)]
        decode_inputs = [[prepare(end + step, 1) for end in ENDS] for step in range(2)]
        other_warm_input = prepare(0, 33)
        live_inputs = [prepare(0, case["length"], case["prompt"], case["tokens"]) for case in live_fixtures]
        report["live_prefill_preparation"] = {
            "unseen_logical_lengths": list(LIVE_LENGTHS),
            "public_prefill_lengths_warmed_before_capture": sorted({*STARTS, *COUNTS}),
            "decode_length_warmed_before_capture": 1,
            "new_case_math_buckets": [32, 1],
            "input_tensors_allocated_before_capture": True,
            "new_case_prefill_executed_before_capture": False,
            "qsa_capacity": 256 if args.layer == 3 else None,
            "qsa_page_permutations": [page.tolist() for page in permutations] if args.layer == 3 else None,
        }
        initial = [snapshot(state) for state in states]

        def prefill(pairs, target):
            with runtime_audit():
                return model.prefill_forward(
                    tuple(pair[0] for pair in pairs),
                    state=tuple(target),
                    token_ids=tuple(pair[1] for pair in pairs) if args.layer == 1 else None,
                )

        def consume(outputs, spans):
            results = []
            for output, (start, count) in zip(outputs, spans):
                actual = ttnn.to_torch(output).reshape(1, count, 10240)
                value = metrics(expected[:, start : start + count], actual, prompt[:, start : start + count])
                results.append(value)
                ttnn.deallocate(output)
            return results

        report["fresh_prefix"] = consume(prefill(prefixes, states[:2]), [(0, start) for start in STARTS])
        checks.extend(metric_passes(value) for value in report["fresh_prefix"])
        report["tuple_continuation"] = consume(prefill(resumed, states[:2]), list(zip(STARTS, COUNTS)))
        checks.extend(metric_passes(value) for value in report["tuple_continuation"])
        post_resume = [snapshot(state) for state in states[:2]]
        if args.layer == 3:
            positions = [int(saved["position"].reshape(-1)[0]) for saved in post_resume]
            assert positions == list(ENDS)
            report["tuple_positions_after_continuation"] = positions
        for index, state in enumerate(states[:2]):
            restore(state, initial[index])
            if args.layer == 3:
                target = state.attention.position
                ttnn.copy_host_to_device_tensor(host(torch.tensor([STARTS[index]], dtype=torch.int32), target), target)
        report["dropped_history"] = consume(prefill(resumed, states[:2]), list(zip(STARTS, COUNTS)))
        for value in report["dropped_history"]:
            detected = value["finite"] and any(
                row["contribution_pcc"] is not None and row["contribution_pcc"] < 0.995 for row in value["per_row"]
            )
            value["negative_control_detected"] = detected
            checks.append(detected)
        for state, saved in zip(states[:2], post_resume):
            restore(state, saved)

        # Warm the third-request prefill and exact tuple decode before capture.
        consume(prefill([other_warm_input], [states[2]]), [(0, 33)])
        restore(states[2], initial[2])
        stable_inputs = decode_inputs[0]

        def decode():
            return model.decode_forward(
                tuple(pair[0] for pair in stable_inputs),
                state=states[:2],
                token_ids=tuple(pair[1] for pair in stable_inputs) if args.layer == 1 else None,
            )

        def refresh(step):
            for index, (source, target) in enumerate(zip(decode_inputs[step], stable_inputs)):
                if step:
                    for src, dst in zip(source, target):
                        if src is not None:
                            ttnn.copy(src, dst)
                else:
                    # Stable input0 was overwritten by step1 during the prior
                    # replay. Restore its original host value explicitly.
                    end = ENDS[index]
                    ttnn.copy_host_to_device_tensor(host(prompt[:, end : end + 1].unsqueeze(0)), target[0])
                    if args.layer == 1:
                        value = torch.zeros(1, 1, 16, dtype=torch.int64)
                        value[..., 0], value[..., 1] = tokens[:, end : end + 1], 1
                        ttnn.copy_host_to_device_tensor(host(value, target[1]), target[1])

        # Input refresh is part of the live trace lifecycle too. Compile the
        # BF16 hidden and (for PLE) UINT32 token device-copy signatures before
        # capture, then restore input0 without mutating request state.
        refresh(1)
        refresh(0)
        with runtime_audit():
            warmed = decode()
        ttnn.synchronize_device(mesh)
        for output in warmed:
            ttnn.deallocate(output)
        for state, saved in zip(states[:2], post_resume):
            restore(state, saved)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        with runtime_audit():
            outputs = decode()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        # Capture records commands without computing these outputs. Acknowledge
        # their exact backing buffers; read values only after a completed replay.
        for output in outputs:
            trace_allocation_tracker.acknowledge_corruptible(output)
        report["capture_outputs_acknowledged"] = len(outputs)
        report["capture_output_readbacks_before_first_replay"] = 0
        report["decode"], report["live_other_prefill"] = [], []
        report["repeated_output_equal"], report["repeated_state_equal"] = [], []
        report["other_state_unchanged_by_decode"] = []
        first_outputs, first_final_states, first_step_states = [], [], []
        first_other_outputs, first_other_states = {}, {}
        for repetition in range(2):
            for case_index, (case, pair) in enumerate(zip(live_fixtures, live_inputs)):
                length = case["length"]
                for state, saved in zip(states[:2], post_resume):
                    restore(state, saved)
                restore(states[2], initial[2])
                other_snapshot = snapshot(states[2])
                for step in range(2):
                    refresh(step)
                    with runtime_audit():
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    observed = [ttnn.to_torch(output).reshape(1, 1, 10240).clone() for output in outputs]
                    for index, actual in enumerate(observed):
                        position = ENDS[index] + step
                        value = metrics(
                            expected[:, position : position + 1], actual, prompt[:, position : position + 1]
                        )
                        report["decode"].append(
                            {
                                "repetition": repetition,
                                "live_prefill_length": length,
                                "step": step,
                                "request": index,
                                **value,
                            }
                        )
                        checks.append(metric_passes(value))
                        if args.layer == 3:
                            assert int(ttnn.to_torch(states[index].attention.position).reshape(-1)[0]) == position + 1
                    current = [snapshot(state) for state in states[:2]]
                    if repetition == 0 and case_index == 0:
                        first_outputs.append(observed)
                        if step == 0:
                            first_step_states = current
                        else:
                            first_final_states = current
                    else:
                        repeated = [torch.equal(a, b) for a, b in zip(first_outputs[step], observed)]
                        report["repeated_output_equal"].append(repeated)
                        checks.extend(repeated)
                        baseline_states = first_step_states if step == 0 else first_final_states
                        repeated = [equal(a, b) for a, b in zip(baseline_states, current)]
                        report["repeated_state_equal"].append(repeated)
                        checks.extend(repeated)
                    unchanged = equal(other_snapshot, snapshot(states[2]))
                    report["other_state_unchanged_by_decode"].append(unchanged)
                    checks.append(unchanged)
                    if step == 0:
                        other_output = prefill([pair], [states[2]])[0]
                        actual = ttnn.to_torch(other_output).reshape(1, length, 10240).clone()
                        ttnn.deallocate(other_output)
                        value = metrics(case["expected"], actual, case["prompt"])
                        unchanged = all(equal(before, snapshot(state)) for before, state in zip(current, states[:2]))
                        other_snapshot = snapshot(states[2])
                        entry = {
                            **value,
                            "repetition": repetition,
                            "logical_length": length,
                            "seed": case["seed"],
                            "first_device_execution_while_trace_live": repetition == 0,
                            "trace_states_unchanged": unchanged,
                            "repeated_output_equal": None,
                            "repeated_state_equal": None,
                        }
                        literal_passed = metric_passes(value)
                        fresh_literal_checks.append(literal_passed)
                        if args.ple_backend_control:
                            from .probe_ple_public_backend import fresh_backend_gate

                            control = case["backend_control"]
                            matched = metrics(control["output"], actual, case["prompt"])
                            gate = fresh_backend_gate(value, matched, control["sensitivity"])
                            entry["ple_backend_control"] = {
                                "matched_metrics": matched,
                                "sensitivity": control["sensitivity"],
                                "gate": gate,
                            }
                            fresh_acceptance_checks.append(gate["passed"])
                        else:
                            fresh_acceptance_checks.append(literal_passed)
                        if repetition == 0:
                            first_other_outputs[length] = actual
                            first_other_states[length] = other_snapshot
                        else:
                            entry["repeated_output_equal"] = torch.equal(first_other_outputs[length], actual)
                            entry["repeated_state_equal"] = equal(first_other_states[length], other_snapshot)
                            checks.extend((entry["repeated_output_equal"], entry["repeated_state_equal"]))
                        if args.layer == 3:
                            entry["position"] = int(other_snapshot["position"].reshape(-1)[0])
                            checks.append(entry["position"] == length)
                        report["live_other_prefill"].append(entry)
                        if args.device_output_cache is not None:
                            device_readbacks["live_other_prefill"].append(
                                {
                                    "logical_length": length,
                                    "repetition": repetition,
                                    "seed": case["seed"],
                                    "prompt_sha256": hashlib.sha256(
                                        case["prompt"].contiguous().view(torch.uint8).numpy().tobytes()
                                    ).hexdigest(),
                                    "tokens_sha256": hashlib.sha256(
                                        case["tokens"].contiguous().view(torch.uint8).numpy().tobytes()
                                    ).hexdigest(),
                                    "literal_sha256": hashlib.sha256(
                                        case["expected"].contiguous().view(torch.uint8).numpy().tobytes()
                                    ).hexdigest(),
                                    "actual": actual,
                                }
                            )
                            args.device_output_cache.parent.mkdir(parents=True, exist_ok=True)
                            torch.save(device_readbacks, args.device_output_cache)
                        checks.append(unchanged)

        if args.layer == 3:
            report["page_remap"] = []
            for index, state in enumerate(states[:2]):
                restore(state, post_resume[index])
                old = permutations[index].flatten().long()
                changed = permutations[index].roll(1, -1).int()
                addresses = {name: tensor.buffer_address() for name, tensor in buffers(state).items()}
                addresses["page_table"] = state.attention.page_table.buffer_address()
                for name in ("k_cache", "v_cache"):
                    original = post_resume[index][name]
                    remapped = torch.empty_like(original)
                    remapped[changed.flatten().long()] = original[old]
                    target = getattr(state.attention, name)
                    ttnn.copy_host_to_device_tensor(host(remapped, target), target)
                table = state.attention.page_table
                ttnn.copy_host_to_device_tensor(host(changed, table), table)
                assert addresses["page_table"] == table.buffer_address()
                assert all(addresses[name] == tensor.buffer_address() for name, tensor in buffers(state).items())
            refresh(0)
            with runtime_audit():
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            for index, output in enumerate(outputs):
                actual = ttnn.to_torch(output).reshape(1, 1, 10240)
                value = metrics(
                    expected[:, ENDS[index] : ENDS[index] + 1], actual, prompt[:, ENDS[index] : ENDS[index] + 1]
                )
                exact = torch.equal(actual, first_outputs[0][index])
                final = snapshot(states[index])
                old = permutations[index].flatten().long()
                changed = permutations[index].roll(1, -1).flatten().long()
                logical_cache_equal = all(
                    torch.equal(final[name][changed], first_step_states[index][name][old])
                    for name in ("k_cache", "v_cache")
                )
                remaining_equal = all(
                    torch.equal(final[name], first_step_states[index][name])
                    for name in ("position", "raw_history", "compressed_keys")
                )
                report["page_remap"].append(
                    {
                        **value,
                        "same_trace_output_equal": exact,
                        "logical_cache_equal": logical_cache_equal,
                        "indexer_and_position_equal": remaining_equal,
                        "stable_buffer_addresses": True,
                    }
                )
                checks.extend((metric_passes(value), exact, logical_cache_equal, remaining_equal))
        report.update(
            capture_count=1,
            replay_count=4 * len(live_fixtures) + int(args.layer == 3),
            runtime_fallback_audit="clean",
            allocation_tracking=True,
            literal_hf_strict_passed=all(checks) and all(fresh_literal_checks),
            fresh_backend_all_matched_passed=(
                all(
                    metric_passes(value["ple_backend_control"]["matched_metrics"])
                    for value in report["live_other_prefill"]
                )
                if args.ple_backend_control
                else None
            ),
            carried_fresh_contribution_findings=[
                {
                    "logical_length": value["logical_length"],
                    "repetition": value["repetition"],
                    "granularity": value["ple_backend_control"]["gate"]["carried"],
                }
                for value in report["live_other_prefill"]
                if value.get("ple_backend_control", {}).get("gate", {}).get("carried")
            ],
            passed=all(checks) and all(fresh_acceptance_checks),
        )
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


@torch.inference_mode()
def main():
    if "--host-self-check" in sys.argv:
        from .probe_ple_public_backend import host_gate_self_checks

        host_gate_self_checks()
        assert "ttnn" not in sys.modules
        print("Public PLE backend classification host checks passed")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--layer", type=int, choices=(0, 1, 3), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/home/ttuser/qwen38-flash/packed_ple_nf4_absmax"))
    parser.add_argument("--expert-dtype", choices=("bfloat16", "bfloat8_b"), default="bfloat16")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument(
        "--ple-backend-control",
        action="store_true",
        help="Layer1 only: explicitly classify reproduced fresh contribution anomalies using actual packed lookup and one six-projection FP64 control; literal whole/resume/decode remain strict",
    )
    parser.add_argument(
        "--device-output-cache", type=Path, help="Save existing live-prefill CPU readbacks for independent diagnostics"
    )
    args = parser.parse_args()
    if args.ple_backend_control and args.layer != 1:
        parser.error("--ple-backend-control applies only to PLE layer1")
    torch.set_num_threads(8)
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
    config._attn_implementation = "eager"
    reference, weights = load_reference(args.snapshot, args.layer, config)
    fixture = make_fixture(reference, config, args.layer)
    live_fixtures = make_live_fixtures(reference, config, args.layer)
    report = {
        "scope": __doc__,
        "layer": args.layer,
        "expert_dtype": args.expert_dtype,
        "starts": list(STARTS),
        "continuation_lengths": list(COUNTS),
        "HF_continuation_controls": fixture[-1],
        "live_prefill_oracles": [
            {"length": case["length"], "seed": case["seed"], "reference": "independent full HF fresh prefill"}
            for case in live_fixtures
        ],
        "threshold": 0.995,
        "device_executed": not args.cpu_only,
        "snapshot": str(args.snapshot),
        "hf_sha256": hashlib.sha256(Path(inspect.getfile(Qwen4ExpTextDecoderLayer)).read_bytes()).hexdigest(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "passed": False,
        "ple_backend_control_enabled": args.ple_backend_control,
        "literal_hf_strict_passed": None,
        "classification_subject_to_stage_review": args.ple_backend_control,
        "device_output_cache": str(args.device_output_cache) if args.device_output_cache is not None else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if args.ple_backend_control:
            from .probe_ple_public_backend import public_backend_controls

            controls, metadata = public_backend_controls(config, weights, live_fixtures, args.snapshot, args.cache_dir)
            assert len(controls) == len(live_fixtures)
            for case, control in zip(live_fixtures, controls):
                case["backend_control"] = control
            report["ple_backend_control"] = metadata
        if args.cpu_only:
            assert "ttnn" not in sys.modules
        else:
            device_run(args, config, weights, fixture, live_fixtures, report)
    except Exception as exc:
        report["error"] = str(exc).split("backtrace:")[0]
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not args.cpu_only and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
