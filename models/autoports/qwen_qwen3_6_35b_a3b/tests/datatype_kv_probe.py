# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reduced real-weight KV-cache controls: one device policy per process, CPU comparison.

Both lengths32 and33 use the leading AIME chat-template tokens, followed by
three teacher-forced tokens from that same prompt. These are KV sensitivity
probes, not full-model accuracy or performance evidence.
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def run(args):
    import torch

    from models.autoports.qwen_qwen3_6_35b_a3b.tests.run_full_model_batched_teacher import load_fixture

    assert os.environ.get("TT_METAL_DEVICE_PROFILER", "0") != "1"
    os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
    os.environ.setdefault("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")
    assert os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
    assert os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"
    entry, provenance = load_fixture(ROOT / "readiness_aime24_chat.refpt")
    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator import build_generator

    torch.set_num_threads(8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(
        status="running",
        command=sys.argv,
        time_unix=time.time(),
        config_path=str(args.precision_config.resolve()),
        layers=[0, 3],
        context=4096,
        batch=32,
        mesh=[1, 4],
        trace_allocation_tracking=1,
        skip_program_cache=0,
        reference=provenance,
        scope="Reduced real layers: KV sensitivity only; no full-model accuracy or speed claim",
        teacher_forcing="Continuation uses following AIME prompt tokens, identical in all32 users",
        cases=[],
    )
    mesh = generator = None
    tensor_cases = {}
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
        generator = build_generator(
            ROOT,
            mesh,
            layer_indices=[0, 3],
            max_seq_len=4096,
            max_batch_size=32,
            host_sampling=False,
            precision_config=str(args.precision_config.resolve()),
        )
        model, cache = generator.model, generator.cache
        assert tuple(model.layer_indices) == (0, 3)
        result["runtime_precision"] = model.precision_summary()
        full_states = next(states for states in cache if "page_table" in states[0])
        width = int(full_states[0]["page_table"].shape[-1])
        pages_per_slot = 2  # ceil((33+3)/32), including the awkward tail.
        assert pages_per_slot * 32 <= width
        pages = torch.stack([torch.arange(width).roll(-slot * pages_per_slot) for slot in range(32)]).int()
        live_pages = pages[:, :pages_per_slot].flatten().tolist()
        assert len(set(live_pages)) == 64
        result["physical_pages_by_slot"] = pages[:, :pages_per_slot].tolist()
        result["aggregate_page_pool_size"] = width
        slots = tuple(range(32))
        # Warm the two relevant prefill lengths and diagnostic host readbacks
        # before any live trace protects persistent allocations.
        for length in (32, 33):
            generator.prefill_forward(torch.ones((1, length), dtype=torch.int32), prompt_lens=[length])
        model.host_logits(generator.sampler_logits)
        for position in generator.positions:
            ttnn.to_torch(ttnn.get_device_tensors(position)[0])
        generator.prepare(prefill_cases=((32, 0), (33, 0)))
        generator.reset()
        assert generator.cache is cache and not generator.host_sampling
        assert "model_fixed" in generator.traces
        baseline_programs = int(mesh.num_program_cache_entries())
        result["trace_ids"] = {key: int(value) for key, value in generator.traces.items()}
        for length in (32, 33):
            generator.reset()
            initial = dict(generator.counters)
            prompt = entry.prompt_tokens[:, :length].repeat(32, 1).int()
            forced_ids = entry.prompt_tokens[0, length : length + 3].tolist()
            collected = []

            def observe(logits):
                rows = logits.reshape(32, -1, model.vocab_size)
                assert rows.shape[1] == 1 and torch.isfinite(rows).all()
                collected.append(rows[:, 0].detach().cpu().clone())

            observe(
                generator.prefill_forward(
                    prompt,
                    prompt_lens=[length] * 32,
                    start_pos=[0] * 32,
                    page_table=pages,
                    kv_cache=cache,
                    slots=slots,
                )
            )
            decode_initial = dict(generator.counters)
            for offset, token in enumerate(forced_ids):
                observe(
                    generator.decode_forward(
                        torch.full((32, 1), int(token), dtype=torch.int32),
                        [length + offset] * 32,
                        slots=slots,
                        kv_cache=cache,
                        refresh_inputs=True,
                        read_tokens=False,
                        return_logits=True,
                    )
                )
            ttnn.synchronize_device(mesh)
            delta = {key: generator.counters[key] - value for key, value in initial.items()}
            decode_delta = {key: generator.counters[key] - value for key, value in decode_initial.items()}
            final_positions = [
                int(ttnn.to_torch(ttnn.get_device_tensors(position)[0]).flatten()[0])
                for position in generator.positions
            ]
            assert final_positions == [length + 3] * 32
            assert delta["captures"] == 0
            assert {key: int(value) for key, value in generator.traces.items()} == result["trace_ids"]
            assert decode_delta["model_replays"] == 3
            assert decode_delta["sampling_replays"] == 0
            assert decode_delta["readbacks"] == 3
            assert decode_delta["page_table_refreshes"] == 0
            assert int(mesh.num_program_cache_entries()) == baseline_programs, "Unwarmed programs after prepare"
            tensor_cases[str(length)] = torch.stack(collected)
            result["cases"].append(
                dict(
                    prompt_length=length,
                    prompt_token_ids=prompt[0].tolist(),
                    teacher_forced_ids=forced_ids,
                    logits_shape=list(tensor_cases[str(length)].shape),
                    final_positions=final_positions,
                    counter_delta=delta,
                    decode_counter_delta=decode_delta,
                    program_cache_entries=int(mesh.num_program_cache_entries()),
                    argmax=tensor_cases[str(length)].argmax(-1).tolist(),
                )
            )
            save_json(args.output.with_suffix(".json"), result)
            print(f"KV_PROBE_CASE_OK prompt={length} batch=32 traced_decode=3", flush=True)
        torch.save({"cases": tensor_cases, "config_path": result["config_path"]}, args.output)
        result["tensor_path"] = str(args.output)
        result["program_cache_entries_after_prepare"] = baseline_programs
        result["status"] = "pass"
    except BaseException:
        result["status"] = "fail"
        result["error"] = traceback.format_exc()
        raise
    finally:
        result["elapsed_seconds"] = time.time() - result["time_unix"]
        save_json(args.output.with_suffix(".json"), result)
        if generator is not None and not generator.closed:
            generator.close()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


def compare(args):
    import torch

    torch.set_num_threads(8)
    control = torch.load(args.control, map_location="cpu", weights_only=True)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=True)
    control_meta = json.loads(args.control.with_suffix(".json").read_text())
    candidate_meta = json.loads(args.candidate.with_suffix(".json").read_text())
    assert control_meta["status"] == candidate_meta["status"] == "pass"
    control_policy = control_meta["runtime_precision"]["selected_policy"]
    candidate_policy = candidate_meta["runtime_precision"]["selected_policy"]
    assert control_policy["kv_cache_dtype"] == "bf16", "The control must use BF16 cache"
    for field in control_policy:
        if field not in ("config_id", "kv_cache_dtype"):
            assert control_policy[field] == candidate_policy[field], f"Non-KV policy differs: {field}"
    assert control["cases"].keys() == candidate["cases"].keys() == {"32", "33"}
    assert control_meta["physical_pages_by_slot"] == candidate_meta["physical_pages_by_slot"]
    for left, right in zip(control_meta["cases"], candidate_meta["cases"]):
        for field in ("prompt_length", "prompt_token_ids", "teacher_forced_ids", "logits_shape"):
            assert left[field] == right[field], field

    def pcc(left, right):
        left, right = left.double().flatten(), right.double().flatten()
        assert torch.isfinite(left).all() and torch.isfinite(right).all()
        if torch.equal(left, right):
            return 1.0
        left, right = left - left.mean(), right - right.mean()
        return float((left @ right) / (left.norm() * right.norm()))

    rows = []
    aggregates = []
    for length, ref in control["cases"].items():
        got = candidate["cases"][length]
        assert ref.shape == got.shape and ref.shape[:2] == (4, 32)
        for step in range(4):
            aggregates.append(dict(prompt_length=int(length), step=step, pcc=pcc(ref[step], got[step])))
            for slot in range(32):
                cross = pcc(ref[step, slot], got[step, slot])
                same_control = pcc(ref[step, 0], ref[step, slot])
                same_candidate = pcc(got[step, 0], got[step, slot])
                top1 = int(ref[step, slot].argmax()) == int(got[step, slot].argmax())
                same_top1 = int(ref[step, 0].argmax()) == int(ref[step, slot].argmax()) and int(
                    got[step, 0].argmax()
                ) == int(got[step, slot].argmax())
                rows.append(
                    dict(
                        prompt_length=int(length),
                        step=step,
                        slot=slot,
                        control_vs_candidate_pcc=cross,
                        control_vs_candidate_top1=top1,
                        control_same_input_pcc=same_control,
                        candidate_same_input_pcc=same_candidate,
                        same_input_top1=same_top1,
                        passed=min(cross, same_control, same_candidate) >= 0.99 and top1 and same_top1,
                    )
                )
    passed = all(row["passed"] for row in rows) and all(row["pcc"] >= 0.995 for row in aggregates)
    result = dict(
        status="pass" if passed else "fail",
        command=sys.argv,
        control=str(args.control),
        candidate=str(args.candidate),
        thresholds={"per_user_pcc": 0.99, "aggregate_pcc": 0.995, "top1": "exact"},
        aggregates=aggregates,
        per_user=rows,
    )
    save_json(args.output, result)
    print(f"KV_COMPARE {result['status']} min_per_user={min(row['control_vs_candidate_pcc'] for row in rows):.8f}")
    assert passed, f"KV sensitivity gate failed; see {args.output}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--precision-config", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True, help="Full-logit .pt output; JSON saved alongside")
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("--control", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "run":
        assert args.output.suffix == ".pt"
        run(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
