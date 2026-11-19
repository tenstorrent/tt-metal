# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Real-weight public-generator replay probe; reduced default or --all-layers.

After sourcing lane-env.sh and activating python_env, use an external bound:
 timeout 3600 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_generator_replay_probe

The default context is 1024, layers are 0/3, and capacity is three fixed slots.
Readback programs are warmed before the sole prepare/capture. No workload
program may be added afterward. --section selects a smaller diagnostic surface.
This is serving/replay evidence, not full-model accuracy or qualitative evidence.
"""

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
os.environ.setdefault("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")
assert os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
assert os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.run_prefill_equivalence_probe import tensor_metrics
from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator import QwenGenerator
from models.autoports.qwen_qwen3_6_35b_a3b.tt.model import HF_MODEL, HF_REVISION, QwenModel

ROOT = Path(__file__).resolve().parents[1]


def host(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).clone()


def delta(after, before):
    return {key: after[key] - before[key] for key in before}


def page_mapping(width, slots, logical_ends, steps):
    """Only live logical pages need disjoint ownership in the aggregate pool."""
    budget = width // 3
    rows = torch.stack([torch.arange(width).roll(-slot * budget) for slot in slots]).to(torch.int32)
    used = []
    for row, end in enumerate(logical_ends):
        count = (end + steps + 31) // 32
        assert count <= budget, ("Request exceeds this probe's per-slot physical allocation", count, budget)
        owned = set(rows[row, :count].tolist())
        assert not any(owned & other for other in used), "Live page ownership overlaps"
        used.append(owned)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-layers", action="store_true", help="Run all40 selected decoder layers; default0/3 remains diagnostic"
    )
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4, help="Fixed decode steps after the initial refreshed decode")
    parser.add_argument("--section", choices=("all", "lengths", "mixed", "sampling"), default="all")
    parser.add_argument("--diagnose-mixed", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "doc/full_model/generator_replay_probe/result.json")
    args = parser.parse_args()
    assert args.context >= 1024 and 1 <= args.steps <= 16
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), ROOT / "tests/run_prefill_equivalence_probe.py"] + [
        ROOT / "tt" / name
        for name in (
            "model.py",
            "generator.py",
            "prefill.py",
            "prefill_transport.py",
            "slot_state.py",
            "multichip_decoder.py",
        )
    ]
    result = {
        "status": "RUNNING",
        "command": sys.argv,
        "time_unix": time.time(),
        "model": HF_MODEL,
        "revision": HF_REVISION,
        "layers": list(range(40)) if args.all_layers else [0, 3],
        "context": args.context,
        "max_batch_size": 3,
        "decode_steps_after_initial": args.steps,
        "scope": (
            "All40 serving and replay acceptance" if args.all_layers else "Reduced-stack serving and replay diagnostics"
        )
        + "; not quality, top-k accuracy, or performance",
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
                "TT_METAL_DEVICE_PROFILER",
                "TT_METAL_WATCHER",
            )
        },
        "cases": [],
        "cache_checkpoints": [],
    }

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    torch.set_num_threads(8)
    mesh = generator = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
        model = QwenModel(mesh, layer_indices=None if args.all_layers else [0, 3], max_seq_len=args.context)
        assert list(model.layer_indices) == result["layers"]
        generator = QwenGenerator(model, max_batch_size=3)
        full_states = next(states for states in generator.cache if "page_table" in states[0])
        page_width = full_states[0]["page_table"].shape[-1]
        # Valid tokenizer IDs, deliberately diverse across requests. These are
        # activation fixtures, not answer-quality prompts for the reduced stack.
        fixture_texts = [
            "Explain how supervised learning uses labeled examples. ",
            "A triangle has three sides. Describe its perimeter and area. ",
            "Translate a short description of a garden into French. ",
        ]
        fixture = [generator.tokenizer.encode(text * 150, add_special_tokens=False) for text in fixture_texts]
        assert all(len(ids) > 300 for ids in fixture)
        result["fixture_token_ids"] = [ids[:300] for ids in fixture]
        diagnostic_hidden = []
        diagnostic_moe = []
        if args.diagnose_mixed:
            for layer in model.layers:
                buffers = [model.upload(torch.zeros(1, 1, 2048), ttnn.bfloat16) for _ in range(3)]
                diagnostic_hidden.append(buffers)
                original = layer.decode_batch

                def observe(hidden_states, *, states, positions, original=original, buffers=buffers):
                    values = original(hidden_states, states=states, positions=positions)
                    for value, buffer in zip(values, buffers):
                        ttnn.copy(ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG), buffer)
                    return values

                layer.decode_batch = observe
                taps = {
                    name: [model.upload(torch.zeros(1, 1, 2048), ttnn.bfloat16) for _ in range(3)]
                    for name in ("input", "output", "shared", "grouped")
                }
                diagnostic_moe.append(taps)

                def save_rows(value, name, taps=taps):
                    value = ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG)
                    for row in range(min(3, value.shape[-2])):
                        ttnn.copy(
                            ttnn.to_memory_config(value[:, row : row + 1, :], ttnn.DRAM_MEMORY_CONFIG), taps[name][row]
                        )

                original_moe = layer._moe

                def observe_moe(x, *, original=original_moe, save=save_rows, **kwargs):
                    value = original(x, **kwargs)
                    if x.shape[-2] in (1, 3):
                        save(x, "input")
                        save(value, "output")
                    return value

                layer._moe = observe_moe
                for name, method_name in (("shared", "_shared"), ("grouped", "_grouped_moe")):
                    original_part = getattr(layer, method_name)

                    def observe_part(x, *args, original=original_part, save=save_rows, name=name, **kwargs):
                        value = original(x, *args, **kwargs)
                        if x.shape[-2] <= 32:
                            save(value, name)
                        return value

                    setattr(layer, method_name, observe_part)

        # Exercise all diagnostic host output forms before capture. Bounded
        # prefill prepare is eager-only and runs once here; generator.prepare
        # below is the sole model/sampling trace capture.
        generator.configure_sampling(
            temperature=[0.0, 0.8, 0.7], top_k=[1, 8, 16], top_p=[1.0, 0.9, 0.8], seed=[3, 5, 7]
        )
        generator.prefill_forward(torch.tensor([fixture[0][:33]]), prompt_lens=[33], return_all_logits=True)
        generator.reset()
        generator.prefill_forward(torch.tensor([fixture[0][:31]]), prompt_lens=[31])
        model.host_logits(generator.sampler_logits)
        for tensor in (generator.tokens, *generator.positions, generator.sampler.tt_sampling.seeds_tt_tensor):
            host(tensor)
        generator.configure_sampling()
        generator.prepare(prefill_cases=((32, 0),))
        ttnn.synchronize_device(mesh)
        prepared_entries = int(mesh.num_program_cache_entries())
        captured = dict(generator.traces)
        captured_count = generator.counters["captures"]
        assert captured_count == 2 and set(captured) == {"model_fixed", "sampling"}
        result["prepared_program_cache_entries"] = prepared_entries
        result["captures_after_prepare"] = captured_count
        result["trace_ids"] = {key: int(value) for key, value in captured.items()}
        save()

        def checkpoint(label):
            gc.collect()
            ttnn.synchronize_device(mesh)
            current = int(mesh.num_program_cache_entries())
            row = dict(label=label, entries=current, delta=current - prepared_entries)
            result["cache_checkpoints"].append(row)
            save()
            assert current == prepared_entries, f"New program after prepare: {row}"
            assert generator.traces == captured and generator.counters["captures"] == captured_count, "Trace recaptured"
            print("REPLAY_CHECKPOINT", json.dumps(row), flush=True)

        def snapshot():
            return {
                "tokens": host(generator.tokens).flatten().to(torch.int64).tolist(),
                "positions": [int(host(value).flatten()[0]) for value in generator.positions],
                "seeds": host(generator.sampler.tt_sampling.seeds_tt_tensor).flatten().to(torch.int64).tolist(),
            }

        def prefill(lengths, slots, starts=None, all_logits=False, pages=None):
            starts = [0] * len(slots) if starts is None else list(starts)
            tokens = torch.zeros(len(slots), max(lengths), dtype=torch.int64)
            for row, (slot, length) in enumerate(zip(slots, lengths)):
                tokens[row, :length] = torch.tensor(fixture[slot][:length])
            pages = page_mapping(page_width, slots, lengths, args.steps + 2) if pages is None else pages
            outputs = generator.prefill_forward(
                tokens,
                prompt_lens=lengths,
                start_pos=starts,
                slots=slots,
                page_table=pages,
                return_all_logits=all_logits,
            )
            assert outputs.shape == (
                len(slots),
                max(end - start for end, start in zip(lengths, starts)) if all_logits else 1,
                model.vocab_size,
            )
            assert torch.isfinite(outputs).all()
            return outputs, pages

        def replay(lengths, slots, pages, *, read_each=False, changed_pages=False, inactive=()):
            """Only initial request admission supplies token/position from host."""
            slots = tuple(slots)
            initial = torch.tensor([fixture[slot][length] for slot, length in zip(slots, lengths)])
            generator.decode_forward(initial, lengths, slots=slots, page_table=pages, read_tokens=False)
            diagnostic = []

            def record_step():
                if args.diagnose_mixed:
                    diagnostic.append(
                        dict(
                            logits=model.host_logits(generator.sampler_logits)[0, 0, list(slots)].clone(),
                            recurrent=[host(generator.cache[0][slot]["recurrent"]) for slot in slots],
                            conv=[host(generator.cache[0][slot]["conv"]) for slot in slots],
                            hidden=[[host(buffers[slot]) for slot in slots] for buffers in diagnostic_hidden],
                            moe=[
                                {name: [host(buffers[slot]) for slot in slots] for name, buffers in taps.items()}
                                for taps in diagnostic_moe
                            ],
                        )
                    )

            record_step()
            before_state = snapshot()
            if changed_pages:
                pages = pages.clone()
                pages[:, [-1, -2]] = pages[:, [-2, -1]]
                # The swapped entries are beyond every request's live prefix;
                # no cache migration is required for this metadata-only update.
                assert all((end + args.steps + 32) // 32 < page_width - 2 for end in lengths)
            if inactive:
                # Resident replay ignores stale host positions, including -1.
                # Explicitly update persistent activity once before the loop;
                # row1's just-admitted token/position remain the baseline.
                assert set(inactive) <= set(slots)
                generator._set_active_slots(tuple(slot for slot in slots if slot not in inactive))
                assert generator.active_slots == tuple(sorted(set(slots) - set(inactive)))
            before = dict(generator.counters)
            outputs = []
            for step in range(args.steps):
                order = tuple(reversed(range(len(slots)))) if step % 2 else tuple(range(len(slots)))
                ordered_slots = tuple(slots[row] for row in order)
                positions = [-1 if slot in inactive else 0 for slot in ordered_slots]
                value = generator.decode_forward(
                    None,
                    positions,
                    slots=ordered_slots,
                    page_table=pages[list(order)],
                    refresh_inputs=False,
                    read_tokens=read_each,
                )
                if read_each:
                    by_slot = dict(zip(ordered_slots, value.tolist()))
                    outputs.append([by_slot[slot] for slot in slots])
                record_step()
            counters = delta(generator.counters, before)
            assert counters["token_refreshes"] == counters["position_refreshes"] == 0, counters
            assert counters["model_replays"] == counters["sampling_replays"] == args.steps, counters
            assert counters["readbacks"] == (args.steps if read_each else 0), counters
            expected_page_refreshes = (
                len(slots) * sum("page_table" in states[0] for states in generator.cache) if changed_pages else 0
            )
            assert counters["page_table_refreshes"] == expected_page_refreshes, (counters, expected_page_refreshes)
            # This is the first token/position read after the fixed-step silent
            # loop; it cannot feed any of that loop's device computations.
            after_state = snapshot()
            if diagnostic:
                torch.save(diagnostic, args.output.with_suffix(f".slots_{'_'.join(map(str, slots))}.pt"))
            for slot, length in zip(slots, lengths):
                advances = 0 if slot in inactive else args.steps
                assert before_state["positions"][slot] == length + 1
                assert after_state["positions"][slot] == length + 1 + advances
                assert after_state["seeds"][slot] == before_state["seeds"][slot] + advances
                if slot in inactive:
                    assert after_state["tokens"][slot] == before_state["tokens"][slot]
            return dict(
                before=before_state,
                after=after_state,
                loop_counters=counters,
                tokens_by_step=outputs,
                slots=list(slots),
                inactive_slots=list(inactive),
            )

        if args.section in ("all", "lengths"):
            for length, start in [(n, 0) for n in (1, 31, 33, 63, 65, 255, 257)] + [
                (33, 31),
                (64, 31),
                (257, 255),
                (288, 255),
            ]:
                repeated = []
                reference_logits = None
                for repetition in range(2):
                    generator.configure_sampling()
                    generator.reset()
                    if start:
                        prefill([start], [0])
                    logits, pages = prefill([length], [0], [start], all_logits=length == 33 and start == 0)
                    if reference_logits is None:
                        reference_logits = logits.clone()
                    else:
                        assert torch.equal(reference_logits, logits), "Repeated public prefill changed logits"
                    repeated.append(replay([length], [0], pages, read_each=bool(repetition)))
                    checkpoint(f"length={length}/start={start}/repeat={repetition}")
                assert repeated[0]["after"] == repeated[1]["after"], "Silent feedback differs from readback-only replay"
                assert repeated[1]["tokens_by_step"][-1][0] == repeated[0]["after"]["tokens"][0]
                result["cases"].append(
                    dict(
                        kind="public_length",
                        length=length,
                        start=start,
                        repeated_logits_exact=True,
                        deterministic_feedback_exact=True,
                        runs=repeated,
                    )
                )
                save()

        if args.section in ("all", "mixed"):
            lengths, slots = [31, 65, 257], [0, 1, 2]
            solo = []
            for slot, length in zip(slots, lengths):
                generator.configure_sampling()
                generator.reset()
                logits, pages = prefill([length], [slot])
                run = replay([length], [slot], pages, read_each=True)
                solo.append((logits.clone(), run))
                checkpoint(f"mixed_reference/slot={slot}")
            mixed_runs = []
            for changed in (False, True):
                generator.configure_sampling()
                generator.reset()
                logits, pages = prefill(lengths, slots)
                comparisons = []
                for row, (reference, _) in enumerate(solo):
                    metric = tensor_metrics(reference, logits[row : row + 1], logits=True)
                    comparisons.append(metric)
                    assert (
                        metric["finite"] and metric["pcc"] >= 0.99999 and metric["top5"] == metric["top100"] == 1.0
                    ), metric
                run = replay(lengths, slots, pages, changed_pages=changed, read_each=True)
                result["mixed_diagnostic"] = dict(solo=[entry[1] for entry in solo], mixed=run)
                save()
                for slot, (_, reference_run) in enumerate(solo):
                    assert (
                        run["after"]["tokens"][slot] == reference_run["after"]["tokens"][slot]
                    ), "Mixed versus solo final greedy token changed"
                    assert run["after"]["positions"][slot] == reference_run["after"]["positions"][slot]
                mixed_runs.append(run)
                result["cases"].append(
                    dict(kind="mixed_vs_solo", changed_unused_pages=changed, logits=comparisons, replay=run)
                )
                checkpoint(f"mixed/changed_pages={changed}")
            assert mixed_runs[0]["after"] == mixed_runs[1]["after"], "Changed unused page entries changed output"
            generator.reset()
            _, pages = prefill(lengths, slots)
            inactive = replay(lengths, slots, pages, inactive=(1,))
            result["cases"].append(dict(kind="inactive_middle_row", replay=inactive))
            checkpoint("inactive_middle_row")

        if args.section in ("all", "sampling"):
            lengths, slots = [31, 33, 65], [0, 1, 2]
            runs = []
            for variant in ("baseline", "repeat", "change_slot_1"):
                generator.configure_sampling(
                    temperature=[0.0, 0.8, 0.7], top_k=[1, 8, 16], top_p=[1.0, 0.9, 0.8], seed=[11, 23, 37]
                )
                generator.reset()
                _, pages = prefill(lengths, slots)
                if variant == "change_slot_1":
                    before_seed = host(generator.sampler.tt_sampling.seeds_tt_tensor).flatten().clone()
                    before_parameters = {key: list(value) for key, value in generator.sampling_values.items()}
                    generator.configure_sampling(temperature=1.2, top_k=4, top_p=0.7, seed=101, slots=[1])
                    after_seed = host(generator.sampler.tt_sampling.seeds_tt_tensor).flatten()
                    assert torch.equal(before_seed[[0, 2]], after_seed[[0, 2]])
                    for name, values in before_parameters.items():
                        assert [generator.sampling_values[name][s] for s in (0, 2)] == [values[s] for s in (0, 2)]
                run = replay(lengths, slots, pages, read_each=True)
                runs.append(run)
                result["cases"].append(dict(kind="per_row_sampling", variant=variant, replay=run))
                checkpoint(f"sampling/{variant}")
            assert (
                runs[0]["tokens_by_step"] == runs[1]["tokens_by_step"] and runs[0]["after"] == runs[1]["after"]
            ), "Matched seeds are nondeterministic"
            for step in range(args.steps):
                for slot in (0, 2):
                    assert (
                        runs[0]["tokens_by_step"][step][slot] == runs[2]["tokens_by_step"][step][slot]
                    ), "Changing slot 1 sampling affected another row"
            result["sampling_isolation"] = {
                "unchanged_slots": [0, 2],
                "changed_slot": 1,
                "seed_and_parameter_isolation": True,
                "output_isolation": True,
                "changed_slot_output_differed": any(
                    a[1] != b[1] for a, b in zip(runs[0]["tokens_by_step"], runs[2]["tokens_by_step"])
                ),
            }
        checkpoint("final")
        result["status"] = "PASS"
        result["counters"] = dict(generator.counters)
        print("GENERATOR_REPLAY_PROBE_OK", flush=True)
    except BaseException:
        result["status"] = "FAIL"
        result["error"] = traceback.format_exc()
        traceback.print_exc()
        raise
    finally:
        if generator is not None:
            result["counters_before_release"] = dict(generator.counters)
            generator.close()
            result["counters_after_release"] = dict(generator.counters)
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        result["elapsed_seconds"] = time.time() - result["time_unix"]
        save()


if __name__ == "__main__":
    main()
