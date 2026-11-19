# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced adapter regression for mixed unknown-to-known prefill horizons.

Run serially under an external timeout. Sampler snapshots are diagnostic reads;
they never supply tokens or state to the model. This is not a performance run.
"""

import argparse
import copy
import json
import os
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator_vllm import OrnithServingDiagnostic
from models.autoports.ornith_ai_ornith_1_5_9b.tt.prefill_plan import plan_prefill


def sampler_snapshot(gen):
    sampler = gen.sampler
    fields = ("prompt_mask", "output_mask", "output_counts", "output_counts_gathered")
    manager = sampler.seed_manager
    return {
        "seeds": [
            ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(sampler.tt_sampling.seeds_tt_tensor)
        ],
        "tables": {
            name: [
                ttnn.to_torch(shard).reshape(32, -1)[:3].clone()
                for shard in ttnn.get_device_tensors(getattr(sampler.tt_penalties, name))
            ]
            for name in fields
        },
        "host_seeds": {
            name: copy.deepcopy(getattr(manager, name)) for name in ("seeds", "seed_counters", "seed_salts")
        },
        "host_rngs": [rng.getstate() for rng in manager.rngs],
        "prompt_shadow": copy.deepcopy(sampler.tt_penalties._prompt_tokens_host),
    }


def assert_sampler_equal(before, after):
    assert all(torch.equal(a, b) for a, b in zip(before["seeds"], after["seeds"], strict=True))
    for name in before["tables"]:
        assert all(torch.equal(a, b) for a, b in zip(before["tables"][name], after["tables"][name], strict=True)), name
    for name in ("host_seeds", "host_rngs"):
        assert before[name] == after[name], name
    if before["prompt_shadow"] is None:
        assert after["prompt_shadow"] is None
    else:
        assert torch.equal(before["prompt_shadow"], after["prompt_shadow"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "layers": [0, 3],
        "batch_capacity": 3,
        "context": 4096,
        "cache_owner": "adapter allocation passed explicitly",
        "decode_trace": True,
        "scope": "Correctness only; diagnostic sampler reads cover all seed words and all four history tables for slots 0,1,2 on every shard.",
        "cases": [],
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    adapter = None
    try:
        hf_config = AutoConfig.from_pretrained(
            os.environ.get("HF_MODEL", "ornith-ai/Ornith-1.5-9B"),
            revision=os.environ.get("HF_REVISION", "489cb97981b8654bcfcf30ce1f94ed1b62e07b53"),
            local_files_only=True,
        )
        adapter = OrnithServingDiagnostic.initialize_vllm_model(hf_config, mesh, 3, 4096)
        cache = adapter.allocate_kv_cache_per_layer([((384, 1, 32, 256), torch.bfloat16, i) for i in range(8)])
        gen = adapter.generator
        gen.page_table = torch.arange(384, dtype=torch.int32).reshape(3, 128)
        gen.bind_serving_cache(cache, page_table=gen.page_table)
        adapter.warmup_model_prefill(cache, enable_trace=False, can_sample_on_device=True)
        adapter.warmup_model_decode(
            cache, enable_trace=True, max_batch_size=3, num_blocks=384, can_sample_on_device=True
        )
        assert cache is gen.cache
        sampler_calls = []

        def forbid_sampling(*_args, **_kwargs):
            sampler_calls.append("unexpected")
            raise AssertionError("Host/intermediate prefill changed sampler history, seeds, or invoked sampling")

        def step(label, ends, originals, starts, expected_rebuilds, expected_host_reads):
            width = max(ends)
            tokens = torch.stack([(torch.arange(width) % 13 + 9707 + row).long() for row in range(len(ends))])
            before = sampler_snapshot(gen)
            host_reads = []
            host_logits = gen.model.host_logits

            def read_logits(logits):
                host_reads.append(tuple(logits.shape))
                return host_logits(logits)

            with (
                patch.object(gen.model, "host_logits", side_effect=read_logits),
                patch.object(gen, "_sample_logits", side_effect=forbid_sampling),
                patch.object(gen.sampling_state, "reset_history", side_effect=forbid_sampling),
                patch.object(gen.sampling_state, "reset_seed_values", side_effect=forbid_sampling),
            ):
                result = adapter.prefill_forward(
                    tokens,
                    page_table=gen.page_table[: len(ends)],
                    kv_cache=cache,
                    prompt_lens=ends,
                    original_prompt_lens=originals,
                    start_pos=starts,
                    empty_slots=list(range(len(ends))),
                )
            if adapter.uses_mrope:
                result, deltas = result
                assert torch.count_nonzero(deltas) == 0
            assert tuple(result.shape) == (len(ends), 1, gen.model.vocab_size)
            assert torch.isfinite(result).all()
            discarded = [0 < end < original for end, original in zip(ends, originals, strict=True)]
            for row, discard in enumerate(discarded):
                if discard:
                    assert torch.count_nonzero(result[row]) == 0
            assert len(host_reads) == expected_host_reads, host_reads
            assert_sampler_equal(before, sampler_snapshot(gen))
            event = gen.prefill_contract.events[-1]
            assert event["rebuilt_prefix_rows"] == expected_rebuilds, event
            known = all(end <= original for end, original in zip(ends, originals, strict=True))
            horizons = originals if known else ends
            metadata = []
            for slot, (start, end, horizon) in enumerate(zip(starts, ends, horizons, strict=True)):
                bound = gen.prefill_runtime.bindings[(id(cache), slot)]
                assert bound["end"] == end and bound["total_length"] == horizon
                plans = []
                for index, layer in enumerate(gen.model.layers):
                    plan = plan_prefill(total_length=horizon, start=start, end=end, kind=layer.kind)
                    assert bound["layers"][index]["history"]["boundary"] == plan.segments[-1].save_boundary_before
                    plans.append({"kind": layer.kind, "profiles": [asdict(s.profile_tag) for s in plan.segments]})
                metadata.append({"slot": slot, "end": end, "horizon": horizon, "plans": plans})
            report["cases"].append(
                {
                    "label": label,
                    "status": "pass",
                    "starts": starts,
                    "ends": ends,
                    "originals": originals,
                    "discard_logits": discarded,
                    "host_logits_readbacks": len(host_reads),
                    "rebuilt_prefix_rows": event["rebuilt_prefix_rows"],
                    "sampler_exactly_unchanged": True,
                    "metadata": metadata,
                }
            )
            save()
            return result

        step("profile_change_initial_mixed", [512, 130], [4096, 129], [0, 0], [], 1)
        step("profile_change_solo_known", [1024], [4096], [512], [0], 0)
        step("missing_tail_initial_mixed", [129, 130], [200, 129], [0, 0], [], 1)
        step("missing_tail_solo_known", [160], [200], [129], [0], 0)
        # A same-generator legacy control computes each suppressed head. Compare
        # terminal logits, following traced decode, complete model state and
        # sampler state against the skipped-head path, including prefix rebuilds.
        state = gen.states[(id(cache), (0, 1, 2))]
        trace_baseline = (dict(state.model_traces), dict(state.sample_traces))
        counters_before = dict(gen.counters)
        programs_before = mesh.num_program_cache_entries()

        def model_snapshot():
            values, seen = [], set()
            for layer_states in cache:
                for slot_state in layer_states:
                    for field in ("conv", "recurrent", "key", "value"):
                        tensor = getattr(slot_state, field)
                        if tensor is not None and id(tensor) not in seen:
                            seen.add(id(tensor))
                            values.extend(ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(tensor))
            return values

        for label, first_end, second_end, original in (
            ("profile", 512, 1024, 2049),
            ("tail", 129, 160, 200),
        ):
            controls = []
            for legacy in (True, False):
                runtime_prefill = gen.prefill_runtime.prefill
                head = gen.prefill_runtime._head
                head_calls = []

                def control_prefill(*args, **kwargs):
                    if legacy and kwargs.get("discard_logits"):
                        kwargs["discard_logits"] = False
                    return runtime_prefill(*args, **kwargs)

                def count_head(*args, **kwargs):
                    head_calls.append((args[1], args[2]))
                    return head(*args, **kwargs)

                before_sampler = sampler_snapshot(gen)
                with (
                    patch.object(gen.prefill_runtime, "prefill", side_effect=control_prefill),
                    patch.object(gen.prefill_runtime, "_head", side_effect=count_head),
                ):
                    step(label + "_first", [first_end, 130], [original, 129], [0, 0], [], 1)
                    step(label + "_resume", [second_end], [original], [first_end], [0], 0)
                    terminal = step(label + "_terminal", [original], [original], [second_end], [], 1)
                # Diagnostic host argmax chooses an identical control input;
                # this probe does not measure or introduce host serving sampling.
                token = int(torch.argmax(terminal[0, 0]))
                following = gen.decode_forward(
                    torch.tensor([[token], [0], [0]]),
                    [original, -1, -1],
                    page_table=gen.page_table,
                    kv_cache=cache,
                    sampling_params=None,
                    reload_inputs=True,
                    read_from_device=True,
                )
                assert_sampler_equal(before_sampler, sampler_snapshot(gen))
                controls.append(
                    {
                        "terminal": terminal.clone(),
                        "following": following.clone(),
                        "model": model_snapshot(),
                        "progress": copy.deepcopy(gen._sampling_progress),
                        "heads": len(head_calls),
                    }
                )
                assert (dict(state.model_traces), dict(state.sample_traces)) == trace_baseline
                assert mesh.num_program_cache_entries() == programs_before
            control, optimized = controls
            assert torch.equal(control["terminal"], optimized["terminal"])
            assert torch.equal(control["following"], optimized["following"])
            assert len(control["model"]) == len(optimized["model"])
            assert all(torch.equal(a, b) for a, b in zip(control["model"], optimized["model"], strict=True))
            assert control["progress"] == optimized["progress"]
            assert control["heads"] == 5 and optimized["heads"] == 2
            report["cases"].append(
                {
                    "label": label + "_skip_vs_legacy_control",
                    "status": "pass",
                    "legacy_head_calls": control["heads"],
                    "optimized_head_calls": optimized["heads"],
                    "terminal_and_traced_decode_bit_equal": True,
                    "all_model_state_bit_equal": True,
                    "sampler_state_and_progress_unchanged": True,
                    "trace_ids_and_program_count_unchanged": True,
                }
            )
            save()
        assert gen.counters["captures"] == counters_before["captures"]
        assert gen.counters["releases"] == counters_before["releases"]
        assert not sampler_calls
        report["status"] = "pass"
    except BaseException as exc:
        report["status"] = "fail"
        report["error"] = repr(exc)
        raise
    finally:
        save()
        if adapter is not None and adapter.generator is not None:
            adapter.generator.close()
        ttnn.close_mesh_device(mesh)
    print("PREFILL_HORIZON_PROBE_PASS", args.output, flush=True)


if __name__ == "__main__":
    main()
