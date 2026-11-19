# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""B3 explicit-command integration probe; reduced default or --all-layers.

Source-only at creation. Use an external timeout after sourcing lane-env.sh:
 timeout 3600 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_serving_commands_probe
This verifies command/state/trace mechanics, not full-model output accuracy.
Add --adapter to exercise the actual vLLM adapter, its cache allocation and
sampling-parameter translation. This remains a direct device probe, not an
allocator-driven serving/page-growth test.
"""

import argparse
import gc
import hashlib
import json
import os
import traceback
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
os.environ.setdefault("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")
assert os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
assert os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator import QwenGenerator
from models.autoports.qwen_qwen3_6_35b_a3b.tt.model import QwenModel

ROOT = Path(__file__).resolve().parents[1]


def host(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).clone()


def local_bits(tensor):
    result = []
    for local in ttnn.get_device_tensors(tensor):
        value = ttnn.to_torch(local).contiguous()
        result.append(value.view(torch.int16 if value.dtype == torch.bfloat16 else torch.int32).clone())
    assert len(result) == 4
    return result


class StaleInputs:
    def __getattribute__(self, name):
        raise AssertionError("Resident call inspected stale host inputs: " + name)

    def __len__(self):
        raise AssertionError("Resident call inspected stale input length")


def build_adapter_probe(adapter_class, mesh, args):
    """Use the real factory and external-cache hook on the diagnostic layer set."""
    previous = os.environ.get("QWEN_VLLM_REDUCED")
    os.environ["QWEN_VLLM_REDUCED"] = "0" if args.all_layers else "1"
    try:
        adapter = adapter_class.initialize_vllm_model(
            SimpleNamespace(
                model_type="qwen3_5_moe", _name_or_path=None if args.model_dir is None else str(args.model_dir)
            ),
            mesh,
            3,
            args.context,
        )
        specs = [
            (tuple(layer["default_shape"]), layer["dtype"], layer["tensor_idx"])
            for layer in adapter.model.cache_spec()["layers"]
        ]
        cache = adapter.allocate_kv_cache_per_layer(specs)
    finally:
        if previous is None:
            os.environ.pop("QWEN_VLLM_REDUCED", None)
        else:
            os.environ["QWEN_VLLM_REDUCED"] = previous
    assert adapter.generator.cache is cache
    assert adapter.decode_input_update_contract == 1
    return adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-layers", action="store_true", help="Run all40 selected decoder layers; default0/3 remains diagnostic"
    )
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--adapter", action="store_true", help="Route serving calls through generator_vllm.py")
    parser.add_argument(
        "--wide-sampling", action="store_true", help="Exercise retained narrow/wide sampling transitions"
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / (
            "readiness_vllm/adapter_commands_probe/result.json"
            if args.adapter
            else "doc/full_model/serving_commands_probe/result.json"
        )
    assert args.context >= 1024
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__)] + [
        ROOT / "tt" / name for name in ("model.py", "generator.py", "serving.py", "serving_state.py", "slot_state.py")
    ]
    if args.adapter:
        paths.append(ROOT / "tt/generator_vllm.py")
    result = dict(
        status="RUNNING",
        layers=list(range(40)) if args.all_layers else [0, 3],
        batch_capacity=3,
        context=args.context,
        tracker=True,
        skip_program_cache=False,
        interface="vllm_adapter" if args.adapter else "generator_v1",
        page_refresh_scope="changed unused tail plus unchanged-table control; not allocator-driven page growth",
        scope="all40 real-weight command integration" if args.all_layers else "reduced real-weight command integration",
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        steps=[],
        cache_checkpoints=[],
    )

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    mesh = generator = adapter = None
    try:
        torch.set_num_threads(8)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
        if args.adapter:
            from vllm_tt_plugin.model_input import TTSamplingParams

            from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator_vllm import QwenForCausalLM

            adapter = build_adapter_probe(QwenForCausalLM, mesh, args)
            model, generator = adapter.model, adapter.generator
            result["adapter_class"] = f"{type(adapter).__module__}:{type(adapter).__name__}"
            result["sampling_parameter_type"] = f"{TTSamplingParams.__module__}:{TTSamplingParams.__name__}"
            result["cache_owner"] = "adapter_external_allocation"
        else:
            model = QwenModel(
                mesh,
                checkpoint=args.model_dir,
                layer_indices=None if args.all_layers else [0, 3],
                max_seq_len=args.context,
            )
            generator = QwenGenerator(model, max_batch_size=3)
        assert list(model.layer_indices) == result["layers"]
        interface = adapter if adapter is not None else generator

        def decode(tokens, positions, **kwargs):
            callback = adapter.decode_forward if adapter is not None else generator.decode_forward_v1
            return callback(tokens, positions, kv_cache=generator.cache, **kwargs)

        def sampling_params(**values):
            if adapter is None:
                return values
            return TTSamplingParams(**values, num_logprobs=[-2] * 3, enable_log_probs=[False] * 3)

        full = next(group for group in generator.cache if "page_table" in group[0])
        linear = next(group for group in generator.cache if "recurrent" in group[0])
        width = full[0]["page_table"].shape[-1]
        pages = torch.stack([torch.arange(width).roll(-slot * (width // 3)) for slot in range(3)]).int()
        fixtures = [
            generator.tokenizer.encode(text * 100, add_special_tokens=False)
            for text in (
                "A triangle has three sides. ",
                "A garden contains trees and flowers. ",
                "A computer executes a sequence of instructions. ",
            )
        ]
        lengths = [33, 47, 65]
        input_tokens = torch.zeros((3, max(lengths)), dtype=torch.int64)
        for slot, length in enumerate(lengths):
            input_tokens[slot, :length] = torch.tensor(fixtures[slot][:length])
        # Warm host output/read forms and all helper state before first capture.
        generator.prefill_forward(torch.tensor([fixtures[0][:33]]), prompt_lens=[33])
        for tensor in (generator.tokens, *generator.positions, generator.sampler.tt_sampling.seeds_tt_tensor):
            host(tensor)
        for state in linear:
            for key in ("recurrent", "conv"):
                local_bits(state[key])
        generator.configure_sampling(
            temperature=[0.0, 0.7, 0.8], top_k=[1, 8, 16], top_p=[1.0, 0.9, 0.8], seed=[11, 22, 33]
        )
        generator.configure_sampling()
        if adapter is not None:
            adapter.warmup_model_prefill(kv_cache=generator.cache, can_sample_on_device=True, enable_trace=False)
            adapter.warmup_model_decode(
                kv_cache=generator.cache,
                max_batch_size=3,
                num_blocks=width,
                can_sample_on_device=True,
                enable_trace=True,
            )
        else:
            generator.prepare(prefill_cases=((32, 0),))
        ttnn.synchronize_device(mesh)
        baseline = int(mesh.num_program_cache_entries())
        captured = dict(generator.traces)
        result["prepared_program_entries"] = baseline
        result["trace_ids"] = {key: int(value) for key, value in captured.items()}

        def checkpoint(label):
            gc.collect()
            ttnn.synchronize_device(mesh)
            entries = int(mesh.num_program_cache_entries())
            result["cache_checkpoints"].append(dict(label=label, entries=entries, delta=entries - baseline))
            save()
            assert entries == baseline, (label, baseline, entries)
            assert generator.traces == captured, "A retained trace was replaced"
            print("SERVING_COMMANDS_CHECKPOINT", label, entries, flush=True)

        def state():
            return dict(
                tokens=host(generator.tokens).reshape(-1).to(torch.int64),
                seeds=host(generator.sampler.tt_sampling.seeds_tt_tensor).reshape(-1).to(torch.int64),
                positions=torch.tensor([int(host(p).reshape(-1)[0]) for p in generator.positions]),
            )

        def counters():
            return dict(generator.counters)

        def difference(before):
            return {key: generator.counters[key] - before[key] for key in before}

        def finish(handle, is_tokens):
            returned, events = interface.read_decode_output(handle, async_read=True)
            assert returned is handle and events
            for event in events:
                ttnn.event_synchronize(event)
            return interface.process_decode_output_host(returned, is_tokens=is_tokens)

        def record(name, **fields):
            result["steps"].append(dict(name=name, **fields))
            save()
            checkpoint(name)

        params = sampling_params(
            temperature=[0.0, 0.0, 0.0], top_k=[1, 1, 1], top_p=[1.0, 1.0, 1.0], seed=[101, 202, 303]
        )
        prefill_kwargs = dict(prompt_lens=lengths, page_table=pages, sampling_params=params, enable_trace=False)
        if adapter is not None:
            prefill_output, deltas = adapter.prefill_forward(
                input_tokens, start_pos=[0, 0, 0], empty_slots=[0, 1, 2], kv_cache=generator.cache, **prefill_kwargs
            )
            assert deltas.tolist() == [0, 0, 0]
        else:
            prefill_handle = generator.prefill_forward(
                input_tokens, slots=[0, 1, 2], read_from_device=False, **prefill_kwargs
            )
            prefill_output = finish(prefill_handle, True)
        prefill_state = state()
        assert prefill_state["positions"].tolist() == lengths, "Prefill sampling advanced positions"
        assert torch.equal(prefill_output[:, 0], prefill_state["tokens"][:3])
        record("prefill_traced_sampling", output=prefill_output[:, 0].tolist(), positions=lengths)
        rows = (2, 0, 1)
        first = decode(
            prefill_output[list(rows)],
            [lengths[s] for s in rows],
            slots=rows,
            page_table=pages[list(rows)],
            sampling_params=params,
            reload_sampling_params=True,
            reset_sampling_state=True,
            read_from_device=False,
        )
        first_state = state()
        output = finish(first, True)
        assert torch.equal(output[:, 0], first_state["tokens"][list(rows)])
        assert first_state["positions"].tolist() == [length + 1 for length in lengths]
        record("admit_unsorted_slots", rows=list(rows), output=output[:, 0].tolist())

        # Two resident token results coexist; the second replay must not replace
        # the first submission's output or consult intentionally unusable hosts.
        before = counters()
        before_positions = state()["positions"]
        handles, expected = [], []
        for _ in range(2):
            handle = decode(
                StaleInputs(), StaleInputs(), sampling_params=params, reload_inputs=False, read_from_device=False
            )
            handles.append(handle)
            expected.append(state()["tokens"][list(rows)].clone())
        got = [finish(handle, True)[:, 0] for handle in handles]
        assert all(torch.equal(actual, wanted) for actual, wanted in zip(got, expected))
        changes = difference(before)
        assert changes["token_refreshes"] == changes["position_refreshes"] == changes["activity_refreshes"] == 0
        assert changes["model_replays"] == changes["sampling_replays"] == 2
        assert torch.equal(state()["positions"], before_positions + 2)
        record("resident_async_ownership", counters=changes, outputs=[value.tolist() for value in got])

        # Keep slot zero active and move two dormant request states. Their model
        # histories, positions, tokens and RNG streams must follow exactly once.
        before_state = state()
        dormant = {slot: {key: local_bits(linear[slot][key]) for key in ("recurrent", "conv")} for slot in (1, 2)}
        initial_rng = [value.clone() for value in generator.initial_host_rng_states]
        remaps = generator.slot_state.remaps
        token_rows = before_state["tokens"][:3].reshape(3, 1)
        decode(
            token_rows,
            [int(before_state["positions"][0]), -1, -1],
            slots=[0, 1, 2],
            sampling_params=params,
            slot_remap=[0, 2, 1],
        )
        after = state()
        assert generator.slot_state.remaps == remaps + 1
        for destination, source in ((1, 2), (2, 1)):
            for key in ("positions", "tokens", "seeds"):
                assert after[key][destination] == before_state[key][source]
            assert torch.equal(generator.initial_host_rng_states[destination], initial_rng[source])
            for key in ("recurrent", "conv"):
                actual = local_bits(linear[destination][key])
                assert all(torch.equal(a, b) for a, b in zip(actual, dormant[source][key])), (destination, key)
        record("simultaneous_dormant_remap", remaps=generator.slot_state.remaps, all_four_chips=True)

        # Only unused tail mappings change; live cache content needs no migration.
        current_pages = torch.cat([slot["_host_pages"] for slot in full], dim=0)
        changed_pages = current_pages.clone()
        changed_pages[:, [-1, -2]] = changed_pages[:, [-2, -1]]
        assert not torch.equal(changed_pages, current_pages)
        full_groups = [group for group in generator.cache if "page_table" in group[0]]
        changed_table_count = sum(len(group) for group in full_groups)
        for label, table, expected_updates in (
            ("page_changed", changed_pages, changed_table_count),
            ("page_unchanged", changed_pages, 0),
        ):
            before = counters()
            before_positions = state()["positions"]
            active = tuple(generator.active_slots)
            decode(
                StaleInputs(),
                StaleInputs(),
                sampling_params=params,
                reload_inputs=False,
                reload_page_table=True,
                page_table=table,
            )
            changes = difference(before)
            assert changes["page_table_refreshes"] == expected_updates, changes
            for group in full_groups:
                for slot, cache_slot in enumerate(group):
                    assert torch.equal(cache_slot["_host_pages"], table[slot : slot + 1])
            assert changes["token_refreshes"] == changes["position_refreshes"] == changes["activity_refreshes"] == 0
            expected_positions = before_positions.clone()
            expected_positions[list(active)] += 1
            assert torch.equal(
                state()["positions"], expected_positions
            ), "Page refresh reset or double-advanced positions"
            assert generator.slot_state.remaps == remaps + 1
            record(label, counters=changes, expected_updates=expected_updates, full_attention_layers=len(full_groups))

        before = state()
        replacement = sampling_params(
            temperature=[0.0, 0.0, 0.0], top_k=[1, 1, 1], top_p=[0.8, 1.0, 1.0], seed=[999, 888, 777]
        )
        decode(
            StaleInputs(), StaleInputs(), sampling_params=replacement, reload_inputs=False, reload_sampling_params=True
        )
        after = state()
        assert after["seeds"][0] == before["seeds"][0] + 1
        assert torch.equal(after["seeds"][1:], before["seeds"][1:])
        assert generator.sampling_values["seed"][0] == 999
        record("parameter_only_preserves_live_rng")

        before = state()
        decode(
            before["tokens"][:3].reshape(3, 1),
            [int(before["positions"][0]), -1, -1],
            sampling_params=params,
            reset_sampling_state=True,
        )
        after = state()
        assert after["seeds"][0] == 999 % 1000000 + 2, "Reset did not use remembered seed plus one sample advance"
        assert torch.equal(after["seeds"][1:], before["seeds"][1:])
        record("reset_without_parameter_reload", seed=int(after["seeds"][0]))

        if args.wide_sampling:
            for top_k, trace_key in ((40, "sampling_64"), (20, "sampling"), (40, "sampling_64")):
                before = state()
                replays = dict(generator.sampling_trace_replays)
                transition = sampling_params(
                    temperature=[0.7, 0.0, 0.0], top_k=[top_k, 1, 1], top_p=[0.95, 1.0, 1.0], seed=[999, 888, 777]
                )
                output = decode(
                    StaleInputs(),
                    StaleInputs(),
                    sampling_params=transition,
                    reload_inputs=False,
                    reload_sampling_params=True,
                )
                after = state()
                assert generator.sampling_trace_replays[trace_key] == replays.get(trace_key, 0) + 1
                other = "sampling" if trace_key == "sampling_64" else "sampling_64"
                assert generator.sampling_trace_replays.get(other, 0) == replays.get(other, 0)
                for field in ("positions", "seeds"):
                    assert after[field][0] == before[field][0] + 1
                    assert torch.equal(after[field][1:], before[field][1:])
                assert torch.equal(after["tokens"][1:], before["tokens"][1:])
                assert int(output[0, 0]) == int(after["tokens"][0])
                record(
                    "sampling_width_transition",
                    top_k=top_k,
                    trace_key=trace_key,
                    sampling_trace_replays=dict(generator.sampling_trace_replays),
                )

            retained_wide = sampling_params(
                temperature=[0.0, 0.7, 0.7], top_k=[1, 40, 64], top_p=[1.0, 0.95, 0.95], seed=[999, 888, 777]
            )
            before = state()
            decode(
                before["tokens"][:3].reshape(3, 1),
                before["positions"].tolist(),
                slots=[0, 1, 2],
                sampling_params=retained_wide,
                reload_sampling_params=True,
            )
            before = state()
            replays = dict(generator.sampling_trace_replays)
            decode(
                before["tokens"][:3].reshape(3, 1),
                [int(before["positions"][0]), -1, -1],
                slots=[0, 1, 2],
                sampling_params=params,
                reload_sampling_params=True,
            )
            after = state()
            assert generator.sampling_values["top_k"][:3] == [1, 40, 64]
            assert generator.sampling_trace_replays["sampling_64"] == replays["sampling_64"] + 1
            for field in ("tokens", "positions", "seeds"):
                assert torch.equal(after[field][1:], before[field][1:])
            for field in ("positions", "seeds"):
                assert after[field][0] == before[field][0] + 1
            record("dormant_wide_rows_remain_bounded", retained_top_k=[1, 40, 64], trace_key="sampling_64")

            before = state()
            replays = dict(generator.sampling_trace_replays)
            decode(
                before["tokens"][:3].reshape(3, 1),
                before["positions"].tolist(),
                slots=[0, 1, 2],
                sampling_params=params,
                reload_sampling_params=True,
            )
            assert generator.sampling_trace_replays["sampling"] == replays["sampling"] + 1
            record("clear_retained_wide_returns_to_narrow", trace_key="sampling")

        # Explicit host-logit mode must never sample, and asynchronous result
        # conversion owns writable storage independent of the next submission.
        before = counters()
        before_state = state()
        host_handle = decode(
            before_state["tokens"][:3].reshape(3, 1),
            [int(before_state["positions"][0]), -1, -1],
            sampling_params=None,
            read_from_device=False,
        )
        logits = finish(host_handle, False)
        assert logits.shape == (3, 1, model.vocab_size) and logits.dtype == torch.float32
        assert torch.isfinite(logits).all()
        assert difference(before)["sampling_replays"] == 0
        logits.fill_(123.0)
        assert torch.equal(state()["seeds"], before_state["seeds"])
        record("host_logits_without_sampling", shape=list(logits.shape))
        generator.output_pool.require_drained()
        result["status"] = "PASS"
    except BaseException as error:
        result.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if generator is not None:
            if adapter is not None:
                adapter.release_persistent_capture()
            else:
                generator.close()
            result["counters"] = generator.counters
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        save()
    print("SERVING_COMMANDS_PROBE_PASS", args.output, flush=True)


if __name__ == "__main__":
    main()
