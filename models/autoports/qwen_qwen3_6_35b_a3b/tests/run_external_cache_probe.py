# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Real layers 0/3, external cache and request-release isolation; serialized.

Source lane-env.sh, then run with an external timeout and log wrapper:
 timeout 900 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_external_cache_probe

This reduced M3/context1024 probe is lifecycle evidence, not model accuracy or
performance. It reads all four chips and records source hashes before opening
hardware. The source was prepared without executing devices.
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
from models.autoports.qwen_qwen3_6_35b_a3b.tests.run_generator_replay_probe import page_mapping
from models.autoports.qwen_qwen3_6_35b_a3b.tt.cache_contract import validate_cache
from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator import QwenGenerator
from models.autoports.qwen_qwen3_6_35b_a3b.tt.model import HF_MODEL, HF_REVISION, QwenModel

ROOT = Path(__file__).resolve().parents[1]


def chips(tensor):
    return tuple(ttnn.to_torch(part).clone() for part in ttnn.get_device_tensors(tensor))


def same(a, b, label):
    assert a.keys() == b.keys(), label
    for key in a:
        assert len(a[key]) == len(b[key]) == 4, (label, key, "expected four chips")
        for chip, (left, right) in enumerate(zip(a[key], b[key])):
            assert torch.equal(left, right), (label, key, chip)


def snapshot(generator):
    tensors = {}
    for index, states in zip(generator.model.layer_indices, generator.cache):
        for slot, state in enumerate(states):
            for key in ("conv", "recurrent", "page_table"):
                if key in state:
                    tensors[f"layer{index}.slot{slot}.{key}"] = state[key]
        if "k" in states[0]:
            for key in ("k", "v"):
                tensors[f"layer{index}.{key}"] = states[0][key]
    tensors.update({f"slot{slot}.position": tensor for slot, tensor in enumerate(generator.positions)})
    tensors.update({f"slot{slot}.activity": tensor for slot, tensor in enumerate(generator.activity)})
    for key in ("tokens", "active_mask"):
        tensors[key] = getattr(generator, key)
    tensors["seeds"] = generator.sampler.tt_sampling.seeds_tt_tensor
    return {key: chips(tensor) for key, tensor in tensors.items()}


def digests(values):
    return {
        key: [hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for tensor in parts]
        for key, parts in values.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "doc/full_model/external_cache_probe/result.json")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__)] + [
        ROOT / "tt" / name
        for name in (
            "model.py",
            "generator.py",
            "prefill.py",
            "prefill_validation.py",
            "cache_contract.py",
            "serving.py",
            "serving_state.py",
            "slot_state.py",
            "multichip_decoder.py",
        )
    ]
    result = dict(
        status="RUNNING",
        command=sys.argv,
        time_unix=time.time(),
        model=HF_MODEL,
        revision=HF_REVISION,
        layers=[0, 3],
        context=1024,
        max_batch_size=3,
        scope="Reduced external-cache/request lifecycle and isolation; no full-model quality or performance claim",
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        cases=[],
    )

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    torch.set_num_threads(8)
    mesh = generator = model = cache = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
        model = QwenModel(mesh, layer_indices=[0, 3], max_seq_len=1024)
        descriptor = model.cache_spec()
        result["cache_descriptor"] = descriptor
        assert model.get_max_tokens_all_users() == descriptor["aggregate_token_capacity"] == 1024
        result["aggregate_token_budget"] = model.get_max_tokens_all_users()
        specs = [(tuple(row["default_shape"]), row["dtype"], 17 + row["tensor_idx"]) for row in descriptor["layers"]]
        cache = model.allocate_kv_cache_per_layer(specs, batch=3)
        full = next(states for states in cache if "k" in states[0])
        assert all(state["k"] is full[0]["k"] and state["v"] is full[0]["v"] for state in full)
        assert len({state["page_table"].buffer_address() for state in full}) == 3
        generator = QwenGenerator(model, max_batch_size=3, kv_cache=cache)
        assert generator.cache is cache
        lengths, slots = [31, 65, 129], [0, 1, 2]
        pages = page_mapping(full[0]["page_table"].shape[-1], slots, lengths, 4)
        fixture = [
            generator.tokenizer.encode(text * 50, add_special_tokens=False)[:length]
            for text, length in zip(
                (
                    "Learning from examples improves a classifier. ",
                    "A triangle has three sides and three angles. ",
                    "Explain how a garden changes through the seasons. ",
                ),
                lengths,
            )
        ]
        assert [len(row) for row in fixture] == lengths
        tokens = torch.nn.utils.rnn.pad_sequence([torch.tensor(row) for row in fixture], batch_first=True)
        result["fixture_token_ids"] = fixture
        result["page_tables"] = pages.tolist()
        params = dict(temperature=[0.0] * 3, top_k=[1] * 3, top_p=[1.0] * 3, seed=[11, 13, 17])
        # Warm diagnostic conversions and each release/reset operation before capture.
        generator.prefill_forward(tokens[:1], prompt_lens=[31], return_all_logits=True)
        snapshot(generator)
        generator.release_request(1)
        generator.configure_sampling(**params)
        generator.prepare(prefill_cases=((32, 0),))
        generator.configure_sampling(**params)
        generator.reset()
        baseline = snapshot(generator)
        traces, entries = dict(generator.traces), int(mesh.num_program_cache_entries())
        result["trace_ids"] = {key: int(value) for key, value in traces.items()}
        result["prepared_program_cache_entries"] = entries

        def stable(label):
            gc.collect()
            ttnn.synchronize_device(mesh)
            assert generator.cache is cache and generator.traces == traces, label
            now = int(mesh.num_program_cache_entries())
            assert now == entries, (label, "post-capture program growth", entries, now)

        def admit():
            prefill = generator.prefill_forward(
                tokens,
                prompt_lens=lengths,
                empty_slots=slots,
                page_tables_per_layer=[pages],
                kv_cache=cache,
                sampling_params=params,
                read_from_device=True,
            )
            outputs = [prefill.tolist()]
            for _ in range(2):
                output = generator.decode_forward_v1(
                    None,
                    None,
                    kv_cache=cache,
                    sampling_params=params,
                    reload_inputs=False,
                    read_from_device=True,
                )
                outputs.append(output.tolist())
            return outputs

        first_outputs = admit()
        before = snapshot(generator)
        result["live_state_sha256"] = digests(before)
        host_rng = [rng.get_state().clone() for rng in generator.host_rng]
        prompt_history = generator.sampler.tt_penalties._prompt_tokens_host.clone()
        output_history = generator.serving_output_history.clone()
        state_addresses = {
            f"{i}.{s}.{key}": value.buffer_address()
            for i, states in enumerate(cache)
            for s, state in enumerate(states)
            for key, value in state.items()
            if key in ("k", "v", "conv", "recurrent", "page_table")
        }

        def rejected(label, call):
            counters = dict(generator.counters)
            try:
                call()
            except (ValueError, TypeError, RuntimeError) as error:
                result["cases"].append(dict(case=label, status="PASS", rejection=str(error)))
            else:
                raise AssertionError((label, "invalid request accepted"))
            same(snapshot(generator), before, label)
            assert generator.counters == counters, (label, "counters changed before rejection")
            assert not generator.slot_state.poisoned and not generator.reload_executor.poisoned, label
            save()

        rejected(
            "bad_later_prefill_row",
            lambda: generator.prefill_forward(
                tokens[:2], prompt_lens=[31, tokens.shape[1] + 1], slots=[0, 1], page_tables_per_layer=[pages[:2]]
            ),
        )
        rejected("noninteger_prefill_tokens", lambda: generator.prefill_forward(tokens[:1].float(), prompt_lens=[31]))
        rejected("invalid_release_slot", lambda: generator.release_request(3))
        rejected(
            "external_cache_rebind",
            lambda: generator.prefill_forward(tokens[:1], prompt_lens=[31], kv_cache=list(cache)),
        )
        rejected(
            "malformed_cache_plan",
            lambda: model.allocate_kv_cache_per_layer([((4, 1, 32, 256), "bfloat16", 17)], batch=3),
        )
        aliased = [list(group) for group in cache]
        aliased[0][1] = aliased[0][0]
        rejected("aliased_recurrent_slot", lambda: validate_cache(model, aliased, batch=3))
        decode_tokens = torch.tensor(first_outputs[-1], dtype=torch.int64)
        decode_positions = [length + 2 for length in lengths]
        duplicate_pages = pages.clone()
        duplicate_pages[0, 1] = duplicate_pages[0, 0]
        overlapping_pages = pages.clone()
        overlapping_pages[1, 0] = overlapping_pages[0, 0]
        for label, invalid_pages in (
            ("decode_duplicate_logical_page", duplicate_pages),
            ("decode_overlapping_submitted_pages", overlapping_pages),
        ):
            rejected(
                label,
                lambda invalid_pages=invalid_pages: generator.decode_forward_v1(
                    decode_tokens,
                    decode_positions,
                    kv_cache=cache,
                    page_tables_per_layer=[invalid_pages],
                    sampling_params=params,
                    read_from_device=True,
                ),
            )
        pending = generator.output_pool.reserve((0,), is_tokens=True)
        try:
            rejected("release_requires_drained_outputs", lambda: generator.release_request(1))
            rejected(
                "prefill_requires_drained_outputs_without_poisoning",
                lambda: generator.prefill_forward(tokens[:1], prompt_lens=[31], page_table=pages[:1]),
            )
        finally:
            generator.output_pool.cancel_unsubmitted(pending)

        generator.release_request(1)
        after = snapshot(generator)
        unchanged = {
            key: value
            for key, value in before.items()
            if ".slot1." not in key and not key.startswith("slot1.") and key not in ("active_mask", "seeds")
        }
        same(unchanged, {key: after[key] for key in unchanged}, "release preserves other requests and entire shared KV")
        for key in after:
            if key.endswith((".slot1.conv", ".slot1.recurrent", "slot1.position", "slot1.activity")):
                assert all(torch.count_nonzero(value) == 0 for value in after[key]), ("released state not zero", key)
            if key.endswith(".slot1.page_table"):
                same({key: before[key]}, {key: after[key]}, "released page mapping stays scheduler-owned")
        for chip in range(4):
            for slot in (0, 2):
                assert after["seeds"][chip].flatten()[slot] == before["seeds"][chip].flatten()[slot]
            assert after["seeds"][chip].flatten()[1] == 1
            assert after["active_mask"][chip].flatten()[:3].tolist() == [1, 0, 1]
        assert generator.active_slots == (0, 2)
        for slot in (0, 2):
            assert torch.equal(host_rng[slot], generator.host_rng[slot].get_state())
            assert torch.equal(prompt_history[slot], generator.sampler.tt_penalties._prompt_tokens_host[slot])
            assert torch.equal(output_history[slot], generator.serving_output_history[slot])
        assert generator.sampler.tt_penalties._prompt_tokens_host[1].eq(-1).all()
        assert generator.serving_output_history[1].eq(-1).all()
        generator.release_request(1)
        same(snapshot(generator), after, "repeated slot release is idempotent")
        assert state_addresses == {
            f"{i}.{s}.{key}": value.buffer_address()
            for i, states in enumerate(cache)
            for s, state in enumerate(states)
            for key, value in state.items()
            if key in ("k", "v", "conv", "recurrent", "page_table")
        }
        stable("slot release")
        result["cases"].append(
            dict(
                case="release_isolation_all_four_chips",
                status="PASS",
                selected_token_policy="resident token retained while inactive; authoritative reload replaces it",
            )
        )

        generator.configure_sampling(**params)
        generator.reset()
        same(snapshot(generator), baseline, "full reset restores same configured state")
        assert generator.serving_rows == (0,) and generator.serving_device_sampling is None
        assert generator.serving_output_history is None and generator.sampler.tt_penalties._prompt_tokens_host is None
        second_outputs = admit()
        assert first_outputs == second_outputs, (
            "reset/re-admission predictions changed",
            first_outputs,
            second_outputs,
        )
        same(snapshot(generator), before, "repeat admission state exact")
        stable("repeat admission")
        result["cases"].append(
            dict(case="reset_and_repeated_external_cache_admission", status="PASS", outputs=first_outputs)
        )
        result["release_state_sha256"] = digests(after)
        captured_count = len(traces)
        generator.release_persistent_capture()
        release_count = generator.counters["releases"]
        assert release_count == captured_count and not generator.traces and generator.closed
        generator.release_persistent_capture()
        assert generator.counters["releases"] == release_count
        result["cases"].append(
            dict(case="persistent_capture_release_exactly_once", status="PASS", released=release_count)
        )
        result["status"] = "PASS"
        result["elapsed_seconds"] = time.time() - result["time_unix"]
        save()
        print("EXTERNAL_CACHE_PROBE_PASS", args.output, flush=True)
    except BaseException:
        result["status"] = "FAIL"
        result["traceback"] = traceback.format_exc()
        save()
        raise
    finally:
        if generator is not None:
            generator.close()
        generator = model = cache = None
        gc.collect()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
