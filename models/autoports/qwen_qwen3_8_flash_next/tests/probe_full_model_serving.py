# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced real-layer serving/feedback gate, including tracker positive control.

Run serially from the configured checkout, after exporting the accepted mesh
descriptor (lane-env clears it). Tracking must be enabled before Python starts::

    TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
      timeout 3600 python -m \
      models.autoports.qwen_qwen3_8_flash_next.tests.probe_full_model_serving \
      --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/serving_probe.json

This is correctness evidence only. Host snapshots deliberately synchronize and
read device state; they are not part of generator performance measurements.
No implementation allocations are acknowledged/exempted by this test.
"""

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import torch
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_shared_pool_generator import decode_geometry_comparison
from models.autoports.qwen_qwen3_8_flash_next.tt.generator import DecodeSubmission, build_generator
from models.autoports.qwen_qwen3_8_flash_next.tt.host_sampling import HostSampler, HostSamplingParams
from models.common.sampling.sampling_params import SamplingParams


def read_ranks(tensor):
    return [ttnn.to_torch(part).clone() for part in ttnn.get_device_tensors(tensor)]


def mutable_cache(cache):
    """Read every mutable decoder state tensor, across every rank."""
    result = {}
    for index, state in enumerate(cache.layers):
        for name in ("conv", "recurrent", "ple_conv", "token_history"):
            value = getattr(state, name, None)
            if value is not None:
                result[f"{index}.{name}"] = read_ranks(value)
        if state.attention is not None:
            for name in ("position", "page_table", "k_cache", "v_cache", "raw_history", "compressed_keys"):
                result[f"{index}.attention.{name}"] = read_ranks(getattr(state.attention, name))
    return result


def equal_cache(expected, actual, label):
    assert expected.keys() == actual.keys(), f"{label}: state fields changed"
    for name in expected:
        for rank, (left, right) in enumerate(zip(expected[name], actual[name], strict=True)):
            assert torch.equal(left, right), f"{label}: {name} rank {rank} changed"


def positions(cache, expected):
    found = []
    for index, state in enumerate(cache.layers):
        if state.attention is not None:
            for rank, value in enumerate(read_ranks(state.attention.position)):
                actual = value.flatten().tolist()
                assert actual == [expected], f"layer {index} rank {rank} QSA position {actual} != {expected}"
                found.append({"layer": index, "rank": rank, "position": actual[0]})
    assert found, "Reduced probe must include QSA"
    return found


def packed(prompts):
    value = torch.zeros(len(prompts), max(map(len, prompts)), dtype=torch.int64)
    for row, prompt in enumerate(prompts):
        value[row, : len(prompt)] = torch.tensor(prompt)
    return value


def trace_ids(gen):
    return {
        "model": {str(key): str(value) for key, value in gen.traces.items()},
        "sampling": {str(key): str(value) for key, value in gen.sampling_traces.items()},
        "prefill": {str(key): str(value) for key, value in gen.prefill_traces.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", default=os.environ.get("QWEN_CHECKPOINT"))
    parser.add_argument("--packed-cache", default=os.environ.get("QWEN_PLE_CACHE"))
    parser.add_argument("--trace-region-size", type=int, default=256 * 1024 * 1024)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new artifact path")
    model_dir = Path(__file__).resolve().parents[1]
    report = {
        "passed": False,
        "scope": "reduced_real_layers_0_1_3",
        "capacity": 128,
        "max_batch_size": 2,
        "regime": "serial",
        "tracking_enabled": bool(trace_allocation_tracker.TRACE_ALLOC_TRACKING),
        "skip_program_cache": os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0"),
        "mesh_descriptor": os.environ.get("TT_MESH_GRAPH_DESC_PATH"),
        "checks": {},
        "source_hashes": {
            str(path.relative_to(model_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((model_dir / "tt").glob("*.py"))
        },
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    started = time.monotonic()
    mesh = gen = None
    current = "setup"
    try:
        assert report["tracking_enabled"], "Start process with TT_METAL_TRACE_ALLOC_TRACKING=1"
        assert report["skip_program_cache"] == "0", "Acceptance must include program-cache allocations"
        assert report["mesh_descriptor"], "Restore the accepted mesh descriptor after sourcing lane-env"
        torch.set_num_threads(8)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(1, 4), trace_region_size=args.trace_region_size, l1_small_size=32768
        )
        options = {"layer_indices": (0, 1, 3), "max_batch_size": 2, "max_seq_len": 128}
        if args.checkpoint:
            options["checkpoint"] = args.checkpoint
        if args.packed_cache:
            options["packed_cache"] = args.packed_cache
        gen = build_generator("models/autoports/qwen_qwen3_8_flash_next", mesh, **options)
        gen.prepare()
        report["prepared_seconds"] = time.monotonic() - started
        report["program_cache_entries_after_prepare"] = mesh.num_program_cache_entries()
        identity = torch.arange(4, dtype=torch.int32).reshape(1, 4)
        prompts = ([785, 13458, 13], [785, 13, 13458, 13, 785] * 6 + [785, 13, 13458])

        def prefill(rows, slots, table=None, **kwargs):
            return gen.prefill_forward(
                packed(rows),
                page_table=identity.repeat(len(rows), 1) if table is None else table,
                kv_cache=[gen.caches[s] for s in slots],
                prompt_lens=list(map(len, rows)),
                slots=slots,
                **kwargs,
            )

        def decode(tokens, pos, slots=(0, 1), table=None, **kwargs):
            return gen.decode_forward(
                torch.tensor(tokens, dtype=torch.int64),
                pos,
                slots=slots,
                page_table=identity.repeat(len(slots), 1) if table is None else table,
                kv_cache=[gen.caches[s] for s in slots],
                **kwargs,
            )

        current = "warmup_idempotence_and_cache_spec"
        assert gen.model_capabilities["fabric_config"] == {"config": ttnn.FabricConfig.FABRIC_1D_RING}
        before = dict(gen.counters)
        before_ids = trace_ids(gen)
        before_programs = mesh.num_program_cache_entries()
        prepared_state = [mutable_cache(cache) for cache in gen.caches]
        prepared_seeds = read_ranks(gen.seed_state.state)
        gen.warmup_model_prefill(kv_cache=gen.caches)
        gen.warmup_model_decode(kv_cache=gen.caches)
        assert gen.counters == before and trace_ids(gen) == before_ids, "Repeated warmup mutated lifecycle counters"
        assert mesh.num_program_cache_entries() == before_programs, "Repeated warmup compiled new programs"
        for slot, cache in enumerate(gen.caches):
            assert cache.position == 0, "Preparation did not restore scheduler position"
            positions(cache, 0)
            equal_cache(prepared_state[slot], mutable_cache(cache), f"Repeated warmup slot {slot}")
        equal_cache({"seed": prepared_seeds}, {"seed": read_ranks(gen.seed_state.state)}, "Repeated warmup RNG")
        spec = gen.cache_spec()
        assert spec["paged_kv_dtype"] == "bfloat16" and spec["page_size"] == 32
        assert [layer["layer"] for layer in spec["layers"]] == [0, 1, 3]
        for description, cache_layer in zip(spec["layers"], gen.caches[0].layers, strict=True):
            for name, geometry in description["tensors"].items():
                tensor = getattr(cache_layer, name, None)
                if tensor is None:
                    tensor = getattr(cache_layer.attention, name)
                assert geometry == {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}, name
        report["checks"][current] = {"passed": True, "cache_spec": spec, "program_cache_entries": before_programs}
        del prepared_state, prepared_seeds

        current = "mixed_prefill_equals_solo"
        solo_logits, solo_states = [], []
        for prompt in prompts:
            solo_logits.append(prefill([prompt], (0,))[0].clone())
            solo_states.append(mutable_cache(gen.caches[0]))
        batch = prefill(prompts, (0, 1))
        for row in range(2):
            assert torch.equal(batch[row], solo_logits[row]), f"Mixed prefill logits row {row} != solo"
            equal_cache(solo_states[row], mutable_cache(gen.caches[row]), f"Mixed prefill row {row}")
        permuted = prefill(prompts[::-1], (1, 0))
        for row, original in enumerate((1, 0)):
            assert torch.equal(permuted[row], solo_logits[original]), f"Permuted prefill row {row} != solo"
            equal_cache(solo_states[original], mutable_cache(gen.caches[original]), f"Permuted slot {original}")
        report["checks"][current] = {
            "passed": True,
            "lengths": list(map(len, prompts)),
            "slot_orders": [[0, 1], [1, 0]],
        }

        current = "ragged_absolute_prefill_continuations"
        starts = (2, 17)
        resumed_solo_logits, resumed_solo_states = [], []
        for row, start in enumerate(starts):
            prefill([prompts[row][:start]], (0,))
            resumed_solo_logits.append(prefill([prompts[row]], (0,), start_pos=[start])[0].clone())
            resumed_solo_states.append(mutable_cache(gen.caches[0]))
        for row, start in enumerate(starts):
            prefill([prompts[row][:start]], (row,))
        resumed = prefill(prompts, (0, 1), start_pos=starts)
        for row in range(2):
            assert torch.equal(
                resumed[row], resumed_solo_logits[row]
            ), f"Ragged continuation row {row} != solo continuation"
            equal_cache(resumed_solo_states[row], mutable_cache(gen.caches[row]), f"Ragged continuation slot {row}")
            assert gen.caches[row].position == len(prompts[row])
            positions(gen.caches[row], len(prompts[row]))
        report["checks"][current] = {
            "passed": True,
            "absolute_starts": list(starts),
            "absolute_ends": list(map(len, prompts)),
            "solo_continuation_outputs_and_all_state_equal": True,
            "full_vs_resumed_accuracy_gate": "separate run_prefill_resume_check with negative controls",
        }
        del resumed_solo_states, resumed_solo_logits

        current = "batched_decode_equals_solo"
        forced = [42, 43]
        solo_decode = []
        for row in range(2):
            prefill([prompts[row]], (0,))
            solo_decode.append(decode([forced[row]], [len(prompts[row])], slots=(0,))[0].clone())
        prefill(prompts, (0, 1))
        batched_decode = decode(forced, list(map(len, prompts)))
        decode_stats = decode_geometry_comparison(torch.stack(solo_decode), batched_decode)
        # The accepted n=1 FP32 matmul reduction and n>1 BF16 native
        # reduction have distinct rounding. Reuse the frozen batch gate;
        # same-geometry feedback, cache, migration and isolation stay exact.
        assert decode_stats["passed"], f"Batched decode failed frozen accuracy gate: {decode_stats}"
        for row in range(2):
            positions(gen.caches[row], len(prompts[row]) + 1)
        report["checks"][current] = {"passed": True, "comparison": decode_stats}

        current = "fixed_slot_and_inactive_row"
        prefill(prompts, (0, 1))
        inactive = mutable_cache(gen.caches[0])
        inactive_position = gen.caches[0].position
        result = decode([999, 43], [-1, len(prompts[1])], sample_on_device=True)
        assert result.shape == (2,) and result[0].item() == -1, "Inactive output must be -1 in row order"
        equal_cache(inactive, mutable_cache(gen.caches[0]), "Inactive slot zero")
        assert gen.caches[0].position == inactive_position, "Inactive scheduler position mutated"
        positions(gen.caches[1], len(prompts[1]) + 1)
        report["checks"][current] = {"passed": True, "output": result.tolist()}

        current = "active_reload_preserves_dormant_feedback"
        prefill(prompts, (0, 1))
        initial = decode([42, 43], list(map(len, prompts)), sample_on_device=True)
        retained = int(initial[1])
        assert retained != 0, "Reload regression control needs a nonzero sampled token to expose zero-filled overwrite"
        dormant_state = mutable_cache(gen.caches[1])
        dormant_position = gen.caches[1].position
        resident_before = read_ranks(gen.decode_tokens)
        for rank, value in enumerate(resident_before):
            assert value.flatten()[1].item() == retained, f"Rank {rank} missing initial sampled feedback"
        refreshed = decode([99, 1234], [len(prompts[0]) + 1, -1], sample_on_device=True, reload_inputs=True)
        assert refreshed[1].item() == -1, "Dormant row must remain inactive in scheduler output"
        for rank, (before_tokens, after_tokens) in enumerate(
            zip(resident_before, read_ranks(gen.decode_tokens), strict=True)
        ):
            assert torch.equal(
                before_tokens.flatten()[1:], after_tokens.flatten()[1:]
            ), f"Rank {rank}: explicit active-slot reload overwrote dormant feedback"
        equal_cache(dormant_state, mutable_cache(gen.caches[1]), "Dormant slot1 during active0 reload")
        assert gen.caches[1].position == dormant_position, "Dormant scheduler position changed during active0 reload"
        positions(gen.caches[1], len(prompts[1]) + 1)
        # Resume using resident feedback. The uninterrupted control must have
        # the same initial B2 decode geometry, followed by a single active row;
        # a B1-only history would change accepted MoE reduction rounding.
        continued = decode([1234], [97], slots=(1,), sample_on_device=True, reload_inputs=False)
        continued_state = mutable_cache(gen.caches[1])
        prefill(prompts, (0, 1))
        control_initial = decode([42, 43], list(map(len, prompts)), sample_on_device=True)
        control_continued = decode([1234], [97], slots=(1,), sample_on_device=True, reload_inputs=False)
        assert int(control_initial[1]) == retained, "Interrupted and uninterrupted initial tokens differ"
        assert torch.equal(
            continued, control_continued
        ), "Dormant feedback continuation differs from uninterrupted control"
        equal_cache(continued_state, mutable_cache(gen.caches[1]), "Dormant feedback continuation versus control")
        report["checks"][current] = {
            "passed": True,
            "active_reloaded_slot": 0,
            "dormant_slot": 1,
            "retained_feedback_token": retained,
            "continued_token": int(continued[0]),
            "all_inactive_token_rows_preserved": True,
            "dormant_cache_and_position_preserved": True,
            "continuation_matches_uninterrupted_control": True,
        }

        current = "on_device_feedback_and_stale_host_reload"
        short = prompts[0]
        prefill([short], (1,))
        first = decode([42], [len(short)], slots=(1,), sample_on_device=True)
        token_address = gen.decode_tokens.buffer_address()
        before = dict(gen.counters)
        device_token = read_ranks(gen.decode_tokens)
        for rank, value in enumerate(device_token):
            assert value.flatten()[1].item() == first[0].item(), f"Rank {rank} sampled feedback not in slot 1"
        # Positive stale values intentionally differ from the actual position;
        # -1 is reserved for inactive-row metadata and is tested above.
        second = decode([1234], [97], slots=(1,), sample_on_device=True, reload_inputs=False)
        assert gen.decode_tokens.buffer_address() == token_address, "Feedback input backing allocation replaced"
        for field in ("token_refreshes", "position_refreshes", "page_table_refreshes"):
            assert gen.counters[field] == before[field], f"Steady-state refresh: {field}"
        observed_positions = positions(gen.caches[1], len(short) + 2)
        for rank, value in enumerate(read_ranks(gen.decode_tokens)):
            assert value.flatten()[1].item() == second[0].item(), f"Rank {rank} second feedback missing"
        # Reproduce the same two steps with authoritative host reload only in
        # this independent control, proving the retained token was consumed.
        prefill([short], (1,))
        control_first = decode([42], [len(short)], slots=(1,), sample_on_device=True)
        control_second = decode(control_first.tolist(), [len(short) + 1], slots=(1,), sample_on_device=True)
        assert torch.equal(first, control_first) and torch.equal(
            second, control_second
        ), "Device feedback differs from reload control"
        report["checks"][current] = {
            "passed": True,
            "tokens": [first.item(), second.item()],
            "persistent_token_address": token_address,
            "positions": observed_positions,
            "steady_state_counter_delta": {
                k: 0 for k in ("token_refreshes", "position_refreshes", "page_table_refreshes")
            },
        }

        current = "page_table_changed_and_unchanged"
        boundary_prompt = short * 10
        prefill([boundary_prompt], (1,))
        page_outputs = [decode([42], [30], slots=(1,), sample_on_device=True).item()]
        count = gen.counters["page_table_refreshes"]
        page_outputs.append(
            decode([1234], [97], slots=(1,), sample_on_device=True, reload_inputs=False, reload_page_table=True).item()
        )
        assert gen.counters["page_table_refreshes"] == count, "Unchanged table copied"
        # Only unused physical pages change, so existing token pages remain
        # valid. The next decode (position 32) must actually write the newly
        # selected physical page, not merely update a spare input tensor.
        changed = torch.tensor([[0, 2, 1, 3]], dtype=torch.int32)
        before_remap = mutable_cache(gen.caches[1])
        page_outputs.append(
            decode(
                [1234],
                [97],
                slots=(1,),
                table=changed,
                sample_on_device=True,
                reload_inputs=False,
                reload_page_table=True,
            ).item()
        )
        assert gen.counters["page_table_refreshes"] == count + 1, "Changed table not copied exactly once"
        for index, state in enumerate(gen.caches[1].layers):
            if state.attention is not None:
                for value in read_ranks(state.attention.page_table):
                    assert torch.equal(value.int(), changed), "Captured QSA table did not receive changed mapping"
                for name in ("k_cache", "v_cache"):
                    for rank, value in enumerate(read_ranks(getattr(state.attention, name))):
                        previous = before_remap[f"{index}.attention.{name}"][rank]
                        assert torch.equal(value[1], previous[1]), f"{name} rank {rank}: wrote old physical page 1"
                        assert not torch.equal(
                            value[2], previous[2]
                        ), f"{name} rank {rank}: new physical page 2 was not written"
        positions(gen.caches[1], 33)
        prefill([boundary_prompt], (1,))
        control = [decode([42], [30], slots=(1,), sample_on_device=True).item()]
        for _ in range(2):
            control.append(decode([1234], [97], slots=(1,), sample_on_device=True, reload_inputs=False).item())
        assert page_outputs == control, "Remapped-page decode differs from identity-mapping control"
        report["checks"][current] = {
            "passed": True,
            "changed_mapping": changed.tolist(),
            "tokens": page_outputs,
            "physical_page_write_verified": 2,
            "old_physical_page_unchanged": 1,
        }

        current = "cross_request_trace_reuse"
        prior_ids, prior_captures, prior_releases = trace_ids(gen), gen.counters["captures"], gen.counters["releases"]
        prefill_count = len(gen.prefill_traces)
        for repeat in range(2):
            prefill(prompts[::-1], (1, 0))
            decode([43, 42], [len(prompts[1]), len(prompts[0])], slots=(1, 0), sample_on_device=True)
            prefill([short], (1,))
            decode([42], [len(short)], slots=(1,), sample_on_device=True)
            assert len(gen.prefill_traces) == prefill_count, "Previously exercised prefill buckets were recaptured"
            if repeat == 0:
                # First encounter with (1, 0) may defer its first capture.
                # Existing signatures must survive it unchanged.
                after_first = trace_ids(gen)
                for family, ids in prior_ids.items():
                    assert all(
                        after_first[family].get(k) == v for k, v in ids.items()
                    ), "Previously captured trace replaced"
                prior_ids, prior_captures = after_first, gen.counters["captures"]
            else:
                assert (
                    trace_ids(gen) == prior_ids and gen.counters["captures"] == prior_captures
                ), "Repeated signature recaptured"
            assert gen.counters["releases"] == prior_releases, "Request handling released traces"
        report["checks"][current] = {
            "passed": True,
            "trace_ids": trace_ids(gen),
            "prefill_cached_trace_count": prefill_count,
        }

        current = "heterogeneous_sampling_and_request_release"
        request_params = (
            SamplingParams(temperature=0.8, top_k=32, top_p=0.95, seed=-7),
            SamplingParams(temperature=1.3, top_k=16, top_p=0.7, seed=2**63 + 19),
        )

        def mixed_params(order):
            return SamplingParams(
                **{
                    name: [getattr(request_params[row], name) for row in order]
                    for name in ("temperature", "top_k", "top_p", "seed")
                }
            )

        sampled_controls, sampled_states = [], []
        for row in range(2):
            first = prefill([prompts[row]], (0,), sampling_params=request_params[row])
            second = decode([1234], [97], slots=(0,), sample_on_device=True, reload_inputs=False)
            sampled_controls.append([int(first[0, 0]), int(second[0])])
            sampled_states.append(mutable_cache(gen.caches[0]))
        # Reordering seeded B2 requests must preserve exact tokens. Compare
        # against B2 logits/sampling, not a different n=1 reduction's draw.
        batch_first = prefill(prompts, (0, 1), sampling_params=mixed_params((0, 1)))
        batch_second = decode([1234, 1234], [97, 97], sample_on_device=True, reload_inputs=False)
        sampled_batch_controls = [[int(batch_first[row, 0]), int(batch_second[row])] for row in range(2)]
        sampled_batch_states = [mutable_cache(cache) for cache in gen.caches]
        for row in range(2):
            assert sampled_batch_controls[row][0] == sampled_controls[row][0], "Mixed sampled prefill differs from solo"
        for order in ((0, 1), (1, 0)):
            first = prefill([prompts[row] for row in order], order, sampling_params=mixed_params(order))
            second = decode([1234, 1234], [97, 97], slots=order, sample_on_device=True, reload_inputs=False)
            for output_row, request in enumerate(order):
                assert [int(first[output_row, 0]), int(second[output_row])] == sampled_batch_controls[
                    request
                ], f"Mixed sampling parameters/row order changed request {request}"
                equal_cache(
                    sampled_batch_states[request],
                    mutable_cache(gen.caches[request]),
                    f"Reordered sampled B2 request {request}",
                )
        initial = prefill(prompts, (0, 1), sampling_params=mixed_params((0, 1)))
        dormant_before = mutable_cache(gen.caches[0])
        dormant_seed = read_ranks(gen.seed_state.state)
        migrated = decode(
            [int(initial[1, 0]), 1234],
            [len(prompts[1]), -1],
            slot_remap=(1, 0),
            sample_on_device=True,
            reload_sampling_params=True,
            reset_sampling_state=True,
            sampling_params=mixed_params((1, 0)),
        )
        assert migrated.tolist() == [sampled_controls[1][1], -1], "Seeded migration changed active output"
        equal_cache(sampled_states[1], mutable_cache(gen.caches[0]), "Migrated active request")
        equal_cache(dormant_before, mutable_cache(gen.caches[1]), "Migrated dormant request")
        for rank, (old, new) in enumerate(zip(dormant_seed, read_ranks(gen.seed_state.state), strict=True)):
            assert torch.equal(old[0], new[1]), f"Migration changed dormant RNG rank {rank}"
        dormant_next = decode([1234], [97], slots=(1,), sample_on_device=True, reload_inputs=False)
        assert int(dormant_next[0]) == sampled_controls[0][1], "Migrated dormant feedback/RNG continuation changed"
        equal_cache(sampled_states[0], mutable_cache(gen.caches[1]), "Migrated dormant continuation")
        # Termination/reuse is a slot operation. A surviving seeded request
        # must retain its private caches, RNG, feedback, and captured programs.
        prefill(prompts, (0, 1), sampling_params=mixed_params((0, 1)))
        survivor = mutable_cache(gen.caches[1])
        seed_before, tokens_before = read_ranks(gen.seed_state.state), read_ranks(gen.decode_tokens)
        ids_before, releases_before = trace_ids(gen), gen.counters["releases"]
        gen.release_request(0)
        gen.release_request(0)
        assert gen.caches[0].position == 0 and gen.request_seeds[0] is None
        positions(gen.caches[0], 0)
        equal_cache(survivor, mutable_cache(gen.caches[1]), "Release leaves survivor state")
        assert gen.caches[1].position == len(prompts[1])
        for rank, (old, new) in enumerate(zip(seed_before, read_ranks(gen.seed_state.state), strict=True)):
            assert torch.equal(old[1:], new[1:]), f"Release mutated other seed slots on rank {rank}"
        for rank, (old, new) in enumerate(zip(tokens_before, read_ranks(gen.decode_tokens), strict=True)):
            assert torch.equal(old.flatten()[1:], new.flatten()[1:]), f"Release mutated other feedback rank {rank}"
        assert trace_ids(gen) == ids_before and gen.counters["releases"] == releases_before
        survivor_next = decode([1234], [97], slots=(1,), sample_on_device=True, reload_inputs=False)
        assert int(survivor_next[0]) == sampled_controls[1][1], "Release changed survivor continuation"
        first_reused = prefill([prompts[0]], (0,), sampling_params=request_params[0])
        second_reused = decode([1234], [97], slots=(0,), sample_on_device=True, reload_inputs=False)
        assert [int(first_reused[0, 0]), int(second_reused[0])] == sampled_controls[
            0
        ], "Same-slot request reuse leaked state"
        report["checks"][current] = {
            "passed": True,
            "sampled_controls": sampled_controls,
            "sampled_batch_controls": sampled_batch_controls,
            "slot_orders": [[0, 1], [1, 0]],
            "released_slot": 0,
            "surviving_slot": 1,
            "request_release_retains_traces": True,
            "seed_and_feedback_survivor_unchanged": True,
            "active_and_dormant_migration_matches_solo": True,
        }
        gen.set_sampling_params(SamplingParams(temperature=0.0, top_k=1, top_p=1.0))
        del survivor, seed_before, tokens_before, sampled_states, sampled_batch_states, dormant_before, dormant_seed

        current = "independent_sampling_reload_commands"
        old_params, new_params = request_params
        old_first = int(prefill([short], (0,), sampling_params=old_params)[0, 0])
        transition_results = []
        for reload_inputs in (True, False):
            forced_token = 42 if reload_inputs else old_first
            prefill([short], (0,), sampling_params=new_params)
            control = decode([forced_token], [len(short)], slots=(0,), sample_on_device=True)
            prefill([short], (0,), sampling_params=old_params)
            before = dict(gen.counters)
            transitioned = decode(
                [forced_token if reload_inputs else 1234],
                [len(short) if reload_inputs else 97],
                slots=(0,),
                sample_on_device=True,
                reload_inputs=reload_inputs,
                reload_sampling_params=True,
                sampling_params=new_params,
            )
            assert torch.equal(transitioned, control), "Changed seed did not use the next logical draw"
            if not reload_inputs:
                for name in ("token_refreshes", "position_refreshes", "page_table_refreshes"):
                    assert gen.counters[name] == before[name], f"Params-only transition refreshed {name}"
            transition_results.append({"reload_inputs": reload_inputs, "token": int(transitioned[0])})
        prefill([short], (0,), sampling_params=new_params)
        control = decode([42], [len(short)], slots=(0,), sample_on_device=True)
        prefill([short], (0,), sampling_params=new_params)
        reset = decode([42], [len(short)], slots=(0,), sample_on_device=True, reset_sampling_state=True)
        assert torch.equal(reset, control), "Independent sampling-state reset changed the logical draw"
        report["checks"][current] = {"passed": True, "parameter_transitions": transition_results, "reset_only": True}
        gen.set_sampling_params(SamplingParams(temperature=0.0, top_k=1, top_p=1.0))

        current = "async_read_immutable_snapshot"
        async_results = []
        for sample_on_device in (True, False):
            first_positions = [len(prompts[1]), len(prompts[0])]
            second_positions = [len(prompts[1]) + 1, -1]
            prefill(prompts[::-1], (1, 0))
            control_first = decode(
                [43, 42], first_positions, slots=(1, 0), sample_on_device=sample_on_device
            ).unsqueeze(1)
            control_second = decode(
                [44, 777], second_positions, slots=(1, 0), sample_on_device=sample_on_device
            ).unsqueeze(1)
            assert not torch.equal(
                control_first, control_second
            ), "Async overwrite control must distinguish the two reads"
            prefill(prompts[::-1], (1, 0))
            readbacks_before = gen.counters["readbacks"]
            draws_before = gen.counters["sampling_replays"]
            raw_seed_state = read_ranks(gen.seed_state.state) if not sample_on_device else None
            first_submission = decode(
                [43, 42],
                first_positions,
                slots=(1, 0),
                sample_on_device=sample_on_device,
                read_from_device=False,
            )
            assert isinstance(first_submission, DecodeSubmission), "Deferred decode did not return DecodeSubmission"
            assert first_submission.slots == (1, 0) and first_submission.active_rows == (0, 1)
            assert first_submission.is_tokens is sample_on_device
            persistent = gen.decode_tokens if sample_on_device else gen.sample_logits
            assert first_submission.tensor.buffer_address() == persistent.buffer_address()
            assert gen.counters["readbacks"] == readbacks_before, "Deferred decode read data before explicit read"
            first_host, first_events = gen.read_decode_output(first_submission, async_read=True)
            assert (
                isinstance(first_host, DecodeSubmission) and first_events
            ), "Async read must return a host handle and events"
            assert (
                first_host.tensor.storage_type() == ttnn.StorageType.HOST
            ), "Async read handle still references device storage"
            # Queue another decode and another read over the same resident
            # output before waiting on or processing the first host snapshot.
            second_submission = decode(
                [44, 777],
                second_positions,
                slots=(1, 0),
                sample_on_device=sample_on_device,
                read_from_device=False,
            )
            assert isinstance(second_submission, DecodeSubmission)
            assert second_submission.slots == (1, 0) and second_submission.active_rows == (0,)
            assert second_submission.tensor.buffer_address() == first_submission.tensor.buffer_address()
            second_host, second_events = gen.read_decode_output(second_submission, async_read=True)
            assert second_events and gen.counters["readbacks"] == readbacks_before + 2
            assert second_host.tensor.storage_type() == ttnn.StorageType.HOST
            assert first_host.tensor is not second_host.tensor, "Two async reads reused the same host tensor handle"
            for event in (*first_events, *second_events):
                ttnn.event_synchronize(event)
            # Deliberately process in reverse completion order. The earlier
            # result must retain both active rows and its own data.
            second_output = gen.process_decode_output_host(second_host)
            first_output = gen.process_decode_output_host(first_host)
            expected_shape = (2, 1) if sample_on_device else (2, 1, gen.model.config.vocab_size)
            assert tuple(first_output.shape) == expected_shape and tuple(second_output.shape) == expected_shape
            assert torch.equal(first_output, control_first), "Earlier async snapshot changed after later decode/read"
            assert torch.equal(second_output, control_second), "Later async snapshot differs from synchronous control"
            assert torch.equal(
                gen.process_decode_output_host(first_host), first_output
            ), "Host processing consumed/mutated snapshot"
            if sample_on_device:
                assert second_output[1, 0].item() == -1, "Inactive scheduler row lost in async token formatting"
            else:
                assert (
                    torch.count_nonzero(second_output[1]) == 0
                ), "Inactive scheduler row lost in async logits formatting"
                assert gen.counters["sampling_replays"] == draws_before, "Raw-logit decode invoked device sampling"
                equal_cache(
                    {"seed": raw_seed_state}, {"seed": read_ranks(gen.seed_state.state)}, "Raw-logit decode RNG"
                )
                # Host processors may mutate their result without affecting
                # the other submission or a later formatting of this one.
                first_output.fill_(-123.0)
                assert torch.equal(gen.process_decode_output_host(first_host), control_first)
                assert torch.equal(gen.process_decode_output_host(second_host), control_second)
            async_results.append(
                {
                    "sample_on_device": sample_on_device,
                    "shape": list(expected_shape),
                    "slots": [1, 0],
                    "first_active_rows": [0, 1],
                    "second_active_rows": [0],
                    "first_event_count": len(first_events),
                    "second_event_count": len(second_events),
                    "delayed_reverse_processing_matches_synchronous": True,
                }
            )
        report["checks"][current] = {"passed": True, "cases": async_results}

        current = "explicit_host_sampling_compatibility"
        host_params = HostSamplingParams(
            temperature=0.9,
            top_k=64,
            top_p=0.95,
            min_p=0.01,
            repetition_penalty=1.1,
            presence_penalty=0.2,
            frequency_penalty=0.1,
            seed=-12345,
        )
        host_runs = []
        original_read_logits = gen._read_logits
        for repeat in range(2):
            read_logits, read_seeds = [], []

            def observe_host_logits(output):
                value = original_read_logits(output)
                read_logits.append(value[0].clone())
                read_seeds.append(read_ranks(gen.seed_state.state))
                return value

            gen._read_logits = observe_host_logits
            before = dict(gen.counters)
            try:
                generated = gen.generate(short, 3, host_sampling=True, sampling_params=host_params)
            finally:
                gen._read_logits = original_read_logits
            assert len(read_logits) == 3 and all(row.shape == (gen.model.config.vocab_size,) for row in read_logits)
            reference = HostSampler(host_params, prompt_token_ids=short)
            expected = [reference.sample(row).token_id for row in read_logits]
            assert generated == expected, "Host generator lost seed, parameter, or output-history semantics"
            assert gen.counters["sampling_replays"] == before["sampling_replays"], "Host mode ran device sampler"
            assert gen.counters["model_replays"] == before["model_replays"] + 2, "Host mode did not trace model decode"
            assert gen.counters["readbacks"] == before["readbacks"] + 3, "Host mode crossed an unexpected read boundary"
            assert gen.counters["token_refreshes"] == before["token_refreshes"] + 3, "Host feedback was not explicit"
            for later in read_seeds[1:]:
                equal_cache({"seed": read_seeds[0]}, {"seed": later}, "Host mode advanced device RNG")
            host_runs.append(generated)
        assert host_runs[0] == host_runs[1], "Explicit host seed failed request-reset reproducibility"

        # A host request must not silently change the subsequent measured route.
        def forbidden_logits_read(_):
            raise AssertionError("Default token-out generation read full logits")

        gen._read_logits = forbidden_logits_read
        before = dict(gen.counters)
        try:
            device_output = gen.generate(short, 3, sampling_params=SamplingParams(temperature=0.0, top_k=1, top_p=1.0))
        finally:
            gen._read_logits = original_read_logits
        assert len(device_output) == 3 and gen.counters["sampling_replays"] == before["sampling_replays"] + 3
        assert gen.counters["token_refreshes"] == before["token_refreshes"], "Device mode fed tokens back from host"
        report["checks"][current] = {
            "passed": True,
            "host_tokens": host_runs,
            "device_tokens_after_host_request": device_output,
            "host_draws_used_full_vocabulary": True,
            "host_device_rng_unchanged": True,
            "default_route_restored_without_full_logits_or_host_feedback": True,
        }

        current = "failed_capture_cleanup"
        before_ids = trace_ids(gen)
        before_state = mutable_cache(gen.caches[1])

        def interrupted_capture():
            # This already-warmed copy is captured, then the Python body fails.
            # No device fault, novel allocation, or incomplete kernel is used.
            ttnn.copy(gen.decode_tokens, gen.saved_tokens)
            raise RuntimeError("serving_probe_deliberate_capture_failure")

        try:
            gen._record_trace(interrupted_capture)
        except RuntimeError as error:
            assert str(error) == "serving_probe_deliberate_capture_failure"
        else:
            raise AssertionError("Deliberate capture exception was swallowed")
        assert trace_ids(gen) == before_ids, "Failed capture entered persistent trace dictionaries"
        equal_cache(before_state, mutable_cache(gen.caches[1]), "Failed capture must not execute")
        prefill([short], (1,))
        decode([42], [len(short)], slots=(1,), sample_on_device=True)
        report["checks"][current] = {"passed": True, "subsequent_prefill_and_decode_replay": True}
        del before_state

        current = "tracker_positive_control"
        # The preceding successful replay establishes a clean baseline. This
        # allocation alone must make the same real model trace unsafe.
        unsafe = ttnn.empty((1, 1, 32, 32), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        rejected = False
        try:
            gen._replay_decode((1,))
        except RuntimeError as exc:
            rejected = "device buffer(s) still alive before" in str(exc)
            report["positive_control_error"] = str(exc)
            positive_log = args.output.parent / "trace_alloc_tracking" / "positive_control.log"
            positive_log.parent.mkdir(parents=True, exist_ok=True)
            positive_log.write_text(
                "TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0\n"
                "Deliberate live allocation before real reduced-model replay:\n" + str(exc) + "\n"
            )
            report["positive_control_log"] = str(positive_log)
        finally:
            ttnn.deallocate(unsafe)
            del unsafe
        assert rejected, "Live postcapture allocation was not rejected by the tracker"
        # End the workload with model replay, including program-cache checks.
        decode([1234], [97], slots=(1,), sample_on_device=True, reload_inputs=False)
        report["checks"][current] = {"passed": True, "rejected_live_allocation": True, "final_clean_replay": True}
        report["program_cache_entries_final"] = mesh.num_program_cache_entries()
        report["counters_before_teardown"] = dict(gen.counters)
        report["trace_ids_before_teardown"] = trace_ids(gen)
        current = "persistent_capture_unload"
        expected_releases = sum(len(group) for group in trace_ids(gen).values())
        before_releases = gen.counters["releases"]
        gen.release_persistent_capture()
        assert all(not group for group in trace_ids(gen).values()), "Unload retained generator traces"
        assert (
            gen.counters["releases"] == before_releases + expected_releases
        ), "Unload did not release every capture once"
        gen.release_persistent_capture()
        gen.close()
        assert gen.counters["releases"] == before_releases + expected_releases, "Repeated close double-released traces"
        report["checks"][current] = {"passed": True, "released_trace_count": expected_releases, "idempotent": True}
        report["passed"] = True
    except BaseException as exc:
        report["failed_check"] = current
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        cleanup_errors = []
        if gen is not None:
            try:
                gen.close()
            except BaseException as error:
                cleanup_errors.append(f"generator.close: {type(error).__name__}: {error}")
        if mesh is not None:
            try:
                ttnn.close_mesh_device(mesh)
                report["caller_closed_mesh"] = True
            except BaseException as error:
                cleanup_errors.append(f"mesh.close: {type(error).__name__}: {error}")
        if cleanup_errors:
            report["passed"] = False
            report["cleanup_errors"] = cleanup_errors
        report["wall_seconds"] = time.monotonic() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if cleanup_errors and "error" not in report:
            raise RuntimeError("; ".join(cleanup_errors))
    print(
        json.dumps({"passed": report["passed"], "checks": list(report["checks"]), "output": str(args.output)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
