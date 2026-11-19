# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real layers 0/3 serving command/ownership probe; externally serialize/timeout.

Calls the full-model generator directly, without a vLLM adapter. This is a
reduced-layer state/trace contract test, not full-model quality/performance.
"""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path


class ForbiddenStaleHost:
    def __getattribute__(self, name):
        raise AssertionError(f"Resident decode accessed stale host argument: {name}")

    def __len__(self):
        raise AssertionError("Resident decode sized stale host argument")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import torch

    import ttnn
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator
    from models.common.sampling.generator import SamplingParams

    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    gen = None
    result = {"status": "running", "layers": [0, 3], "batch": 3, "inactive_slot": 1}
    try:
        gen = build_generator(None, mesh, layer_indices=[0, 3], max_batch_size=3, max_seq_len=256)
        params = SamplingParams(temperature=[0.8] * 3, top_k=32, top_p=0.9, seed=[42, 43, 44])
        lengths, inactive, pages = [33, 35, 37], 1, 8
        prompts = torch.zeros((3, max(lengths)), dtype=torch.int64)
        for slot, length in enumerate(lengths):
            prompts[slot, :length] = torch.arange(length) + 100 + slot * 41
        result["phase"] = "mixed_prefill"
        gen.prefill_forward(prompts, page_table=gen.page_table, kv_cache=gen.cache, prompt_lens=lengths)
        buffers = {
            "conv": (gen.cache[0][0].conv, 3),
            "recurrent": (gen.cache[0][0].recurrent, 1),
            "key": (gen.cache[1][0].key, 1),
            "value": (gen.cache[1][0].value, 1),
        }
        initial = {name: ttnn.clone(value) for name, (value, _) in buffers.items()}

        def restore_initial():
            for name, (value, _) in buffers.items():
                ttnn.copy(initial[name], value)

        def state_host():
            return {
                name: ttnn.to_torch(value, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=dim))
                for name, (value, dim) in buffers.items()
            }

        def row(value, name, slot):
            if name == "conv":
                return value[:, :, slot : slot + 1, :]
            if name == "recurrent":
                return value[slot : slot + 1]
            return value[slot * pages : (slot + 1) * pages]

        def hashes(snapshot):
            return {
                name: [
                    hashlib.sha256(row(value, name, slot).contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                    for slot in range(3)
                ]
                for name, value in snapshot.items()
            }

        def assert_state_equal(actual, expected, slots=range(3)):
            for name in buffers:
                for slot in slots:
                    assert torch.equal(row(actual[name], name, slot), row(expected[name], name, slot)), (name, slot)

        def seed_host():
            return ttnn.to_torch(ttnn.get_device_tensors(gen.sampler.tt_sampling.seeds_tt_tensor)[0]).clone()

        def resident_positions():
            state = gen.states[(id(gen.cache), (0, 1, 2))]
            return [int(ttnn.to_torch(ttnn.get_device_tensors(pos)[0]).item()) for pos in state.positions]

        def finish(handle, *, is_tokens=True):
            before, seed = dict(gen.counters), seed_host()
            unqueued = gen.submissions.get(handle.submission_id, (None, None, 1))[2] is None
            read_handle, events = gen.read_decode_output(handle, async_read=True)
            assert read_handle is handle and events, "Async read must retain exact submission and transfer events"
            for event in events:
                ttnn.event_synchronize(event)
            output = gen.process_decode_output_host(read_handle, is_tokens=is_tokens)
            repeated = gen.process_decode_output_host(read_handle, is_tokens=is_tokens)
            assert output is repeated, "Completed read must be idempotent"
            expected_counters = dict(before)
            expected_counters["readbacks"] += int(unqueued)
            assert gen.counters == expected_counters, "Readback may only queue the one deferred transfer"
            assert torch.equal(seed_host(), seed), "Readback mutated RNG state"
            expected_shape = (3, 1) if is_tokens else (3, 1, gen.model.vocab_size)
            assert tuple(output.shape) == expected_shape
            return output.clone()

        tokens = torch.tensor([[300], [301], [302]])
        positions = [33, -1, 37]
        stale = ForbiddenStaleHost()

        def admit(**updates):
            kwargs = {
                "page_table": gen.page_table,
                "kv_cache": gen.cache,
                "sampling_params": params,
                "read_from_device": False,
                "reload_inputs": True,
                "reload_sampling_params": True,
                "reset_sampling_state": True,
                "prompt_tokens": prompts,
            }
            kwargs.update(updates)
            reads = gen.counters["readbacks"]
            handle = gen.decode_forward(tokens, positions, **kwargs)
            assert gen.counters["readbacks"] == reads
            assert isinstance(handle.token_tensor, ttnn.Tensor)
            assert gen.submissions[handle.submission_id][1:] == (None, None)
            return handle

        def steady(**updates):
            kwargs = {
                "page_table": stale,
                "kv_cache": gen.cache,
                "sampling_params": params,
                "read_from_device": False,
                "reload_inputs": False,
            }
            kwargs.update(updates)
            reads = gen.counters["readbacks"]
            handle = gen.decode_forward(stale, stale, **kwargs)
            assert gen.counters["readbacks"] == reads
            assert isinstance(handle.token_tensor, ttnn.Tensor)
            assert gen.submissions[handle.submission_id][1:] == (None, None)
            return handle

        result["phase"] = "synchronous_reference"
        before = state_host()
        expected = [finish(admit())]
        first_state, first_seed = state_host(), seed_host()
        assert_state_equal(first_state, before, [inactive])
        expected.append(finish(steady()))
        second_state = state_host()
        assert not torch.equal(
            expected[0][[0, 2]], expected[1][[0, 2]]
        ), "Ownership stimulus produced identical active tokens; use a discriminating seed/prompt before claiming pass"
        assert_state_equal(second_state, before, [inactive])
        assert seed_host().reshape(-1)[inactive] == first_seed.reshape(-1)[inactive]
        expected.append(finish(steady()))
        third_state = state_host()
        assert resident_positions() == [36, -1, 40]

        result["phase"] = "invalid_transition_preflight"
        bad_tokens = tokens.clone()
        bad_tokens[0, 0] = gen.model.vocab_size
        bad_table = gen.page_table.clone()
        bad_table[0, -1] = gen.cache[1][0].key.shape[0]
        rejected = []
        for name, changes in (
            ("token_range", {"tokens": bad_tokens}),
            ("position_range", {"start_pos": [-2, -1, 37]}),
            ("position_dtype", {"start_pos": [33.5, -1.0, 37.0]}),
            ("table_range", {"page_table": bad_table}),
            ("table_dtype", {"page_table": gen.page_table.float()}),
            ("table_geometry", {"page_table": gen.page_table[:, :-1]}),
            ("temperature_nan", {"sampling_params": replace(params, temperature=[float("nan")] * 3)}),
            ("top_p_range", {"sampling_params": replace(params, top_p=1.01)}),
            ("top_k_unsupported", {"sampling_params": replace(params, top_k=65)}),
            ("history_token_range", {"output_tokens": [[gen.model.vocab_size], [], []]}),
            ("history_rows", {"prompt_tokens": [[1], [2]]}),
        ):
            kwargs = {
                "tokens": tokens,
                "start_pos": positions,
                "page_table": gen.page_table,
                "kv_cache": gen.cache,
                "sampling_params": params,
                "read_from_device": False,
                "reload_inputs": True,
                "reload_sampling_params": True,
                "reset_sampling_state": True,
                "slot_remap": [2, 1, 0],
                "prompt_tokens": prompts,
            }
            kwargs.update(changes)
            counters, seeds_before = dict(gen.counters), seed_host()
            try:
                gen.decode_forward(**kwargs)
            except (TypeError, ValueError):
                rejected.append(name)
            else:
                raise AssertionError(f"Malformed transition was accepted: {name}")
            assert gen.counters == counters, (name, "mutated counters")
            assert torch.equal(seed_host(), seeds_before), (name, "mutated seeds")
            assert_state_equal(state_host(), third_state)
            assert resident_positions() == [36, -1, 40]
        result["invalid_transitions_rejected_before_mutation"] = rejected

        result["phase"] = "two_pending_submissions"
        restore_initial()
        first = admit()
        counters = dict(gen.counters)
        try:
            steady()
        except ValueError as error:
            assert "Queue deferred output reads" in str(error)
        else:
            raise AssertionError("Unqueued borrowed output must prevent overwrite")
        assert gen.counters == counters
        result["deferred_reads_and_overwrite_guard"] = "pass"
        first_read, first_events = gen.read_decode_output(first, async_read=True)
        counters = dict(gen.counters)
        # Deliberately submit again before processing or waiting on the first.
        second = steady()
        assert first.submission_id != second.submission_id
        assert first.completion is None and second.completion is None
        for name in ("token_refreshes", "position_refreshes", "page_table_refreshes", "captures"):
            assert gen.counters[name] == counters[name], name
        assert gen.counters["model_replays"] == counters["model_replays"] + 1
        assert gen.counters["sampling_replays"] == counters["sampling_replays"] + 1
        for event in first_events:
            ttnn.event_synchronize(event)
        actual_first = finish(first_read)
        actual_second = finish(second)
        print(
            "PENDING_OUTPUT_CONTROL",
            json.dumps(
                {
                    "first": actual_first.tolist(),
                    "expected_first": expected[0].tolist(),
                    "second": actual_second.tolist(),
                    "expected_second": expected[1].tolist(),
                }
            ),
            flush=True,
        )
        assert torch.equal(actual_first, expected[0])
        assert torch.equal(actual_second, expected[1])
        assert_state_equal(state_host(), second_state)
        assert resident_positions() == [35, -1, 39]

        result["phase"] = "page_only_reload"
        changed = gen.page_table.clone()
        changed[0, [-2, -1]] = changed[0, [-1, -2]]  # One changed row; same causal history.
        counters = dict(gen.counters)
        actual_third = finish(steady(page_table=changed, reload_page_table=True))
        assert torch.equal(actual_third, expected[2])
        assert_state_equal(state_host(), third_state)
        assert gen.counters["page_table_refreshes"] == counters["page_table_refreshes"] + 1
        for name in ("token_refreshes", "position_refreshes", "captures"):
            assert gen.counters[name] == counters[name], name
        counters = dict(gen.counters)
        finish(steady(page_table=changed.clone(), reload_page_table=False))
        assert gen.counters["page_table_refreshes"] == counters["page_table_refreshes"]
        counters = dict(gen.counters)
        finish(steady(page_table=changed.clone(), reload_page_table=True))
        # An explicit reload with an equal clone must not transfer any row.
        assert gen.counters["page_table_refreshes"] == counters["page_table_refreshes"]
        for name in ("token_refreshes", "position_refreshes", "captures"):
            assert gen.counters[name] == counters[name], name
        assert_state_equal(state_host(), before, [inactive])
        assert resident_positions() == [38, -1, 42]

        result["phase"] = "whole_pool_remap_after_drain"
        premove, seed = state_host(), seed_host()
        remap = [2, 1, 0]
        moved = gen.decode_forward(
            tokens,
            [-1, -1, -1],
            page_table=changed,
            kv_cache=gen.cache,
            sampling_params=params,
            read_from_device=False,
            reload_inputs=True,
            slot_remap=remap,
        )
        finish(moved)
        postmove = state_host()
        for name in ("conv", "recurrent"):
            for destination, source in enumerate(remap):
                assert torch.equal(row(postmove[name], name, destination), row(premove[name], name, source))
        for name in ("key", "value"):
            assert torch.equal(postmove[name], premove[name]), "Logical slot remap must not move physical KV pages"
        assert torch.equal(seed_host().reshape(-1)[:3], seed.reshape(-1)[remap])
        assert resident_positions() == [-1, -1, -1]

        result["phase"] = "single_slot_prefill_reset"
        gen.prefill_forward(
            prompts[:1],
            page_table=gen.page_table[:1],
            kv_cache=gen.cache,
            prompt_lens=[lengths[0]],
            start_pos=[0],
            empty_slots=[0],
        )
        postreset = state_host()
        assert_state_equal(postreset, postmove, [1, 2])
        for name in ("conv", "recurrent"):
            assert torch.equal(row(postreset[name], name, 0), row(before[name], name, 0)), name
        for name in ("key", "value"):
            actual = row(postreset[name], name, 0).permute(1, 0, 2, 3).flatten(1, 2)[:, : lengths[0]]
            original = row(before[name], name, 0).permute(1, 0, 2, 3).flatten(1, 2)[:, : lengths[0]]
            assert torch.equal(actual, original), name

        result["phase"] = "host_sampling_default"
        restore_initial()
        counters = dict(gen.counters)
        host_handle = gen.decode_forward(
            tokens,
            positions,
            page_table=gen.page_table,
            kv_cache=gen.cache,
            read_from_device=False,
            reload_inputs=True,
        )  # Absent sampling_params explicitly selects the host-logits contract.
        host_logits = finish(host_handle, is_tokens=False)
        assert host_logits.dtype == torch.float32 and torch.isfinite(host_logits[[0, 2]]).all()
        assert gen.counters["sampling_replays"] == counters["sampling_replays"]
        assert_state_equal(state_host(), before, [inactive])
        original_host = gen.process_decode_output_host(host_handle, is_tokens=False)
        original_host[0, 0, 0] = 12345  # The actual returned full-logits buffer is writable.
        owned_host = original_host.clone()
        next_host = gen.decode_forward(
            tokens,
            [34, -1, 38],
            page_table=gen.page_table,
            kv_cache=gen.cache,
            read_from_device=False,
            reload_inputs=True,
        )
        finish(next_host, is_tokens=False)
        assert torch.equal(original_host[[0, 2]], owned_host[[0, 2]]), "Later host submission overwrote prior output"
        assert gen.counters["sampling_replays"] == counters["sampling_replays"]

        result["phase"] = "resident_remap_without_reload"
        restore_initial()
        finish(admit())
        oracle_output = finish(steady())
        oracle_state, oracle_seed = state_host(), seed_host()
        restore_initial()
        finish(admit())
        state = gen.states[(id(gen.cache), (0, 1, 2))]
        old_tokens = ttnn.to_torch(ttnn.get_device_tensors(state.tokens)[0]).reshape(-1)[:3].clone()
        old_positions, old_tables = resident_positions(), state.host_tables.clone()
        resident_remap = [1, 2, 0]  # Move the inactive gap as well as both live requests.
        observed = {}
        original_execute = gen._execute

        def inspect_remapped_inputs(current, *, host_sampling=False):
            # Diagnostic reads outside any performance claim: observe inputs
            # immediately before the original production trace submission.
            observed["tokens"] = ttnn.to_torch(ttnn.get_device_tensors(current.tokens)[0]).reshape(-1)[:3].clone()
            observed["positions"] = resident_positions()
            observed["tables"] = torch.cat(
                [ttnn.to_torch(ttnn.get_device_tensors(table)[0]) for table in current.tables], dim=0
            )
            original_execute(current, host_sampling=host_sampling)

        counters = dict(gen.counters)
        gen._execute = inspect_remapped_inputs
        try:
            remapped_output = finish(steady(slot_remap=resident_remap))
        finally:
            gen._execute = original_execute
        assert torch.equal(observed["tokens"], old_tokens[resident_remap])
        assert observed["positions"] == [old_positions[source] for source in resident_remap]
        assert torch.equal(observed["tables"], old_tables[resident_remap])
        assert torch.equal(state.host_tables, old_tables[resident_remap])
        assert state.batch_state.active == (False, True, True) and state.masked
        assert resident_positions() == [-1, 39, 35]
        assert torch.equal(remapped_output[[1, 2]], oracle_output[[2, 0]])
        remapped_state = state_host()
        for name in ("conv", "recurrent"):
            for destination, source in enumerate(resident_remap):
                assert torch.equal(row(remapped_state[name], name, destination), row(oracle_state[name], name, source))
        for name in ("key", "value"):
            assert torch.equal(
                remapped_state[name], oracle_state[name]
            ), "Global KV pages must retain physical ownership"
        assert torch.equal(seed_host().reshape(-1)[:3], oracle_seed.reshape(-1)[resident_remap])
        for name in ("token_refreshes", "position_refreshes", "page_table_refreshes", "captures"):
            assert gen.counters[name] == counters[name], name
        result.update(
            status="pass",
            phase="complete",
            mixed_prompt_lengths=lengths,
            reference_tokens=[value.tolist() for value in expected],
            initial_hashes=hashes(before),
            first_decode_hashes=hashes(first_state),
            two_pending_submissions=True,
            stale_host_inputs_unread=True,
            changed_and_unchanged_page_tables=True,
            forced_same_page_table_reload=True,
            read_is_observational=True,
            inactive_state_and_seed_preserved=True,
            whole_pool_remap=remap,
            resident_remap_without_host_reload=resident_remap,
            single_slot_reset_preserves_peers=True,
            host_logits_shape=list(original_host.shape),
            counters=dict(gen.counters),
            limitation="Reduced real layers 0/3; no full-model accuracy or throughput claim.",
        )
    except Exception as error:
        result.update(status="fail", error=repr(error))
        raise
    finally:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
