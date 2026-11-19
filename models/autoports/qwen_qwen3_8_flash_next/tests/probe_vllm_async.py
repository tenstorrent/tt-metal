# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Adapter reload/async correctness with scheduler-owned shared KV storage.

Run only under the device owner's serialization and a bounded timeout, using
the pinned serving environment. This diagnostic synchronizes and reads cache
state; it is not performance or full-model acceptance evidence. For example::

    QWEN_VLLM_LAYER_INDICES=0,1,3 TT_METAL_TRACE_ALLOC_TRACKING=1 \
      TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 timeout 3600 python -m \
      models.autoports.qwen_qwen3_8_flash_next.tests.probe_vllm_async \
      --output models/autoports/qwen_qwen3_8_flash_next/doc/vllm_integration/adapter_async.json

The caller must restore the accepted mesh descriptor after sourcing lane-env.
All model calls pass the exact opaque pool returned by the adapter allocator.
Host token feedback appears only in explicitly labelled authoritative controls.
"""

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.generator import DecodeSubmission
from models.autoports.qwen_qwen3_8_flash_next.tt.generator_vllm import QwenFlashNextForCausalLM
from models.common.sampling.sampling_params import SamplingParams


def read_ranks(tensor):
    return [ttnn.to_torch(part).clone() for part in ttnn.get_device_tensors(tensor)]


def snapshot(cache, pages):
    """Private mutable state plus this request's pages, never other pool pages."""
    result = {}
    for index, layer in enumerate(cache.layers):
        for name in ("conv", "recurrent", "ple_conv", "token_history"):
            tensor = getattr(layer, name, None)
            if tensor is not None:
                result[f"{index}.{name}"] = read_ranks(tensor)
        if layer.attention is not None:
            for name in ("position", "page_table", "raw_history", "compressed_keys", "k_cache", "v_cache"):
                values = read_ranks(getattr(layer.attention, name))
                if name in ("k_cache", "v_cache"):
                    values = [value[pages].clone() for value in values]
                result[f"{index}.attention.{name}"] = values
    return result


def equal_state(expected, actual, label):
    assert expected.keys() == actual.keys(), f"{label}: state fields changed"
    for name in expected:
        for rank, (left, right) in enumerate(zip(expected[name], actual[name], strict=True)):
            assert torch.equal(left, right), f"{label}: {name} rank {rank} changed"


def check_position(cache, expected):
    assert cache.position == expected, f"Host position {cache.position} != {expected}"
    observed = []
    for index, layer in enumerate(cache.layers):
        if layer.attention is not None:
            for rank, value in enumerate(read_ranks(layer.attention.position)):
                assert value.flatten().tolist() == [expected], f"Layer {index} rank {rank}: stale device position"
                observed.append(dict(layer=index, rank=rank, position=expected))
    assert observed, "The diagnostic must include an indexed-attention layer"
    return observed


def counter_delta(before, after):
    return {name: after[name] - value for name, value in before.items() if after[name] != value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output artifact path")
    from vllm.transformers_utils.config import get_config

    os.environ.setdefault("QWEN_VLLM_LAYER_INDICES", "0,1,3")
    checkpoint = os.environ.get(
        "QWEN_CHECKPOINT",
        "/home/ttuser/qwen38-flash/hf/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/812d96e10a0dd82b69d749e68693698cf6c36daa",
    )
    report = dict(
        passed=False,
        scope="adapter_device_diagnostic_not_server_acceptance",
        regime="serial",
        context=128,
        max_batch_size=2,
        pool_blocks=16,
        selected_layers=os.environ["QWEN_VLLM_LAYER_INDICES"],
        sampling_mode="canonical_device_greedy_token_out",
        tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING"),
        skip_program_cache=os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"),
        checks={},
    )
    mesh = adapter = None
    current = "setup"
    started = time.monotonic()
    try:
        assert report["tracking"] == "1", "Enable trace allocation tracking before Python starts"
        assert report["skip_program_cache"] == "0", "Include program cache allocations in tracking"
        assert os.environ.get("TT_MESH_GRAPH_DESC_PATH"), "Restore the accepted mesh descriptor"
        torch.set_num_threads(8)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0, l1_small_size=32768)
        adapter = QwenFlashNextForCausalLM.initialize_vllm_model(
            get_config(checkpoint, trust_remote_code=False), mesh, 2, 128
        )
        pool = adapter.allocate_kv_cache_per_layer([((16, 1, 32, 256), torch.bfloat16, i) for i in range(12)])
        gen = adapter.generator
        assert pool is gen.shared_pool
        for cache in gen.caches:
            for layer in cache.layers:
                if layer.attention is not None:
                    assert id(layer.attention.k_cache) in pool.shared_tensor_ids
                    assert id(layer.attention.v_cache) in pool.shared_tensor_ids
        report["scheduler_pool"] = dict(
            exact_allocator_return_passed_to_all_calls=True,
            all_request_kv_tensors_alias_pool=True,
            tensor_indices=list(pool.pairs),
            bytes_per_rank=pool.bytes_per_rank,
        )
        adapter.warmup_model_prefill(kv_cache=pool)
        adapter.warmup_model_decode(kv_cache=pool)
        report["prepared_seconds"] = time.monotonic() - started
        report["initial_counters"] = dict(gen.counters)
        sampling = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
        # Reserve disjoint physical pages for both scheduler slots, with spare
        # page13 available to grow slot1's second logical page.
        tables = torch.tensor([[1, 2, 3, 4], [9, 10, 11, 12]], dtype=torch.int32)
        short = [785, 13458, 13]
        prompts = (short, [785, 13, 13458, 13, 785] * 6 + [785, 13, 13458])

        def prefill(rows, slots):
            tokens = torch.zeros(len(rows), max(map(len, rows)), dtype=torch.int64)
            for row, prompt in enumerate(rows):
                tokens[row, : len(prompt)] = torch.tensor(prompt)
            result, rope_delta = adapter.prefill_forward(
                tokens=tokens,
                prompt_lens=list(map(len, rows)),
                start_pos=[0] * len(rows),
                empty_slots=slots,
                page_table=tables[list(slots)],
                kv_cache=pool,
                sampling_params=sampling,
            )
            assert rope_delta.tolist() == [0] * len(rows)
            return result

        def decode(tokens, positions, slots=(1,), table=None, **kwargs):
            return adapter.decode_forward(
                tokens=torch.tensor(tokens, dtype=torch.int64),
                start_pos=positions,
                slots=slots,
                page_table=tables[list(slots)] if table is None else table,
                kv_cache=pool,
                sampling_params=sampling,
                **kwargs,
            )

        current = "stale_host_tokens_and_positions"
        prefill([short], (1,))
        first = decode([42], [3])
        stale_token = (int(first[0]) + 1) % gen.model.config.vocab_size
        token_address = gen.decode_tokens.buffer_address()
        before = dict(gen.counters)
        second = decode([stale_token], [97], reload_inputs=False)
        delta = counter_delta(before, gen.counters)
        for name in ("token_refreshes", "position_refreshes", "page_table_refreshes"):
            assert gen.counters[name] == before[name], f"Stale-input replay refreshed {name}"
        assert gen.decode_tokens.buffer_address() == token_address
        observed_positions = check_position(gen.caches[1], 5)
        for rank, value in enumerate(read_ranks(gen.decode_tokens)):
            assert int(value.flatten()[1]) == int(second[0]), f"Rank {rank}: feedback did not stay resident"
        replay_state = snapshot(gen.caches[1], tables[1].tolist())
        # Authoritative host reload is a diagnostic control only.
        prefill([short], (1,))
        control_first = decode([42], [3])
        control_second = decode(control_first.tolist(), [4], reload_inputs=True)
        assert torch.equal(first, control_first) and torch.equal(second, control_second)
        equal_state(replay_state, snapshot(gen.caches[1], tables[1].tolist()), current)
        report["checks"][current] = dict(
            passed=True,
            slot=1,
            stale_host_token=stale_token,
            stale_host_position=97,
            authoritative_positions=[3, 4],
            tokens=[int(first[0]), int(second[0])],
            counter_delta=delta,
            device_positions=observed_positions,
            persistent_token_address=token_address,
            all_request_state_matches_reload_control=True,
        )

        current = "page_table_only_growth"
        prefill([short * 10], (1,))
        tokens = [int(decode([42], [30])[0])]
        before = dict(gen.counters)
        tokens.append(int(decode([stale_token], [97], reload_inputs=False, reload_page_table=True)[0]))
        assert gen.counters["page_table_refreshes"] == before["page_table_refreshes"], "Unchanged table copied"
        changed = torch.tensor([[9, 13, 11, 12]], dtype=torch.int32)
        physical_before = {}
        for index, layer in enumerate(gen.caches[1].layers):
            if layer.attention is not None:
                for name in ("k_cache", "v_cache"):
                    physical_before[index, name] = read_ranks(getattr(layer.attention, name))
        tokens.append(int(decode([stale_token], [97], table=changed, reload_inputs=False, reload_page_table=True)[0]))
        delta = counter_delta(before, gen.counters)
        assert delta.get("page_table_refreshes") == 1, "Changed scheduler table must copy exactly once"
        for name in ("token_refreshes", "position_refreshes"):
            assert gen.counters[name] == before[name], f"Page-only growth refreshed {name}"
        physical_checks = []
        for index, layer in enumerate(gen.caches[1].layers):
            if layer.attention is not None:
                for value in read_ranks(layer.attention.page_table):
                    assert torch.equal(value.int(), changed), "Captured attention table did not receive growth"
                for name in ("k_cache", "v_cache"):
                    for rank, value in enumerate(read_ranks(getattr(layer.attention, name))):
                        previous = physical_before[index, name][rank]
                        assert torch.equal(value[10], previous[10]), f"{name} rank{rank}: wrote old page10"
                        assert not torch.equal(value[13], previous[13]), f"{name} rank{rank}: new page13 unwritten"
                        physical_checks.append(dict(layer=index, tensor=name, rank=rank, written=13, untouched=10))
        observed_positions = check_position(gen.caches[1], 33)
        prefill([short * 10], (1,))
        control = [int(decode([42], [30])[0])]
        for _ in range(2):
            control.append(int(decode([stale_token], [97], reload_inputs=False)[0]))
        assert tokens == control, "New physical mapping changed the continuation"
        report["checks"][current] = dict(
            passed=True,
            slot=1,
            logical_write_position=32,
            original_mapping=tables[1].tolist(),
            grown_mapping=changed[0].tolist(),
            physical_checks=physical_checks,
            tokens=tokens,
            counter_delta=delta,
            device_positions=observed_positions,
        )
        del physical_before

        current = "inactive_slot_preservation"
        prefill(prompts, (0, 1))
        inactive = snapshot(gen.caches[0], tables[0].tolist())
        result = decode([999, 43], [-1, len(prompts[1])], slots=(0, 1))
        assert result.shape == (2,) and int(result[0]) == -1
        equal_state(inactive, snapshot(gen.caches[0], tables[0].tolist()), current)
        check_position(gen.caches[0], len(prompts[0]))
        check_position(gen.caches[1], len(prompts[1]) + 1)
        report["checks"][current] = dict(passed=True, inactive_slot=0, active_slot=1, output=result.tolist())

        current = "two_queued_async_snapshots"
        slots = (1, 0)
        first_positions = [len(prompts[1]), len(prompts[0])]
        second_positions = [len(prompts[1]) + 1, -1]
        prefill(prompts[::-1], slots)
        control_first = decode([43, 42], first_positions, slots=slots).unsqueeze(1)
        control_second = decode([44, 777], second_positions, slots=slots).unsqueeze(1)
        assert not torch.equal(control_first, control_second), "Async overwrite control must distinguish reads"
        control_states = [snapshot(cache, tables[slot].tolist()) for slot, cache in enumerate(gen.caches)]
        prefill(prompts[::-1], slots)
        before = dict(gen.counters)
        first_submission = decode([43, 42], first_positions, slots=slots, read_from_device=False)
        assert isinstance(first_submission, DecodeSubmission) and first_submission.is_tokens
        assert first_submission.slots == slots and first_submission.active_rows == (0, 1)
        assert gen.counters["readbacks"] == before["readbacks"], "Deferred decode performed an implicit read"
        first_host, first_events = adapter.read_decode_output(first_submission, async_read=True)
        second_submission = decode([44, 777], second_positions, slots=slots, read_from_device=False)
        assert isinstance(second_submission, DecodeSubmission) and second_submission.is_tokens
        assert second_submission.slots == slots and second_submission.active_rows == (0,)
        assert first_submission.tensor.buffer_address() == second_submission.tensor.buffer_address()
        second_host, second_events = adapter.read_decode_output(second_submission, async_read=True)
        assert first_events and second_events, "Async reads must return completion events"
        assert first_host.tensor is not second_host.tensor, "Async reads reused a host tensor handle"
        assert first_host.tensor.storage_type() == second_host.tensor.storage_type() == ttnn.StorageType.HOST
        assert gen.counters["readbacks"] == before["readbacks"] + 2
        assert gen.counters["synchronizations"] == before["synchronizations"], "Async submission synchronized"
        assert gen.counters["captures"] == before["captures"], "Known decode signatures recaptured"
        assert gen.counters["releases"] == before["releases"], "Async submission retired a live trace"
        queued_delta = counter_delta(before, gen.counters)
        # Both reads were queued before the first wait. Format in reverse order
        # to catch mutable data and metadata accidentally shared across handles.
        for event in (*first_events, *second_events):
            ttnn.event_synchronize(event)
        second_output = adapter.process_decode_output_host(second_host)
        first_output = adapter.process_decode_output_host(first_host)
        assert tuple(first_output.shape) == tuple(second_output.shape) == (2, 1)
        assert torch.equal(first_output, control_first), "Earlier snapshot overwritten by later decode"
        assert torch.equal(second_output, control_second), "Later snapshot differs from synchronous control"
        assert first_host.slots == slots and first_host.active_rows == (0, 1)
        assert second_host.slots == slots and second_host.active_rows == (0,)
        assert torch.equal(
            adapter.process_decode_output_host(first_host), first_output
        ), "Host processing mutated handle"
        assert int(second_output[1, 0]) == -1, "Inactive scheduler row lost during formatting"
        for slot, cache in enumerate(gen.caches):
            equal_state(control_states[slot], snapshot(cache, tables[slot].tolist()), f"Async final slot{slot}")
        report["checks"][current] = dict(
            passed=True,
            slot_order=list(slots),
            active_rows=[[0, 1], [0]],
            queued_before_wait=2,
            format_order=[2, 1, 1],
            outputs=[first_output.tolist(), second_output.tolist()],
            same_device_output_address=first_submission.tensor.buffer_address(),
            distinct_host_handles=True,
            immutable_submission_metadata=True,
            final_state_matches_synchronous_controls=True,
            queued_counter_delta=queued_delta,
        )
        report["final_counters"] = dict(gen.counters)
        report["program_cache_entries"] = mesh.num_program_cache_entries()
        report["passed"] = True
    except Exception:
        report["failed_check"] = current
        report["error"] = traceback.format_exc()
        raise
    finally:
        try:
            if adapter is not None:
                adapter.release_persistent_capture()
            report["capture_cleanup_completed"] = True
        except Exception:
            report["passed"] = False
            report["capture_cleanup_error"] = traceback.format_exc()
            raise
        finally:
            try:
                if mesh is not None:
                    ttnn.close_mesh_device(mesh)
                report["mesh_close_completed"] = True
            except Exception:
                report["passed"] = False
                report["mesh_close_error"] = traceback.format_exc()
                raise
            finally:
                report["seconds"] = time.monotonic() - started
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("ADAPTER_ASYNC_PROBE_PASS", flush=True)


if __name__ == "__main__":
    main()
