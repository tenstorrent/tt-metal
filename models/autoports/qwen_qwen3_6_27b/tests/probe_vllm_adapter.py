# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Representative adapter mask, deferred readback and resident-input contract."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.generator_vllm import DecodeOutput, QwenVLLM
from models.autoports.qwen_qwen3_6_27b.tt.page_table_ownership import page_table_for_attention


def first(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).flatten().clone()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-prepare", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    adapter = None
    try:
        adapter = QwenVLLM.initialize_vllm_model(None, mesh, 4, 262144)
        cache = adapter.allocate_kv_cache((8196, 1, 32, 256), torch.bfloat16, 16)
        gen = adapter.generator
        assert gen.cache is cache
        # Physical pool and virtual width match serving. This first diagnostic
        # restricts preparation only; the live server uses public full prepare.
        if not args.full_prepare:
            for engine in (gen.prefill_single, gen.prefill_uniform):
                engine.prepare(_diagnostic_lengths=[1, 32, 33, 65])
        adapter.warmup_model_prefill(kv_cache=cache)
        tracking_control = False
        if os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1":
            assert (
                os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "0"
            ), "Acceptance tracking must include program-cache allocations"
            unsafe = gen.model.io._upload(torch.zeros(1, 1, 1, 32), ttnn.uint32)
            try:
                gen._execute("decode")
            except RuntimeError as error:
                assert "will be corrupted on replay" in str(error), str(error)
                tracking_control = True
                print("TRACKING_POSITIVE_CONTROL " + str(error).splitlines()[0], flush=True)
            finally:
                del unsafe
            assert tracking_control, "Trace allocation tracking did not reject a live late allocation"
        table = torch.zeros_like(gen.page_host)
        table[:, :32] = torch.arange(1, 129, dtype=torch.int32).reshape(4, 32)
        tokens = torch.arange(70000, 70130, dtype=torch.int32).reshape(2, 65)
        slots = [3, 1]
        params = SimpleNamespace(temperature=[1.7, 2.3], top_k=[32, 17], top_p=[1.0, 1.0], seed=[0, -1])
        sampler = gen.sampling.tt_sampling
        controls = (
            gen.tokens,
            sampler.seeds_tt_tensor,
            sampler.k_tensor,
            sampler.p_tensor,
            sampler.temp_tensor,
            sampler._greedy_col,
        )

        def snapshot():
            return [first(tensor) for tensor in controls]

        def prefill(ends, starts, mask, *, rows=(0, 1)):
            selected = SimpleNamespace(**{key: [value[row] for row in rows] for key, value in vars(params).items()})
            output = adapter.prefill_forward(
                tokens[list(rows)],
                page_table=table[[slots[row] for row in rows]],
                kv_cache=cache,
                prompt_lens=ends,
                start_pos=starts,
                empty_slots=[slots[row] for row in rows],
                sampling_params=selected,
                sampling_mask=mask,
            )
            if adapter.request_specific_rope:
                output, deltas = output
                assert deltas.tolist() == [0] * len(rows)
            return output

        # Fully intermediate chunks publish state without any sample or D2H.
        gen.configure_sampling(temperature=[0.7] * 4, top_k=[5] * 4, seed=[10, 11, 12, 13])
        gen._write(torch.arange(32, dtype=torch.int32).reshape(1, 1, 1, 32), gen.tokens)
        before = snapshot()
        counters = dict(gen.counters)
        state_before = first(cache[0].recurrent)
        result = prefill([33, 33], [0, 0], [False, False])
        assert result.tolist() == [[0], [0]]
        assert all(torch.equal(a, b) for a, b in zip(before, snapshot()))
        assert not torch.equal(state_before, first(cache[0].recurrent)), "Intermediate state was not published"
        for key in ("sample_replays", "token_readbacks", "full_logits_readbacks", "sampling_param_refreshes"):
            assert gen.counters[key] == counters.get(key, 0), key
        assert gen.request_seeds == [10, 11, 12, 13]

        # Mixed rows sample only the final row; all other sampler slots retain
        # device values. The false row advances recurrent/cache state normally.
        before = snapshot()
        reads = gen.counters["full_logits_readbacks"]
        mixed = prefill([65, 65], [33, 33], [True, False])
        assert mixed[1, 0] == 0
        after = snapshot()
        assert all(torch.equal(a[[0, 1, 2]], b[[0, 1, 2]]) for a, b in zip(before, after))
        assert gen.request_seeds == [10, 11, 12, 0]
        assert gen.counters["full_logits_readbacks"] == reads
        # Same real layer/chunk path, identical seeded final token in isolation.
        prefill([33], [0], [False], rows=(0,))
        solo = prefill([65], [33], [True], rows=(0,))
        assert int(mixed[0, 0]) == int(solo[0, 0]), "Mixed mask changed final-row seeded result"

        # Two enqueued reads must own independent host storage. Known values
        # make this discriminating even when the reduced model repeats tokens.
        pending = []
        for value in (123, 456):
            gen._write(torch.full((1, 1, 1, 32), value, dtype=torch.int32), gen.tokens)
            pending.append(adapter.read_decode_output(DecodeOutput(gen.tokens, True), async_read=True))
        for value, (host, events) in zip((123, 456), pending):
            for event in events:
                ttnn.event_synchronize(event)
            assert adapter.process_decode_output_host(host, is_tokens=True).tolist() == [[value]] * 4
        # Non-overlapped plugin finalization skips read_decode_output entirely.
        assert adapter.process_decode_output_host(DecodeOutput(gen.tokens, True, 1), is_tokens=True).tolist() == [[456]]

        # Force authoritative inputs once, then queue two deferred reads around
        # resident replays whose host token/position/table arguments are stale.
        greedy = SimpleNamespace(temperature=[0.0] * 4, top_k=[1] * 4, top_p=[0.0] * 4, seed=[100] * 4)
        positions = torch.tensor([-1, -1, -1, 65], dtype=torch.int32)
        kwargs = dict(page_table=table, kv_cache=cache, sampling_params=greedy, read_from_device=False)
        output = adapter.decode_forward(
            torch.ones(4, 1, dtype=torch.int32),
            positions,
            reload_sampling_params=True,
            reset_sampling_state=True,
            **kwargs,
        )
        host1, events1 = adapter.read_decode_output(output, async_read=True)
        baseline_counters = dict(gen.counters)
        adapter.decode_forward(
            torch.zeros(4, 1, dtype=torch.int32),
            torch.zeros(4, dtype=torch.int32),
            reload_inputs=False,
            **{**kwargs, "page_table": table.flip(0)},
        )
        host2, events2 = adapter.read_decode_output(output, async_read=True)
        for event in events1 + events2:
            ttnn.event_synchronize(event)
        result1 = adapter.process_decode_output_host(host1, is_tokens=True).clone()
        result2 = adapter.process_decode_output_host(host2, is_tokens=True).clone()
        assert int(first(gen.positions)[3]) == 67
        assert int(first(gen.rope_positions)[3]) == 67
        assert int(first(gen.tokens)[3]) == int(result2[3, 0])
        for key in ("token_refreshes", "position_refreshes", "rope_refreshes", "page_table_refreshes"):
            assert gen.counters[key] == baseline_counters.get(key, 0), key
        # A page-only reload must not replace resident token/position state.
        table[:, 3] = torch.arange(130, 134, dtype=torch.int32)
        adapter.decode_forward(
            torch.zeros(4, 1, dtype=torch.int32),
            torch.zeros(4, dtype=torch.int32),
            reload_inputs=False,
            reload_page_table=True,
            **kwargs,
        )
        assert int(first(gen.positions)[3]) == 68
        assert gen.counters["page_table_refreshes"] == baseline_counters.get("page_table_refreshes", 0) + 1
        assert gen.counters["token_refreshes"] == baseline_counters.get("token_refreshes", 0)
        # Compare B1 against the fixed-capacity trace from the identical
        # prefilling state. Snapshot inactive state to detect prefix overruns.
        prefix_greedy = SimpleNamespace(temperature=[0.0], top_k=[1], top_p=[0.0], seed=[100])

        def prefix_trial(bucket):
            gen.reset()
            initial = adapter.prefill_forward(
                tokens[:1],
                page_table=table[:1],
                kv_cache=cache,
                prompt_lens=[65],
                start_pos=[0],
                empty_slots=[0],
                sampling_params=prefix_greedy,
                sampling_mask=[True],
            )
            if adapter.request_specific_rope:
                initial = initial[0]
            inactive = [first(value).reshape(4, -1)[1:].clone() for value in (cache[0].conv, cache[0].recurrent)]
            inp = torch.zeros(bucket, 1, dtype=torch.int32)
            inp[0] = initial[0]
            pos = torch.full((bucket,), -1, dtype=torch.int32)
            pos[0] = 65
            pp = prefix_greedy if bucket == 1 else greedy
            outputs = []
            table_writes = gen.counters["page_table_refreshes"]
            for step in range(3):
                out = adapter.decode_forward(
                    inp if step == 0 else torch.zeros_like(inp),
                    pos if step == 0 else torch.zeros_like(pos),
                    page_table=table[:bucket],
                    kv_cache=cache,
                    sampling_params=pp,
                    reload_inputs=step == 0,
                    reload_sampling_params=step == 0,
                    reset_sampling_state=step == 0,
                )
                assert out.shape == (bucket, 1)
                outputs.append(int(out[0, 0]))
            if bucket == 1:
                assert gen.full_page_tables_dirty
                assert gen.counters["page_table_refreshes"] == table_writes
            else:
                assert not gen.full_page_tables_dirty
            assert int(first(gen.positions)[0]) == 68
            for before_state, value in zip(inactive, (cache[0].conv, cache[0].recurrent)):
                assert torch.equal(before_state, first(value).reshape(4, -1)[1:]), "Inactive prefix state changed"
            active_state = [first(value).reshape(4, -1)[0].clone() for value in (cache[0].conv, cache[0].recurrent)]
            return outputs, active_state

        fixed_tokens, fixed_state = prefix_trial(4)
        prefix_tokens, prefix_state = prefix_trial(1)
        assert all(torch.equal(a, b) for a, b in zip(fixed_state, prefix_state)), "Prefix active state differs"
        assert fixed_tokens == prefix_tokens, (fixed_tokens, prefix_tokens)
        # Exercise both bucket transitions without resets. Row zero retains
        # its device state; authoritative token/positions delimit transitions.
        for bucket in (4, 1):
            inp = torch.zeros(bucket, 1, dtype=torch.int32)
            inp[0, 0] = int(first(gen.tokens)[0])
            pos = torch.full((bucket,), -1, dtype=torch.int32)
            pos[0] = int(first(gen.positions)[0])
            out = adapter.decode_forward(
                inp,
                pos,
                page_table=table[:bucket],
                kv_cache=cache,
                sampling_params=prefix_greedy if bucket == 1 else greedy,
                reload_inputs=True,
                reload_sampling_params=True,
            )
            assert out.shape == (bucket, 1)
            assert int(first(gen.positions)[0]) == int(pos[0]) + 1
        assert gen.full_page_tables_dirty
        assert torch.equal(first(gen.single_decode_page_table), gen.page_host[0].flatten())
        before_flush = gen.counters["page_table_refreshes"]
        preserved_active = first(cache[0].recurrent).reshape(4, -1)[0].clone()
        adapter.prefill_forward(
            tokens[:1],
            page_table=gen.page_host[1:2].clone(),
            kv_cache=cache,
            prompt_lens=[65],
            start_pos=[0],
            empty_slots=[1],
            sampling_params=prefix_greedy,
            sampling_mask=[False],
        )
        assert torch.equal(preserved_active, first(cache[0].recurrent).reshape(4, -1)[0])
        assert not gen.full_page_tables_dirty
        assert gen.counters["page_table_refreshes"] > before_flush
        attention_state = next(state for layer, state in zip(gen.model.layers, cache) if layer.kind == "full_attention")
        assert torch.equal(first(page_table_for_attention(attention_state)), gen.page_host.flatten())
        assert gen.counters["full_logits_readbacks"] == 0

        def replay_retained_request():
            # Slot0 keeps its existing request while slot3 prefills. Explicit
            # authoritative inputs make this a trace-lifetime check, independent
            # of the async stale-input contract exercised above.
            position = int(first(gen.positions)[0])
            token = int(first(gen.tokens)[0])
            output = adapter.decode_forward(
                torch.tensor([[token]], dtype=torch.int32),
                torch.tensor([position], dtype=torch.int32),
                page_table=gen.page_host[:1].clone(),
                kv_cache=cache,
                sampling_params=prefix_greedy,
                reload_inputs=True,
                reload_sampling_params=True,
                read_from_device=False,
            )
            host, events = adapter.read_decode_output(output, async_read=True)
            for event in events:
                ttnn.event_synchronize(event)
            decoded = adapter.process_decode_output_host(host, is_tokens=True)
            assert decoded.shape == (1, 1)
            assert int(first(gen.positions)[0]) == position + 1
            return dict(position=position, token=int(decoded[0, 0]))

        # End even the diagnostic mode with replay: a final eager prefill alone
        # would not trigger the allocation tracker's pre-replay validation.
        final_replay = replay_retained_request()
        tile_continuations = []
        if args.full_prepare:
            retained_traces = dict(gen.traces)
            retained_captures = gen.counters["captures"]
            retained_retirements = gen.counters["retirements"]
            retained_programs = int(mesh.num_program_cache_entries())
            tile_tokens = torch.arange(71000, 71513, dtype=torch.int32)[None]
            for end in (65, 513):
                # Both public calls preserve their real boundaries. Start32 is
                # tile-aligned but not512-aligned; end513 also crosses512 with
                # a one-token tail. No per-case preparation is permitted.
                replays = [replay_retained_request()]
                kwargs_tile = dict(
                    page_table=gen.page_host[3:4].clone(),
                    kv_cache=cache,
                    empty_slots=[3],
                    sampling_params=prefix_greedy,
                )
                adapter.prefill_forward(
                    tile_tokens,
                    prompt_lens=[32],
                    start_pos=[0],
                    sampling_mask=[False],
                    **kwargs_tile,
                )
                replays.append(replay_retained_request())
                active_state = [
                    (tensor, first(tensor).reshape(4, -1)[0].clone())
                    for state in cache
                    if hasattr(state, "recurrent")
                    for tensor in (state.conv, state.recurrent)
                ]
                head_steps = gen.prefill_single.counters["unaligned_attention_steps"]
                samples_before = gen.counters["sample_replays"]
                sampled = adapter.prefill_forward(
                    tile_tokens,
                    prompt_lens=[end],
                    start_pos=[32],
                    sampling_mask=[True],
                    **kwargs_tile,
                )
                if adapter.request_specific_rope:
                    sampled = sampled[0]
                assert sampled.shape == (1, 1)
                assert (
                    gen.prefill_single.counters["unaligned_attention_steps"] == head_steps
                ), "Tile-aligned continuation unexpectedly used the serial attention head"
                assert gen.counters["sample_replays"] == samples_before + 1
                for tensor, previous in active_state:
                    assert torch.equal(
                        previous, first(tensor).reshape(4, -1)[0]
                    ), "Tile continuation changed the retained request's state"
                replays.append(replay_retained_request())
                assert dict(gen.traces) == retained_traces, "Tile continuation replaced or added a trace"
                assert gen.counters["captures"] == retained_captures
                assert gen.counters["retirements"] == retained_retirements
                assert (
                    int(mesh.num_program_cache_entries()) == retained_programs
                ), "Tile continuation compiled a program after generator-wide preparation"
                assert gen.counters["full_logits_readbacks"] == 0
                tile_continuations.append(
                    dict(
                        boundaries=[0, 32, end],
                        slot=3,
                        retained_decode_slot=0,
                        retained_decode_replays=replays,
                        sampled_token=int(sampled[0, 0]),
                        trace_ids={key: int(value) for key, value in gen.traces.items()},
                        captures=gen.counters["captures"],
                        retirements=gen.counters["retirements"],
                        program_cache_entries=int(mesh.num_program_cache_entries()),
                        unaligned_attention_step_delta=gen.prefill_single.counters["unaligned_attention_steps"]
                        - head_steps,
                        active_request_state_preserved=True,
                    )
                )
                final_replay = replays[-1]
        result = dict(
            passed=True,
            layers=adapter.model.layer_indices,
            context=adapter.max_seq_len,
            cache_shape=[8196, 1, 32, 256],
            preparation="full" if args.full_prepare else "diagnostic",
            all_intermediate_no_sample_or_read=True,
            mixed_preserves_masked_state=True,
            mixed_token=mixed.tolist(),
            solo_token=solo.tolist(),
            async_tokens=[result1.tolist(), result2.tolist()],
            deferred_read_known_values=[123, 456],
            tracking_positive_control=tracking_control,
            stale_inputs_ignored=True,
            page_only_refresh=True,
            prefix_matches_capacity=fixed_tokens == prefix_tokens,
            prefix_tokens=prefix_tokens,
            inactive_prefix_state_preserved=True,
            active_prefix_state_exact=True,
            deferred_full_page_tables=True,
            unchanged_host_map_prefill_flushes_dirty_tables=True,
            bucket_transitions=[4, 1, 4, 1],
            tile_aligned_continuations=tile_continuations,
            tile_continuation_trace_baseline=(
                dict(
                    trace_ids={key: int(value) for key, value in retained_traces.items()},
                    captures=retained_captures,
                    retirements=retained_retirements,
                    program_cache_entries=retained_programs,
                )
                if args.full_prepare
                else None
            ),
            near_context_end=(
                "Not run here: actual context is262144; this short retained-trace check uses "
                "valid starts32 and a512-boundary tail. The reduced continuation probe covers "
                "the shortened final fill-table branch at its explicit2048 allocation."
            ),
            final_decode_replay=final_replay,
            counters=dict(gen.counters),
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        if adapter is not None:
            adapter.release_persistent_capture()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
