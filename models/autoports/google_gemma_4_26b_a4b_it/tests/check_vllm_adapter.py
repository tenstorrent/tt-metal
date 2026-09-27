# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced adapter token feedback and nonuniform page-table regression."""

import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import Gemma4Generator
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm import AutoportGemma4ForCausalLM
from models.common.sampling.generator import SamplingParams


def read(value):
    return ttnn.to_torch(ttnn.get_device_tensors(value)[0]).clone()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    gen = None
    try:
        gen = Gemma4Generator(mesh, max_seq_len=256, layer_indices=(0, 5), trace_debug=True)
        prompt = [2] + [100] * 32
        expected = gen.generate(prompt, 9, stop_on_eos=False)
        gen._release_trace()
        adapter = AutoportGemma4ForCausalLM(gen, 2)
        specs = [((8, 2, 32, 256), torch.bfloat16, i) for i in range(30)]
        specs[5] = ((8, 1, 32, 512), torch.bfloat16, 0)
        cache = adapter.allocate_kv_cache_per_layer(specs)
        assert all(a.buffer_address() == b.buffer_address() for a, b in zip(cache[0], cache[1]))
        table = torch.arange(8, dtype=torch.int32).reshape(1, -1)
        tables = [table] * 30
        tables[5] = table.flip(1)
        params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
        first = adapter.prefill_forward(
            torch.tensor([prompt]),
            table,
            cache,
            [33],
            sampling_params=params,
            page_tables_per_layer=tables,
            start_pos=torch.tensor([0]),
        )
        observed = [int(first.item())]
        for step in range(3):
            prior = read(gen.tokens) if step else None
            prior_positions = read(gen.cache_positions) if step else None
            output = adapter.decode_forward(
                first if step == 0 else torch.tensor([[12345]]),
                torch.tensor([33]),
                table,
                cache,
                sampling_params=params,
                page_tables_per_layer=tables,
                reset_batch=step == 0,
                read_from_device=False,
            )
            host, events = adapter.read_decode_output(output, async_read=True)
            for event in events:
                ttnn.event_synchronize(event)
            observed.append(int(adapter.process_decode_output_host(host, is_tokens=True).item()))
            if step:
                assert torch.equal(read(gen.consumed_tokens), prior)
                assert torch.equal(read(gen.consumed_positions), prior_positions)
                assert torch.equal(read(gen.cache_positions), prior_positions + 1)
        assert observed == expected[:4], (observed, expected)
        assert gen.cache is cache
        assert gen.counters["page_table_refreshes"] == 0
        trace_id = gen.trace_id
        changed = tables[5].clone()
        # These logical pages are outside this short decode's read window.
        # This checks refresh and trace retention, not remapping live KV data.
        changed[:, [5, 6]] = changed[:, [6, 5]]
        tables[5] = changed
        for step in range(2):
            output = adapter.decode_forward(
                torch.tensor([[observed[-1]]]) if step == 0 else torch.tensor([[12345]]),
                torch.tensor([36]) if step == 0 else torch.tensor([33]),
                table,
                cache,
                sampling_params=params,
                page_tables_per_layer=tables,
                reset_batch=step == 0,
                read_from_device=False,
            )
            if step == 0:
                # The synchronous plugin path finalizes a raw distributed
                # device tensor without calling read_decode_output first.
                host_tokens = adapter.process_decode_output_host(output, is_tokens=True)
            else:
                host, events = adapter.read_decode_output(output, async_read=True)
                for event in events:
                    ttnn.event_synchronize(event)
                host_tokens = adapter.process_decode_output_host(host, is_tokens=True)
            observed.append(int(host_tokens.item()))
            assert gen.trace_id == trace_id
            assert gen.counters["page_table_refreshes"] == 1
            assert torch.equal(read(gen.table[1]), changed)
        assert observed == expected[:6], (observed, expected)

        bindings = (gen.tokens, gen.positions, gen.cache_positions, gen.public_tokens, gen.public_tokens_view)
        page_bindings = tuple(gen.table)
        refresh_counters = {
            name: gen.counters[name]
            for name in ("token_refreshes", "position_refreshes", "cache_position_refreshes", "page_table_refreshes")
        }
        queued = []
        for _ in range(2):
            output = adapter.decode_forward(
                torch.tensor([[12345]]),
                torch.tensor([33]),
                table,
                cache,
                sampling_params=params,
                page_tables_per_layer=tables,
                reset_batch=False,
                read_from_device=False,
            )
            assert output is bindings[-1]
            current = (gen.tokens, gen.positions, gen.cache_positions, gen.public_tokens, gen.public_tokens_view)
            assert all(value is prior for value, prior in zip(current, bindings))
            assert all(value is prior for value, prior in zip(gen.table, page_bindings))
            assert gen.trace_id == trace_id
            queued.append(adapter.read_decode_output(output, async_read=True))
        # Both decode/read pairs must be submitted before any event wait or
        # host conversion, as in the plugin's overlapped steady decode path.
        assert queued[0][0] is not queued[1][0]
        queued_tokens = []
        for host, events in queued:
            for event in events:
                ttnn.event_synchronize(event)
            tokens = adapter.process_decode_output_host(host, is_tokens=True)
            assert tokens.shape == (1, 1)
            queued_tokens.append(tokens)
            observed.append(int(tokens.item()))
        assert observed == expected[:8], (observed, expected)
        assert queued_tokens[0].data_ptr() != queued_tokens[1].data_ptr()
        assert all(gen.counters[name] == value for name, value in refresh_counters.items())
        snapshots = [tokens.clone() for tokens in queued_tokens]

        # Move the same request to slot 1, leaving slot 0 inactive. The sole
        # active row retains its original cache mapping, so no KV data moves.
        rebound_tables = [value.repeat(2, 1) for value in tables]
        rebound = adapter.decode_forward(
            torch.tensor([[0], [observed[-1]]]),
            torch.tensor([-1, len(prompt) + len(observed) - 1]),
            rebound_tables[0],
            cache,
            sampling_params=params,
            page_tables_per_layer=rebound_tables,
            reset_batch=True,
            read_from_device=False,
        )
        assert gen.batch == 2 and gen.active_slots == (1,)
        assert gen.cache is cache
        assert rebound is gen.public_tokens_view and rebound is not bindings[-1]
        host, events = adapter.read_decode_output(rebound, async_read=True)
        for event in events:
            ttnn.event_synchronize(event)
        rebound_tokens = adapter.process_decode_output_host(host, is_tokens=True)
        assert rebound_tokens.shape == (2, 1)
        observed.append(int(rebound_tokens[1].item()))
        assert observed == expected, (observed, expected)
        for (prior_host, _), prior_tokens, snapshot in zip(queued, queued_tokens, snapshots):
            torch.testing.assert_close(prior_tokens, snapshot)
            delayed = adapter.process_decode_output_host(prior_host, is_tokens=True)
            assert delayed.shape == (1, 1)
            torch.testing.assert_close(delayed, snapshot)
        report = {
            "scope": "reduced layers 0 and 5; direct adapter, not server performance",
            "nonaligned_prompt_tokens": 33,
            "external_cache_identity": True,
            "distinct_per_layer_page_tables": True,
            "hybrid_pool_aliases_with_distinct_geometry": True,
            "stale_token_position_feedback": True,
            "async_read": True,
            "queued_decode_reads_before_first_wait": 2,
            "queued_async_tokens": [int(tokens.item()) for tokens in queued_tokens],
            "persistent_decode_tensor_identity": True,
            "queued_reads_have_independent_host_storage": True,
            "host_outputs_stable_after_batch_rebind": True,
            "batch_rebind_rows": 2,
            "batch_rebind_active_slots": [1],
            "changed_page_table_single_refresh_without_recapture": True,
            "changed_page_table_scope": "unused logical columns 5 and 6; live-page growth is covered by serving",
            "matches_standalone_tokens": True,
            "tokens": observed,
            "runtime_precision": gen.model.precision_summary(),
            "counters": gen.counters,
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        if gen is not None:
            gen.teardown()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
