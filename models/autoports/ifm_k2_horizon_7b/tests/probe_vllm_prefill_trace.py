"""External-cache prefill trace equivalence and bounded request lifecycle."""

import argparse
import json
from pathlib import Path

import torch

import ttnn

from ..tt.generator_vllm import K2HorizonForCausalLM
from .probe_vllm_adapter import params, read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(16)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
    adapter = None
    result = dict(layers=args.layers, cases=[])
    try:
        adapter = K2HorizonForCausalLM(mesh, max_batch_size=4, num_layers=args.layers)
        gen = adapter.generator
        cache = adapter.allocate_kv_cache((4 * 256, 2, 32, 128), torch.bfloat16, args.layers)
        table = torch.arange(4 * 256, dtype=torch.int32).reshape(4, 256)

        def generate(prompt, mapping, traced):
            first = adapter.prefill_forward(
                prompt,
                mapping[:1],
                cache,
                [prompt.shape[1]],
                sampling_params=params(1),
                enable_trace=traced,
            )
            output = [int(first[0, 0])]
            positions = torch.tensor([prompt.shape[1], -1, -1, -1])
            tokens = torch.zeros((4, 1), dtype=torch.long)
            tokens[0, 0] = first[0, 0]
            for step in range(3):
                device = adapter.decode_forward(
                    tokens,
                    positions,
                    mapping,
                    cache,
                    sampling_params=params(4),
                    # The live plugin drains and supplies authoritative state
                    # on allocator page growth, including table bucket changes.
                    reset_batch=step == 0 or int(positions[0]) % 32 == 0,
                    read_from_device=False,
                )
                host, events = adapter.read_decode_output(device, async_read=True)
                for event in events:
                    ttnn.event_synchronize(event)
                emitted = adapter.process_decode_output_host(host, is_tokens=True)
                output.append(int(emitted[0, 0]))
                tokens = emitted
                positions[0] += 1
            return output

        for length in (1, 31, 32, 33, 127, 128, 129, 224, 225, 255, 256, 257, 4095, 4096, 4097):
            prompt = (torch.arange(length) % 1000 + 500).reshape(1, -1)
            expected = generate(prompt, table, False)
            actual = generate(prompt, table, True)
            assert actual == expected, (length, actual, expected)
            before = gen.counters.copy()
            repeat = generate(prompt, table, True)
            assert repeat == expected
            delta = dict(gen.counters - before)
            if length <= 4096:
                assert delta.get("prefill_replays", 0) == 1, delta
                if length != 4095:
                    assert delta.get("release_synchronizations", 0) == 0, delta
            assert gen.kv_cache is None
            result["cases"].append(dict(length=length, tokens=actual, repeat_counters=delta))
            print("PREFILL_TRACE_CASE", length, flush=True)

        # Same compiled length, different token data and active physical pages.
        prompt = (torch.arange(127) + 900).reshape(1, -1)
        changed = table.flip(1).contiguous()
        expected = generate(prompt, changed, False)
        generate(prompt - 300, table, True)
        before = gen.counters.copy()
        actual = generate(prompt, changed, True)
        assert expected == actual
        assert gen.counters["prefill_page_table_refreshes"] == before["prefill_page_table_refreshes"] + 1
        assert torch.equal(read(gen.prefill_state["tokens"]).long(), prompt)
        assert torch.equal(read(gen.prefill_state["table"]), adapter._table(changed[:1], 127))
        result["changed_prompt_and_active_page_map"] = dict(tokens=actual, counters=dict(gen.counters - before))
        # Reduced models can share an argmax despite differing logits. Compare
        # the complete terminal tensor diagnostically, outside any timed path.
        traced_logits = gen._read_logits(gen.prefill_state["logits"]).clone()
        eager_logits = gen.prefill_forward(
            prompt,
            page_table=adapter._table(changed[:1], 127),
            kv_cache=cache,
            prompt_lens=[127],
            sampling_mode="host",
        )
        assert torch.equal(traced_logits.reshape_as(eager_logits), eager_logits)
        generate(prompt - 300, table, True)
        different_logits = gen._read_logits(gen.prefill_state["logits"]).clone()
        assert not torch.equal(different_logits, traced_logits)
        result["prefill_logits_exact_and_changed_tokens_observed"] = True
        result["standalone_cache_unset"] = gen.kv_cache is None
        external_before = read(cache[0][0])
        standalone = gen.generate(prompt[0].tolist(), 4)
        assert standalone == expected, (standalone, expected)
        assert torch.equal(external_before, read(cache[0][0]))
        result["standalone_generation_and_cache_isolation"] = True
        result["passed"] = True
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    finally:
        if adapter is not None:
            adapter.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
