"""Four-layer hardware smoke for the production Qwen3.6 vLLM adapter.

``--batch`` defaults to 2, which is what this harness was validated at, but the
served batch is 32 and the two are not interchangeable: the fused KDA conv decode
path needs ``kernel * batch`` tile aligned, so at batch 2 it never engages and the
linear state stays in the composite tiled ``[1, batch, channels, kernel]``
layout. ``prefill_forward(empty_slots=...)`` and ``slot_remap`` both reach into
that state, so at batch 2 they only ever exercise one of the two layouts the
model can hold. Running this at ``--batch 32`` is what catches the other.
"""

import argparse

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.generator_vllm import DEFAULT_SNAPSHOT, Qwen36ForCausalLM
from models.common.sampling import SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--n-layers", type=int, default=4)
    args = parser.parse_args()
    batch = args.batch
    blocks_per_row = 4

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    try:
        config = AutoConfig.from_pretrained(DEFAULT_SNAPSHOT, local_files_only=True)
        adapter = Qwen36ForCausalLM.initialize_vllm_model(
            config, mesh, max_batch_size=batch, max_seq_len=256, n_layers=args.n_layers
        )
        kda = [
            bool(getattr(layer, "linear_kda_decode_ready", False))
            for layer in adapter.generator.model.layers
            if layer.layer_kind == "linear_attention"
        ]
        print(f"REDUCED_INIT_OK batch={batch} fused_kda_decode={sum(kda)}/{len(kda)}", flush=True)
        cache = adapter.allocate_kv_cache((blocks_per_row * batch, 1, 64, 256), torch.bfloat16, 1)
        tokens = torch.randint(0, 1000, (batch, 65), dtype=torch.long)
        page_table = torch.tensor(
            [[blocks_per_row * row + i for i in range(blocks_per_row)] for row in range(batch)], dtype=torch.int32
        )
        prompt_lens = [65, 63] + [64] * (batch - 2) if batch >= 2 else [65]
        positions = torch.tensor(prompt_lens)
        params = SamplingParams(temperature=[0.0] * batch, top_k=[1] * batch, top_p=[1.0] * batch)
        output, rope_deltas = adapter.prefill_forward(
            tokens, page_table, cache, prompt_lens, sampling_params=params, empty_slots=list(range(batch))
        )
        assert rope_deltas.tolist() == [0] * batch
        print("REDUCED_PREFILL_OK", tuple(output[0].shape), flush=True)
        device_output = adapter.decode_forward(
            output[0].reshape(batch, 1),
            positions,
            page_table,
            cache,
            sampling_params=params,
            reset_batch=True,
            read_from_device=False,
        )
        host_output, events = adapter.read_decode_output(device_output, async_read=True)
        for event in events:
            ttnn.event_synchronize(event)
        sampled, _ = adapter.process_decode_output_host(host_output, is_tokens=True)
        print("REDUCED_DECODE_OK", sampled.tolist(), adapter.generator.trace_counters, flush=True)
        counters_before_stale = dict(adapter.generator.trace_counters)
        stale_device_output = adapter.decode_forward(
            torch.full((batch, 1), 999, dtype=torch.long),
            positions,
            page_table,
            cache,
            sampling_params=params,
            reset_batch=False,
            read_from_device=False,
        )
        stale_host, events = adapter.read_decode_output(stale_device_output, async_read=True)
        for event in events:
            ttnn.event_synchronize(event)
        stale_sampled, _ = adapter.process_decode_output_host(stale_host, is_tokens=True)
        counters_after_stale = dict(adapter.generator.trace_counters)
        assert stale_sampled.shape == sampled.shape
        assert stale_sampled.ne(999).all()
        assert counters_after_stale["replays"] == counters_before_stale["replays"] + 1
        for name in ("token_host_refreshes", "position_host_refreshes", "page_table_refreshes", "readbacks"):
            assert counters_after_stale[name] == counters_before_stale[name]
        print(
            "REDUCED_STALE_INPUT_OK",
            stale_sampled.tolist(),
            counters_before_stale,
            counters_after_stale,
            flush=True,
        )
        # remap[new] = old.  A rotation by one is the batch-2 flip generalized,
        # so the batch-2 behaviour this harness recorded is unchanged.
        remap = [(row + 1) % batch for row in range(batch)]
        swapped_device_output = adapter.decode_forward(
            torch.roll(stale_sampled, -1, dims=0).reshape(batch, 1),
            torch.roll(positions, -1, dims=0) + 1,
            page_table[remap],
            cache,
            sampling_params=params,
            reset_batch=True,
            slot_remap=torch.tensor(remap, dtype=torch.int32),
            read_from_device=False,
        )
        swapped_host, events = adapter.read_decode_output(swapped_device_output, async_read=True)
        for event in events:
            ttnn.event_synchronize(event)
        swapped, _ = adapter.process_decode_output_host(swapped_host, is_tokens=True)
        print("REDUCED_SLOT_REMAP_OK", swapped.tolist(), flush=True)
        adapter.teardown()
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
