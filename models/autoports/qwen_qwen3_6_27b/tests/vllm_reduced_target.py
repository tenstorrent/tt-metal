"""Four-layer hardware smoke for the production Qwen3.6 vLLM adapter."""

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.generator_vllm import DEFAULT_SNAPSHOT, Qwen36ForCausalLM
from models.common.sampling import SamplingParams


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    try:
        config = AutoConfig.from_pretrained(DEFAULT_SNAPSHOT, local_files_only=True)
        adapter = Qwen36ForCausalLM.initialize_vllm_model(config, mesh, max_batch_size=2, max_seq_len=256, n_layers=4)
        print("REDUCED_INIT_OK", flush=True)
        cache = adapter.allocate_kv_cache((8, 1, 64, 256), torch.bfloat16, 1)
        tokens = torch.randint(0, 1000, (2, 65), dtype=torch.long)
        page_table = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32)
        params = SamplingParams(temperature=[0.0, 0.0], top_k=[1, 1], top_p=[1.0, 1.0])
        output, rope_deltas = adapter.prefill_forward(
            tokens, page_table, cache, [65, 63], sampling_params=params, empty_slots=[0, 1]
        )
        assert rope_deltas.tolist() == [0, 0]
        print("REDUCED_PREFILL_OK", tuple(output[0].shape), flush=True)
        device_output = adapter.decode_forward(
            output[0].reshape(2, 1),
            torch.tensor([65, 63]),
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
            torch.full((2, 1), 999, dtype=torch.long),
            torch.tensor([65, 63]),
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
        swapped_device_output = adapter.decode_forward(
            stale_sampled.flip(0).reshape(2, 1),
            torch.tensor([64, 66]),
            page_table.flip(0),
            cache,
            sampling_params=params,
            reset_batch=True,
            slot_remap=torch.tensor([1, 0], dtype=torch.int32),
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
