"""Device contract checks through the serving adapter; no server/profiler."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

import ttnn

from ..tt.generator_vllm import K2HorizonForCausalLM


def read(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).clone()


def params(batch):
    return SimpleNamespace(top_k=[1] * batch, top_p=[0.0] * batch, temperature=[0.0] * batch, seed=[17] * batch)


def lifecycle(adapter):
    """Repeat a known prefill after decode retires and recreates batch buckets."""
    gen = adapter.generator
    batch = adapter.max_batch_size
    assert batch >= 4
    cache = adapter.allocate_kv_cache((batch * 8, 2, 32, 128), torch.bfloat16, gen.model.num_layers)
    table = torch.arange(batch * 8, dtype=torch.int32).reshape(batch, 8)
    prompt = torch.arange(127).reshape(1, -1) + 500
    records = []
    for cycle in range(3):
        first = adapter.prefill_forward(prompt, table[:1], cache, [127], sampling_params=params(1))
        for active in (1, 4, 2, 1):
            tokens = torch.full((batch, 1), int(first[0, 0]))
            positions = torch.full((batch,), -1, dtype=torch.int32)
            positions[:active] = 127
            adapter.decode_forward(tokens, positions, table, cache, reset_batch=True, sampling_params=params(batch))
            records.append(dict(cycle=cycle, active=active, pool=gen.model.pool.inventory()))
        print("LIFECYCLE_CYCLE", cycle, flush=True)
    # Once the B1 prefill scratch is restored, repeats retain the decode trace.
    for repeat in range(2):
        before = gen.counters.copy()
        first = adapter.prefill_forward(prompt, table[:1], cache, [127], sampling_params=params(1))
        tokens = torch.full((batch, 1), int(first[0, 0]))
        positions = torch.full((batch,), -1, dtype=torch.int32)
        positions[0] = 127
        adapter.decode_forward(tokens, positions, table, cache, reset_batch=True, sampling_params=params(batch))
        if repeat:
            assert gen.counters["release_synchronizations"] == before["release_synchronizations"]
    return dict(
        lifecycle=True,
        layers=gen.model.num_layers,
        records=records,
        warmed_batch1_trace_retained=True,
        counters=dict(gen.counters),
    )


def run(adapter):
    gen = adapter.generator
    batch = adapter.max_batch_size
    cache = adapter.allocate_kv_cache((batch * 8, 2, 32, 128), torch.bfloat16, gen.model.num_layers)
    table = torch.arange(batch * 8, dtype=torch.int32).reshape(batch, 8)
    lengths = [31 + 2 * (i % 2) for i in range(batch)]
    prompts = (torch.arange(batch * 33).reshape(batch, 33) % 1000 + 500).long()
    first = adapter.prefill_forward(prompts, table, cache, lengths, sampling_params=params(batch))
    assert first.shape == (batch, 1)
    pos = torch.tensor(lengths)
    adapter.decode_forward(
        first, pos, table, cache, reset_batch=True, sampling_params=params(batch), read_from_device=False
    )
    # Minimal deferred read, separate event wait and host formatting.
    host, events = adapter.read_decode_output(gen.state["tokens"], async_read=True)
    for event in events:
        ttnn.event_synchronize(event)
    second = adapter.process_decode_output_host(host, is_tokens=True)
    assert torch.equal(second.reshape(-1), read(gen.state["tokens"]).reshape(-1)[:batch].long())
    counters = gen.counters.copy()
    # Deliberately stale scheduler inputs must not overwrite device feedback.
    adapter.decode_forward(
        torch.ones_like(first),
        torch.zeros_like(pos),
        table,
        cache,
        sampling_params=params(batch),
        read_from_device=False,
    )
    third = adapter.process_decode_output_host(adapter.read_decode_output(gen.state["tokens"]), is_tokens=True)
    assert torch.equal(read(gen.state["positions"])[:batch], pos + 2)
    for key in ("token_refreshes", "position_refreshes", "rope_refreshes", "page_table_refreshes"):
        assert gen.counters[key] == counters[key], key
    steady_counters = dict(gen.counters - counters)

    # Refill identical inputs and force authoritative host state for the control.
    first_control = adapter.prefill_forward(prompts, table, cache, lengths, sampling_params=params(batch))
    control2 = adapter.decode_forward(first_control, pos, table, cache, reset_batch=True, sampling_params=params(batch))
    control3 = adapter.decode_forward(control2, pos + 1, table, cache, reset_batch=True, sampling_params=params(batch))
    assert torch.equal(second, control2) and torch.equal(third, control3)

    # At position 32 a new logical page becomes active. Give its physical page
    # a different mapping and verify copy-only refresh preserves token/position.
    one = torch.tensor([[500 + i for i in range(31)]] * batch)
    begin = adapter.prefill_forward(one, table, cache, [31] * batch, sampling_params=params(batch))
    adapter.decode_forward(
        begin, torch.full((batch,), 31), table, cache, reset_batch=True, sampling_params=params(batch)
    )
    changed = table.clone()
    changed[:, 1] = table[:, 7]
    previous = read(gen.state["tokens"]).reshape(-1)[:batch].long()
    counters = gen.counters.copy()
    output = adapter.decode_forward(
        torch.ones_like(begin), torch.zeros(batch, dtype=torch.int32), changed, cache, sampling_params=params(batch)
    )
    assert torch.equal(read(gen.state["positions"])[:batch], torch.full((batch,), 33))
    assert gen.counters["page_table_refreshes"] == counters["page_table_refreshes"] + 1
    for key in ("token_refreshes", "position_refreshes", "rope_refreshes"):
        assert gen.counters[key] == counters[key]
    assert torch.equal(read(gen.state["table"]), changed)
    # Same exact prior token/cache state with a full refresh is the control.
    expected = adapter.decode_forward(
        previous.reshape(batch, 1),
        torch.full((batch,), 32),
        changed,
        cache,
        sampling_params=params(batch),
        reset_batch=True,
    )
    assert torch.equal(output, expected)

    # Diagnostic host mode tests logits, never the performance sampling route.
    os.environ["K2_VLLM_ALLOW_HOST_SAMPLING"] = "1"
    adapter.allow_host_sampling = True
    logits1 = adapter.prefill_forward(prompts, table, cache, lengths)
    logits2 = adapter.prefill_forward(prompts, table, cache, lengths)
    assert torch.equal(logits1, logits2)
    perm = torch.arange(batch - 1, -1, -1)
    logits3 = adapter.prefill_forward(prompts[perm], table, cache, [lengths[i] for i in perm])
    assert torch.equal(logits1[perm], logits3)
    baseline = gen.prefill_forward(prompts, page_table=table, kv_cache=cache, prompt_lens=lengths, sampling_mode="host")
    assert torch.equal(logits1, baseline)
    reference_logprobs = torch.log_softmax(logits1.float(), dim=-1)
    values, indices = reference_logprobs[:, 0].topk(20)
    serving_logprob_controls = [
        dict(
            prompt=prompts[i, : lengths[i]].tolist(),
            top_logprobs={str(int(k)): float(v) for k, v in zip(indices[i], values[i])},
        )
        for i in range(batch)
    ]
    before = read(cache[0][0])
    gen.reset()
    assert gen.kv_cache is None and torch.equal(before, read(cache[0][0]))
    return dict(
        layers=gen.model.num_layers,
        batch=batch,
        prompt_lengths=lengths,
        async_split=True,
        device_feedback_matches_forced_control=True,
        stale_inputs_ignored=True,
        steady_counters=steady_counters,
        used_page_refresh_preserves_device_state=True,
        logits_repeat_and_batch_permutation_exact=True,
        standalone_logits_exact=True,
        external_cache_survives_reset=True,
        serving_logprob_controls=serving_logprob_controls,
        counters=dict(gen.counters),
        precision=gen.model.precision_config,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--output", required=True)
    p.add_argument("--lifecycle", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(16)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
    adapter = None
    try:
        adapter = K2HorizonForCausalLM(mesh, max_batch_size=a.batch, num_layers=a.layers)
        result = lifecycle(adapter) if a.lifecycle else run(adapter)
        Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
        print("ADAPTER_CONTRACT_PASS", a.output, flush=True)
    finally:
        if adapter is not None:
            adapter.generator.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
