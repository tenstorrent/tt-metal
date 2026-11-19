# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reduced TP4 adapter regression for the canonical wide sampling traces.

Run in a fresh, idle hardware process from the tt-metal root:
  TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
    QWEN_VLLM_DIAGNOSTIC_LAYERS=0,3 python -m \
    models.autoports.qwen_qwen3_6_27b.tests.probe_vllm_wide_sampling \
    --output /tmp/vllm-wide-sampling.json --draws 128

Synthetic full-vocabulary logits isolate sampling from model accuracy. Model
layers 0/3 and the scheduler-owned 8196-page cache still use the real adapter
preparation, persistent token output, sampling graph and penalty sibling graph.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
os.environ.setdefault("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")
os.environ.setdefault("QWEN_VLLM_DIAGNOSTIC_LAYERS", "0,3")

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.generator_vllm import QwenVLLM

TEMPERATURES = [0.5, 1.0, 2.0, 0.7]
TOP_K = [33, 40, 63, 64]
SEEDS = [17, 101, 2027, 31337]


def first(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).clone()


def addresses(tensor):
    return [int(part.buffer_address()) for part in ttnn.get_device_tensors(tensor)]


def synthetic_logits(vocab):
    assert vocab % 4 == 0
    shard_width = vocab // 4
    # Each row has sixteen distinct IDs on each shard. Token IDs are unrelated
    # to rank, including the ranks crossing the old 32-candidate boundary.
    rank = torch.arange(64)
    candidate_ids = torch.stack(
        [((rank + row) % 4) * shard_width + 1000 + row * 193 + (rank // 4) * 17 for row in range(4)]
    ).long()
    assert (candidate_ids < vocab).all()
    values = torch.full((1, 1, 32, vocab), -100.0, dtype=torch.bfloat16)
    ranked_logits = (-rank.float() / 128).to(torch.bfloat16)
    for row in range(4):
        values[0, 0, row, candidate_ids[row]] = ranked_logits
    return values, candidate_ids


def inspect_samples(samples, candidate_ids):
    matches = samples[:, :, None] == candidate_ids[None, :, :]
    assert matches.any(-1).all(), "Sample trace emitted an ID outside the 64 supplied candidates"
    ranks = matches.int().argmax(-1)
    assert (ranks < torch.tensor(TOP_K)[None, :]).all(), "Sample trace exceeded a row's top-k"
    counts = torch.stack([torch.bincount(ranks[:, row], minlength=64) for row in range(4)])
    # Pool eligible independent rows for boundary occupancy: with only 128 draws
    # the individual k33 row can legitimately miss rank33. Individual rank64 is
    # also low power here; the focused op probe checks it with more draws/users.
    assert (counts[:, 32:40].sum(0) > 0).all(), "Ranks33..40 were not all reached across eligible rows"
    assert (counts[1:, 32:].sum(1) > 0).all(), "A wide row never sampled beyond rank32"
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=128)
    args = parser.parse_args()
    if args.draws < 128:
        parser.error("--draws must be at least 128 for boundary occupancy checks")
    assert os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
    assert os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"
    assert os.environ["QWEN_VLLM_DIAGNOSTIC_LAYERS"] == "0,3"
    torch.set_num_threads(8)
    result = dict(
        passed=False,
        layers=[0, 3],
        batch=4,
        external_cache_pages=8196,
        draws=args.draws,
        temperature=TEMPERATURES,
        top_k=TOP_K,
        top_p=1.0,
        seeds=SEEDS,
        trace_allocation_tracking=True,
        program_cache_tracking=True,
        scope="Synthetic logits through prepared QwenVLLM generator sampling traces",
    )
    mesh = None
    adapter = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
        adapter = QwenVLLM.initialize_vllm_model(None, mesh, 4, 262144)
        cache = adapter.allocate_kv_cache((8196, 1, 32, 256), torch.bfloat16, 16)
        gen = adapter.generator
        for engine in {gen.prefill_single, gen.prefill_uniform}:
            engine.prepare(_diagnostic_lengths=[1, 2, 3, 32, 33, 65])
        adapter.warmup_model_prefill(kv_cache=cache)
        assert gen.prepared and gen.cache is cache
        baseline_traces = dict(gen.traces)
        captures = gen.counters["captures"]
        retirements = gen.counters["retirements"]
        logits_reads = gen.counters["full_logits_readbacks"]
        token_addresses = addresses(gen.tokens)
        sampler = gen.sampling.tt_sampling
        params = dict(temperature=TEMPERATURES, top_k=TOP_K, top_p=[1.0] * 4, seed=SEEDS)

        # Own the full host tensor and its TT wrapper through every queued copy.
        # All subsequent draws consume the persistent gen.logits allocation.
        logits, candidate_ids = synthetic_logits(gen.model.config.vocab_size)
        mapper = ttnn.ShardTensor2dMesh(mesh, dims=(None, 3), mesh_shape=mesh.shape)
        logits_host = ttnn.from_torch(logits, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
        ttnn.copy_host_to_device_tensor(logits_host, gen.logits)
        gen.configure_sampling(**params, reset_rng=True)
        mask = first(gen.sample_active_rows).flatten()
        assert torch.equal(mask[:4], torch.ones(4, dtype=mask.dtype)) and not mask[4:].any()
        observed_temps = first(sampler.temp_tensor).flatten()[:4].float()
        expected_temps = (1 / torch.tensor(TEMPERATURES)).to(torch.bfloat16).float()
        assert torch.equal(observed_temps, expected_temps), "Public temperatures were not converted exactly once to 1/T"
        assert first(sampler.k_tensor).flatten()[:4].long().tolist() == TOP_K
        assert torch.equal(first(sampler.p_tensor).flatten()[:4].float(), torch.ones(4))
        assert gen.sampling_values["temperature"] == TEMPERATURES

        def draw(n):
            tokens = []
            for index in range(n):
                gen._execute("sample")
                tokens.append(gen._read_tokens())
                assert addresses(gen.tokens) == token_addresses, "Recurrent token-out allocation changed"
                if (index + 1) % 64 == 0:
                    print(json.dumps(dict(progress="sample_sequence", draws=index + 1)), flush=True)
            return torch.tensor(tokens, dtype=torch.int64)

        before_seeds = first(sampler.seeds_tt_tensor).flatten().long()
        sampled = draw(args.draws)
        after_seeds = first(sampler.seeds_tt_tensor).flatten().long()
        assert torch.equal(
            after_seeds[:4], before_seeds[:4] + args.draws
        ), "Trace did not advance active RNG seeds once per draw"
        assert torch.equal(after_seeds[4:], before_seeds[4:]), "Trace advanced inactive RNG slots"
        assert torch.equal(first(gen.sample_active_rows).flatten(), mask), "Sample trace changed the active mask"
        counts = inspect_samples(sampled, candidate_ids)
        result.update(
            candidate_ids_by_rank=candidate_ids.tolist(),
            rank_counts=counts.tolist(),
            boundary_rank33_through40_counts=counts[:, 32:40].sum(0).tolist(),
            observed_reciprocal_temperatures=observed_temps.tolist(),
            first_tokens=sampled[:8].tolist(),
            active_seed_advance=args.draws,
            inactive_seeds_preserved=True,
        )

        # Reconfigure the same public parameters: cache reuse must still honor
        # reset_rng=True and reproduce the complete request-owned sequence.
        gen.configure_sampling(**params, reset_rng=True)
        assert torch.equal(first(sampler.seeds_tt_tensor).flatten().long()[:4], before_seeds[:4])
        repeated = draw(args.draws)
        assert torch.equal(sampled, repeated), "Same public seeds did not reproduce the complete sampled sequence"
        result["same_seed_sequence_equal"] = True

        gen.configure_sampling(temperature=0.0, top_k=1, top_p=0.0, seed=SEEDS)
        greedy = draw(1)[0]
        assert torch.equal(greedy, candidate_ids[:, 0]), "Greedy k1 changed under the wide sampler"
        result["greedy_tokens"] = greedy.tolist()

        # Exercise the already-prepared counted sibling once. A presence penalty
        # removes the original maximum in row0; the other rows are inactive.
        gen.configure_sampling(temperature=0.0, top_k=1, top_p=0.0, seed=SEEDS, presence_penalty=[2.0, 0.0, 0.0, 0.0])
        history = torch.full((4, 1), -1, dtype=torch.int64)
        history[0, 0] = candidate_ids[0, 0]
        gen.penalty_history.reset(slots=[0, 1, 2, 3], output_tokens=history)
        gen.penalty_history_slots.update([0, 1, 2, 3])
        gen._write(torch.nn.functional.pad(torch.tensor([[1.0, 0.0, 0.0, 0.0]]), (0, 28)), gen.sample_active_rows)
        sibling = gen._sample_trace_key([0])
        assert sibling == "sample_penalties"
        gen._execute(sibling)
        penalized = torch.tensor(gen._read_tokens())
        assert penalized[0] == candidate_ids[0, 1]
        assert torch.equal(penalized[1:], greedy[1:]), "Penalty trace overwrote inactive token slots"
        history_counts = first(gen.sampling.tt_penalties.output_counts_gathered)
        assert int(history_counts[0, candidate_ids[0, 0]]) == 1
        assert int(history_counts[0, candidate_ids[0, 1]]) == 1
        assert not history_counts[1:4].any(), "Penalty sibling counted inactive rows"
        result["penalty_sibling"] = dict(trace=sibling, token=int(penalized[0]), expected=int(candidate_ids[0, 1]))

        # End with the original neutral sample trace after history/mask changes,
        # so allocation tracking observes every operation performed above.
        gen.penalty_history.reset()
        gen.penalty_history_slots.clear()
        gen._write(mask.reshape(1, 32), gen.sample_active_rows)
        gen.configure_sampling(**params, reset_rng=True)
        final = draw(1)[0]
        assert torch.equal(final, sampled[0]), "Neutral replay changed after the penalty sibling"
        assert addresses(gen.tokens) == token_addresses
        assert gen.traces == baseline_traces
        assert gen.counters["captures"] == captures and gen.counters["retirements"] == retirements
        assert gen.counters["full_logits_readbacks"] == logits_reads
        result.update(
            passed=True,
            persistent_token_addresses=token_addresses,
            no_full_logits_readbacks=True,
            traces_stable=True,
            traces={key: int(value) for key, value in gen.traces.items()},
            counters=dict(gen.counters),
        )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {
                    key: value
                    for key, value in result.items()
                    if key not in ("candidate_ids_by_rank", "rank_counts", "counters")
                }
            ),
            flush=True,
        )
        if adapter is not None and adapter.generator is not None:
            adapter.generator.close()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
