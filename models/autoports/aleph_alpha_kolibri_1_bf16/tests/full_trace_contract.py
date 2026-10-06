# SPDX-License-Identifier: Apache-2.0
"""Exact persistent-state and lifetime probe, one real layer of each kind."""

import argparse
import json
import os
import time
from pathlib import Path

import torch

import ttnn

from ..tt.generator import build_generator
from .full_provenance import provenance
from .optimized_coverage import runtime_audit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--full", action="store_true")
    p.add_argument("--capacity", type=int, default=1024)
    args = p.parse_args()
    assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "0"
    torch.set_num_threads(8)
    started = time.monotonic()
    source = provenance()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    gen = None

    def read(t):
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).flatten().tolist()

    try:
        gen = build_generator(
            None, mesh, layer_indices=None if args.full else [0, 4], capacity=args.capacity, batch_size=args.batch
        )
        with runtime_audit():
            gen._decode()
            gen._sample()
            hidden = gen._prefill_bucket(32, args.capacity)
            del hidden
        # Direct model-prefill's default logits output is a sibling consumer
        # of the terminal geometry. Compare its M128 projection with the
        # generator's bounded-tile contract before any live captures exist.
        hidden = gen._prefill_bucket(128, args.capacity)
        whole = gen.read_logits(gen.model.terminal(hidden))
        tiled = torch.cat(
            [
                gen.read_logits(gen.model.terminal(ttnn.slice(hidden, (0, 0, start, 0), (1, 1, start + 32, 2560))))
                for start in range(0, 128, 32)
            ],
            dim=2,
        )
        assert tuple(whole.shape) == (1, 1, 128, 128000)
        assert torch.equal(whole, tiled)
        del hidden, whole, tiled
        gen.prepare()
        trace_ids = dict(gen.traces)
        gen.reset()
        b = args.batch
        positions = [31 + i for i in range(b)]
        if b > 1:
            positions[-1] = -1
        gen.bind([42 + i for i in range(b)], positions)
        snapshots = []
        counts = gen.counters.copy()
        for step in range(3):
            before = dict(tokens=read(gen.tokens), positions=read(gen.positions), rope=read(gen.rope_positions))
            gen.replay()
            after = dict(tokens=read(gen.tokens), positions=read(gen.positions), rope=read(gen.rope_positions))
            if snapshots:
                assert before["tokens"] == snapshots[-1]["after"]["tokens"]
            assert after["positions"] == [v + 1 if v >= 0 else -1 for v in before["positions"]]
            assert after["rope"] == [v + 1 if v >= 0 else -1 for v in before["rope"]]
            snapshots.append(dict(before=before, after=after))
        for name in ("token_refreshes", "position_refreshes", "rope_refreshes", "page_table_refreshes"):
            assert gen.counters[name] == counts[name], name
        if b > 1:
            for pair in gen.state.layers:
                for cache in pair:
                    host = ttnn.to_torch(ttnn.get_device_tensors(cache)[0])
                    pages_per_slot = host.shape[0] // b
                    assert torch.count_nonzero(host[-pages_per_slot:]).item() == 0, "Inactive slot KV was modified"
                    del host
        pages = {key: value.clone() for key, value in gen.state.host_page_tables.items()}
        count = gen.counters["page_table_refreshes"]
        gen.refresh_page_tables(pages)
        assert gen.counters["page_table_refreshes"] == count
        pages["full"][:, [0, 1]] = pages["full"][:, [1, 0]]
        gen.refresh_page_tables(pages)
        assert gen.counters["page_table_refreshes"] == count + 1
        assert torch.equal(ttnn.to_torch(ttnn.get_device_tensors(gen.state.page_tables["full"])[0]), pages["full"])
        gen.replay()
        gen.read_tokens()
        # A caller may mutate an exposed host map in place. The refresh cache
        # must compare against an independent snapshot of device contents.
        in_place = gen.state.host_page_tables["full"]
        in_place[:, [0, 1]] = in_place[:, [1, 0]]
        before_copy = gen.counters["page_table_refreshes"]
        gen.refresh_page_tables({"full": in_place})
        assert gen.counters["page_table_refreshes"] == before_copy + 1
        assert torch.equal(ttnn.to_torch(ttnn.get_device_tensors(gen.state.page_tables["full"])[0]), in_place)
        # Sampling parameters are persistent tensor values, not execution keys.
        gen.copy(torch.full((b,), 8, dtype=torch.int32), gen.k)
        gen.copy(torch.full((b,), 0.9), gen.p)
        gen.replay()
        gen.read_tokens()
        gen.copy(torch.ones(b, dtype=torch.int32), gen.k)
        gen.copy(torch.zeros(b), gen.p)
        gen.replay(mode="argmax")
        gen.read_tokens()
        gen.replay()
        gen.read_tokens()
        assert gen.traces == trace_ids
        # The common manual_seed UINT32_MAX sentinel must never be exposed by
        # a user seed. Repeating an edge seed after intervening RNG advancement
        # must reproduce the same sampled sequence from fixed real logits.
        seed_checks = []
        for edge in (4294967295, -1):
            sequences = []
            for seed in (edge, 12345, edge, edge % 1000000):
                gen.set_sampling(top_k=32, top_p=1.0, temperature=2.0, seed=seed)
                assert read(gen.sampler._seeds) == [seed % 1000000 + 1] * b
                values = []
                for _ in range(4):
                    ttnn.execute_trace(mesh, gen.traces["split"], cq_id=0, blocking=False)
                    values.append(gen.read_tokens().tolist())
                sequences.append(values)
            assert sequences[0] == sequences[2] == sequences[3]
            seed_checks.append(dict(seed=edge, normalized=edge % 1000000 + 1, sequences=sequences))
        gen.set_sampling()
        gen.reset()
        # New logical shape within prepared buckets, followed by existing traces.
        prompts = torch.full((b, 65), 42, dtype=torch.long)
        lens = [33 + (i % 3) * 16 for i in range(b)]
        sampled = gen.prefill_forward(
            prompts, page_table=gen.state.host_page_tables, kv_cache=gen.state, prompt_lens=lens
        )
        if b > 3:
            assert sampled[0] == sampled[3], "Equal prompts differ by fixed slot"
        # Explicit full-logit compatibility followed by token-out replay must
        # not introduce late program-cache allocations or invalidate traces.
        compat = gen.prefill_logits([42] * 65)
        assert tuple(compat.shape) == (1, 65, 128000)
        gen.bind([42] * b, [65] * b)
        gen.replay()
        gen.read_tokens()
        assert gen.traces == trace_ids
        report = dict(
            provenance=source,
            runtime_forward_audit=True,
            layer_count=len(gen.model.layers),
            capacity=args.capacity,
            batch=b,
            seconds=time.monotonic() - started,
            trace_ids={key: str(value) for key, value in trace_ids.items()},
            snapshots=snapshots,
            counters=dict(gen.counters),
            tracking=True,
            program_cache_tracking=True,
            exact_feedback=True,
            page_change=True,
            unchanged_page_skip=True,
            cross_request_reuse=True,
            seed_edge_checks=seed_checks,
            modes=["split_greedy", "split_sampled", "argmax", "split_greedy"],
        )
        target = (
            Path(os.environ.get("FULL_ARTIFACT_DIR", Path(__file__).resolve().parents[1] / "doc/full_model"))
            / f"trace_contract_{'full_' if args.full else ''}b{b}.json"
        )
        target.write_text(json.dumps(report, indent=2) + "\n")
        print("TRACE_CONTRACT_PASS", json.dumps(report), flush=True)
    finally:
        if gen:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
