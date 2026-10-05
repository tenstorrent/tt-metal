"""Focused persistent-input, feedback, page-table and sampling replay evidence."""

import argparse
import json
from pathlib import Path

import torch

import ttnn

from ..tt.generator import K2Generator


def read(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).clone()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--output", default="models/demos/k2_horizon_7b_qb2/doc/full_model/trace_contract.json")
    p.add_argument("--strategy", default="split")
    args = p.parse_args()
    torch.set_num_threads(16)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    gen = None
    result = {"layers": args.layers, "records": []}
    try:
        gen = K2Generator(mesh, override_num_layers=args.layers)
        prompt = gen.tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is 2 plus 3?"}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        print("TRACE_PROBE_INITIAL_GENERATE", len(prompt), flush=True)
        gen.generate(prompt, 1, strategy=args.strategy)
        print("TRACE_PROBE_INITIAL_GENERATE_COMPLETE", flush=True)
        state = gen.state
        buffer_ids = {name: state[name].buffer_address() for name in ["tokens", "positions", "rope", "table"]}
        before = gen.counters.copy()
        for step in range(3):
            print("TRACE_PROBE_FEEDBACK", step, flush=True)
            consumed = read(state["tokens"]).reshape(-1).tolist()
            pos_before = read(state["positions"]).reshape(-1).tolist()
            rope_before = read(state["rope"]).reshape(-1).tolist()
            if step:
                assert consumed == result["records"][-1]["sampled_tokens"]
            assert pos_before[0] == len(prompt) + step
            assert rope_before[0] == pos_before[0]
            gen.decode_forward(
                page_table=gen.page_table, kv_cache=gen.kv_cache, strategy=args.strategy, read_from_device=False
            )
            sampled = read(state["tokens"]).reshape(-1).tolist()
            pos_after = read(state["positions"]).reshape(-1).tolist()
            assert pos_after[0] == pos_before[0] + 1
            assert pos_after[1:] == [-1] * 31
            for name, addr in buffer_ids.items():
                assert state[name].buffer_address() == addr
            logits = gen._read_logits(state["logits"])[0, 0, 0]
            assert logits[int(sampled[0])] == logits.max(), "Split path must be semantically greedy"
            result["records"].append(
                {
                    "consumed_tokens": consumed,
                    "sampled_tokens": sampled,
                    "positions_before": pos_before,
                    "positions_after": pos_after,
                    "rope_before": rope_before,
                }
            )
        result["steady_state_counters"] = dict(gen.counters - before)
        for name in ["token_refreshes", "position_refreshes", "rope_refreshes", "page_table_refreshes"]:
            assert result["steady_state_counters"].get(name, 0) == 0
        before = gen.counters.copy()
        gen.refresh_page_table(gen.page_table)
        assert gen.counters["page_table_refreshes"] == before["page_table_refreshes"]
        # Recreate one fixed decode state and preserve exact rank-local KV.
        print("TRACE_PROBE_PAGE_REMAP", flush=True)
        gen.generate(prompt, 1, strategy=args.strategy)
        current_token = read(state["tokens"]).reshape(-1)[:1].int()
        cache_host = [
            tuple(ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1)).clone() for t in pair)
            for pair in gen.kv_cache
        ]
        position = torch.tensor([len(prompt)], dtype=torch.int32)
        gen.decode_forward(current_token, position, page_table=gen.page_table, kv_cache=gen.kv_cache)
        baseline = gen._read_logits(state["logits"]).clone()
        baseline_tokens = read(state["tokens"]).clone()
        permutation = torch.arange(gen.page_table.numel()).flip(0)
        remap = permutation[gen.page_table.long()].int()
        for pair, host_pair in zip(gen.kv_cache, cache_host):
            for tensor, host in zip(pair, host_pair):
                remapped = torch.empty_like(host)
                remapped[permutation] = host
                upload = ttnn.from_torch(
                    remapped, dtype=tensor.dtype, layout=tensor.layout, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1)
                )
                ttnn.copy_host_to_device_tensor(upload, tensor)
        before = gen.counters.copy()
        gen.decode_forward(current_token, position, page_table=remap, kv_cache=gen.kv_cache)
        assert torch.equal(read(state["table"]), remap)
        assert gen.counters["page_table_refreshes"] == before["page_table_refreshes"] + 1
        remapped_logits = gen._read_logits(state["logits"])
        assert torch.equal(baseline, remapped_logits), "Cache/page permutation must preserve logits"
        assert torch.equal(baseline_tokens, read(state["tokens"]))
        after_remap = read(state["tokens"]).reshape(-1).tolist()
        result["page_table"] = {
            "unchanged_copies": 0,
            "changed_copies": 1,
            "changed_host_table": remap.tolist(),
            "sampled_after_remap": after_remap,
            "remapped_cache_logits_exact": True,
        }
        # Distinct external scheduler inputs are consumed at the same addresses.
        gen.decode_forward(torch.tensor([42]), position + 1, page_table=remap, kv_cache=gen.kv_cache)
        different_logits = gen._read_logits(state["logits"])
        assert not torch.equal(baseline, different_logits)
        assert int(read(state["positions"])[0]) == len(prompt) + 2
        assert int(read(state["rope"]).reshape(-1)[0]) == len(prompt) + 2
        result["different_scheduler_inputs"] = {
            "tokens": [int(current_token[0]), 42],
            "positions": [len(prompt), len(prompt) + 1],
            "logits_changed": True,
        }
        for name, addr in buffer_ids.items():
            assert state[name].buffer_address() == addr

        # First/cached seeded requests, strategy switches, and nontrivial runtime
        # sampling contents exercise warmup state without a host seed loop.
        print("TRACE_PROBE_STOCHASTIC", flush=True)
        stochastic = []
        for index in range(2):
            print("TRACE_PROBE_STOCHASTIC_REQUEST", index, flush=True)
            stochastic.append(gen.generate(prompt, 32, top_k=32, top_p=1.0, temperature=20.0, seed=2**64 - 1))
        assert stochastic[0] == stochastic[1]
        assert len(set(stochastic[0])) > 1
        print("TRACE_PROBE_SPLIT", flush=True)
        split = gen.generate(prompt, 16, strategy="split")
        print("TRACE_PROBE_ARGMAX", flush=True)
        argmax = gen.generate(prompt, 16, strategy="argmax")
        assert split == argmax
        print("TRACE_PROBE_SPLIT_AFTER_ARGMAX", flush=True)
        repeated = gen.generate(prompt, 16, strategy="split")
        assert repeated == split
        result["sampling"] = {
            "seeded_repeated_tokens": stochastic,
            "greedy_split": split,
            "greedy_argmax": argmax,
            "split_after_switch": repeated,
        }

        # First fused-preparation boundary and an awkward tile/page tail.
        result["prepared_prefill_repeated"] = []
        for length in [224, 225, 257]:
            medium = (prompt * ((length + len(prompt) - 1) // len(prompt)))[:length]
            first = gen.generate(medium, 4)
            second = gen.generate(medium, 4)
            assert first == second
            result["prepared_prefill_repeated"].append({"length": length, "tokens": first, "repeat_exact": True})

        # A request whose eager AGMM scratch changes 4096 -> 256 must leave no
        # live allocation that invalidates decode traces on its second pass.
        long_prompt = (prompt * ((4352 + len(prompt) - 1) // len(prompt)))[:4352]
        long_outputs = [gen.generate(long_prompt, 4), gen.generate(long_prompt, 4)]
        assert long_outputs[0] == long_outputs[1]
        result["chunked_repeated"] = {"prompt_len": 4352, "tokens": long_outputs}
        result["buffer_addresses"] = buffer_ids
        result["pass"] = True
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
