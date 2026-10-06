# SPDX-License-Identifier: Apache-2.0
"""All-layer request reuse, qualitative, capacity and primary 128/128 evidence."""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch

import ttnn

from ..tt.generator import KolibriGenerator, build_generator
from .full_memory import memory_views, persistent_tensors
from .full_provenance import provenance
from .full_qualitative import run_suite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--capacity", type=int, default=1048576)
    parser.add_argument("--tag", default="full_validation")
    parser.add_argument("--reduced", action="store_true")
    parser.add_argument("--qualitative", action="store_true")
    parser.add_argument("--batch-reconfigure", action="store_true")
    parser.add_argument("--long-prompt", type=int, default=8193)
    args = parser.parse_args()
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[1]
    out = Path(os.environ.get("FULL_ARTIFACT_DIR", root / "doc/full_model"))
    result = dict(provenance=provenance(), batch=args.batch, capacity=args.capacity)

    def save():
        (out / f"{args.tag}.json").write_text(json.dumps(result, indent=2) + "\n")

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
    gen = None
    try:
        tick = time.monotonic()
        gen = build_generator(
            root, mesh, capacity=args.capacity, batch_size=args.batch, layer_indices=[0, 4] if args.reduced else None
        )
        gen.prepare()
        result["setup_seconds"] = time.monotonic() - tick
        result["allocator"] = memory_views(mesh)
        result["persistent_tensors"] = persistent_tensors(gen)
        result["layer_count"] = len(gen.model.layers)
        saved_traces = dict(gen.traces)
        save()
        print("PREPARED", result["setup_seconds"], flush=True)
        if args.qualitative:
            result["qualitative"] = run_suite(gen, "tt")
            save()
        # Equal real prompt at different fixed slots, mixed lengths, inactive slot.
        prompts = torch.full((args.batch, 129), 42, dtype=torch.long)
        lens = [65 + i % 3 * 32 for i in range(args.batch)]
        if args.batch > 1:
            lens[-1] = 0
        gen.reset()
        first = gen.prefill_forward(
            prompts, page_table=gen.state.host_page_tables, kv_cache=gen.state, prompt_lens=lens
        )
        if args.batch > 3:
            assert first[0] == first[3]
        before = gen.counters.copy()
        state_before = {
            k: ttnn.to_torch(ttnn.get_device_tensors(v)[0]).flatten().tolist()
            for k, v in (("tokens", gen.tokens), ("positions", gen.positions), ("rope", gen.rope_positions))
        }
        gen.replay()
        state_after = {
            k: ttnn.to_torch(ttnn.get_device_tensors(v)[0]).flatten().tolist()
            for k, v in (("tokens", gen.tokens), ("positions", gen.positions), ("rope", gen.rope_positions))
        }
        assert state_after["positions"] == [p + 1 if p >= 0 else -1 for p in state_before["positions"]]
        assert state_after["rope"] == state_after["positions"]
        for key in ("token_refreshes", "position_refreshes", "rope_refreshes", "page_table_refreshes"):
            assert gen.counters[key] == before[key]
        gen.reset()
        second = gen.prefill_forward(
            prompts, page_table=gen.state.host_page_tables, kv_cache=gen.state, prompt_lens=lens
        )
        assert torch.equal(first, second), "Repeated prompt token outputs differ"
        result["mixed_batch"] = dict(
            lengths=lens,
            first=first.tolist(),
            repeat=second.tolist(),
            trace_before=state_before,
            trace_after=state_after,
        )
        # Full-logit compatibility is explicit. The same live traces survive it.
        logits1 = gen.prefill_logits([42] * 65)
        logits2 = gen.prefill_logits([42] * 65)
        assert torch.equal(logits1, logits2), (logits1 - logits2).abs().max()
        result["deterministic_logits"] = True
        del logits1, logits2
        pages = {k: v.clone() for k, v in gen.state.host_page_tables.items()}
        initial = gen.counters["page_table_refreshes"]
        gen.refresh_page_tables(pages)
        assert gen.counters["page_table_refreshes"] == initial
        pages["full"][:, [0, 1]] = pages["full"][:, [1, 0]]
        gen.refresh_page_tables(pages)
        assert gen.counters["page_table_refreshes"] == initial + 1
        gen.bind([42] * args.batch, [64] * args.batch)
        gen.replay()
        gen.read_tokens()
        gen.set_sampling(top_k=8, top_p=0.9, temperature=0.8, seed=17)
        gen.replay()
        gen.read_tokens()
        gen.set_sampling()
        gen.replay()
        gen.read_tokens()
        result["page_table_and_sampling_reuse"] = True
        save()
        if args.long_prompt:
            tick = time.monotonic()
            original_prefill = gen._prefill_bucket

            def progress_prefill(n, bound, alignment=32):
                value = original_prefill(n, bound, alignment)
                print("LONG_PREFILL_PHYSICAL_END", bound, flush=True)
                return value

            gen._prefill_bucket = progress_prefill
            try:
                long_gen_len = min(34, args.capacity - args.long_prompt + 1)
                ids = gen.generate([42] * args.long_prompt, long_gen_len, stop_on_eos=False)
            finally:
                del gen._prefill_bucket
                del original_prefill, progress_prefill
            result["long_prompt"] = dict(
                length=args.long_prompt,
                ids=ids,
                last_consumed_position=args.long_prompt + len(ids) - 2,
                seconds=time.monotonic() - tick,
            )
            save()
        # Exercise last valid position with allocated cache and nonaligned suffix.
        # This is an address/capacity probe, not a claim to have populated a 1M prefix.
        gen.reset()
        suffix = torch.full((1, 33), 42, dtype=torch.long)
        boundary = gen.prefill_forward(
            suffix,
            page_table=gen.state.host_page_tables,
            kv_cache=gen.state,
            prompt_lens=[33],
            start_pos=[args.capacity - 33],
            slots=[0],
        )
        result["last_position_probe"] = dict(
            start=args.capacity - 33, prompt_len=33, initialized_prefix=False, output=boundary.tolist()
        )
        save()
        # Matched primary batch-1 prompt128/generate128, caller token readback included.
        if args.batch == 1:
            prompt = json.loads((out / "reference_metadata.json").read_text())["prompt_token_ids"][:128]
            if len(prompt) != 128:
                raise ValueError("Primary benchmark needs exactly 128 prompt tokens")
            primary = []
            for teacher in (False, True):
                for repeat in range(3):
                    gen.generate(
                        prompt, 128, next_input=(lambda step, pred: 42) if teacher else None, stop_on_eos=False
                    )
                    primary.append(dict(repeat=repeat, **gen.last_generation_metrics))
            result["primary_128_128"] = primary
            # Paired warmed full-generator TTFT with the same final norm and
            # cache policy, including reset/input prep/terminal/first readback.
            comparisons = []
            for length in (31, 128, 8193):
                request = (prompt * ((length + len(prompt) - 1) // len(prompt)))[:length]
                for repeat in range(3):
                    ids = []
                    for traced in (False, True) if repeat % 2 == 0 else (True, False):
                        gen.trace_prefill = traced
                        tokens = gen.generate(request, 2, stop_on_eos=False)
                        ids.append(tokens)
                        comparisons.append(
                            dict(
                                length=length,
                                repeat=repeat,
                                traced_prefill=traced,
                                tokens=tokens,
                                **gen.last_generation_metrics,
                            )
                        )
                    assert ids[0] == ids[1]
            gen.trace_prefill = True
            result["prefill_comparison"] = comparisons
            result[
                "prefill_comparison_input"
            ] = "Primary chat-template token prefix, repeated to the requested length; performance workload, not a quality prompt."
            save()
            # Same real prefix and identical greedy semantics for split vs argmax.
            terminal = {}
            terminal_counters = {}
            for mode in ("split", "argmax", "logits"):
                gen.reset()
                gen._prefill(prompt[:-1])
                gen.bind([prompt[-1]], [127])
                gen.replay(sample=mode != "logits", mode="split" if mode == "logits" else mode)
                ttnn.synchronize_device(mesh)
                before_counts = gen.counters.copy()
                tick = time.monotonic()
                for _ in range(128):
                    if mode == "split":
                        gen.decode_forward(None, None, page_table=None, kv_cache=gen.state, read_from_device=False)
                    else:
                        gen.replay(sample=mode != "logits", mode="split" if mode == "logits" else mode)
                ttnn.synchronize_device(mesh)
                terminal[mode] = (time.monotonic() - tick) * 1000 / 128
                terminal_counters[mode] = dict(gen.counters - before_counts)
                for name in (
                    "token_refreshes",
                    "position_refreshes",
                    "rope_refreshes",
                    "page_table_refreshes",
                    "token_readbacks",
                    "logit_readbacks",
                    "synchronizations",
                ):
                    assert terminal_counters[mode].get(name, 0) == 0, (mode, name)
            result["device_traced_ms"] = terminal
            result["device_traced_counters"] = terminal_counters
            # Explicit host-greedy compatibility, separate from measured path.
            compat = gen.generate(prompt, 8, host_sampling=True, stop_on_eos=False)
            result["host_greedy_compatibility"] = dict(tokens=compat, metrics=gen.last_generation_metrics)
        assert gen.traces == saved_traces
        result["lifetime_trace_ids"] = {k: str(v) for k, v in saved_traces.items()}
        if args.batch_reconfigure:
            # Explicit model reconfiguration ends the B1 trace lifetime. Keep
            # the same 50 device-weight layers; allocate the B32 request state.
            retained_model = gen.model
            gen.close()
            del gen
            gc.collect()
            gen = KolibriGenerator(retained_model, batch_size=32, capacity=8192)
            gen.prepare()
            batch_traces = dict(gen.traces)
            batch_prompts = torch.full((32, 129), 42, dtype=torch.long)
            batch_lengths = [65 + i % 3 * 32 for i in range(32)]
            batch_lengths[-1] = 0
            first = gen.prefill_forward(
                batch_prompts, page_table=gen.state.host_page_tables, kv_cache=gen.state, prompt_lens=batch_lengths
            )
            assert first[0] == first[3]
            assert int(first[0]) == result["mixed_batch"]["first"][0], "B1/B32 token-out mismatch"
            batch_pos = [length if length else -1 for length in batch_lengths]
            logits = gen.decode_forward(
                [42] * 32, batch_pos, page_table=gen.state.host_page_tables, kv_cache=gen.state, sample_on_device=False
            )
            assert torch.equal(logits[0], logits[3]), "Equal batch slots have unequal logits"
            batch_counts = gen.counters.copy()
            for _ in range(4):
                gen.decode_forward(None, None, page_table=None, kv_cache=gen.state)
            for key in ("token_refreshes", "position_refreshes", "rope_refreshes", "page_table_refreshes"):
                assert gen.counters[key] == batch_counts[key]
            gen.reset()
            repeat = gen.prefill_forward(
                batch_prompts, page_table=gen.state.host_page_tables, kv_cache=gen.state, prompt_lens=batch_lengths
            )
            assert torch.equal(first, repeat)
            assert gen.traces == batch_traces
            result["batch32"] = dict(
                capacity=8192,
                layer_count=len(gen.model.layers),
                prompt_lengths=batch_lengths,
                first=first.tolist(),
                repeat=repeat.tolist(),
                equal_slot_logits=True,
                b1_token_match=True,
                feedback_without_host_refresh=True,
                allocator=memory_views(mesh),
                trace_ids={k: str(v) for k, v in batch_traces.items()},
            )
        if args.reduced:
            # Reconfiguration is a lifetime boundary; external allocation is
            # supported without rebinding a live captured graph.
            cache_state, model = gen.state, gen.model
            batch_size, capacity = gen.batch_size, gen.logical_capacity
            addresses = [[v.buffer_address() for v in pair] for pair in cache_state.layers]
            gen.close()
            gen = KolibriGenerator(model, batch_size=batch_size, capacity=capacity, cache_state=cache_state)
            gen.prepare()
            assert not gen.owns_cache
            assert addresses == [[v.buffer_address() for v in pair] for pair in gen.state.layers]
            values = gen.generate([42] * 65, 4, stop_on_eos=False)
            result["external_cache"] = dict(addresses_preserved=True, tokens=values)
        result["passed"] = True
        save()
        print("FULL_VALIDATION_PASS", args.tag, flush=True)
    finally:
        if gen:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
