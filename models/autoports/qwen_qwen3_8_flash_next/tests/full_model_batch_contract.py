"""Mixed fixed-row requests, external cache binding and device feedback probe."""

import argparse
import hashlib
import json
from pathlib import Path
import torch
import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.generator import build_generator
from models.autoports.qwen_qwen3_8_flash_next.tt.model import ModelCache


def host(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,1,3")
    ap.add_argument("--adopt-cache", action="store_true", help="Adopt exported buffers without a second allocation")
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--tag", default="batch_contract")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--trace-region-size", type=int, default=None)
    ap.add_argument("--stage", default="full_model")
    ap.add_argument("--prefill-trace-max", type=int, default=None)
    args = ap.parse_args()
    torch.set_num_threads(8)
    ttnn.CONFIG.throw_exception_on_fallback = True
    root = Path("models/autoports/qwen_qwen3_8_flash_next")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    trace_bytes = args.trace_region_size or (
        (512000000 if args.batch > 1 else 768000000) if args.layers == "all" else 160000000
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=trace_bytes)
    try:
        gen = build_generator(
            root,
            mesh,
            layer_indices=None if args.layers == "all" else [int(x) for x in args.layers.split(",") if x],
            batch_size=args.batch,
            max_seq_len=args.context,
            max_requests=args.batch + 2,
            checkpoint=args.checkpoint,
            prefill_trace_max=args.prefill_trace_max,
        )

        def empty_like(tensor):
            return ttnn.empty(
                tuple(tensor.shape),
                dtype=tensor.dtype,
                layout=tensor.layout,
                memory_config=tensor.memory_config(),
                device=mesh,
            )

        external_rows = []
        for row in ([] if args.adopt_cache else gen.model.cache.layers):
            external_rows.append(
                {
                    key: (
                        type(value)(**{k: empty_like(v) for k, v in vars(value).items()})
                        if key == "ple_cache"
                        else empty_like(value)
                    )
                    for key, value in row.items()
                }
            )
        external = ModelCache([dict(row) for row in gen.model.cache.layers] if args.adopt_cache else external_rows)
        gen.prepare(kv_cache=external)
        assert gen.model.cache is external and gen.kv_cache is external
        b = args.batch
        slots = list(range(2, b + 2))[::-1]
        lengths = [7 if i % 2 == 0 else 31 for i in range(b)]
        if b > 2:
            lengths[-1] = 0
        tokens = torch.tensor([[100 + j % 100 for j in range(max(lengths))] for _ in range(b)])
        active = [n > 0 for n in lengths]
        gen.reset()
        logits = gen.prefill_forward(
            tokens,
            page_table=gen.page_table,
            kv_cache=external,
            prompt_lens=lengths,
            slots=slots,
            return_all_logits=True,
        )
        assert tuple(logits.shape) == (b, max(lengths), gen.model.cfg.vocab_size)
        if b > 2:
            assert torch.count_nonzero(logits[-1]) == 0
        full_row_last = gen._read_logits(gen.tt_logits).clone()
        gen._replay("greedy")
        first = host(gen.tt_token).reshape(-1).tolist()
        before = dict(gen.counters)
        steps = []
        for _ in range(3):
            consumed = host(gen.tt_token).reshape(-1).tolist()
            pos = host(gen.tt_pos)[:, 0].tolist()
            gen._replay("decode")
            gen._replay("greedy")
            feedback = host(gen.tt_token).reshape(-1).tolist()
            nextpos = host(gen.tt_pos)[:, 0].tolist()
            assert nextpos == [p + 1 if a else 0 for p, a in zip(pos, active)]
            if steps:
                assert consumed == steps[-1]["feedback"]
            steps.append(dict(consumed=consumed, positions=pos, feedback=feedback, next_positions=nextpos))
        delta = {k: gen.counters[k] - before.get(k, 0) for k in gen.counters}
        for key in ("token_refreshes", "position_refreshes", "page_table_refreshes", "synchronizations"):
            assert delta.get(key, 0) == 0, (key, delta)
        # Compare the changed B>1 last-row projection against the established
        # full-row head, including shorter rows retaining their last logits.
        trace_ids = dict(gen.traces)
        gen.reset()
        inactive_before = gen._read_logits(gen.tt_logits)[~torch.tensor(active)].clone()
        gen.prefill_forward(
            tokens, page_table=gen.page_table, kv_cache=external,
            prompt_lens=lengths, slots=slots, read_tokens=False,
        )
        selected_last = gen._read_logits(gen.tt_logits)
        active_mask = torch.tensor(active)
        assert torch.equal(full_row_last[active_mask], selected_last[active_mask]), "Last-row head logits"
        assert torch.equal(inactive_before, selected_last[~active_mask]), "Inactive logits retained"
        selected_first = host(gen.tt_token).reshape(-1).tolist()
        assert all(x == y for x, y, a in zip(first, selected_first, active) if a), "Last-row greedy tokens"
        selected_steps = []
        for control in steps:
            consumed = host(gen.tt_token).reshape(-1).tolist()
            pos = host(gen.tt_pos)[:, 0].tolist()
            gen._replay("decode")
            gen._replay("greedy")
            feedback = host(gen.tt_token).reshape(-1).tolist()
            nextpos = host(gen.tt_pos)[:, 0].tolist()
            assert pos == control["positions"] and nextpos == control["next_positions"]
            for key, values in (("consumed", consumed), ("feedback", feedback)):
                assert all(x == y for x, y, a in zip(control[key], values, active) if a), key
            selected_steps.append(dict(consumed=consumed, positions=pos, feedback=feedback, next_positions=nextpos))
        assert gen.traces == trace_ids and gen.counters["captures"] == len(trace_ids)
        # Repeat exactly, then move identical prompts between lane/slot pairs.
        gen.reset()
        repeat = gen.prefill_forward(
            tokens,
            page_table=gen.page_table,
            kv_cache=external,
            prompt_lens=lengths,
            slots=slots,
            return_all_logits=True,
        )
        assert torch.equal(logits, repeat), "Reset must restore identical logits"
        common = torch.tensor([[100, 101, 102, 103, 104, 105, 106] for _ in range(b)])
        gen.reset()
        equal = gen.prefill_forward(
            common,
            page_table=gen.page_table,
            kv_cache=external,
            prompt_lens=[7] * b,
            slots=slots,
            return_all_logits=True,
        )
        for lane in range(1, b):
            assert torch.equal(equal[0], equal[lane]), ("batch lane logits", lane)
        # Same physical4+2+1 decomposition across an external call boundary.
        gen.reset()
        prefix = gen.prefill_forward(
            common[:, :4],
            page_table=gen.page_table,
            kv_cache=external,
            prompt_lens=[4] * b,
            slots=slots,
            return_all_logits=True,
        )
        tail = gen.prefill_forward(
            common[:, 4:],
            page_table=gen.page_table,
            kv_cache=external,
            prompt_lens=[3] * b,
            start_pos=[4] * b,
            slots=slots,
            return_all_logits=True,
        )
        assert torch.equal(torch.cat([prefix, tail], dim=1), equal), "Explicit continuation prefill"
        # Activity-only scheduler updates must retain logical slots and pages,
        # including requests temporarily routed to private inactive storage.
        pages = gen.page_table.flip(1)
        distinct = common + torch.arange(b)[:, None] * 13
        scheduler_runs = []
        for explicit in (True, False):
            gen.reset()
            gen.prefill_forward(distinct, page_table=pages, kv_cache=external, prompt_lens=[7] * b, slots=slots)
            observations = []
            for enabled, positions in (([False] + [True] * (b - 1), [7] * b), ([True] * b, [7] + [8] * (b - 1))):
                gen.decode_forward(
                    107 + torch.arange(b)[:, None] * 13,
                    positions,
                    page_table=pages if explicit else None,
                    kv_cache=external,
                    slots=slots if explicit else None,
                    active_rows=enabled,
                    read_tokens=False,
                )
                physical_slots = host(gen.tt_slots)[:, 0].tolist()
                assert physical_slots == [
                    s if a else gen.model.max_requests + i for i, (s, a) in enumerate(zip(slots, enabled))
                ]
                physical_pages = host(gen.tt_page)
                for i, enabled_row in enumerate(enabled):
                    if enabled_row:
                        assert torch.equal(physical_pages[i], pages[i])
                observations.append((host(gen.tt_token), gen._read_logits(gen.tt_logits), host(gen.tt_pos)))
            scheduler_runs.append(observations)
        for explicit, retained in zip(*scheduler_runs):
            assert all(torch.equal(x, y) for x, y in zip(explicit, retained)), "Retained scheduler mappings"
        single = common[0].tolist()
        device_output = gen.generate(single, 4)
        gen.sampling_mode = "host"
        host_output = gen.generate(single, 4)
        gen.sampling_mode = "device"
        assert host_output == device_output, ("host compatibility", host_output, device_output)
        gen.set_sampling_params(top_k=8, top_p=0.8, temperature=0.7, seed=123)
        sampled = gen.generate(single, 4, sampling="sample")
        gen.set_sampling_params(top_k=8, top_p=0.8, temperature=0.7, seed=123)
        assert gen.generate(single, 4, sampling="sample") == sampled, "Seeded sampled replay"
        gen.set_sampling_params(top_k=1, top_p=0.0, temperature=1.0, seed=321)
        assert gen.generate(single, 4, sampling="sample") == device_output, "Updated parameters select greedy"
        highlevel = gen.prefill_logits(single)
        assert highlevel.shape == (1, len(single), gen.model.cfg.vocab_size)
        ids = dict(gen.traces)
        gen.reset()
        gen.prefill_forward(
            tokens,
            page_table=gen.page_table.flip(1),
            kv_cache=external,
            prompt_lens=lengths,
            slots=slots,
            active_rows=active,
        )
        assert gen.traces == ids and gen.counters["captures"] == len(ids)
        result = dict(
            args=vars(args),
            slots=slots,
            lengths=lengths,
            first_tokens=first,
            steps=steps,
            steady_state_counter_delta=delta,
            repeated_logits_exact=True,
            batch_lane_logits_exact=True,
            external_prefill_continuation_exact=True,
            scheduler_activity_retains_slots_and_pages=True,
            external_cache=True,
            external_cache_distinct_allocation=not args.adopt_cache,
            last_row_differential=dict(
                active_sampler_logits_exact=True, inactive_logits_retained=True,
                active_greedy_tokens_exact=True, active_feedback_steps_exact=True,
                first_tokens=selected_first, steps=selected_steps,
                trace_ids_retained=True,
            ),
            counters=dict(gen.counters),
        )
        result["highlevel_single_request_tokens"] = device_output
        result["host_compatibility_equal"] = True
        result["sample_parameter_update_and_seed_repeat"] = True
        result["seeded_sample_tokens"] = sampled
        result["source_sha256"] = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / "tt").glob("*.py")
        }
        result["canonical_eos"] = gen.model.generation_config.eos_token_id
        result["trace_region_bytes"] = trace_bytes
        result["trace_ids"] = {k: str(v) for k, v in gen.traces.items()}
        (root / "doc" / args.stage / f"{args.tag}.json").write_text(json.dumps(result, indent=2) + "\n")
        print("BATCH_CONTRACT_PASS", json.dumps(result), flush=True)
        gen.close()
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
