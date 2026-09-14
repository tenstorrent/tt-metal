# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare serving-owned prefill traces with eager controls on the same adapter.

Device reads here are correctness diagnostics, never serving performance evidence.
Run after every server has stopped; --full selects the complete model.
"""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator, configure_fabric
from models.autoports.qwen_qwen3_8_27b.tt.generator_vllm import Qwen38ForCausalLM


def forbidden(*args, **kwargs):
    raise AssertionError("Warmed serving request dispatched eager prefill/sampling or recaptured")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--lengths", default="31,33,127,128,129,4095,4096,4097")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--multi", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    report = dict(full=args.full, batch=args.batch, context=262144, rows=[], status="running")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    configure_fabric()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=134217728)
    gen = None
    try:
        gen = build_generator(
            Path(__file__).resolve().parents[1],
            mesh,
            layer_indices=None if args.full else [0, 3],
            precision_config=Path(__file__).resolve().parents[1] / "doc/datatype_sweep/selected_precision_config.json",
        )
        adapter = Qwen38ForCausalLM(gen, args.batch, 262144)
        adapter.allocate_kv_cache((8192 + args.batch, 1, 32, 256), None, len(gen.model.layers))
        base = gen.tokenizer.encode("A quiet library contains shelves of books. " * 600, add_special_tokens=False)
        params = SimpleNamespace(temperature=[0.0], top_k=[1], top_p=[1.0], seed=[17])
        decode_params = SimpleNamespace(
            temperature=[0.0] * args.batch,
            top_k=[1] * args.batch,
            top_p=[1.0] * args.batch,
            seed=[17] * args.batch,
        )
        captured = {}
        original_owned = gen._prefill_for_generate
        original_public = gen.prefill_forward

        def host_logits(output):
            return torch.cat([ttnn.to_torch(t)[:, :, :1, :].float() for t in ttnn.get_device_tensors(output)], -1)

        def owned(tokens, **kwargs):
            output = original_owned(tokens, **kwargs)
            captured["logits"] = host_logits(output)
            return output

        def eager(tokens, **kwargs):
            outputs = original_public(tokens, **kwargs)
            captured["logits"] = host_logits(outputs[0])
            return gen.sample_prefill(outputs)

        def public(*a, **kw):
            outputs = original_public(*a, **kw)
            captured["logits"] = host_logits(outputs[0])
            return outputs

        def request(prompt, table, *, control=False, guarded=False):
            captured.clear()
            before = gen.counters.copy()
            identities = (
                gen.trace,
                gen.sample_trace,
                gen.prefill_trace,
                getattr(gen, "prefill_sample_trace", None),
                id(gen.tokens),
                id(gen.positions),
                id(gen.rope_indices),
                id(gen.page_table),
            )
            with (
                patch.object(gen, "serving_prefill_tokens", eager) if control else nullcontext(),
                patch.object(gen, "_prefill_for_generate", owned),
                patch.object(gen, "prefill_forward", public),
                patch.object(gen.model, "prefill", forbidden) if guarded and len(prompt) <= 4096 else nullcontext(),
                patch.object(gen, "_sampling_step", forbidden) if guarded else nullcontext(),
                patch.object(gen, "_capture", forbidden) if guarded else nullcontext(),
            ):
                first, _ = adapter.prefill_forward(
                    torch.tensor([prompt], dtype=torch.int32),
                    page_table=table[:1],
                    kv_cache=adapter.cache,
                    prompt_lens=[len(prompt)],
                    sampling_params=params,
                    empty_slots=[0],
                )
                values = [int(first[0, 0])]
                positions = torch.full((args.batch,), -1, dtype=torch.int32)
                positions[0] = len(prompt)
                tokens = torch.zeros(args.batch, 1, dtype=torch.int32)
                tokens[0, 0] = values[-1]
                for step in range(3):
                    device = adapter.decode_forward(
                        tokens,
                        positions,
                        table,
                        adapter.cache,
                        read_from_device=False,
                        sampling_params=decode_params,
                        reset_batch=step == 0,
                    )
                    host, events = adapter.read_decode_output(device, async_read=True)
                    for event in events:
                        ttnn.event_synchronize(event)
                    values.append(int(adapter.process_decode_output_host(host, is_tokens=True)[0, 0]))
            if guarded:
                assert identities == (
                    gen.trace,
                    gen.sample_trace,
                    gen.prefill_trace,
                    gen.prefill_sample_trace,
                    id(gen.tokens),
                    id(gen.positions),
                    id(gen.rope_indices),
                    id(gen.page_table),
                )
            position = int(ttnn.to_torch(ttnn.get_device_tensors(gen.positions)[0]).reshape(-1)[0])
            rope = int(ttnn.to_torch(ttnn.get_device_tensors(gen.rope_indices)[0]).reshape(-1)[0])
            seed = int(ttnn.to_torch(ttnn.get_device_tensors(gen.sampler.seeds_tt_tensor)[0]).reshape(-1)[0])
            assert position == rope == len(prompt) + 3, (position, rope)
            assert seed == 17 + len(prompt) + 4, seed
            return (
                dict(
                    tokens=values,
                    counters=dict(gen.counters - before),
                    position=position,
                    rope=rope,
                    seed=seed,
                    stable_identities=guarded,
                ),
                captured["logits"].clone(),
            )

        for length in map(int, args.lengths.split(",")):
            prompt = base[:length]
            assert len(prompt) == length
            table = torch.zeros(args.batch, 8192, dtype=torch.int32)
            needed = (length + 4 + 31) // 32
            table[0, :needed] = torch.arange(1, needed + 1)
            control, expected = request(prompt, table, control=True)
            warm, actual = request(prompt, table)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert warm["tokens"] == control["tokens"]
            repeat, repeated = request(prompt, table, guarded=True)
            torch.testing.assert_close(repeated, expected, rtol=0, atol=0)
            assert repeat["tokens"] == control["tokens"]
            if length <= 4096:
                assert repeat["counters"].get("prefill_replays") == 1, repeat
                assert repeat["counters"].get("prefill_token_refreshes") == 1, repeat
                assert repeat["counters"].get("page_table_refreshes", 0) == 0, repeat
            changed_prompt = prompt.copy()
            changed_prompt[-1] = next(token for token in base if token != prompt[-1])
            changed_table = table.clone()
            changed_table[0, :needed] = torch.arange(needed, 0, -1)
            changed, changed_logits = request(changed_prompt, changed_table, guarded=True)
            assert not torch.equal(changed_logits, expected), "Changed prompt produced stale logits"
            changed_control, changed_expected = request(changed_prompt, changed_table, control=True)
            torch.testing.assert_close(changed_logits, changed_expected, rtol=0, atol=0)
            assert changed["tokens"] == changed_control["tokens"]
            report["rows"].append(
                dict(
                    length=length,
                    control=control,
                    warm=warm,
                    repeat=repeat,
                    changed=changed,
                    exact_logits=True,
                    exact_tokens=True,
                    changed_logits=True,
                    changed_control=changed_control,
                )
            )
            save()
            print("PREFILL_TRACE_CASE_PASS", length, flush=True)
        if args.multi:
            assert args.batch == 4, "Multi-row control uses fixed slots1/3 in a four-slot cache"
            slots, lengths = [1, 3], [31, 45]
            table = torch.zeros(4, 8192, dtype=torch.int32)
            for slot, length in zip(slots, lengths):
                table[slot, :4] = torch.arange(1 + slot * 128, 5 + slot * 128)
            prompts = torch.zeros(2, max(lengths), dtype=torch.int32)
            for row, length in enumerate(lengths):
                prompts[row, :length] = torch.tensor(base[row : row + length])
            multi_params = SimpleNamespace(temperature=[0.0] * 2, top_k=[1] * 2, top_p=[1.0] * 2, seed=[17, 23])
            multi_decode_params = SimpleNamespace(
                temperature=[0.0] * 4, top_k=[1] * 4, top_p=[1.0] * 4, seed=[1, 17, 1, 23]
            )

            def multi_request(*, control=False, guarded=False):
                logits_rows = []
                before = gen.counters.copy()

                def capture_public(*a, **kw):
                    outputs = original_public(*a, **kw)
                    logits_rows.extend(host_logits(output) for output in outputs)
                    return outputs

                def eager_multi(tokens, **kwargs):
                    outputs = []
                    for row, (start, end, slot) in enumerate(
                        zip(kwargs["start_pos"], kwargs["prompt_lens"], kwargs["slots"])
                    ):
                        outputs.extend(
                            capture_public(
                                tokens[row : row + 1, start:end],
                                page_table=kwargs["page_table"],
                                kv_cache=kwargs["kv_cache"],
                                prompt_lens=[end - start],
                                start_pos=[start],
                                slots=[slot],
                            )
                        )
                    return gen.sample_prefill(outputs)

                with (
                    patch.object(gen, "serving_prefill_tokens", eager_multi) if control else nullcontext(),
                    patch.object(gen, "prefill_forward", capture_public),
                    patch.object(gen, "_sampling_step", forbidden) if guarded else nullcontext(),
                    patch.object(gen, "_capture", forbidden) if guarded else nullcontext(),
                ):
                    first, _ = adapter.prefill_forward(
                        prompts,
                        page_table=table[slots],
                        kv_cache=adapter.cache,
                        prompt_lens=lengths,
                        sampling_params=multi_params,
                        empty_slots=slots,
                    )
                    values = [first[:, 0].tolist()]
                    positions = torch.tensor([-1, lengths[0], -1, lengths[1]], dtype=torch.int32)
                    tokens = torch.zeros(4, 1, dtype=torch.int32)
                    tokens[slots, 0] = first[:, 0].int()
                    for step in range(3):
                        device = adapter.decode_forward(
                            tokens,
                            positions,
                            table,
                            adapter.cache,
                            read_from_device=False,
                            sampling_params=multi_decode_params,
                            reset_batch=step == 0,
                        )
                        host, events = adapter.read_decode_output(device, async_read=True)
                        for event in events:
                            ttnn.event_synchronize(event)
                        values.append(adapter.process_decode_output_host(host, is_tokens=True)[slots, 0].tolist())
                return values, torch.stack(logits_rows), dict(gen.counters - before)

            control, expected, _ = multi_request(control=True)
            warm, actual, warm_counts = multi_request()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert warm == control
            repeated, actual, repeat_counts = multi_request(guarded=True)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert repeated == control
            assert repeat_counts.get("prefill_sampling_replays") == 1, repeat_counts
            assert repeat_counts.get("page_table_refreshes", 0) == 0, repeat_counts
            report["multirow"] = dict(
                slots=slots,
                lengths=lengths,
                exact_logits=True,
                exact_tokens=True,
                tokens=repeated,
                warm_counters=warm_counts,
                repeat_counters=repeat_counts,
            )
            save()
            print("MULTIROW_PREFILL_SAMPLING_TRACE_PASS", flush=True)
        report["status"] = "pass"
        save()
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
