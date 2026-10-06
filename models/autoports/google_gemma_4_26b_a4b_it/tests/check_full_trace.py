# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Persistent feedback, page-table refresh, reset and mixed-slot regression."""
import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import build_generator


def read(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).clone()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--all-layers", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 4), trace_region_size=1000000000 if args.all_layers and args.batch > 1 else 100000000
    )
    gen = None
    report = {}
    try:
        gen = build_generator(
            None, mesh, max_seq_len=1024, layer_indices=None if args.all_layers else (0, 5), trace_debug=True
        )
        if args.batch == 1:
            output = gen.generate([2] + [100] * 32, 4, stop_on_eos=False)
            prior = read(gen.tokens)
            pos = read(gen.cache_positions)
            gen._replay()
            assert torch.equal(read(gen.consumed_tokens), prior)
            assert torch.equal(read(gen.consumed_positions), pos)
            assert torch.equal(read(gen.cache_positions), pos + 1)
            report["feedback_and_positions"] = True
            before = gen.counters.get("page_table_refreshes", 0)
            table = gen.table_host.clone()
            position = read(gen.cache_positions).flatten().int()
            gen.decode_forward(None, position, page_table=table, kv_cache=gen.cache, device_feedback=True)
            assert gen.counters.get("page_table_refreshes", 0) == before
            changed = table.clone()
            changed[0, -1], changed[0, -2] = table[0, -2], table[0, -1]
            address = gen.table.buffer_address()
            gen.decode_forward(None, position + 1, page_table=changed, kv_cache=gen.cache, device_feedback=True)
            assert gen.counters["page_table_refreshes"] == before + 1
            assert gen.table.buffer_address() == address and torch.equal(read(gen.table), changed)
            report["changed_and_unchanged_page_table"] = True
            trace = gen.trace_id
            cache = gen.cache
            token = gen.tokens
            gen.reset()
            assert gen.trace_id == trace and gen.cache is cache and gen.tokens is token
            assert all(torch.count_nonzero(read(t)) == 0 for pair in gen.cache for t in pair)
            assert torch.count_nonzero(read(gen.tokens).long()) == 0
            assert torch.count_nonzero(read(gen.cache_positions)) == 0
            again = gen.generate([2] + [100] * 32, 4, stop_on_eos=False)
            assert again == output
            assert gen.trace_id == trace and gen.metrics["reused_request_trace"]
            changed_prompt = [2] + [101] * 32
            reused = gen.generate(changed_prompt, 4, stop_on_eos=False)
            assert gen.trace_id == trace and gen.metrics["reused_request_trace"]
            gen._release_trace()
            cold = gen.generate(changed_prompt, 4, stop_on_eos=False)
            assert cold == reused
            report["reset_and_repeated_generation"] = True
        else:
            batch = args.batch
            caches, table = gen.model.allocate_cache(slots=batch, context=256)
            lengths = [31, 63, 127] if batch == 3 else [31 if i % 2 == 0 else 127 for i in range(batch)]
            active = [i for i in range(batch) if batch != 3 or i != 1]
            prompts = torch.stack([torch.full((127,), 100 + slot, dtype=torch.int64) for slot in active])
            prefill = gen.prefill_forward(
                prompts, page_table=table, kv_cache=caches, prompt_lens=[lengths[i] for i in active], slots=active
            )
            assert tuple(prefill.shape) == (len(active), 1, gen.model.config.vocab_size // 4)
            batched_prefill = gen._read_logits(prefill).reshape(len(active), -1)
            positions = torch.tensor([lengths[i] if i in active else -1 for i in range(batch)], dtype=torch.int32)
            inactive_before = [read(t)[table[1]].clone() for pair in caches for t in pair] if batch == 3 else []
            decoded = gen.decode_forward(torch.full((batch,), 100), positions, page_table=table, kv_cache=caches)
            assert tuple(decoded.shape) == (batch,)
            batched_logits = (
                gen._read_logits(gen.trace_logits)
                .reshape(32 if gen.trace_logits.shape[-2] == 32 else batch, -1)[:batch]
                .clone()
            )
            prior = read(gen.tokens)
            gen.decode_forward(
                None,
                positions + torch.tensor([int(i in active) for i in range(batch)]),
                page_table=table,
                kv_cache=caches,
                device_feedback=True,
            )
            assert torch.equal(read(gen.consumed_tokens), prior)
            actual_pos = read(gen.cache_positions).flatten()
            assert actual_pos.tolist() == [lengths[i] + 2 if i in active else -1 for i in range(batch)]
            if batch == 3:
                after = [read(t)[table[1]].clone() for pair in caches for t in pair]
                assert all(torch.equal(a, b) for a, b in zip(inactive_before, after))
                assert int(read(gen.positions).flatten()[1]) in (-1, 4294967295)
            second_batched = (
                gen._read_logits(gen.trace_logits)
                .reshape(32 if gen.trace_logits.shape[-2] == 32 else batch, -1)[:batch]
                .clone()
            )
            gen._release_trace()
            isolated_cache, isolated_table = gen.model.allocate_cache(slots=1, context=256)
            isolated_device_table = gen.model.upload(isolated_table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            slot_checks = []
            for prompt_row, slot in enumerate(active):
                if args.all_layers and prompt_row not in (0, len(active) - 1):
                    continue
                prefill_reference = gen.prefill_forward(
                    prompts[prompt_row : prompt_row + 1],
                    page_table=isolated_table,
                    kv_cache=isolated_cache,
                    prompt_lens=[lengths[slot]],
                )
                reference_prefill = gen._read_logits(prefill_reference).flatten()
                prefill_pcc = float(torch.corrcoef(torch.stack([reference_prefill, batched_prefill[prompt_row]]))[0, 1])
                assert prefill_pcc >= 0.999, (slot, prefill_pcc)
                single_logits = gen.model.decode_forward(
                    gen.model.upload(torch.tensor([[[[100]]]], dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                    current_pos=gen.model.upload(
                        positions[slot : slot + 1].reshape(1, 1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
                    ),
                    cache_pos=gen.model.upload(positions[slot : slot + 1], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                    page_table=isolated_device_table,
                    kv_cache=isolated_cache,
                    batch=1,
                )
                reference = gen._read_logits(single_logits).flatten()
                actual = batched_logits[slot]
                pcc = float(torch.corrcoef(torch.stack([reference, actual]))[0, 1])
                top5 = int(actual.argmax()) in reference.topk(5).indices.tolist()
                assert pcc >= 0.999 and top5, (slot, pcc, top5)
                second_logits = gen.model.decode_forward(
                    gen.model.upload(
                        prior.flatten()[slot : slot + 1].reshape(1, 1, 1, 1).int(), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
                    ),
                    current_pos=gen.model.upload(
                        (positions[slot : slot + 1] + 1).reshape(1, 1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
                    ),
                    cache_pos=gen.model.upload(positions[slot : slot + 1] + 1, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                    page_table=isolated_device_table,
                    kv_cache=isolated_cache,
                    batch=1,
                )
                second_reference = gen._read_logits(second_logits).flatten()
                second_pcc = float(torch.corrcoef(torch.stack([second_reference, second_batched[slot]]))[0, 1])
                assert second_pcc >= 0.999
                slot_checks.append(
                    dict(
                        slot=slot,
                        prefill_pcc=prefill_pcc,
                        decode_pcc=pcc,
                        second_decode_pcc=second_pcc,
                        top5_match=top5,
                    )
                )
            report["isolated_slot_comparison"] = slot_checks
            report.update(
                batch=batch,
                active_slots=active,
                mixed_lengths=lengths,
                feedback=True,
                inactive_cache_unchanged=True if batch == 3 else None,
            )
        report["precision_runtime"] = gen.model.precision_summary()
        report["layer_count"] = len(gen.model.layers)
        report["counters"] = gen.counters
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(report, flush=True)
    finally:
        if gen is not None:
            gen.teardown()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
