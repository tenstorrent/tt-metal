# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-shape trace feedback, page ownership, fixed-slot and batch checks."""

import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_27b.tests.run_optimized_decoder import device_only
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator


def host(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=3)
    p.add_argument("--full", action="store_true")
    p.add_argument("--qualitative", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
    gen = None
    try:
        gen = build_generator("models/autoports/qwen_qwen3_8_27b", mesh, layer_indices=None if a.full else [0, 3])
        model_step, sampling_step = gen._model_step, gen._sampling_step

        def guarded_model():
            with device_only():
                return model_step()

        def guarded_sampling(logits):
            with device_only():
                return sampling_step(logits)

        gen._model_step, gen._sampling_step = guarded_model, guarded_sampling
        b = a.batch
        cache = gen._ensure_cache(b, 128)
        gen.bind_cache(cache, gen.page_table)
        assert not gen.owns_cache
        gen.reset()
        slots = [0] if b == 1 else [b - 1, 0]
        lengths = [33] if b == 1 else [31, 33]
        ids = torch.full((len(slots), max(lengths)), 1596, dtype=torch.int64)
        before = [
            {name: host(getattr(s, name)).clone() for name in ("recurrent", "conv")}
            for s in cache.layers
            if s.recurrent is not None
        ]
        print("MIXED_PREFILL", flush=True)
        logits = gen.prefill_forward(ids, page_table=gen.page_table, kv_cache=cache, prompt_lens=lengths, slots=slots)
        assert len(logits) == len(slots)
        del logits
        positions = torch.full((b,), -1, dtype=torch.int32)
        for slot, length in zip(slots, lengths):
            positions[slot] = length
        tok = torch.full((b, 1), 1596, dtype=torch.int32)
        start_counts = gen.counters.copy()
        print("CAPTURE_DECODE", flush=True)
        output = gen.decode_forward(tok, positions, page_table=gen.page_table, kv_cache=cache, active_slots=slots)
        persistent_id = gen.tokens.buffer_address()
        exact = host(gen.tokens).reshape(-1)[:b].long()
        assert torch.equal(output, exact)
        after_pos = host(gen.positions).reshape(-1).int()
        expected = positions.clone()
        expected[slots] += 1
        assert torch.equal(after_pos, expected), (after_pos, expected)
        first_logits = gen._host_logits(gen.logits).clone()
        before_steady = gen.counters.copy()
        feedback = exact.clone()
        for step in range(2):
            assert torch.equal(host(gen.tokens).reshape(-1)[:b].long(), feedback)
            with device_only():
                device_result = gen.decode_forward(page_table=gen.page_table, kv_cache=cache, read_from_device=False)
            assert device_result is gen.tokens
            feedback = host(device_result).reshape(-1)[:b].long()
            assert gen.tokens.buffer_address() == persistent_id
        expected[slots] += 2
        assert torch.equal(host(gen.positions).reshape(-1).int(), expected)
        assert torch.equal(host(gen.rope_indices).reshape(-1)[slots].long(), expected[slots].long())
        for key in ("token_refreshes", "position_refreshes", "rope_refreshes", "page_table_refreshes"):
            assert gen.counters[key] == before_steady[key], (key, dict(gen.counters))
        final_logits = gen._host_logits(gen.logits)
        assert not torch.equal(first_logits, final_logits)
        inactive = [i for i in range(b) if i not in slots]
        for old, state in zip(before, [s for s in cache.layers if s.recurrent is not None]):
            for name, previous in old.items():
                now = host(getattr(state, name))
                if inactive:
                    assert torch.equal(previous[inactive], now[inactive]), f"inactive {name} changed"
        print("PAGE_TABLE_CASES", flush=True)
        # Establish an exact same-input oracle before physical-page permutation.
        # Allocate backups before capture so the trace allocator sees them live.
        gen._release_traces()
        backups = [
            (tensor, ttnn.clone(tensor))
            for state in cache.layers
            for tensor in (state.recurrent, state.conv)
            if tensor is not None
        ]
        backups += [
            (tensor, ttnn.clone(tensor))
            for tensor in (gen.tokens, gen.positions, gen.rope_indices, gen.sampler.seeds_tt_tensor)
        ]
        budget = gen.remaining_steps
        gen.decode_forward(page_table=gen.page_table, kv_cache=cache)
        canonical_logits = gen._host_logits(gen.logits).clone()
        for tensor, backup in backups:
            ttnn.copy(backup, tensor)
        gen.remaining_steps = budget
        # Preserve exact cache contents while changing physical page addresses.
        pages = host(gen.page_table).int()
        permutation = torch.arange(cache.num_pages - 1, -1, -1)
        for state in cache.layers:
            if state.key is None:
                continue
            for tensor in (state.key, state.value):
                rank_parts = [ttnn.to_torch(v)[permutation].contiguous() for v in ttnn.get_device_tensors(tensor)]
                changed = ttnn.from_torch(
                    torch.cat(rank_parts, dim=1),
                    device=mesh,
                    dtype=tensor.dtype,
                    layout=tensor.layout,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
                )
                ttnn.copy(changed, tensor)
                ttnn.deallocate(changed)
        changed_pages = (cache.num_pages - 1 - pages).int()
        n = gen.counters["page_table_refreshes"]
        gen.decode_forward(page_table=changed_pages, kv_cache=cache)
        assert gen.counters["page_table_refreshes"] == n + 1
        assert torch.equal(canonical_logits, gen._host_logits(gen.logits)), "physical page remapping changed logits"
        del backups, canonical_logits
        gen.decode_forward(page_table=changed_pages.clone(), kv_cache=cache)
        assert gen.counters["page_table_refreshes"] == n + 1
        # Device mutation invalidates host shadow; switching back must copy.
        gen._refresh_table(gen.page_table)
        gen._refresh_table(changed_pages)
        assert gen.counters["page_table_refreshes"] == n + 2
        print("SAMPLING_MODES", flush=True)
        gen.set_sampling_params(top_k=8, top_p=0.9, temperature=0.8, seed=12)
        sampled = gen.decode_forward(page_table=gen.page_table, kv_cache=cache)
        assert ((sampled >= 0) & (sampled < gen.model.config.vocab_size)).all()
        gen.set_sampling_params()
        gen.decode_forward(page_table=gen.page_table, kv_cache=cache)
        print("BATCH_LOGIT_DETERMINISM", flush=True)
        gen.reset()
        identical = torch.full((b, 33), 1596, dtype=torch.int64)
        same = gen.prefill_forward(
            identical, page_table=gen.page_table, kv_cache=cache, prompt_lens=[33] * b, return_all_logits=True
        )
        for row in range(1, b):
            assert torch.equal(same[0], same[row]), f"prefill logits differ at slot {row}"
        gen.reset()
        repeated = gen.prefill_forward(
            identical, page_table=gen.page_table, kv_cache=cache, prompt_lens=[33] * b, return_all_logits=True
        )
        assert torch.equal(same, repeated), "prefill repeat logits differ"
        del same, repeated
        gen.decode_forward(
            tok, torch.full((b,), 33), page_table=gen.page_table, kv_cache=cache, active_slots=list(range(b))
        )
        decoded = gen._host_logits(gen.logits)[0, 0, :b]
        for row in range(1, b):
            assert torch.equal(decoded[0], decoded[row]), f"decode logits differ at slot {row}"
        print("UNALIGNED_CONTINUATION", flush=True)
        gen.reset()
        prefix = gen.prefill_forward(identical[:, :31], page_table=gen.page_table, kv_cache=cache, prompt_lens=[31] * b)
        del prefix
        continued = gen.prefill_forward(
            identical[:, 31:],
            page_table=gen.page_table,
            kv_cache=cache,
            prompt_lens=[2] * b,
            start_pos=[31] * b,
            return_all_logits=True,
        )
        gen.reset()
        whole = gen.prefill_forward(
            identical, page_table=gen.page_table, kv_cache=cache, prompt_lens=[33] * b, return_all_logits=True
        )
        correlation = torch.corrcoef(torch.stack([continued.flatten(), whole[:, 31:].flatten()]))[0, 1].item()
        assert correlation >= 0.999, correlation
        del continued, whole
        gen.decode_forward(tok, torch.full((b,), 33), page_table=gen.page_table, kv_cache=cache)
        sampling_replays = gen.counters["sampling_replays"]
        compatibility = gen.decode_forward(page_table=gen.page_table, kv_cache=cache, host_sampling=True)
        assert compatibility.shape == (b, gen.model.config.vocab_size)
        assert gen.counters["sampling_replays"] == sampling_replays
        # High-level compatibility owns host feedback explicitly and should
        # agree with the common device greedy sampler for the same prompt.
        host_tokens = gen.generate([1596] * 33, 4, host_sampling=True)
        device_tokens = gen.generate([1596] * 33, 4)
        assert host_tokens == device_tokens, (host_tokens, device_tokens)
        report = dict(
            external_cache_binding=True,
            unaligned_continuation_pcc=correlation,
            explicit_host_sampling_equal=True,
            device_only_model_and_sampling_guard=True,
            logit_batch_and_repeat_determinism=True,
            batch=b,
            full=a.full,
            slots=slots,
            prompt_lengths=lengths,
            persistent_feedback=True,
            positions_coherent=True,
            inactive_state_unchanged=True,
            page_table_changed_and_unchanged=True,
            physical_page_remapping_logits_equal=True,
            greedy_sampled_alternation=True,
            steady_state_refresh_counts={
                k: 0 for k in ("token_refreshes", "position_refreshes", "rope_refreshes", "page_table_refreshes")
            },
            counters=dict(gen.counters),
        )
        a.output.write_text(json.dumps(report, indent=2) + "\n")
        if a.qualitative:
            assert a.full, "Qualitative output requires the full layer stack"
            from models.autoports.qwen_qwen3_8_27b.tests.tt_qualitative import run

            report["qualitative"] = str(
                run(gen, Path("models/autoports/qwen_qwen3_8_27b"), control_name="hf_qualitative_256.json")
            )
            a.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
