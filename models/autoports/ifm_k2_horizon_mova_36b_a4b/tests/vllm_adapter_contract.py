# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced real-layer adapter contract; authoritative control versus resident replay."""

import argparse
import json
import os
from pathlib import Path

import torch
from vllm_tt_plugin.model_input import TTSamplingParams

import ttnn

from ..tt.generator_vllm import K2HorizonForCausalLM


def check_intermediate_prefill(adapter, cache):
    g = adapter.generator
    tokens = torch.stack([torch.arange(110, 241), torch.arange(310, 441)])
    table = torch.arange(1, 33, dtype=torch.int32).reshape(2, 16)
    params = TTSamplingParams(
        temperature=[0.7, 0.7],
        top_k=[16, 16],
        top_p=[1.0, 1.0],
        seed=[41, 42],
        presence_penalty=[0.2, 0.2],
        frequency_penalty=[0.1, 0.1],
        repetition_penalty=[1.1, 1.1],
        enable_log_probs=[False, False],
        num_logprobs=[-1, -1],
    )

    def prefill(ends, starts, mask, host=False):
        return adapter.prefill_forward(
            tokens,
            page_table=table,
            kv_cache=cache,
            prompt_lens=ends,
            start_pos=starts,
            empty_slots=[0, 1],
            enable_trace=True,
            sampling_params=None if host else params,
            intermediate_prefill_mask=torch.tensor(mask, dtype=torch.bool),
        )

    def state():
        tensors = dict(
            tokens=g.tokens,
            seeds=g.sampler.tt_sampling.seeds_tt_tensor,
            counts=g.sampler.tt_penalties.output_counts,
            mask=g.sampler.tt_penalties.output_mask,
            prompt=g.sampler.tt_penalties.prompt_mask,
        )
        return {
            name: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).reshape(32, -1).clone() for name, t in tensors.items()
        }

    control = prefill([129, 131], [0, 0], [False, False])
    before = state()
    counters = g.counters.copy()
    prefill([128, 128], [0, 0], [True, True])
    after = state()
    assert all(torch.equal(before[k], after[k]) for k in before), "Intermediate-only prefill changed sampler state"
    for key in ("sampling_replays", "token_readbacks", "logit_readbacks", "prefill_seed_resets"):
        assert g.counters[key] == counters[key], key
    mixed = prefill([129, 129], [128, 128], [False, True])
    after_mixed = state()
    assert all(torch.equal(after[k][1:], after_mixed[k][1:]) for k in before), "Mixed prefill changed non-final slots"
    assert torch.equal(control[0], mixed[0])
    final = prefill([129, 131], [128, 129], [True, False])
    assert torch.equal(control[1], final[1])
    assert g.counters["logit_readbacks"] == counters["logit_readbacks"]
    # Explicit host controls still suppress intermediate logits; final rows
    # return actual logits while intermediate placeholders remain zero.
    before_host = state()
    host = prefill([128, 128], [0, 0], [True, True], host=True)
    assert host.shape == (2, 1, g.model.vocab_size) and not host.any()
    assert all(torch.equal(before_host[k], state()[k]) for k in before_host)
    assert g.counters["logit_readbacks"] == counters["logit_readbacks"]
    mixed_host = prefill([129, 129], [128, 128], [False, True], host=True)
    assert mixed_host[0].abs().sum() > 0 and not mixed_host[1].any()
    assert g.counters["logit_readbacks"] == counters["logit_readbacks"] + 1
    return dict(
        status="pass",
        intermediate_state_preserved=True,
        mixed_inactive_state_preserved=True,
        chunked_seeded_tokens_match_control=True,
        ordinary_logit_reads=0,
        explicit_host_final_logit_reads=1,
        partial_tail_lengths=[1, 2],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "readiness_vllm/adapter_contract.json"
    )
    args = parser.parse_args()
    os.environ["K2_VLLM_LAYERS"] = "[0,3]"
    os.environ["K2_VLLM_HOST_COMPAT"] = "1"
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    adapter = None
    try:
        adapter = K2HorizonForCausalLM.initialize_vllm_model(None, mesh, 4, max_seq_len=1024)
        cache = adapter.allocate_kv_cache((68, 2, 64, 128), torch.bfloat16, 48)
        adapter.warmup_model_prefill(cache, False)
        adapter.warmup_model_decode(cache, True)
        g = adapter.generator
        p1 = check_intermediate_prefill(adapter, cache)
        # Same populated staging buffer and real selected-policy weights: only
        # eager versus captured terminal dispatch differs.
        g._prefill_head()
        eager_head = g.read_logits(g.logits).clone()
        g._replay_prefill_head()
        traced_head = g.read_logits(g.logits).clone()
        assert torch.equal(eager_head, traced_head), "Traced prefill terminal changed logits"
        head_trace = g.prefill_head_trace
        lens = [63, 127, 129, 131]
        tokens = torch.stack([torch.arange(10 + 100 * s, 10 + 100 * s + 131) for s in range(4)])
        params = TTSamplingParams(
            temperature=[0.0] * 4,
            top_k=[0] * 4,
            top_p=[1.0] * 4,
            seed=[1, 2, 3, 4],
            enable_log_probs=[False] * 4,
            num_logprobs=[-1] * 4,
        )
        runs = []
        for resident in (False, True):
            table = torch.zeros(4, 16, dtype=torch.int32)
            maps = torch.arange(1, 65, dtype=torch.int32).reshape(4, 16).flip(1)
            allocated = [0] * 4

            def grow(positions):
                changed = False
                for s, p in enumerate(positions):
                    needed = (int(p) + 1 + 63) // 64
                    if needed > allocated[s]:
                        table[s, allocated[s] : needed] = maps[s, allocated[s] : needed]
                        allocated[s] = needed
                        changed = True
                return changed

            grow([n - 1 for n in lens])
            first = adapter.prefill_forward(
                tokens,
                page_table=table,
                kv_cache=cache,
                prompt_lens=lens,
                sampling_params=params,
                empty_slots=[0, 1, 2, 3],
                enable_trace=True,
            )
            positions = torch.tensor(lens, dtype=torch.int32)
            current = first
            output = [first.reshape(-1).tolist()]
            base = g.counters.copy()
            growth = []
            pending = []
            for step in range(160):
                changed = grow(positions)
                if changed:
                    growth.append({"step": step, "positions": positions.tolist(), "blocks": allocated.copy()})
                reload_inputs = not resident or step == 0
                result = adapter.decode_forward(
                    current if reload_inputs else torch.full_like(current, 999999),
                    positions if reload_inputs else torch.full_like(positions, -999),
                    page_table=table,
                    kv_cache=cache,
                    sampling_params=params,
                    enable_trace=True,
                    read_from_device=False,
                    reload_inputs=reload_inputs,
                    reload_page_table=changed and not reload_inputs,
                    reload_sampling_params=step == 0,
                    reset_sampling_state=step == 0,
                )
                assert ttnn.is_tensor_storage_on_device(result.values)
                host, events = adapter.read_decode_output(result, async_read=True)
                assert len(ttnn.get_device_tensors(host.values)) == 1
                pending.append((host, events))
                # Queue the next resident submission before consuming this result.
                if not resident or len(pending) > 1:
                    old, ev = pending.pop(0)
                    for e in ev:
                        ttnn.event_synchronize(e)
                    current = adapter.process_decode_output_host(old, is_tokens=True)
                    output.append(current.reshape(-1).tolist())
                positions += 1
            for old, ev in pending:
                for e in ev:
                    ttnn.event_synchronize(e)
                output.append(adapter.process_decode_output_host(old, is_tokens=True).reshape(-1).tolist())
            actual = ttnn.to_torch(ttnn.get_device_tensors(g.positions)[0]).reshape(-1)
            assert actual.tolist() == positions.tolist(), (actual, positions)
            delta = dict(g.counters - base)
            assert delta["async_reads"] == 160
            assert delta["device_output_submissions"] == 160
            assert delta["token_readback_replicas"] == delta["token_readbacks"] == 160
            assert g.prefill_head_trace == head_trace
            if resident:
                assert delta["token_refreshes"] == 1 and delta["position_refreshes"] == 1
                assert delta["page_table_refreshes"] == 1 + len([x for x in growth if x["step"] != 0])
            runs.append(
                dict(
                    resident=resident,
                    outputs=output,
                    growth=growth,
                    counter_delta=delta,
                    final_positions=actual.tolist(),
                )
            )
        assert runs[0]["outputs"] == runs[1]["outputs"], "Resident token feedback differs from authoritative control"
        # Explicitly refreshed tokens/positions change the actual trace inputs and logits.
        logits = []
        for value, pos in [(35, 333), (75, 335)]:
            out = adapter.decode_forward(
                torch.full((4, 1), value),
                torch.full((4,), pos),
                page_table=maps,
                kv_cache=cache,
                enable_trace=True,
                read_from_device=False,
            )
            host, events = adapter.read_decode_output(out, async_read=True)
            for e in events:
                ttnn.event_synchronize(e)
            logits.append(adapter.process_decode_output_host(host, is_tokens=False))
            assert ttnn.to_torch(ttnn.get_device_tensors(g.tokens)[0]).reshape(-1)[:4].tolist() == [value] * 4
        assert not torch.equal(*logits)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                dict(
                    status="pass",
                    intermediate_prefill=p1,
                    layers=[0, 3],
                    cache_owner="vllm",
                    prompt_lengths=lens,
                    decode_steps=160,
                    tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING"),
                    stale_token_position_ignored=True,
                    refreshed_inputs_change_logits=True,
                    terminal_eager_trace_bitwise=True,
                    prefill_head_captures=g.counters["prefill_head_captures"],
                    token_readback_replicas_per_submission=1,
                    async_device_output_verified=True,
                    runs=runs,
                ),
                indent=2,
            )
        )
        print("ADAPTER_CONTRACT_PASS")
    finally:
        if adapter:
            adapter.release_persistent_capture()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
