# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reduced final-prefill sampling, fixed slots, and absolute seed regression."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.generator import QwenGenerator
from models.autoports.qwen_qwen3_6_27b.tt.model import Model
from models.common.sampling.generator import _hash_request_seed_to_device_seed


def first(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).flatten().clone()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    generator = None
    try:
        model = Model(mesh, layer_indices=[0, 3])
        page_table = torch.arange(128, dtype=torch.int32).reshape(4, 32)
        cache = model.create_cache(page_table.flip(0))
        from models.autoports.qwen_qwen3_6_27b.tt.page_table_ownership import page_table_for_attention

        table_tensor = page_table_for_attention(cache[1])
        address = table_tensor.buffer_address()
        bad_table = page_table.clone()
        bad_table[0, 0] = 128
        try:
            QwenGenerator(model, batch_size=4, max_seq_len=1024, kv_cache=cache, page_table=bad_table)
        except ValueError as error:
            assert "Initial page address" in str(error)
        else:
            raise AssertionError("Accepted initial out-of-bounds external page table")
        generator = QwenGenerator(model, batch_size=4, max_seq_len=1024, kv_cache=cache, page_table=page_table)
        assert torch.equal(first(table_tensor).reshape(4, 32), page_table), "Initial external page table stayed stale"
        assert table_tensor.buffer_address() == address, "Initial binding replaced external table storage"
        # Restricted preparation is deliberate: this isolates the new contract,
        # not the full public preparation acceptance battery.
        for engine in (generator.prefill_single, generator.prefill_uniform):
            engine.prepare(_diagnostic_lengths=[1, 33, 512])
        generator.prepare()
        generator.configure_sampling(temperature=[0.7, 0.8, 0.9, 1.0], top_k=[2, 3, 4, 5], seed=[10, 11, 12, 13])
        generator._write(torch.arange(32, dtype=torch.int32).reshape(1, 1, 1, 32), generator.tokens)
        sampler = generator.sampling.tt_sampling
        controls = (
            generator.tokens,
            sampler.seeds_tt_tensor,
            sampler.k_tensor,
            sampler.p_tensor,
            sampler.temp_tensor,
            sampler._greedy_col,
        )
        before = [first(tensor) for tensor in controls]
        original_mask = first(generator.sample_active_rows)
        slots, ends = [3, 1], [33, 513]
        tokens = torch.arange(70000, 70000 + 2 * 514, dtype=torch.int32).reshape(2, 514)
        table = generator.page_host[slots].clone()
        call = dict(page_table=table, kv_cache=generator.cache, prompt_lens=ends, empty_slots=slots)
        # Explicit diagnostic host control; the measured sampled call must not
        # read logits back. The second fresh prefill resets the same GDN slots.
        expected = generator.prefill_forward(tokens, **call).argmax(dim=-1)
        params = SimpleNamespace(
            temperature=[0.0, 0.0],
            top_k=[-1, -1],
            top_p=[1.0, 1.0],
            seed=[0, -1],
            enable_log_probs=[False, False],
            num_logprobs=[-1, -1],
        )
        prior_reads = generator.counters["full_logits_readbacks"]
        prior_samples = generator.counters["sample_replays"]
        output = generator.prefill_forward(tokens, sampling_params=params, enable_trace=False, **call)
        assert output.dtype == torch.int64 and tuple(output.shape) == (2, 1)
        assert torch.equal(output, expected), "Final sampled tokens differ from greedy logits in request order"
        assert generator.counters["full_logits_readbacks"] == prior_reads, "Sampled prefill read full logits"
        assert generator.counters["sample_replays"] == prior_samples + 1, "Intermediate chunks advanced sampling"
        after = [first(tensor) for tensor in controls]
        assert all(torch.equal(a[[0, 2]], b[[0, 2]]) for a, b in zip(before, after)), "Omitted sampler slots changed"
        assert torch.equal(original_mask, first(generator.sample_active_rows)), "Prefill changed decode active mask"
        assert generator.request_seeds == [10, -1, 12, 0]
        for row, slot in enumerate(slots):
            assert int(after[0][slot]) == int(output[row, 0])
            assert int(after[1][slot]) == _hash_request_seed_to_device_seed(params.seed[row], 0) + ends[row]

        # A non-final scheduler chunk has no sampling_params and must not touch
        # token feedback, seeds, or per-slot sampling parameters.
        before_chunk = [first(tensor) for tensor in controls]
        generator.prefill_forward(
            tokens[1:2],
            page_table=table[1:2],
            kv_cache=generator.cache,
            prompt_lens=[514],
            start_pos=[513],
            empty_slots=[1],
        )
        assert all(torch.equal(a, first(b)) for a, b in zip(before_chunk, controls))
        assert generator.counters["sample_replays"] == prior_samples + 1
        # A different request's final suffix samples from its absolute input
        # position, independent of whether any token history was supplied.
        resumed = SimpleNamespace(temperature=0.0, top_k=-1, top_p=1.0, seed=0)
        generator.prefill_forward(
            tokens[:1],
            page_table=table[:1],
            kv_cache=generator.cache,
            prompt_lens=[34],
            start_pos=[33],
            empty_slots=[3],
            sampling_params=resumed,
        )
        assert int(first(sampler.seeds_tt_tensor)[3]) == _hash_request_seed_to_device_seed(0, 0) + 34

        # Unsupported device features reject before a request state mutation.
        for field, value in (
            ("presence_penalty", 0.5),
            ("frequency_penalty", 0.5),
            ("repetition_penalty", 1.5),
            ("enable_log_probs", True),
        ):
            rejected = SimpleNamespace(**vars(params))
            setattr(rejected, field, value)
            before_reject = [first(tensor) for tensor in controls]
            try:
                generator.prefill_forward(tokens, sampling_params=rejected, **call)
            except ValueError as error:
                assert "host-logits" in str(error)
            else:
                raise AssertionError(f"Accepted unsupported {field}")
            assert all(torch.equal(a, first(b)) for a, b in zip(before_reject, controls))

        # Replay after every new operation checks stable captured allocations
        # when the parent invokes this probe with trace allocation tracking.
        reset_seeds = [(1 << 63) - 1, -(1 << 63), 0, -1]
        reset_params = SimpleNamespace(
            temperature=[0.7] * 4,
            top_k=[20] * 4,
            top_p=[0.9] * 4,
            seed=reset_seeds,
            enable_log_probs=False,
            num_logprobs=-1,
        )
        reset_positions = torch.tensor([-1, 514, -1, 34])
        generator.decode_forward(
            torch.tensor([[1], [int(output[1, 0])], [2], [int(output[0, 0])]]),
            reset_positions,
            page_table=generator.page_host,
            kv_cache=generator.cache,
            reload_sampling_params=True,
            reset_sampling_state=True,
            sampling_params=reset_params,
        )
        reset_actual = first(sampler.seeds_tt_tensor)
        for slot in (1, 3):
            assert (
                int(reset_actual[slot])
                == _hash_request_seed_to_device_seed(reset_seeds[slot], 0) + int(reset_positions[slot]) + 1
            )
        before_resident = dict(generator.counters)
        # All four host inputs are intentionally stale; resident command owns
        # authority and advances the same device RNG stream without history.
        generator.decode_forward(
            torch.zeros(4, 1, dtype=torch.int32),
            torch.zeros(4, dtype=torch.int32),
            page_table=page_table.flip(0),
            kv_cache=generator.cache,
            reload_inputs=False,
        )
        resident_actual = first(sampler.seeds_tt_tensor)
        for slot in (1, 3):
            assert int(resident_actual[slot]) == int(reset_actual[slot]) + 1
        for name in ("token_refreshes", "position_refreshes", "rope_refreshes", "page_table_refreshes"):
            assert generator.counters[name] == before_resident.get(name, 0)

        # The random stream belongs to the request, including seed 0/-1; neither
        # submitted row order nor physical destination slot may select a different
        # stream. Use unequal lengths/parameters and actual final prefill logits.
        stochastic_ends = [33, 1]
        stochastic_fields = dict(temperature=[1.7, 2.3], top_k=[32, 17], top_p=[1.0, 1.0], seed=[0, -1])
        stochastic_cases = (
            ("baseline", [0, 1], [3, 1]),
            ("repeat", [0, 1], [3, 1]),
            ("request_permutation", [1, 0], [1, 3]),
            ("slot_permutation", [0, 1], [1, 3]),
        )
        stochastic_outputs = {}
        for name, order, destinations in stochastic_cases:
            request_params = SimpleNamespace(
                **{field: [values[index] for index in order] for field, values in stochastic_fields.items()},
                enable_log_probs=False,
                num_logprobs=-1,
            )
            omitted_before = [first(tensor)[[0, 2]] for tensor in controls]
            mask_before = first(generator.sample_active_rows)
            reads_before = generator.counters["full_logits_readbacks"]
            samples_before = generator.counters["sample_replays"]
            sampled = generator.prefill_forward(
                tokens[order, :33],
                page_table=page_table[destinations],
                kv_cache=generator.cache,
                prompt_lens=[stochastic_ends[index] for index in order],
                empty_slots=destinations,
                sampling_params=request_params,
            )
            assert tuple(sampled.shape) == (2, 1) and sampled.dtype == torch.int64
            assert generator.counters["full_logits_readbacks"] == reads_before
            assert generator.counters["sample_replays"] == samples_before + 1
            assert torch.equal(mask_before, first(generator.sample_active_rows))
            assert all(torch.equal(old, first(tensor)[[0, 2]]) for old, tensor in zip(omitted_before, controls))
            observed_seeds, observed_tokens = first(sampler.seeds_tt_tensor), first(generator.tokens)
            for row, (request, slot) in enumerate(zip(order, destinations)):
                assert int(observed_tokens[slot]) == int(sampled[row, 0])
                assert int(observed_seeds[slot]) == (
                    _hash_request_seed_to_device_seed(stochastic_fields["seed"][request], 0) + stochastic_ends[request]
                )
            # Compare in request identity order, not submission order.
            inverse_order = [order.index(request) for request in range(2)]
            canonical = sampled[inverse_order]
            stochastic_outputs[name] = canonical.tolist()
            if name == "baseline":
                stochastic_baseline = canonical.clone()
            else:
                assert torch.equal(canonical, stochastic_baseline), f"Stochastic request output changed for {name}"

        # End with authoritative then resident replay after mask restoration.
        # This exercises trace allocation tracking and device token feedback;
        # no full-logit readback or host-sampled token is used as feedback.
        generator.decode_forward(
            torch.tensor([[0], [int(sampled[0, 0])], [0], [int(sampled[1, 0])]]),
            torch.tensor([-1, 33, -1, 1]),
            page_table=generator.page_host,
            kv_cache=generator.cache,
        )
        final_reads = generator.counters["full_logits_readbacks"]
        generator.decode_forward(
            torch.zeros(4, 1, dtype=torch.int32),
            torch.zeros(4, dtype=torch.int32),
            page_table=page_table.flip(0),
            kv_cache=generator.cache,
            reload_inputs=False,
        )
        assert generator.counters["full_logits_readbacks"] == final_reads
        result = dict(
            passed=True,
            output=output.tolist(),
            seed_reset_values=reset_seeds,
            seeded_reset_without_history=True,
            resident_stale_inputs_ignored=True,
            stochastic_outputs=stochastic_outputs,
            stochastic_repeat_and_slot_permutation=True,
            counters=dict(generator.counters),
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        if generator is not None:
            generator.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
