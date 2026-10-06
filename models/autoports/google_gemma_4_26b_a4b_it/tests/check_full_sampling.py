# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced real-terminal sampled modes, host compatibility and runtime audit."""
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.runtime_audit import device_only
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import build_generator
from models.common.sampling.generator import SamplingParams


def read(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).clone()


def main():
    torch.set_num_threads(8)
    root = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/full_model")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    gen = None
    report = {}
    try:
        gen = build_generator(None, mesh, max_seq_len=2048, layer_indices=(0, 5), trace_debug=True)
        prompt = [2] + [100] * 32
        greedy = gen.generate(prompt, 8, stop_on_eos=False)
        with device_only():
            gen._replay()
        report["traced_replay_has_no_host_tensor_operations"] = True
        gen._release_trace()
        prompt_device = gen.model.upload(
            torch.tensor(prompt).reshape(1, 1, 1, -1).int(), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
        )
        with device_only():
            gen.model.prefill_forward(prompt_device, page_table=gen.table, kv_cache=gen.cache)
        report["prefill_has_no_host_tensor_fallback"] = True
        with device_only():
            logits = gen._forward()
            gen.sampler.precompile(logits, tt_out_tok=gen.tokens)
        report["model_and_common_sampler_have_no_host_fallback"] = True
        gen.configure_sampling(SamplingParams(temperature=0.8, top_k=16, top_p=0.9))
        unseeded_first = read(gen.sampler.tt_sampling.seeds_tt_tensor)
        gen.configure_sampling(SamplingParams(temperature=0.8, top_k=16, top_p=0.9))
        assert not torch.equal(unseeded_first, read(gen.sampler.tt_sampling.seeds_tt_tensor))
        report["omitted_seed_uses_fresh_request_entropy"] = True
        sampled = SamplingParams(temperature=0.8, top_k=16, top_p=0.9, seed=42)
        first = gen.generate(prompt, 8, sampling_params=sampled, stop_on_eos=False)
        seed = read(gen.sampler.tt_sampling.seeds_tt_tensor)
        with device_only():
            gen._replay()
        assert torch.equal(read(gen.sampler.tt_sampling.seeds_tt_tensor).long(), seed.long() + 1)
        assert torch.equal(read(gen.consumed_tokens).flatten()[:1], torch.tensor(first[-1:], dtype=torch.uint32))
        again_greedy = gen.generate(prompt, 8, stop_on_eos=False)
        second = gen.generate(prompt, 8, sampling_params=sampled, stop_on_eos=False)
        assert first == second and greedy == again_greedy
        report["alternating_greedy_seeded_sampling_repeatable"] = True
        report["sampled_seed_advance_and_token_feedback"] = True
        report["sampled_counters"] = gen.counters.copy()
        gen.host_sampling = True
        try:
            gen.configure_sampling(sampled)
        except ValueError:
            report["host_mode_rejects_unsupported_sampled_policy"] = True
        else:
            raise AssertionError("Host compatibility silently accepted sampled parameters")
        host = gen.generate(prompt, 8, stop_on_eos=False)
        assert gen.counters["full_logits_readbacks"] == 7 and gen.counters["token_refreshes"] == 7
        report["host_tokens"] = host
        report["device_tokens"] = greedy
        print("HOST_DEVICE_COMPARISON", host, greedy, flush=True)
        if host != greedy:
            gen.host_sampling = False
            first_logits = gen.prefill_logits(prompt)[0, -1]
            first_maximum = float(first_logits.max())
            assert float(first_logits[host[0]]) == first_maximum
            assert float(first_logits[greedy[0]]) == first_maximum
            report["prefill_host_device_greedy_maximum"] = first_maximum
            oracle = []

            def force_host_prefix(step, predicted):
                if step > 0:
                    values = gen._read_logits(gen.trace_logits)[..., :1, :].flatten()
                    maximum = float(values.max())
                    oracle.append(
                        dict(
                            step=step,
                            predicted=predicted,
                            host_token=host[step],
                            predicted_value=float(values[predicted]),
                            host_value=float(values[host[step]]),
                            maximum=maximum,
                            tied_maxima=int((values == maximum).sum()),
                        )
                    )
                    (root / "sampling_greedy_oracle.json").write_text(json.dumps(oracle, indent=2) + "\n")
                    assert float(values[predicted]) == maximum and float(values[host[step]]) == maximum, oracle[-1]
                return host[step]

            checked = gen.generate(prompt, 8, next_input=force_host_prefix, stop_on_eos=False)
            assert checked[0] == greedy[0]
            report["host_device_common_prefix_greedy_oracle"] = oracle
            gen.host_sampling = True
        report["explicit_host_sampling_is_semantically_greedy"] = True
        gen.host_sampling = False
        penalized = SamplingParams(
            temperature=0.8,
            top_k=16,
            top_p=0.9,
            seed=0,
            presence_penalty=0.1,
            frequency_penalty=0.2,
            repetition_penalty=1.1,
        )
        penalized_tokens = gen.generate(prompt, 4, sampling_params=penalized, stop_on_eos=False)
        counts = read(gen.sampler.tt_penalties.output_counts_gathered).reshape(32, -1)[0]
        assert int(counts.sum()) == len(penalized_tokens)
        report["traced_penalty_counts"] = True
        saved_trace = gen.trace_id
        saved_tokens = read(gen.tokens)
        saved_positions = read(gen.positions)
        for option in ({"enable_log_probs": True}, {"num_logprobs": 1}):
            unsupported = SamplingParams(temperature=0.0, top_k=1, top_p=1.0, **option)
            for action in (
                lambda: gen.configure_sampling(unsupported),
                lambda: gen.generate(prompt, 4, sampling_params=unsupported),
            ):
                try:
                    action()
                except ValueError as error:
                    assert "log-probability" in str(error)
                else:
                    raise AssertionError("Unsupported TP4 logprobs were silently accepted")
                assert gen.trace_id == saved_trace
                assert torch.equal(saved_tokens, read(gen.tokens))
                assert torch.equal(saved_positions, read(gen.positions))
        report["unsupported_tp4_logprobs_rejected_before_request_mutation"] = True
        received = []
        gen.generate(prompt, 1, next_input=lambda step, token: received.append((step, token)) or token)
        assert len(received) == 1
        report["single_token_callback"] = True
        (root / "sampling_contract.json").write_text(json.dumps(report, indent=2) + "\n")
        print(report, flush=True)
    finally:
        if gen is not None:
            gen.teardown()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
