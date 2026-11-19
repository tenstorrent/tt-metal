# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced real-model seeded resume, mode transitions, and common penalties.

Parent runs this serialized device probe. The frozen-logit oracle reproduces
BF16 penalty arithmetic only; every device sample uses the common split trace.
All cache/position/logit snapshot allocations precede the first trace capture.
"""


def main():
    import argparse
    import hashlib
    import json
    from dataclasses import replace
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import torch

    import ttnn
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator
    from models.common.sampling.generator import SamplingParams

    torch.set_num_threads(8)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"layers": [0, 3], "batch": 3, "context": 512, "cases": []}
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    gen = None
    try:
        gen = build_generator(None, mesh, layer_indices=[0, 3], max_batch_size=3, max_seq_len=512)
        params = SamplingParams(
            temperature=[0.8] * 3,
            top_k=32,
            top_p=0.9,
            seed=[42] * 3,
            presence_penalty=0.5,
            frequency_penalty=0.25,
            repetition_penalty=1.125,
        )
        lengths = [33, 35, 37]
        histories = [list(range(100 + 100 * slot, 100 + 100 * slot + length)) for slot, length in enumerate(lengths)]
        prompts = torch.zeros(3, max(lengths), dtype=torch.int64)
        for slot, history in enumerate(histories):
            prompts[slot, : len(history)] = torch.tensor(history)

        def record(case, passed, **details):
            item = {"case": case, "passed": bool(passed), **details}
            report["cases"].append(item)
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(item), flush=True)
            if not passed:
                raise AssertionError(case)

        def read(tensor):
            return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).clone()

        def seeds():
            return read(gen.sampler.tt_sampling.seeds_tt_tensor).reshape(-1).long()

        def sampler_rows(slots):
            result = {}
            for name in ("prompt_mask", "output_mask", "output_counts", "output_counts_gathered"):
                for rank, shard in enumerate(ttnn.get_device_tensors(getattr(gen.sampler.tt_penalties, name))):
                    value = ttnn.to_torch(shard).reshape(32, -1)[slots].float()
                    result[f"{name}.{rank}"] = hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
            result["seeds"] = seeds()[slots].tolist()
            return result

        state = gen.prepare_decode(kv_cache=gen.cache, page_table=gen.page_table, slots=(0, 1, 2))
        unsampled = gen.prefill_contract.forward(
            prompts, page_table=gen.page_table, kv_cache=gen.cache, prompt_lens=lengths, sample_on_device=True
        )
        frozen = ttnn.clone(unsampled.logits)
        frozen_host = gen.model.host_logits(frozen).reshape(32, gen.model.vocab_size)[:3].float()
        # Compile copying this fixed real head output into the matching decode
        # sampling input geometry before any trace capture.
        ttnn.copy(frozen, state.logits)
        first = gen.prefill_forward(
            prompts, page_table=gen.page_table, kv_cache=gen.cache, prompt_lens=lengths, sampling_params=params
        )
        prefill_seeds = seeds()
        cache_buffers, seen = [], set()
        for layer_states in gen.cache:
            for st in layer_states:
                for name in ("conv", "recurrent", "key", "value"):
                    tensor = getattr(st, name)
                    if tensor is not None and id(tensor) not in seen:
                        cache_buffers.append(tensor)
                        seen.add(id(tensor))
        prefill_cache = [ttnn.clone(tensor) for tensor in cache_buffers]
        resume_cache = [ttnn.zeros_like(tensor) for tensor in cache_buffers]
        resume_positions = [ttnn.zeros_like(pos) for pos in state.positions]
        # Warm snapshot copies before capture, preserving original state.
        for tensor, snapshot in zip(cache_buffers, resume_cache, strict=True):
            ttnn.copy(tensor, snapshot)
            ttnn.copy(snapshot, tensor)
        for pos, snapshot in zip(state.positions, resume_positions, strict=True):
            ttnn.copy(pos, snapshot)
            ttnn.copy(snapshot, pos)
        gen.capture(state)
        for tensor, snapshot in zip(cache_buffers, prefill_cache, strict=True):
            ttnn.copy(snapshot, tensor)
        # Capture is not admission; the first public decode explicitly restores
        # complete prompt/output histories and the matching seedstream position.
        baseline = mesh.num_program_cache_entries()
        captures = gen.counters["captures"]
        traces = (dict(state.model_traces), dict(state.sample_traces))
        mesh.set_program_cache_misses_allowed(False)

        def decode(tokens=None, positions=None, *, reset=False, output_history=None, scoped=None, new_params=None):
            reload = tokens is not None
            return gen.decode_forward(
                tokens,
                positions,
                page_table=gen.page_table if reload else None,
                kv_cache=gen.cache,
                sampling_params=params if new_params is None else new_params,
                reload_inputs=reload,
                reload_sampling_params=reset or new_params is not None,
                reset_sampling_state=reset,
                sampling_state_slots=scoped,
                prompt_tokens=histories if reset else None,
                output_tokens=output_history,
            )

        initial_outputs = [[int(value)] for value in first[:, 0]]
        second = decode(first, lengths, reset=True, output_history=initial_outputs)
        record("prefill_sample_in_seed_history", torch.equal(seeds()[:3], prefill_seeds[:3] + 1))
        for tensor, snapshot in zip(cache_buffers, resume_cache, strict=True):
            ttnn.copy(tensor, snapshot)
        for pos, snapshot in zip(state.positions, resume_positions, strict=True):
            ttnn.copy(pos, snapshot)
        expected = [decode(), decode()]
        expected_seeds = seeds()
        expected_history = sampler_rows([0, 1, 2])
        for tensor, snapshot in zip(cache_buffers, resume_cache, strict=True):
            ttnn.copy(snapshot, tensor)
        for pos, snapshot in zip(state.positions, resume_positions, strict=True):
            ttnn.copy(snapshot, pos)
        outputs_so_far = [[int(first[slot, 0]), int(second[slot, 0])] for slot in range(3)]
        resumed = [
            decode(second, [length + 1 for length in lengths], reset=True, output_history=outputs_so_far),
            decode(),
        ]
        record(
            "seeded_history_resume_matches_uninterrupted",
            all(torch.equal(a, b) for a, b in zip(expected, resumed, strict=True))
            and torch.equal(seeds()[:3], expected_seeds[:3])
            and sampler_rows([0, 1, 2]) == expected_history,
            uninterrupted=[token.reshape(-1).tolist() for token in expected],
            resumed=[token.reshape(-1).tolist() for token in resumed],
        )
        for mode, controls in (
            ("greedy", replace(params, temperature=[0.0] * 3, top_k=1, top_p=1.0)),
            ("sampled", params),
        ):
            before_seeds = seeds()
            counters = dict(gen.counters)
            decode(new_params=controls)
            record(
                "mode_reload_preserves_seedstream",
                torch.equal(seeds()[:3], before_seeds[:3] + 1)
                and torch.equal(seeds()[3:], before_seeds[3:])
                and gen.counters["token_refreshes"] == counters["token_refreshes"]
                and gen.counters["position_refreshes"] == counters["position_refreshes"]
                and gen.counters["sampling_replays"] == counters["sampling_replays"] + 1,
                mode=mode,
            )
        peers = sampler_rows([1, 2])
        decode(
            second,
            [lengths[0] + 1, -1, -1],
            reset=True,
            scoped=[0],
            output_history=[outputs_so_far[0]],
        )
        record("scoped_history_reset_preserves_inactive_peers", sampler_rows([1, 2]) == peers)

        # Discriminating penalty oracle: suppress the eight leading real logits
        # with prompt repetition and repeated output counts. Dyadic parameters
        # make each BF16 rounding boundary explicit and reproducible.
        penalty_params = replace(
            params,
            temperature=[0.0] * 3,
            top_k=1,
            top_p=1.0,
            presence_penalty=64.0,
            frequency_penalty=4.0,
            repetition_penalty=2.0,
        )
        top = frozen_host.topk(8, dim=-1).indices.tolist()
        penalty_prompts = [row[:4] for row in top]
        penalty_outputs = [[token for rank, token in enumerate(row) for _ in range(rank + 1)] for row in top]
        adjusted = frozen_host.to(torch.bfloat16).float()
        for slot in range(3):
            counts = torch.bincount(torch.tensor(penalty_outputs[slot]), minlength=gen.model.vocab_size)
            combined = counts > 0
            combined[penalty_prompts[slot]] = True
            scale = torch.ones(gen.model.vocab_size)
            scale[combined & (adjusted[slot] > 0)] = 0.5
            scale[combined & (adjusted[slot] <= 0)] = 2.0
            adjusted[slot] = (adjusted[slot] * scale).bfloat16().float()
            adjusted[slot] = (adjusted[slot] - (counts > 0).float() * 64.0).bfloat16().float()
            adjusted[slot] = (adjusted[slot] - (counts.bfloat16().float() * 4.0).bfloat16().float()).bfloat16().float()
        wanted = adjusted.argmax(dim=-1)
        assert bool((wanted != frozen_host.argmax(dim=-1)).all()), "Penalty stimulus must change every greedy winner"
        gen.configure_sampling(penalty_params)
        gen._reset_sampling_state(slots=[0, 1, 2], prompt_tokens=penalty_prompts, output_tokens=penalty_outputs)
        gen.sampling_state.refresh_active_slots([0, 1, 2])
        ttnn.copy(frozen, state.trace_logits[False])
        original_host_logits = gen.model.host_logits

        def forbid_logits(*_args, **_kwargs):
            raise AssertionError("Traced common sampler crossed a host-logits boundary")

        gen.model.host_logits = forbid_logits
        try:
            ttnn.execute_trace(mesh, state.sample_traces[(False, True)], cq_id=0, blocking=True)
        finally:
            gen.model.host_logits = original_host_logits
        actual = read(state.tokens).reshape(-1)[:3].long()
        record(
            "frozen_real_logits_common_penalties_match_host_greedy",
            torch.equal(actual, wanted),
            actual=actual.tolist(),
            expected=wanted.tolist(),
            unadjusted=frozen_host.argmax(dim=-1).tolist(),
        )
        # Fresh seed=None admissions must acquire independent device streams.
        # Frozen real logits isolate RNG behavior from token feedback/cache state.
        unseeded = replace(
            params,
            temperature=[1.0] * 3,
            seed=[None] * 3,
            top_k=32,
            top_p=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty=1.0,
        )
        unseeded_tokens, unseeded_streams = [], []
        for _ in range(8):
            gen.configure_sampling(unseeded)
            gen._reset_sampling_state(slots=[0, 1, 2], prompt_tokens=histories)
            unseeded_streams.append(seeds()[:3].tolist())
            gen.sampling_state.refresh_active_slots([0, 1, 2])
            ttnn.copy(frozen, state.trace_logits[False])
            ttnn.execute_trace(mesh, state.sample_traces[(False, True)], cq_id=0, blocking=True)
            unseeded_tokens.append(read(state.tokens).reshape(-1)[:3].long().tolist())
        record(
            "unseeded_frozen_real_logits_first_token_variety",
            len({tuple(row) for row in unseeded_streams}) == 8 and len({tuple(row) for row in unseeded_tokens}) > 1,
            streams=unseeded_streams,
            tokens=unseeded_tokens,
        )
        record(
            "transition_program_and_trace_reuse",
            mesh.num_program_cache_entries() == baseline
            and gen.counters["captures"] == captures
            and state.model_traces == traces[0]
            and state.sample_traces == traces[1],
            prepared_programs=baseline,
            final_programs=mesh.num_program_cache_entries(),
        )
        report["passed"] = True
    except BaseException as error:
        report["passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")
        mesh.set_program_cache_misses_allowed(True)
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
