# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced adapter replay regression: original prompt boundaries and sampler state.

Device execution is parent-owned and serialized. Capture happens before requests.
The host-final case deliberately feeds the uninterrupted run's next token, so it
checks resumed state reconstruction without claiming host/TT RNG equivalence.
"""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.vllm_adapter_decode_probe import build_adapter
from models.common.sampling.generator import SamplingParams, _hash_request_seed_to_device_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "layers": [0, 3], "batch": 3, "context": 256, "cases": []}

    def record(name, passed, **details):
        report["cases"].append({"name": name, "passed": bool(passed), **details})
        output.write_text(json.dumps(report, indent=2) + "\n")
        if not passed:
            raise AssertionError(name)

    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    gen = None
    try:
        gen = build_adapter(None, mesh, max_batch_size=3, max_seq_len=256)
        state = gen.prepare_decode(kv_cache=gen.cache, page_table=gen.page_table, slots=(0, 1, 2))
        gen.capture(state)
        captures = gen.counters["captures"]
        lengths = [33, 35, 37]
        prompts = [list(range(100 + 100 * row, 100 + 100 * row + length)) for row, length in enumerate(lengths)]
        request_seeds = [42, 43, 44]
        bases = [(_hash_request_seed_to_device_seed(seed, 0) % 500_000) + 1 for seed in request_seeds]
        params = SamplingParams(
            temperature=[0.8] * 3,
            top_k=32,
            top_p=0.9,
            seed=request_seeds,
            presence_penalty=0.5,
            frequency_penalty=0.25,
            repetition_penalty=1.125,
        )

        def packed(rows):
            result = torch.zeros((len(rows), max(map(len, rows))), dtype=torch.int64)
            for row, values in enumerate(rows):
                result[row, : len(values)] = torch.tensor(values)
            return result

        def prefill(rows, originals, *, starts=None, ends=None, device=True, slots=(0, 1, 2)):
            with patch.object(gen.sampling_state, "reset_history", wraps=gen.sampling_state.reset_history) as reset:
                result = gen.prefill_forward(
                    packed(rows),
                    page_table=gen.page_table[list(slots)],
                    kv_cache=gen.cache,
                    prompt_lens=list(map(len, rows)) if ends is None else list(ends),
                    original_prompt_lens=list(originals),
                    start_pos=starts,
                    empty_slots=list(slots),
                    # The adapter derives the canonical request horizon from the
                    # authoritative original boundary, including unknown replay.
                    sampling_params=params if device else None,
                )
            assert reset.call_count == int(device), ("history reset count", device, reset.call_count)
            return result[0] if isinstance(result, tuple) else result

        def decode(tokens, positions, histories=None, *, reset=True):
            handle = gen.decode_forward(
                tokens,
                positions,
                page_table=gen.page_table if reset else None,
                kv_cache=gen.cache,
                sampling_params=params,
                reload_inputs=reset,
                reload_sampling_params=reset,
                reset_sampling_state=reset,
                read_from_device=False,
                prompt_tokens=prompts if reset else None,
                output_tokens=histories if reset else None,
            )
            read, events = gen.read_decode_output(handle, async_read=True)
            for event in events:
                ttnn.event_synchronize(event)
            return gen.process_decode_output_host(read, is_tokens=True)

        def verify_state(name, prompt_rows, output_rows, *, slots=(0, 1, 2)):
            # Every expected element comes from authoritative token histories,
            # independently of the generator's boundary/offset metadata.
            expected_counts = torch.stack(
                [
                    torch.bincount(torch.tensor(row, dtype=torch.int64), minlength=gen.sampler.tt_penalties.vocab_size)
                    for row in output_rows
                ]
            )
            expected_prompt = torch.stack(
                [
                    torch.bincount(torch.tensor(row, dtype=torch.int64), minlength=gen.sampler.tt_penalties.vocab_size)
                    > 0
                    for row in prompt_rows
                ]
            ).long()
            expected = {
                "prompt_mask": expected_prompt,
                "output_mask": (expected_counts > 0).long(),
                "output_counts": expected_counts,
                "output_counts_gathered": expected_counts,
            }
            failures = []
            for field, wanted in expected.items():
                tensor = getattr(gen.sampler.tt_penalties, field)
                for rank, shard in enumerate(ttnn.get_device_tensors(tensor)):
                    actual = ttnn.to_torch(shard).reshape(32, -1)[list(slots)].long()
                    # TP4 sampling_dp=1: gathered counts are replicated [32,V]
                    # on every chip; the other three tables shard vocab [32,V/4].
                    # Both count tables accumulate outputs; masks are counts>0.
                    target = wanted if field == "output_counts_gathered" else wanted.chunk(4, dim=1)[rank]
                    if actual.shape != target.shape:
                        failures.append(
                            {
                                "field": field,
                                "rank": rank,
                                "actual_shape": list(actual.shape),
                                "expected_shape": list(target.shape),
                            }
                        )
                    elif not torch.equal(actual, target):
                        failures.append(
                            {"field": field, "rank": rank, "different_elements": int((actual != target).sum())}
                        )
            seed_tensor = gen.sampler.tt_sampling.seeds_tt_tensor
            wanted_seeds = [base + len(row) for base, row in zip(bases, output_rows)]
            actual_seeds = []
            for shard in ttnn.get_device_tensors(seed_tensor):
                observed = ttnn.to_torch(shard).reshape(-1)[list(slots)].long().tolist()
                actual_seeds.append(observed)
                if observed != wanted_seeds:
                    failures.append({"field": "seed", "actual": observed, "expected": wanted_seeds})
            record(
                name,
                not failures,
                seeds=actual_seeds,
                expected_seeds=wanted_seeds,
                output_counts=[len(row) for row in output_rows],
                mismatches=failures,
            )

        first = prefill(prompts, lengths)
        generated = [[int(first[row, 0])] for row in range(3)]
        verify_state("uninterrupted_prefill", prompts, generated)
        token = first
        for step in range(3):
            token = decode(
                token if step == 0 else None,
                lengths if step == 0 else None,
                generated if step == 0 else None,
                reset=step == 0,
            )
            for row in range(3):
                generated[row].append(int(token[row, 0]))
        verify_state("uninterrupted_four_outputs", prompts, generated)
        record("uninterrupted_outputs_recorded", True, tokens=generated)

        # Replay only the first two generated outputs. Its next draw must be
        # the third output, with two existing output counts rather than zero.
        replay_rows = [prompt + tokens[:2] for prompt, tokens in zip(prompts, generated)]
        replay_ends = list(map(len, replay_rows))
        expected_third = torch.tensor([[tokens[2]] for tokens in generated])
        expected_fourth = torch.tensor([[tokens[3]] for tokens in generated])
        third = prefill(replay_rows, lengths)
        verify_state(
            "resumed_prefill_seed_and_histories",
            prompts,
            [tokens[:2] + [int(third[row, 0])] for row, tokens in enumerate(generated)],
        )
        record(
            "resumed_third_matches_uninterrupted",
            torch.equal(third, expected_third),
            actual=third.tolist(),
            expected=expected_third.tolist(),
        )
        fourth = decode(third, replay_ends, [tokens[:3] for tokens in generated])
        verify_state(
            "resumed_decode_seed_and_histories",
            prompts,
            [tokens[:3] + [int(fourth[row, 0])] for row, tokens in enumerate(generated)],
        )
        record(
            "resumed_fourth_matches_uninterrupted",
            torch.equal(fourth, expected_fourth),
            actual=fourth.tolist(),
            expected=expected_fourth.tolist(),
        )

        # Original boundary exceeds the first chunk and lies inside the second.
        # Intermediate chunks intentionally use the host path and no RNG draw.
        prefill(replay_rows, lengths, ends=[32, 32, 32], device=False)
        crossing = [length + 1 for length in lengths]
        prefill(replay_rows, lengths, starts=[32, 32, 32], ends=crossing, device=False)
        third = prefill(replay_rows, lengths, starts=crossing)
        verify_state(
            "chunked_replay_seed_and_histories",
            prompts,
            [tokens[:2] + [int(third[row, 0])] for row, tokens in enumerate(generated)],
        )
        record(
            "chunked_replay_matches_uninterrupted",
            torch.equal(third, expected_third),
            actual=third.tolist(),
            expected=expected_third.tolist(),
        )

        # Host-final prefill supplies full logits. Feed the baseline host result
        # deliberately: this is a state test, not a host sampler parity claim.
        logits = prefill(replay_rows, lengths, device=False)
        record(
            "host_final_logits_geometry", tuple(logits.shape) == (3, 1, gen.model.vocab_size), actual=list(logits.shape)
        )
        fourth = decode(expected_third, replay_ends, [tokens[:3] for tokens in generated])
        verify_state(
            "host_final_to_device_seed_and_histories",
            prompts,
            [tokens[:3] + [int(fourth[row, 0])] for row, tokens in enumerate(generated)],
        )
        record(
            "host_final_to_device_matches_uninterrupted",
            torch.equal(fourth, expected_fourth),
            actual=fourth.tolist(),
            expected=expected_fourth.tolist(),
        )

        # Identical token prefixes, different original boundary: these are NEW
        # requests, with zero prior outputs. Reuse slots in a different row order.
        slots = (2, 0, 1)
        fresh = prefill(replay_rows, replay_ends, slots=slots)
        verify_state(
            "fresh_prefix_collision_and_slot_reuse",
            replay_rows,
            [[int(fresh[row, 0])] for row in range(3)],
            slots=slots,
        )
        record("no_recapture", gen.counters["captures"] == captures, before=captures, after=gen.counters["captures"])
        report["counters"] = dict(gen.counters)
        report["status"] = "pass"
        output.write_text(json.dumps(report, indent=2) + "\n")
        print("ORIGINAL_BOUNDARY_PROBE_PASS", flush=True)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        output.write_text(json.dumps(report, indent=2) + "\n")
        raise
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
