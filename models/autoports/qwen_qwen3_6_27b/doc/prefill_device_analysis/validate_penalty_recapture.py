# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TP4, B32 recapture preserves penalty history and replay counts one real token."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.generator import build_generator
from models.common.sampling import SamplingParams, format_sampling_params

BUFFERS = ("output_mask", "output_counts", "output_counts_gathered")


def read_history(penalties):
    return {
        name: [
            ttnn.to_torch(shard).clone().to(torch.int64) for shard in ttnn.get_device_tensors(getattr(penalties, name))
        ]
        for name in BUFFERS
    }


def read_rank_vectors(tensor):
    return [ttnn.to_torch(shard).reshape(-1).to(torch.int64) for shard in ttnn.get_device_tensors(tensor)]


def empty_view_controls(generator, mesh, repetitions):
    """Measure only slot-view entry/exit; verify every touched cache tensor."""
    generator._release_traces()

    def cache_hashes():
        hashes = {}
        for index, layer in enumerate(generator.model.layers):
            names = ("batch_indices",) if layer.layer_kind == "full_attention" else ("conv", "recurrent")
            for name in names:
                hashes[f"{index}/{name}"] = [
                    hashlib.sha256(ttnn.to_torch(shard).contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                    for shard in ttnn.get_device_tensors(layer.caches[name])
                ]
        return hashes

    before = cache_hashes()
    with generator.model.single_slot_prefill_view(0):
        pass
    ttnn.synchronize_device(mesh)
    seconds = []
    for _ in range(repetitions):
        start = time.perf_counter()
        with generator.model.single_slot_prefill_view(0):
            pass
        ttnn.synchronize_device(mesh)
        seconds.append(time.perf_counter() - start)
    after = cache_hashes()
    result = {"seconds": seconds, "before_cache_hashes": before, "after_cache_hashes": after}
    result["mismatched_cache_keys"] = [key for key in before if before[key] != after[key]]
    print("EMPTY_SLOT_VIEW_CONTROL", json.dumps({"seconds": seconds, "mismatches": result["mismatched_cache_keys"]}))
    return result


def sampling_rank_controls(
    generator, mesh, steps, prompt_ids=None, save_case=None, refresh_unseeded_each_step=False, selected_cases=None
):
    """One active request: compare TP seeds/tokens for traced and eager sampling."""
    cases = {}
    for label, seed, use_trace in (
        ("unseeded_traced", None, True),
        ("unseeded_eager", None, False),
        ("seed42_traced", 42, True),
        ("seed42_eager", 42, False),
    ):
        if selected_cases is not None and label not in selected_cases:
            continue
        generator.reset()
        logical_length = len(prompt_ids) if prompt_ids is not None else 32
        physical_length = ((logical_length + 127) // 128) * 128 if prompt_ids is not None else 32
        tokens = torch.zeros((32, physical_length), dtype=torch.long)
        tokens[0, :logical_length] = torch.tensor(prompt_ids) if prompt_ids is not None else torch.arange(101, 133)
        positions = torch.full((32,), -1, dtype=torch.int32)
        positions[0] = logical_length
        params = SamplingParams(temperature=0.7, top_k=20, top_p=0.9, seed=seed)
        formatted = format_sampling_params(params, 32)
        generator.sampling.apply_prefill_state(
            sampling_params=formatted,
            prompt_tokens=tokens,
            empty_slots=[0],
            replicate_seeds=False,
        )
        ttnn.synchronize_device(mesh)
        prefill_start = time.perf_counter()
        logits = generator.prefill_forward(
            tokens,
            page_table=generator._page_table,
            kv_cache=generator.kv_cache,
            prompt_lens=[logical_length] + [0] * 31,
            read_from_device=False,
        )
        ttnn.synchronize_device(mesh)
        prefill_seconds = time.perf_counter() - prefill_start
        sample_start = time.perf_counter()
        sampler_logits = ttnn.reshape(logits, (1, 1, 32, logits.shape[-1]))
        sampled, _ = generator.sampling.sample(sampler_logits, enable_trace=False)
        ttnn.synchronize_device(mesh)
        prefill_sampling_seconds = time.perf_counter() - sample_start
        initial_tokens = read_rank_vectors(sampled)
        ttnn.deallocate(sampled)
        if sampler_logits is not logits:
            ttnn.deallocate(sampler_logits)
        ttnn.deallocate(logits)
        # Public setup replicates host inputs. Start decode ranks from the same
        # token so a prefill mismatch cannot silently contaminate this control.
        setup_start = time.perf_counter()
        generator.setup_token_out_decode(
            initial_tokens[0][:32],
            positions,
            page_table=generator.page_table_host,
            kv_cache=generator.kv_cache,
            active_mask=positions >= 0,
            sampling_params=params,
        )
        ttnn.synchronize_device(mesh)
        setup_seconds = time.perf_counter() - setup_start
        generator.sampling.seed_manager.deactivate_slots_except([0])
        generator.sampling.seed_manager.reset_seed_from_slots_if_needed(formatted.seed, [0])
        generator.sampling.seed_manager.align_seed_counters_to_positions(formatted.seed, [0], positions.tolist())
        case = {
            "request_seed": seed,
            "force_internal_sampling_trace": use_trace,
            "refresh_unseeded_each_step": refresh_unseeded_each_step and seed is None,
            "production_reseed_unseeded_each_step": generator.sampling.seed_manager.reseed_unseeded_each_step,
            "prefill_tokens_by_rank": [int(rank[0]) for rank in initial_tokens],
            "prompt_logical_length": logical_length,
            "prompt_physical_length": physical_length,
            "synchronized_prefill_seconds": prefill_seconds,
            "synchronized_prefill_sampling_seconds": prefill_sampling_seconds,
            "synchronized_setup_seconds": setup_seconds,
            "steps": [],
        }
        generated_ids = [int(initial_tokens[0][0])]
        cases[label] = case
        manager = generator.sampling.seed_manager
        original_trace_predicate = manager.has_active_request_seed
        # This test-local override changes only sample()'s trace/eager branch.
        # Seed registration, counters and per-step device uploads stay real.
        manager.has_active_request_seed = lambda: not use_trace
        try:
            for step in range(steps):
                if refresh_unseeded_each_step and seed is None:
                    # Isolated intervention: request another ordinary entropy
                    # seed upload, retaining the real unseeded trace predicate.
                    manager._reseted = True
                seed_start = time.perf_counter()
                manager.get_new_values([0])
                seed_update_seconds = time.perf_counter() - seed_start
                sampled = generator.token_out_decode_step(readback=False)
                ttnn.synchronize_device(mesh)
                rank_tokens = read_rank_vectors(sampled)
                rank_seeds = read_rank_vectors(generator.sampling.tt_sampling.seeds_tt_tensor)
                row = {
                    "step": step,
                    "tokens_by_rank": [int(rank[0]) for rank in rank_tokens],
                    "seeds_by_rank": [int(rank[0]) for rank in rank_seeds],
                    "all_seed_vectors_equal": all(torch.equal(rank_seeds[0], rank) for rank in rank_seeds[1:]),
                    "internal_sampling_trace": use_trace,
                    "actual_explicit_seed_active": manager._active_request_seed,
                    "seed_update_host_seconds": seed_update_seconds,
                }
                row["active_tokens_equal"] = len(set(row["tokens_by_rank"])) == 1
                case["steps"].append(row)
                generated_ids.append(row["tokens_by_rank"][0])
                print("SAMPLING_RANK_CONTROL", label, json.dumps(row), flush=True)
                if prompt_ids is not None and generated_ids[-1] == generator.tokenizer.eos_token_id:
                    break
        finally:
            manager.has_active_request_seed = original_trace_predicate
            case["generated_token_ids"] = generated_ids
            case["generated_text"] = generator.tokenizer.decode(generated_ids, skip_special_tokens=False)
            case["stopped_on_eos"] = generated_ids[-1] == generator.tokenizer.eos_token_id
            if save_case is not None:
                save_case(cases)
            print("SAMPLING_TEXT_CONTROL", label, json.dumps(case["generated_text"]), flush=True)
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--max-context", type=int, default=128)
    parser.add_argument("--chat-prompt-id", type=int)
    parser.add_argument("--skip-penalty-check", action="store_true")
    parser.add_argument("--refresh-unseeded-each-step", action="store_true")
    parser.add_argument("--empty-view-repetitions", type=int, default=0)
    parser.add_argument(
        "--sampling-cases",
        nargs="+",
        choices=("unseeded_traced", "unseeded_eager", "seed42_traced", "seed42_eager"),
    )
    parser.add_argument("--sampling-consistency-steps", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(8)
    result = {"invocation": sys.argv, "batch": 32, "num_layers": args.num_layers, "max_context": args.max_context}
    prompt_ids = None
    if args.chat_prompt_id is not None:
        prompt_path = Path(__file__).parent / "artifacts/qualitative_prompt_format.json"
        metadata = json.loads(prompt_path.read_text())
        prompt = next(case for case in metadata["cases"] if case["id"] == args.chat_prompt_id)
        prompt_ids = prompt["prompt_token_ids"]
        result["prompt"] = prompt
        result["model"] = metadata["model"]
        result["revision"] = metadata["revision"]
        if len(prompt_ids) + args.sampling_consistency_steps >= args.max_context:
            raise ValueError("prompt and generated tokens must fit max_context")

    def save_cases(cases):
        result["sampling_rank_controls"] = cases
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    generator = None
    mesh = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
        generator = build_generator(
            Path(__file__).parents[2], mesh, batch=32, num_layers=args.num_layers, max_context=args.max_context
        )
        if not args.skip_penalty_check:
            generator.reset()
            tokens = torch.arange(32 * 32, dtype=torch.long).reshape(32, 32) + 100
            logits = generator.prefill_forward(
                tokens,
                page_table=generator._page_table,
                kv_cache=generator.kv_cache,
                prompt_lens=[32] * 32,
            )
            first_tokens = logits.reshape(32, -1).argmax(dim=-1)
            params = SamplingParams(
                temperature=1.0,
                top_k=1,
                top_p=0.0,
                presence_penalty=0.75,
                frequency_penalty=0.5,
                repetition_penalty=1.25,
            )
            generator.sampling.apply_prefill_state(
                sampling_params=format_sampling_params(params, 32),
                prompt_tokens=tokens,
                empty_slots=list(range(32)),
                replicate_seeds=False,
            )
            penalties = generator.sampling.tt_penalties
            shard_width = generator.model.padded_vocab_size // 4
            # Every vocabulary shard carries nonzero history. The repeated first
            # token also makes preservation of counts stronger than a mask check.
            history = torch.stack([torch.arange(32) + 23 + shard * shard_width for shard in range(4)], dim=1)
            history = torch.cat([history, history[:, :1]], dim=1)
            if int(history.max()) >= generator.model.vocab_size:
                raise AssertionError("constructed history reaches padded vocabulary")
            generator.sampling.reset_output_state(history)
            ttnn.synchronize_device(mesh)
            before = read_history(penalties)
            result["buffer_shapes_per_device"] = {
                name: [list(shard.shape) for shard in shards] for name, shards in before.items()
            }
            if not all(bool(shard.ne(0).any()) for shards in before.values() for shard in shards):
                raise AssertionError("all three buffers on all four devices must have nonzero history")

            generator.setup_token_out_decode(
                first_tokens,
                torch.full((32,), 32, dtype=torch.int32),
                page_table=generator.page_table_host,
                kv_cache=generator.kv_cache,
                sampling_params=params,
                preserve_sampling_history=True,
            )
            ttnn.synchronize_device(mesh)
            after_setup = read_history(penalties)
            result["setup_mismatches"] = {
                name: [int(left.ne(right).sum()) for left, right in zip(before[name], after_setup[name])]
                for name in BUFFERS
            }
            if any(count for counts in result["setup_mismatches"].values() for count in counts):
                raise AssertionError("recapture changed live output history")
            print("PENALTY_RECAPTURE_HISTORY_EXACT", json.dumps(result["setup_mismatches"]), flush=True)

            generator.sampling.seed_manager.get_new_values(list(range(32)))
            sampled_tt = generator.token_out_decode_step(readback=False)
            ttnn.synchronize_device(mesh)
            sampled = ttnn.to_torch(ttnn.get_device_tensors(sampled_tt)[0]).reshape(-1)[:32].to(torch.int64)
            if sampled.numel() != 32 or not bool(((sampled >= 0) & (sampled < generator.model.vocab_size)).all()):
                raise AssertionError("replay did not produce 32 valid vocabulary tokens")
            after_replay = read_history(penalties)
            expected_global = before["output_counts_gathered"][0].clone()
            expected_global.scatter_add_(1, sampled[:, None], torch.ones((32, 1), dtype=torch.int64))
            expected_shards = [
                expected_global[:, shard * shard_width : (shard + 1) * shard_width] for shard in range(4)
            ]
            expected = {
                "output_mask": [shard.gt(0).to(torch.int64) for shard in expected_shards],
                "output_counts": expected_shards,
                "output_counts_gathered": [expected_global] * 4,
            }
            result["replay_mismatches"] = {
                name: [int(left.ne(right).sum()) for left, right in zip(expected[name], after_replay[name])]
                for name in BUFFERS
            }
            if any(count for counts in result["replay_mismatches"].values() for count in counts):
                raise AssertionError("replay history differs from exactly one sampled token per row")
            result["sampled_tokens"] = sampled.tolist()
            result["sampling_uses_internal_trace"] = not generator.sampling.seed_manager.has_active_request_seed()
            result["trace_counters"] = dict(generator.trace_counters)
            result["status"] = "PENALTY_RECAPTURE_AND_REPLAY_EXACT"
            print(result["status"], json.dumps(result["replay_mismatches"]), flush=True)
        if args.sampling_consistency_steps:
            result["sampling_rank_controls"] = sampling_rank_controls(
                generator,
                mesh,
                args.sampling_consistency_steps,
                prompt_ids=prompt_ids,
                save_case=save_cases,
                refresh_unseeded_each_step=args.refresh_unseeded_each_step,
                selected_cases=args.sampling_cases,
            )
            result["sampling_rank_agreement"] = all(
                len(set(case["prefill_tokens_by_rank"])) == 1
                and all(row["active_tokens_equal"] and row["all_seed_vectors_equal"] for row in case["steps"])
                for case in result["sampling_rank_controls"].values()
            )
            result["status"] = "SAMPLING_RANK_CONTROLS_COMPLETE" if args.skip_penalty_check else result["status"]
            if {"seed42_traced", "seed42_eager"} <= result["sampling_rank_controls"].keys():
                result["seed42_trace_eager_tokens_equal"] = (
                    result["sampling_rank_controls"]["seed42_traced"]["generated_token_ids"]
                    == result["sampling_rank_controls"]["seed42_eager"]["generated_token_ids"]
                )
            if not result["sampling_rank_agreement"]:
                raise AssertionError("TP4 sampling seeds or active token IDs disagree; see recorded controls")
        if args.empty_view_repetitions:
            result["empty_slot_view"] = empty_view_controls(generator, mesh, args.empty_view_repetitions)
            if result["empty_slot_view"]["mismatched_cache_keys"]:
                raise AssertionError("empty slot-view roundtrip changed cache contents")
    except Exception as exc:
        result["status"] = "FAILED"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if generator is not None:
                generator.teardown()
            if mesh is not None:
                ttnn.close_mesh_device(mesh)
        finally:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
