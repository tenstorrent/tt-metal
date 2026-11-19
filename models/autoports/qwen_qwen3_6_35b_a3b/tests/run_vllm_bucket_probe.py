# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reduced bucket/route exactness and retained-trace discriminator, without a server.

Run after sourcing lane-env.sh, with an external timeout. Requires the generator
constructor's decode_buckets=(2, 32), model_trace_key(active_slots), and retained
model_2/model_fixed/sampling/sampling_64 traces. Every reference forces the already-captured
fixed32 trace. Optional B4 adds a retained bucket; route masking compares a
separate opt-in generator against the unchanged generator. No recapture occurs.
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
os.environ.setdefault("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")
assert os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
assert os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.run_serving_commands_probe import StaleInputs, host, local_bits
from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator import QwenGenerator
from models.autoports.qwen_qwen3_6_35b_a3b.tt.model import QwenModel

ROOT = Path(__file__).resolve().parents[1]
CAPACITY = 32
OWNERS = (0, 1, 17, 23)
LENGTHS = (33, 47, 65, 49)


def compare_tree(reference, candidate, path="state"):
    """Exact tensor comparison; no tolerance or hash-only verdict."""
    if isinstance(reference, torch.Tensor):
        if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
            raise AssertionError(f"{path}: tensor metadata differs")
        if not torch.equal(reference, candidate):
            different = reference != candidate
            index = int(different.reshape(-1).nonzero()[0])
            raise AssertionError(
                f"{path}: {int(different.sum())} differing elements; first_flat_index={index}, "
                f"reference={reference.reshape(-1)[index].item()}, candidate={candidate.reshape(-1)[index].item()}"
            )
    elif isinstance(reference, dict):
        assert reference.keys() == candidate.keys(), path
        for key in reference:
            compare_tree(reference[key], candidate[key], f"{path}.{key}")
    elif isinstance(reference, (list, tuple)):
        assert len(reference) == len(candidate), path
        for index, (left, right) in enumerate(zip(reference, candidate)):
            compare_tree(left, right, f"{path}[{index}]")
    else:
        assert reference == candidate, (path, reference, candidate)


def main():
    global OWNERS, LENGTHS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--trace-region-size", type=int, default=768 * 1024 * 1024)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--four-row-bucket", action="store_true", help="Compare retained 2/4/32 against fixed32")
    parser.add_argument("--route-mask", action="store_true", help="Compare opt-in inactive route suppression")
    parser.add_argument(
        "--route-plan", action="store_true", help="Compare fused integer routing against unfused routing"
    )
    parser.add_argument("--solo-attention", action="store_true")
    parser.add_argument("--packed-postreduce-weights", action="store_true")
    parser.add_argument("--native-qk-reader", action="store_true")
    parser.add_argument("--prune-postreduce-padding", action="store_true")
    parser.add_argument(
        "--hidden-row-mask",
        action="store_true",
        help="Compare fused tiled row extraction/masking with the original path",
    )
    parser.add_argument(
        "--conditional-state-commit", action="store_true", help="Compare direct conditional state commit with staging"
    )
    parser.add_argument(
        "--ep-weight-nd-sharding", action="store_true", help="Compare ND gate/up weights with interleaved"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "readiness_vllm/bucket_probe/result.json")
    args = parser.parse_args()
    assert not args.solo_attention or args.route_mask
    assert not args.route_plan or args.route_mask
    separate_reference = (
        args.route_mask
        or args.ep_weight_nd_sharding
        or args.conditional_state_commit
        or args.hidden_row_mask
        or args.packed_postreduce_weights
        or args.native_qk_reader
        or args.prune_postreduce_padding
    )
    buckets = (2, 4, 32) if args.four_row_bucket else (2, 32)
    if args.four_row_bucket:
        OWNERS = (0, 1, 2, 3, 17)
        LENGTHS = (33, 47, 65, 49, 67)
    if args.route_mask:
        assert args.four_row_bucket
        OWNERS += (31,)
        LENGTHS += (69,)
    assert args.context >= 1024 and 3 <= args.steps <= 8
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(
        status="RUNNING",
        command=sys.argv,
        source_sha256={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "tt").glob("*.py"))
        },
        layers=[0, 3],
        capacity=CAPACITY,
        context=args.context,
        trace_region_size=args.trace_region_size,
        tracking=True,
        skip_program_cache=False,
        owners=list(OWNERS),
        candidate_buckets=list(buckets),
        route_mask=args.route_mask,
        route_plan=args.route_plan,
        solo_attention=args.solo_attention,
        hidden_row_mask=args.hidden_row_mask,
        packed_postreduce_weights=args.packed_postreduce_weights,
        native_qk_reader=args.native_qk_reader,
        prune_postreduce_padding=args.prune_postreduce_padding,
        reference_hidden_row_mask=False,
        conditional_state_commit=args.conditional_state_commit,
        reference_conditional_state_commit=False,
        ep_weight_nd_sharding=args.ep_weight_nd_sharding,
        reference_ep_weight_nd_sharding=False,
        separate_fixed32_reference=separate_reference,
        prompt_lengths=list(LENGTHS),
        comparison="exact full-vocabulary active-row logits, tokens, positions, seeds, all-chip recurrent/conv and full KV pools",
        limitations=[
            "Reduced layers0/3, not full-model accuracy, qualitative quality or serving performance.",
            "Direct generator v1 commands, not plugin scheduler/bucket admission or allocator-driven page growth.",
            "Queued-output check preserves two live leases; diagnostic snapshots synchronize, so this is not overlap performance.",
            "Transition inputs are deliberately host-authoritative; resident inputs are unreadable sentinels.",
            "Greedy and native32 top-k8; the B4 variant also tests native64 top-k64. Not the full sampling suite.",
            "No request-state permutation or changed page mapping in this discriminator; adapter/server checks cover those separately.",
        ],
        cases=[],
        checkpoints=[],
    )

    native_root = ROOT.parents[2] / "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill"
    result["source_sha256"].update(
        {
            "repo:" + str(path.relative_to(ROOT.parents[2])): hashlib.sha256(path.read_bytes()).hexdigest()
            for family in ("moe_fused_swiglu", "post_combine_reduce")
            for path in sorted((native_root / family).rglob("*"))
            if path.suffix in (".cpp", ".hpp")
        }
    )

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    mesh = generator = None
    generators = {}
    try:
        torch.set_num_threads(8)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=args.trace_region_size)
        model_args = dict(checkpoint=args.model_dir, layer_indices=[0, 3], max_seq_len=args.context)
        model = QwenModel(
            mesh,
            **model_args,
            **({"fuse_hidden_row_mask": True} if args.hidden_row_mask else {}),
            **({"packed_postreduce_weights": True} if args.packed_postreduce_weights else {}),
            **({"native_qk_reader": True} if args.native_qk_reader else {}),
            **({"prune_postreduce_padding": True} if args.prune_postreduce_padding else {}),
            **({"fuse_recurrent_state_commit": True} if args.conditional_state_commit else {}),
            **({"ep_weight_nd_sharding": True} if args.ep_weight_nd_sharding else {}),
            **(
                {
                    "suppress_inactive_moe_routes": True,
                    "specialize_solo_slot0_attention": args.solo_attention,
                    **({"fuse_moe_route_plan": True} if args.route_plan else {}),
                }
                if args.route_mask
                else {}
            ),
        )
        generator = QwenGenerator(model, max_batch_size=CAPACITY, decode_buckets=buckets)
        generators["candidate"] = generator
        if separate_reference:
            reference_model = QwenModel(
                mesh,
                **model_args,
                suppress_inactive_moe_routes=args.route_plan,
            )
            generators["reference"] = QwenGenerator(reference_model, max_batch_size=CAPACITY, decode_buckets=(32,))
        if args.route_mask:
            from models.autoports.qwen_qwen3_6_35b_a3b.tt.moe_route_mask import mask_inactive_routes

            route_ids = torch.arange(256, dtype=torch.int32).reshape(1, 1, 32, 8)
            device_ids = ttnn.from_torch(
                route_ids,
                dtype=ttnn.uint32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            result["route_helper_cases"] = []
            for slots, active in (
                ((0, 1), (0,)),
                ((0, 1), (17,)),
                ((0, 1, 2, 3), (0, 3, 17)),
                (tuple(range(32)), tuple(range(32))),
                (tuple(range(32)), (31,)),
                ((31, 0), (31,)),
            ):
                generator._set_active_slots(active)
                masked = mask_inactive_routes(
                    device_ids, generator.active_mask, model.layers[0].ep_reduce_table, slots, mesh
                )
                for rank, tensor in enumerate(ttnn.get_device_tensors(masked)):
                    expected = route_ids.clone()
                    for row in range(32):
                        if row >= len(slots) or slots[row] not in active:
                            expected[0, 0, row, :] = 64 if rank == 0 else 0
                    actual = ttnn.to_torch(tensor).to(torch.int32)
                    assert torch.equal(actual, expected), ("route_helper", rank, slots, active)
                result["route_helper_cases"].append(dict(slots=list(slots), active=list(active), exact_all_chips=True))
                ttnn.deallocate(masked)
            generator._set_active_slots((0,))
            ttnn.deallocate(device_ids)
            save()
        full_groups = [group for group in generator.cache if "page_table" in group[0]]
        width = full_groups[0][0]["page_table"].shape[-1]
        pages = torch.zeros((CAPACITY, width), dtype=torch.int32)
        for ordinal, slot in enumerate(OWNERS):
            pages[slot] = torch.arange(width).roll(-ordinal * (width // len(OWNERS)))
            assert (LENGTHS[ordinal] + args.steps + 31) // 32 <= width // len(OWNERS)
        texts = (
            "A triangle has three sides and three angles. ",
            "A garden contains trees, flowers and a small pond. ",
            "A computer follows a sequence of precise instructions. ",
            "The distant mountain is covered with snow in winter. ",
        )
        if args.four_row_bucket:
            texts += ("The river flows through the valley toward the sea. ",)
        if args.route_mask:
            texts += ("The moon reflects sunlight above the ocean. ",)
        fixtures = [generator.tokenizer.encode(text * 50, add_special_tokens=False) for text in texts]
        prompts = torch.zeros((len(OWNERS), max(LENGTHS)), dtype=torch.int64)
        for row, length in enumerate(LENGTHS):
            prompts[row, :length] = torch.tensor(fixtures[row][:length])
        result["prompt_token_ids"] = [row[:length].tolist() for row, length in zip(prompts, LENGTHS)]

        def snapshot(active=()):
            cache = {}
            for layer_index, group in zip(model.layer_indices, generator.cache):
                if "recurrent" in group[0]:
                    cache[str(layer_index)] = {
                        str(slot): {key: local_bits(group[slot][key]) for key in ("recurrent", "conv")}
                        for slot in range(CAPACITY)
                    }
                else:
                    # Shared physical pools: read each once, covering every page
                    # including dormant owners and unowned storage on all chips.
                    cache[str(layer_index)] = {key: local_bits(group[0][key]) for key in ("k", "v")}
            logits = model.host_logits(generator.sampler_logits)[0, 0]
            if active:
                logits = logits[list(active)].clone()
                assert torch.isfinite(logits).all()
            else:
                logits = torch.empty(0)
            return dict(
                cache=cache,
                tokens=host(generator.tokens).reshape(-1).to(torch.int64),
                seeds=host(generator.sampler.tt_sampling.seeds_tt_tensor).reshape(-1).to(torch.int64),
                positions=torch.tensor([int(host(value).reshape(-1)[0]) for value in generator.positions]),
                logits=logits,
            )

        # Warm observation paths before any capture. Diagnostic host tensor
        # snapshots are not retained device allocations.
        retained = {}
        for name, generator in generators.items():
            model = generator.model
            generator.prefill_forward(
                prompts, prompt_lens=LENGTHS, start_pos=[0] * len(OWNERS), slots=OWNERS, page_table=pages[list(OWNERS)]
            )
            generator.configure_sampling(
                temperature=[0.7] * CAPACITY,
                top_k=[8] * CAPACITY,
                top_p=[0.9] * CAPACITY,
                seed=[101 + slot for slot in range(CAPACITY)],
            )
            generator.configure_sampling()
            snapshot()
            if separate_reference:
                generator.prepare_without_capture()
        for name, generator in generators.items():
            model = generator.model
            generator.prepare()
            traces = dict(generator.traces)
            capture_count = generator.counters["captures"]
            expected_model_keys = {generator._bucket_trace_key(bucket) for bucket in generator.decode_buckets}
            if generator.solo_slot0_attention:
                expected_model_keys.add("model_solo0")
            assert set(traces) == expected_model_keys | {"sampling", "sampling_64"}, traces
            assert capture_count == len(expected_model_keys) + 2
            retained[name] = (traces, capture_count)

        def select(name):
            nonlocal generator, model, traces, capture_count
            generator = generators[name]
            model = generator.model
            traces, capture_count = retained[name]

        select("candidate")
        entries = int(mesh.num_program_cache_entries())
        result.update(trace_ids={key: str(value) for key, value in traces.items()}, prepared_program_entries=entries)

        def checkpoint(label):
            gc.collect()
            ttnn.synchronize_device(mesh)
            current = int(mesh.num_program_cache_entries())
            row = dict(label=label, program_entries=current, delta=current - entries)
            result["checkpoints"].append(row)
            save()
            assert current == entries, row
            assert generator.traces == traces and generator.counters["captures"] == capture_count
            assert generator.counters["releases"] == 0
            print("BUCKET_CHECKPOINT", json.dumps(row), flush=True)

        def setup(sampled):
            generator.output_pool.require_drained()
            generator.reset()
            generator.prefill_forward(
                prompts, prompt_lens=LENGTHS, start_pos=[0] * len(OWNERS), slots=OWNERS, page_table=pages[list(OWNERS)]
            )
            generator._refresh([fixtures[row][length] for row, length in enumerate(LENGTHS)], LENGTHS, OWNERS)
            values = dict(
                temperature=[0.7 if sampled else 0.0] * CAPACITY,
                top_k=[64 if sampled == 64 else (8 if sampled else 1)] * CAPACITY,
                top_p=[0.9 if sampled else 1.0] * CAPACITY,
                seed=[101 + slot for slot in range(CAPACITY)],
            )
            generator.configure_sampling(**values)
            return values

        def trajectory(label, schedule, sampled, buckets, reference=None):
            generator.decode_buckets = buckets
            params = setup(sampled)
            initial = snapshot()
            checkpoint(label + "/setup")
            if reference is not None:
                compare_tree(reference["initial"], initial, label + ".initial")
            frames, pending, previous_active = [], [], None
            summaries = []
            max_live_leases = 0
            try:
                for step, active in enumerate(schedule):
                    active = tuple(active)
                    changed = active != previous_active
                    if changed:
                        assert not pending, "Drain prior-layout output before a transition"
                    old = initial if not frames else frames[-1]
                    before = dict(generator.counters)
                    replays_before = dict(generator.trace_replays)
                    positions = [int(old["positions"][slot]) if slot in active else -1 for slot in range(CAPACITY)]
                    key = generator.model_trace_key(active)
                    handle = generator.decode_forward_v1(
                        old["tokens"].reshape(CAPACITY, 1) if changed else StaleInputs(),
                        positions if changed else StaleInputs(),
                        kv_cache=generator.cache,
                        page_table=pages if changed else None,
                        sampling_params=params,
                        reload_inputs=changed,
                        reload_sampling_params=changed,
                        reset_sampling_state=changed,
                        read_from_device=False,
                        enable_trace=True,
                    )
                    frame = snapshot(active)
                    replays_after = dict(generator.trace_replays)
                    for trace_key in set(replays_before) | set(replays_after) | {key}:
                        assert replays_after.get(trace_key, 0) - replays_before.get(trace_key, 0) == int(
                            trace_key == key
                        ), (label, step, trace_key, replays_before, replays_after)
                    expected_positions = old["positions"].clone()
                    expected_positions[list(active)] += 1
                    assert torch.equal(frame["positions"], expected_positions), (label, step, "positions")
                    dormant = [slot for slot in range(CAPACITY) if slot not in active]
                    assert torch.equal(frame["tokens"][dormant], old["tokens"][dormant])
                    assert torch.equal(frame["seeds"][dormant], old["seeds"][dormant])
                    for index, layer in zip(model.layer_indices, model.layers):
                        if layer.kind == "linear_attention":
                            for slot in dormant:
                                compare_tree(
                                    old["cache"][str(index)][str(slot)],
                                    frame["cache"][str(index)][str(slot)],
                                    f"{label}.step{step}.dormant{slot}.layer{index}",
                                )
                        else:
                            for slot in set(OWNERS) - set(active):
                                owned_pages = pages[slot, : width // len(OWNERS)].long()
                                for field in ("k", "v"):
                                    for chip, (prior, current) in enumerate(
                                        zip(old["cache"][str(index)][field], frame["cache"][str(index)][field])
                                    ):
                                        compare_tree(
                                            prior[owned_pages],
                                            current[owned_pages],
                                            f"{label}.step{step}.dormant{slot}.layer{index}.{field}.chip{chip}",
                                        )
                    if not changed:
                        for name in (
                            "token_refreshes",
                            "position_refreshes",
                            "page_table_refreshes",
                            "activity_refreshes",
                        ):
                            assert generator.counters[name] == before[name], (label, step, name)
                    frames.append(frame)
                    pending.append((handle, frame))
                    live_leases = sum(lease is not None for lease in generator.output_pool.leases)
                    max_live_leases = max(max_live_leases, live_leases)
                    # Steps1/2 in a steady sequence leave two snapshots leased.
                    keep = step == 1 and step + 1 < len(schedule) and tuple(schedule[step + 1]) == active
                    if not keep:
                        for output, owned_frame in pending:
                            returned, events = generator.read_decode_output(output, async_read=True)
                            assert returned is output and events
                            tokens = generator.process_decode_output_host(returned, is_tokens=True)
                            assert torch.equal(tokens[:, 0], owned_frame["tokens"])
                            owned_frame["output"] = tokens
                        pending.clear()
                    summaries.append(
                        dict(
                            step=step,
                            active=list(active),
                            trace=key,
                            token_ids=frame["tokens"][list(active)].tolist(),
                            positions=frame["positions"][list(active)].tolist(),
                            authoritative_reload=changed,
                            live_output_leases=live_leases,
                            trace_replay_delta={
                                name: replays_after.get(name, 0) - replays_before.get(name, 0)
                                for name in sorted(set(replays_after) | set(replays_before))
                            },
                            counter_delta={name: generator.counters[name] - value for name, value in before.items()},
                        )
                    )
                    previous_active = active
                assert not pending
                if len(set(map(tuple, schedule))) == 1:
                    assert max_live_leases == 2, (label, max_live_leases)
                if reference is not None:
                    for step, frame in enumerate(frames):
                        compare_tree(reference["frames"][step], frame, f"{label}.step{step}")
                checkpoint(label + "/complete")
                return dict(initial=initial, frames=frames, summary=summaries)
            finally:
                # Include any output published before a failed snapshot/assertion.
                for handle in list(generator.output_pool.leases):
                    if handle is not None:
                        generator.output_pool.discard(handle)
                generator.output_pool.require_drained()

        schedules = [(f"steady_{'-'.join(map(str, active))}", [active] * args.steps) for active in ((0,), (1,), (0, 1))]
        schedules.append(("transitions_2_32_2_32", [(0,), (0, 1, 17), (0,), (0, 17)]))
        if args.four_row_bucket:
            schedules += [
                ("steady_four_row_bucket", [(0, 1, 3)] * args.steps),
                ("steady_four_active", [(0, 1, 2, 3)] * args.steps),
                ("transitions_2_4_32_4_2", [(0,), (0, 1, 3), OWNERS, (1, 3), (1,)]),
            ]
        if args.route_mask:
            schedules += [(f"steady_high_{slot}", [(slot,)] * args.steps) for slot in (2, 3, 17, 31)]
            schedules += [("inactive_predecessors", [(17, 31), (0, 17, 31), (31,), (2, 31), (0,)])]
        for sampled in (False, True, 64) if args.four_row_bucket else (False, True):
            for name, schedule in schedules:
                label = ("seeded64" if sampled == 64 else ("seeded" if sampled else "greedy")) + "/" + name
                row = dict(name=label, status="RUNNING", active_schedule=[list(active) for active in schedule])
                result["cases"].append(row)
                save()
                select("reference" if separate_reference else "candidate")
                reference = trajectory(label + "/fixed32", schedule, sampled, (32,))
                row["reference"] = reference["summary"]
                save()
                select("candidate")
                candidate = trajectory(label + "/candidate", schedule, sampled, buckets, reference=reference)
                row.update(
                    status="PASS", exact_initial_state=True, exact_all_steps=True, candidate=candidate["summary"]
                )
                del reference, candidate
                gc.collect()
                save()
        generator.decode_buckets = buckets
        result.update(
            status="PASS",
            counters_before_shutdown=dict(generator.counters),
            trace_replays=dict(generator.trace_replays),
        )
    except BaseException as error:
        result.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        try:
            for name, instance in generators.items():
                instance.close()
                result[
                    "counters_after_shutdown" if name == "candidate" else "reference_counters_after_shutdown"
                ] = dict(instance.counters)
                assert instance.counters["releases"] == instance.counters["captures"]
            if mesh is not None:
                ttnn.close_mesh_device(mesh)
        except BaseException as error:
            result.update(status="FAIL", cleanup_error=repr(error), cleanup_traceback=traceback.format_exc())
            raise
        finally:
            save()
    print("VLLM_BUCKET_PROBE_PASS", args.output, flush=True)


if __name__ == "__main__":
    main()
