# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Mixed linear/full stack, resumed prefill and heterogeneous traced requests.

Serially runs optimized chip 0, closes it, then runs the four-chip decoder mesh.
Two requests share a physical KV pool with disjoint permuted pages, nonaligned
prefix boundaries and different current positions. The same decode trace is replayed with two distinct page permutations;
all inputs, states, positions, plans and saved state buffers exist before capture.
No boundary conversion is inserted between the two decoder layers.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.reference import config, pcc, weights
from models.autoports.qwen_qwen3_6_35b_a3b.tests.run_multichip_decoder import _state_checks, _write
from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only


def make_inputs(args, cfg):
    if len(args.prefix_lengths) != len(args.continuation_lengths):
        raise ValueError("Each request needs a prefix length and continuation length")
    lengths = [a + b for a, b in zip(args.prefix_lengths, args.continuation_lengths)]
    if min(args.prefix_lengths + args.continuation_lengths) <= 0:
        raise ValueError("Prefix and continuation lengths must be positive")
    if max(lengths) + args.decode_steps > args.context:
        raise ValueError("Context must fit every request and its decode steps")
    if len(set(lengths)) < 2:
        raise ValueError("Use distinct final lengths to test heterogeneous current positions")
    torch.manual_seed(args.seed)
    blocks = (args.context + 31) // 32
    independent_pages = [torch.randperm(blocks) for _ in lengths]
    pool_pages = (len(lengths) * blocks + 8 + 7) // 8 * 8
    pages = []
    guard_pages = []
    for _ in range(2):
        permutation = torch.randperm(pool_pages)
        pages.append([permutation[r * blocks : (r + 1) * blocks].clone() for r in range(len(lengths))])
        guard_pages.append(permutation[len(lengths) * blocks :].clone())
        assert len(torch.cat(pages[-1]).unique()) == len(lengths) * blocks
    assert all(not torch.equal(pages[0][r], pages[1][r]) for r in range(len(lengths)))
    return dict(
        x=[torch.randn(1, n, cfg.hidden_size, dtype=torch.bfloat16) * 0.05 for n in lengths],
        xd=torch.randn(args.decode_steps, len(lengths), 1, cfg.hidden_size, dtype=torch.bfloat16) * 0.05,
        pages=pages,
        independent_pages=independent_pages,
        pool_pages=pool_pages,
        guard_pages=guard_pages,
        lengths=lengths,
    )


def written_cache_mask(inputs, phase, decode_steps):
    mask = torch.zeros((inputs["pool_pages"], 32), dtype=torch.bool)
    for pages, length in zip(inputs["pages"][phase], inputs["lengths"]):
        positions = torch.arange(length + decode_steps)
        mask[pages[positions // 32], positions % 32] = True
    assert int(mask.sum()) == sum(length + decode_steps for length in inputs["lengths"])
    return mask


def unwritten_cache_nonzero(physical, mask):
    return int(torch.count_nonzero(physical.permute(0, 2, 1, 3)[~mask]))


def _dram_census(mesh, stage, previous=None):
    """Allocator's per-bank view; multiply by bank count for device totals."""
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    fields = (
        "num_banks",
        "total_bytes_per_bank",
        "total_bytes_allocated_per_bank",
        "total_bytes_free_per_bank",
        "largest_contiguous_bytes_free_per_bank",
    )
    row = dict(stage=stage, **{key: int(getattr(view, key)) for key in fields})
    for short, field in (
        ("capacity", "total_bytes_per_bank"),
        ("allocated", "total_bytes_allocated_per_bank"),
        ("free", "total_bytes_free_per_bank"),
    ):
        row[short + "_bytes_per_device"] = row["num_banks"] * row[field]
    if previous is not None:
        row["allocated_delta_bytes_per_bank"] = (
            row["total_bytes_allocated_per_bank"] - previous["total_bytes_allocated_per_bank"]
        )
        row["allocated_delta_bytes_per_device"] = (
            row["allocated_bytes_per_device"] - previous["allocated_bytes_per_device"]
        )
    return row


def run_side(args, cfg, state_dicts, inputs, *, multichip):
    if multichip:
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.multichip_decoder import MultichipDecoder as Decoder

        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=args.trace_region_size)
    else:
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.optimized_decoder import OptimizedDecoder as Decoder

        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=args.trace_region_size
        )
    trace = None
    allocator_census = []
    census_path = str(Path(args.output).with_suffix("")) + (".multichip" if multichip else ".baseline") + ".dram.json"

    def census(stage):
        ttnn.synchronize_device(mesh)
        allocator_census.append(_dram_census(mesh, stage, allocator_census[-1] if allocator_census else None))
        # Preserve each observed checkpoint even if a later gate fails.
        _write(census_path, allocator_census)
        print("DRAM_CENSUS", json.dumps(allocator_census[-1]), flush=True)

    try:
        census("before_loading")
        build_kwargs = json.loads(args.mesh_build_kwargs if multichip else args.baseline_build_kwargs)
        workspace = None
        if multichip:
            from models.autoports.qwen_qwen3_6_35b_a3b.tt.multichip_decoder import CollectiveWorkspace

            workspace = CollectiveWorkspace(mesh, num_links=build_kwargs.get("num_links", 2))
            build_kwargs["workspace"] = workspace
            census("shared_workspace_constructed")
        decoders = []
        for index, sd in zip((0, 3), state_dicts):
            decoders.append(
                Decoder.from_state_dict(sd, hf_config=cfg, layer_idx=index, mesh_device=mesh, **build_kwargs)
            )
            census(f"layer_{index}_loaded")
        if multichip:
            assert all(d.collective_workspace is workspace for d in decoders)
            assert decoders[0].ep_outputs is decoders[1].ep_outputs
            assert decoders[0].ccl is decoders[1].ccl
        assert [d.kind for d in decoders] == ["linear_attention", "full_attention"]
        sharded = multichip and getattr(decoders[0], "residual_layout", "sharded") != "replicated"
        assert not multichip or all(
            getattr(d, "residual_layout", "sharded") == decoders[0].residual_layout for d in decoders
        )
        mapper = ttnn.ShardTensorToMesh(mesh, dim=-1) if sharded else ttnn.ReplicateTensorToMesh(mesh)
        batch = len(inputs["lengths"])

        def upload(t):
            return ttnn.from_torch(t, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)

        output_replica_checks = 0

        def host(t):
            nonlocal output_replica_checks
            if sharded:
                return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).clone()
            parts = [ttnn.to_torch(part) for part in ttnn.get_device_tensors(t)]
            for rank, part in enumerate(parts[1:], 1):
                assert torch.equal(parts[0], part), ("output replicas differ", rank)
                output_replica_checks += 1
            return parts[0].clone()

        def new_states(phase=None):
            linear = [decoders[0].new_recurrent_state() for _ in range(batch)]
            if phase is None:
                full = [
                    decoders[1].new_kv_state(context_length=args.context, page_order=pages)
                    for pages in inputs["independent_pages"]
                ]
            else:
                full = []
                cache = None
                for pages in inputs["pages"][phase]:
                    state = decoders[1].new_kv_state(
                        context_length=args.context,
                        page_order=pages,
                        num_physical_pages=inputs["pool_pages"],
                        kv_cache=cache,
                    )
                    if cache is None:
                        cache = (state["k"], state["v"])
                    else:
                        assert state["k"] is cache[0] and state["v"] is cache[1]
                    full.append(state)
            return [linear, full]

        states = [new_states(phase) for phase in range(2)]
        whole_states = new_states()
        independent_states = new_states()
        state_keys = (("conv", "recurrent"), ("k", "v", "page_table"))
        tensor_keys = (("conv", "recurrent"), ("k", "v"))
        prefix_inputs = [upload(t[:, :n]) for t, n in zip(inputs["x"], args.prefix_lengths)]
        continuation_inputs = [upload(t[:, n:]) for t, n in zip(inputs["x"], args.prefix_lengths)]
        whole_inputs = [upload(t) for t in inputs["x"]]
        step_inputs = [
            [upload(inputs["xd"][step, row : row + 1]) for row in range(batch)] for step in range(args.decode_steps)
        ]
        stable_inputs = [ttnn.clone(t) for t in step_inputs[0]]
        prefix_plans = [[d.prepare_prefill(seq_len=n) for n in args.prefix_lengths] for d in decoders]
        continuation_plans = [
            [
                d.prepare_prefill(seq_len=n, start_pos=start)
                for start, n in zip(args.prefix_lengths, args.continuation_lengths)
            ]
            for d in decoders
        ]
        whole_plans = [[d.prepare_prefill(seq_len=n) for n in inputs["lengths"]] for d in decoders]
        step_positions = [
            [decoders[1].prepare_positions(start_pos=length + step, seq_len=1) for length in inputs["lengths"]]
            for step in range(args.decode_steps)
        ]
        stable_positions = [{key: ttnn.clone(t) for key, t in pos.items()} for pos in step_positions[0]]
        if multichip:
            for d in decoders:
                d.prepare_decode_batch(batch)

        census("caches_inputs_positions_and_workspace_prepared")

        def prefill(rows, active_states, plans):
            for d, group, group_plans in zip(decoders, active_states, plans):
                rows = [d.prefill_forward(t, state=s, plan=p) for t, s, p in zip(rows, group, group_plans)]
            return rows

        def host_rows(rows, lengths):
            return [host(t).reshape(1, length, cfg.hidden_size) for t, length in zip(rows, lengths)]

        prefixes, continuations = [], []
        for phase in range(2):
            with device_only():
                output = prefill(prefix_inputs, states[phase], prefix_plans)
            prefixes.append(host_rows(output, args.prefix_lengths))
            del output
            with device_only():
                output = prefill(continuation_inputs, states[phase], continuation_plans)
            continuations.append(host_rows(output, args.continuation_lengths))
            del output
        # The same resumed schedule on independent pools controls shared-page
        # ownership without conflating pool layout with chunking numerics.
        with device_only():
            output = prefill(prefix_inputs, independent_states, prefix_plans)
        independent_prefix = host_rows(output, args.prefix_lengths)
        del output
        with device_only():
            output = prefill(continuation_inputs, independent_states, continuation_plans)
        independent_continuation = host_rows(output, args.continuation_lengths)
        del output
        for phase in range(2):
            assert all(
                torch.equal(a, b) for a, b in zip(prefixes[phase], independent_prefix)
            ), "shared/independent prefix mismatch"
            assert all(
                torch.equal(a, b) for a, b in zip(continuations[phase], independent_continuation)
            ), "shared/independent continuation mismatch"
        # Independent one-shot control proves continuation preserves old state.
        with device_only():
            output = prefill(whole_inputs, whole_states, whole_plans)
        whole = host_rows(output, inputs["lengths"])
        del output
        resumed_pcc = [
            [pcc(expected, torch.cat((a, b), dim=1)) for expected, a, b in zip(whole, pre, cont)]
            for pre, cont in zip(prefixes, continuations)
        ]
        assert min(value for phase in resumed_pcc for value in phase) >= 0.995, resumed_pcc
        snapshots = [
            [
                [{key: ttnn.clone(s[key]) for key in keys} for s in group]
                for group, keys in zip(phase_states, state_keys)
            ]
            for phase_states in states
        ]
        stable_states = states[0]
        census("prefill_controls_and_saved_state_prepared")

        def restore(phase):
            for depth, (target_group, source_group, keys) in enumerate(
                zip(stable_states, snapshots[phase], state_keys)
            ):
                for row, (target, source) in enumerate(zip(target_group, source_group)):
                    for key in keys:
                        # Both request dictionaries alias this physical pool;
                        # restore it once, then refresh each row's own table.
                        if depth == 1 and row > 0 and key in ("k", "v"):
                            continue
                        ttnn.copy(source[key], target[key])

        def refresh(step):
            for target, source in zip(stable_inputs, step_inputs[step]):
                ttnn.copy(source, target)
            for target, source in zip(stable_positions, step_positions[step]):
                for key in source:
                    ttnn.copy(source[key], target[key])

        def decode(active_states=None):
            active_states = stable_states if active_states is None else active_states
            rows = decoders[0].decode_batch(stable_inputs, states=active_states[0], positions=[None] * batch)
            return decoders[1].decode_batch(rows, states=active_states[1], positions=stable_positions)

        def host_decode(rows):
            return torch.cat([host(t).reshape(1, 1, cfg.hidden_size) for t in rows])

        def host_state(active_states=None):
            active_states = stable_states if active_states is None else active_states
            return [
                [
                    {key: [ttnn.to_torch(t).clone() for t in ttnn.get_device_tensors(s[key])] for key in keys}
                    for s in group
                ]
                for group, keys in zip(active_states, tensor_keys)
            ]

        independent_decode = []
        for step in range(args.decode_steps):
            refresh(step)
            with device_only():
                output = decode(independent_states)
            independent_decode.append(host_decode(output))
            del output
        independent_decode = torch.stack(independent_decode)
        independent_final_states = host_state(independent_states)
        eager = []
        for phase in range(2):
            restore(phase)
            step_outputs = []
            for step in range(args.decode_steps):
                refresh(step)
                with device_only():
                    output = decode()
                step_outputs.append(host_decode(output))
                del output
            eager.append(torch.stack(step_outputs))
            assert torch.equal(eager[-1], independent_decode), ("shared/independent decode mismatch", phase)
        restore(0)
        refresh(0)
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        with device_only():
            traced = decode()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        from ttnn.tools import trace_allocation_tracker

        # Each final output is replay-overwritten, and kept live until release.
        # All weights, CCL buffers, inputs, positions, physical state and saved
        # snapshots retain their allocations while either page mapping is used.
        for output in traced:
            trace_allocation_tracker.acknowledge_corruptible(output)
        census("decode_trace_live")
        replay_streams, final_states = {}, {}
        shared_pool_checks = []
        for repeat in range(args.stress_repeats):
            for phase in range(2):
                restore(phase)  # Includes refreshed page table at the same address.
                step_outputs = []
                for step in range(args.decode_steps):
                    refresh(step)
                    with device_only():
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    step_outputs.append(host_decode(traced))
                stream = torch.stack(step_outputs)
                assert torch.equal(eager[phase], stream), ("eager/trace mismatch", phase, repeat)
                if phase in replay_streams:
                    assert torch.equal(replay_streams[phase], stream), ("nondeterminism", phase, repeat)
                replay_streams[phase] = stream
                final_states[phase] = host_state()
                # Every cell outside the two requests' written logical ranges
                # must remain zero, including guard pages and owned future tails.
                written_mask = written_cache_mask(inputs, phase, args.decode_steps)
                for key in ("k", "v"):
                    for rank, physical in enumerate(final_states[phase][1][0][key]):
                        nonzero = unwritten_cache_nonzero(physical, written_mask)
                        assert nonzero == 0, ("unowned/future KV cells modified", phase, repeat, key, rank, nonzero)
                        shared_pool_checks.append(
                            dict(phase=phase, repeat=repeat, tensor=key, rank=rank, unwritten_nonzero=nonzero)
                        )
                for depth, (shared_group, independent_group) in enumerate(
                    zip(final_states[phase], independent_final_states)
                ):
                    for row, (shared, independent) in enumerate(zip(shared_group, independent_group)):
                        for key in tensor_keys[depth]:
                            for rank, (actual, expected) in enumerate(zip(shared[key], independent[key])):
                                if depth == 1:
                                    length = inputs["lengths"][row] + args.decode_steps
                                    actual = (
                                        actual[inputs["pages"][phase][row]]
                                        .permute(1, 0, 2, 3)
                                        .flatten(1, 2)[:, :length]
                                    )
                                    expected = (
                                        expected[inputs["independent_pages"][row]]
                                        .permute(1, 0, 2, 3)
                                        .flatten(1, 2)[:, :length]
                                    )
                                assert torch.equal(actual, expected), (
                                    "shared/independent state mismatch",
                                    phase,
                                    repeat,
                                    depth,
                                    row,
                                    key,
                                    rank,
                                )
                for row, state in enumerate(stable_states[1]):
                    pages = inputs["pages"][phase][row].to(torch.int32)
                    expected_table = torch.cat((pages, pages[-1:].expand((-len(pages)) % 8)))
                    for table in ttnn.get_device_tensors(state["page_table"]):
                        assert torch.equal(ttnn.to_torch(table).reshape(-1), expected_table), (
                            "page-table refresh mismatch",
                            phase,
                            row,
                        )
        print("MIXED_STACK_SIDE_DONE", Decoder.__name__, flush=True)
        return dict(
            prefix=prefixes,
            continuation=continuations,
            whole=whole,
            independent_prefix=independent_prefix,
            independent_continuation=independent_continuation,
            independent_decode=independent_decode,
            independent_states=independent_final_states,
            decode=replay_streams,
            states=final_states,
            summary=dict(
                decoder=Decoder.__name__,
                layers=[0, 3],
                kinds=[d.kind for d in decoders],
                residual_layout="hidden-sharded" if sharded else "replicated",
                one_shot_vs_resumed_pcc=resumed_pcc,
                all_step_eager_trace_equal=True,
                deterministic=True,
                trace_capture_count=1,
                trace_replay_count=2 * args.stress_repeats * args.decode_steps,
                page_table_phases=2,
                shared_kv_pool=True,
                shared_kv_pool_pages=inputs["pool_pages"],
                shared_kv_request_count=batch,
                shared_kv_disjoint_page_ownership=True,
                shared_kv_guard_pages=[pages.tolist() for pages in inputs["guard_pages"]],
                shared_kv_vs_independent_bit_equal=True,
                shared_kv_vs_independent_state_bit_equal=True,
                shared_kv_page_tables_exact=True,
                shared_kv_untouched_checks=shared_pool_checks,
                output_replicas_bit_equal=True if multichip and not sharded else None,
                output_replica_checks=output_replica_checks,
                page_table_refresh="copy saved table and matching physical cache to stable tensors between request phases",
                positions=[[length + step for length in inputs["lengths"]] for step in range(args.decode_steps)],
                runtime_fallback_audit="pass",
                shared_collective_workspace=multichip,
                collective_workspace_lifetime=(
                    "single serial CQ0 workspace retained through trace release" if multichip else None
                ),
                allocator_census=allocator_census,
                allocator_census_artifact=census_path,
                loaded_decoder_bytes_per_device={
                    row["stage"]: row["allocated_delta_bytes_per_device"]
                    for row in allocator_census
                    if row["stage"] in ("layer_0_loaded", "layer_3_loaded")
                },
                allocator_interpretation="Allocator per-bank view multiplied by num_banks; layer-load deltas include all resident constructor allocations. Cache/workspace preparation and live-trace deltas are separate; test holds multiple independent cache controls.",
            ),
        )
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


def run(args):
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record

    if args.decode_steps < 2 or args.stress_repeats < 2:
        raise ValueError("Changing-input deterministic trace validation needs at least two steps and repeats")
    torch.set_num_threads(8)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record(str(output))
    provenance_path = str(output) + ".provenance.json"
    provenance = json.loads(Path(provenance_path).read_text())
    provenance.update(device="serial 1x1 chip 0 baseline then 1x4 Blackhole mesh", decoder="mixed linear/full stack")
    _write(provenance_path, provenance)
    cfg = config()
    inputs = make_inputs(args, cfg)
    state_dicts = [weights(index) for index in (0, 3)]
    baseline = run_side(args, cfg, state_dicts, inputs, multichip=False)
    candidate = run_side(args, cfg, state_dicts, inputs, multichip=True)
    tensor_path = output.with_suffix(".pt")
    torch.save(dict(inputs=inputs, baseline=baseline, multichip=candidate), tensor_path)
    comparisons = []
    for phase in range(2):
        for mode in ("prefix", "continuation"):
            # The whole logical output is the gate. Per-request rows remain
            # diagnostics, independently of their differing token counts.
            expected = torch.cat([t.reshape(-1) for t in baseline[mode][phase]])
            actual = torch.cat([t.reshape(-1) for t in candidate[mode][phase]])
            comparisons.append(dict(phase=phase, mode=mode, pcc=pcc(expected, actual)))
        for step, (expected, actual) in enumerate(zip(baseline["decode"][phase], candidate["decode"][phase])):
            comparisons.append(dict(phase=phase, mode="decode", step=step, pcc=pcc(expected, actual)))
    state_checks = []
    for phase in range(2):
        for index, length in enumerate(inputs["lengths"]):
            base = {"states": [[group[index]] for group in baseline["states"][phase]]}
            multi = {"states": [[group[index]] for group in candidate["states"][phase]]}
            checks = _state_checks(
                base,
                multi,
                {"pages": [inputs["pages"][phase][index]]},
                SimpleNamespace(length=length, decode_steps=args.decode_steps),
            )
            state_checks.extend(dict(check, phase=phase, request=index) for check in checks)
    scores = dict(
        layers=[0, 3],
        prefix_lengths=args.prefix_lengths,
        continuation_lengths=args.continuation_lengths,
        context=args.context,
        decode_steps=args.decode_steps,
        baseline=baseline["summary"],
        multichip=candidate["summary"],
        output_comparisons=comparisons,
        state_checks=state_checks,
        seed=args.seed,
        tensor_artifact=str(tensor_path),
        trace_allocation_tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1",
        passed=all(c["pcc"] >= 0.995 for c in comparisons + state_checks),
    )
    _write(str(output), scores)
    print(json.dumps(scores, indent=2), flush=True)
    assert scores["passed"], "Mixed-stack whole-output and state PCC must pass 0.995"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=[33, 35])
    parser.add_argument("--continuation-lengths", type=int, nargs="+", default=[32, 34])
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--stress-repeats", type=int, default=3)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--trace-region-size", type=int, default=67108864)
    parser.add_argument("--baseline-build-kwargs", default="{}")
    parser.add_argument("--mesh-build-kwargs", default="{}")
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
