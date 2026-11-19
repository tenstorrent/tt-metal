# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Serial optimized-single-chip / mesh validation with persistent paired inputs.

Run under an external bounded timeout. All tensor conversion, state restoration,
input refresh, and output composition are outside the measured layer. A baseline
can be saved in one process and loaded in another (useful for profiler runs).
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.reference import config, pcc, weights
from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only


def _write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def _inputs(args, c):
    torch.manual_seed(args.seed)
    x = torch.randn(args.batch, args.length, c.hidden_size, dtype=torch.bfloat16) * 0.05
    xd = torch.randn(args.decode_steps, args.batch, 1, c.hidden_size, dtype=torch.bfloat16) * 0.05
    if args.activation_file:
        payload = torch.load(args.activation_file, weights_only=True)
        assert payload["metadata"]["checkpoint_revision"] == "345d467b117d1fd5eda9b879a5988de9ae5f47a7"
        row = payload["layers"][str(args.layer)]
        assert tuple(row["x"].shape) == tuple(x.shape)
        x = row["x"]
        if row["xd"].ndim == 4:
            assert row["xd"].shape[0] >= args.decode_steps
            assert tuple(row["xd"].shape[1:]) == tuple(xd.shape[1:])
            xd = row["xd"][: args.decode_steps].clone()
        else:
            assert args.batch == 1
            xd[0] = row["xd"]
    pages = [torch.randperm((args.context + 31) // 32) for _ in range(args.batch)]
    return dict(x=x, xd=xd, pages=pages)


def _run_side(args, c, sd, inputs, *, multichip):
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
    try:
        dec = Decoder.from_state_dict(
            sd,
            hf_config=c,
            layer_idx=args.layer,
            mesh_device=mesh,
            **json.loads(args.mesh_build_kwargs if multichip else args.baseline_build_kwargs),
        )
        # Setup-only geometry experiment: preserve weights/dtype/fidelity and
        # rebuild the same optimized DRAM projection family for local K/N.
        if multichip and args.mesh_projection_geometry:
            dec.projection_geometry = json.loads(args.mesh_projection_geometry)
            for key, weight in dec.mesh_weights.items():
                role = {
                    "self_attn.o_proj.weight": "attention_out",
                    "mlp.shared_expert.down_proj.weight": "shared_down",
                }.get(key, key)
                dec.mesh_decode_weights[key] = dec._make_dram_projection(weight, role)
            dec.precision_policy["mesh_projection_geometry_experiment"] = dec.projection_geometry
        print("LOADED", Decoder.__name__, flush=True)
        sharded = multichip and getattr(dec, "residual_layout", "sharded") != "replicated"
        mapper = ttnn.ShardTensorToMesh(mesh, dim=-1) if sharded else ttnn.ReplicateTensorToMesh(mesh)

        def upload(t):
            return ttnn.from_torch(t, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)

        def host(t):
            if sharded:
                return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).clone()
            replicas = [ttnn.to_torch(part).clone() for part in ttnn.get_device_tensors(t)]
            assert all(torch.equal(replicas[0], part) for part in replicas[1:]), "residual replicas disagree"
            return replicas[0]

        states = [
            [
                (
                    dec.new_recurrent_state()
                    if dec.kind == "linear_attention"
                    else dec.new_kv_state(context_length=args.context, page_order=inputs["pages"][row])
                )
                for row in range(args.batch)
            ]
            for _ in range(args.stack_depth)
        ]
        state_keys = ("conv", "recurrent") if dec.kind == "linear_attention" else ("k", "v")
        plan = dec.prepare_prefill(seq_len=args.length, **({"chunk_size": args.chunk_size} if args.chunk_size else {}))
        tx = [upload(inputs["x"][r : r + 1]) for r in range(args.batch)]
        step_inputs = [
            [upload(inputs["xd"][step, r : r + 1]) for r in range(args.batch)] for step in range(args.decode_steps)
        ]
        stable_inputs = [ttnn.clone(t) for t in step_inputs[0]]
        # Every position tensor is allocated before capture, including future RoPE
        # values. Copy into stable addresses before replay; never recapture.
        step_positions = [
            [
                dec.prepare_positions(start_pos=args.length + step, seq_len=1) if dec.kind == "full_attention" else None
                for _ in range(args.batch)
            ]
            for step in range(args.decode_steps)
        ]
        stable_positions = [
            {key: ttnn.clone(t) for key, t in row.items()} if row else None for row in step_positions[0]
        ]

        zero_buffers = [
            [{key: ttnn.zeros_like(state[key]) for key in state_keys} for state in group] for group in states
        ]

        def zero_states():
            for group, zeros in zip(states, zero_buffers):
                for state, zero in zip(group, zeros):
                    for key in state_keys:
                        ttnn.copy(zero[key], state[key])

        def prefill():
            rows = tx
            for group in states:
                rows = [dec.prefill_forward(t, state=s, plan=plan) for t, s in zip(rows, group)]
            return rows

        if multichip:
            dec.prepare_decode_batch(args.batch)
        empty_expert_rank_check = None
        if multichip and args.batch == 32 and dec.local_expert_dispatch:
            # Force all selections onto rank 0 and poison persistent output
            # storage. Ranks 1-3 must produce finite zeros without reading stale
            # expert regions; rank 0's zero-input expert outputs are also zero.
            probe = ttnn.from_torch(
                torch.zeros(1, 32, 2048),
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            probe_ids = ttnn.from_torch(
                torch.arange(8).reshape(1, 1, 1, 8).expand(1, 1, 32, 8).contiguous(),
                device=mesh,
                dtype=ttnn.uint32,
                layout=ttnn.TILE_LAYOUT,
            )
            probe_scores = ttnn.from_torch(
                torch.full((1, 1, 32, 8), 0.125),
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            for buffer in dec.ep_outputs.values():
                poison = ttnn.add(buffer, 7.0)
                ttnn.copy(poison, buffer)
                del poison
            with device_only():
                probe_out = dec._grouped_moe(probe, probe_ids, probe_scores)
            for part in ttnn.get_device_tensors(probe_out):
                assert torch.count_nonzero(ttnn.to_torch(part)) == 0, "empty expert rank read poisoned output"
            del probe, probe_ids, probe_scores, probe_out
            empty_expert_rank_check = True

        def decode():
            rows = stable_inputs
            for group in states:
                rows = dec.decode_batch(rows, states=group, positions=stable_positions)
            return rows

        def refresh(step):
            for target, source in zip(stable_inputs, step_inputs[step]):
                ttnn.copy(source, target)
            for target, source in zip(stable_positions, step_positions[step]):
                if source:
                    for key in source:
                        ttnn.copy(source[key], target[key])

        with device_only():
            warm = prefill()
        print("PREFILL_DONE", Decoder.__name__, flush=True)
        prefill_output = torch.cat([host(t).reshape(1, args.length, c.hidden_size) for t in warm])
        del warm
        prefill_samples = []
        for _ in range(args.prefill_repeats):
            zero_states()
            ttnn.synchronize_device(mesh)
            begin = time.perf_counter()
            with device_only():
                warm = prefill()
            ttnn.synchronize_device(mesh)
            prefill_samples.append((time.perf_counter() - begin) * 1000)
            assert torch.equal(
                prefill_output, torch.cat([host(t).reshape(1, args.length, c.hidden_size) for t in warm])
            )
            del warm
        if args.profile:
            from tracy import signpost

            zero_states()
            ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)
            signpost("PERF_PREFILL")
            with device_only():
                out = prefill()
            ttnn.synchronize_device(mesh)
            signpost("PERF_PREFILL_END")
            ttnn.ReadDeviceProfiler(mesh)
            del out
        snapshots = [[{key: ttnn.clone(state[key]) for key in state_keys} for state in group] for group in states]

        prefill_states = None
        if getattr(args, "save_prefill_states", False):
            # Optional host-only profiler seed: exact state after the already
            # validated prefill, outside all measured execution windows.
            prefill_states = [
                [
                    {key: [ttnn.to_torch(t).clone() for t in ttnn.get_device_tensors(saved[key])] for key in state_keys}
                    for saved in group
                ]
                for group in snapshots
            ]

        def restore():
            for group, saved in zip(states, snapshots):
                for state, snap in zip(group, saved):
                    for key in state_keys:
                        ttnn.copy(snap[key], state[key])

        # Prepare copies and every decode step before capture. The changing
        # eager stream is an independent direct control for every replay step.
        eager_rows = []
        for step in range(args.decode_steps):
            refresh(step)
            with device_only():
                warm = decode()
            eager_rows.append(torch.cat([host(t).reshape(1, 1, c.hidden_size) for t in warm]))
            del warm
        eager_stream = torch.stack(eager_rows)
        restore()
        refresh(0)
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        with device_only():
            traced = decode()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        from ttnn.tools import trace_allocation_tracker

        # Only final outputs are replay-overwritten. Inputs, weights, position
        # tensors, state, snapshots and CCL state stay live through trace release.
        for t in traced:
            trace_allocation_tracker.acknowledge_corruptible(t)
        streams = []
        for _ in range(args.stress_repeats):
            restore()
            outputs = []
            for step in range(args.decode_steps):
                refresh(step)
                with device_only():
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                outputs.append(torch.cat([host(t).reshape(1, 1, c.hidden_size) for t in traced]))
            streams.append(torch.stack(outputs))
        assert torch.equal(streams[0], eager_stream), "changing-input replay differs from eager stream"
        assert all(torch.equal(streams[0], stream) for stream in streams[1:]), "replay is nondeterministic"
        # Exercise prepared eager prefill while this same decode trace is live.
        # All transient prefill outputs are released before replay; weights,
        # caches, CCL resources and grouped output workspace retain ownership.
        zero_states()
        with device_only():
            live_prefill = prefill()
        del live_prefill
        refresh(0)
        with device_only():
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        live_result = torch.cat([host(t).reshape(1, 1, c.hidden_size) for t in traced])
        assert torch.equal(live_result, eager_stream[0]), "live-trace prefill changed replay"
        decode_samples = []
        refresh(0)
        for _ in range(args.repeats):
            restore()
            ttnn.synchronize_device(mesh)
            begin = time.perf_counter()
            with device_only():
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            decode_samples.append((time.perf_counter() - begin) * 1000)
        functional_sequence = None
        if multichip and getattr(args, "functional_sequence", False):
            from models.autoports.qwen_qwen3_6_35b_a3b.tests.multichip_functional import sequence_bound

            functional_samples = []
            for _ in range(args.functional_repeats):
                restore()
                ttnn.synchronize_device(mesh)
                for step in range(128):
                    refresh(step)
                    ttnn.synchronize_device(mesh)
                    with device_only():
                        begin = time.perf_counter()
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                        elapsed_ms = (time.perf_counter() - begin) * 1000
                    functional_samples.append(elapsed_ms)
                    observed = torch.cat([host(t).reshape(1, 1, c.hidden_size) for t in traced])
                    assert torch.equal(observed, eager_stream[step]), "FUNCTIONAL changing-input trace mismatch"
            bound = sequence_bound(args.layer)
            mean_ms = statistics.mean(functional_samples)
            fraction = bound["roofline_ms_per_token"] / mean_ms
            functional_sequence = dict(
                prompt_length=128,
                decode_length=128,
                batch=1,
                mesh=[1, 4],
                repeats=args.functional_repeats,
                samples_ms=functional_samples,
                decode_ms_per_token_mean=mean_ms,
                positions=list(range(128, 256)),
                repeated_bit_equal=True,
                all_steps_eager_trace_equal=True,
                trace_capture_count=1,
                input_and_position_refresh_each_step=True,
                state_reset_between_repeats=True,
                timing_boundary="blocking trace replay only; per-step device input/position copies, synchronization, "
                "host output checks, audit-context setup/teardown and between-sequence state reset excluded; "
                "no host uploads inside timing",
                theoretical=bound,
                roofline_fraction=fraction,
                floor_fraction=0.1,
                passed=fraction >= 0.1,
            )
        if args.profile:
            from tracy import signpost

            restore()
            ttnn.ReadDeviceProfiler(mesh)
            signpost("PERF_DECODE")
            for _ in range(args.repeats):
                with device_only():
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                ttnn.ReadDeviceProfiler(mesh)
            signpost("PERF_DECODE_END")
        result = dict(
            prefill=prefill_output,
            decode=streams[0],
            summary=dict(
                decoder=Decoder.__name__,
                kind=dec.kind,
                residual_layout="hidden-sharded" if sharded else "replicated",
                output_replicas_equal=True if not sharded else None,
                empty_expert_ranks_zero_after_poison=empty_expert_rank_check,
                prefill_samples_ms=prefill_samples,
                prefill_warmed_ms=statistics.median(prefill_samples),
                decode_samples_ms=decode_samples,
                decode_traced_warmed_ms=statistics.median(decode_samples),
                deterministic=True,
                all_step_eager_trace_equal=True,
                live_trace_prefill_replay_equal=True,
                replay_count=args.stress_repeats * args.decode_steps,
                trace_capture_count=1,
                changing_inputs=not torch.equal(inputs["xd"][0], inputs["xd"][-1]),
                positions=list(range(args.length, args.length + args.decode_steps)),
                state_shapes={key: list(states[0][0][key].shape) for key in state_keys},
                runtime_fallback_audit="pass",
                precision_policy=getattr(dec, "precision_policy", None),
                regime="serial; chips 0,1,2,3" if multichip else "serial; chip 0",
            ),
        )
        if prefill_states is not None:
            result["prefill_states"] = prefill_states
        if functional_sequence is not None:
            result["summary"]["functional_sequence"] = functional_sequence
        # Preserve physical state exactly; mesh-specific head assembly belongs
        # to a decoder-provided hook, not an assumed host concat dimension.
        restore()
        for step in range(args.decode_steps):
            refresh(step)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        result["states"] = [
            [
                {key: [ttnn.to_torch(t).clone() for t in ttnn.get_device_tensors(state[key])] for key in state_keys}
                for state in group
            ]
            for group in states
        ]
        return result
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


def _state_checks(baseline, candidate, inputs, args):
    """Reconstruct documented rank ownership, including duplicated GQA KV heads."""
    checks = []
    for depth, (base_group, mesh_group) in enumerate(zip(baseline["states"], candidate["states"])):
        for row, (base_state, mesh_state) in enumerate(zip(base_group, mesh_group)):
            for key, base_parts in base_state.items():
                expected, parts = base_parts[0], mesh_state[key]
                if key in ("k", "v"):
                    # TP4 over 16 Q / 2 KV heads: ranks 0,1 own head 0;
                    # ranks 2,3 own head 1. Paired replicas must agree.
                    assert torch.equal(parts[0], parts[1]) and torch.equal(parts[2], parts[3]), key
                    actual = torch.cat([parts[0], parts[2]], dim=1)
                    # Compare logical written cells, so unused zero cache pages
                    # cannot inflate the result. Page order is shared exactly.
                    length = args.length + args.decode_steps
                    logical = inputs["pages"][row]
                    expected = expected[logical].permute(1, 0, 2, 3).flatten(1, 2)[:, :length]
                    actual = actual[logical].permute(1, 0, 2, 3).flatten(1, 2)[:, :length]
                elif key == "conv":
                    # Per-rank packed [q512,k512,v1024], global [q2048,k2048,v4096].
                    split = [torch.split(t, (512, 512, 1024), dim=-1) for t in parts]
                    actual = torch.cat([torch.cat([t[group] for t in split], dim=-1) for group in range(3)], dim=-1)
                else:
                    actual = torch.cat(parts, dim=1)
                checks.append(dict(stack_index=depth, row=row, tensor=key, pcc=pcc(expected, actual)))
    return checks


def run(args):
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record

    if args.profile and os.environ.get("TT_METAL_WATCHER"):
        raise ValueError("watcher and profiler runs must be separate")
    if args.context < args.length + args.decode_steps:
        raise ValueError("context must contain prefill and all decode steps")
    if getattr(args, "functional_sequence", False):
        if (args.length, args.decode_steps, args.batch, args.stack_depth) != (128, 128, 1, 1):
            raise ValueError("FUNCTIONAL requires length128, decode-steps128, batch1 and stack-depth1")
        if args.functional_repeats < 2 or args.profile or args.baseline_only:
            raise ValueError("FUNCTIONAL requires >=2 repeats and unprofiled multichip execution")
        if any(
            os.environ.get(key) not in (None, "", "0")
            for key in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER", "TT_METAL_TRACE_ALLOC_TRACKING")
        ):
            raise ValueError("FUNCTIONAL timing requires watcher, profiler and allocation tracking off")
    if (
        min(args.batch, args.decode_steps, args.repeats, args.prefill_repeats, args.stress_repeats, args.stack_depth)
        < 1
    ):
        raise ValueError("batch, step, repeat and stack counts must be positive")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    record(args.output)
    provenance_path = args.output + ".provenance.json"
    provenance = json.loads(Path(provenance_path).read_text())
    provenance.update(device="Blackhole 1x4 target; optimized baseline 1x1 physical chip 0", decoder="multichip")
    _write(provenance_path, provenance)
    torch.set_num_threads(8)
    c, sd = config(), weights(args.layer)
    signature = {
        k: getattr(args, k)
        for k in (
            "layer",
            "length",
            "context",
            "batch",
            "decode_steps",
            "stack_depth",
            "seed",
            "baseline_build_kwargs",
            "activation_file",
        )
    }
    if args.baseline_load:
        saved = torch.load(args.baseline_load, weights_only=True)
        assert saved["signature"] == signature, "saved baseline does not match requested test"
        inputs, baseline = saved["inputs"], saved["baseline"]
    else:
        inputs = _inputs(args, c)
        baseline = _run_side(args, c, sd, inputs, multichip=False)
        if args.baseline_save:
            Path(args.baseline_save).parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(signature=signature, inputs=inputs, baseline=baseline), args.baseline_save)
    scores = dict(
        **signature,
        activation_source=args.activation_file or "seeded random BF16 std0.05; distinct batch rows and decode steps",
        page_order="independently permuted per row; same tables on baseline and mesh",
        baseline=baseline["summary"],
        trace_allocation_tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1",
    )
    if args.baseline_only:
        _write(args.output, scores)
        return
    mesh_result = _run_side(args, c, sd, inputs, multichip=True)
    scores["multichip"] = mesh_result["summary"]
    scores["prefill_pcc"] = pcc(baseline["prefill"], mesh_result["prefill"])
    scores["decode_pcc"] = [pcc(a, b) for a, b in zip(baseline["decode"], mesh_result["decode"])]
    scores["prefill_per_row_pcc"] = [pcc(a, b) for a, b in zip(baseline["prefill"], mesh_result["prefill"])]
    scores["decode_per_row_pcc"] = [
        [pcc(a, b) for a, b in zip(base_step, mesh_step)]
        for base_step, mesh_step in zip(baseline["decode"], mesh_result["decode"])
    ]
    for mode, key in (("prefill", "prefill_warmed_ms"), ("decode", "decode_traced_warmed_ms")):
        speedup = baseline["summary"][key] / mesh_result["summary"][key]
        scores[mode + "_speedup"] = speedup
        scores[mode + "_efficiency"] = speedup / 4
    tensor_path = str(Path(args.output).with_suffix(".pt"))
    torch.save(dict(signature=signature, inputs=inputs, baseline=baseline, multichip=mesh_result), tensor_path)
    scores["tensor_artifact"] = tensor_path
    scores["state_checks"] = _state_checks(baseline, mesh_result, inputs, args)
    scores["passed"] = (
        scores["prefill_pcc"] >= 0.995
        and min(scores["decode_pcc"]) >= 0.995
        and all(check["pcc"] >= 0.995 for check in scores["state_checks"])
    )
    if getattr(args, "functional_sequence", False):
        scores["passed"] = scores["passed"] and scores["multichip"]["functional_sequence"]["passed"]
    _write(args.output, scores)
    print(json.dumps(scores, indent=2), flush=True)
    assert scores["passed"], "whole-output PCC must be >=0.995; requested FUNCTIONAL floor must pass"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--length", type=int, default=33)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--stack-depth", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--prefill-repeats", type=int, default=3)
    parser.add_argument("--stress-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--trace-region-size", type=int, default=67108864)
    parser.add_argument("--activation-file")
    parser.add_argument("--baseline-save")
    parser.add_argument("--baseline-load")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--baseline-build-kwargs", default="{}")
    parser.add_argument("--mesh-build-kwargs", default="{}")
    parser.add_argument("--mesh-projection-geometry")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--save-prefill-states", action="store_true")
    parser.add_argument("--functional-sequence", action="store_true")
    parser.add_argument("--functional-repeats", type=int, default=3)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
