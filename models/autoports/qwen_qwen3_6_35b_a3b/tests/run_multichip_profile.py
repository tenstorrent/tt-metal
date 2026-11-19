# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Small B32 profiling fixture seeded by a fully accepted paired decoder run.

Only row 0 is prefetched here. All 32 exact post-prefill states come from the
paired seed. One eager decode warms dispatch; two changing-input trace steps
are checked in two reset streams. Broad correctness remains in the seed run.
"""

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def _validate_seed(seed, acceptance, provenance, layer, model_root):
    """Fail before opening hardware on a stale, partial, or incompatible seed."""
    signature = seed["signature"]
    assert signature["layer"] == layer
    assert signature["batch"] == 32 and signature["length"] == 32
    assert signature["stack_depth"] == 1 and signature["decode_steps"] >= 2
    assert signature["context"] >= signature["length"] + signature["decode_steps"]
    assert signature["activation_file"], "real-text seed required"
    assert signature["baseline_build_kwargs"] == "{}"
    assert acceptance["passed"] is True
    for key, value in signature.items():
        assert acceptance[key] == value, f"seed acceptance signature mismatch: {key}"
    assert "--save-prefill-states" in provenance["command"]
    assert "--baseline-load" not in provenance["command"]
    for name in ("multichip_decoder.py", "optimized_decoder.py"):
        key = "tt/" + name
        digest = hashlib.sha256((model_root / key).read_bytes()).hexdigest()
        assert provenance["source_sha256"][key] == digest, f"stale seed source: {key}"
    inputs = seed["inputs"]
    assert tuple(inputs["x"].shape) == (32, 32, 2048)
    assert tuple(inputs["xd"].shape) == (signature["decode_steps"], 32, 1, 2048)
    assert not torch.equal(inputs["xd"][0], inputs["xd"][1]), "decode inputs must change"
    assert len(inputs["pages"]) == 32
    pages = (signature["context"] + 31) // 32
    for order in inputs["pages"]:
        assert torch.equal(order.sort().values, torch.arange(pages)), "invalid seed page permutation"
    for side, rank_count in (("baseline", 1), ("multichip", 4)):
        result = seed[side]
        assert result["summary"] == acceptance[side], "seed summary differs from acceptance"
        assert result["summary"]["runtime_fallback_audit"] == "pass"
        assert result["summary"]["deterministic"] is True
        assert result["summary"]["all_step_eager_trace_equal"] is True
        assert tuple(result["prefill"].shape) == (32, 32, 2048)
        assert tuple(result["decode"].shape) == (signature["decode_steps"], 32, 1, 2048)
        assert len(result["prefill_states"]) == 1 and len(result["prefill_states"][0]) == 32
        expected_keys = {"conv", "recurrent"} if result["summary"]["kind"] == "linear_attention" else {"k", "v"}
        for state in result["prefill_states"][0]:
            assert set(state) == expected_keys
            for parts in state.values():
                assert len(parts) == rank_count
                assert all(part.shape == parts[0].shape and torch.isfinite(part).all() for part in parts)
    assert seed["multichip"]["summary"]["residual_layout"] == "replicated"
    assert acceptance["prefill_pcc"] >= 0.995 and min(acceptance["decode_pcc"]) >= 0.995
    return signature


def run(args):
    # Keep imports below seed validation so validation can be exercised on CPU.
    if os.environ.get("TT_METAL_WATCHER"):
        raise ValueError("watcher and profiling runs must be separate")
    if os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1":
        raise ValueError("this narrow profiler fixture requires trace tracking disabled")
    seed = torch.load(args.seed, map_location="cpu", weights_only=True)
    acceptance_path = Path(args.seed).with_suffix(".json")
    acceptance = json.loads(acceptance_path.read_text())
    assert Path(acceptance["tensor_artifact"]).resolve() == Path(args.seed).resolve()
    seed_provenance = json.loads(Path(str(acceptance_path) + ".provenance.json").read_text())
    signature = _validate_seed(seed, acceptance, seed_provenance, args.layer, Path(__file__).parents[1])
    from tracy import signpost
    from ttnn.tools import trace_allocation_tracker

    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.reference import config, pcc, weights
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.multichip_decoder import MultichipDecoder

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    record(args.output)
    provenance_path = args.output + ".provenance.json"
    provenance = json.loads(Path(provenance_path).read_text())
    provenance.update(
        device="Blackhole 1x4 target; chips 0,1,2,3",
        decoder="multichip",
        seed_artifact=args.seed,
        seed_acceptance=str(acceptance_path),
    )
    _write(provenance_path, provenance)
    torch.set_num_threads(8)
    c, sd = config(), weights(args.layer)
    inputs = seed["inputs"]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=67108864)
    trace = None
    try:
        dec = MultichipDecoder.from_state_dict(sd, hf_config=c, layer_idx=args.layer, mesh_device=mesh)
        assert dec.residual_layout == "replicated"
        assert dec.precision_policy == seed["multichip"]["summary"]["precision_policy"]
        state_keys = ("conv", "recurrent") if dec.kind == "linear_attention" else ("k", "v")
        states = [
            (
                dec.new_recurrent_state()
                if dec.kind == "linear_attention"
                else dec.new_kv_state(context_length=signature["context"], page_order=inputs["pages"][row])
            )
            for row in range(32)
        ]
        dec.prepare_decode_batch(32)
        plan = dec.prepare_prefill(seq_len=32)

        def upload(t):
            return ttnn.from_torch(
                t,
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        replica_checks = 0

        def host(t):
            nonlocal replica_checks
            parts = [ttnn.to_torch(part).clone() for part in ttnn.get_device_tensors(t)]
            assert len(parts) == 4 and all(torch.equal(parts[0], part) for part in parts[1:]), "output replicas differ"
            replica_checks += 1
            return parts[0]

        def host_rows(rows):
            return torch.cat([host(t).reshape(1, 1, c.hidden_size) for t in rows])

        tx = upload(inputs["x"][:1])
        step_inputs = [[upload(inputs["xd"][step, row : row + 1]) for row in range(32)] for step in range(2)]
        stable_inputs = [ttnn.clone(t) for t in step_inputs[0]]
        positions = [
            [
                dec.prepare_positions(start_pos=32 + step, seq_len=1) if dec.kind == "full_attention" else None
                for _ in range(32)
            ]
            for step in range(2)
        ]
        stable_positions = [{key: ttnn.clone(t) for key, t in row.items()} if row else None for row in positions[0]]
        snapshots = []
        for state, saved in zip(states, seed["multichip"]["prefill_states"][0]):
            snapshots.append(
                {
                    key: ttnn.from_torch(
                        torch.cat(saved[key], dim=0),
                        device=mesh,
                        dtype=state[key].dtype,
                        layout=state[key].layout,
                        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
                    )
                    for key in state_keys
                }
            )
        zero_row = {key: ttnn.zeros_like(states[0][key]) for key in state_keys}

        def restore():
            for state, saved in zip(states, snapshots):
                for key in state_keys:
                    ttnn.copy(saved[key], state[key])

        def refresh(step):
            for target, source in zip(stable_inputs, step_inputs[step]):
                ttnn.copy(source, target)
            for target, source in zip(stable_positions, positions[step]):
                if source:
                    for key in source:
                        ttnn.copy(source[key], target[key])

        def decode():
            return dec.decode_batch(stable_inputs, states=states, positions=stable_positions)

        with device_only():
            warm = dec.prefill_forward(tx, state=states[0], plan=plan)
        warm_host = host(warm).reshape(1, 32, c.hidden_size)
        del warm
        for key in state_keys:
            ttnn.copy(zero_row[key], states[0][key])
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
        signpost("PERF_PREFILL")
        with device_only():
            profiled_prefill = dec.prefill_forward(tx, state=states[0], plan=plan)
        ttnn.synchronize_device(mesh)
        signpost("PERF_PREFILL_END")
        ttnn.ReadDeviceProfiler(mesh)
        prefill_host = host(profiled_prefill).reshape(1, 32, c.hidden_size)
        del profiled_prefill
        assert torch.equal(prefill_host, warm_host), "prefill warm/profile differ"
        prefill_pcc = pcc(seed["baseline"]["prefill"][:1], prefill_host)
        assert prefill_pcc >= 0.995

        restore()
        upload_checks = []
        for row, (state, saved) in enumerate(zip(states, seed["multichip"]["prefill_states"][0])):
            for key in state_keys:
                parts = ttnn.get_device_tensors(state[key])
                assert len(parts) == 4
                for rank, (part, expected) in enumerate(zip(parts, saved[key])):
                    assert torch.equal(ttnn.to_torch(part), expected), f"seed state upload changed {row}/{key}/{rank}"
                    upload_checks.append(dict(row=row, tensor=key, rank=rank, bit_equal=True))
        refresh(0)
        with device_only():
            eager = decode()
        eager_host = host_rows(eager)
        del eager
        eager_pcc = pcc(seed["baseline"]["decode"][0], eager_host)
        assert eager_pcc >= 0.995
        ttnn.ReadDeviceProfiler(mesh)
        restore()
        refresh(0)
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        with device_only():
            traced = decode()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        for t in traced:
            trace_allocation_tracker.acknowledge_corruptible(t)
        ttnn.ReadDeviceProfiler(mesh)
        streams = []
        for _ in range(2):
            restore()
            output = []
            for step in range(2):
                refresh(step)
                with device_only():
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                output.append(host_rows(traced))
                ttnn.ReadDeviceProfiler(mesh)
            streams.append(torch.stack(output))
        assert torch.equal(streams[0], streams[1]), "replay streams differ"
        assert torch.equal(streams[0][0], eager_host), "step 0 eager/trace differ"
        seed_mesh_equal = [torch.equal(streams[0][step], seed["multichip"]["decode"][step]) for step in range(2)]
        assert all(seed_mesh_equal), "replay differs from accepted seed mesh outputs"
        decode_pcc = [pcc(seed["baseline"]["decode"][step], streams[0][step]) for step in range(2)]
        assert min(decode_pcc) >= 0.995

        restore()
        refresh(0)
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
        samples = []
        signpost("PERF_DECODE")
        for _ in range(16):
            begin = time.perf_counter()
            with device_only():
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            samples.append((time.perf_counter() - begin) * 1000)
            ttnn.ReadDeviceProfiler(mesh)
        signpost("PERF_DECODE_END")

        result = dict(
            profile_fixture="seeded_b32_v1",
            layer=args.layer,
            length=32,
            context=signature["context"],
            batch=32,
            decode_steps=2,
            stack_depth=1,
            activation_file=signature["activation_file"],
            activation_source=acceptance["activation_source"],
            seed_artifact=args.seed,
            seed_acceptance=str(acceptance_path),
            phase_scope={
                "prefill": "one representative request from batch 32",
                "decode": "one complete batch; not divided by batch size",
            },
            prefill_request_count=1,
            profile_decode_replays=16,
            profile_replay_state_policy="restore once before PERF_DECODE; fixed input/position, evolving state; profiling only",
            prefill_pcc=prefill_pcc,
            decode_pcc=decode_pcc,
            eager_step0_pcc=eager_pcc,
            step0_eager_trace_equal=True,
            replay_seed_mesh_equal=seed_mesh_equal,
            prefill_state_upload_checks=upload_checks,
            trace_allocation_tracking=False,
            multichip=dict(
                decoder="MultichipDecoder",
                kind=dec.kind,
                residual_layout="replicated",
                output_replicas_equal=True,
                output_replica_checks=replica_checks,
                deterministic=True,
                runtime_fallback_audit="pass",
                decode_samples_ms=samples,
                decode_traced_warmed_ms=statistics.median(samples),
                timing_regime="profiled trace execute wall time; profiler drains excluded",
                precision_policy=dec.precision_policy,
                trace_capture_count=1,
                replay_count=4,
                changing_inputs=True,
                positions=[32, 33],
            ),
            passed=True,
        )
        tensor_path = str(Path(args.output).with_suffix(".pt"))
        torch.save(dict(prefill_row0=prefill_host, eager_step0=eager_host, trace_streams=streams), tensor_path)
        result["tensor_artifact"] = tensor_path
        _write(args.output, result)
        print(json.dumps(result, indent=2), flush=True)
    finally:
        try:
            if trace is not None:
                ttnn.release_trace(mesh, trace)
        finally:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, choices=(0, 3), required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
