# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Identical recorded-input optimized/TP4 parity and warmed trace runner."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_8_27b.tests.reference import load_config, load_layer_weights
from models.autoports.qwen_qwen3_8_27b.tests.run_optimized_decoder import device_only, pcc
from models.autoports.qwen_qwen3_8_27b.tt.multichip_decoder import MultichipDecoder
from models.autoports.qwen_qwen3_8_27b.tt.optimized_decoder import OptimizedDecoder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--policy", default="{}")
    p.add_argument("--policy-file", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--compare-dir", type=Path, default=Path("/home/mvasiljevic/qwen38-full-rerun/tt-metal-cache/multichip_compare")
    )
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--prefill-repeats", type=int, default=5)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--trace-prefill", action="store_true")
    p.add_argument("--projection-bench", action="store_true")
    p.add_argument("--queued-timing", action="store_true")
    p.add_argument("--cache-pcc-diagnostic", action="store_true")
    p.add_argument("--continuation", action="store_true")
    p.add_argument("--stack", action="store_true")
    p.add_argument("--prefill-only", action="store_true")
    p.add_argument("--capacity", action="store_true")
    p.add_argument(
        "--capacity-plan",
        type=Path,
        default=Path("models/autoports/qwen_qwen3_8_27b/doc/multichip_decoder/memory_capacity_plan.json"),
    )
    p.add_argument("--stress-iterations", type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(8)
    snapshot = (
        "/home/mvasiljevic/hf-cache/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    )
    config = load_config(snapshot)
    weights = load_layer_weights(snapshot, args.layer)
    recorded = torch.load(
        Path("/home/mvasiljevic/qwen38-full-rerun/tt-metal-cache/optimized_decoder_activations_long")
        / f"layer{args.layer}.pt",
        weights_only=True,
    )
    n, b = args.length, args.batch
    x = recorded[torch.arange(b) % recorded.shape[0]][:, torch.arange(n + 2) % recorded.shape[1]].clone()
    rope = Qwen3_5TextRotaryEmbedding(config)
    cos, sin = rope(x, torch.arange(n + 2)[None].expand(b, -1))
    torch.manual_seed(727)
    per_user = (min(n + 2, config.max_position_embeddings) + 31) // 32
    table = torch.randperm(per_user * b + 3)[: per_user * b].reshape(b, per_user).int()
    policy = json.loads(args.policy_file.read_text() if args.policy_file else args.policy)
    if not args.baseline:
        ttnn.set_fabric_config(
            ttnn.FabricConfig.FABRIC_1D_RING if policy.get("ring", True) else ttnn.FabricConfig.FABRIC_1D
        )
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1 if args.baseline else 4),
        physical_device_ids=[3] if args.baseline else [],
        trace_region_size=40000000,
    )
    trace = None
    prefill_trace = None
    source_hash = hashlib.sha256(
        Path(
            (OptimizedDecoder if args.baseline else MultichipDecoder).__module__.replace(".", "/") + ".py"
        ).read_bytes()
    ).hexdigest()
    report = dict(
        layer=args.layer,
        batch=b,
        length=n,
        baseline=args.baseline,
        policy=policy,
        stack=args.stack,
        continuation=args.continuation,
    )
    try:
        decoder = (OptimizedDecoder if args.baseline else MultichipDecoder).from_state_dict(
            weights,
            hf_config=config,
            layer_idx=args.layer,
            mesh_device=mesh,
            **({} if args.baseline else {"policy": policy}),
        )
        report["effective_policy"] = decoder.policy
        report["mesh_shape"] = list(mesh.shape)

        next_decoder = None
        if args.stack:
            assert args.layer == 0, "Stack fixture is linear then full attention"
            next_decoder = (OptimizedDecoder if args.baseline else MultichipDecoder).from_state_dict(
                load_layer_weights(snapshot, 3),
                hf_config=config,
                layer_idx=3,
                mesh_device=mesh,
                **({} if args.baseline else {"policy": policy, "ccl": decoder.ccl}),
            )
            report["next_layer_effective_policy"] = next_decoder.policy
            if not args.baseline and decoder.policy.get("direct_allreduce"):
                assert next_decoder.allreduce_buffer is decoder.allreduce_buffer
                report["stack_shared_collective_workspace"] = True

        def upload(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, shard=False):
            return ttnn.from_torch(
                t.contiguous(),
                dtype=dtype,
                layout=layout,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=(
                    None
                    if args.baseline
                    else ttnn.ShardTensorToMesh(mesh, dim=-1)
                    if shard
                    else ttnn.ReplicateTensorToMesh(mesh)
                ),
            )

        sharded = not args.baseline and decoder.sharded_residual

        def host(t, dim=None):
            if args.baseline:
                return ttnn.to_torch(t)
            if dim is not None:
                return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=dim))
            parts = [ttnn.to_torch(v) for v in ttnn.get_device_tensors(t)]
            for part in parts[1:]:
                assert pcc(parts[0], part) > 0.99999, "replicated outputs differ"
            return parts[0]

        state = decoder.allocate_state(batch_size=b, num_pages=per_user * b + 3)
        inputs = upload(x[:, :n], shard=sharded)
        cc, ss = upload(cos[:, :n]), upload(sin[:, :n])
        pt = upload(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        pos = upload(torch.full((b,), n, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        dx = upload(x[:, n : n + 1], shard=sharded)
        dc, ds = upload(cos[:, n : n + 1]), upload(sin[:, n : n + 1])
        reservation = None
        if args.capacity and not args.baseline:
            plan = json.loads(args.capacity_plan.read_text())
            reserve_bytes = plan["capacity_probe_reserved_bytes_per_device"]
            remaining_tiles = (reserve_bytes + 1087) // 1088
            reservation = []
            while remaining_tiles:
                tiles = min(remaining_tiles, 1024**3 // 1088)
                reservation.append(
                    ttnn.empty(
                        [1, 1, tiles * 32, 32],
                        dtype=ttnn.bfloat8_b,
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                )
                remaining_tiles -= tiles
            report["capacity_reservation_bytes_per_device"] = ((reserve_bytes + 1087) // 1088) * 1088
            report["capacity_reservation_slabs"] = len(reservation)
        states = [state]
        if next_decoder:
            states.append(next_decoder.allocate_state(batch_size=b, num_pages=per_user * b + 3))
        tensors = {f"{i}.{k}": v for i, st in enumerate(states) for k, v in vars(st).items() if v is not None}
        initial = {k: ttnn.clone(v) for k, v in tensors.items()}

        def restore(saved):
            for k, v in saved.items():
                ttnn.copy(v, tensors[k])

        positions = upload(torch.arange(n)[:, None].expand(n, b).int(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

        def single_prefill():
            with device_only():
                if args.continuation and n > 1:
                    cut = min(31, n - 1)
                    first = decoder.prefill_forward(
                        inputs[:, :cut], state=state, cos=cc[:, :cut], sin=ss[:, :cut], page_table=pt
                    )
                    second = decoder.prefill_forward(
                        inputs[:, cut:],
                        state=state,
                        start_pos=cut,
                        cos=cc[:, cut:],
                        sin=ss[:, cut:],
                        page_table=pt,
                        positions=positions[cut:],
                    )
                    return ttnn.concat(
                        [
                            ttnn.to_memory_config(first, ttnn.DRAM_MEMORY_CONFIG),
                            ttnn.to_memory_config(second, ttnn.DRAM_MEMORY_CONFIG),
                        ],
                        dim=1,
                    )
                return decoder.prefill_forward(inputs, state=state, cos=cc, sin=ss, page_table=pt)

        def prefill():
            result = single_prefill()
            if next_decoder:
                with device_only():
                    result = next_decoder.prefill_forward(result, state=states[1], cos=cc, sin=ss, page_table=pt)
            return result

        def decode():
            with device_only():
                result = decoder.decode_forward(dx, state=state, cos=dc, sin=ds, page_table=pt, current_pos=pos)
                if next_decoder:
                    result = next_decoder.decode_forward(
                        result, state=states[1], cos=dc, sin=ds, page_table=pt, current_pos=pos
                    )
                return result

        print("PREFILL_BEGIN", flush=True)
        result = prefill()
        pre = host(result, -1 if sharded else None)
        del result
        prefix = {k: ttnn.clone(v) for k, v in tensors.items()}

        def state_host():
            values = {}
            for index, current_state in enumerate(states):
                for name, tensor in vars(current_state).items():
                    if tensor is None:
                        continue
                    key = f"layer{index}.{name}"
                    if args.baseline:
                        values[key] = host(tensor)
                    elif name == "conv":
                        parts = [ttnn.to_torch(v) for v in ttnn.get_device_tensors(tensor)]
                        values[key] = torch.cat(
                            [
                                torch.cat([part[..., a:z] for part in parts], dim=-1)
                                for a, z in ((0, 512), (512, 1024), (1024, 2560))
                            ],
                            dim=-1,
                        )
                    else:
                        values[key] = host(tensor, 1)
            mapped = set(table.flatten().tolist())
            unused = sorted(set(range(per_user * b + 3)) - mapped)
            for key, value in values.items():
                if key.endswith((".key", ".value")):
                    assert torch.count_nonzero(value[unused]) == 0, f"Unowned cache pages modified: {key}"
            if n > 4097:
                sampled_pages = torch.tensor(sorted(set(table[:, [0, -2, -1]].flatten().tolist() + unused)))
                values = {
                    key: value[sampled_pages] if key.endswith((".key", ".value")) else value
                    for key, value in values.items()
                }
            return values

        state_outputs = state_host()
        print("PREFILL_DONE", flush=True)
        if args.capacity:
            view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
            report["memory_view"] = {
                "banks": view.num_banks,
                "bytes_per_bank": view.total_bytes_per_bank,
                "allocated_bytes_per_bank": view.total_bytes_allocated_per_bank,
                "scope": "Representative allocator view after prefill output release, not peak usage; reservation plus decoder, inputs and cache remain live.",
            }
        if args.prefill_only:
            args.compare_dir.mkdir(parents=True, exist_ok=True)
            fixture = args.compare_dir / f"l{args.layer}_b{b}_s{n}_prefill_only.pt"
            if args.baseline:
                torch.save(pre, fixture)
            else:
                baseline = torch.load(fixture, weights_only=True)
                report["pcc"] = {"prefill": pcc(baseline, pre)}
                assert report["pcc"]["prefill"] >= 0.995
            report["source_sha256"] = source_hash
            report["effective_policy"] = decoder.policy
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)
            return
        times = []
        for _ in range(1 if args.capacity or args.profile else args.prefill_repeats):
            restore(initial)
            ttnn.synchronize_device(mesh)
            start = time.perf_counter()
            result = prefill()
            ttnn.synchronize_device(mesh)
            times.append((time.perf_counter() - start) * 1000)
            del result
        report["prefill_ms"] = statistics.median(times)
        report["prefill_samples_ms"] = times
        if args.trace_prefill:
            original_prefill_input = ttnn.clone(inputs)
            changed_prefill_input = ttnn.neg(inputs)
            restore(initial)
            warmed = prefill()
            ttnn.synchronize_device(mesh)
            del warmed
            restore(initial)
            ttnn.synchronize_device(mesh)
            prefill_trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            traced_prefill = prefill()
            ttnn.end_trace_capture(mesh, prefill_trace, cq_id=0)
            traced_times = []
            for _ in range(1 if args.profile else args.prefill_repeats):
                restore(initial)
                ttnn.synchronize_device(mesh)
                start = time.perf_counter()
                ttnn.execute_trace(mesh, prefill_trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                traced_times.append((time.perf_counter() - start) * 1000)
                assert torch.equal(pre, host(traced_prefill, -1 if sharded else None))
            for key, value in state_host().items():
                assert torch.equal(state_outputs[key], value), f"Prefill trace state mismatch: {key}"
            ttnn.copy(changed_prefill_input, inputs)
            restore(initial)
            changed_prefill = prefill()
            changed_pre = host(changed_prefill, -1 if sharded else None)
            changed_pre_state = state_host()
            del changed_prefill
            assert not torch.equal(pre, changed_pre), "Changed prefill inputs ignored"
            restore(initial)
            ttnn.execute_trace(mesh, prefill_trace, cq_id=0, blocking=True)
            assert torch.equal(changed_pre, host(traced_prefill, -1 if sharded else None))
            for key, value in state_host().items():
                assert torch.equal(changed_pre_state[key], value), f"Changed prefill state mismatch: {key}"
            ttnn.copy(original_prefill_input, inputs)
            report["traced_prefill_ms"] = statistics.median(traced_times)
            report["traced_prefill_samples_ms"] = traced_times
            report["prefill_trace_bitwise_equal"] = True
            report["changed_prefill_trace_bitwise_equal"] = True
            report[
                "prefill_trace_contract"
            ] = "Same device-only TP4 forward; stable input/state/page-table addresses; input/state refresh and readback outside trace. Eager prefill remains the primary before/after metric."
        if args.profile:
            from tracy import signpost

            restore(initial)
            ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)
            signpost("PERF_PREFILL")
            if args.trace_prefill:
                ttnn.execute_trace(mesh, prefill_trace, cq_id=0, blocking=False)
            else:
                profiled = prefill()
            report["profile_prefill_mode"] = "trace" if args.trace_prefill else "eager"
            ttnn.synchronize_device(mesh)
            signpost("PERF_PREFILL_END")
            ttnn.ReadDeviceProfiler(mesh)
        if prefill_trace is not None:
            ttnn.release_trace(mesh, prefill_trace)
            prefill_trace = None
            del traced_prefill, original_prefill_input, changed_prefill_input
        restore(prefix)
        projection_inputs = {}
        original_linear = decoder._linear
        if args.projection_bench:

            def record_projection(x, name, activation=None, keep_sharded=False):
                projection_inputs[name] = ttnn.clone(x)
                return original_linear(x, name, activation=activation, keep_sharded=keep_sharded)

            decoder._linear = record_projection
        expected = host(decode(), -1 if sharded else None)
        if args.projection_bench:
            decoder._linear = original_linear
            report["projection_bench"] = {}
            for name, activation_input in projection_inputs.items():

                def projection():
                    return OptimizedDecoder._linear(decoder, activation_input, name, keep_sharded=True)

                eager = projection()
                oracle = host(eager, -1)
                del eager
                projection_trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                projected = projection()
                ttnn.end_trace_capture(mesh, projection_trace, cq_id=0)
                samples = []
                try:
                    for _ in range(5):
                        ttnn.synchronize_device(mesh)
                        started = time.perf_counter()
                        for _ in range(20):
                            ttnn.execute_trace(mesh, projection_trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        samples.append((time.perf_counter() - started) * 50)
                    assert torch.equal(oracle, host(projected, -1)), name
                    weight = decoder.dram_weights[name + ".weight"]
                    role = decoder._role(name)
                    report["projection_bench"][name] = {
                        "ms": statistics.median(samples),
                        "samples_ms": samples,
                        "input_shape": list(activation_input.shape),
                        "weight_shape": list(weight.shape),
                        "weight_dtype": str(weight.dtype),
                        "weight_memory": str(weight.memory_config()),
                        "input_memory": str(activation_input.memory_config()),
                        "output_memory": str(projected.memory_config()),
                        "readers": decoder.policy[role + "_readers"],
                        "cores": decoder.policy[role + "_cores"],
                        "in0_block_w": decoder.policy[role + "_block"],
                        "fidelity": decoder.policy[role + "_fidelity"],
                        "fp32_dest_acc_en": decoder.projection_configs[role].fp32_dest_acc_en,
                        "trace_bitwise_equal": True,
                        "input_source": "cloned real checkpoint decoder activation before projection",
                        "timing_scope": "isolated projection and its input/output layout operations; 20 queued trace replays",
                    }
                finally:
                    ttnn.release_trace(mesh, projection_trace)
                del projected
            projection_inputs.clear()
        last_position_state = state_host() if args.capacity else {}
        changed_inputs = [
            upload(x[:, n + 1 : n + 2], shard=sharded),
            upload(cos[:, n + 1 : n + 2]),
            upload(sin[:, n + 1 : n + 2]),
            upload(
                (torch.full((b,), n - 1, dtype=torch.int32) - torch.arange(b, dtype=torch.int32) % min(n, 4)),
                ttnn.int32,
                ttnn.ROW_MAJOR_LAYOUT,
            ),
        ]
        changed_table = torch.roll(table, 1, dims=-1) if n <= 4097 else table
        changed_inputs.append(upload(changed_table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT))
        changed_prefix = dict(prefix)
        changed_prefix_host = dict(state_outputs)
        if n <= 4097:
            # Remap physical storage with the table, preserving valid logical history.
            for index, current_state in enumerate(states):
                for name in ("key", "value"):
                    key = f"layer{index}.{name}"
                    if key not in state_outputs:
                        continue
                    original = state_outputs[key]
                    remapped = original.clone()
                    remapped[changed_table.flatten().long()] = original[table.flatten().long()]
                    changed_prefix_host[key] = remapped
                    changed_prefix[f"{index}.{name}"] = ttnn.from_torch(
                        remapped,
                        dtype=getattr(current_state, name).dtype,
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=None if args.baseline else ttnn.ShardTensorToMesh(mesh, dim=1),
                    )
        original_inputs = [ttnn.clone(v) for v in (dx, dc, ds, pos, pt)]

        def refresh(values):
            for source, dest in zip(values, (dx, dc, ds, pos, pt)):
                ttnn.copy(source, dest)

        refresh(changed_inputs)
        restore(changed_prefix)
        changed_expected = host(decode(), -1 if sharded else None)
        changed_state = state_host()
        if n <= 4097:
            changed_positions = n - 1 - torch.arange(b) % min(n, 4)
            changed_table = torch.roll(table, 1, dims=-1)
            allowed = torch.zeros(per_user * b + 3, 1, 32, 1, dtype=torch.bool)
            for user, position in enumerate(changed_positions.tolist()):
                allowed[changed_table[user, position // 32], :, position % 32, :] = True
            for key, value in changed_state.items():
                if key.endswith((".key", ".value")):
                    mask = ~allowed.expand_as(value)
                    assert torch.equal(value[mask], changed_prefix_host[key][mask]), f"Unexpected cache write: {key}"
            report["cache_write_ownership"] = "passed: changed page-table/current-position rows only"
        refresh(original_inputs)
        restore(prefix)
        if args.stress_iterations:
            # Compute the eager oracle before capture reserves intermediate L1
            # allocations; release each temporary before the next invocation.
            for iteration in range(args.stress_iterations):
                stress_output = decode()
                if iteration + 1 < args.stress_iterations:
                    del stress_output
            stress_expected = host(stress_output, -1 if sharded else None)
            stress_state = state_host()
            del stress_output
            restore(prefix)
        decode()
        if args.profile:
            ttnn.ReadDeviceProfiler(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        traced = decode()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        if args.profile:
            ttnn.ReadDeviceProfiler(mesh)
        times = []
        for _ in range(1 if args.profile else args.repeats):
            restore(prefix)
            ttnn.synchronize_device(mesh)
            start = time.perf_counter()
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            times.append((time.perf_counter() - start) * 1000)
            actual = host(traced, -1 if sharded else None)
            assert torch.equal(expected, actual), "trace differs from eager"
            if args.profile:
                ttnn.ReadDeviceProfiler(mesh)
        report["decode_ms"] = statistics.median(times)
        report["decode_samples_ms"] = times
        report["trace_bitwise_equal"] = True
        refresh(changed_inputs)
        restore(changed_prefix)
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        changed_actual = host(traced, -1 if sharded else None)
        assert torch.equal(changed_expected, changed_actual), "changed-input replay mismatch"
        assert not torch.equal(expected, changed_actual), "changed inputs ignored"
        for key, value in (state_host()).items():
            assert torch.equal(changed_state[key], value), f"Changed state replay mismatch: {key}"
        report["changed_input_replay_bitwise_equal"] = True
        report["runtime_fallback_audit"] = "passed: NoTorch plus forbidden host conversions during every forward"
        report["post_decode_state_bitwise_equal"] = True
        if args.queued_timing:
            # Supplement the inherited one-replay latency with amortized dispatch.
            # Evolving-state equality is checked independently by queued stress.
            queued_times = []
            refresh(original_inputs)
            for _ in range(5):
                restore(prefix)
                ttnn.synchronize_device(mesh)
                start = time.perf_counter()
                for _ in range(20):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                queued_times.append((time.perf_counter() - start) * 50)
            report["queued_decode_ms"] = statistics.median(queued_times)
            report["queued_decode_samples_ms"] = queued_times
            refresh(changed_inputs)
            restore(changed_prefix)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        if args.stress_iterations:
            # Serial model execution with evolving state and no host barrier
            # between invocations exercises shared collective workspace reuse.
            refresh(original_inputs)
            restore(prefix)
            for _ in range(args.stress_iterations):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            assert torch.equal(stress_expected, host(traced, -1 if sharded else None))
            for key, value in state_host().items():
                assert torch.equal(stress_state[key], value), f"Queued stress state mismatch: {key}"
            report["queued_stress_iterations"] = args.stress_iterations
            report["queued_stress_bitwise_equal"] = True
        if args.profile:
            from tracy import signpost

            restore(changed_prefix)
            ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)
            signpost("PERF_DECODE")
            profile_start = time.perf_counter()
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            report["signposted_decode_host_ms"] = (time.perf_counter() - profile_start) * 1000
            signpost("PERF_DECODE_END")
            ttnn.ReadDeviceProfiler(mesh)

        def logical_state(values, active_table, valid_tokens=n):
            if n > 4097:
                mapped = set(table.flatten().tolist())
                unused = sorted(set(range(per_user * b + 3)) - mapped)
                sampled = sorted(set(table[:, [0, -2, -1]].flatten().tolist() + unused))
                values = {k: v.clone() for k, v in values.items()}
                for key, value in values.items():
                    if key.endswith((".key", ".value")):
                        for user in range(b):
                            for virtual_page, physical_page in enumerate(active_table[user].tolist()):
                                if physical_page not in sampled:
                                    continue
                                valid_rows = max(0, min(32, valid_tokens - virtual_page * 32))
                                value[sampled.index(physical_page), :, valid_rows:] = 0
                return values
            valid = torch.zeros(per_user * b + 3, 1, 32, 1, dtype=torch.bool)
            for user in range(b):
                for position in range(valid_tokens):
                    valid[active_table[user, position // 32], :, position % 32, :] = True
            return {
                key: value.masked_fill(~valid.expand_as(value), 0) if key.endswith((".key", ".value")) else value
                for key, value in values.items()
            }

        state_outputs = logical_state(state_outputs, table)
        changed_state = logical_state(changed_state, changed_table)
        report["cache_comparison_scope"] = (
            "Sampled first/end/unowned pages, valid prefix rows only; complete outputs compared."
            if n > 4097
            else "All logical prefix rows; padded future rows excluded. Raw replay and write-ownership are exact."
        )
        outputs = {
            "prefill": pre,
            "decode": expected,
            "changed_decode": changed_expected,
            **state_outputs,
            **{"last_position." + k: v for k, v in logical_state(last_position_state, table, n + 1).items()},
            **{"changed." + k: v for k, v in changed_state.items()},
        }
        if args.stress_iterations:
            outputs["stress_decode"] = stress_expected
            outputs.update({"stress." + k: v for k, v in logical_state(stress_state, table, n + 1).items()})
        report["finite_outputs"] = {k: bool(torch.isfinite(v).all()) for k, v in outputs.items()}
        if not all(report["finite_outputs"].values()):
            report["source_sha256"] = source_hash
            report["effective_policy"] = decoder.policy
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            raise AssertionError(report["finite_outputs"])
        args.compare_dir.mkdir(parents=True, exist_ok=True)
        fixture = args.compare_dir / f"l{args.layer}_b{b}_s{n}_stack{int(args.stack)}_c{int(args.continuation)}.pt"
        if args.baseline:
            torch.save(outputs, fixture)
        else:
            base = torch.load(fixture, weights_only=True)
            report["pcc"] = {k: pcc(base[k], v) for k, v in outputs.items()}
            report["per_user_pcc"] = {
                k: [pcc(base[k][i], v[i]) for i in range(b)]
                for k, v in outputs.items()
                if k in ("prefill", "decode", "changed_decode", "stress_decode")
            }
            if min(report["pcc"].values()) < 0.995:
                torch.save(outputs, args.compare_dir / (args.output.stem + "_failed.pt"))
                report["source_sha256"] = source_hash
                report["effective_policy"] = decoder.policy
                args.output.write_text(json.dumps(report, indent=2) + "\n")
            checked_pcc = {
                key: value
                for key, value in report["pcc"].items()
                if not (args.cache_pcc_diagnostic and key.endswith((".key", ".value")))
            }
            if args.cache_pcc_diagnostic:
                report["cache_pcc_diagnostic_only"] = True
            assert min(checked_pcc.values()) >= 0.995, report
            assert min(v for values in report["per_user_pcc"].values() for v in values) >= 0.995, report
        report["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report["source_sha256"] = source_hash
        report["effective_policy"] = decoder.policy
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        if prefill_trace is not None:
            ttnn.release_trace(mesh, prefill_trace)
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
