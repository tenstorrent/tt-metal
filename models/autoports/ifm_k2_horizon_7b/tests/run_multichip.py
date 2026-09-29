"""Serialized optimized-single-chip versus TP4 real-weight layer comparison."""

import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from tracy import signpost

import ttnn

from ..tt.multichip_decoder import MultichipDecoder
from ..tt.optimized_decoder import OptimizedDecoder
from .run_functional import load_reference, pcc
from .sweep_optimized import real_activations


def upload(x, mesh, *, shard=None, integer=False):
    return ttnn.from_torch(
        x.contiguous(),
        device=mesh,
        dtype=ttnn.int32 if integer else ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT if integer else ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=shard) if shard is not None else ttnn.ReplicateTensorToMesh(mesh),
    )


def read(x, mesh, shard=None):
    if mesh.get_num_devices() == 1:
        return ttnn.to_torch(x)
    if shard is not None:
        return ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=shard))
    return ttnn.to_torch(ttnn.get_device_tensors(x)[0])


def refresh(dst, x, mesh, *, shard=None, integer=False):
    host = ttnn.from_torch(
        x.contiguous(),
        dtype=ttnn.int32 if integer else ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT if integer else ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=shard) if shard is not None else ttnn.ReplicateTensorToMesh(mesh),
    )
    ttnn.copy_host_to_device_tensor(host, dst)


@torch.no_grad()
def run(args):
    torch.set_num_threads(16)
    from .multichip_candidates import policy_for

    runtime_hash = hashlib.sha256(
        Path("models/autoports/ifm_k2_horizon_7b/tt/multichip_decoder.py").read_bytes()
    ).hexdigest()
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    baseline_hash = hashlib.sha256(
        Path("models/autoports/ifm_k2_horizon_7b/tt/optimized_decoder.py").read_bytes()
    ).hexdigest()
    candidate_hash = hashlib.sha256(Path(__file__).with_name("multichip_candidates.py").read_bytes()).hexdigest()
    stage5_candidate_hash = hashlib.sha256(
        Path(__file__).with_name("optimized_multichip_candidates.py").read_bytes()
    ).hexdigest()
    if args.stack > 1 and args.candidate:
        raise ValueError(
            "Candidate experiments currently support one layer; stack tests exercise the actual implementation."
        )
    if args.hf_check and (args.stack != 1 or args.remap or args.heterogeneous):
        raise ValueError("Direct HF control requires one layer and a sequential, unmapped logical cache stream")
    config, state, hf, rope_fn = load_reference()
    if args.stack > 1:
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        from .run_functional import MODEL, REVISION

        index = json.loads(Path(hf_hub_download(MODEL, "model.safetensors.index.json", revision=REVISION)).read_text())
        wanted = [f"model.layers.{i}." for i in range(1, args.stack)]
        files = {v for k, v in index["weight_map"].items() if any(k.startswith(p) for p in wanted)}
        for filename in files:
            with safe_open(hf_hub_download(MODEL, filename, revision=REVISION), framework="pt") as f:
                state.update({k: f.get_tensor(k) for k in f.keys() if any(k.startswith(p) for p in wanted)})
    seq, batch = args.seq, args.batch
    inputs = real_activations(batch * (seq + args.steps)).reshape(batch, seq + args.steps, 4096)
    capacity = math.ceil((seq + args.steps) / 32) * 32
    pages = capacity // 32
    torch.manual_seed(123)
    table = torch.randperm(batch * pages).reshape(batch, pages).int()
    outputs = {}
    cache_controls = {}
    measurements = {}
    records = {}
    for multi in (False, True):
        label = "multichip" if multi else "baseline"
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING if multi else ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(1, 4) if multi else ttnn.MeshShape(1, 1),
            **({} if multi else {"physical_device_ids": [0]}),
            trace_region_size=0,
        )
        trace = None
        try:
            layer_kwargs = (
                {
                    "residual_sharded": not args.replicated,
                    "ccl_dtype": args.ccl_dtype,
                    "num_links": args.links,
                    "policy": policy_for(args.policy) if args.policy else None,
                }
                if multi
                else {}
            )
            if multi:
                from ..tt.collective_buffers import DecodeCollectiveBuffers

                layer_kwargs["collective_buffers"] = DecodeCollectiveBuffers(mesh)
            layer = (MultichipDecoder if multi else OptimizedDecoder).from_state_dict(
                state, hf_config=config, layer_idx=0, mesh_device=mesh, **layer_kwargs
            )
            if multi and args.slice_q:
                original_qk = layer._decode_qk

                def corrected_qk(q, k, rope):
                    q, k = original_qk(q, k, rope)
                    print("QK_LOGICAL_SHAPES", q.shape, k.shape, flush=True)
                    return q[:, :, :8, :], k

                layer._decode_qk = corrected_qk
            if multi and args.candidate:
                from .multichip_candidates import configure

                configure(layer, args.candidate, state)
            from .runtime_audit import instrument

            layers = [layer]
            for idx in range(1, args.stack):
                layers.append(
                    (MultichipDecoder if multi else OptimizedDecoder).from_state_dict(
                        state, hf_config=config, layer_idx=idx, mesh_device=mesh, **layer_kwargs
                    )
                )
            audit_counts = [instrument(obj) for obj in layers]
            rs = -1 if multi and not args.replicated else None
            tt_table = upload(table, mesh, integer=True)
            all_caches = [
                tuple(
                    ttnn.zeros(
                        (batch * pages, 2 if multi else 8, 32, 128),
                        device=mesh,
                        dtype=obj.kv_dtype,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    for _ in range(2)
                )
                for obj in layers
            ]

            def forward_prefill(value, rope, plan):
                for obj, cache in zip(layers, all_caches):
                    value = obj.prefill_forward(value, rope=rope, kv_cache=cache, page_table=tt_table, plan=plan)
                return value

            x = inputs[:, :seq]
            rope = rope_fn(x, torch.arange(seq).expand(batch, -1))
            tx = upload(x.unsqueeze(0), mesh, shard=rs)
            tr = tuple(upload(r.unsqueeze(1), mesh) for r in rope)
            plan = layer.prepare_prefill(seq_len=seq)
            if args.split:
                split = args.split
                first = layer.prepare_prefill(seq_len=split)
                second = layer.prepare_prefill(seq_len=seq - split, start_pos=split)

                def prefill():
                    a = forward_prefill(tx[:, :, :split, :], tuple(r[:, :, :split, :] for r in tr), first)
                    b = forward_prefill(tx[:, :, split:, :], tuple(r[:, :, split:, :] for r in tr), second)
                    return ttnn.concat([a, b], dim=2)

            else:
                prefill = lambda: forward_prefill(tx, tr, plan)
            result = prefill()
            ttnn.synchronize_device(mesh)
            signpost(label.upper() + "_PERF_PREFILL")
            t0 = time.perf_counter()
            result = prefill()
            ttnn.synchronize_device(mesh)
            prefill_us = (time.perf_counter() - t0) * 1e6
            signpost(label.upper() + "_PERF_PREFILL_END")
            values = [read(result, mesh, rs)]
            if multi:
                print("PREFILL_PCC", pcc(values[0], outputs["baseline"][0]), flush=True)
            pc = []
            repeated = True
            decode_times = []
            for step in range(args.steps):
                pos = torch.full((batch,), seq + step, dtype=torch.int32)
                if args.heterogeneous:
                    pos -= torch.arange(batch, dtype=torch.int32) % 2
                dx = inputs[:, seq + step : seq + step + 1]
                rr = rope_fn(dx, pos[:, None])
                packed = tuple(r.unsqueeze(0).expand(1, batch, 32, 128) for r in rr)
                if step == 0:
                    td = upload(dx.unsqueeze(0), mesh, shard=rs)
                    # decode layout [1,1,B,H]
                    if batch > 1:
                        td = upload(dx.transpose(0, 1).unsqueeze(0), mesh, shard=rs)
                    dr = tuple(upload(r, mesh) for r in packed)
                    tp = upload(pos, mesh, integer=True)

                    def decode():
                        value = td
                        for obj, cache in zip(layers, all_caches):
                            value = obj.decode_forward(
                                value, rope=dr, kv_cache=cache, page_table=tt_table, current_pos=tp
                            )
                        return value

                    dout = decode()
                    ttnn.synchronize_device(mesh)
                    ccl_keys_before_capture = set(layer.collective_buffers.tensors) if multi else set()
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    dout = decode()
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    if multi:
                        assert set(layer.collective_buffers.tensors) == ccl_keys_before_capture
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                else:
                    if args.remap:
                        mapping = table.flip(0) if step % 2 else table
                        refresh(tt_table, mapping, mesh, integer=True)
                    refresh(td, dx.transpose(0, 1).unsqueeze(0), mesh, shard=rs)
                    for dst, src in zip(dr, packed):
                        refresh(dst, src, mesh)
                    refresh(tp, pos, mesh, integer=True)
                signpost(f"{label.upper()}_PERF_DECODE_{step:03d}")
                t0 = time.perf_counter()
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                decode_times.append((time.perf_counter() - t0) * 1e6)
                signpost(f"{label.upper()}_PERF_DECODE_{step:03d}_END")
                actual = read(dout, mesh, rs)
                values.append(actual)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                repeated &= torch.equal(actual, read(dout, mesh, rs))
            outputs[label] = values
            cache_controls[label] = [read(c, mesh, 1 if multi else None) for cache in all_caches for c in cache]
            measurements[label] = {
                "prefill_host_us": prefill_us,
                "decode_trace_host_mean_us": sum(decode_times) / len(decode_times),
                "decode_steps": args.steps,
                "replay_bitwise": repeated,
                "mesh_ids": mesh.get_device_ids(),
                "runtime_audit": audit_counts,
                "updated_input_outputs_changed": all(not torch.equal(a, b) for a, b in zip(values[1:], values[2:])),
                "orchestration": {
                    "timed_trace_submissions": args.steps,
                    "timed_synchronizations": args.steps,
                    "readbacks_inside_timed_windows": 0,
                    "validation_input_refreshes": args.steps - 1,
                    "validation_position_refreshes": args.steps - 1,
                    "validation_rope_refreshes": 2 * (args.steps - 1),
                    "validation_page_table_refreshes": args.steps - 1 if args.remap else 0,
                    "scope": "Teacher-forced layer validation refreshes occur outside timed windows; no generator/sampling claim",
                },
            }
            if multi:
                measurements[label]["weight_allocations"] = layer.weight_allocations
                measurements[label]["collective_buffer_pool"] = layer.collective_buffers.inventory()
                measurements[label]["shared_collective_pool"] = all(
                    obj.collective_buffers is layer.collective_buffers for obj in layers
                )
                measurements[label]["collective_allocations_during_capture"] = 0
            if args.burst_replays:
                expected = read(dout, mesh, rs)
                signpost(label.upper() + "_PERF_BURST")
                t0 = time.perf_counter()
                for _ in range(args.burst_replays):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                measurements[label]["stationary_burst_host_us"] = (time.perf_counter() - t0) * 1e6 / args.burst_replays
                signpost(label.upper() + "_PERF_BURST_END")
                measurements[label]["stationary_burst_replays"] = args.burst_replays
                measurements[label]["stationary_burst_bitwise"] = torch.equal(expected, read(dout, mesh, rs))
                assert measurements[label]["stationary_burst_bitwise"]
            print(label, measurements[label], flush=True)
            if trace is not None:
                ttnn.release_trace(mesh, trace)
        finally:
            ttnn.close_mesh_device(mesh)
        gc.collect()
    pp = pcc(outputs["multichip"][0], outputs["baseline"][0])
    dp = [pcc(a, b) for a, b in zip(outputs["multichip"][1:], outputs["baseline"][1:])]
    kp = [pcc(a, b) for a, b in zip(cache_controls["multichip"], cache_controls["baseline"])]
    head_kp = [
        [pcc(a[:, h], b[:, h]) for h in range(8)]
        for a, b in zip(cache_controls["multichip"], cache_controls["baseline"])
    ]
    from ..tt.optimized_decoder import PrecisionPolicy

    compared_policy = policy_for(args.policy) if args.policy else PrecisionPolicy()
    cache_projection_changed = compared_policy.attention != PrecisionPolicy().attention
    hf_accuracy = None
    if args.hf_check:
        from transformers import DynamicCache

        cache = DynamicCache(config=config)
        hx = inputs[:, :seq]
        hr = rope_fn(hx, torch.arange(seq).expand(batch, -1))
        expected = hf(hx, position_embeddings=hr, past_key_values=cache)
        hf_prefill = pcc(outputs["multichip"][0], expected)
        hf_decode = []
        for step in range(args.steps):
            hx = inputs[:, seq + step : seq + step + 1]
            hr = rope_fn(hx, torch.full((batch, 1), seq + step, dtype=torch.int64))
            expected = hf(hx, position_embeddings=hr, past_key_values=cache)
            hf_decode.append(pcc(outputs["multichip"][step + 1], expected))
        hf_accuracy = {
            "prefill_pcc": hf_prefill,
            "decode_pcc": hf_decode,
            "prefill_tokens": seq * batch,
            "decode_tokens": args.steps * batch,
            "source": "Direct real-checkpoint Hugging Face layer with its own sequential KV cache",
        }
    result = {
        "seq": seq,
        "batch": batch,
        "steps": args.steps,
        "residual_sharded": not args.replicated,
        "ccl_dtype": args.ccl_dtype,
        "command_args": vars(args),
        "runner_sha256": runner_hash,
        "candidate_sha256": candidate_hash,
        "stage5_candidate_sha256": stage5_candidate_hash,
        "prefill_pcc": pp,
        "hf_accuracy": hf_accuracy,
        "decode_pcc": dp,
        "cache_pcc": kp,
        "cache_per_head_pcc": head_kp,
        "cache_pcc_is_diagnostic": cache_projection_changed,
        "cache_comparison_basis": {
            "baseline_attention_weights": PrecisionPolicy().attention,
            "candidate_attention_weights": compared_policy.attention,
            "reason": "Different projection weight precision changes cached K/V; all sequential cache-consuming output PCCs remain acceptance gates."
            if cache_projection_changed
            else "Matching projection precision: aggregate and per-head cache PCCs are acceptance gates.",
        },
        "measurements": measurements,
        "baseline_sha256": baseline_hash,
        "implementation_sha256": runtime_hash,
        "collective_buffers_sha256": hashlib.sha256(
            Path(__file__).resolve().parents[1].joinpath("tt/collective_buffers.py").read_bytes()
        ).hexdigest(),
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    assert pp >= 0.995 and min(dp) >= 0.995, result
    assert all(m["replay_bitwise"] for m in measurements.values())
    if not cache_projection_changed:
        assert min(kp) >= 0.995 and min(min(row) for row in head_kp) >= 0.995, head_kp
    assert all(m["updated_input_outputs_changed"] for m in measurements.values())
    if hf_accuracy:
        assert hf_accuracy["prefill_pcc"] >= 0.995 and min(hf_accuracy["decode_pcc"]) >= 0.995, hf_accuracy


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seq", type=int, default=33)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--replicated", action="store_true")
    p.add_argument("--ccl-dtype", default="bfloat16")
    p.add_argument("--links", type=int, default=2)
    p.add_argument("--slice-q", action="store_true")
    p.add_argument("--split", type=int)
    p.add_argument("--remap", action="store_true")
    p.add_argument("--heterogeneous", action="store_true")
    p.add_argument("--candidate")
    p.add_argument("--policy")
    p.add_argument("--stack", type=int, default=1, choices=[1, 2])
    p.add_argument("--output", required=True)
    p.add_argument("--burst-replays", type=int, default=0)
    p.add_argument("--hf-check", action="store_true")
    run(p.parse_args())
