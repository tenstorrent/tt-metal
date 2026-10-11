# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serial real-weight single-chip/TP4 equivalence and trace benchmark runner."""

import argparse
import gzip
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tests.prepare_optimized_inputs import load_state
from models.autoports.meta_models_muse_glimmer_30b.tests.run_optimized import (
    CONFIG,
    ROOT,
    AutoConfig,
    MuseGlimmerTextRotaryEmbedding,
    device_only,
    pcc,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import MultichipDecoder
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import OptimizedDecoder
from models.autoports.meta_models_muse_glimmer_30b.tt.prepared_prefill import PreparedPrefill


class DecoderPair:
    """Different layer kinds connected without a boundary conversion."""

    def __init__(self, layers):
        self.layers = layers

    def allocate_cache(self, **kwargs):
        return tuple(layer.allocate_cache(**kwargs) for layer in self.layers)

    def prefill_forward(self, x, *, kv_cache, **kwargs):
        for layer, cache in zip(self.layers, kv_cache):
            x = layer.prefill_forward(x, kv_cache=cache, **kwargs)
        return x

    def decode_forward(self, x, *, kv_cache, **kwargs):
        for layer, cache in zip(self.layers, kv_cache):
            x = layer.decode_forward(x, kv_cache=cache, **kwargs)
        return x


def main(args):
    if args.prefill_repeats < 2:
        raise ValueError("Acceptance requires at least two repeated prefills")
    torch.set_num_threads(8)
    torch.manual_seed(123)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cfg = AutoConfig.from_pretrained(CONFIG).text_config
    state = load_state(args.layer)
    recorded = torch.load(ROOT / f"doc/optimized_decoder/inputs/layer_{args.layer}.pt", weights_only=True)

    def activation(start, count):
        return torch.cat(
            [recorded[:, (torch.arange(start, start + count) + 17 * b) % recorded.shape[1]] for b in range(args.batch)],
            0,
        ).contiguous()

    rotary = MuseGlimmerTextRotaryEmbedding(cfg)
    pages_per_user = 4096 if args.pages is None else args.pages // args.batch
    if args.pages is not None:
        assert args.pages % args.batch == 0 and pages_per_user * 32 >= args.length + args.continuation
    physical = torch.randperm(args.batch * pages_per_user).reshape(args.batch, pages_per_user).int()
    # Native fused cache update requires table width <= physical page count.
    # A smaller caller-selected pool therefore carries a shorter table; the
    # default pool/table retain the complete 131072-token context.
    pt = physical.contiguous()
    slots = list(reversed(range(args.batch)))
    x = activation(0, args.length)
    pos = torch.full((args.batch,), min(args.length, 131071), dtype=torch.int32)
    tok = activation(int(pos[0]), 1)
    cos, sin = rotary(tok, pos[:, None])
    inputs_host = dict(x=tok.transpose(0, 1)[None], current_pos=pos, rope_cos=cos[None], rope_sin=sin[None])
    suffix = ("_stack" if args.stack else "") + (f"_pages{args.pages}" if args.pages is not None else "")
    reference_path = (
        args.output.parent / f"baseline_l{args.layer}_b{args.batch}_s{args.length}_c{args.continuation}{suffix}.pt"
    )
    results = []
    overrides = {}
    if args.mode == "single":
        cls = OptimizedDecoder
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
        sharded = False
    else:
        cls = MultichipDecoder
        overrides = dict(sharded_residual=args.mode == "sharded")
        if args.policy:
            overrides.update(json.loads(args.policy))
        if args.policy_file:
            overrides.update(json.loads(args.policy_file.read_text()))
        requested_policy = dict(cls.POLICY, **overrides)
        ttnn.set_fabric_config(
            ttnn.FabricConfig.FABRIC_1D_RING
            if requested_policy.get("topology") == "Ring"
            else ttnn.FabricConfig.FABRIC_1D
        )
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
        sharded = requested_policy["sharded_residual"]
    row = dict(
        mode=args.mode,
        layer=args.layer,
        length=args.length,
        batch=args.batch,
        policy=cls.POLICY,
        command=sys.argv,
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__).resolve(),
                ROOT / "tt/optimized_decoder.py",
                ROOT / "tt/multichip_decoder.py",
                ROOT / "tt/prepared_prefill.py",
            )
        },
        environment={
            k: os.environ.get(k)
            for k in (
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
                "TT_METAL_WATCHER",
                "TT_METAL_DEVICE_PROFILER",
            )
        },
    )
    args.output.with_suffix(".provenance.json").write_text(json.dumps(row, indent=2, default=str) + "\n")
    for source in (
        Path(__file__).resolve(),
        ROOT / "tt/multichip_decoder.py",
        ROOT / "tt/optimized_decoder.py",
        ROOT / "tt/prepared_prefill.py",
    ):
        args.output.with_name(args.output.stem + "_" + source.name + ".gz").write_bytes(
            gzip.compress(source.read_bytes(), mtime=0)
        )

    def upload(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, residual=False, device=True):
        mapper = ttnn.ShardTensorToMesh(mesh, dim=3) if sharded and residual else ttnn.ReplicateTensorToMesh(mesh)
        return ttnn.from_torch(value, dtype=dtype, layout=layout, device=mesh if device else None, mesh_mapper=mapper)

    def read(t, residual=True):
        pieces = [ttnn.to_torch(v).clone() for v in ttnn.get_device_tensors(t)]
        finite = all(bool(torch.isfinite(v).all()) for v in pieces)
        equal = all(torch.equal(pieces[0], v) for v in pieces[1:])
        if not finite or (residual and not sharded and not equal):
            torch.save(pieces, args.output.with_suffix(".replica_failure.pt"))
            diagnostics = []
            for i, v in enumerate(pieces):
                finite_pair = torch.isfinite(pieces[0]) & torch.isfinite(v)
                mismatch = finite_pair & (pieces[0] != v)
                delta = (pieces[0][finite_pair].float() - v[finite_pair].float()).abs()
                diagnostics.append(
                    dict(
                        rank=i,
                        shape=list(v.shape),
                        nan_count=int(torch.isnan(v).sum()),
                        inf_count=int(torch.isinf(v).sum()),
                        finite_mismatch_count=int(mismatch.sum()),
                        nan_mask_disagreement=int((torch.isnan(pieces[0]) != torch.isnan(v)).sum()),
                        inf_mask_disagreement=int((torch.isinf(pieces[0]) != torch.isinf(v)).sum()),
                        first_finite_mismatches=mismatch.nonzero()[:8].tolist(),
                        max_finite_delta=float(delta.max()) if delta.numel() else None,
                    )
                )
            print("REPLICA_FAILURE", diagnostics, flush=True)
        assert finite, "nonfinite logical output"
        if sharded and residual:
            return torch.cat(pieces, -1)
        if residual:
            assert equal, "replicas differ"
        return pieces[0]

    data = {}
    prepared_prefill = None
    baseline = None if args.mode == "single" else torch.load(reference_path, weights_only=True)

    def check(name, got):
        data[name] = got
        if baseline is not None:
            score = pcc(baseline[name], got)
            row[name + "_pcc"] = score
            print(name, score, flush=True)
            assert score >= 0.995, (name, score)
            if not name.startswith("cache_"):
                a, b = baseline[name], got
                if (
                    a.shape[0] == 1
                    and a.shape[2] == args.batch
                    and name not in ("prefill", "continuation", "stack_prefill")
                ):
                    a, b = a[0, 0], b[0, 0]
                per_user = [pcc(a[i], b[i]) for i in range(args.batch)]
                row[name + "_per_user_pcc"] = per_user
                assert min(per_user) >= 0.995, (name, per_user)

    try:
        factory_options = {} if args.mode == "single" else dict(policy=overrides)
        layer = cls.from_state_dict(state, hf_config=cfg, layer_idx=args.layer, mesh_device=mesh, **factory_options)
        row["policy"] = layer.policy
        if args.stack:
            assert args.layer == 0 and not args.microbench
            extra = {} if args.mode == "single" else dict(ccl=layer.ccl, policy=overrides)
            second = cls.from_state_dict(load_state(3), hf_config=cfg, layer_idx=3, mesh_device=mesh, **extra)
            layer = DecoderPair([layer, second])
            row["stacked_layer_indices"] = [0, 3]
        kv = layer.allocate_cache(
            batch_size=args.batch,
            **(dict(num_pages=args.pages) if args.pages is not None and args.mode != "single" else {}),
        )
        row["physical_page_budget"] = args.pages
        modules = layer.layers if args.stack else [layer]
        row["tensor_contracts"] = [
            dict(
                layer=m.layer_idx,
                policy=m.policy,
                weights={
                    name: dict(
                        shape=list(w.shape),
                        padded_shape=list(w.padded_shape),
                        dtype=str(w.dtype),
                        memory_config=str(w.memory_config()),
                    )
                    for name, w in m.weights.items()
                },
                residual_memory=str(m.residual_mem),
                ccl_semaphore_cores=str(m.ccl.ccl_cores) if args.mode != "single" else None,
                program_configs={name: str(pc) for name, pc in m.decode_pc.items()},
            )
            for m in modules
        ]
        row["cache_shapes"] = (
            [[list(t.shape) for t in pair] for pair in kv] if args.stack else [list(t.shape) for t in kv]
        )
        xp = upload(x[:, None], residual=True)
        if args.l1_prefill_input:
            assert args.batch == 1 and args.length == 1024 and not args.stack and args.mode == "replicated"
            memory = ttnn.create_sharded_memory_config(
                (128, 832),
                ttnn.CoreGrid(x=8, y=8),
                ttnn.ShardStrategy.BLOCK,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            xp = ttnn.to_memory_config(xp, memory)
            original_prefill_input = read(xp).clone()
        tp = upload(pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        dp = upload(pt[slots].contiguous(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        inputs = {
            k: upload(
                v,
                ttnn.int32 if k == "current_pos" else ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT if k == "current_pos" else ttnn.TILE_LAYOUT,
                residual=k == "x",
            )
            for k, v in inputs_host.items()
        }
        if args.mode != "single" and args.length <= 64 and not args.eager_prefill and not args.prefill_trace_trial:
            with device_only():
                warm = layer.decode_forward(**inputs, kv_cache=kv, page_table=dp)
            read(warm)
            prepared_prefill = PreparedPrefill(
                layer,
                mesh=mesh,
                kv_cache=kv,
                page_table=tp,
                batch=args.batch,
                policy=json.loads(args.prepared_prefill_policy) if args.prepared_prefill_policy else None,
            )
            prepared_prefill.prepare()
            # Diagnostic BF8 cache reads compile slice/untilize programs too.
            # Prepare those exact signatures before either trace is live.
            if args.validate and not args.stack:
                for cache in kv:
                    for user in range(args.batch):
                        for position in sorted(
                            {0, min(31, args.length - 1), min(32, args.length - 1), args.length - 1}
                        ):
                            physical = int(pt[slots[user], position // 32])
                            view = ttnn.slice(
                                cache,
                                [physical, 0, position % 32, 0],
                                [physical + 1, 1, position % 32 + 1, 128],
                            )
                            for value in ttnn.get_device_tensors(view):
                                ttnn.to_torch(value)
                            del view, value
            prepared_prefill.capture()
            del warm

        def prefill(x, *, start_pos=0):
            if prepared_prefill is not None and start_pos == 0 and x.shape[-2] <= 64:
                return prepared_prefill(x, page_table=tp, slots=slots)
            return layer.prefill_forward(x, kv_cache=kv, page_table=tp, slots=slots, start_pos=start_pos)

        print("PREFILL", flush=True)
        with device_only():
            y = prefill(xp)
        check("prefill", read(y))
        del y
        for _ in range(args.prefill_repeats):
            with device_only():
                y = prefill(xp)
            assert torch.equal(data["prefill"], read(y))
            del y
        row["prefill_deterministic"] = True
        row["prefill_repeat_count"] = args.prefill_repeats
        if args.l1_prefill_input:
            row["caller_l1_input_preserved"] = torch.equal(original_prefill_input, read(xp))
            assert row["caller_l1_input_preserved"]
        if args.validate and not args.stack:
            for cache_name, cache in zip(("k", "v"), kv):
                values = []
                for user in range(args.batch):
                    for position in sorted({0, min(31, args.length - 1), min(32, args.length - 1), args.length - 1}):
                        physical = int(pt[slots[user], position // 32])
                        view = ttnn.slice(
                            cache,
                            [physical, 0, position % 32, 0],
                            [physical + 1, 2 if args.mode == "single" else 1, position % 32 + 1, 128],
                        )
                        parts = [ttnn.to_torch(v) for v in ttnn.get_device_tensors(view)]
                        if args.mode == "single":
                            parts = [parts[0][:, h : h + 1] for h in (0, 0, 1, 1)]
                        else:
                            assert torch.equal(parts[0], parts[1]) and torch.equal(
                                parts[2], parts[3]
                            ), "KV pair ownership differs"
                        values.append(torch.cat(parts, 1))
                        del view
                check("cache_" + cache_name, torch.cat(values, 0))
        if args.continuation:
            tail = upload(activation(args.length, args.continuation)[:, None], residual=True)
            with device_only():
                y = prefill(tail, start_pos=args.length)
            check("continuation", read(y))
            del y, tail
        if args.perf:
            from tracy import signpost

            samples = []
            signpost("PERF_PREFILL")
            for _ in range(3):
                ttnn.synchronize_device(mesh)
                start = time.perf_counter()
                for __ in range(1 if args.profile else 3):
                    y = prefill(xp)
                    del y
                ttnn.synchronize_device(mesh)
                samples.append((time.perf_counter() - start) * 1000 / (1 if args.profile else 3))
            signpost("PERF_PREFILL_END")
            row["prefill_ms"] = statistics.median(samples)
            row["prefill_samples_ms"] = samples
        if args.prefill_trace_trial:
            from ttnn.tools import trace_allocation_tracker

            prefill_tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            with device_only():
                prefill_out = layer.prefill_forward(xp, kv_cache=kv, page_table=tp, slots=slots)
            ttnn.end_trace_capture(mesh, prefill_tid, cq_id=0)
            trace_allocation_tracker.acknowledge_corruptible(prefill_out)
            trial_scores = []
            for variant in range(3):
                new_x = activation(113 * variant, args.length)[:, None]
                new_pt = torch.roll(pt, variant, dims=1)
                ttnn.copy_host_to_device_tensor(upload(new_x, residual=True, device=False), xp)
                ttnn.copy_host_to_device_tensor(upload(new_pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, device=False), tp)
                with device_only():
                    reference = layer.prefill_forward(xp, kv_cache=kv, page_table=tp, slots=slots)
                reference_host = read(reference)
                del reference
                for _ in range(3):
                    ttnn.execute_trace(mesh, prefill_tid, cq_id=0, blocking=True)
                    score = pcc(reference_host, read(prefill_out))
                    assert score >= 0.995, score
                    trial_scores.append(score)
            times = []
            for _ in range(5):
                ttnn.synchronize_device(mesh)
                start = time.perf_counter()
                for __ in range(100):
                    ttnn.execute_trace(mesh, prefill_tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                times.append((time.perf_counter() - start) * 1000 / 100)
            row["prefill_trace_trial"] = dict(
                trace_id=prefill_tid,
                capture_count=1,
                refreshed_variants=3,
                pcc=trial_scores,
                samples_ms=times,
                median_ms=statistics.median(times),
            )
            ttnn.release_trace(mesh, prefill_tid)
            del prefill_out
            ttnn.copy_host_to_device_tensor(upload(x[:, None], residual=True, device=False), xp)
            ttnn.copy_host_to_device_tensor(upload(pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, device=False), tp)
            with device_only():
                y = layer.prefill_forward(xp, kv_cache=kv, page_table=tp, slots=slots)
            del y
        print("DECODE_WARM", flush=True)
        captured_inputs = {}
        original_linear = None if args.stack else layer._linear
        if args.microbench:

            def capture_linear(tensor, name):
                captured_inputs[name] = ttnn.clone(ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG))
                return original_linear(tensor, name)

            layer._linear = capture_linear
        for _ in range(2):
            with device_only():
                y = layer.decode_forward(**inputs, kv_cache=kv, page_table=dp)
            del y
        if args.microbench:
            layer._linear = original_linear
            micro = []
            for name, tensor in captured_inputs.items():
                # Measure the projection and its required input/output layouts;
                # collective topology is measured by the complete decoder below.
                for _ in range(2):
                    y = OptimizedDecoder._linear(layer, tensor, name)
                    del y
                micro_tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                micro_out = OptimizedDecoder._linear(layer, tensor, name)
                ttnn.end_trace_capture(mesh, micro_tid, cq_id=0)
                ttnn.execute_trace(mesh, micro_tid, cq_id=0, blocking=True)
                times = []
                if args.profile:
                    from tracy import signpost

                    micro_signpost = "PERF_MICRO_" + name.replace(".", "_").upper()
                    signpost(micro_signpost)
                micro_replays = 3 if args.profile else 100
                for _ in range(1 if args.profile else 3):
                    start = time.perf_counter()
                    for __ in range(micro_replays):
                        ttnn.execute_trace(mesh, micro_tid, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    times.append((time.perf_counter() - start) * 1000 / micro_replays)
                if args.profile:
                    signpost(micro_signpost + "_END")
                micro.append(
                    dict(
                        name=name,
                        ms=statistics.median(times),
                        samples_ms=times,
                        weight_shape=list(layer.weights[name].shape),
                        input_memory=str(layer.decode_mem[name]),
                        weight_memory=str(layer.weights[name].memory_config()),
                        program_config=str(layer.decode_pc[name]),
                        dtype=str(layer.weights[name].dtype),
                        kernel_config=str(layer.mm_ck[name]),
                    )
                )
                # Preserve every rank's exact recorded input and output for
                # cross-reader agreement checks, outside the timed window.
                torch.save(
                    {
                        "inputs": [ttnn.to_torch(v).clone() for v in ttnn.get_device_tensors(tensor)],
                        "outputs": [ttnn.to_torch(v).clone() for v in ttnn.get_device_tensors(micro_out)],
                    },
                    args.output.with_suffix(".micro_" + name + ".pt"),
                )
                ttnn.release_trace(mesh, micro_tid)
                del micro_out
            captured_inputs.clear()
            row["projection_microbench"] = micro
        ttnn.synchronize_device(mesh)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        with device_only():
            out = layer.decode_forward(**inputs, kv_cache=kv, page_table=dp)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        from ttnn.tools import trace_allocation_tracker

        trace_allocation_tracker.acknowledge_corruptible(out)
        # Only this output's backing allocation is exempted; no aliases retained.
        # Each result is consumed before another replay; all other inputs predate capture.
        samples = []
        for _ in range(3):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            samples.append(read(out))
        assert all(torch.equal(samples[0], v) for v in samples)
        check("decode", samples[0])
        row.update(trace_id=tid, capture_count=1, repeat_equal=True)
        if args.perf or args.decode_perf:
            from tracy import signpost

            times = []
            signpost("PERF_DECODE")
            for _ in range(1 if args.profile else 5):
                ttnn.synchronize_device(mesh)
                start = time.perf_counter()
                for __ in range(20 if args.profile else 100):
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                times.append((time.perf_counter() - start) * 1000 / (20 if args.profile else 100))
            signpost("PERF_DECODE_END")
            row["decode_ms"] = statistics.median(times)
            row["decode_samples_ms"] = times
        if args.validate:
            positions = torch.tensor([(17 + 23 * b) % args.length for b in range(args.batch)], dtype=torch.int32)
            token = torch.cat([x[b : b + 1, int(positions[b]) : int(positions[b]) + 1] for b in range(args.batch)], 0)
            rc, rs = rotary(token, positions[:, None])
            refreshed = dict(x=token.transpose(0, 1)[None], current_pos=positions, rope_cos=rc[None], rope_sin=rs[None])
            for key, value in refreshed.items():
                ttnn.copy_host_to_device_tensor(
                    upload(value, inputs[key].dtype, inputs[key].layout, residual=key == "x", device=False), inputs[key]
                )
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            check("nonuniform_positions", read(out))
            row["nonuniform_positions"] = positions.tolist()
        for step in range(args.stress):
            # A new request at position0, then position1, with new page ownership.
            token = activation(113 + step, 1)
            positions = torch.full((args.batch,), step % 2, dtype=torch.int32)
            rc, rs = rotary(token, positions[:, None])
            refreshed = dict(x=token.transpose(0, 1)[None], current_pos=positions, rope_cos=rc[None], rope_sin=rs[None])
            for key, value in refreshed.items():
                host = upload(value, inputs[key].dtype, inputs[key].layout, residual=key == "x", device=False)
                ttnn.copy_host_to_device_tensor(host, inputs[key])
            if step % 2 == 0:
                new_pt = torch.roll(pt, step + 1, dims=1)
                ttnn.copy_host_to_device_tensor(upload(new_pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, device=False), dp)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            check(f"refresh_{step}", read(out))
        row["updated_input_replays"] = args.stress
        row["prefill_execution"] = "prepared physical32/64 traces" if prepared_prefill else "eager"
        if prepared_prefill is not None:
            row["prefill_trace_policy"] = prepared_prefill.policy
            row["prefill_captured_policies"] = prepared_prefill.captured_policies
            row["prefill_setup_ms"] = dict(prepare=prepared_prefill.prepare_ms, capture=prepared_prefill.capture_ms)
            row["prefill_trace_captures"] = prepared_prefill.capture_count
            row["prefill_trace_replays"] = prepared_prefill.replay_count
            row["prepared_prefill_source_sha256"] = hashlib.sha256(
                (ROOT / "tt/prepared_prefill.py").read_bytes()
            ).hexdigest()
        row["request_recaptures"] = 0
        ttnn.release_trace(mesh, tid)
        if os.environ.get("TT_METAL_WATCHER") == "10":
            # Allow the required 10-second watcher to sample even a short case.
            ttnn.synchronize_device(mesh)
            time.sleep(11)
            row["watcher_dwell_seconds"] = 11
        if args.mode == "single":
            torch.save(data, reference_path)
        row["status"] = "pass"
        args.output.write_text(json.dumps(row, indent=2, default=str) + "\n")
        print("ALL_PASS", row.get("prefill_ms"), row.get("decode_ms"), flush=True)
    finally:
        if prepared_prefill is not None:
            prepared_prefill.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "replicated", "sharded"], required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--pages", type=int)
    parser.add_argument("--continuation", type=int, default=0)
    parser.add_argument("--stress", type=int, default=2)
    parser.add_argument("--prefill-repeats", type=int, default=2)
    parser.add_argument("--policy")
    parser.add_argument("--policy-file", type=Path)
    parser.add_argument("--perf", action="store_true")
    parser.add_argument("--decode-perf", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--microbench", action="store_true")
    parser.add_argument("--l1-prefill-input", action="store_true")
    parser.add_argument("--prefill-trace-trial", action="store_true")
    parser.add_argument("--eager-prefill", action="store_true")
    parser.add_argument("--prepared-prefill-policy")
    parser.add_argument("--stack", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
