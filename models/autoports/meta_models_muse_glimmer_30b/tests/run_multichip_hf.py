# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Additional independent HF/TP4 decoder accuracy control. Use the separate HF reference environment.

All torch work is setup/reference/comparison. Device forwards are guarded against
host conversion. Checkpoint loading reads only the selected layer's tensors.
"""

import argparse
import gc
import gzip
import hashlib
import inspect
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.muse_glimmer.modeling_muse_glimmer import (
    MuseGlimmerTextDecoderLayer,
    MuseGlimmerTextRotaryEmbedding,
)

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import MultichipDecoder as OptimizedDecoder

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "tests/assets"
SNAPSHOT = (
    Path(os.environ.get("HF_HOME", "/mnt/models/huggingface"))
    / "hub/models--meta-models--Muse-Glimmer-30B/snapshots/a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
)


def weights(config, layer_idx, real, evidence_dir):
    with torch.device("meta"):
        ref = MuseGlimmerTextDecoderLayer(config, layer_idx)
    if real:
        index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())["weight_map"]
        prefix = f"model.language_model.layers.{layer_idx}."
        state = {}
        for filename in set(index[k] for k in index if k.startswith(prefix)):
            with safe_open(SNAPSHOT / filename, framework="pt", device="cpu") as f:
                for key in index:
                    if key.startswith(prefix) and index[key] == filename:
                        state[key[len(prefix) :]] = f.get_tensor(key)
        stats = [
            {
                "name": k,
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "mean": v.float().mean().item(),
                "std": v.float().std().item(),
            }
            for k, v in state.items()
        ]
        (evidence_dir / f"weight_stats_{layer_idx}.json").write_text(json.dumps(stats, indent=2))
    else:
        stats_path = ROOT / f"doc/functional_decoder/weight_stats_{layer_idx}.json"
        stats = {v["name"]: v for v in json.loads(stats_path.read_text())}
        g = torch.Generator().manual_seed(42 + layer_idx)
        state = {}
        for k, v in ref.state_dict().items():
            assert list(v.shape) == stats[k]["shape"], k
            mean = stats[k]["mean"]
            std = stats[k]["std"]
            state[k] = (torch.randn(v.shape, generator=g) * std + mean).bfloat16()
    ref.load_state_dict(state, assign=True)
    return ref.eval(), state


class LayerCache:
    def __init__(self, window=None):
        self.k = self.v = None
        self.window = window

    def update(self, k, v, layer_idx):
        self.k = k if self.k is None else torch.cat((self.k, k), dim=2)
        self.v = v if self.v is None else torch.cat((self.v, v), dim=2)
        result = self.k, self.v
        if self.window:
            self.k, self.v = self.k[:, :, -(self.window - 1) :], self.v[:, :, -(self.window - 1) :]
        return result


@torch.no_grad()
def reference(ref, x, start, cache, rotary):
    pieces = []
    # The official layer performs every calculation; stream only its mask/cache
    # inputs so long-context evidence never materializes a square 131K mask.
    for off in range(0, x.shape[1], 512):
        chunk = x[:, off : off + 512]
        pos = torch.arange(start + off, start + off + chunk.shape[1])
        retained = 0 if cache.k is None else cache.k.shape[2]
        keys = torch.arange(start + off - retained, start + off + chunk.shape[1])
        visible = keys[None, :] <= pos[:, None]
        if ref.self_attn.sliding_window:
            visible &= keys[None, :] > pos[:, None] - ref.self_attn.sliding_window
        mask = torch.zeros(visible.shape, dtype=x.dtype).masked_fill_(~visible, float("-inf"))[None, None]
        rope = rotary(chunk, pos[None]) if ref.config.layer_rope_theta[ref.self_attn.layer_idx] else None
        pieces.append(ref(chunk, position_embeddings=rope, attention_mask=mask, past_key_values=cache))
    return torch.cat(pieces, dim=1)


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    score = (torch.dot(a, b) / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))).item()
    # Parallel dot/norm reductions can differ by a few ulps at millions of
    # elements, including a tensor compared with itself. This only bounds the
    # mathematical range; it does not relax the independent 0.995 gate.
    assert -1.000000001 <= score <= 1.000000001, score
    return max(-1.0, min(1.0, score))


def to_device(x, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(x, device=device, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def read(x):
    pieces = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(x)]
    assert all(torch.equal(pieces[0], p) for p in pieces[1:]), "Replicas differ"
    return pieces[0]


@contextmanager
def device_only():
    old = {name: getattr(ttnn, name) for name in ("from_torch", "to_torch", "as_tensor")}

    def forbidden(*a, **k):
        raise AssertionError("Host fallback inside a decoder pass")

    def audit(frame, event, arg):
        module = frame.f_globals.get("__name__", "")
        if event == "call" and (module == "torch" or module.startswith("torch.")):
            raise AssertionError(f"Torch call inside decoder: {module}.{frame.f_code.co_name}")

    previous_profile = sys.getprofile()
    try:
        sys.setprofile(audit)
        for name in old:
            setattr(ttnn, name, forbidden)
        yield
    finally:
        sys.setprofile(previous_profile)
        for name, value in old.items():
            setattr(ttnn, name, value)


def run(args):
    import transformers

    provenance = {
        "command": sys.argv,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "snapshot": str(SNAPSHOT),
        "trace_alloc_tracking": os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING", "0"),
        "trace_alloc_skip_program_cache": os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0"),
        "watcher": os.environ.get("TT_METAL_WATCHER"),
        "device_profiler": os.environ.get("TT_METAL_DEVICE_PROFILER"),
        "profiler_program_support_count": os.environ.get("TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT"),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__).resolve(),
                ROOT / "tt/multichip_decoder.py",
                ROOT / "tt/optimized_decoder.py",
                ROOT / "tt/functional_decoder.py",
            )
        },
        "dependency_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(inspect.getfile(MuseGlimmerTextDecoderLayer)),
                Path("ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp"),
                Path("ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"),
            )
        },
        "seed": 123,
        "pcc_dtype": "float64",
        "acceptance": 0.995,
    }
    args.output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2))
    torch.set_num_threads(8)
    torch.manual_seed(123)
    cfg = AutoConfig.from_pretrained(CONFIG).text_config
    cfg._attn_implementation = "sdpa"
    ref, state = weights(cfg, args.layer, args.real, args.output.parent)
    rotary = MuseGlimmerTextRotaryEmbedding(cfg)
    recorded = torch.load(ROOT / f"doc/optimized_decoder/inputs/layer_{args.layer}.pt", weights_only=True)

    def activation(start, count):
        return torch.cat(
            [recorded[:, (torch.arange(start, start + count) + 17 * b) % recorded.shape[1]] for b in range(args.batch)],
            0,
        ).contiguous()

    provenance[
        "input_source"
    ] = "recorded official-HF checkpoint text activations; cycled for contexts longer than recording"
    overrides = json.loads(args.policy) if args.policy else {}
    provenance["policy_overrides"] = overrides
    args.output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2))
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    try:
        layer = OptimizedDecoder.from_state_dict(
            state, hf_config=cfg, layer_idx=args.layer, mesh_device=mesh, policy=overrides
        )
        provenance["policy"] = layer.policy
        args.output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2))
        for key, source in {"runner": Path(__file__), "decoder": ROOT / "tt/multichip_decoder.py"}.items():
            args.output.with_suffix(f".{key}.py.gz").write_bytes(gzip.compress(source.read_bytes(), mtime=0))
        del state
        kv = layer.allocate_cache(batch_size=args.batch)
        results = []
        for length in args.lengths:
            print(f"CASE layer={args.layer} batch={args.batch} length={length}", flush=True)
            x = activation(0, length)
            pt = torch.randperm(args.batch * 4096).reshape(args.batch, 4096).int()
            tt_pt = to_device(pt, mesh, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            tt_x = to_device(x[:, None], mesh)
            t0 = time.perf_counter()
            slots = list(reversed(range(args.batch)))
            with device_only():
                tt_y = layer.prefill_forward(tt_x, kv_cache=kv, page_table=tt_pt, slots=slots)
            got = read(tt_y)[:, 0]
            print(f"TT prefill done {time.perf_counter()-t0:.3f}s", flush=True)
            cache = LayerCache(window=layer.window if length > 4096 else None)
            expected = reference(ref, x, 0, cache, rotary)
            random_positions = torch.randint(0, length, (args.batch,), dtype=torch.int32)
            random_expected = expected[torch.arange(args.batch), random_positions.long()][:, None].clone()
            last_expected = expected[:, -1:].clone()
            score = pcc(expected, got)
            prefill_user_pcc = [pcc(expected[b], got[b]) for b in range(args.batch)]
            assert min(prefill_user_pcc) >= 0.995, prefill_user_pcc
            print(f"PREFILL_PCC {score}", flush=True)
            assert score >= 0.995, score
            if length <= 4096:
                with device_only():
                    repeated = layer.prefill_forward(tt_x, kv_cache=kv, page_table=tt_pt, slots=slots)
                assert torch.equal(got, read(repeated)[:, 0])
                del repeated
            if args.control:
                old_chunk = layer.CHUNK_SIZE
                layer.CHUNK_SIZE = (length + 31) // 32 * 32
                with device_only():
                    unchunked = layer.prefill_forward(tt_x, kv_cache=kv, page_table=tt_pt, slots=slots)
                control_score = pcc(got, read(unchunked)[:, 0])
                assert control_score >= 0.995
                layer.CHUNK_SIZE = old_chunk
                del unchunked
                print(f"UNCHUNKED_CONTROL_PCC {control_score}", flush=True)
            row = {
                "layer": args.layer,
                "kind": layer.kind,
                "batch": args.batch,
                "length": length,
                "real_weights": args.real,
                "prefill_pcc": score,
                "prefill_user_pcc": prefill_user_pcc,
                "prefill_repeated_equal": True if length <= 4096 else None,
            }
            if args.control:
                row["unchunked_control_pcc"] = control_score
            if args.continuation:
                tail = activation(length, args.continuation)
                tt_tail = to_device(tail[:, None], mesh)
                with device_only():
                    continued = layer.prefill_forward(
                        tt_tail, kv_cache=kv, page_table=tt_pt, start_pos=length, slots=slots
                    )
                wanted = reference(ref, tail, length, cache, rotary)
                continuation_score = pcc(wanted, read(continued)[:, 0])
                print(f"CONTINUATION_PCC {continuation_score}", flush=True)
                assert continuation_score >= 0.995
                row["continuation"] = {"start": length, "length": args.continuation, "pcc": continuation_score}
                length += args.continuation
                del tt_tail, continued, tail, wanted
            if args.perf:
                from tracy import signpost

                with device_only():
                    warmed = layer.prefill_forward(tt_x, kv_cache=kv, page_table=tt_pt, slots=slots)
                del warmed
                ttnn.synchronize_device(mesh)
                signpost("PERF_PREFILL")
                t0 = time.perf_counter()
                for _ in range(3):
                    with device_only():
                        warmed = layer.prefill_forward(tt_x, kv_cache=kv, page_table=tt_pt, slots=slots)
                    del warmed
                ttnn.synchronize_device(mesh)
                row["prefill_warmed_wall_ms"] = (time.perf_counter() - t0) * 1000 / 3
                signpost("PERF_PREFILL_END")
            if length <= cfg.max_position_embeddings:
                at_end = length == cfg.max_position_embeddings
                token = x[:, -1:] if at_end else activation(length, 1)
                decode_pos = length - 1 if at_end else length
                if at_end:
                    cache.k, cache.v = cache.k[:, :, :-1], cache.v[:, :, :-1]
                pos = torch.full((args.batch,), decode_pos, dtype=torch.int32)
                decode_pt = pt[slots].contiguous()
                tt_decode_pt = to_device(decode_pt, mesh, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
                cos, sin = rotary(token, pos[:, None])
                inputs = {
                    "x": to_device(token.transpose(0, 1)[None], mesh),
                    "current_pos": to_device(pos, mesh, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                    "rope_cos": to_device(cos[None], mesh),
                    "rope_sin": to_device(sin[None], mesh),
                }
                with device_only():
                    warm = layer.decode_forward(**inputs, kv_cache=kv, page_table=tt_decode_pt)
                del warm
                ttnn.synchronize_device(mesh)
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                with device_only():
                    out = layer.decode_forward(**inputs, kv_cache=kv, page_table=tt_decode_pt)
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                # out is created by this trace and consumed immediately after each replay.
                from ttnn.tools import trace_allocation_tracker

                trace_allocation_tracker.acknowledge_corruptible(out)
                samples = []
                for _ in range(3):
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                    samples.append(read(out)[0].transpose(0, 1).clone())
                assert torch.equal(samples[0], samples[1]) and torch.equal(samples[1], samples[2])
                expected = last_expected if at_end else reference(ref, token, decode_pos, cache, rotary)
                score = pcc(expected, samples[-1])
                decode_user_pcc = [pcc(expected[b], samples[-1][b]) for b in range(args.batch)]
                assert min(decode_user_pcc) >= 0.995, decode_user_pcc
                print(f"TRACED_DECODE_PCC {score}", flush=True)
                assert score >= 0.995, score
                row.update(
                    decode_user_pcc=decode_user_pcc,
                    decode_pcc=score,
                    decode_position=decode_pos,
                    repeated_replay_equal=True,
                    trace_id=tid,
                )
                if args.perf:
                    ttnn.synchronize_device(mesh)
                    signpost("PERF_DECODE")
                    t0 = time.perf_counter()
                    for _ in range(20):
                        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    row["decode_warmed_trace_wall_ms"] = (time.perf_counter() - t0) * 1000 / 20
                    signpost("PERF_DECODE_END")
                # Nonuniform per-user current positions; prior prefill output is
                # the reference for rewriting the identical token at that position.
                rt = x[torch.arange(args.batch), random_positions.long()][:, None]
                rc, rs = rotary(rt, random_positions[:, None])
                for name, value in {
                    "x": rt.transpose(0, 1)[None],
                    "current_pos": random_positions,
                    "rope_cos": rc[None],
                    "rope_sin": rs[None],
                }.items():
                    host = ttnn.from_torch(value, dtype=inputs[name].dtype, layout=inputs[name].layout)
                    ttnn.copy_host_to_device_tensor(host, inputs[name])
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                random_output = read(out)[0].transpose(0, 1)
                random_score = pcc(random_expected, random_output)
                random_user_pcc = [pcc(random_expected[b], random_output[b]) for b in range(args.batch)]
                row["random_positions"] = {
                    "positions": random_positions.tolist(),
                    "pcc": random_score,
                    "per_user_pcc": random_user_pcc,
                    "trace_id": tid,
                }
                args.output.write_text(json.dumps(results + [row], indent=2, default=str))
                print(f"RANDOM_POSITIONS {random_positions.tolist()}", flush=True)
                print(f"RANDOM_USER_PCC {random_user_pcc}", flush=True)
                if random_score < 0.995 or min(random_user_pcc) < 0.995:
                    torch.save(
                        {
                            "input": rt,
                            "expected": random_expected,
                            "actual": random_output,
                            "positions": random_positions,
                            "page_table": decode_pt,
                        },
                        args.output.with_suffix(".failure.pt"),
                    )
                assert random_score >= 0.995, random_score
                assert min(random_user_pcc) >= 0.995, random_user_pcc
                print(f"RANDOM_POSITION_PCC {random_score}", flush=True)
                # Same trace, refreshed token/position/RoPE and changed ownership.
                fresh_cache = LayerCache()
                refreshed = []
                new_pt = torch.randperm(args.batch * 4096).reshape(args.batch, 4096).int()
                for step in (0, 1):
                    new_token = activation(step + 113, 1)
                    new_pos = torch.full((args.batch,), step, dtype=torch.int32)
                    nc, ns = rotary(new_token, new_pos[:, None])
                    refresh = {
                        "x": new_token.transpose(0, 1)[None],
                        "current_pos": new_pos,
                        "rope_cos": nc[None],
                        "rope_sin": ns[None],
                    }
                    for name, value in refresh.items():
                        host = ttnn.from_torch(value, dtype=inputs[name].dtype, layout=inputs[name].layout)
                        ttnn.copy_host_to_device_tensor(host, inputs[name])
                    host_pt = ttnn.from_torch(new_pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
                    ttnn.copy_host_to_device_tensor(host_pt, tt_decode_pt)
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                    updated = read(out)[0].transpose(0, 1)
                    wanted = reference(ref, new_token, step, fresh_cache, rotary)
                    update_score = pcc(wanted, updated)
                    print(f"REFRESHED_TRACE_PCC step={step} {update_score}", flush=True)
                    assert update_score >= 0.995
                    update_user_pcc = [pcc(wanted[b], updated[b]) for b in range(args.batch)]
                    assert min(update_user_pcc) >= 0.995, update_user_pcc
                    assert not torch.equal(samples[-1], updated)
                    refreshed.append(
                        {"position": step, "pcc": update_score, "per_user_pcc": update_user_pcc, "trace_id": tid}
                    )
                row["refreshed_replays"] = refreshed
                row["trace_capture_count"] = 1
                row["trace_replay_count"] = 6 + (20 if args.perf else 0)
                ttnn.release_trace(mesh, tid)
                del out, inputs, samples, tt_decode_pt
            results.append(row)
            args.output.write_text(json.dumps(results, indent=2, default=str))
            del tt_y, tt_x, tt_pt, got, x, expected, cache
            gc.collect()
        print("ALL_PASS", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", help="JSON overrides of the multichip policy")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--lengths", type=int, nargs="+", default=[32])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--perf", action="store_true")
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--continuation", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "doc/multichip_decoder/hf_result.json")
    run(parser.parse_args())
