"""Serialized baseline/TP4 comparison with identical pinned real weights and inputs."""

import faulthandler
import signal

faulthandler.register(signal.SIGUSR1, all_threads=True)
import argparse
import ast
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

import ttnn
from models.autoports.google_gemma_4_31b_it.tests.run_optimized_decoder import pcc, weights
from models.autoports.google_gemma_4_31b_it.tt.multichip_decoder import MultichipDecoder
from models.autoports.google_gemma_4_31b_it.tt.optimized_decoder import HF_MODEL, HF_REVISION, OptimizedDecoder


def forbidden_read(*args, **kwargs):
    raise AssertionError("runtime host read")


def run(a):
    torch.set_num_threads(16)
    torch.manual_seed(123)
    c = AutoConfig.from_pretrained(HF_MODEL, revision=HF_REVISION).text_config
    state = weights(a.layer)
    fixture = Path(a.inputs or f"bringup/artifacts/optimized_decoder_layer{a.layer}_inputs2050.pt")
    source = torch.load(fixture, weights_only=True, mmap=True)
    raw = source[:, : a.seq + 1]
    if a.distinct_offsets:
        assert source.shape[0] == 1 and source.shape[1] >= a.seq + 1 + 17 * (a.batch - 1)
        x = torch.stack([source[0, 17 * i : 17 * i + a.seq + 1] for i in range(a.batch)])
    else:
        x = (raw.expand(a.batch, -1, -1) if raw.shape[0] == 1 else raw[: a.batch]).contiguous()
    decode_position = min(a.seq, c.max_position_embeddings - 1)
    assert x.shape[1] == decode_position + 1
    cos, sin = Gemma4TextRotaryEmbedding(c)(
        x, torch.arange(a.seq + (a.replays if a.stress else 1)).expand(a.batch, -1), layer_type=c.layer_types[a.layer]
    )
    blocks = (a.seq + (a.replays if a.stress else 1) + 31) // 32
    pages = (torch.randperm(a.batch * blocks) + 3).reshape(a.batch, blocks).to(torch.int32)
    results, tensors, stress_outputs, cache_outputs = {}, {}, {}, {}
    for count in [1, 4]:
        if count == 4:
            ttnn.set_fabric_config(
                ttnn.FabricConfig.FABRIC_1D_RING
                if not a.candidate
                or (json.loads(a.candidate).get("ring_fused") or json.loads(a.candidate).get("ring_all"))
                else ttnn.FabricConfig.FABRIC_1D
            )
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, count), trace_region_size=0)
        try:
            empty_allocated_per_bank = int(
                ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank
            )
            cls = OptimizedDecoder if count == 1 else MultichipDecoder
            if count == 4 and a.candidate:
                from models.autoports.google_gemma_4_31b_it.tests.multichip_candidate import MultichipCandidate

                cls = MultichipCandidate
            print("SETUP", count, flush=True)
            model = cls.from_state_dict(
                state,
                hf_config=c,
                layer_idx=a.layer,
                mesh_device=mesh,
                **(
                    {
                        "residual_sharded": not a.replicated,
                        "sliding_cache_tokens": (
                            2048 if a.cyclic and c.layer_types[a.layer] == "sliding_attention" else None
                        ),
                        **({"candidate": json.loads(a.candidate)} if a.candidate else {}),
                    }
                    if count == 4
                    else {}
                ),
            )
            print("WEIGHTS_READY", count, flush=True)

            def memory():
                view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
                return {
                    k: int(getattr(view, k))
                    for k in (
                        "num_banks",
                        "total_bytes_per_bank",
                        "total_bytes_allocated_per_bank",
                        "largest_contiguous_bytes_free_per_bank",
                    )
                }

            weight_memory = memory()

            def tt(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, residual=False, on_device=True):
                mapper = (
                    ttnn.ShardTensorToMesh(mesh, dim=3)
                    if residual and count == 4 and not a.replicated
                    else ttnn.ReplicateTensorToMesh(mesh)
                )
                print("UPLOAD", count, tuple(t.shape), str(layout), flush=True)
                host_tensor = ttnn.from_torch(
                    t, device=None, dtype=dtype, layout=layout, mesh_mapper=mapper if count > 1 else None
                )
                return ttnn.to_device(host_tensor, mesh) if on_device else host_tensor

            def host(t):
                if count == 4 and not a.replicated:
                    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=3))
                return ttnn.to_torch(ttnn.get_device_tensors(t)[0])

            current_pages = pages
            physical_blocks = a.batch * blocks + 3
            if count == 4 and a.cyclic and c.layer_types[a.layer] == "sliding_attention":
                pool = (torch.randperm(a.batch * 64) + 3).reshape(a.batch, 64)
                current_pages = pool[:, torch.arange(max(blocks, 64)) % 64].to(torch.int32)
                physical_blocks = a.batch * 64 + 3
            cache = model.allocate_cache(num_blocks=physical_blocks)
            cache_memory = memory()
            page = tt(current_pages, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            px = tt(x[:, : a.seq].unsqueeze(1), residual=True)
            pc, ps = tt(cos[:, : a.seq].unsqueeze(1)), tt(sin[:, : a.seq].unsqueeze(1))
            dx = tt(x[:, decode_position : decode_position + 1].reshape(1, 1, a.batch, -1), residual=True)
            kw = dict(
                cos=tt(cos[:, decode_position : decode_position + 1].reshape(1, a.batch, 1, -1)),
                sin=tt(sin[:, decode_position : decode_position + 1].reshape(1, a.batch, 1, -1)),
                kv_cache=cache,
                page_table=page,
                current_pos=tt(
                    torch.full((a.batch,), decode_position, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
                ),
            )
            print("PREFILL_START", count, flush=True)
            with (
                patch("ttnn.from_torch", side_effect=AssertionError("runtime host conversion")),
                patch(
                    "ttnn.to_torch",
                    new=ttnn.to_torch if a.candidate and json.loads(a.candidate).get("diagnose") else forbidden_read,
                ),
            ):
                pref = model.prefill_forward(px, cos=pc, sin=ps, kv_cache=cache, page_table=page)
                print("DECODE_START", count, flush=True)
                dec = model.decode_forward(dx, **kw)
            print("HOST_READ_START", count, flush=True)
            selected_pref = pref
            if a.seq > 4096:
                selected_pref = ttnn.slice(pref, (0, 0, a.seq - 32, 0), (a.batch, 1, a.seq, pref.shape[-1]))
            tensors[count] = (host(selected_pref), host(dec))
            del selected_pref
            cache_outputs[count] = []
            selected_blocks = [decode_position // 32] if a.cyclic else sorted({0, decode_position // 32})
            for cache_tensor in cache:
                unused = ttnn.slice(cache_tensor, (0, 0, 0, 0), (3, model.nkv, 32, model.head_dim))
                for local in ttnn.get_device_tensors(unused):
                    assert torch.count_nonzero(ttnn.to_torch(local)) == 0
                snapshots = []
                for user in range(a.batch):
                    for logical_block in selected_blocks:
                        physical = int(current_pages[user, logical_block])
                        valid = decode_position % 32 + 1 if logical_block == decode_position // 32 else 32
                        piece = ttnn.slice(
                            cache_tensor, (physical, 0, 0, 0), (physical + 1, model.nkv, valid, model.head_dim)
                        )
                        snapshots.append(
                            ttnn.to_torch(piece, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1))
                            if count == 4
                            else ttnn.to_torch(piece)
                        )
                cache_outputs[count].append(torch.cat([t.flatten() for t in snapshots]))
            del unused, piece, snapshots, local, cache_tensor
            ttnn.synchronize_device(mesh)
            start = time.perf_counter()
            for _ in range(3):
                dec = model.decode_forward(dx, **kw)
            ttnn.synchronize_device(mesh)
            results[str(count)] = {
                "empty_allocated_per_bank": empty_allocated_per_bank,
                "weight_memory": weight_memory,
                "cache_memory": cache_memory,
                "decode_eager_host_us": (time.perf_counter() - start) * 1e6 / 3,
            }
            start = time.perf_counter()
            warm_pref = model.prefill_forward(px, cos=pc, sin=ps, kv_cache=cache, page_table=page)
            ttnn.synchronize_device(mesh)
            results[str(count)]["prefill_warm_host_us"] = (time.perf_counter() - start) * 1e6
            del warm_pref
            if a.stress:
                cache2 = model.allocate_cache(num_blocks=physical_blocks)
                page2 = tt(current_pages.flip(0), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
                other = model.prefill_forward(px, cos=pc, sin=ps, kv_cache=cache2, page_table=page2)
                del other
                stress_host = []
                for step in range(a.replays):
                    positions = torch.arange(a.batch, dtype=torch.int32).remainder(min(a.seq, 7)).neg() + a.seq + step
                    idx = torch.arange(a.batch)
                    stress_host.append(
                        (
                            tt(x[:, step % a.seq].reshape(1, 1, a.batch, -1), residual=True, on_device=False),
                            tt(cos[idx, positions.long()].reshape(1, a.batch, 1, -1), on_device=False),
                            tt(sin[idx, positions.long()].reshape(1, a.batch, 1, -1), on_device=False),
                            tt(positions, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, on_device=False),
                            tt(
                                current_pages if step < a.replays // 2 else current_pages.flip(0),
                                ttnn.int32,
                                ttnn.ROW_MAJOR_LAYOUT,
                                on_device=False,
                            ),
                        )
                    )
            from models.autoports.google_gemma_4_31b_it.tests.optimized_trace import capture

            tid, traced = capture(mesh, lambda: model.decode_forward(dx, **kw))
            try:
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                assert torch.equal(host(traced), tensors[count][1])
                start = time.perf_counter()
                for _ in range(a.replays):
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                results[str(count)]["decode_trace_host_us"] = (time.perf_counter() - start) * 1e6 / a.replays
                results[str(count)]["trace_replays"] = a.replays + 1
                assert torch.equal(host(traced), tensors[count][1])
                if a.stress:
                    stress_outputs[count] = []
                    for step, row in enumerate(stress_host):
                        if step == a.replays // 2:
                            other = model.prefill_forward(px, cos=pc, sin=ps, kv_cache=cache2, page_table=page2)
                            del other
                        for source, dest in zip(row, (dx, kw["cos"], kw["sin"], kw["current_pos"], page)):
                            ttnn.copy_host_to_device_tensor(source, dest)
                        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                        stress_outputs[count].append(host(traced))
                    results[str(count)].update(
                        stress_replays=a.replays, trace_id=str(tid), captures=1, interleaved_other_cache_prefill=True
                    )
            finally:
                ttnn.release_trace(mesh, tid)
            del traced
            if a.stress:
                del cache2, page2, stress_host, row, source, dest
            print("DONE", count, flush=True)
        finally:
            ttnn.close_mesh_device(mesh)
        del model, cache, pref, dec, px, pc, ps, dx, kw, page
    finite = {
        str(count): {"prefill": bool(torch.isfinite(pair[0]).all()), "decode": bool(torch.isfinite(pair[1]).all())}
        for count, pair in tensors.items()
    }
    results["runtime_fallback_audit"] = {
        "throw_exception_on_fallback": bool(ttnn.CONFIG.throw_exception_on_fallback),
        "prefill_decode_host_conversion_guard": True,
    }
    results["finite"] = finite
    results["cache_pcc"] = [pcc(x, y) for x, y in zip(cache_outputs[1], cache_outputs[4])]
    results["unused_pages_unchanged"] = True
    assert min(results["cache_pcc"]) >= 0.995
    if a.stress:
        values = [pcc(x, y) for x, y in zip(stress_outputs[1], stress_outputs[4])]
        results["stress_min_pcc"] = min(values)
        results["stress_final_pcc"] = values[-1]
        assert all(v >= 0.995 for v in values), values

    results.update(
        prefill_pcc=pcc(tensors[1][0], tensors[4][0]),
        decode_pcc=pcc(tensors[1][1], tensors[4][1]),
        layer_type=c.layer_types[a.layer],
        batch=a.batch,
        seq=a.seq,
        residual="replicated" if a.replicated else "sharded",
        candidate=json.loads(a.candidate) if a.candidate else None,
    )
    print(json.dumps(results), flush=True)
    Path(a.result).write_text(json.dumps(results, indent=2) + "\n")
    assert min(results["prefill_pcc"], results["decode_pcc"]) >= 0.995


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--seq", type=int, default=33)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--cyclic", action="store_true")
    p.add_argument("--distinct-offsets", action="store_true")
    p.add_argument("--stress", action="store_true")
    p.add_argument("--candidate")
    p.add_argument("--inputs")
    p.add_argument("--replays", type=int, default=128)
    p.add_argument("--replicated", action="store_true")
    p.add_argument("--result", required=True)
    args = p.parse_args()
    source_paths = [
        Path(__file__),
        Path(__file__).parents[1] / "tt/multichip_decoder.py",
        Path(__file__).parents[1] / "tt/optimized_decoder.py",
        Path(__file__).with_name("multichip_candidate.py"),
    ]
    Path(args.result).with_suffix(".manifest.json").write_text(
        json.dumps(
            dict(
                command=sys.argv,
                source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                semantic_hashes={
                    str(p): hashlib.sha256(
                        ast.dump(ast.parse(p.read_text()), include_attributes=False).encode()
                    ).hexdigest()
                    for p in source_paths
                },
                env={
                    k: os.getenv(k)
                    for k in [
                        "TT_METAL_WATCHER",
                        "TT_METAL_TRACE_ALLOC_TRACKING",
                        "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
                    ]
                },
            ),
            indent=2,
        )
        + "\n"
    )
    with ttnn.manage_config("throw_exception_on_fallback", True):
        run(args)
