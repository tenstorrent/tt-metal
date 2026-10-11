# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Persistent decode trace interleaved with prepared short traces and eager long/append prefills."""

import argparse
import gc
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerTextDecoderLayer
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tests.prepare_optimized_inputs import load_state
from models.autoports.meta_models_muse_glimmer_30b.tests.run_optimized import (
    CONFIG,
    ROOT,
    AutoConfig,
    LayerCache,
    MuseGlimmerTextRotaryEmbedding,
    device_only,
    pcc,
    reference,
    to_device,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import MultichipDecoder
from models.autoports.meta_models_muse_glimmer_30b.tt.prepared_prefill import PreparedPrefill


def read(tensor):
    pieces = [ttnn.to_torch(v) for v in ttnn.get_device_tensors(tensor)]
    assert all(bool(torch.isfinite(p).all()) for p in pieces)
    assert all(torch.equal(pieces[0], p) for p in pieces[1:])
    return pieces[0]


def run(layer_idx, output=None, policy=None, batch=2, short_only=False, prepared_policy=None):
    assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "0"
    torch.set_num_threads(8)
    torch.manual_seed(251)
    cfg = AutoConfig.from_pretrained(CONFIG).text_config
    cfg._attn_implementation = "sdpa"
    state = load_state(layer_idx)
    with torch.device("meta"):
        ref = MuseGlimmerTextDecoderLayer(cfg, layer_idx)
    ref.load_state_dict(state, assign=True)
    ref.eval()
    recorded = torch.load(ROOT / f"doc/optimized_decoder/inputs/layer_{layer_idx}.pt", weights_only=True)

    def activation(start, count):
        return torch.cat(
            [recorded[:, (torch.arange(start, start + count) + 17 * b) % recorded.shape[1]] for b in range(batch)], 0
        ).contiguous()

    rotary = MuseGlimmerTextRotaryEmbedding(cfg)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    tid = None
    prefill = None
    try:
        layer = MultichipDecoder.from_state_dict(
            state,
            hf_config=cfg,
            layer_idx=layer_idx,
            mesh_device=mesh,
            policy=json.loads(policy) if policy else None,
        )
        kv = layer.allocate_cache(batch_size=batch)
        pt = torch.arange(batch * 4096, dtype=torch.int32).reshape(batch, 4096)
        tt_pt = to_device(pt, mesh, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        tt_decode_pt = to_device(pt, mesh, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        template = torch.zeros(1, 1, batch, 6656, dtype=torch.bfloat16)
        inputs = {
            "x": to_device(template, mesh),
            "current_pos": to_device(torch.zeros(batch, dtype=torch.int32), mesh, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            "rope_cos": to_device(torch.ones(1, batch, 1, 128), mesh),
            "rope_sin": to_device(torch.zeros(1, batch, 1, 128), mesh),
        }
        ordered, reversed_slots = list(range(batch)), list(reversed(range(batch)))
        lengths = [1, 2, 17, 31, 32, 33, 63, 64, 65]
        if not short_only:
            lengths += [1023, 1024, 1025, 2049, 4095, 4096, 4097, 4159]
        cases = [(0, n, ordered) for n in lengths]
        cases += (
            [(33, 32, reversed_slots)]
            if short_only
            else [(33, 1057, reversed_slots), (1024, 1025, ordered), (2048, 33, reversed_slots)]
        )
        short_inputs = {
            n: to_device(torch.zeros(batch, 1, n, 6656, dtype=torch.bfloat16), mesh)
            for n in (1, 2, 17, 31, 32, 33, 63, 64)
        }
        prefill = PreparedPrefill(
            layer,
            mesh=mesh,
            kv_cache=kv,
            page_table=tt_pt,
            batch=batch,
            policy=json.loads(prepared_policy) if prepared_policy else None,
        )
        # Decode creates persistent L1 CCL staging. Interleaved norm planning
        # includes live allocator occupancy in its program-cache key, so this
        # state must exist before preparing prefill. Retain one decode output
        # too, matching the live residual footprint of the eventual trace.
        with device_only():
            warm = layer.decode_forward(**inputs, kv_cache=kv, page_table=tt_decode_pt)
        prefill.prepare()
        # Warm every physical shape/offset/slot path before capture. All temporary
        # tensors die before recording. KV from this warmup is replaced per request.
        for prefix, tail, slots in cases:
            for start, length in [(0, prefix), (prefix, tail)] if prefix else [(0, tail)]:
                x = to_device(torch.zeros(batch, 1, length, 6656, dtype=torch.bfloat16), mesh)
                with device_only():
                    y = layer.prefill_forward(x, kv_cache=kv, page_table=tt_pt, start_pos=start, slots=slots)
                del x, y
        prefill.capture()
        del warm
        gc.collect()
        ttnn.synchronize_device(mesh)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        with device_only():
            out = layer.decode_forward(**inputs, kv_cache=kv, page_table=tt_decode_pt)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        # Only this exact backing allocation is exempt. No aliases persist; read
        # immediately after its trace. Eager prefill may overwrite it before that.
        trace_allocation_tracker.acknowledge_corruptible(out)
        rows = []
        for prefix, tail, slots in cases + [(0, 33, reversed_slots), (0, 31, ordered)]:
            length = prefix + tail
            pt = torch.randperm(batch * 4096).reshape(batch, 4096).int()
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), tt_pt)
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(pt[slots].contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), tt_decode_pt
            )
            x = activation(0, length)
            cache = LayerCache()
            prefill_scores = []
            prefill_user_scores = []
            for start, count in [(0, prefix), (prefix, tail)] if prefix else [(0, tail)]:
                part = x[:, start : start + count]
                if start == 0 and count <= 64:
                    tt_x = short_inputs[count]
                    ttnn.copy_host_to_device_tensor(
                        ttnn.from_torch(part[:, None], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), tt_x
                    )
                    with device_only():
                        y = prefill(tt_x, page_table=tt_pt, slots=slots)
                else:
                    tt_x = to_device(part[:, None], mesh)
                    with device_only():
                        y = layer.prefill_forward(tt_x, kv_cache=kv, page_table=tt_pt, start_pos=start, slots=slots)
                got = read(y)[:, 0]
                del y, tt_x
                gc.collect()
                expected = reference(ref, part, start, cache, rotary)
                prefill_score = pcc(got, expected)
                user_scores = [pcc(got[b], expected[b]) for b in range(batch)]
                assert min(user_scores) >= 0.995, (start, count, user_scores)
                prefill_user_scores.append(user_scores)
                prefill_scores.append(prefill_score)
            token = activation(length, 1)
            pos = torch.full((batch,), length, dtype=torch.int32)
            cos, sin = rotary(token, pos[:, None])
            refresh = {
                "x": token.transpose(0, 1)[None],
                "current_pos": pos,
                "rope_cos": cos[None],
                "rope_sin": sin[None],
            }
            for name, value in refresh.items():
                host = ttnn.from_torch(value, dtype=inputs[name].dtype, layout=inputs[name].layout)
                ttnn.copy_host_to_device_tensor(host, inputs[name])
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            got = read(out)[0].transpose(0, 1)
            expected = reference(ref, token, length, cache, rotary)
            decode_score = pcc(got, expected)
            decode_user_scores = [pcc(got[b], expected[b]) for b in range(batch)]
            if min(decode_user_scores) < 0.995:
                print("FAILED_DECODE", length, slots, decode_user_scores, flush=True)
                torch.save(
                    dict(
                        length=length,
                        slots=slots,
                        input=x,
                        expected=expected,
                        got=got,
                        page_table=pt,
                        prefill_user_pcc=prefill_user_scores,
                        decode_user_pcc=decode_user_scores,
                    ),
                    output.with_suffix(".failure.pt"),
                )
            assert min(decode_user_scores) >= 0.995, (length, decode_user_scores)
            rows.append(
                {
                    "length": length,
                    "prefix_length": prefix,
                    "tail_length": tail,
                    "slots": slots,
                    "trace_id": tid,
                    "prefill_pcc": min(prefill_scores),
                    "decode_pcc": decode_score,
                    "prefill_user_pcc": prefill_user_scores,
                    "decode_user_pcc": decode_user_scores,
                }
            )
            print("REUSED", rows[-1], flush=True)
        report = {
            "command": sys.argv,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "decoder_sha256": hashlib.sha256((ROOT / "tt/multichip_decoder.py").read_bytes()).hexdigest(),
            "inherited_decoder_sha256": hashlib.sha256((ROOT / "tt/optimized_decoder.py").read_bytes()).hexdigest(),
            "prepared_prefill_sha256": hashlib.sha256((ROOT / "tt/prepared_prefill.py").read_bytes()).hexdigest(),
            "layer": layer_idx,
            "policy": layer.policy,
            "tracking_enabled": True,
            "program_cache_included": True,
            "preparation_order": "bind all input pools; decode persistent CCL setup; retain representative decode output; prepare short buckets and AR buffers before eager variants; capture two prefill buckets; release representative output; capture decode once",
            "batch": batch,
            "short_only": short_only,
            "prefill_trace_policy": prefill.policy,
            "prefill_captured_policies": prefill.captured_policies,
            "prefill_setup_ms": dict(prepare=prefill.prepare_ms, capture=prefill.capture_ms),
            "capture_count": 1,
            "prefill_capture_count": prefill.capture_count,
            "prefill_trace_ids": dict(prefill.traces),
            "prefill_replay_count": prefill.replay_count,
            "request_release_count": 0,
            "request_recapture_count": 0,
            "teardown_release_count": 1,
            "execution_signature": f"TP4 replicated B{batch} BF16 DRAM x/position/RoPE/page buffers bound before capture; full context paged KV",
            "rows": rows,
        }
        prefill.close()
        report["prefill_teardown_release_count"] = prefill.release_count
        destination = output or ROOT / f"doc/multichip_decoder/trace_reuse_{layer_idx}.json"
        destination.write_text(json.dumps(report, indent=2, default=str))
        for key, source in {"runner": Path(__file__), "prepared_prefill": ROOT / "tt/prepared_prefill.py"}.items():
            destination.with_suffix(f".{key}.py.gz").write_bytes(gzip.compress(source.read_bytes(), mtime=0))
        print("TRACE_REUSE_PASS", flush=True)
    finally:
        if tid is not None:
            ttnn.release_trace(mesh, tid)
        if prefill is not None:
            prefill.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--policy")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--short-only", action="store_true")
    parser.add_argument("--prepared-policy")
    args = parser.parse_args()
    run(args.layer, args.output, args.policy, args.batch, args.short_only, args.prepared_policy)
