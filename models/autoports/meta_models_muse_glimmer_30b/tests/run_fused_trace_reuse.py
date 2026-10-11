# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Persistent decode trace interleaved with prepared eager prefill variants."""

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tests.run_fused import (
    CONFIG,
    ROOT,
    AutoConfig,
    FusedDecoder,
    LayerCache,
    MuseGlimmerTextRotaryEmbedding,
    device_only,
    pcc,
    read,
    reference,
    to_device,
    weights,
)


def run(layer_idx):
    assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "0"
    torch.set_num_threads(8)
    torch.manual_seed(251)
    cfg = AutoConfig.from_pretrained(CONFIG).text_config
    cfg._attn_implementation = "sdpa"
    ref, state = weights(cfg, layer_idx, True)
    rotary = MuseGlimmerTextRotaryEmbedding(cfg)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    tid = None
    try:
        layer = FusedDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh)
        kv = layer.allocate_cache(batch_size=2)
        pt = torch.arange(8192, dtype=torch.int32).reshape(2, 4096)
        tt_pt = to_device(pt, mesh, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        tt_decode_pt = to_device(pt, mesh, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        template = torch.zeros(1, 1, 2, 6656, dtype=torch.bfloat16)
        inputs = {
            "x": to_device(template, mesh),
            "current_pos": to_device(torch.zeros(2, dtype=torch.int32), mesh, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            "rope_cos": to_device(torch.ones(1, 2, 1, 128), mesh),
            "rope_sin": to_device(torch.zeros(1, 2, 1, 128), mesh),
        }
        cases = [(0, n, [0, 1]) for n in [31, 32, 33, 1023, 1024, 1025, 2049, 17]]
        cases += [(33, 1057, [1, 0]), (1024, 1025, [0, 1]), (2048, 33, [1, 0])]
        # Warm every physical shape/offset/slot path before capture. All temporary
        # tensors die before recording. KV from this warmup is replaced per request.
        for prefix, tail, slots in cases:
            for start, length in [(0, prefix), (prefix, tail)] if prefix else [(0, tail)]:
                x = to_device(torch.zeros(2, 1, length, 6656, dtype=torch.bfloat16), mesh)
                with device_only():
                    y = layer.prefill_forward(x, kv_cache=kv, page_table=tt_pt, start_pos=start, slots=slots)
                del x, y
        with device_only():
            warm = layer.decode_forward(**inputs, kv_cache=kv, page_table=tt_decode_pt)
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
        for prefix, tail, slots in cases + [(0, 33, [1, 0]), (0, 31, [0, 1])]:
            length = prefix + tail
            pt = torch.randperm(8192).reshape(2, 4096).int()
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), tt_pt)
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(pt[slots].contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), tt_decode_pt
            )
            x = torch.randn(2, length, 6656).bfloat16()
            cache = LayerCache()
            prefill_scores = []
            for start, count in [(0, prefix), (prefix, tail)] if prefix else [(0, tail)]:
                part = x[:, start : start + count]
                tt_x = to_device(part[:, None], mesh)
                with device_only():
                    y = layer.prefill_forward(tt_x, kv_cache=kv, page_table=tt_pt, start_pos=start, slots=slots)
                got = read(y)[:, 0]
                del y, tt_x
                gc.collect()
                expected = reference(ref, part, start, cache, rotary)
                prefill_score = pcc(got, expected)
                assert prefill_score >= 0.995
                prefill_scores.append(prefill_score)
            token = torch.randn(2, 1, 6656).bfloat16()
            pos = torch.full((2,), length, dtype=torch.int32)
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
            assert decode_score >= 0.995
            rows.append(
                {
                    "length": length,
                    "prefix_length": prefix,
                    "tail_length": tail,
                    "slots": slots,
                    "trace_id": tid,
                    "prefill_pcc": min(prefill_scores),
                    "decode_pcc": decode_score,
                }
            )
            print("REUSED", rows[-1], flush=True)
        report = {
            "command": sys.argv,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "decoder_sha256": hashlib.sha256((ROOT / "tt/fused_decoder.py").read_bytes()).hexdigest(),
            "layer": layer_idx,
            "tracking_enabled": True,
            "program_cache_included": True,
            "batch": 2,
            "capture_count": 1,
            "request_release_count": 0,
            "request_recapture_count": 0,
            "teardown_release_count": 1,
            "execution_signature": "B2 BF16 DRAM x/position/RoPE/page buffers bound before capture; full context paged KV",
            "rows": rows,
        }
        (ROOT / f"doc/fused_decoder/trace_reuse_{layer_idx}.json").write_text(json.dumps(report, indent=2, default=str))
        print("TRACE_REUSE_PASS", flush=True)
    finally:
        if tid is not None:
            ttnn.release_trace(mesh, tid)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    run(parser.parse_args().layer)
