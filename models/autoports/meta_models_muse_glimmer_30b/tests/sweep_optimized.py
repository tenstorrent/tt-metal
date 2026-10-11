# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serial real-weight layer equivalence and warmed trace fusion experiments."""

import argparse
import gc
import gzip
import hashlib
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch
from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerTextDecoderLayer

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tests.prepare_optimized_inputs import load_state
from models.autoports.meta_models_muse_glimmer_30b.tests.run_functional import (
    CONFIG,
    ROOT,
    AutoConfig,
    LayerCache,
    MuseGlimmerTextRotaryEmbedding,
    device_only,
    read,
    reference,
    to_device,
)
from models.autoports.meta_models_muse_glimmer_30b.tests.run_optimized import pcc
from models.autoports.meta_models_muse_glimmer_30b.tt.fused_decoder import FusedDecoder
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import OptimizedDecoder


def main(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [ROOT / "tt/fused_decoder.py", ROOT / "tt/optimized_decoder.py", args.policies, Path(__file__)]
    }
    for source in (ROOT / "tt/optimized_decoder.py", Path(__file__)):
        args.output.with_name(args.output.stem + "_" + source.name + ".gz").write_bytes(
            gzip.compress(source.read_bytes(), mtime=0)
        )
    torch.set_num_threads(8)
    torch.manual_seed(123)
    cfg = AutoConfig.from_pretrained(CONFIG).text_config
    cfg._attn_implementation = "sdpa"
    state = load_state(args.layer)
    with torch.device("meta"):
        ref = MuseGlimmerTextDecoderLayer(cfg, args.layer)
    ref.load_state_dict(state, assign=True)
    ref.eval()
    rotary = MuseGlimmerTextRotaryEmbedding(cfg)
    recorded = torch.load(ROOT / f"doc/optimized_decoder/inputs/layer_{args.layer}.pt", weights_only=True)
    x = recorded[:, torch.arange(args.length) % recorded.shape[1]].repeat(args.batch, 1, 1).contiguous()
    tok = recorded[:, [args.length % recorded.shape[1]]].repeat(args.batch, 1, 1).contiguous()
    assert x.shape[1] == args.length and tok.shape[1] == 1
    policies = json.loads(args.policies.read_text())
    policies["default"] = dict(OptimizedDecoder.POLICY)
    pt = torch.randperm(args.batch * 4096).reshape(args.batch, 4096).int()
    pos = torch.full((args.batch,), args.length, dtype=torch.int32)
    cos, sin = rotary(tok, pos[:, None])
    cache = LayerCache()
    expected = reference(ref, x, 0, cache, rotary)
    expected_decode = reference(ref, tok, args.length, cache, rotary)
    del ref, cache
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    results = []
    baseline_p = baseline_d = None
    try:
        xp = to_device(x[:, None], mesh)
        inputs = dict(
            x=to_device(tok.transpose(0, 1)[None], mesh),
            current_pos=to_device(pos, mesh, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            rope_cos=to_device(cos[None], mesh),
            rope_sin=to_device(sin[None], mesh),
        )
        tp = to_device(pt, mesh, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        for name in args.candidates:
            tid = None
            layer = None
            kv = None
            decode_inputs = None
            print("CANDIDATE", name, flush=True)
            cls = FusedDecoder if name == "fused" else OptimizedDecoder
            if name != "fused":
                cls.POLICY = policies[name]
            row = dict(
                candidate=name,
                layer=args.layer,
                batch=args.batch,
                length=args.length,
                policy=policies.get(name),
                inputs="recorded official-HF text activations",
                decode_input_sharded=getattr(args, "decode_input_sharded", False),
            )
            try:
                layer = cls.from_state_dict(state, hf_config=cfg, layer_idx=args.layer, mesh_device=mesh)
                decode_inputs = dict(inputs)
                if getattr(args, "decode_input_sharded", False):
                    assert name != "fused", "Separate optimized input-layout control only"
                    decode_inputs["x"] = ttnn.to_memory_config(inputs["x"], layer.residual_mem)
                kv = layer.allocate_cache(batch_size=args.batch)
                with device_only():
                    y = layer.prefill_forward(xp, kv_cache=kv, page_table=tp)
                got = read(y)[:, 0].clone()
                del y
                row["prefill_hf_pcc"] = pcc(expected, got)
                if baseline_p is None:
                    baseline_p = got
                row["prefill_first_candidate_pcc"] = pcc(baseline_p, got)
                assert row["prefill_hf_pcc"] >= 0.995, row
                for _ in range(2):
                    with device_only():
                        y = layer.prefill_forward(xp, kv_cache=kv, page_table=tp)
                    del y
                prefill = []
                for _ in range(3):
                    ttnn.synchronize_device(mesh)
                    t0 = time.perf_counter()
                    for __ in range(5):
                        with device_only():
                            y = layer.prefill_forward(xp, kv_cache=kv, page_table=tp)
                        del y
                    ttnn.synchronize_device(mesh)
                    prefill.append((time.perf_counter() - t0) * 1000 / 5)
                with device_only():
                    y = layer.decode_forward(**decode_inputs, kv_cache=kv, page_table=tp)
                del y
                ttnn.synchronize_device(mesh)
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                with device_only():
                    out = layer.decode_forward(**decode_inputs, kv_cache=kv, page_table=tp)
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                from ttnn.tools import trace_allocation_tracker

                trace_allocation_tracker.acknowledge_corruptible(out)
                for _ in range(3):
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                gotd = read(out)[0].transpose(0, 1).clone()
                row["decode_hf_pcc"] = pcc(expected_decode, gotd)
                if baseline_d is None:
                    baseline_d = gotd
                row["decode_first_candidate_pcc"] = pcc(baseline_d, gotd)
                assert row["decode_hf_pcc"] >= 0.995, row
                decode = []
                for _ in range(5):
                    ttnn.synchronize_device(mesh)
                    t0 = time.perf_counter()
                    for __ in range(100):
                        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    decode.append((time.perf_counter() - t0) * 1000 / 100)
                row.update(
                    prefill_ms=statistics.median(prefill),
                    decode_ms=statistics.median(decode),
                    prefill_samples_ms=prefill,
                    decode_samples_ms=decode,
                    status="pass",
                )
                del out
            except Exception as e:
                row.update(status="error", error=str(e))
                traceback.print_exc()
            finally:
                if tid is not None:
                    ttnn.release_trace(mesh, tid)
                del kv, layer, decode_inputs
                gc.collect()
            results.append(row)
            args.output.write_text(
                json.dumps(
                    dict(
                        command=sys.argv,
                        rows=results,
                        source_sha256=source_hashes,
                    ),
                    indent=2,
                )
            )
            print("RESULT", json.dumps(row), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--length", type=int, default=1024)
    p.add_argument("--decode-input-sharded", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--policies", type=Path, default=ROOT / "doc/optimized_decoder/policies.json")
    p.add_argument(
        "--candidates",
        nargs="+",
        default=["fused", "bfp8", "mlp_lofi", "mlp4_gateup", "mlp4_all", "attention4", "sharded8"],
    )
    main(p.parse_args())
