# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serial real-weight layer equivalence and warmed trace fusion experiments."""

import argparse
import gc
import hashlib
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tests.fusion_candidates import FusedDecoder
from models.autoports.meta_models_muse_glimmer_30b.tests.run_functional import (
    CONFIG,
    ROOT,
    AutoConfig,
    FunctionalDecoder,
    LayerCache,
    MuseGlimmerTextRotaryEmbedding,
    device_only,
    pcc,
    read,
    reference,
    to_device,
)
from models.autoports.meta_models_muse_glimmer_30b.tests.run_fused import weights


def main(args):
    torch.set_num_threads(8)
    torch.manual_seed(123)
    cfg = AutoConfig.from_pretrained(CONFIG).text_config
    cfg._attn_implementation = "sdpa"
    ref, state = weights(cfg, args.layer, True)
    rotary = MuseGlimmerTextRotaryEmbedding(cfg)
    x = torch.randn(args.batch, args.length, 6656).bfloat16()
    tok = torch.randn(args.batch, 1, 6656).bfloat16()
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
            print("CANDIDATE", name, flush=True)
            flags = frozenset(name.split("+")) if name != "baseline" else frozenset()
            cls = FunctionalDecoder if name == "baseline" else FusedDecoder
            if name == "final":
                from models.autoports.meta_models_muse_glimmer_30b.tt.fused_decoder import FusedDecoder as FinalDecoder

                cls = FinalDecoder
            if name == "physical_join":
                from models.autoports.meta_models_muse_glimmer_30b.tests.probe_physical_join import PhysicalJoinDecoder

                cls = PhysicalJoinDecoder
            cls.FUSIONS = flags
            row = dict(candidate=name, layer=args.layer, batch=args.batch, length=args.length)
            try:
                layer = cls.from_state_dict(state, hf_config=cfg, layer_idx=args.layer, mesh_device=mesh)
                kv = layer.allocate_cache(batch_size=args.batch)
                with device_only():
                    y = layer.prefill_forward(xp, kv_cache=kv, page_table=tp)
                got = read(y)[:, 0].clone()
                del y
                row["prefill_hf_pcc"] = pcc(expected, got)
                if baseline_p is None:
                    baseline_p = got
                row["prefill_functional_pcc"] = pcc(baseline_p, got)
                assert min(row["prefill_hf_pcc"], row["prefill_functional_pcc"]) >= 0.995, row
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
                    y = layer.decode_forward(**inputs, kv_cache=kv, page_table=tp)
                del y
                ttnn.synchronize_device(mesh)
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                with device_only():
                    out = layer.decode_forward(**inputs, kv_cache=kv, page_table=tp)
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                for _ in range(3):
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                gotd = read(out)[0].transpose(0, 1).clone()
                row["decode_hf_pcc"] = pcc(expected_decode, gotd)
                if baseline_d is None:
                    baseline_d = gotd
                row["decode_functional_pcc"] = pcc(baseline_d, gotd)
                assert min(row["decode_hf_pcc"], row["decode_functional_pcc"]) >= 0.995, row
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
                del kv, layer
                gc.collect()
            results.append(row)
            args.output.write_text(
                json.dumps(
                    dict(
                        command=sys.argv,
                        rows=results,
                        source_sha256={
                            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [
                                ROOT / "tt/fused_decoder.py",
                                ROOT / "tests/fusion_candidates.py",
                                ROOT / "tests/probe_physical_join.py",
                                Path(__file__),
                            ]
                        },
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
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--candidates",
        nargs="+",
        default=[
            "baseline",
            "rope",
            "qscale",
            "concat",
            "cache",
            "binary_act",
            "matmul_act",
            "pack_qkv",
            "pack_qkv+decode_heads",
            "pack_qkvg",
            "pack_mlp",
            "pack_mlp+swiglu",
        ],
    )
    main(p.parse_args())
