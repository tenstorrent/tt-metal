# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Same-input equivalence and warmed trace timing for individual graph rewrites."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.run_decoder import load_layer
from models.autoports.google_gemma_4_26b_a4b_it.tests.runtime_audit import device_only
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import FunctionalDecoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.fused_decoder import (
    BroadcastQKV,
    BroadcastRouter,
    FusedSharedMLP,
    PackedExperts,
)
from models.autoports.google_gemma_4_26b_a4b_it.tt.precision_ops import rms_norm
from models.common.utility_functions import comp_pcc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(8)
    cfg = AutoConfig.from_pretrained(Path(__file__).parent).text_config
    hf = load_layer(cfg, args.layer, True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    results = []
    try:
        decoder = FunctionalDecoder.from_state_dict(
            hf.state_dict(), hf_config=cfg, layer_idx=args.layer, mesh_device=mesh
        )

        def upload(x):
            return ttnn.from_torch(x, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        def compare(name, baseline, candidates):
            with device_only():
                reference = baseline()
            host = ttnn.to_torch(reference).float()
            reference.deallocate(True)
            for candidate, fn in [("baseline", baseline)] + candidates:
                with device_only():
                    warm = fn()
                actual = ttnn.to_torch(warm).float()
                passing, pcc = comp_pcc(host, actual, 0.995)
                warm.deallocate(True)
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                with device_only():
                    output = fn()
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                times = []
                for _ in range(3):
                    start = time.perf_counter_ns()
                    for _ in range(30):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    times.append((time.perf_counter_ns() - start) / 30000)
                ttnn.release_trace(mesh, trace)
                output.deallocate(True)
                row = dict(
                    component=name,
                    candidate=candidate,
                    pcc=float(pcc),
                    passed=bool(passing),
                    exact_equal=torch.equal(host, actual),
                    max_abs=float((host - actual).abs().max()),
                    traced_host_us=statistics.median(times),
                )
                results.append(row)
                args.output.write_text(
                    json.dumps(
                        dict(layer_type=cfg.layer_types[args.layer], real_weights=True, results=results), indent=2
                    )
                    + "\n"
                )
                print(row, flush=True)
                assert passing, row

        x = upload(torch.randn(1, 1, 1, cfg.hidden_size).bfloat16())
        normed = rms_norm(x, cfg.rms_norm_eps, decoder.input_norm_weight)
        qkv = decoder.layer.self_attn.source.weights.wqkv
        candidates = []
        for group in [256, 512, 1024, 2048, 4096, 8192, 16384]:
            fused = BroadcastQKV(qkv, group)
            candidates.append((f"broadcast_{group}", lambda f=fused: f(normed)))
        compare("qkv", lambda: qkv(normed), candidates)
        router = decoder.layer.moe.router
        fused_router = BroadcastRouter(router)
        compare("router", lambda: router(normed), [("broadcast_scale", lambda: fused_router(normed))])
        for length in [1, 32]:
            inp = upload(torch.randn(1, 1, length, cfg.hidden_size).bfloat16())
            shared = decoder.layer.shared_mlp
            fused_shared = FusedSharedMLP(shared)
            compare(f"shared_{length}", lambda: shared(inp), [("gelu_mul", lambda: fused_shared(inp))])
            routes = router(inp)
            experts = decoder.layer.moe.experts
            packed = PackedExperts(experts, fused_gelu=False)
            activated = PackedExperts(experts, fused_gelu=True, fuse_decode_gelu=True)
            compare(
                f"experts_{length}",
                lambda: experts(ttnn.clone(inp), routes),
                [
                    ("packed", lambda: packed(ttnn.clone(inp), routes)),
                    ("packed_gelu", lambda: activated(ttnn.clone(inp), routes)),
                ],
            )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
