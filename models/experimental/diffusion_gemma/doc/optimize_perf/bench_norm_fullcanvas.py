# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""dg-08 L1-residency pass HIGH-4 — collapse the chunked RMSNorm to one full-canvas width-sharded norm.

Per-norm correctness + timing, single mesh-open (contended box). For each candidate RMSNorm on a
256-row canvas: PCC(chunked-path, full-canvas-path) — RMSNorm is per-row so block_h=8 is per-row
EQUIVALENT to 8x block_h=1 (expect ~0.99999x, NOT exactly 1.0: the bf16 reduction/accumulation ORDER
differs) — and chunked vs full-canvas device time (the slice/concat glue the full-canvas path removes).
Covers BOTH branches taken by ``_chunked_norm_forward``: the WEIGHTED gemma4 ``norm.forward`` fast-path
(input_layernorm etc., has tt_weight) AND the with_scale=False / tt_weight=None branch (``_rms_norm_dram``
vs ``_fullcanvas_norm(weight=None)``), exercised by a ``_NoWeightNorm`` stub. Also logs, via
RESULT_NORM_KIND, whether each REAL denoise-path layer norm actually takes the no-weight branch (so a
default-flip gate knows if that branch even fires at >32 rows in practice).

Run (device-free window):
  DG_CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it \
    python models/experimental/diffusion_gemma/doc/optimize_perf/bench_norm_fullcanvas.py --iters 40

Markers: RESULT_NORM name=.. chunked_ms=.. full_ms=.. pcc=.. ; RESULT_NORM_KIND name=.. with_scale=.. has_weight=..
"""
from __future__ import annotations

import argparse
import os
import time

import torch
from loguru import logger

import ttnn
from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
from models.experimental.diffusion_gemma.tt import denoise_forward as DF

CKPT = os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it")


def _pcc(a, b):
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom == 0:
        return 1.0 if a.norm() == b.norm() else 0.0
    return (torch.dot(a, b) / denom).item()


def _to_host(t):
    dev = t.device()
    if dev is not None and hasattr(dev, "get_num_devices") and dev.get_num_devices() > 1:
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
    return ttnn.to_torch(t).float()


def _time(fn, iters, mesh):
    fn()
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - t0) * 1e3 / iters


def run(iters, canvas_length):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT, None)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=1300000000)
    try:
        mi = build_tt_model_from_checkpoint_dir(
            mesh, CKPT, max_batch_size=1, max_seq_len=512, num_layers=2, create_kv_cache=True
        )
        tt_model = mi.tt_model
        H = tt_model.hf_config.hidden_size
        layer = tt_model.layers[0]
        moe_layer = None
        for lyr in tt_model.layers:
            if getattr(lyr, "enable_moe_block", False):
                moe_layer = lyr
                break

        # Report which REAL denoise-path layer norms take the no-weight (with_scale=False,
        # tt_weight=None) branch at >32 rows -> tells a default-flip gate if _fullcanvas_norm(weight=None)
        # even fires in practice. Inspect a representative set across the layer + MoE + router.
        real_norms = [
            ("input_layernorm", getattr(layer, "input_layernorm", None)),
            ("post_attention_layernorm", getattr(layer, "post_attention_layernorm", None)),
            ("pre_feedforward_layernorm", getattr(layer, "pre_feedforward_layernorm", None)),
            ("post_feedforward_layernorm", getattr(layer, "post_feedforward_layernorm", None)),
        ]
        if moe_layer is not None:
            real_norms += [
                ("moe.post_feedforward_layernorm", getattr(moe_layer, "post_feedforward_layernorm", None)),
                ("moe.router.norm", getattr(getattr(moe_layer, "moe", None), "router", None)),
            ]
        for nm, obj in real_norms:
            if obj is None:
                continue
            n = getattr(obj, "norm", obj)  # router exposes .norm
            ws = getattr(n, "with_scale", None)
            hw = getattr(n, "tt_weight", None) is not None
            print(f"RESULT_NORM_KIND name={nm} with_scale={ws} has_weight={hw}", flush=True)

        # Candidate norms to PCC/time: weighted layer norms + a with_scale=False no-weight STUB that
        # forces _chunked_norm_forward's no-weight branch (_rms_norm_dram vs _fullcanvas_norm(weight=None)).
        class _NoWeightNorm:
            with_scale = False
            tt_weight = None

            def __init__(self, eps):
                self.eps = eps

            def forward(self, x):  # only used on the <=32-row path; >32 rows gate to dram/fullcanvas
                return ttnn.rms_norm(x, epsilon=self.eps)

        eps = float(getattr(layer.input_layernorm, "eps", 1e-6))
        norms = [("input_layernorm", layer.input_layernorm)]
        if moe_layer is not None:
            norms.append(("post_feedforward_layernorm", moe_layer.post_feedforward_layernorm))
        norms.append(("noweight_stub", _NoWeightNorm(eps)))

        def mk_hidden(scale=1.0):
            host = torch.randn(1, 1, canvas_length, H, dtype=torch.float32) * scale
            return ttnn.from_torch(
                host,
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        for name, norm in norms:
            hs = mk_hidden()

            os.environ["DG_NORM_FULLCANVAS"] = "0"
            chunked = DF._chunked_norm_forward(norm, hs)
            chunked_host = _to_host(chunked)
            chunked.deallocate(True)

            os.environ["DG_NORM_FULLCANVAS"] = "1"
            try:
                full = DF._chunked_norm_forward(norm, hs)
            except Exception as e:
                print(f"RESULT_NORM name={name} ERROR {type(e).__name__}: {str(e)[:220]}", flush=True)
                logger.warning(f"{name} fullcanvas failed: {type(e).__name__}: {str(e)[:400]}")
                hs.deallocate(True)
                continue
            full_host = _to_host(full)
            full.deallocate(True)
            pcc = _pcc(chunked_host, full_host)

            os.environ["DG_NORM_FULLCANVAS"] = "0"
            c_ms = _time(lambda: DF._chunked_norm_forward(norm, hs).deallocate(True), iters, mesh)
            os.environ["DG_NORM_FULLCANVAS"] = "1"
            f_ms = _time(lambda: DF._chunked_norm_forward(norm, hs).deallocate(True), iters, mesh)

            print(f"RESULT_NORM name={name} chunked_ms={c_ms:.4f} full_ms={f_ms:.4f} pcc={pcc:.6f}", flush=True)
            hs.deallocate(True)

        os.environ["DG_NORM_FULLCANVAS"] = "0"
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--canvas-length", type=int, default=256)
    args = ap.parse_args()
    run(args.iters, args.canvas_length)


if __name__ == "__main__":
    main()
