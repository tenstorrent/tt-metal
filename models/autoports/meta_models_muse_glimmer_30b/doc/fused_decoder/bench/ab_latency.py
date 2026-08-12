# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Wall-clock A/B: functional vs fused decoder, warmed prefill and traced decode.

Fast iteration harness for the fusing loop.  The committed before/after numbers
come from the Tracy device-profiler runs (``tt-perf-report``); this script is
the cheap screen used to accept or reject a candidate rewrite.

Every window is measured ``--rounds`` times and reported as ``min (all rounds)``
so a sub-1 % delta can be told apart from the harness's own jitter — prefill
here carries a ~1-2 % round-to-round spread, which is exactly the size of
several of the rejected candidates' deltas.

    python .../bench/ab_latency.py [--impl functional,fused] [--kinds sliding,full]
"""
from __future__ import annotations

import argparse
import time

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tests import reference as R
from models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder import (
    FunctionalDecoder,
    reference_layer_indices,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.fused_decoder import FusedDecoder

PAGE_BLOCK = 64
MAX_SEQ = 16384
IMPLS = {"functional": FunctionalDecoder, "fused": FusedDecoder}


def _load_variants():
    """Candidate rewrites live next to this script (see ``variants.py``)."""
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location("mg_fused_variants", pathlib.Path(__file__).with_name("variants.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    IMPLS.update(
        packed_gate_up=module.PackedGateUpDecoder,
        swiglu=module.SwigluDecoder,
        packed_qkv_gate=module.PackedQkvGateDecoder,
        fused_kv_update=module.FusedKvUpdateDecoder,
    )


def page_table(mesh, batch, max_seq_len, seed=7):
    blocks = (max_seq_len + PAGE_BLOCK - 1) // PAGE_BLOCK
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(batch * blocks, generator=gen).reshape(batch, blocks).to(torch.int32)
    return ttnn.from_torch(
        perm, device=mesh, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def to_dev(mesh, hidden):
    flat = hidden.reshape(1, 1, hidden.shape[0] * hidden.shape[1], hidden.shape[2])
    return ttnn.from_torch(
        flat, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def pos_tensors(mesh, positions):
    cur = ttnn.from_torch(
        positions.to(torch.int32),
        device=mesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    rope = ttnn.from_torch(
        positions.reshape(1, -1).to(torch.int32),
        device=mesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return cur, rope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="functional,fused")
    ap.add_argument("--kinds", default="sliding,full")
    ap.add_argument("--prefill-seq", type=int, default=8192)
    ap.add_argument("--decode-iters", type=int, default=64)
    ap.add_argument("--decode-context", type=int, default=2048)
    ap.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="measurement rounds per window; the reported number is the minimum, "
        "and the per-round spread is printed so sub-1%% deltas can be judged",
    )
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if any(name not in IMPLS for name in args.impl.split(",")):
        _load_variants()

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    ttnn.SetDefaultDevice(mesh)
    try:
        idxs = reference_layer_indices(R.hf_config())
        for kind in args.kinds.split(","):
            layer_idx = idxs[kind]
            state_dict = R.synthetic_state_dict(layer_idx)
            for impl in args.impl.split(","):
                cls = IMPLS[impl]
                dec = cls.from_state_dict(
                    state_dict,
                    hf_config=R.hf_config(),
                    layer_idx=layer_idx,
                    mesh_device=mesh,
                    max_batch_size=1,
                    max_seq_len=MAX_SEQ,
                    page_block_size=PAGE_BLOCK,
                    prefill_chunk_size=8192,
                )
                pt = page_table(mesh, 1, MAX_SEQ, seed=3)

                # ---- warmed prefill
                tt_hidden = to_dev(mesh, R.synthetic_hidden_states(1, args.prefill_seq, seed=42))
                for _ in range(2):
                    ttnn.deallocate(dec.prefill_forward(tt_hidden, page_table=pt, user_id=0))
                prefill_rounds = []
                for _ in range(args.rounds):
                    ttnn.synchronize_device(mesh)
                    t0 = time.perf_counter()
                    for _ in range(3):
                        ttnn.deallocate(dec.prefill_forward(tt_hidden, page_table=pt, user_id=0))
                    ttnn.synchronize_device(mesh)
                    prefill_rounds.append((time.perf_counter() - t0) / 3 * 1e3)
                prefill_ms = min(prefill_rounds)
                ttnn.deallocate(tt_hidden)

                # ---- traced decode
                tt_token = to_dev(mesh, R.synthetic_hidden_states(1, 1, seed=44))
                cur, rope = pos_tensors(mesh, torch.tensor([args.decode_context]))
                warm = dec.decode_forward(tt_token, current_pos=cur, page_table=pt, rope_pos_ids=rope)
                ttnn.deallocate(warm)
                ttnn.synchronize_device(mesh)
                trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
                tt_out = dec.decode_forward(tt_token, current_pos=cur, page_table=pt, rope_pos_ids=rope)
                ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
                ttnn.synchronize_device(mesh)
                for _ in range(8):
                    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                decode_rounds = []
                for _ in range(args.rounds):
                    ttnn.synchronize_device(mesh)
                    t0 = time.perf_counter()
                    for _ in range(args.decode_iters):
                        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    decode_rounds.append((time.perf_counter() - t0) / args.decode_iters * 1e3)
                decode_ms = min(decode_rounds)
                ttnn.release_trace(mesh, trace_id)
                for t in (tt_token, cur, rope, pt, tt_out):
                    ttnn.deallocate(t)
                print(
                    f"AB {args.tag:12s} impl={impl:16s} kind={kind:8s} "
                    f"prefill{args.prefill_seq}=min {prefill_ms:7.2f} ms "
                    f"({'/'.join(f'{r:.2f}' for r in prefill_rounds)})  "
                    f"traced_decode@{args.decode_context}=min {decode_ms:6.3f} ms/token "
                    f"({'/'.join(f'{r:.3f}' for r in decode_rounds)})",
                    flush=True,
                )
                del dec
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
