# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Post-selection verification for the datatype-sweep winner, through the NORMAL construction path.

Builds the model via build_generator with NO precision kwargs, so it loads
doc/datatype_sweep/selected_precision_config.json by default (the required config artifact). Then:

  1. PROPAGATION CHECK: asserts load_selected_precision_policy() resolved to the selected file (not
     in-code defaults) and that the ACTUAL on-device weight dtypes + per-group math fidelities match
     the selected policy field-by-field. This proves build_generator (the factory the vLLM adapter
     also uses) consumes every selected weight/activation/CCL/KV/fidelity field.
  2. POST-SELECTION TOKEN-OUT: the same warmed no-readback token-out benchmark used by optimized full
     model (prompt128/gen128, capture->warm8->measure), recorded separately from teacher-forcing.

  cd /tmp && TT_METAL_HOME=<tree> PYTHONPATH=<repo> python <this>.py --out <json>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.poolside_laguna_xs_2_1.tt.generator import build_generator
from models.autoports.poolside_laguna_xs_2_1.tt.model import (
    SELECTED_PRECISION_CONFIG_PATH,
    load_selected_precision_policy,
)
from models.autoports.poolside_laguna_xs_2_1.tt.optimized_decoder import _DTYPE_TO_STR

MD = Path("/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1")


def _fid_of(dec, attr):
    ck = getattr(dec, attr)
    for name in ("LoFi", "HiFi2", "HiFi4"):
        if ck is dec._ck_by_fid[name]:
            return name
    return "?"


def _capture(mesh, body):
    body()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    body()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    return tid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", type=int, default=128)
    ap.add_argument("--gen", type=int, default=128)
    args = ap.parse_args()

    pol, src = load_selected_precision_policy()
    assert src == SELECTED_PRECISION_CONFIG_PATH, f"selected config not resolved from file: {src}"

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=1_500_000_000)
    res = {"selected_config_path": src, "selected_policy": pol.to_dict()}
    try:
        gen = build_generator(MD, mesh, max_seq_len=max(4096, args.prompt + 4 * args.gen + 64))
        m = gen.model
        moe = next(d for d in m.layers if d.cfg.is_moe)
        l0 = m.layers[0]
        # ---- propagation check: built vs selected ----
        built = {
            "attn_qkv": _DTYPE_TO_STR[moe.w["wqkv"].dtype],
            "attn_o": _DTYPE_TO_STR[moe.w["wo"].dtype],
            "moe_ff13": _DTYPE_TO_STR[moe.w["exp_gate_up"].dtype],
            "moe_ff2": _DTYPE_TO_STR[moe.w["exp_down"].dtype],
            "shared_ff13": _DTYPE_TO_STR[moe.w["sh_gate_up"].dtype],
            "shared_ff2": _DTYPE_TO_STR[moe.w["sh_down"].dtype],
            "dense_ff13": _DTYPE_TO_STR[l0.w["mlp_gate_up"].dtype],
            "dense_ff2": _DTYPE_TO_STR[l0.w["mlp_down"].dtype],
            "router": _DTYPE_TO_STR[moe.w["gate_w"].dtype],
            "lm_head": _DTYPE_TO_STR[m.lm_head_w.dtype],
            "kv_cache": _DTYPE_TO_STR[moe.policy.kv_cache],
            "ccl": _DTYPE_TO_STR[moe.policy.ccl],
            "fid_attn_qkv": _fid_of(moe, "_ck_qkv"),
            "fid_attn_o": _fid_of(moe, "_ck_o"),
            "fid_dense": _fid_of(l0, "_ck_dense"),
            "fid_shared": _fid_of(moe, "_ck_shared"),
            "fid_router": _fid_of(moe, "_ck_router"),
            "fid_moe": _fid_of(moe, "_ck_moe"),
        }
        sel = pol.to_dict()
        mism = {k: (built[k], sel[k]) for k in built if built[k] != sel[k]}
        res["propagation_built_vs_selected"] = built
        res["propagation_mismatches"] = mism
        res["propagation_ok"] = len(mism) == 0
        assert not mism, f"propagation mismatch: {mism}"

        # ---- post-selection warmed token-out benchmark (normal construction path) ----
        P, G = args.prompt, args.gen
        torch.manual_seed(0)
        prompt = torch.randint(0, gen.vocab, (P,), dtype=torch.int64).tolist()
        gen._ensure_cache(1, P + 4 * G + 64)
        kv, pt = gen._kv_cache, gen._page_table

        def prefill_ttft():
            x = m.embed_prefill(gen._tokens_to_device(torch.tensor(prompt)))
            h = m.prefill_layers(x, kv, pt, user_id=0, start_pos=0)
            last = ttnn.slice(h, [0, P - 1, 0], [1, P, gen.hidden])
            tb = gen._rep(torch.zeros([1, 1, 1, 1], dtype=torch.int32), ttnn.uint32)
            gen._greedy_sample(m.lm_head_shards_decode(ttnn.reshape(last, (1, 1, 1, gen.hidden))), 1, tb)
            return tb

        prefill_ttft()
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        prefill_ttft()
        ttnn.synchronize_device(mesh)
        ttft_ms = (time.perf_counter() - t0) * 1e3

        tok = gen._rep(torch.zeros([1, 1, 1, 1], dtype=torch.int32), ttnn.uint32)
        cur = gen._rep(torch.zeros([1], dtype=torch.int32), ttnn.int32)
        ridx = gen._rep(torch.zeros([1, 1], dtype=torch.int32), ttnn.uint32)
        fixed = int(prompt[-1])

        def stage(pos):
            ttnn.copy_host_to_device_tensor(gen._host_rank4_tok(fixed), tok)
            ttnn.copy_host_to_device_tensor(gen._host_pos(pos), cur)
            ttnn.copy_host_to_device_tensor(gen._host_ridx(pos), ridx)

        def step_to():
            h = m.embed_decode(ttnn.reshape(tok, (1, 1)))
            h = m.decode_layers(h, cur, ridx, pt, kv)
            gen._greedy_sample(m.lm_head_shards_decode(h), 1, tok)
            ttnn.plus_one(cur, skip_negative_entries=True)
            ttnn.plus_one(ridx)

        def measure(tid):
            stage(P)
            ttnn.synchronize_device(mesh)
            t = time.perf_counter()
            for _ in range(G):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            return (time.perf_counter() - t) / G * 1e3

        stage(P)
        tid = _capture(mesh, step_to)
        stage(P)
        for _ in range(8):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh)
        to_ms = measure(tid)

        # + readback variant
        def measure_rb(tid):
            stage(P)
            ttnn.synchronize_device(mesh)
            t = time.perf_counter()
            for _ in range(G):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                gen._read_token(tok, 1)
            return (time.perf_counter() - t) / G * 1e3

        to_rb_ms = measure_rb(tid)
        ttnn.release_trace(mesh, tid)

        res["post_selection_tokenout"] = {
            "regime": "warmed no-readback traced token-out via normal build_generator(selected config); batch-1",
            "workload": f"prompt{P}/gen{G}",
            "ttft_ms": round(ttft_ms, 1),
            "token_out_decode_ms_tok": round(to_ms, 3),
            "token_out_decode_tsu": round(1e3 / to_ms, 2),
            "token_out_plus_readback_ms_tok": round(to_rb_ms, 3),
            "token_out_plus_readback_tsu": round(1e3 / to_rb_ms, 2),
        }
        print("POST_SELECTION", json.dumps(res["post_selection_tokenout"]))
        print("PROPAGATION_OK", res["propagation_ok"])
    finally:
        try:
            gen.teardown()
        except Exception:
            pass
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    import os as _os
    import sys as _sys

    main()
    _sys.stdout.flush()
    _sys.stderr.flush()
    _os._exit(0)
