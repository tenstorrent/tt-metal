# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Optimization-loop driver (not a pytest): quick PCC sanity + traced warmed
decode timing + warmed prefill timing for one decoder class / dtype arm.

    python .../dev_optimize.py --kind moe --decoder fused \
        --weight-dtype bf8 --expert-dtype bf4 [--skip-pcc] [--prefill]

Timing methodology matches tests/test_fused_perf.py: prefill to ctx 1023,
trace-capture one decode step at position 1023, 3 warm replays, then 32 timed
replays (wall / 32 = ms per token). Prefill: warmed second run at S=2048.
"""

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.zai_org_glm_4_7_flash.tests import utils

DTYPES = {"bf16": ttnn.bfloat16, "bf8": ttnn.bfloat8_b, "bf4": ttnn.bfloat4_b}


def decoder_cls(name):
    if name == "functional":
        from models.autoports.zai_org_glm_4_7_flash.tt.functional_decoder import FunctionalDecoder

        return FunctionalDecoder
    if name == "fused":
        from models.autoports.zai_org_glm_4_7_flash.tt.fused_decoder import FusedDecoder

        return FusedDecoder
    if name == "optimized":
        from models.autoports.zai_org_glm_4_7_flash.tt.optimized_decoder import OptimizedDecoder

        return OptimizedDecoder
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="moe", choices=["moe", "dense"])
    ap.add_argument("--decoder", default="fused", choices=["functional", "fused", "optimized"])
    ap.add_argument("--weight-dtype", default=None, choices=list(DTYPES))
    ap.add_argument("--expert-dtype", default=None, choices=list(DTYPES))
    ap.add_argument("--cache-dtype", default="bf16", choices=["bf16", "bf8"])
    ap.add_argument("--real", action="store_true", help="real checkpoint weights")
    ap.add_argument("--skip-pcc", action="store_true")
    ap.add_argument("--prefill", action="store_true", help="also time warmed prefill S=2048")
    ap.add_argument("--decode-iters", type=int, default=32)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--attn-fidelity", default=None)
    ap.add_argument("--mlp-fidelity", default=None)
    ap.add_argument("--expert-fidelity", default=None)
    ap.add_argument("--prefill-proj-fidelity", default=None)
    ap.add_argument("--prefill-expert-fidelity", default=None)
    ap.add_argument("--attn-weight-dtype", default=None, choices=list(DTYPES))
    ap.add_argument("--mlp-gateup-dtype", default=None, choices=list(DTYPES))
    ap.add_argument("--mlp-down-dtype", default=None, choices=list(DTYPES))
    ap.add_argument("--check-ties", action="store_true")
    args = ap.parse_args()

    cfg = utils.hf_config()
    layer_idx = utils.LAYER_KINDS[args.kind]
    sd = utils.load_real_layer_state_dict(cfg, layer_idx) if args.real else utils.synth_layer_state_dict(cfg, layer_idx)
    cls = decoder_cls(args.decoder)
    if args.attn_weight_dtype:
        cls.attn_weight_dtype = DTYPES[args.attn_weight_dtype]
        print(f"attn_weight_dtype -> {args.attn_weight_dtype}")
    if args.mlp_gateup_dtype:
        cls.mlp_gateup_dtype = DTYPES[args.mlp_gateup_dtype]
        print(f"mlp_gateup_dtype -> {args.mlp_gateup_dtype}")
    if args.mlp_down_dtype:
        cls.mlp_down_dtype = DTYPES[args.mlp_down_dtype]
        print(f"mlp_down_dtype -> {args.mlp_down_dtype}")
    for attr in (
        "attn_fidelity",
        "mlp_fidelity",
        "expert_fidelity",
        "prefill_proj_fidelity",
        "prefill_expert_fidelity",
    ):
        v = getattr(args, attr)
        if v:
            setattr(cls, attr, v)
            results_fid = v
            print(f"{attr} -> {v}")

    kwargs = dict(
        hf_config=cfg,
        layer_idx=layer_idx,
        mesh_device=None,  # filled below
        max_batch_size=1,
        max_context=4096,
        prefill_chunk_size=1024,
    )
    if args.weight_dtype:
        kwargs["weight_dtype"] = DTYPES[args.weight_dtype]
    if args.expert_dtype:
        kwargs["expert_dtype"] = DTYPES[args.expert_dtype]

    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=0)
    results = {
        "kind": args.kind,
        "decoder": args.decoder,
        "weight_dtype": args.weight_dtype,
        "expert_dtype": args.expert_dtype,
        "cache_dtype": args.cache_dtype,
        "real": args.real,
        "tag": args.tag,
    }
    try:
        kwargs["mesh_device"] = device
        dec = cls.from_state_dict(sd, **kwargs)
        paged = dec.paged_config
        cache_dtype = ttnn.bfloat16 if args.cache_dtype == "bf16" else ttnn.bfloat8_b
        try:
            cache = dec.allocate_kv_cache(dtype=cache_dtype)
        except TypeError:
            cache = dec.allocate_kv_cache()
        pt_torch = utils.make_page_table(1, paged.max_num_blocks, seed=3)
        pt = ttnn.from_torch(pt_torch, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

        # ---------------- quick PCC sanity (prefill 509 + 3 decode steps) ----
        if not args.skip_pcc:
            S, n_steps = 509, 3
            x = utils.synth_activations(cfg, layer_idx, S + n_steps, seed=7)
            hf_layer = utils.build_hf_layer(cfg, layer_idx, sd)
            ref = utils.hf_forward(cfg, hf_layer, x)
            if args.check_ties:
                ties = utils.router_tie_positions(cfg, hf_layer, x)
                near = {p: g for p, g in ties.items() if p >= S - 2}
                results["router_ties_at_decode_positions"] = near
                print(f"router sub-ulp ties at decode positions: {near} (all ties: {sorted(ties)})")
            x_tt = ttnn.from_torch(x[:, :S].unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            out = dec.prefill_forward(x_tt, kv_cache=cache, page_table=pt, user_id=0, seq_len=S)
            p_pre = utils.pcc(ref[0, :S], ttnn.to_torch(out).float()[0, 0, :S])
            ttnn.deallocate(out)
            ttnn.deallocate(x_tt)
            steps = []
            for i in range(n_steps):
                pos = S + i
                x_tt_d = ttnn.from_torch(
                    x[:, pos : pos + 1].unsqueeze(0).permute(0, 2, 1, 3),
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                )
                cur = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=device)
                rot = ttnn.from_torch(torch.tensor([[pos]], dtype=torch.uint32), device=device)
                out_d = dec.decode_forward(x_tt_d, kv_cache=cache, page_table=pt, cur_pos_tensor=cur, rot_idxs=rot)
                steps.append(utils.pcc(ref[0, pos], ttnn.to_torch(out_d).float()[0, 0, 0]))
                for t in (out_d, x_tt_d, cur, rot):
                    ttnn.deallocate(t)
            results["pcc_prefill_509"] = round(p_pre, 6)
            results["pcc_decode_steps"] = [round(p, 6) for p in steps]
            print(f"PCC prefill509={p_pre:.6f} decode={['%.6f' % p for p in steps]}")
            ttnn.deallocate(cache)
            cache = dec.allocate_kv_cache(dtype=cache_dtype) if args.cache_dtype == "bf8" else dec.allocate_kv_cache()

        # ---------------- traced decode timing at ctx 1024 -------------------
        S = 1023
        x = utils.synth_activations(cfg, layer_idx, S + 2, seed=7)
        x_tt = ttnn.from_torch(x[:, :S].unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        out = dec.prefill_forward(x_tt, kv_cache=cache, page_table=pt, user_id=0, seq_len=S)
        ttnn.deallocate(out)
        ttnn.deallocate(x_tt)

        pos = S
        x_dev = ttnn.from_torch(
            x[:, pos : pos + 1].unsqueeze(0).permute(0, 2, 1, 3),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        pos_dev = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=device)
        rot_dev = ttnn.from_torch(torch.tensor([[pos]], dtype=torch.uint32), device=device)

        out_c = dec.decode_forward(x_dev, kv_cache=cache, page_table=pt, cur_pos_tensor=pos_dev, rot_idxs=rot_dev)
        ttnn.deallocate(out_c)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        out_t = dec.decode_forward(x_dev, kv_cache=cache, page_table=pt, cur_pos_tensor=pos_dev, rot_idxs=rot_dev)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        for _ in range(3):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        try:
            ttnn.ReadDeviceProfiler(device)
        except Exception:
            pass

        try:
            from tracy import signpost

            signpost(f"PERF_DECODE_{args.kind.upper()}_{args.decoder.upper()}")
        except Exception:
            signpost = None
        t0 = time.perf_counter()
        for _ in range(args.decode_iters):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        if signpost is not None:
            signpost(f"PERF_DECODE_{args.kind.upper()}_{args.decoder.upper()}_END")
            try:
                ttnn.ReadDeviceProfiler(device)
            except Exception:
                pass
        ms_tok = (t1 - t0) / args.decode_iters * 1000
        assert not torch.isnan(ttnn.to_torch(out_t)).any()
        ttnn.release_trace(device, tid)
        results["decode_ms_per_token"] = round(ms_tok, 4)
        print(f"TRACED DECODE {args.kind} ctx1024: {ms_tok:.4f} ms/token")

        # ---------------- warmed prefill timing S=2048 -----------------------
        if args.prefill:
            ttnn.deallocate(cache)
            cache = dec.allocate_kv_cache(dtype=cache_dtype) if args.cache_dtype == "bf8" else dec.allocate_kv_cache()
            S2 = 2048
            xp = utils.synth_activations(cfg, layer_idx, S2, seed=7)
            xp_tt = ttnn.from_torch(xp.unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            out = dec.prefill_forward(xp_tt, kv_cache=cache, page_table=pt, user_id=0, seq_len=S2)
            ttnn.deallocate(out)
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            out = dec.prefill_forward(xp_tt, kv_cache=cache, page_table=pt, user_id=0, seq_len=S2)
            ttnn.synchronize_device(device)
            t1 = time.perf_counter()
            ttnn.deallocate(out)
            results["prefill_2048_ms"] = round((t1 - t0) * 1000, 2)
            results["prefill_tokens_per_s"] = round(S2 / (t1 - t0), 1)
            print(f"WARMED PREFILL {args.kind} S=2048: {(t1-t0)*1000:.1f} ms ({S2/(t1-t0):.0f} t/s)")
    finally:
        ttnn.close_device(device)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=1))
        print(f"wrote {args.json_out}")
    print("RESULT", json.dumps(results))


if __name__ == "__main__":
    main()
