# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Whole-layer A/B: traced decode ms/token, warmed prefill ms, and PCC vs HF.

The isolated matmul sweeps (``decode_matmul_sweep.py``) rank a projection in
isolation; this ranks a *candidate layer*.  Both are needed, and they disagree in
one important way: the isolated probe has ~1.5 MB of L1 free for circular
buffers, while a real decode step has ~232 KB of live L1 tensors at the MLP, so
the largest ``in0_block_w`` the isolated sweep accepts can fail in the layer.
Candidates are therefore always re-ranked here before anything is shipped
($optimize: "Measure the whole layer").

Each candidate is a (precision policy, decode geometry) pair.  Failures are
reported as ``FAILED`` with the exact op message rather than aborting the run, so
one illegal geometry does not cost the whole sweep.

    python .../bench/layer_ab.py --candidates shipped,mlp_bfp8
    python .../bench/layer_ab.py --list
"""

from __future__ import annotations

import argparse
import time
import traceback

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tests import reference as R
from models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder import reference_layer_indices
from models.autoports.meta_models_muse_glimmer_30b.tt.fused_decoder import FusedDecoder
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import (
    BOUNDARY_CORES,
    DECODE_MATMUL,
    OptimizedDecoder,
    PrecisionPolicy,
)

PAGE_BLOCK = 64
MAX_SEQ = 16384
BF16, BFP8, BFP4 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b


def policy(name, attn=BFP8, gate_up=BFP4, down=BFP4, kv=BFP8, act=BF16, fid=ttnn.MathFidelity.LoFi):
    return PrecisionPolicy(
        name=name,
        attn_weight_dtype=attn,
        mlp_gate_up_weight_dtype=gate_up,
        mlp_down_weight_dtype=down,
        kv_cache_dtype=kv,
        activation_dtype=act,
        decode_math_fidelity=fid,
        prefill_math_fidelity=fid,
    )


def geometry(**overrides):
    """``DECODE_MATMUL`` with per-role ``(cores, in0_block_w)`` overrides."""
    table = dict(DECODE_MATMUL)
    for key, value in overrides.items():
        role, dtype = key.rsplit("__", 1)
        table[(role, {"bfp4": BFP4, "bfp8": BFP8, "bf16": BF16}[dtype])] = value
    return table


#: ``name -> (policy, geometry, boundary_cores)``.  Grouped by the question each
#: block answers; see ``doc/optimized_decoder/work_log.md``.
CANDIDATES: dict[str, tuple[PrecisionPolicy, dict, int]] = {
    # -- precision policy, one tensor group at a time -----------------------
    "all_bfp8": (
        policy("all-bfp8", gate_up=BFP8, down=BFP8),
        geometry(mlp_gate__bfp8=(26, 2), mlp_up__bfp8=(26, 2), mlp_down__bfp8=(26, 8)),
        BOUNDARY_CORES,
    ),
    "all_bfp8_52": (
        policy("all-bfp8", gate_up=BFP8, down=BFP8),
        geometry(mlp_gate__bfp8=(52, 4), mlp_up__bfp8=(52, 4), mlp_down__bfp8=(52, 12)),
        BOUNDARY_CORES,
    ),
    "all_bfp8_26bw4": (
        policy("all-bfp8", gate_up=BFP8, down=BFP8),
        geometry(mlp_gate__bfp8=(26, 4), mlp_up__bfp8=(26, 4), mlp_down__bfp8=(26, 4)),
        BOUNDARY_CORES,
    ),
    "gateup_bfp4": (
        policy("gateup-bfp4-down-bfp8", down=BFP8),
        geometry(mlp_down__bfp8=(26, 4)),
        BOUNDARY_CORES,
    ),
    "gateup_bfp4_dbw8": (
        policy("gateup-bfp4-down-bfp8", down=BFP8),
        geometry(mlp_down__bfp8=(26, 8)),
        BOUNDARY_CORES,
    ),
    "down_bfp4_gateup_bfp8": (
        policy("gateup-bfp8-down-bfp4", gate_up=BFP8),
        geometry(mlp_gate__bfp8=(26, 2), mlp_up__bfp8=(26, 2)),
        BOUNDARY_CORES,
    ),
    "mlp_bfp4": (policy("attn-bfp8-mlp-bfp4"), DECODE_MATMUL, BOUNDARY_CORES),
    "all_bfp4": (policy("all-bfp4", attn=BFP4), DECODE_MATMUL, BOUNDARY_CORES),
    "attn_bfp4_mlp_bfp8": (
        policy("attn-bfp4-mlp-bfp8", attn=BFP4, gate_up=BFP8, down=BFP8),
        DECODE_MATMUL,
        BOUNDARY_CORES,
    ),
    # -- activation dtype, separately from weight dtype ---------------------
    "act_bfp8": (policy("attn-bfp8-mlp-bfp4-act-bfp8", act=BFP8), DECODE_MATMUL, BOUNDARY_CORES),
    # -- KV cache dtype (OPT-002) -------------------------------------------
    "mlp_bfp4_kv_bf16": (policy("attn-bfp8-mlp-bfp4-kv-bf16", kv=BF16), DECODE_MATMUL, BOUNDARY_CORES),
    "all_bfp8_kv_bf16": (policy("all-bfp8-kv-bf16", gate_up=BFP8, down=BFP8, kv=BF16), DECODE_MATMUL, BOUNDARY_CORES),
    # -- math fidelity, dtype held fixed ------------------------------------
    "mlp_bfp4_hifi2": (policy("attn-bfp8-mlp-bfp4-hifi2", fid=ttnn.MathFidelity.HiFi2), DECODE_MATMUL, BOUNDARY_CORES),
    "mlp_bfp4_hifi4": (policy("attn-bfp8-mlp-bfp4-hifi4", fid=ttnn.MathFidelity.HiFi4), DECODE_MATMUL, BOUNDARY_CORES),
    # -- in-layer geometry: gate/up in0_block_w under the real L1 budget ----
    "mlp_bfp4_gu_bw4": (
        policy("attn-bfp8-mlp-bfp4"),
        geometry(mlp_gate__bfp4=(13, 4), mlp_up__bfp4=(13, 4)),
        BOUNDARY_CORES,
    ),
    "mlp_bfp4_gu_bw2": (
        policy("attn-bfp8-mlp-bfp4"),
        geometry(mlp_gate__bfp4=(13, 2), mlp_up__bfp4=(13, 2)),
        BOUNDARY_CORES,
    ),
    "mlp_bfp4_gu26_bw4": (
        policy("attn-bfp8-mlp-bfp4"),
        geometry(mlp_gate__bfp4=(26, 4), mlp_up__bfp4=(26, 4), mlp_down__bfp4=(26, 24)),
        BOUNDARY_CORES,
    ),
    "mlp_bfp4_gu26_bw8": (
        policy("attn-bfp8-mlp-bfp4"),
        geometry(mlp_gate__bfp4=(26, 8), mlp_up__bfp4=(26, 8), mlp_down__bfp4=(26, 24)),
        BOUNDARY_CORES,
    ),
    "mlp_bfp4_gu52_bw4": (
        policy("attn-bfp8-mlp-bfp4"),
        geometry(mlp_gate__bfp4=(52, 4), mlp_up__bfp4=(52, 4), mlp_down__bfp4=(52, 12)),
        BOUNDARY_CORES,
    ),
    "mlp_bfp4_down_bw12": (policy("attn-bfp8-mlp-bfp4"), geometry(mlp_down__bfp4=(26, 12)), BOUNDARY_CORES),
    "mlp_bfp4_down_bw8": (policy("attn-bfp8-mlp-bfp4"), geometry(mlp_down__bfp4=(26, 8)), BOUNDARY_CORES),
    "mlp_bfp4_down_bw4": (policy("attn-bfp8-mlp-bfp4"), geometry(mlp_down__bfp4=(26, 4)), BOUNDARY_CORES),
    # -- boundary grid 16 (the round-2 winner) and its neighbourhood ---------
    "b16_oproj_bw4": (
        policy("attn-bfp8-mlp-bfp4"),
        geometry(wqkv__bfp8=(16, 13), attn_gate__bfp8=(16, 13), o_proj__bfp8=(16, 4)),
        16,
    ),
    "b16_oproj_bw2": (
        policy("attn-bfp8-mlp-bfp4"),
        geometry(wqkv__bfp8=(16, 13), attn_gate__bfp8=(16, 13), o_proj__bfp8=(16, 2)),
        16,
    ),
    "b16_mlp52": (
        policy("attn-bfp8-mlp-bfp4"),
        geometry(
            wqkv__bfp8=(16, 13),
            attn_gate__bfp8=(16, 13),
            o_proj__bfp8=(16, 8),
            mlp_gate__bfp4=(52, 4),
            mlp_up__bfp4=(52, 4),
            mlp_down__bfp4=(52, 12),
        ),
        16,
    ),
    "b16_mlp16": (
        policy("attn-bfp8-mlp-bfp4"),
        geometry(
            wqkv__bfp8=(16, 13),
            attn_gate__bfp8=(16, 13),
            o_proj__bfp8=(16, 8),
            mlp_gate__bfp4=(16, 13),
            mlp_up__bfp4=(16, 13),
            mlp_down__bfp4=(16, 39),
        ),
        16,
    ),
    "b16_sdpa88": (
        policy("attn-bfp8-mlp-bfp4"),
        geometry(wqkv__bfp8=(16, 13), attn_gate__bfp8=(16, 13), o_proj__bfp8=(16, 8)),
        16,
    ),
    # -- packed candidates at an in-layer-legal in0_block_w -----------------
    "b16_all_bfp8": (
        policy("all-bfp8", gate_up=BFP8, down=BFP8),
        geometry(
            wqkv__bfp8=(16, 13),
            attn_gate__bfp8=(16, 13),
            o_proj__bfp8=(16, 8),
            mlp_gate__bfp8=(26, 2),
            mlp_up__bfp8=(26, 2),
            mlp_down__bfp8=(26, 8),
        ),
        16,
    ),
    # -- in-layer geometry: attention in0_block_w ---------------------------
    "attn_bw2": (
        policy("attn-bfp8-mlp-bfp4"),
        geometry(wqkv__bfp8=(BOUNDARY_CORES, 2), attn_gate__bfp8=(BOUNDARY_CORES, 2)),
        BOUNDARY_CORES,
    ),
    "attn_oproj_bw8": (policy("attn-bfp8-mlp-bfp4"), geometry(o_proj__bfp8=(BOUNDARY_CORES, 8)), BOUNDARY_CORES),
    # -- boundary grid ------------------------------------------------------
    "boundary16": (
        policy("attn-bfp8-mlp-bfp4"),
        geometry(wqkv__bfp8=(16, 13), attn_gate__bfp8=(16, 13), o_proj__bfp8=(16, 8)),
        16,
    ),
    "boundary4": (
        policy("attn-bfp8-mlp-bfp4"),
        geometry(wqkv__bfp8=(4, 13), attn_gate__bfp8=(4, 13), o_proj__bfp8=(4, 4)),
        4,
    ),
}


#: The packed candidates need their own ``in0_block_w``: a packed output is 2x
#: wide, so the value the split path uses overflows L1 in the layer.  These are
#: the largest that compile, so the packed family is rejected on a *measured*
#: loss rather than on a first API error ($optimize OPT-001/OPT-010).
#: Set from ``--sdpa``; a one-element list so ``build`` can read it without a
#: global statement.
SDPA_OVERRIDE: list = [None]

#: ``name -> extra OptimizedDecoder kwargs``, on top of the shipped policy and
#: geometry.  These are candidates that change one constructor knob rather than a
#: precision policy or a geometry table, so they cannot be expressed in
#: ``CANDIDATES``.
KWARG_CANDIDATES: dict[str, dict] = {
    # Fold SiLU into the MLP gate matmul and sigmoid into the attention-gate
    # matmul instead of leaving them on the ttnn.mul that consumes each
    # (DECODE_FUSED_ACTIVATION).  Measured against "mlp_bfp4", which is the same
    # layer with the activations on the elementwise ops.
    "fused_act": {"decode_fused_activation": True},
}

PACKED_GEOMETRY = {
    "packed_qkv_gate": {"decode_matmul": geometry(wqkv__bfp8=(BOUNDARY_CORES, 2))},
    "packed_gate_up": {
        "decode_matmul": geometry(mlp_gate__bfp4=(52, 2), mlp_up__bfp4=(52, 2), mlp_down__bfp4=(52, 12))
    },
}


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


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().to(torch.float64), b.flatten().to(torch.float64)
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _variant_classes():
    """Rejected candidates live next to this script; import lazily."""
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "mg_optimized_variants", pathlib.Path(__file__).with_name("variants.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {
        "packed_qkv_gate": module.PackedQkvGateDecoder,
        "packed_gate_up": module.PackedGateUpDecoder,
        "fused_sdpa": module.FusedSdpaDecoder,
    }


def build(mesh, name, layer_idx, state_dict, decode_context):
    # 131071 is the last valid position of the advertised 131072-token context.
    max_seq = MAX_SEQ if decode_context < MAX_SEQ else 131072
    if name == "fused":
        return FusedDecoder.from_state_dict(
            state_dict,
            hf_config=R.hf_config(),
            layer_idx=layer_idx,
            mesh_device=mesh,
            max_batch_size=1,
            max_seq_len=max_seq,
            page_block_size=PAGE_BLOCK,
            prefill_chunk_size=8192,
        )
    variants = _variant_classes()
    if name in variants:
        extra = PACKED_GEOMETRY.get(name, {})
        return variants[name].from_state_dict(
            state_dict,
            hf_config=R.hf_config(),
            layer_idx=layer_idx,
            mesh_device=mesh,
            max_batch_size=1,
            max_seq_len=max_seq,
            page_block_size=PAGE_BLOCK,
            prefill_chunk_size=8192,
            **extra,
        )
    extra_kwargs = KWARG_CANDIDATES.get(name, {})
    pol, table, boundary = CANDIDATES["mlp_bfp4" if name in KWARG_CANDIDATES else name]
    return OptimizedDecoder.from_state_dict(
        state_dict,
        hf_config=R.hf_config(),
        layer_idx=layer_idx,
        mesh_device=mesh,
        max_batch_size=1,
        max_seq_len=max_seq,
        page_block_size=PAGE_BLOCK,
        prefill_chunk_size=8192,
        precision=pol,
        decode_matmul=table,
        boundary_cores=boundary,
        decode_sdpa=SDPA_OVERRIDE[0],
        **extra_kwargs,
    )


def run_candidate(mesh, name, kind, layer_idx, state_dict, ref_layer, args):
    dec = build(mesh, name, layer_idx, state_dict, args.decode_context)
    pt = page_table(mesh, 1, dec.config.max_seq_len, seed=3)
    result = {"name": name, "kind": kind}

    # ---- PCC: prefill then a decode step off the prefilled cache
    hidden = R.synthetic_hidden_states(1, args.pcc_seq_len, seed=42)
    ref_out, ref_cache = R.reference_prefill(ref_layer, layer_idx, hidden)
    tt_out = dec.prefill_forward(to_dev(mesh, hidden), page_table=pt, user_id=0)
    result["prefill_pcc"] = pcc(ttnn.to_torch(tt_out).reshape(1, args.pcc_seq_len, -1), ref_out)
    ttnn.deallocate(tt_out)
    token = R.synthetic_hidden_states(1, 1, seed=100)
    ref_dec = R.reference_decode(
        ref_layer, layer_idx, token, past_key_values=ref_cache, positions=torch.tensor([args.pcc_seq_len])
    )
    cur, rope = pos_tensors(mesh, torch.tensor([args.pcc_seq_len]))
    tt_dec = dec.decode_forward(to_dev(mesh, token), current_pos=cur, page_table=pt, rope_pos_ids=rope)
    result["decode_pcc"] = pcc(ttnn.to_torch(tt_dec).reshape(1, 1, -1), ref_dec)
    ttnn.deallocate(tt_dec)
    for t in (cur, rope):
        ttnn.deallocate(t)

    # ---- warmed prefill
    tt_hidden = to_dev(mesh, R.synthetic_hidden_states(1, args.prefill_seq, seed=43))
    for _ in range(2):
        ttnn.deallocate(dec.prefill_forward(tt_hidden, page_table=pt, user_id=0))
    best = float("inf")
    for _ in range(args.rounds):
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(2):
            ttnn.deallocate(dec.prefill_forward(tt_hidden, page_table=pt, user_id=0))
        ttnn.synchronize_device(mesh)
        best = min(best, (time.perf_counter() - t0) / 2 * 1e3)
    result["prefill_ms"] = best
    ttnn.deallocate(tt_hidden)

    # ---- traced decode
    tt_token = to_dev(mesh, R.synthetic_hidden_states(1, 1, seed=44))
    cur, rope = pos_tensors(mesh, torch.tensor([args.decode_context]))
    ttnn.deallocate(dec.decode_forward(tt_token, current_pos=cur, page_table=pt, rope_pos_ids=rope))
    ttnn.synchronize_device(mesh)
    trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
    tt_trace_out = dec.decode_forward(tt_token, current_pos=cur, page_table=pt, rope_pos_ids=rope)
    ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh)
    for _ in range(8):
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    best = float("inf")
    per_round = []
    for _ in range(args.rounds):
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(args.decode_iters):
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        per_round.append((time.perf_counter() - t0) / args.decode_iters * 1e3)
        best = min(best, per_round[-1])
    result["decode_ms"] = best
    result["decode_rounds"] = per_round
    ttnn.release_trace(mesh, trace_id)
    for t in (tt_token, cur, rope, pt, tt_trace_out):
        ttnn.deallocate(t)
    del dec
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="fused,mlp_bfp4")
    ap.add_argument("--kinds", default="sliding,full")
    ap.add_argument("--pcc-seq-len", type=int, default=100)
    ap.add_argument("--prefill-seq", type=int, default=8192)
    ap.add_argument("--decode-context", type=int, default=2048)
    ap.add_argument("--decode-iters", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--real-weights", action="store_true", help="use the released bf16 checkpoint for this layer")
    ap.add_argument("--sdpa", default="", help="override decode SDPA as gx,gy,q_chunk,k_chunk")
    args = ap.parse_args()
    if args.sdpa:
        SDPA_OVERRIDE[0] = tuple(int(v) for v in args.sdpa.split(","))
    if args.list:
        for name, (pol, _, boundary) in CANDIDATES.items():
            print(f"{name:22s} {pol.name:32s} boundary={boundary}")
        return

    names = args.candidates.split(",")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=90112 * 12)
    ttnn.SetDefaultDevice(mesh)
    try:
        idxs = reference_layer_indices(R.hf_config())
        for kind in args.kinds.split(","):
            layer_idx = idxs[kind]
            state_dict = R.real_state_dict(layer_idx) if args.real_weights else R.synthetic_state_dict(layer_idx)
            ref_layer = R.reference_layer(layer_idx, state_dict)
            for name in names:
                try:
                    r = run_candidate(mesh, name, kind, layer_idx, state_dict, ref_layer, args)
                    print(
                        f"AB{'[real]' if args.real_weights else '      '} {name:22s} kind={kind:8s} "
                        f"prefill{args.prefill_seq}={r['prefill_ms']:8.2f} ms  "
                        f"traced_decode@{args.decode_context}={r['decode_ms']:7.4f} ms/token  "
                        f"prefill_pcc={r['prefill_pcc']:.6f} decode_pcc={r['decode_pcc']:.6f}  "
                        f"({'/'.join(f'{v:.4f}' for v in r['decode_rounds'])})",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    msg = " | ".join(l.strip() for l in str(exc).strip().splitlines() if l.strip())
                    print(f"FAILED {name:22s} kind={kind:8s} {msg[:400]}", flush=True)
                    traceback.print_exc()
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
