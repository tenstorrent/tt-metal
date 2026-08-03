# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Datatype-sweep driver for Laguna-XS-2.1 on the 1x4 Blackhole mesh.

Evaluates one or more precision-policy candidates with the OFFICIAL readiness
teacher-forcing scoring (`models.common.readiness_check.run_teacher_forcing._run_one_entry`
+ `TokenAccuracy`) against the AIME24 chat-template reference (192-token prompt, 100 forced
tokens, top-100). Reports, per candidate: top-1/top-5/top-100, TTFT, and trace-verified
teacher-forcing decode t/s/u (the sweep RANKING metric). Also dumps a "built-policy readback"
(actual on-device weight dtypes + per-group math fidelity) proving the recorded dtype/fidelity
fields are consumed by the measured runtime path.

Two ways to vary policy, chosen to minimise the ~5-min-per-build weight-load cost:
  * WEIGHT-CHANGING candidates (attn/shared/dense/expert/lm_head/qk_norm dtype) require a fresh
    build: pass `--base-config <candidate.json>` and a single variant with an empty `mutate`.
  * WEIGHT-INVARIANT candidates (compute fidelity, CCL dtype, KV-cache dtype) are applied
    IN-PROCESS on one shared build: list several variants whose `mutate` only touches
    fid_*/ccl/kv_cache/logits. Traces are released and recaptured per variant so the new
    fidelity/CCL/KV is what the measured trace actually runs.

Spec JSON (`--spec path`):
  {"base_config": "configs/base.json" | null,
   "variants": [{"id": "C0", "mutate": {}}, {"id": "C7", "mutate": {"fid_moe": "HiFi2"}}, ...]}

  cd /tmp && TT_METAL_HOME=<tree> PYTHONPATH=<repo> python \
    <this>.py --spec <spec.json> --out-dir <candidate_results>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.poolside_laguna_xs_2_1.tt.generator import build_generator
from models.autoports.poolside_laguna_xs_2_1.tt.optimized_decoder import _DTYPE_TO_STR, _STR_TO_DTYPE, PrecisionPolicy
from models.common.readiness_check.run_teacher_forcing import _run_one_entry
from models.common.readiness_check.teacher_forcing import TokenAccuracy

MD = Path("/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1")
REF = MD / "readiness_aime24_chat.refpt"

_WEIGHT_FIELDS = {
    "attn_qkv",
    "attn_o",
    "attn_gate",
    "dense_ff13",
    "dense_ff2",
    "moe_ff13",
    "moe_ff2",
    "shared_ff13",
    "shared_ff2",
    "router",
    "qk_norm",
    "lm_head",
}
_FID_TO_GROUP = {
    "fid_attn_qkv": "_ck_qkv",
    "fid_attn_o": "_ck_o",
    "fid_attn_gate": "_ck_gate",
    "fid_dense": "_ck_dense",
    "fid_shared": "_ck_shared",
    "fid_router": "_ck_router",
    "fid_moe": "_ck_moe",
}


def _fid_of(dec, attr):
    """Resolve a group's math fidelity by VALUE (the compute-config's math_fidelity enum), so it is
    robust to value-identical-but-different-object compute configs."""
    ck = getattr(dec, attr)
    mf = getattr(ck, "math_fidelity", None)
    if mf is not None:
        s = str(mf).split(".")[-1]
        if s in ("LoFi", "HiFi2", "HiFi4"):
            return s
    for name in ("LoFi", "HiFi2", "HiFi4"):  # fallback: identity
        if ck is dec._ck_by_fid[name]:
            return name
    return "?"


def _built_readback(model):
    """Read the ACTUAL on-device weight dtypes + per-group math fidelity from the built model,
    proving the policy fields are consumed by the measured runtime path."""
    l0 = model.layers[0]  # dense MLP layer (layer 0)
    moe = next((d for d in model.layers if d.cfg.is_moe), model.layers[-1])
    rb = {
        "lm_head_weight_dtype": _DTYPE_TO_STR.get(model.lm_head_w.dtype, str(model.lm_head_w.dtype)),
        "attn_wqkv_dtype": _DTYPE_TO_STR.get(moe.w["wqkv"].dtype, str(moe.w["wqkv"].dtype)),
        "attn_wo_dtype": _DTYPE_TO_STR.get(moe.w["wo"].dtype, str(moe.w["wo"].dtype)),
        "moe_expert_gate_up_dtype": _DTYPE_TO_STR.get(moe.w["exp_gate_up"].dtype, str(moe.w["exp_gate_up"].dtype)),
        "moe_expert_down_dtype": _DTYPE_TO_STR.get(moe.w["exp_down"].dtype, str(moe.w["exp_down"].dtype)),
        "shared_gate_up_dtype": _DTYPE_TO_STR.get(moe.w["sh_gate_up"].dtype, str(moe.w["sh_gate_up"].dtype)),
        "shared_down_dtype": _DTYPE_TO_STR.get(moe.w["sh_down"].dtype, str(moe.w["sh_down"].dtype)),
        "dense_gate_up_dtype": _DTYPE_TO_STR.get(l0.w["mlp_gate_up"].dtype, str(l0.w["mlp_gate_up"].dtype)),
        "dense_down_dtype": _DTYPE_TO_STR.get(l0.w["mlp_down"].dtype, str(l0.w["mlp_down"].dtype)),
        "router_dtype": _DTYPE_TO_STR.get(moe.w["gate_w"].dtype, str(moe.w["gate_w"].dtype)),
        "policy_ccl": _DTYPE_TO_STR.get(moe.policy.ccl, str(moe.policy.ccl)),
        "policy_kv_cache": _DTYPE_TO_STR.get(moe.policy.kv_cache, str(moe.policy.kv_cache)),
        "fid_attn_qkv": _fid_of(moe, "_ck_qkv"),
        "fid_attn_o": _fid_of(moe, "_ck_o"),
        "fid_attn_gate": _fid_of(moe, "_ck_gate"),
        "fid_dense": _fid_of(l0, "_ck_dense"),
        "fid_shared": _fid_of(moe, "_ck_shared"),
        "fid_router": _fid_of(moe, "_ck_router"),
        "fid_moe": _fid_of(moe, "_ck_moe"),
    }
    return rb


def _snapshot_base(model):
    """Snapshot the as-built weight-invariant state PER LAYER (each layer's own fidelity ck objects +
    ccl/kv/logits dtypes) so each variant is applied from a clean base without accumulating mutations
    and without cross-assigning one layer's compute-config objects to another."""
    snap = []
    for dec in model.layers:
        e = {
            "ck": {attr: getattr(dec, attr) for attr in _FID_TO_GROUP.values()},
            "ccl": dec.policy.ccl,
            "kv": dec.policy.kv_cache,
            "logits": dec.policy.logits,
        }
        snap.append(e)
    return snap


def _restore_base(gen, snap):
    for dec, e in zip(gen.model.layers, snap):
        for attr, ck in e["ck"].items():
            setattr(dec, attr, ck)
        dec.policy.ccl = e["ccl"]
        dec.policy.kv_cache = e["kv"]
        dec.policy.logits = e["logits"]


def _apply_mutation(gen, mutate):
    """Apply a WEIGHT-INVARIANT mutation in-process to all decoder layers."""
    model = gen.model
    for f in mutate:
        if f in _WEIGHT_FIELDS:
            raise ValueError(f"weight field '{f}' cannot be mutated in-process; use --base-config")
    kv_changed = False
    for dec in model.layers:
        for f, v in mutate.items():
            if f in _FID_TO_GROUP:
                setattr(dec, _FID_TO_GROUP[f], dec._ck_by_fid[v])
            elif f == "ccl":
                dec.policy.ccl = _STR_TO_DTYPE[str(v).lower()]
            elif f == "kv_cache":
                dec.policy.kv_cache = _STR_TO_DTYPE[str(v).lower()]
                kv_changed = True
            elif f == "logits":
                dec.policy.logits = _STR_TO_DTYPE[str(v).lower()]
            else:
                raise ValueError(f"unknown mutate field {f}")
    if kv_changed:
        gen._kv_cache = None
        gen._trace = {}


def _effective_policy_dict(gen, base_policy, mutate):
    """The full effective policy for this variant = base weight policy + weight-invariant overrides."""
    d = dict(base_policy.to_dict())
    d.update(mutate)
    # normalise dtype-valued overrides to strings
    for f in ("ccl", "kv_cache", "logits"):
        if f in d and not isinstance(d[f], str):
            d[f] = _DTYPE_TO_STR[d[f]]
    return d


def _capture(mesh, body):
    body()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    body()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    return tid


def _perf_tokenout(gen, mesh, P=128, G=128):
    """Warmed token-out no-readback + logits-only traced decode t/s/u (perf_full_model methodology:
    capture -> warm 8 -> measure non-blocking + one sync -> release). Drift-resistant, so it is a
    clean corroboration metric even as a second measurement in the process."""
    m = gen.model
    torch.manual_seed(0)
    prompt = torch.randint(0, gen.vocab, (P,), dtype=torch.int64).tolist()
    gen._kv_cache = None
    gen._trace = {}
    gen._ensure_cache(1, P + 4 * G + 64)
    kv, pt = gen._kv_cache, gen._page_table
    tok = gen._rep(torch.zeros([1, 1, 1, 1], dtype=torch.int32), ttnn.uint32)
    cur = gen._rep(torch.zeros([1], dtype=torch.int32), ttnn.int32)
    ridx = gen._rep(torch.zeros([1, 1], dtype=torch.int32), ttnn.uint32)
    _hold = {}
    fixed_tok = int(prompt[-1])

    def stage(pos):
        ttnn.copy_host_to_device_tensor(gen._host_rank4_tok(fixed_tok), tok)
        ttnn.copy_host_to_device_tensor(gen._host_pos(pos), cur)
        ttnn.copy_host_to_device_tensor(gen._host_ridx(pos), ridx)

    def step_logits():
        h = m.embed_decode(ttnn.reshape(tok, (1, 1)))
        h = m.decode_layers(h, cur, ridx, pt, kv)
        _hold["l"] = m.lm_head_shards_decode(h)
        ttnn.plus_one(cur, skip_negative_entries=True)
        ttnn.plus_one(ridx)

    def step_tokenout():
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
    tid_l = _capture(mesh, step_logits)
    stage(P)
    for _ in range(8):
        ttnn.execute_trace(mesh, tid_l, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh)
    lo = measure(tid_l)
    ttnn.release_trace(mesh, tid_l)
    stage(P)
    tid_t = _capture(mesh, step_tokenout)
    stage(P)
    for _ in range(8):
        ttnn.execute_trace(mesh, tid_t, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh)
    to = measure(tid_t)
    ttnn.release_trace(mesh, tid_t)
    gen._kv_cache = None
    gen._trace = {}
    return {
        "tokenout_workload": f"prompt{P}/gen{G}",
        "logits_only_decode_ms_tok": round(lo, 3),
        "logits_only_decode_tsu": round(1e3 / lo, 2),
        "token_out_decode_ms_tok": round(to, 3),
        "token_out_decode_tsu": round(1e3 / to, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-seq-len", type=int, default=16384)
    ap.add_argument(
        "--perf-tokenout",
        action="store_true",
        help="also run the tight warmed token-out benchmark (single-variant fresh builds only)",
    )
    args = ap.parse_args()

    spec = json.loads(Path(args.spec).read_text())
    base_cfg = spec.get("base_config")
    base_cfg_path = str((MD / base_cfg)) if base_cfg and not Path(base_cfg).is_absolute() else base_cfg
    variants = spec["variants"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if base_cfg_path:
        base_policy = PrecisionPolicy.from_dict(json.loads(Path(base_cfg_path).read_text())["policy"])
    else:
        base_policy = PrecisionPolicy()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=1_500_000_000)
    results = []
    try:
        gen = build_generator(
            MD,
            mesh,
            max_seq_len=args.max_seq_len,
            precision_config_path=base_cfg_path,  # None -> in-code defaults
        )
        readback_base = _built_readback(gen.model)
        base_snap = _snapshot_base(gen.model)
        acc = TokenAccuracy(REF)
        for v in variants:
            vid = v["id"]
            mutate = v.get("mutate", {})
            t0 = time.perf_counter()
            try:
                gen.teardown()
                gen._kv_cache = None  # force fresh cache each variant (base KV dtype restored below)
                _restore_base(gen, base_snap)
                _apply_mutation(gen, mutate)
                acc.reset()
                stats = _run_one_entry(generator=gen, acc=acc, entry_idx=0)
                readback = _built_readback(gen.model)
                rec = {
                    "id": vid,
                    "status": "ok",
                    "policy": _effective_policy_dict(gen, base_policy, mutate),
                    "built_readback": readback,
                    "top1": stats["top1"],
                    "top5": stats["top5"],
                    "top100": stats["top100"],
                    "matches_top1": stats["matches_top1"],
                    "matches_top5": stats["matches_top5"],
                    "matches_top100": stats["matches_top100"],
                    "total": stats["total"],
                    "k": stats["k"],
                    "ttft_ms": round(stats.get("ttft_ms", 0.0), 2),
                    "teacher_decode_tsu": round(stats.get("decode_t/s/u", 0.0), 3),
                    "teacher_decode_ms_tok": round(1e3 / stats["decode_t/s/u"], 3)
                    if stats.get("decode_t/s/u")
                    else None,
                    "decode_tokens": stats.get("decode_tokens"),
                    "decode_elapsed_s": round(stats.get("decode_elapsed_s", 0.0), 4),
                    "eval_wall_s": round(time.perf_counter() - t0, 1),
                }
                if args.perf_tokenout:
                    gen.teardown()
                    rec.update(_perf_tokenout(gen, mesh))
            except Exception as e:
                rec = {
                    "id": vid,
                    "status": "FAILED",
                    "error": repr(e)[:400],
                    "policy": _effective_policy_dict(gen, base_policy, mutate),
                }
            results.append(rec)
            (out_dir / f"{vid}.json").write_text(json.dumps(rec, indent=2))
            print("SWEEP_RESULT", json.dumps(rec))
    finally:
        try:
            gen.teardown()
        except Exception:
            pass
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    (out_dir / f"_run_{Path(args.spec).stem}.json").write_text(json.dumps(results, indent=2))
    print("SWEEP_DONE", len(results), "variants")


if __name__ == "__main__":
    import os as _os
    import sys as _sys

    main()
    _sys.stdout.flush()
    _sys.stderr.flush()
    _os._exit(0)
