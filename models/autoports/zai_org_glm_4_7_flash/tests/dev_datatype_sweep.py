# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-model ($datatype-sweep stage) candidate driver for GLM-4.7-Flash.

Builds the real 47-layer model through the exact same factory the readiness
harness uses (`tt/generator.py::build_generator`), with a candidate
dtype/compute-fidelity policy applied via real constructor kwargs (dtype
groups already exposed at the model level: `expert_dtype`, `weight_dtype`,
`cache_dtype`, `lm_head_dtype`, `lm_head_fidelity`) and/or a dynamically
subclassed decoder (`OptimizedDecoder`/`SharedRopeDecoder` class attributes:
`attn_fidelity`, `mlp_fidelity`, `expert_fidelity`, `router_fidelity`,
`prefill_proj_fidelity`, `prefill_expert_fidelity`, `attn_weight_dtype`,
`mlp_gateup_dtype`, `mlp_down_dtype`, `dense_mlp_dtype`). Then runs the shared
AIME24 chat-template prefill-check and teacher-forcing readiness checks
against the one already-open mesh device, reusing the shared harness's own
per-entry evaluation functions verbatim (`run_prefill_check._run_one_entry_prefill`,
`run_teacher_forcing._run_one_entry`) so the accuracy numbers are directly
comparable to `doc/optimized_full_model/logs/run_{prefill_check,teacher_forcing}.log`.

Building the model once (instead of once per readiness script, as the CLI
tools do) lets this driver introspect the constructed model's real dtype and
compute-kernel-config objects immediately after construction and record them
in the output JSON's `policy_snapshot` -- proof the candidate's requested
policy actually reached the measured runtime path, not just this script's
request.

    python -m models.autoports.zai_org_glm_4_7_flash.tests.dev_datatype_sweep \\
        --config-id C00_baseline \\
        --out doc/datatype_sweep/runs/C00_baseline.json

    python -m models.autoports.zai_org_glm_4_7_flash.tests.dev_datatype_sweep \\
        --config-id C01_lmhead_bf4_lofi --lm-head-dtype bf4 --lm-head-fidelity lofi \\
        --out doc/datatype_sweep/runs/C01_lmhead_bf4_lofi.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import ttnn
from models.autoports.zai_org_glm_4_7_flash.tt.generator import build_generator
from models.autoports.zai_org_glm_4_7_flash.tt.model import SharedRopeDecoder
from models.autoports.zai_org_glm_4_7_flash.tt.optimized_decoder import FIDELITY
from models.common.readiness_check.mesh_device import close_readiness_mesh_device, open_readiness_mesh_device
from models.common.readiness_check.run_prefill_check import _run_one_entry_prefill
from models.common.readiness_check.run_teacher_forcing import _run_one_entry as _run_one_entry_tf
from models.common.readiness_check.schema import load_reference
from models.common.readiness_check.teacher_forcing import TokenAccuracy

MODEL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = MODEL_DIR / "readiness_aime24_chat.refpt"
DTYPES = {"bf16": ttnn.bfloat16, "bf8": ttnn.bfloat8_b, "bf4": ttnn.bfloat4_b}
FIDELITY_CHOICES = sorted(FIDELITY)

CLASS_ATTR_FIDELITY_ARGS = (
    "attn_fidelity",
    "mlp_fidelity",
    "expert_fidelity",
    "router_fidelity",
    "prefill_proj_fidelity",
    "prefill_expert_fidelity",
)
CLASS_ATTR_DTYPE_ARGS = ("attn_weight_dtype", "mlp_gateup_dtype", "mlp_down_dtype", "dense_mlp_dtype")


def build_decoder_cls(args) -> type:
    """A fresh SharedRopeDecoder subclass carrying only the requested overrides
    (unset fields inherit the shipped default from OptimizedDecoder). Returns
    the base class unchanged (no subclass) when nothing is overridden, so the
    baseline candidate builds the literal shipped class, not a look-alike."""
    attrs: dict = {}
    for name in CLASS_ATTR_FIDELITY_ARGS:
        val = getattr(args, name)
        if val is not None:
            attrs[name] = val
    for name in CLASS_ATTR_DTYPE_ARGS:
        val = getattr(args, name)
        if val is not None:
            attrs[name] = DTYPES[val]
    if not attrs:
        return SharedRopeDecoder
    return type(f"SweepDecoder_{args.config_id}", (SharedRopeDecoder,), attrs)


def _fid(ck) -> dict:
    return {"math_fidelity": str(ck.math_fidelity), "fp32_dest_acc_en": bool(ck.fp32_dest_acc_en)}


def policy_snapshot(model) -> dict:
    """Introspect the just-built model's real dtype/compute-kernel-config
    objects (not the requested kwargs) -- the propagation-check proof."""
    dense = next((layer for layer in model.layers if layer.layer_kind == "dense"), None)
    moe = next((layer for layer in model.layers if layer.layer_kind == "moe"), None)
    snap = {
        "expert_dtype": str(model.expert_dtype),
        "weight_dtype": str(model.weight_dtype),
        "cache_dtype": str(model.cache_dtype),
        "embed_dtype": str(model.embed_dtype),
        "lm_head_dtype": str(model.lm_head_weight.dtype),
        "lm_head_fidelity": model.lm_head_fidelity,
        "ck_lm_head": _fid(model.ck_lm_head),
        "decoder_cls": model.decoder_cls_name,
    }
    if moe is not None:
        snap["moe_layer"] = {
            "attn_weight_dtype": str(moe.wqkv_a_ds.dtype),
            "ck_attn": _fid(moe.ck_attn),
            "shared_gate_up_dtype": str(moe.shared_gate.dtype),
            "shared_down_dtype": str(moe.shared_down_ds.dtype),
            "expert_gate_up_dtype": str(moe.experts_gate_up.dtype),
            "expert_down_dtype": str(moe.experts_down.dtype),
            "ck_mlp_shared": _fid(moe.ck_mlp),
            "ck_expert": _fid(moe.ck_expert),
            "router_dtype": str(moe.gate_w.dtype),
            "ck_router": _fid(moe.ck_router),
        }
    if dense is not None:
        snap["dense_layer"] = {
            "attn_weight_dtype": str(dense.wqkv_a_ds.dtype),
            "mlp_gate_dtype": str(dense.mlp_gate.dtype),
            "dense_down_dtype": str(dense.dense_down_ds.dtype),
            "ck_mlp_dense": _fid(dense.ck_mlp),
        }
    return snap


def _aggregate(per_entry: list[dict], *, timed: bool) -> dict:
    total = sum(e["total"] for e in per_entry)
    if not total:
        return {}
    agg = {
        "top1": sum(e["matches_top1"] for e in per_entry) / total,
        "top5": sum(e["matches_top5"] for e in per_entry) / total,
        "top100": sum(e["matches_top100"] for e in per_entry) / total,
        "matches_top1": sum(e["matches_top1"] for e in per_entry),
        "matches_top5": sum(e["matches_top5"] for e in per_entry),
        "matches_top100": sum(e["matches_top100"] for e in per_entry),
        "total": total,
        "k": per_entry[0]["k"],
    }
    if timed:
        total_elapsed_s = sum(e.get("elapsed_s", 0.0) for e in per_entry)
        ttft_values = [e["ttft_ms"] for e in per_entry if e.get("ttft_ms") is not None]
        decode_tokens = sum(e.get("decode_tokens", 0.0) for e in per_entry)
        decode_elapsed_s = sum(e.get("decode_elapsed_s", 0.0) for e in per_entry)
        agg["elapsed_s"] = total_elapsed_s
        agg["e2e_t/s/u"] = (total / total_elapsed_s) if total_elapsed_s > 0 else 0.0
        if ttft_values:
            agg["ttft_ms"] = sum(ttft_values) / len(ttft_values)
        if decode_elapsed_s > 0:
            agg["decode_tokens"] = decode_tokens
            agg["decode_elapsed_s"] = decode_elapsed_s
            agg["decode_t/s/u"] = decode_tokens / decode_elapsed_s
    return agg


def run_candidate(args) -> dict:
    build_kwargs: dict = {"decoder_cls": build_decoder_cls(args)}
    if args.expert_dtype:
        build_kwargs["expert_dtype"] = DTYPES[args.expert_dtype]
    if args.weight_dtype:
        build_kwargs["weight_dtype"] = DTYPES[args.weight_dtype]
    if args.cache_dtype:
        build_kwargs["cache_dtype"] = DTYPES[args.cache_dtype]
    if args.lm_head_dtype:
        build_kwargs["lm_head_dtype"] = DTYPES[args.lm_head_dtype]
    if args.lm_head_fidelity:
        build_kwargs["lm_head_fidelity"] = args.lm_head_fidelity

    mesh_device = open_readiness_mesh_device(
        args.mesh_device, None, trace_region_size=args.trace_region_size, l1_small_size=args.l1_small_size
    )
    result: dict = {
        "config_id": args.config_id,
        "build_kwargs_requested": {
            "expert_dtype": args.expert_dtype,
            "weight_dtype": args.weight_dtype,
            "cache_dtype": args.cache_dtype,
            "lm_head_dtype": args.lm_head_dtype,
            "lm_head_fidelity": args.lm_head_fidelity,
            **{name: getattr(args, name) for name in CLASS_ATTR_FIDELITY_ARGS},
            **{name: getattr(args, name) for name in CLASS_ATTR_DTYPE_ARGS},
        },
        "reference": str(args.reference),
        "mesh_device": args.mesh_device,
    }
    t_build0 = time.perf_counter()
    try:
        generator = build_generator(model_dir=MODEL_DIR, mesh_device=mesh_device, **build_kwargs)
        result["build_s"] = time.perf_counter() - t_build0
        result["policy_snapshot"] = policy_snapshot(generator.model)

        reference = load_reference(args.reference)
        prefill_entries = []
        for entry_idx, entry in enumerate(reference.entries):
            if entry_idx > 0:
                generator.reset()
            prefill_entries.append(_run_one_entry_prefill(generator=generator, entry=entry, reference=reference))
        result["prefill_check"] = {"per_entry": prefill_entries, "aggregate": _aggregate(prefill_entries, timed=False)}

        generator.reset()
        acc = TokenAccuracy(args.reference)
        tf_entries = []
        for entry_idx in range(acc.num_entries):
            if entry_idx > 0:
                generator.reset()
            tf_entries.append(_run_one_entry_tf(generator=generator, acc=acc, entry_idx=entry_idx))
        result["teacher_forcing"] = {"per_entry": tf_entries, "aggregate": _aggregate(tf_entries, timed=True)}
    finally:
        teardown = locals().get("generator")
        if teardown is not None:
            fn = getattr(teardown, "teardown", None)
            if callable(fn):
                fn()
        close_readiness_mesh_device(mesh_device, None)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-id", required=True)
    ap.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    ap.add_argument("--mesh-device", default="N150")
    ap.add_argument("--trace-region-size", type=int, default=350_000_000)
    ap.add_argument("--l1-small-size", type=int, default=32_768)
    ap.add_argument("--out", type=Path, required=True)
    # model-level dtype/fidelity kwargs (from_pretrained)
    ap.add_argument("--expert-dtype", default=None, choices=list(DTYPES))
    ap.add_argument("--weight-dtype", default=None, choices=list(DTYPES))
    ap.add_argument("--cache-dtype", default=None, choices=list(DTYPES))
    ap.add_argument("--lm-head-dtype", default=None, choices=list(DTYPES))
    ap.add_argument("--lm-head-fidelity", default=None, choices=FIDELITY_CHOICES)
    # decoder class-attribute dtype/fidelity overrides
    for name in CLASS_ATTR_FIDELITY_ARGS:
        ap.add_argument(f"--{name.replace('_', '-')}", default=None, choices=FIDELITY_CHOICES)
    for name in CLASS_ATTR_DTYPE_ARGS:
        ap.add_argument(f"--{name.replace('_', '-')}", default=None, choices=list(DTYPES))
    args = ap.parse_args()

    result = run_candidate(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str))
    pf = result["prefill_check"]["aggregate"]
    tf = result["teacher_forcing"]["aggregate"]
    print(
        f"{args.config_id}: prefill top1={pf.get('top1'):.3f} top5={pf.get('top5'):.3f} top100={pf.get('top100'):.3f} | "
        f"tf top1={tf.get('top1'):.3f} top5={tf.get('top5'):.3f} top100={tf.get('top100'):.3f} "
        f"TTFT={tf.get('ttft_ms', float('nan')):.2f}ms decode={tf.get('decode_t/s/u', float('nan')):.2f}t/s/u"
    )
    print(f"CANDIDATE_DONE {args.config_id} -> {args.out}")


if __name__ == "__main__":
    main()
