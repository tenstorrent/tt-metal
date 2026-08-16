# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tier A of the sweep: does each candidate reach the device, and as what?

A 48-layer teacher-forcing row is ~3 minutes. Twenty-three of them is over an
hour, and a row that turns out to be *unconstructible* (an op with no kernel for
that dtype, a block width that gets clamped, a CCL that will not run at
``bfloat8_b``) would waste the whole three minutes to find that out. So every
candidate is first built at **two layers**, which is ~10 s, and three things are
recorded:

1. ``fallback_audit`` -- what the config actually put on the device. In
   particular the **resolved** ``in0_block_w``, because
   ``_tuned_sparse_matmul_config`` silently clamps an illegal one and the row
   would otherwise be recorded under the width it *asked for* rather than the
   width it got.
2. the per-die expert byte count, so a dtype change is visible as an allocation
   change rather than a claim.
3. four generated tokens, so "reaches the device" means "and still runs".

Nothing here reads ``candidate.config.<field>`` for its output. Every reported
value is read back off the built model.

Two layers is enough for all of that and for **none** of the accuracy or t/s/u
numbers -- those need 48 layers and are tier B.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.doc.datatype_sweep.probes.candidates import (  # noqa: E402
    CANDIDATES,
    STACKED,
)
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.generator import build_generator  # noqa: E402
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.model import DEFAULT_TRACE_REGION_SIZE  # noqa: E402

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parents[1]

# The audit keys that are *device readback* rather than config echo. These are
# what a sweep row is allowed to quote.
READBACK_KEYS = [
    "device_experts_gate_up_dtype",
    "device_experts_down_dtype",
    "device_attention_qkv_dtype",
    "device_attention_wo_dtype",
    "device_attention_qkv_decode_dtype",
    "device_router_dtype",
    "device_norm_weight_dtype",
    "device_expert_bytes_per_die",
    "lm_head_weight_dtype",
    "embedding_weight_dtype",
    "kv_cache_dtype",
    "kv_cache_dtype_source",
    "gate_up_in0_block_w",
    "down_in0_block_w",
    "expert_math_fidelity",
    "attention_math_fidelity",
    "router_window_math_fidelity",
    "ccl_dtype",
    "activation_dtype",
    # Added by the stage-07 review. Without these four the audit for a row that
    # only moves ``lm_head_fidelity`` / ``norm_fidelity`` / ``logits_dtype`` /
    # ``sampling_dtype`` was byte-identical to the baseline's, so a row whose
    # lever was never wired up looked exactly like a row whose lever did
    # nothing. ``norm_fidelity`` turned out to be the former.
    "lm_head_math_fidelity",
    "norm_math_fidelity",
    "logits_dtype_observed",
    "sampling_dtype_observed",
    "dram_sharded_taken",
    "expert_intermediate_buffer",
    "norm_shard_feeds_qkv_directly",
]


def probe_one(mesh, cand, layers: int) -> dict:
    row = {"config_id": cand.cid, "group": cand.group, "delta": cand.delta, "why": cand.why}
    gen = None
    try:
        gen = build_generator(
            MODEL_DIR,
            mesh,
            override_num_layers=layers,
            max_context_len=512,
            max_batch_size=1,
            precision=cand.config,
        )
        # Allocate the KV cache before auditing so kv_cache_dtype is a readback.
        gen._ensure_kv_cache()
        # Generate BEFORE auditing: ``logits_dtype_observed`` /
        # ``sampling_dtype_observed`` are read off the tensors the terminal path
        # produced, so they are ``None`` until something has run.
        ids = gen.tokenizer("def fib(n):", add_special_tokens=False)["input_ids"]
        out = gen.generate(ids, 4, enable_trace=True, sampling_mode="device", top_k=1)
        row["tokens"] = [int(t) for t in out]
        audit = gen.model.runtime_fallback_audit()
        row["audit"] = {k: audit.get(k) for k in READBACK_KEYS}
        row["status"] = "ok"
        # Did any requested block width get silently clamped?
        req_gu = cand.config.experts_gate_up_in0_block_w
        req_dn = cand.config.experts_down_in0_block_w
        row["block_width_requested"] = [req_gu, req_dn]
        row["block_width_resolved"] = [audit["gate_up_in0_block_w"], audit["down_in0_block_w"]]
        row["block_width_clamped"] = [req_gu, req_dn] != [audit["gate_up_in0_block_w"], audit["down_in0_block_w"]]
    except Exception as exc:  # noqa: BLE001 - a failing candidate is a result
        row["status"] = "error"
        row["error"] = repr(exc)
        row["traceback"] = traceback.format_exc()[-2000:]
    finally:
        if gen is not None:
            try:
                gen.teardown()
            except Exception:  # noqa: BLE001
                pass
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--only", default="", help="comma-separated config ids; default all")
    ap.add_argument("--out", default="structural_probe.json")
    args = ap.parse_args()

    pool = CANDIDATES + STACKED
    wanted = [c for c in pool if not args.only or c.cid in args.only.split(",")]

    # Merge rather than overwrite: a later ``--only`` pass for the stacked rows
    # must not discard the first pass's twenty-three results.
    out_file = HERE / args.out
    rows = json.loads(out_file.read_text()) if out_file.exists() else []
    rows = [r for r in rows if r["config_id"] not in {c.cid for c in wanted}]

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    try:
        for i, cand in enumerate(wanted, 1):
            print(f"[{i}/{len(wanted)}] {cand.cid} :: {cand.delta}", flush=True)
            row = probe_one(mesh, cand, args.layers)
            rows.append(row)
            print(
                f"    -> {row['status']}"
                + (f"  clamped={row.get('block_width_clamped')}" if row["status"] == "ok" else f"  {row.get('error')}"),
                flush=True,
            )
            out_file.write_text(json.dumps(rows, indent=2))
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    out_file.write_text(json.dumps(rows, indent=2))
    bad = [r["config_id"] for r in rows if r["status"] != "ok"]
    clamped = [r["config_id"] for r in rows if r.get("block_width_clamped")]
    print(f"\n{len(rows)} probed; {len(bad)} failed: {bad}; clamped widths: {clamped}")


if __name__ == "__main__":
    main()
