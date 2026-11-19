# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""POLICY-CHANGING DIAGNOSTIC: never full-model acceptance evidence.

Runs the existing reduced shared-pool probe unchanged under a scoped decode-only
reduction replacement. Default BF16 keeps native HiFi4/BF16 DEST but changes its
arithmetic schedule; FP32 additionally changes accumulation precision. Neither
mode selects a production repair. Prefill and n=1 use their original methods.

Serialized device-owner command (author performs CPU checks only)::

 TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto \
 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
 timeout 1800 python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_batch_reduction_counterfactual \
 --accumulation bf16 --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/batch_reduction_counterfactual_bf16.json

Use a different --output with --accumulation fp32 for the precision-changing
control. --checkpoint and --packed-cache pass through to the original probe.
"""

import argparse
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch


def batched_reduction(tt, moe, combined, weights, ids_rm, n, compute):
    """One vectorized batched matmul, with the n=1 ownership-mask semantics.

    All n requests remain one device operation: [1,n,1,10] @ [1,n,10,2560].
    The explicit BF16 row-major conversion removes padded M=32 rows before
    merging the batch dimension into the public token dimension. This avoids
    interpreting per-batch padding as logical tokens; there is no per-row loop.
    """
    ids = tt.to_layout(tt.typecast(tt.reshape(ids_rm, (1, n, 1, 10)), tt.float32), tt.TILE_LAYOUT)
    owned = tt.logical_and(tt.ge(ids, moe.workspace.rank_lo), tt.lt(ids, moe.workspace.rank_hi))
    owned = tt.reshape(tt.typecast(owned, tt.bfloat16), (1, n, 10, 1))
    values = tt.to_layout(tt.reshape(combined, (1, n, 10, 2560)), tt.TILE_LAYOUT)
    values = tt.where(owned, values, 0.0)
    scores = tt.to_layout(tt.reshape(weights, (1, n, 1, 10)), tt.TILE_LAYOUT)
    result = tt.matmul(
        scores, values, dtype=tt.bfloat16, compute_kernel_config=compute, memory_config=tt.DRAM_MEMORY_CONFIG
    )
    return tt.to_layout(tt.reshape(tt.to_layout(result, tt.ROW_MAJOR_LAYOUT), (1, 1, n, 2560)), tt.TILE_LAYOUT)


@contextmanager
def scoped_counterfactual(tt, decoder_class, moe_class, accumulation, stats):
    original_decode, original_reduce = decoder_class.decode_forward, moe_class._reduce
    depth = 0
    configs = {}

    def decode(self, *args, **kwargs):
        nonlocal depth
        depth += 1
        try:
            return original_decode(self, *args, **kwargs)
        finally:
            depth -= 1

    def reduce(self, combined, weights, ids_rm, n):
        if depth == 0 or n == 1:
            key = "outside_decode_original_calls" if depth == 0 else "single_decode_original_calls"
            stats[key] = stats.get(key, 0) + 1
            return original_reduce(self, combined, weights, ids_rm, n)
        if accumulation == "fp32":
            compute = self.compute  # Exact original n=1 config, explicitly precision-changing for n>1.
        else:
            key = id(self)
            if key not in configs:
                # Native post_combine_reduce factory explicitly sets HiFi4,
                # fp32_dest_acc_en=false, dst_full_sync_en=false. Descriptor
                # default math_approx_mode is false (not TTNN's default true).
                # Its packer L1 accumulation is kernel-controlled; this test
                # deliberately removes those per-expert stores, not widens DEST.
                configs[key] = tt.init_device_compute_kernel_config(
                    self.loader.mesh_device.arch(),
                    math_fidelity=tt.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=False,
                    dst_full_sync_en=False,
                    packer_l1_acc=False,
                )
            compute = configs[key]
        stats["batched_decode_replacement_calls"] = stats.get("batched_decode_replacement_calls", 0) + 1
        counts = stats.setdefault("logical_batch_sizes", {})
        counts[str(n)] = counts.get(str(n), 0) + 1
        return batched_reduction(tt, self, combined, weights, ids_rm, n, compute)

    with patch.object(decoder_class, "decode_forward", decode), patch.object(moe_class, "_reduce", reduce):
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accumulation", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--checkpoint")
    parser.add_argument("--packed-cache")
    args = parser.parse_args()
    if "counterfactual" not in args.output.name:
        parser.error("Use an output filename containing 'counterfactual'; never overwrite acceptance evidence")
    if args.output.exists():
        parser.error("Output already exists; choose a new counterfactual artifact name")
    if os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") != "1":
        parser.error("Set TT_METAL_TRACE_ALLOC_TRACKING=1 before imports")
    if os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") != "0":
        parser.error("Keep program-cache allocation tracking enabled")
    import ttnn as tt
    from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MeshMoe, MultichipDecoder

    from . import probe_shared_pool_generator as original_probe

    stats = {}
    label = "POLICY-CHANGING DIAGNOSTIC — NOT ACCEPTANCE"
    report = {
        "scope": label,
        "acceptance_eligible": False,
        "passed": False,
        "counterfactual_accumulation": args.accumulation,
        "precision_changed": args.accumulation == "fp32",
        "arithmetic_schedule_changed": True,
        "prefill_unchanged": True,
        "single_decode_unchanged": True,
        "dispatch_counts_note": "Python warmup/capture dispatch counts, not trace replay execution counts.",
        "compute_contract": {
            "math_fidelity": "HiFi4",
            "fp32_dest_acc_en": args.accumulation == "fp32",
            "math_approx_mode": False,
            "dst_full_sync_en": False,
            "packer_l1_acc": False,
        },
    }
    # Even an interrupted run leaves an explicitly disqualified artifact.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    raw = args.output.with_name(args.output.stem + ".POLICY_CHANGING_DIAGNOSTIC.raw.json")
    assert not raw.exists(), "Refuse to reuse an old raw diagnostic"
    forwarded = [original_probe.__file__, "--output", str(raw)]
    for name in ("checkpoint", "packed_cache"):
        if getattr(args, name):
            forwarded += ["--" + name.replace("_", "-"), getattr(args, name)]
    print(label, args.accumulation, flush=True)
    try:
        with patch.object(sys, "argv", forwarded), scoped_counterfactual(
            tt, MultichipDecoder, MeshMoe, args.accumulation, stats
        ):
            original_probe.main()
        assert stats.get("batched_decode_replacement_calls", 0) > 0, "Counterfactual did not execute"
        assert stats.get("outside_decode_original_calls", 0) > 0, "Missing unchanged prefill control"
        assert stats.get("single_decode_original_calls", 0) > 0, "Missing unchanged single-row control"
    finally:
        if raw.exists():
            base = json.loads(raw.read_text())
            report["diagnostic_checks_passed"] = base.pop("passed", False)
            base.pop("scope", None)
            report.update(base)
            raw.unlink()
        report["scope"] = label
        report["passed"] = False  # Cannot be consumed as an acceptance pass.
        report["acceptance_eligible"] = False
        report["dispatch_counts"] = stats
        report["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report["original_probe_sha256"] = hashlib.sha256(Path(original_probe.__file__).read_bytes()).hexdigest()
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
