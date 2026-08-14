# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""What moving the softcap into L1 costs the decode step's L1 high-water mark.

The stage review was right to reject "none of the three changes allocates anything":
``ttnn.tanh`` and ``ttnn.multiply`` have no in-place form here, so change 1 replaces two
DRAM-interleaved transients with two **width-sharded L1** ones.  The long-lived DRAM
budget is unchanged, but the traced decode step's L1 high-water mark is not, and this
model has a documented 7,296 B of headroom in the main L1 pool.

This probe measures it rather than arguing it: free L1 per bank at the boundary layout,
then after each op of the LM-head tail, in both orders.  Read-only with respect to the
model; it allocates and frees its own tensors.

Usage::

    python doc/optimized_full_model/bench/l1_highwater_probe.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

import ttnn

ROOT = pathlib.Path(__file__).resolve().parents[3]
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))

from models.autoports.meta_models_muse_glimmer_30b.tt import model as model_mod  # noqa: E402
from models.autoports.meta_models_muse_glimmer_30b.tt.generator import (  # noqa: E402
    DEFAULT_TRACE_REGION_SIZE,
    build_generator,
    clear_generator_cache,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (  # noqa: E402
    close_multichip_mesh,
    open_multichip_mesh,
)

OUT = ROOT / "doc/optimized_full_model"


def say(*args) -> None:
    print(*args, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="0,3")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--out", default="l1_highwater_probe.json")
    args = parser.parse_args()

    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    mesh = open_multichip_mesh(trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    summary: dict = {}
    generator = None
    try:
        generator = build_generator(
            ROOT, mesh, max_seq_len=args.max_seq_len, layer_indices=[int(i) for i in args.layers.split(",")]
        )
        model = generator.model
        rows = model_mod.TERMINAL_ROWS

        def l1():
            view = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
            return {
                "free_per_bank": int(view.total_bytes_free_per_bank),
                "allocated_per_bank": int(view.total_bytes_allocated_per_bank),
                "largest_contiguous_free_per_bank": int(view.largest_contiguous_bytes_free_per_bank),
                "banks": int(view.num_banks),
            }

        summary["l1_total_bytes_per_bank"] = int(ttnn.get_memory_view(mesh, ttnn.BufferType.L1).total_bytes_per_bank)
        summary["baseline"] = l1()
        say(f"L1 baseline {summary['baseline']}")

        hidden = ttnn.from_torch(
            torch.randn(1, 1, rows, model.config.hidden_size, generator=torch.Generator().manual_seed(5)).to(
                torch.bfloat16
            ),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=model.boundary_memcfg(rows),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        head = model.lm_head
        try:
            for label, in_l1 in (("softcap_in_l1", True), ("softcap_in_dram", False)):
                head.softcap_in_l1 = in_l1
                steps: dict = {}
                hidden_in, owned = head._as_input(hidden)
                steps["after_lm_head_input_reshard"] = l1()
                logits = ttnn.linear(
                    hidden_in,
                    head.weight,
                    dtype=head.output_dtype,
                    memory_config=head.output_memcfg,
                    program_config=head.program_config,
                    compute_kernel_config=head.compute_kernel_config,
                )
                if owned:
                    ttnn.deallocate(hidden_in)
                steps["after_matmul"] = l1()
                if not in_l1:
                    interleaved = ttnn.sharded_to_interleaved(logits, ttnn.DRAM_MEMORY_CONFIG)
                    ttnn.deallocate(logits)
                    logits = interleaved
                    steps["after_sharded_to_interleaved"] = l1()
                memcfg = logits.memory_config()
                capped = ttnn.tanh(logits, memory_config=memcfg)
                steps["after_tanh"] = l1()
                scaled = ttnn.multiply(capped, head.softcap, memory_config=memcfg)
                steps["after_multiply"] = l1()
                ttnn.deallocate(logits)
                ttnn.deallocate(capped)
                if scaled.is_sharded():
                    out = ttnn.sharded_to_interleaved(scaled, ttnn.DRAM_MEMORY_CONFIG)
                    ttnn.deallocate(scaled)
                    scaled = out
                    steps["after_final_sharded_to_interleaved"] = l1()
                ttnn.deallocate(scaled)
                steps["after_free"] = l1()
                peak = max(s["allocated_per_bank"] for s in steps.values())
                summary[label] = {
                    "steps": steps,
                    "peak_allocated_per_bank": peak,
                    "peak_over_baseline_per_bank": peak - summary["baseline"]["allocated_per_bank"],
                }
                say(f"L1 {label}: peak allocated/bank = {peak} B (+{peak - summary['baseline']['allocated_per_bank']})")
                for name, snap in steps.items():
                    say(
                        f"L1   {name:<38} allocated/bank={snap['allocated_per_bank']:>9} free={snap['free_per_bank']:>9}"
                    )
        finally:
            head.softcap_in_l1 = model_mod.LM_HEAD_SOFTCAP_IN_L1
            ttnn.deallocate(hidden)

        delta = (
            summary["softcap_in_l1"]["peak_allocated_per_bank"] - summary["softcap_in_dram"]["peak_allocated_per_bank"]
        )
        summary["l1_peak_delta_per_bank_bytes"] = delta
        summary["l1_free_per_bank_at_peak_with_change"] = (
            summary["l1_total_bytes_per_bank"] - summary["softcap_in_l1"]["peak_allocated_per_bank"]
        )
        say(
            f"L1 change 1 costs {delta} B/bank of peak L1; "
            f"{summary['l1_free_per_bank_at_peak_with_change']} B/bank still free at the peak"
        )
        say("L1_OK")
        return 0
    finally:
        path = OUT / args.out
        path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
        say(f"L1 summary -> {path}")
        if generator is not None:
            generator.teardown()
        clear_generator_cache()
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    raise SystemExit(main())
