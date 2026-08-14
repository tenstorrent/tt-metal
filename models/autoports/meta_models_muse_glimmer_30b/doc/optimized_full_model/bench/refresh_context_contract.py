# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Rebuild ``doc/context_contract.json`` for the optimized-full-model stage.

The full-model stage's builder, loaded by path, pointed at this stage's evidence
files and stamped with this stage's name.  Nothing about the *capability* changes
here -- the three optimizations are layout changes and allocate nothing new -- so
the point of re-running it is that the contract's measured byte budget and
performance rows come from this stage's runs rather than being inherited on trust.

Usage::

    python doc/optimized_full_model/bench/refresh_context_contract.py
    python doc/optimized_full_model/bench/refresh_context_contract.py --check
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))


def main() -> int:
    path = ROOT / "doc/full_model/bench/refresh_context_contract.py"
    spec = importlib.util.spec_from_file_location("muse_glimmer_full_model_contract", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.EVIDENCE = ROOT / "doc/optimized_full_model"
    module.STAGE = "optimized_full_model"

    evidence = module.EVIDENCE
    original_build = module.build

    def build(previous: dict) -> dict:
        contract = original_build(previous)
        # The two persistent/peak allocations this stage adds, from their own measured
        # artifacts rather than from prose.  ``$optimize`` requires the contract to move
        # when trace buffers or activation memory move, and both of these do.
        trace = json.loads((evidence / "prefill_trace_probe.json").read_text())
        l1 = json.loads((evidence / "l1_highwater_probe.json").read_text())
        contract.setdefault("implementation", {})["optimized_full_model_allocations"] = {
            "prefill_trace_default": False,
            "prefill_trace_retained_dram_bytes_per_device_at_128_rows": trace["capture_retained_dram_bytes"],
            "prefill_trace_retained_dram_source": (
                "doc/optimized_full_model/prefill_trace_probe.json (measured with the decode and "
                "sampling traces already captured)"
            ),
            "prefill_trace_max_entries_default": 1,
            "prefill_trace_bound": (
                "one entry per 32-row padded-length bucket; retained DRAM scales with the padded row "
                "count, so prefill_trace_max_entries x padded_rows must be budgeted against "
                "trace_region_bytes and DRAM by any caller that raises either. Only the 128-row, "
                "1-entry configuration is measured."
            ),
            "decode_peak_l1_delta_bytes_per_bank": l1["l1_peak_delta_per_bank_bytes"],
            "decode_peak_l1_free_per_bank_at_that_peak": l1["l1_free_per_bank_at_peak_with_change"],
            "l1_bytes_per_bank": l1["l1_total_bytes_per_bank"],
            "decode_peak_l1_source": (
                "doc/optimized_full_model/l1_highwater_probe.json (the LM-head softcap moved onto the "
                "matmul's own width-sharded L1 output; measured on an otherwise-idle device, so it is "
                "the terminal path's own peak rather than the step's)"
            ),
        }
        notes = contract.get("notes", "")
        contract["notes"] = notes.replace(
            "The full-model stage takes no capability reduction",
            "The optimized-full-model stage takes no capability reduction",
        )
        return contract

    module.build = build
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
