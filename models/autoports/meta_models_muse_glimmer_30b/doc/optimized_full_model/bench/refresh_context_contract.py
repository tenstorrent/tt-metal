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
        # The parent builder hardcodes ``doc/full_model/...`` in these two provenance
        # strings, so overriding ``EVIDENCE`` moved the *numbers* to this stage's runs but
        # left them labelled with the previous stage's paths.  Round 4 of the stage review
        # found that; it is the same false-provenance defect rounds 2 and 3 found in
        # ``perf_summary.json`` and ``evidence_perf_before.json``.  Re-point them at the
        # directory the values were actually read from, derived rather than retyped.
        stage_dir = evidence.relative_to(ROOT.parents[2]).as_posix()
        stage_dir = stage_dir[stage_dir.index("doc/") :]
        if "performance" in contract:
            contract["performance"]["source"] = f"{stage_dir}/evidence_perf.json"
        budget = contract.get("byte_budget_at_full_context")
        if isinstance(budget, dict) and "measured_from" in budget:
            budget["measured_from"] = f"{stage_dir}/evidence_accuracy.json (capability_report over the built model)"
        # Round 5 found the same defect in the sibling fields: ``tested.commands`` named the
        # previous stage's harness, and ``tested.prefill_misses.note`` its miss file, while
        # every *value* under ``tested`` is this stage's.  Round 6 then found that fixing it
        # with a blanket substitution over the subtree was too blunt and broke two things:
        # it rewrote the ``qualitative.py --arm hf`` command, which this stage deliberately
        # did **not** run (it reuses the previous stage's HF control via
        # ``--reuse-hf-control``), and it rewrote a ``evidence_misses_*.json`` glob that
        # matches the previous stage's ``_bfp4`` suffix but nothing here.  So the commands
        # are now stated explicitly as the ones this stage actually ran, and the miss note
        # names the file that exists.
        tested = contract.get("tested")
        if isinstance(tested, dict):
            commands = tested.get("commands")
            if isinstance(commands, list):
                tested["commands"] = [
                    (
                        # The HF arm is the previous stage's control, reused rather than re-run.
                        "python doc/full_model/bench/qualitative.py --arm hf "
                        "(run by the previous stage; this stage reuses its output rather than "
                        "re-running it, via the wrapper's --reuse-hf-control)"
                        if "--arm hf" in command
                        else command.replace("doc/full_model/", f"{stage_dir}/")
                    )
                    for command in commands
                ]
                if not any("--arm compare" in command for command in tested["commands"]):
                    tested["commands"].append(f"python {stage_dir}/bench/qualitative.py --arm compare")
            misses = tested.get("prefill_misses")
            if isinstance(misses, dict) and "note" in misses:
                miss_files = sorted(path.name for path in evidence.glob("evidence_misses*.json"))
                misses["note"] = f"per-position detail in {stage_dir}/" + (
                    miss_files[0] if miss_files else "evidence_misses.json"
                )
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
        notes = notes.replace(
            "The full-model stage takes no capability reduction",
            "The optimized-full-model stage takes no capability reduction",
        )
        # Round 7: the parent hardcodes a restore key in this sentence, so it named the wrong
        # predecessor -- the previous top level was ``full_model``, whose block is separately
        # present.  Derive it from the blocks actually nested here instead of asserting one.
        nested = [key for key in contract if isinstance(contract.get(key), dict) and "stage" in contract[key]]
        previous = "full_model" if "full_model" in contract else (nested[0] if nested else "the previous stage")
        for wrong in ("optimized_multichip_decoder", "full_model"):
            notes = notes.replace(
                f"the previous top level is now the '{wrong}' block.",
                f"the previous top level is now the '{previous}' block.",
            )
        contract["notes"] = notes
        # The commands recorded are the ones this stage ran, including the wrapper flag that
        # makes the TT and compare arms reuse the previous stage's HF control.
        tested = contract.get("tested")
        if isinstance(tested, dict) and isinstance(tested.get("commands"), list):
            tested["commands"] = [
                (
                    command + " --reuse-hf-control"
                    if f"{stage_dir}/bench/qualitative.py" in command
                    and "--reuse-hf-control" not in command
                    and ("--arm tt" in command or "--arm compare" in command)
                    else command
                )
                for command in tested["commands"]
            ]
        return contract

    module.build = build
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
