"""Predict remaining diagnostic time from the completed serial pilot; no inference."""
import json
import re
import time
from pathlib import Path

D = Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
split = json.loads((D / "setup/swe85-pilot-split85/split-report.json").read_text())
counts = split["counts_per_precision"]["rest"]["total"]
predictions = {}
for mode, free_rate in [("served", 35.07), ("full", 17.17)]:
    path = D / f"setup/swe85-pilot-split83/runs/pilot/{mode}"
    result = json.loads((path / "fullreply.sphinx-doc__sphinx-10323.call68.json").read_text())
    wall = result["continuations"]["chosen"]["diagnostic_wall_s_including_prefill"]
    log = (D / f"setup/swe85-pilot-execution83/{mode}.log").read_text()
    progress = float(re.search(r"PRECISION_TF_PROGRESS .* chosen 256 480 ([\d.]+)", log)[1])
    per_token = (wall - progress) / 224
    residual = wall - 480 * per_token
    load = json.loads((path / "load.json").read_text())["wall_s"]
    prefill = residual * counts["prefill_input_tokens_proxy"] / 64139
    teacher_forcing = per_token * counts["teacher_forcing_positions"]
    free = counts["free_run_token_cap"] / free_rate
    seconds = load + prefill + teacher_forcing + free
    predictions[mode] = {
        "pilot_load_s": load,
        "pilot_branch_wall_s": wall,
        "pilot_progress256_s": progress,
        "estimated_tf_seconds_per_token": per_token,
        "estimated_prefill_residual_s": residual,
        "remaining_load_s": load,
        "remaining_prefill_s": prefill,
        "remaining_tf_s": teacher_forcing,
        "remaining_free_s": free,
        "predicted_hours": seconds / 3600,
        "planning_interval_hours": [0.7 * seconds / 3600, 1.5 * seconds / 3600],
        "timeout_s": 14400 if mode == "served" else 21600,
    }
record = {
    "status": "cost_gate_pass_pending_scoped_fixture_review",
    "at_unix": time.time(),
    "pilot": "setup/swe85-pilot-split83/pilot.json",
    "remainder": "setup/swe85-pilot-split85/rest.json",
    "counts_per_precision": counts,
    "prediction": predictions,
    "basis": "Serial measured prefix64139 with480 TF positions. TF slope estimated from256-position progress to completion; remaining branch wall is prefill/warmup residual scaled by input token count. Free generation uses prior short matched TT diagnostic35.07/17.17 tokens/s. Not serving rates.",
    "limitations": [
        "Single prefix and rounded progress timestamp; prefill nonlinearity and short-input overhead uncertain.",
        "No profiler or added synchronization; ordinary host logit/token readbacks included.",
        "Longest anomaly prefix76697 modestly extrapolates pilot. 50% upper margin; inspect completed jobs against prediction.",
        "Three semantic coverage gaps require supplemental controls outside frozen161 manifest.",
    ],
    "full_timeout_override_reason": "Predicted upper bound exceeds4h;6h timeout admits measured cost without truncating progressing diagnostics, below12h per-run limit and remaining stage budget.",
}
(D / "setup/swe85-remainder-cost-gate.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(predictions))
