#!/usr/bin/env python3
"""Check the autonomous diffusion-bringup capability contract.

Diffusion analog of ``check_context_contract.py`` (which is autoregressive/context-length specific).
A diffusion pipeline advertises a *capability envelope* — the modalities, resolutions, frame counts,
fps, audio rate, latent shapes, and denoise-step counts it can generate — rather than a context length.
This runner-side guardrail asserts that a ``capability_contract.json`` exists, is structurally sane, and
that any REDUCTION of an advertised capability is justified by a hard device limit with evidence (mirrors
the context contract's DRAM-reason rule). It deliberately does not infer quality; stage-review and the
qualitative-check gate handle perceptual quality.

Exit codes: 0 pass, 1 advisory, 2 critical, 3 error.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

CONTRACT_NAME = "capability_contract.json"

# A reduction of advertised capability is only acceptable for a hard physical limit.
HARD_LIMIT_REASONS = {
    "device_dram",
    "dram",
    "device_dram_capacity",
    "hardware_dram_capacity",
    "l1_capacity",
    "device_l1",
}

# Keys that describe the generative envelope. At least the starred ones must be present.
REQUIRED_KEYS = ("modalities", "denoise_steps")
RECOMMENDED_KEYS = ("resolution", "num_frames", "fps", "audio", "latent_shapes")


def _find_contract(model_dir: str, hf_model: str) -> Path | None:
    """Locate capability_contract.json under a model dir or a slugged autoport dir."""
    candidates: list[Path] = []
    if model_dir:
        candidates += [Path(model_dir) / CONTRACT_NAME, Path(model_dir) / "doc" / CONTRACT_NAME]
    if hf_model:
        slug = re.sub(r"[^a-z0-9]+", "_", hf_model.lower()).strip("_")
        for root in (Path("models/autoports"), Path("models/tt_dit/models"), Path("models/tt_dit/pipelines")):
            candidates += list(root.glob(f"**/*{slug}*/**/{CONTRACT_NAME}")) if root.exists() else []
    for c in candidates:
        if c.is_file():
            return c
    # last resort: any capability_contract.json in the tree (single-model checkouts)
    hits = list(Path(".").glob(f"**/{CONTRACT_NAME}"))
    return hits[0] if len(hits) == 1 else None


def _check_reductions(contract: dict[str, Any]) -> tuple[int, list[str]]:
    """Every declared reduction needs a hard-limit reason + evidence. Returns (worst_code, msgs)."""
    worst = 0
    msgs: list[str] = []
    reductions = contract.get("reductions") or contract.get("capability_reductions") or []
    if not isinstance(reductions, list):
        return 2, ["'reductions' must be a list of {field, advertised, supported, reason, evidence}."]
    for r in reductions:
        field = r.get("field", "<unnamed>")
        reason = str(r.get("reason", "")).lower()
        evidence = r.get("evidence")
        if reason not in HARD_LIMIT_REASONS:
            worst = max(worst, 2)
            msgs.append(
                f"reduction of '{field}' has reason '{reason or 'MISSING'}' — only a hard device limit "
                f"({sorted(HARD_LIMIT_REASONS)}) may reduce advertised capability."
            )
        if not evidence:
            worst = max(worst, 2)
            msgs.append(f"reduction of '{field}' has no 'evidence' (largest-feasible measurement required).")
    return worst, msgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--hf-model", default="")
    ap.add_argument("--stage", default="")
    ap.add_argument("--require-contract", action="store_true")
    args = ap.parse_args()

    if not args.model_dir and not args.hf_model:
        print("Neither --model-dir nor --hf-model was provided.", file=sys.stderr)
        return 3

    path = _find_contract(args.model_dir, args.hf_model)
    if path is None:
        msg = f"{CONTRACT_NAME} not found for the target model."
        if args.require_contract:
            print(msg, file=sys.stderr)
            return 2
        print(f"ADVISORY: {msg}", file=sys.stderr)
        return 1

    try:
        contract = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as err:
        print(f"could not read/parse {path}: {err}", file=sys.stderr)
        return 3

    worst = 0
    msgs: list[str] = []

    missing_required = [k for k in REQUIRED_KEYS if k not in contract]
    if missing_required:
        worst = max(worst, 2)
        msgs.append(f"contract missing required keys: {missing_required}")

    missing_reco = [k for k in RECOMMENDED_KEYS if k not in contract]
    if missing_reco:
        worst = max(worst, 1)
        msgs.append(f"contract missing recommended keys (advisory): {missing_reco}")

    # modalities must be a non-empty list; denoise_steps must record what was validated.
    mods = contract.get("modalities")
    if not isinstance(mods, list) or not mods:
        worst = max(worst, 2)
        msgs.append("'modalities' must be a non-empty list (e.g. ['video','audio']).")
    ds = contract.get("denoise_steps")
    if isinstance(ds, dict) and not ds.get("validated"):
        worst = max(worst, 1)
        msgs.append("'denoise_steps.validated' is empty — record the step counts actually generated.")

    rcode, rmsgs = _check_reductions(contract)
    worst = max(worst, rcode)
    msgs += rmsgs

    label = {0: "PASS", 1: "ADVISORY", 2: "CRITICAL"}[worst]
    print(f"[check_diffusion_contract] {label} — {path} (stage={args.stage or 'n/a'})")
    for m in msgs:
        print(f"  - {m}", file=sys.stderr if worst >= 2 else sys.stdout)
    return worst


if __name__ == "__main__":
    sys.exit(main())
