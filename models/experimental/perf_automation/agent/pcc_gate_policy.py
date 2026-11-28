"""Decide where optimize's correctness gate comes from — and refuse when there is no ground truth.

Optimize reverts any edit whose PCC drops, so the gate is the only thing standing between a perf
lever and a silently degraded model. That makes "no usable gate" a stop condition, not something to
paper over.

The failure this exists to prevent (llama3_1_8b_p150, 2026-07-25): a file named `test_pcc.py` was
supplied, but it asserted top-1 TOKEN ACCURACY and never printed a PCC. The threshold extractor
found none and returned its 0.99 default, which satisfied the fatal `no_pcc_threshold` check, and
the verdict path returned "ok" whenever no PCC was parsed. So every edit was waved through with
correctness never numerically checked -- and token accuracy cannot substitute: quantization error
is proportional to magnitude, so it preserves the argmax ranking while wrecking the values. Measured
on realistic logits: bf4_b sat at PCC 0.513 with 100% top-1 match, clearing an 0.86 accuracy floor.

POLICY (deliberately conservative):
  1. A gate was SUPPLIED via --pcc-test -> use it. Always. The operator's choice is never silently
     replaced; if it declares no threshold that is a loud warning, not a substitution.
  2. NO gate supplied, but the model resolves to a cached HF reference -> generate an HF-referenced
     gate: ground truth exists, so absolute correctness is checkable.
  3. Neither -> STOP. Without an HF reference and without a supplied gate there is no ground truth,
     and a fabricated gate would only manufacture confidence.

A baseline-referenced variant (compare each edit against the UNEDITED model's own output) was
considered and REJECTED: it has no gate file, and a threshold declared in a gate file is the only
way this tool learns a floor. It would therefore fall back to PERF_MCP_PCC_MIN, default 0.0 -- every
edit passing while the run reports that a gate is active. A gate that cannot fail is worse than an
absent one, because it manufactures the appearance of enforcement.
"""

from __future__ import annotations

import os
from pathlib import Path

USE_SUPPLIED = "use_supplied"
GENERATE_FROM_HF = "generate_from_hf"
STOP = "stop"


def _hf_hub_root() -> Path:
    env = os.environ.get("HF_HOME")
    if env:
        return Path(env) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def hf_reference_available(model_id: str | None) -> bool:
    """Is there a locally cached HF snapshot with weights for this model id?

    Requires actual weight files, not just a directory: an aborted download leaves the folder behind
    and would otherwise read as "reference available", producing a gate that cannot run.
    """
    if not model_id or "/" not in str(model_id):
        return False
    org, _, name = str(model_id).partition("/")
    snaps = _hf_hub_root() / f"models--{org}--{name}" / "snapshots"
    try:
        for snap in snaps.iterdir():
            if not snap.is_dir():
                continue
            for p in snap.rglob("*"):
                if p.suffix.lower() in (".safetensors", ".bin", ".pt", ".pth") and p.stat().st_size > 0:
                    return True
    except OSError:
        return False
    return False


def supplied_gate_is_usable(pcc_test_path) -> tuple[bool, str]:
    """Does the supplied file actually work as a PCC gate? Returns (usable, reason).

    Checked statically so a bad gate is caught at discovery rather than after hours of running:
    the file must exist AND declare a numeric PCC threshold. A file that merely *runs* is not
    enough -- that is exactly how a token-accuracy test passed for a PCC gate.
    """
    if not pcc_test_path:
        return False, "no --pcc-test supplied"
    path = Path(str(pcc_test_path).split("::", 1)[0])
    if not path.is_file():
        return False, "supplied gate not found: %s" % path
    try:
        from .model_files import pcc_threshold_is_declared
    except Exception:  # noqa: BLE001
        return True, "threshold check unavailable; accepting the supplied gate"
    if not pcc_threshold_is_declared(path):
        return False, (
            "%s declares no PCC threshold, so it does not compute a PCC. A gate that cannot fail on "
            "numerical damage is not a correctness gate (token-accuracy style checks stay ~100%% "
            "while PCC collapses under quantization)." % path.name
        )
    return True, "supplied gate declares a PCC threshold"


def decide(pcc_test_path=None, model_id: str | None = None) -> dict:
    """Pick the gate source. Returns {action, reason, model_id, warning?}.

    A SUPPLIED gate always wins -- the HF path is a fallback for when none was given, never a
    silent replacement for the operator's choice. If the supplied file turns out not to emit a PCC,
    the run fails at the gate (run_pcc returns pcc_low) rather than being quietly substituted.
    """
    usable, why = supplied_gate_is_usable(pcc_test_path)
    if usable:
        return {"action": USE_SUPPLIED, "reason": why, "model_id": model_id}

    # A file WAS supplied but declares no threshold. Still use it -- do not override the operator --
    # but say so loudly, because the runtime gate will reject every edit if it emits no PCC.
    if pcc_test_path and Path(str(pcc_test_path).split("::", 1)[0]).is_file():
        return {
            "action": USE_SUPPLIED,
            "reason": why,
            "model_id": model_id,
            "warning": (
                "the supplied gate declares no PCC threshold. It is being used as given, but if it "
                "does not print `PCC: <float>` every edit will be rejected as unverified."
            ),
        }

    if hf_reference_available(model_id):
        return {
            "action": GENERATE_FROM_HF,
            "reason": "%s; a cached HF reference exists for %s, so an HF-referenced gate can be "
            "generated (ground truth available)" % (why, model_id),
            "model_id": model_id,
        }

    return {
        "action": STOP,
        "reason": (
            "%s, and no cached HF reference for %r. There is no ground truth to check correctness "
            "against, so optimize would be free to commit edits that silently degrade the model. "
            "PLEASE GIVE A PCC TEST TO RUN OPTIMIZE: pass --pcc-test <file>::<test>. It must "
            "compare the model against a reference, print `PCC: <float>`, and declare its own "
            "numeric threshold."
        )
        % (why, model_id),
        "model_id": model_id,
    }
