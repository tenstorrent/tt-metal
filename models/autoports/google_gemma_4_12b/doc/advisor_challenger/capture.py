# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Model-filled copy of advisor-challenger/scripts/capture_template.py.

Capture one frozen-incumbent Gemma-4 decoder step at decode batch 32.  Shapes
and the synthetic state builder come from the model's decoder test; precision
comes from the already-frozen incumbent record, never constructor defaults.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import ttnn

TT_METAL_ROOT = os.environ.get("TT_METAL_ROOT", "/home/mvasiljevic/tt-metal")
ADVISOR_PIN = os.environ.get("CHALLENGER_ADVISOR_PIN", "618cd4e75d")
if TT_METAL_ROOT not in sys.path:
    sys.path.append(TT_METAL_ROOT)

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))
from common import build_state, decode as decode_state, functional

# FILL 1: model and one representative layer per kind.
MODEL_DIR = "google_gemma_4_12b"
LAYER_KIND = os.environ["CHALLENGER_LAYER_KIND"]
BATCH = int(os.environ.get("SHARD_ADVISE_BATCH", "32"))
if BATCH != 32:
    raise SystemExit(f"advisor-challenger capture requires decode batch 32, got {BATCH}")

# FILL 2: executed incumbent policy and weight dtypes.
INCUMBENT = HERE / "incumbent.json"
with INCUMBENT.open() as fh:
    _incumbent = json.load(fh)
SHIPPED_POLICY = _incumbent["shipped_policy"]
SHIPPED_DTYPES = _incumbent["shipped_weight_dtypes"]
if "constructor_default" in (_incumbent.get("shipped_policy_source") or "").lower():
    raise SystemExit("refusing to capture constructor defaults instead of executed policy")

OUT_DIR = HERE / "shard_advise" / LAYER_KIND
_STATE = None


def _build(device):
    """FILL 3/4: real model config/shapes and its synthetic test weights."""
    global _STATE
    _STATE = build_state(device, SHIPPED_POLICY, LAYER_KIND, BATCH)
    return _STATE["decoder"]


def make_inputs(device):
    """ttnn-advise input hook; build once with the incumbent policy."""
    _build(device)
    return _STATE["hidden"]


def decode(hidden):
    """FILL 5: exactly one incumbent decode step."""
    _STATE["hidden"] = hidden
    return decode_state(_STATE)


def _record_traced_dtypes(out_dir: Path) -> None:
    advisor_home = os.environ["TTMLIR_ADVISOR_HOME"]
    commit = subprocess.run(
        ["git", "-C", advisor_home, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "layer_kind": LAYER_KIND,
        "layer_idx": _STATE["layer_idx"] if _STATE else functional._find_layer_idx(functional._hf_text_config(), LAYER_KIND),
        "batch": BATCH,
        "traced_weight_dtypes": SHIPPED_DTYPES,
        "shipped_weight_dtypes": SHIPPED_DTYPES,
        "policy_source": _incumbent["shipped_policy_source"],
        "advisor_commit": commit,
        "advisor_pin_expected": ADVISOR_PIN,
        "advisor_home": advisor_home,
        "capture_template": ".agents/skills/advisor-challenger/scripts/capture_template.py",
    }
    (out_dir / "traced_dtypes.json").write_text(json.dumps(data, indent=2) + "\n")


def _patch_report(out_dir: Path) -> None:
    path = out_dir / "report.json"
    report = json.loads(path.read_text())
    report.update(
        {
            "traced_weight_dtypes": SHIPPED_DTYPES,
            "capture_policy_source": _incumbent["shipped_policy_source"],
            "capture_batch": BATCH,
            "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "capture_template": ".agents/skills/advisor-challenger/scripts/capture_template.py",
        }
    )
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    # Run after `ttnn-advise capture capture.py:decode ...` to persist the
    # template-mandated executed-policy and advisor-pin provenance.
    _record_traced_dtypes(OUT_DIR)
    _patch_report(OUT_DIR)
