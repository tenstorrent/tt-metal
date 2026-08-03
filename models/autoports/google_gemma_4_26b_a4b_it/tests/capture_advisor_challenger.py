# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 model hooks for advisor-challenger capture_template.py."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
repo_packages = ROOT / "python_env/lib/python3.12/site-packages"
if str(repo_packages) not in sys.path:
    sys.path.append(str(repo_packages))

import ttnn

from models.autoports.google_gemma_4_26b_a4b_it.tests.advisor_challenger_harness import _decode_state

MODEL_DIR = "google_gemma_4_26b_a4b_it"
LAYER_KIND = os.environ["CHALLENGER_LAYER_KIND"]
BATCH = int(os.environ["SHARD_ADVISE_BATCH"])
assert BATCH == int(os.environ["CHALLENGER_DECODE_BATCH"]) == 1
INCUMBENT_PATH = ROOT / f"models/autoports/{MODEL_DIR}/doc/advisor_challenger/incumbent.json"
INCUMBENT = json.loads(INCUMBENT_PATH.read_text())
SHIPPED_POLICY = INCUMBENT["shipped_policy"]
SHIPPED_DTYPES = INCUMBENT["shipped_weight_dtypes"]

_DECODER = None
_KWARGS = None


def decode(hidden):
    """Capture the shipped decode prefix through the dense MLP.

    Routed experts use sparse_matmul, which is terminal in the advisor tracer;
    their measured share is retained as unreachable rather than suppressed.
    """
    residual = hidden
    attn_in = _DECODER._rms_norm(hidden, _DECODER.weights.input_ln)
    attn_out = _DECODER._attention_decode(attn_in, cache_position_modulo=None, **_KWARGS)
    attn_out = _DECODER._rms_norm(attn_out, _DECODER.weights.post_attn_ln)
    hidden = ttnn.add(residual, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    mlp_in = _DECODER._rms_norm(hidden, _DECODER.weights.pre_ff_ln)
    mlp_out = _DECODER._dense_mlp(mlp_in)
    return _DECODER._rms_norm(mlp_out, _DECODER.weights.post_ff_ln_1)


def make_inputs(device):
    global _DECODER, _KWARGS
    _DECODER, hidden, _KWARGS = _decode_state(device, SHIPPED_POLICY)
    return (hidden,)


def record_capture(out_dir):
    commit = subprocess.check_output(
        ["git", "-C", os.environ["TTMLIR_ADVISOR_HOME"], "rev-parse", "HEAD"], text=True
    ).strip()
    payload = {
        "layer_kind": LAYER_KIND,
        "layer_idx": {"sliding_attention": 0, "full_attention": 5}[LAYER_KIND],
        "batch": BATCH,
        "capture_batch": BATCH,
        "requested_decode_batch": 1,
        "traced_weight_dtypes": SHIPPED_DTYPES,
        "shipped_weight_dtypes": SHIPPED_DTYPES,
        "policy_source": INCUMBENT["shipped_policy_source"],
        "advisor_commit": commit,
        "advisor_pin_expected": "618cd4e75d",
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "traced_dtypes.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    record_capture(os.environ["CHALLENGER_OUT_DIR"])
