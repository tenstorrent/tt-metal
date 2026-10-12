# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Extract fixed startup milestones without publishing server log content."""

import json
import re
import sys
from pathlib import Path

STAGES = (
    ("mesh_open", "Attempting to open mesh device"),
    ("weight_loading", "Loading Gemma4 layer "),
    ("weights_loaded", "Gemma4 model loaded"),
    ("model_warmup", "Warming model trace"),
    ("sampling_warmup", "Warming sampling trace"),
    ("model_trace_capture", "Capturing model trace"),
    ("sampling_trace_capture", "Capturing sampling trace"),
)

FAILURES = (
    ("active_ethernet_timeout", "Timed out while waiting for active ethernet core"),
    ("ethernet_heartbeat", "ETH core heartbeat check failed"),
    ("missing_weights", "GatedRepoError"),
    ("missing_weights", "LocalEntryNotFoundError"),
    ("missing_module", "ModuleNotFoundError"),
    ("import_error", "ImportError"),
    ("out_of_memory", "Out of memory"),
)


def startup_status(text):
    """Return only fixed labels, booleans and a bounded layer index."""
    positions = [(text.rfind(marker), stage) for stage, marker in STAGES]
    position, stage = max(positions)
    layers = [int(value) for value in re.findall(r"Loading Gemma4 layer (\d{1,2})\b", text)]
    failure_category = next((category for category, marker in FAILURES if marker in text), "unknown")
    return {
        "last_stage": stage if position >= 0 else "unknown",
        "last_layer_started": next((layer for layer in reversed(layers) if 0 <= layer < 30), None),
        "exception_logged": "Traceback (most recent call last)" in text,
        "failure_category": failure_category,
    }


if __name__ == "__main__":
    log_path, output_path = map(Path, sys.argv[1:])
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    output_path.write_text(json.dumps(startup_status(text), indent=2) + "\n")
