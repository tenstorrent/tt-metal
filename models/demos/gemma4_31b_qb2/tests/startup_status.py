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


def startup_status(text):
    """Return only fixed labels, booleans and a bounded layer index."""
    positions = [(text.rfind(marker), stage) for stage, marker in STAGES]
    position, stage = max(positions)
    layers = [int(value) for value in re.findall(r"Loading Gemma4 layer (\d{1,2})\b", text)]
    return {
        "last_stage": stage if position >= 0 else "unknown",
        "last_layer_started": next((layer for layer in reversed(layers) if 0 <= layer < 60), None),
        "exception_logged": "Traceback (most recent call last)" in text,
    }


if __name__ == "__main__":
    log_path, output_path = map(Path, sys.argv[1:])
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    output_path.write_text(json.dumps(startup_status(text), indent=2) + "\n")
