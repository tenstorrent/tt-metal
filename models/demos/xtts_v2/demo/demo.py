# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCAFFOLD-TODO: auto-generated skeleton demo for `coqui/XTTS-v2`.

This file was emitted by ``scaffold_demo_folder`` because the backend
``XTTS-v2 (multilingual TTS)`` has ``use_module_tree=True`` and no sibling template
existed at ``models/demos/xtts_v2``. It runs the HF reference once on
CPU (no tt-metal device traffic yet) and prints the category marker
(``==ASR 0 - OUTPUT``) so the planner's correctness gate can validate output
shape and the iterate loop has a runnable starting point.

To graduate this demo onto tt-metal, the per-component bring-up loop
(``up --auto``) will progressively replace the HF forward call with
TTNN components under ``_stubs/`` and ``tt/``. The marker emission
stays intact -- the correctness comparator reads it regardless of
whether the underlying compute is CPU or device.
"""
from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path
from typing import Any

import pytest


HF_MODEL_ID = "coqui/XTTS-v2"
CATEGORY = "STT"
MARKER = "==ASR 0 - OUTPUT"


def _emit_marker(payload: Any) -> None:
    print(MARKER)
    if payload is None:
        print("(no payload)")
        return
    try:
        import numpy as np

        if hasattr(payload, "detach"):
            arr = payload.detach().to("cpu").to(dtype_=None) if False else payload.detach().cpu().float().numpy()
        else:
            arr = np.asarray(payload)
        buf = io.BytesIO()
        np.save(buf, arr)
        print(base64.b64encode(buf.getvalue()).decode("ascii"))
    except Exception as exc:
        print(f"(marker payload serialization failed: {type(exc).__name__}: {exc})")
        print(repr(payload)[:512])


@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_demo(device_params, device):
    import torch
    from transformers import AutoModel, AutoTokenizer

    os.environ.setdefault("HF_MODEL", HF_MODEL_ID)
    tok = AutoTokenizer.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        HF_MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    sentence = os.environ.get("TT_PLANNER_PROBE_INPUT", "The quick brown fox jumps over the lazy dog.")
    inputs = tok(sentence, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)

    if MARKER.startswith("==EMBED"):
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None and isinstance(out, (tuple, list)) and out:
            hidden = out[0]
        if hidden is not None:
            _emit_marker(hidden[0, -1, :])
        else:
            _emit_marker(None)
    elif MARKER.startswith("==CLASS"):
        logits = getattr(out, "logits", None)
        if logits is None:
            logits = out
        _emit_marker(logits[0])
    else:
        last = getattr(out, "last_hidden_state", None)
        _emit_marker(last[0, -1, :] if last is not None else None)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__ + "::test_demo", "-svv"]))
