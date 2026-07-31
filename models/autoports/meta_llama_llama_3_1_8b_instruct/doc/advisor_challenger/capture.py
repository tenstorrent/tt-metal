# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Batch-32 dense-layer capture filled from capture_template.py."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch
import ttnn

_ROOT = Path(__file__).parents[5]
_TEMPLATE = _ROOT / ".agents/skills/advisor-challenger/scripts/capture_template.py"
_SPEC = importlib.util.spec_from_file_location("advisor_challenger_capture_template", _TEMPLATE)
template = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(template)

_DECODER = None
_KWARGS = None


def _build(device):
    from models.autoports.meta_llama_llama_3_1_8b_instruct.doc.advisor_challenger.harness import build

    policy = template.SHIPPED_POLICY
    decoder, hidden, current_pos, rot_mats, page_table = build(device, policy)
    return decoder, hidden, {"current_pos": current_pos, "rot_mats": rot_mats, "page_table": page_table}


def decode(hidden):
    return _DECODER.decode_forward(hidden, **_KWARGS)


def make_inputs(device):
    global _DECODER, _KWARGS
    _DECODER, hidden, _KWARGS = _build(device)
    template._record_traced_dtypes(os.environ["CHALLENGER_OUT_DIR"])
    return (hidden,)


if __name__ == "__main__":
    if os.environ.get("CHALLENGER_FINALIZE_CAPTURE") == "1":
        template.finalize_report_metadata(os.environ["CHALLENGER_OUT_DIR"])
        raise SystemExit(0)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        make_inputs(mesh)
        print("capture target builds: kind=dense idx=0 batch=32")
    finally:
        ttnn.close_mesh_device(mesh)
