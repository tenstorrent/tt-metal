# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""North-Mini hooks for the advisor-challenger capture template."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch
import ttnn

# The pinned advisor environment intentionally carries TTNN/torch but not the
# model-side Hugging Face helpers. Append (never prepend) the live checkout's
# Python packages after importing advisor TTNN, so model config loading does
# not replace the pinned tracer runtime.
sys.path.append(str(_ROOT / "python_env/lib/python3.12/site-packages") if "_ROOT" in globals() else
                "/home/mvasiljevic/tt-metal/python_env/lib/python3.12/site-packages")
from transformers import AutoConfig

_ROOT = Path(__file__).resolve().parents[5]
_SPEC = importlib.util.spec_from_file_location(
    "advisor_challenger_capture_template",
    _ROOT / ".agents/skills/advisor-challenger/scripts/capture_template.py",
)
assert _SPEC and _SPEC.loader
_TEMPLATE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TEMPLATE)

from models.autoports.coherelabs_north_mini_code_1_0.tests.test_functional_decoder import (
    REAL_REVISION,
    _decode_inputs,
    _page_table,
    _synthetic_state,
    _to_tt,
)
from models.autoports.coherelabs_north_mini_code_1_0.tt.functional_decoder import MODEL_ID
from models.autoports.coherelabs_north_mini_code_1_0.tt.optimized_decoder import OptimizedDecoder

MODEL_DIR = _TEMPLATE.MODEL_DIR
LAYER_KIND = _TEMPLATE.LAYER_KIND
LAYER_IDX = _TEMPLATE.LAYER_IDX
BATCH = _TEMPLATE.BATCH
SHIPPED_POLICY = _TEMPLATE.SHIPPED_POLICY
SHIPPED_DTYPES = _TEMPLATE.SHIPPED_DTYPES

_DECODER = None
_STATE = None


def _config():
    return AutoConfig.from_pretrained(MODEL_ID, revision=REAL_REVISION, local_files_only=True)


def _synthetic_state_dict(config):
    return _synthetic_state(config, LAYER_IDX, sparse_weights=LAYER_IDX != 0)


def _build(device):
    global _DECODER, _STATE
    config = _config()
    state_dict = _synthetic_state_dict(config)
    constructor_policy = {k: v for k, v in SHIPPED_POLICY.items() if k != "layer_idx"}
    _DECODER = OptimizedDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=device,
        batch=BATCH,
        max_cache_len=32,
        **constructor_policy,
    )
    generator = torch.Generator().manual_seed(20260803 + LAYER_IDX)
    hidden = _to_tt(
        (torch.randn(1, BATCH, 1, config.hidden_size, generator=generator) * 0.02).to(torch.bfloat16),
        device,
    )
    key_cache, value_cache = _DECODER.create_paged_kv_cache()
    page_table = _to_tt(
        _page_table(BATCH, 1), device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    current, cos, sin = _decode_inputs(_DECODER, config, device, [0] * BATCH)
    _STATE = {
        "hidden": hidden,
        "kwargs": {
            "key_cache": key_cache,
            "value_cache": value_cache,
            "page_table": page_table,
            "current_positions": current,
            "position_cos": cos,
            "position_sin": sin,
        },
    }
    _TEMPLATE._DECODER = _DECODER
    _TEMPLATE._CONFIG = config
    _TEMPLATE._WEIGHTS = _DECODER
    return _DECODER


def decode(hidden=None):
    if _DECODER is None:
        raise RuntimeError("capture template did not build the decoder")
    hidden = _STATE["hidden"] if hidden is None else hidden
    if os.environ.get("CHALLENGER_CAPTURE_ATTENTION_ONLY") == "1":
        normalized = ttnn.rms_norm(hidden, epsilon=_DECODER.eps, weight=_DECODER.weights["norm"])
        return _DECODER._attention_decode(normalized, **_STATE["kwargs"])
    return _DECODER.decode_forward(hidden, **_STATE["kwargs"])


def make_inputs(device):
    """The capture-template hook consumed by ``ttnn-advise capture``."""
    _build(device)
    _TEMPLATE._record_traced_dtypes(
        str(Path(__file__).parent / "shard_advise" / LAYER_KIND)
    )
    return (_STATE["hidden"],)


_TEMPLATE._config = _config
_TEMPLATE._synthetic_state_dict = _synthetic_state_dict
_TEMPLATE._build = _build
_TEMPLATE.decode = decode


if __name__ == "__main__":
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        _build(device)
        _TEMPLATE._record_traced_dtypes(str(Path(__file__).parent / "shard_advise" / LAYER_KIND))
        print(f"capture target builds: kind={LAYER_KIND} idx={LAYER_IDX} batch={BATCH}")
    finally:
        ttnn.close_mesh_device(device)
