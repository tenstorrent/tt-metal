# SPDX-License-Identifier: Apache-2.0
"""North-Mini advisor capture target, adapted from capture_template.py."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import torch

import ttnn

TT_METAL_ROOT = os.environ.get("TT_METAL_ROOT", "/home/mvasiljevic/tt-metal")
if TT_METAL_ROOT not in sys.path:
    sys.path.append(TT_METAL_ROOT)

MODEL_DIR = "coherelabs_north_mini_code_1_0"
LAYER_KIND = os.environ["CHALLENGER_LAYER_KIND"]
LAYER_IDX = int(os.environ["CHALLENGER_LAYER_IDX"])
BATCH = int(os.environ.get("SHARD_ADVISE_BATCH", "1"))
INCUMBENT = os.environ.get(
    "CHALLENGER_INCUMBENT_JSON",
    f"{TT_METAL_ROOT}/models/autoports/{MODEL_DIR}/doc/advisor_challenger/incumbent.json",
)
with open(INCUMBENT) as fh:
    FROZEN = json.load(fh)
SHIPPED_POLICY = FROZEN["shipped_policy"]
SHIPPED_DTYPES = FROZEN["shipped_weight_dtypes"]

_DECODER = None
_KWARGS = None


def _build(device):
    from models.autoports.coherelabs_north_mini_code_1_0.tests import test_functional_decoder as tests
    from models.autoports.coherelabs_north_mini_code_1_0.tt.optimized_decoder import OptimizedDecoder

    config = tests._config()
    state = tests._synthetic_state(config, LAYER_IDX)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=device,
        batch=BATCH,
        max_cache_len=32,
        candidate=SHIPPED_POLICY["candidate"],
    )
    generator = torch.Generator().manual_seed(31000 + LAYER_IDX)
    hidden = tests._to_tt(
        (torch.randn(1, BATCH, 1, config.hidden_size, generator=generator) * 0.02).to(torch.bfloat16), device
    )
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = tests._to_tt(tests._page_table(BATCH, 1), device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    current, cos, sin = tests._decode_inputs(decoder, config, device, [0] * BATCH)
    kwargs = {
        "key_cache": key_cache,
        "value_cache": value_cache,
        "page_table": page_table,
        "current_positions": current,
        "position_cos": cos if decoder.use_rope else None,
        "position_sin": sin if decoder.use_rope else None,
    }
    return decoder, kwargs, hidden


def decode(hidden):
    # The pinned direct tracer cannot consume TracedTensor inputs in
    # paged_fused_update_cache, and sparse_matmul is terminal by contract.
    # Preserve the real shipped path up to those tracer boundaries.
    if _DECODER.use_sharded_dense_residual:
        hidden = ttnn.to_memory_config(hidden, _DECODER.dense_residual_memory_config)
        normalized = ttnn.rms_norm(
            hidden,
            epsilon=_DECODER.eps,
            weight=_DECODER.weights["norm"],
            program_config=_DECODER.dense_norm_program_config,
            memory_config=_DECODER.dense_residual_memory_config,
        )
    else:
        normalized = ttnn.rms_norm(hidden, epsilon=_DECODER.eps, weight=_DECODER.weights["norm"])
    query, key, value = _DECODER._qkv_decode(normalized, _KWARGS["position_cos"], _KWARGS["position_sin"])
    if _DECODER.mlp_type == "dense":
        return _DECODER._dense_mlp(normalized)
    return query


def make_inputs(device):
    global _DECODER, _KWARGS
    _DECODER, _KWARGS, hidden = _build(device)
    return (hidden,)


def record_traced_dtypes(out_dir):
    commit = subprocess.run(
        ["git", "-C", os.environ["TTMLIR_ADVISOR_HOME"], "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "traced_dtypes.json"), "w") as fh:
        json.dump(
            {
                "layer_kind": LAYER_KIND,
                "layer_idx": LAYER_IDX,
                "batch": BATCH,
                "traced_weight_dtypes": SHIPPED_DTYPES,
                "shipped_weight_dtypes": SHIPPED_DTYPES,
                "policy_source": FROZEN["shipped_policy_source"],
                "advisor_commit": commit,
                "advisor_pin_expected": "618cd4e75d",
                "advisor_home": os.environ["TTMLIR_ADVISOR_HOME"],
            },
            fh,
            indent=2,
        )
