# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Model-filled advisor-challenger capture template for North-Mini batch-1 decode."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.coherelabs_north_mini_code_1_0.tests.test_functional_decoder import (
    REAL_REVISION,
    _decode_inputs,
    _page_table,
    _synthetic_state,
    _to_tt,
)
from models.autoports.coherelabs_north_mini_code_1_0.tt.optimized_decoder import MODEL_ID, OptimizedDecoder

HERE = Path(__file__).resolve().parent
ADVISOR_PIN = "618cd4e75d"
MODEL_DIR = "coherelabs_north_mini_code_1_0"
LAYER_KIND = os.environ["CHALLENGER_LAYER_KIND"]
BATCH = int(os.environ.get("SHARD_ADVISE_BATCH", "1"))
if BATCH != 1:
    raise SystemExit(f"advisor-challenger capture requires decode batch 1, got {BATCH}")
LAYER_BY_KIND = {"dense_full_forced_rope": 0, "sliding_rope_moe": 1, "full_no_rope_moe": 4}
LAYER_IDX = LAYER_BY_KIND[LAYER_KIND]

with (HERE / "incumbent.json").open() as fh:
    _incumbent = json.load(fh)
SHIPPED_POLICY = _incumbent["shipped_policy"]
SHIPPED_DTYPES = _incumbent["shipped_weight_dtypes"]
if "constructor_default" in (_incumbent.get("shipped_policy_source") or "").lower():
    raise SystemExit("refusing to capture constructor defaults instead of executed policy")

OUT_DIR = HERE / "shard_advise" / LAYER_KIND
_STATE = None


def _build(device):
    global _STATE
    config = AutoConfig.from_pretrained(MODEL_ID, revision=REAL_REVISION, local_files_only=True)
    state = _synthetic_state(config, LAYER_IDX, sparse_weights=True)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=device,
        batch=BATCH,
        max_cache_len=32,
        **SHIPPED_POLICY,
    )
    generator = torch.Generator().manual_seed(18000 + LAYER_IDX)
    hidden = _to_tt(
        (torch.randn(1, BATCH, 1, config.hidden_size, generator=generator) * 0.02).to(torch.bfloat16), device
    )
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(_page_table(BATCH, 1), device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    current, cos, sin = _decode_inputs(decoder, config, device, [0] * BATCH)
    _STATE = {
        "decoder": decoder,
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
    return decoder


def make_inputs(device):
    _build(device)
    return _STATE["hidden"]


def decode(hidden):
    _STATE["hidden"] = hidden
    decoder = _STATE["decoder"]
    if decoder.mlp_type == "dense":
        return decoder.decode_forward(hidden, **_STATE["kwargs"])

    # The pinned tracer cannot lower the exact route-presence ones_like and
    # sparse_matmul is terminal immediately afterward.  Preserve the shipped
    # residual/norm/attention prefix so advice for the reachable portion is
    # still measured; the sparse expert suffix is explicitly unreachable.
    residual = ttnn.to_memory_config(hidden, decoder.decode_residual_memory_config)
    normalized = ttnn.rms_norm(
        residual,
        epsilon=decoder.eps,
        weight=decoder.weights["norm"],
        program_config=decoder.decode_norm_program_config,
        memory_config=decoder.decode_residual_memory_config,
        compute_kernel_config=decoder.decode_norm_compute_config,
    )
    attention = decoder._attention_decode(
        normalized,
        key_cache=_STATE["kwargs"]["key_cache"],
        value_cache=_STATE["kwargs"]["value_cache"],
        page_table=_STATE["kwargs"]["page_table"],
        current_positions=_STATE["kwargs"]["current_positions"],
        position_cos=_STATE["kwargs"]["position_cos"],
        position_sin=_STATE["kwargs"]["position_sin"],
    )
    attention = ttnn.to_memory_config(attention, decoder.decode_residual_memory_config)
    return ttnn.add(residual, attention, memory_config=decoder.decode_residual_memory_config)


def record_capture():
    advisor_home = os.environ["TTMLIR_ADVISOR_HOME"]
    commit = subprocess.run(
        ["git", "-C", advisor_home, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    provenance = {
        "layer_kind": LAYER_KIND,
        "layer_idx": LAYER_IDX,
        "batch": BATCH,
        "traced_weight_dtypes": SHIPPED_DTYPES,
        "shipped_weight_dtypes": SHIPPED_DTYPES,
        "policy_source": _incumbent["shipped_policy_source"],
        "advisor_commit": commit,
        "advisor_pin_expected": ADVISOR_PIN,
        "advisor_home": advisor_home,
        "capture_template": ".agents/skills/advisor-challenger/scripts/capture_template.py",
    }
    (OUT_DIR / "traced_dtypes.json").write_text(json.dumps(provenance, indent=2) + "\n")
    report_path = OUT_DIR / "report.json"
    report = json.loads(report_path.read_text())
    report.update(
        {
            "traced_weight_dtypes": SHIPPED_DTYPES,
            "capture_policy_source": _incumbent["shipped_policy_source"],
            "capture_batch": BATCH,
            "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "capture_template": ".agents/skills/advisor-challenger/scripts/capture_template.py",
        }
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    record_capture()
