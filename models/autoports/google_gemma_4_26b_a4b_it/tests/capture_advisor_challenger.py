# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 copy/fill hooks for advisor-challenger's ``capture_template.py``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MODEL_DIR = "google_gemma_4_26b_a4b_it"
ROOT = Path(__file__).resolve().parents[4]
CAPTURE_TEMPLATE = ROOT / ".agents/skills/advisor-challenger/scripts/capture_template.py"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
for extra in (ROOT / "python_env/lib/python3.12/site-packages", ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.append(str(extra))
from models.autoports.google_gemma_4_26b_a4b_it.tests.advisor_challenger_harness import _decode_state
import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import (
    HIDDEN_SIZE, NUM_Q_HEADS, _make_decode_height_sharded_memory_config,
    _make_decode_rope_memory_config,
)
LAYER_KIND = os.environ["CHALLENGER_LAYER_KIND"]
BATCH = int(os.environ["SHARD_ADVISE_BATCH"])
assert BATCH == int(os.environ["CHALLENGER_DECODE_BATCH"]) == 1
INCUMBENT_PATH = Path(os.environ.get(
    "CHALLENGER_INCUMBENT_JSON",
    ROOT / f"models/autoports/{MODEL_DIR}/doc/advisor_challenger/incumbent.json",
))
INCUMBENT = json.loads(INCUMBENT_PATH.read_text())
_STATE = None


def make_inputs(device):
    global _STATE
    _STATE = _decode_state(device, INCUMBENT["shipped_policy"])
    return (_STATE[1],)


def _attention_decode(x, decoder, kwargs):
    kind = decoder.layer_kind
    batch = x.shape[-2]
    xqkv = ttnn.linear(x, decoder.weights.qkv, dtype=decoder.activation_dtype,
                       memory_config=ttnn.DRAM_MEMORY_CONFIG,
                       compute_kernel_config=decoder.attention_compute_config)
    head_mem = _make_decode_height_sharded_memory_config(decoder.mesh_device, batch, kind.head_dim)
    q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads_decode(
        xqkv, num_heads=NUM_Q_HEADS, num_kv_heads=kind.num_kv_heads, memory_config=head_mem)
    q_heads = ttnn.to_memory_config(q_heads, ttnn.L1_MEMORY_CONFIG, dtype=q_heads.dtype)
    k_heads = ttnn.to_memory_config(k_heads, ttnn.L1_MEMORY_CONFIG, dtype=k_heads.dtype)
    v_heads = ttnn.to_memory_config(v_heads, ttnn.L1_MEMORY_CONFIG, dtype=v_heads.dtype)
    q_heads = decoder._rms_norm(q_heads, decoder.weights.q_norm)
    k_heads = decoder._rms_norm(k_heads, decoder.weights.k_norm)
    v_heads = decoder._rms_norm(v_heads, None)
    if kind.name == "full_attention":
        q_heads = ttnn.transpose(q_heads, 1, 2)
        k_heads = ttnn.transpose(k_heads, 1, 2)
        q_heads = ttnn.experimental.rotary_embedding_hf(
            q_heads, kwargs["position_cos"], kwargs["position_sin"], is_decode_mode=False)
        k_heads = ttnn.experimental.rotary_embedding_hf(
            k_heads, kwargs["position_cos"], kwargs["position_sin"], is_decode_mode=False)
        q_heads = ttnn.transpose(q_heads, 1, 2)
        k_heads = ttnn.transpose(k_heads, 1, 2)
        q_heads = ttnn.to_memory_config(q_heads, head_mem, dtype=q_heads.dtype)
        k_heads = ttnn.to_memory_config(k_heads, head_mem, dtype=k_heads.dtype)
        v_heads = ttnn.to_memory_config(v_heads, head_mem, dtype=v_heads.dtype)
    else:
        q_heads = ttnn.to_memory_config(q_heads, head_mem, dtype=q_heads.dtype)
        k_heads = ttnn.to_memory_config(k_heads, head_mem, dtype=k_heads.dtype)
        v_heads = ttnn.to_memory_config(v_heads, head_mem, dtype=v_heads.dtype)
        rope_mem = _make_decode_rope_memory_config(decoder.mesh_device, batch, kind.head_dim)
        cos = ttnn.interleaved_to_sharded(kwargs["position_cos"], rope_mem)
        sin = ttnn.interleaved_to_sharded(kwargs["position_sin"], rope_mem)
        q_heads = ttnn.experimental.rotary_embedding_hf(q_heads, cos, sin, is_decode_mode=True)
        k_heads = ttnn.experimental.rotary_embedding_hf(k_heads, cos, sin, is_decode_mode=True)
    key_cache, value_cache = kwargs["kv_cache"]
    view = decoder._cache_view_kwargs(prefill=False)
    ttnn.experimental.paged_update_cache(key_cache, k_heads, update_idxs_tensor=kwargs["current_pos"],
        page_table=kwargs["page_table"], **view)
    ttnn.experimental.paged_update_cache(value_cache, v_heads, update_idxs_tensor=kwargs["current_pos"],
        page_table=kwargs["page_table"], **view)
    attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q_heads, key_cache, value_cache, page_table_tensor=kwargs["page_table"],
        cur_pos_tensor=kwargs["current_pos"], scale=1.0, sliding_window_size=kind.sliding_window,
        program_config=decoder.sdpa_program_config, memory_config=ttnn.DRAM_MEMORY_CONFIG, **view)
    attn_out = ttnn.to_memory_config(attn_out, head_mem, dtype=attn_out.dtype)
    attn_out = ttnn.experimental.nlp_concat_heads_decode(attn_out, num_heads=NUM_Q_HEADS)
    attn_out = ttnn.sharded_to_interleaved(attn_out, ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.linear(attn_out, decoder.weights.o_proj, dtype=decoder.activation_dtype,
                       memory_config=ttnn.DRAM_MEMORY_CONFIG,
                       compute_kernel_config=decoder.attention_compute_config)


def decode(hidden):
    decoder, _, kwargs = _STATE
    residual = decoder._residual_sharded(hidden)
    attn_out = _attention_decode(decoder._rms_norm(hidden, decoder.weights.input_ln), decoder, kwargs)
    hidden = ttnn.add(residual, decoder._residual_norm(decoder._residual_sharded(attn_out),
        decoder.weights.post_attn_ln), memory_config=decoder.decode_residual_memory_config)
    mlp_in = decoder._residual_interleaved(decoder._residual_norm(hidden, decoder.weights.pre_ff_ln))
    mlp = decoder._dense_mlp(mlp_in, fold_activation=False)
    return decoder._residual_norm(decoder._residual_sharded(mlp), decoder.weights.post_ff_ln_1)


def record_capture(out_dir):
    commit = subprocess.check_output(
        ["git", "-C", os.environ["TTMLIR_ADVISOR_HOME"], "rev-parse", "HEAD"], text=True
    ).strip()
    out = Path(out_dir)
    traced_path = out / "traced_dtypes.json"
    previous = json.loads(traced_path.read_text()) if traced_path.is_file() else {}
    payload = {
        "layer_kind": LAYER_KIND,
        "layer_idx": {"sliding_attention": 0, "full_attention": 5}[LAYER_KIND],
        "batch": BATCH, "capture_batch": BATCH, "requested_decode_batch": 1,
        "traced_weight_dtypes": INCUMBENT["shipped_weight_dtypes"],
        "shipped_weight_dtypes": INCUMBENT["shipped_weight_dtypes"],
        "policy_source": INCUMBENT["shipped_policy_source"],
        "advisor_commit": commit, "advisor_pin_expected": "618cd4e75d",
        "captured_at": previous.get("captured_at") or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "capture_template": str(CAPTURE_TEMPLATE.relative_to(ROOT)),
    }
    out.mkdir(parents=True, exist_ok=True)
    traced_path.write_text(json.dumps(payload, indent=2) + "\n")
    report_path = out / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        report.update(
            capture_batch=BATCH,
            captured_at=payload["captured_at"],
            capture_policy_source=str(INCUMBENT_PATH),
            traced_weight_dtypes=INCUMBENT["shipped_weight_dtypes"],
            capture_template=payload["capture_template"],
            allow_bf16_dram_sharded_matmul=True,
        )
        report["uncapturable"] = {
            "ops": ["ttnn.sparse_matmul"],
            "reason": "Gemma-4 MoE expert suffix is tracer-terminal at sparse_matmul",
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    record_capture(os.environ["CHALLENGER_OUT_DIR"])
