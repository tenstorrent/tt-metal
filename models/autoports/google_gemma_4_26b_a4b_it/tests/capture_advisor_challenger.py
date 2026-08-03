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
repo_packages = ROOT / "python_env/lib/python3.12/site-packages"
repo_tools = ROOT / "tools"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(repo_packages) not in sys.path:
    sys.path.append(str(repo_packages))
if str(repo_tools) not in sys.path:
    sys.path.append(str(repo_tools))

import ttnn

from models.autoports.google_gemma_4_26b_a4b_it.tests.advisor_challenger_harness import _decode_state
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import (
    HIDDEN_SIZE,
    NUM_Q_HEADS,
    _make_decode_height_sharded_memory_config,
    _make_decode_rope_memory_config,
)

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


def _attention_decode_capture(x):
    """Mirror the shipped attention path without dynamically querying a traced tensor."""
    kind = _DECODER.layer_kind
    batch = x.shape[-2]
    xqkv = ttnn.linear(x, _DECODER.weights.qkv, dtype=_DECODER.activation_dtype,
                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
    head_mem = _make_decode_height_sharded_memory_config(_DECODER.mesh_device, batch, kind.head_dim)
    q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads_decode(
        xqkv, num_heads=NUM_Q_HEADS, num_kv_heads=kind.num_kv_heads, memory_config=head_mem)
    q_heads = ttnn.to_memory_config(q_heads, ttnn.L1_MEMORY_CONFIG, dtype=q_heads.dtype)
    k_heads = ttnn.to_memory_config(k_heads, ttnn.L1_MEMORY_CONFIG, dtype=k_heads.dtype)
    v_heads = ttnn.to_memory_config(v_heads, ttnn.L1_MEMORY_CONFIG, dtype=v_heads.dtype)
    q_heads = _DECODER._rms_norm(q_heads, _DECODER.weights.q_norm)
    k_heads = _DECODER._rms_norm(k_heads, _DECODER.weights.k_norm)
    v_heads = _DECODER._rms_norm(v_heads, None)
    if kind.name == "full_attention":
        q_heads = ttnn.transpose(q_heads, 1, 2)
        k_heads = ttnn.transpose(k_heads, 1, 2)
        q_heads = ttnn.experimental.rotary_embedding_hf(
            q_heads, _KWARGS["position_cos"], _KWARGS["position_sin"], is_decode_mode=False)
        k_heads = ttnn.experimental.rotary_embedding_hf(
            k_heads, _KWARGS["position_cos"], _KWARGS["position_sin"], is_decode_mode=False)
        q_heads = ttnn.transpose(q_heads, 1, 2)
        k_heads = ttnn.transpose(k_heads, 1, 2)
        q_heads = ttnn.to_memory_config(q_heads, head_mem, dtype=q_heads.dtype)
        k_heads = ttnn.to_memory_config(k_heads, head_mem, dtype=k_heads.dtype)
        v_heads = ttnn.to_memory_config(v_heads, head_mem, dtype=v_heads.dtype)
    else:
        q_heads = ttnn.to_memory_config(q_heads, head_mem, dtype=q_heads.dtype)
        k_heads = ttnn.to_memory_config(k_heads, head_mem, dtype=k_heads.dtype)
        v_heads = ttnn.to_memory_config(v_heads, head_mem, dtype=v_heads.dtype)
        rope_mem = _make_decode_rope_memory_config(_DECODER.mesh_device, batch, kind.head_dim)
        cos = ttnn.interleaved_to_sharded(_KWARGS["position_cos"], rope_mem)
        sin = ttnn.interleaved_to_sharded(_KWARGS["position_sin"], rope_mem)
        q_heads = ttnn.experimental.rotary_embedding_hf(q_heads, cos, sin, is_decode_mode=True)
        k_heads = ttnn.experimental.rotary_embedding_hf(k_heads, cos, sin, is_decode_mode=True)
    key_cache, value_cache = _KWARGS["kv_cache"]
    update_kwargs = _DECODER._cache_view_kwargs(prefill=False)
    ttnn.experimental.paged_update_cache(
        key_cache, k_heads, update_idxs_tensor=_KWARGS["current_pos"],
        page_table=_KWARGS["page_table"], **update_kwargs)
    ttnn.experimental.paged_update_cache(
        value_cache, v_heads, update_idxs_tensor=_KWARGS["current_pos"],
        page_table=_KWARGS["page_table"], **update_kwargs)
    sdpa_kwargs = _DECODER._cache_view_kwargs(prefill=False)
    attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q_heads, key_cache, value_cache, page_table_tensor=_KWARGS["page_table"],
        cur_pos_tensor=_KWARGS["current_pos"], scale=1.0,
        sliding_window_size=kind.sliding_window, program_config=_DECODER.sdpa_program_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG, **sdpa_kwargs)
    attn_out = ttnn.to_memory_config(attn_out, head_mem, dtype=attn_out.dtype)
    attn_out = ttnn.experimental.nlp_concat_heads_decode(attn_out, num_heads=NUM_Q_HEADS)
    attn_out = ttnn.sharded_to_interleaved(attn_out, ttnn.DRAM_MEMORY_CONFIG)
    attn_out = ttnn.linear(attn_out, _DECODER.weights.o_proj, dtype=_DECODER.activation_dtype,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if attn_out.shape[-2] != batch:
        attn_out = ttnn.slice(attn_out, starts=[0, 0, 0, 0], ends=[1, 1, batch, HIDDEN_SIZE],
                              steps=[1, 1, 1, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return attn_out


def decode(hidden):
    """Capture the shipped decode prefix; sparse experts are tracer-terminal."""
    residual = hidden
    attn_in = _DECODER._rms_norm(hidden, _DECODER.weights.input_ln)
    attn_out = _attention_decode_capture(attn_in)
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
        "layer_kind": LAYER_KIND, "layer_idx": {"sliding_attention": 0, "full_attention": 5}[LAYER_KIND],
        "batch": BATCH, "capture_batch": BATCH, "requested_decode_batch": 1,
        "traced_weight_dtypes": SHIPPED_DTYPES, "shipped_weight_dtypes": SHIPPED_DTYPES,
        "policy_source": INCUMBENT["shipped_policy_source"], "advisor_commit": commit,
        "advisor_pin_expected": "618cd4e75d", "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "traced_dtypes.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    record_capture(os.environ["CHALLENGER_OUT_DIR"])
