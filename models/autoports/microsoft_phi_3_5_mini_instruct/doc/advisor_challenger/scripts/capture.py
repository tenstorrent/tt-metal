# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Capture target template for `$advisor-challenger`. Copy per model, fill the marked sections.

THE ONE THING THIS TEMPLATE EXISTS TO FIX
-----------------------------------------
Three of the four capture scripts written before this template constructed the decoder with NO dtype or
policy argument -- `OptimizedDecoder.from_state_dict(state, hf_config=cfg, layer_idx=..., mesh_device=...)`
-- and therefore traced the CLASS DEFAULTS rather than the precision the model actually ships. On north
that meant the capture traced bf16 attention while the cell shipped bfp8, so the advisor excluded two
matmuls from DRAM-sharding consideration for a dtype the model never used. The stage then had to find DS
BFP8 attention by hand, worth -10.0%.

`SETUP.md` B.1 lists what to reuse from the model's test builders -- "config, synthetic state dict, paged
KV cache, rope, current_pos". It used to stop there, and that omission is what produced the captures above;
it now names the shipped policy too and points this stage here. Recording what was traced is still this
template's job, because only an artifact lets the gate verify it.

So: CONSTRUCT WITH THE SHIPPED POLICY, and record what you traced so the gate can verify it.
The shipped policy comes from what EXECUTED -- the final tt-perf-report CSV or the selected candidate
JSON -- never from `resolved_policy.constructor_defaults`, which are class defaults.

A synthetic state dict is fine: the advisor reasons about layout, not values. What must be real is the
DTYPE and the SHAPES.
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace
from types import MethodType

import torch

import ttnn

TT_METAL_ROOT = os.environ.get("TT_METAL_ROOT", "/home/mvasiljevic/tt-metal")
# $shard-advise SETUP.md A.1 pins tt-mlir here so runs stay comparable. Recorded with every capture.
ADVISOR_PIN = os.environ.get("CHALLENGER_ADVISOR_PIN", "618cd4e75d")
if TT_METAL_ROOT not in sys.path:
    sys.path.append(TT_METAL_ROOT)

# ---- FILL 1: the model under test -------------------------------------------------------------
MODEL_DIR = os.environ["CHALLENGER_MODEL_DIR"]          # e.g. google_gemma_4_26b_a4b_it
LAYER_KIND = os.environ.get("CHALLENGER_LAYER_KIND", "dense")
LAYER_IDX = int(os.environ.get("CHALLENGER_LAYER_IDX", "0"))
BATCH = int(os.environ.get("SHARD_ADVISE_BATCH", "1"))   # MUST match the incumbent's DECODE_BATCH
MAX_CONTEXT = int(os.environ.get("CHALLENGER_MAX_CONTEXT", "64"))

# One capture per LAYER KIND. A single capture lets the whole search follow it to one kind: qwen's arm
# captured LAYER_IDX=3 (a full_attention layer) and then ran candidate variants on that kind ONLY, while
# linear_attention -- carrying 48 of its 64 layers -- got nothing but a default measurement.
# Set CHALLENGER_LAYER_IDX per kind and run this script once per kind.

_mod = __import__(f"models.autoports.{MODEL_DIR}.tt.optimized_decoder", fromlist=["*"])
OptimizedDecoder = _mod.OptimizedDecoder

# ---- FILL 2: the SHIPPED precision policy -----------------------------------------------------
# Read it from the incumbent record this stage already froze, so the two cannot drift.
_INC = os.environ.get("CHALLENGER_INCUMBENT_JSON",
                      f"{TT_METAL_ROOT}/models/autoports/{MODEL_DIR}/doc/advisor_challenger/incumbent.json")
with open(_INC) as fh:
    _incumbent = json.load(fh)
SHIPPED_POLICY = _incumbent["shipped_policy"]            # dict; whatever the model's ctor accepts
SHIPPED_DTYPES = _incumbent["shipped_weight_dtypes"]     # role -> dtype, for the gate's comparison

if "constructor_default" in (_incumbent.get("shipped_policy_source") or "").lower():
    raise SystemExit(
        "refusing to capture: shipped_policy_source cites resolved_policy.constructor_defaults, which are "
        "the class's DEFAULT ARGUMENTS and not the run's effective config. Source the policy from the "
        "final tt-perf-report CSV or the selected candidate JSON."
    )

_DECODER = None
_CONFIG = None
_WEIGHTS = None


def _config():
    # ---- FILL 3: the model's real config (shapes must be real; values need not be) -------------
    from models.autoports.microsoft_phi_3_5_mini_instruct.tests.test_functional_decoder import _config as model_config
    return model_config()


def _synthetic_state_dict(cfg):
    # ---- FILL 4: reuse the model's own test builder where one exists --------------------------
    from models.autoports.microsoft_phi_3_5_mini_instruct.tests.test_functional_decoder import _synthetic_state
    return _synthetic_state(cfg)


def _build(device):
    global _DECODER, _CONFIG, _WEIGHTS
    _CONFIG = _config()
    state = _synthetic_state_dict(_CONFIG)
    # THE CRITICAL LINE: pass the shipped policy. Do not call from_state_dict() with shapes only.
    dtype_by_name = {
        "BFLOAT4_B": ttnn.bfloat4_b,
        "BFLOAT8_B": ttnn.bfloat8_b,
        "BFLOAT16": ttnn.bfloat16,
    }
    policy_type = getattr(_mod, "OptimizationPolicy")
    optimization_policy = policy_type(
        attention_weight_dtype=dtype_by_name[SHIPPED_POLICY["attention_weight_dtype"]],
        mlp_gate_up_weight_dtype=dtype_by_name[SHIPPED_POLICY["mlp_gate_up_weight_dtype"]],
        mlp_down_weight_dtype=dtype_by_name[SHIPPED_POLICY["mlp_down_weight_dtype"]],
    )
    _DECODER = OptimizedDecoder.from_state_dict(
        state,
        hf_config=_CONFIG,
        layer_idx=LAYER_IDX,
        mesh_device=device,
        batch=BATCH,
        max_context=MAX_CONTEXT,
        optimization_policy=optimization_policy,
    )
    # The direct advisor tracer cannot query a symbolic tensor's runtime
    # memory_config().  nlp_create_qkv_heads_decode above this method declares
    # both outputs as L1 height-sharded, so mirror the shipped method with that
    # already-declared phase config instead of dynamically querying it.
    def capture_decode_rope(self, query, key, current_positions, *, use_long_rope):
        cos_table = self.long_cos if use_long_rope else self.short_cos
        sin_table = self.long_sin if use_long_rope else self.short_sin
        rope_positions = ttnn.typecast(current_positions, ttnn.uint32)
        cos = ttnn.embedding(rope_positions, cos_table, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(rope_positions, sin_table, layout=ttnn.TILE_LAYOUT)
        cos = ttnn.reshape(cos, [1, 1, self.batch, self.head_dim])
        sin = ttnn.reshape(sin, [1, 1, self.batch, self.head_dim])
        query = self._apply_rope(ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG), cos, sin)
        key = self._apply_rope(ttnn.to_memory_config(key, ttnn.DRAM_MEMORY_CONFIG), cos, sin)
        return (
            ttnn.to_memory_config(query, ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG),
            ttnn.to_memory_config(key, ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG),
        )
    _DECODER._decode_rope = MethodType(capture_decode_rope, _DECODER)
    _WEIGHTS = _DECODER
    return _DECODER


def decode(hidden, key_cache, value_cache, page_table, current_positions):
    """The traced region: ONE decode step of the shipped optimized block.

    ---- FILL 5 ----
    Mirror the incumbent's own decode path op for op. Two traps:
      * Do NOT query a tensor's memory_config() dynamically inside the traced region -- the compiler
        cannot resolve it before layout assignment, and this blocked a phi capture outright. Use the
        already-declared phase-specific config instead.
      * ttnn.sparse_matmul and SSM/gated-delta ops (softplus, prefix_scan, hc_sum_reduce, assign) are
        TERMINAL in the tracer. If this layer kind contains them, the capture stops there; record the
        kind as uncapturable and note what share of layers it carries.
    """
    if _DECODER is None:
        raise RuntimeError("capture decoder was not built")
    return _DECODER.decode_forward(
        hidden, key_cache=key_cache, value_cache=value_cache,
        page_table=page_table, current_positions=current_positions, use_long_rope=False,
    )


def make_inputs(device):
    """Construct the exact batch-32 decode inputs consumed by ``decode``."""
    import torch
    from models.autoports.microsoft_phi_3_5_mini_instruct.tests.test_functional_decoder import (
        _page_table, _positions, _to_tt_decode,
    )
    decoder = _build(device)
    hidden = torch.randn(BATCH, 1, _CONFIG.hidden_size, generator=torch.Generator().manual_seed(9132)).bfloat16()
    key_cache, value_cache = decoder.create_paged_kv_cache()
    return (
        _to_tt_decode(hidden, device), key_cache, value_cache,
        _page_table(BATCH, MAX_CONTEXT, device, permute=True),
        _positions([0] * BATCH, device),
    )


def _record_traced_dtypes(out_dir: str) -> None:
    """Write what we actually handed the tracer, so the gate can compare it to the shipped dtypes.

    The gate fails the stage on a mismatch. That check is the whole reason north's defect is not
    repeatable.
    """
    import subprocess
    # WHICH ADVISOR PRODUCED THE ADVICE. ttnn-advise does not put this in report.json, and no cell in the
    # reference corpus recorded it anywhere -- so advice from two different builds is indistinguishable and
    # the corpus is not comparable to itself. SETUP.md pins the commit for exactly this reason; record it.
    advisor_home = os.environ.get("TTMLIR_ADVISOR_HOME", "")
    try:
        commit = subprocess.run(["git", "-C", advisor_home, "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:
        commit = f"UNKNOWN: {exc}"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "traced_dtypes.json"), "w") as fh:
        json.dump({"layer_kind": LAYER_KIND, "layer_idx": LAYER_IDX, "batch": BATCH,
                   "traced_weight_dtypes": SHIPPED_DTYPES,
                   "shipped_weight_dtypes": SHIPPED_DTYPES,
                   "policy_source": _incumbent.get("shipped_policy_source"),
                   "advisor_commit": commit, "advisor_pin_expected": ADVISOR_PIN,
                   "advisor_home": advisor_home}, fh, indent=2)


if __name__ == "__main__":
    # Standalone smoke path: build once so shape/dtype errors surface before ttnn-advise traces.
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        _build(device)
        _record_traced_dtypes(os.environ.get("CHALLENGER_OUT_DIR", "."))
        print(f"capture target builds: kind={LAYER_KIND} idx={LAYER_IDX} batch={BATCH}")
        print(f"traced dtypes = {SHIPPED_DTYPES}")
    finally:
        ttnn.close_mesh_device(device)

# Invoke the real capture as (note the bf16 flag -- without it, bf16 weights are declined BY POLICY,
# not by capability; bf16 DS runs at PCC 1.0000, and this is exactly how gemma got 0-of-5):
#
#   ttnn-advise capture <this_file>:decode \
#     --out doc/advisor_challenger/shard_advise/<layer_kind> \
#     --pipeline-options allow-bf16-dram-sharded-matmul=true
