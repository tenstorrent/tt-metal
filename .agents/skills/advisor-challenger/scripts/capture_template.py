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

import torch

import ttnn

# BUG FIX (v3): this defaulted to one developer's absolute path, so the template only worked on that host.
TT_METAL_ROOT = os.environ.get("TT_METAL_ROOT") or os.environ.get("TT_METAL_HOME") or os.getcwd()
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
    raise NotImplementedError("fill in the model's hf_config-shaped SimpleNamespace")


def _synthetic_state_dict(cfg):
    # ---- FILL 4: reuse the model's own test builder where one exists --------------------------
    raise NotImplementedError("build or import the synthetic state dict")


def _build(device):
    global _DECODER, _CONFIG, _WEIGHTS
    _CONFIG = _config()
    state = _synthetic_state_dict(_CONFIG)
    # THE CRITICAL LINE: pass the shipped policy. Do not call from_state_dict() with shapes only.
    _DECODER = OptimizedDecoder.from_state_dict(
        state,
        hf_config=_CONFIG,
        layer_idx=LAYER_IDX,
        mesh_device=device,
        batch=BATCH,
        **SHIPPED_POLICY,
    )
    _WEIGHTS = _DECODER
    return _DECODER


def decode(hidden):
    """The traced region: ONE decode step of the shipped optimized block.

    ---- FILL 5 ----
    Mirror the incumbent's own decode path op for op. Two traps:
      * Do NOT query a tensor's memory_config() dynamically inside the traced region -- the compiler
        cannot resolve it before layout assignment, and this blocked a phi capture outright. Use the
        already-declared phase-specific config instead.
      * Do NOT assume an op is terminal in the tracer. `sparse_matmul`, mutable-state `ttnn.copy`,
        `paged_fused_update_cache`, `ones_like`, `pow` and `repeat_interleave` all have handlers at this
        pin, and of the ten ops the reference corpus recorded as blockers four already had one and one did
        not exist at all. ATTEMPT the trace, read the traceback frames (they name which of ttnn-jit's two
        tracers is running), and only then record a kind as uncapturable -- with its share of layers, and
        in `CAPTURE_SCOPE["stopped_at"]`.
    """
    raise NotImplementedError("mirror the incumbent's decode path")


# ---- FILL 6: what this capture actually attempted --------------------------------------------
# Fifteen capture scripts were written for the reference corpus, 54 to 290 lines, and nothing compared them:
# five stopped at the same terminal op in four different places, from 30 ops captured down to 5, and two
# invented private env knobs whose values were recorded nowhere. So a cross-cell coverage number silently
# mixed captures that attempted very different amounts of the layer.
#
# SUBSTITUTIONS ARE THE IMPORTANT ONE. If you replace a model method before tracing -- four cells replaced
# `_decode_rope`, one replaced three more -- then the advice for that region is advice for YOUR STAND-IN, and
# a change to the real method cannot reach the capture at all. Name every one.
CAPTURE_SCOPE = {
    "ops_attempted": [],          # e.g. ["rms_norm", "linear", "rope", "sdpa", "mlp"]
    "methods_substituted": {},    # e.g. {"_decode_rope": "DRAM-staging stand-in; tracer cannot resolve
                                  #        memory_config() before layout assignment"}
    "env_knobs": {},              # every private knob this capture reads, with its VALUE
    "stopped_at": None,           # the op the trace terminated on, if any
}


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
                   "advisor_home": advisor_home,
                   "capture_scope": CAPTURE_SCOPE}, fh, indent=2)


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
