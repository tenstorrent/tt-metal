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

import hashlib
import json
import os
import shutil
import socket
import sys
from datetime import datetime, timezone
from types import MethodType, SimpleNamespace

import torch

import ttnn

# BUG FIX (v3): this defaulted to one developer's absolute path, so the template only worked on that host.
TT_METAL_ROOT = os.environ.get("TT_METAL_ROOT") or os.environ.get("TT_METAL_HOME") or os.getcwd()
# TWO PINS, because they answer different questions.
#
# ADVISOR_PIN is the toolchain this stage runs -- tt-mlir at the v3 tracer commit. OPTIMIZER_PIN is the
# commit the PLACEMENT logic must still be at, and the v3 branch is that commit plus three files under
# tools/ttnn-jit/ and nothing under lib/ or include/. Recording both is what lets a reader check the claim
# the whole v2-vs-v3 comparison rests on: that the optimizer did not move.
#
# Getting this wrong is not cosmetic. The gate fails a capture whose advisor_commit does not start with the
# expected pin, so a single stale pin string fails every cell of the run -- which is what this fixes.
ADVISOR_PIN = os.environ.get("CHALLENGER_ADVISOR_PIN", "97724a1170")
OPTIMIZER_PIN = os.environ.get("CHALLENGER_OPTIMIZER_PIN", "618cd4e75d")
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
    from transformers import AutoConfig
    from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import MODEL_ID

    cfg = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True).text_config
    assert len(cfg.layer_types) == cfg.num_hidden_layers
    assert cfg.layer_types[LAYER_IDX] == LAYER_KIND
    return cfg


def _synthetic_state_dict(cfg):
    # ---- FILL 4: reuse the model's own test builder where one exists --------------------------
    if LAYER_KIND == "full_attention":
        from models.autoports.qwen_qwen3_6_27b.tests.full_attention_synthetic_pcc import _state
    elif LAYER_KIND == "linear_attention":
        from models.autoports.qwen_qwen3_6_27b.tests.linear_attention_synthetic_pcc import _state
    else:
        raise AssertionError(LAYER_KIND)
    return _state(cfg)


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
        max_context=64,
        page_size=64,
        **SHIPPED_POLICY,
    )
    def _rms_norm_decode_without_dynamic_query(self, hidden_states, name):
        return ttnn.rms_norm(
            hidden_states,
            epsilon=self.eps,
            weight=self.weights[name],
            memory_config=self.decode_residual_memory_config,
            program_config=self.decode_norm_program_config,
            compute_kernel_config=self.norm_compute_kernel_config,
        )
    def _decode_linear_without_dynamic_query(
        self, hidden_states, weight_name, *, k, n, in0_block_w, fused_activation=None, compute_kernel_config=None
    ):
        storage_cores = self.policy.decode_storage_cores
        output_memcfg = _mod._l1_width_memory_config(rows=ttnn.TILE_SIZE, width=n, cores=storage_cores)
        return ttnn.linear(
            hidden_states,
            self.weights[weight_name],
            memory_config=output_memcfg,
            program_config=_mod._decode_program(
                k=k, n=n, in0_block_w=in0_block_w, cores=storage_cores, fused_activation=fused_activation
            ),
            compute_kernel_config=compute_kernel_config or (
                self.qkv_compute_kernel_config if weight_name.startswith("qkv") else
                self.o_compute_kernel_config if weight_name.startswith("o_proj") else
                self.mlp_compute_kernel_config
            ),
            dtype=ttnn.bfloat16,
        )
    _DECODER._rms_norm_decode = MethodType(_rms_norm_decode_without_dynamic_query, _DECODER)
    _DECODER._decode_linear = MethodType(_decode_linear_without_dynamic_query, _DECODER)
    _WEIGHTS = _DECODER
    return _DECODER


def make_inputs(device):
    """Concrete batch-32 decode input required by the advisor capture CLI."""
    decoder = _build(device)
    torch.manual_seed(20260810)
    hidden = (torch.randn(BATCH, 1, _CONFIG.hidden_size) * 0.2).bfloat16()
    hidden_tt = ttnn.from_torch(
        hidden.reshape(1, 1, BATCH, _CONFIG.hidden_size),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    page_table = ttnn.from_torch(
        torch.arange(BATCH, dtype=torch.int32).reshape(BATCH, 1),
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
    )
    positions = ttnn.from_torch(
        torch.zeros(BATCH, dtype=torch.uint32),
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
    )
    assert decoder.layer_kind == LAYER_KIND
    return (hidden_tt, page_table, positions)


def decode(hidden, page_table, positions):
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
    global _DECODER
    if _DECODER is None:
        _build(hidden.device())
    return _DECODER.decode_forward(
        hidden_states=hidden,
        page_table=page_table,
        current_positions=positions,
    )


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
    "ops_attempted": ["input_rms_norm", "token_mixer_decode", "residual_add", "post_rms_norm", "mlp", "residual_add"],
    "methods_substituted": {
        "_rms_norm_decode": "same rms_norm call with the dynamic memory_config() guard omitted; decode_forward already converts residual to decode_residual_memory_config",
        "_decode_linear": "same linear call with the dynamic input memory_config() guard omitted; traced producer layouts are assigned by the advisor"
    },    # e.g. {"_decode_rope": "DRAM-staging stand-in; tracer cannot resolve
                                  #        memory_config() before layout assignment"}
    "env_knobs": {"SHARD_ADVISE_BATCH": BATCH, "CHALLENGER_LAYER_KIND": LAYER_KIND, "CHALLENGER_LAYER_IDX": LAYER_IDX},
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

    def _git(*args):
        try:
            return subprocess.run(["git", "-C", advisor_home, *args],
                                  capture_output=True, text=True, check=True).stdout.strip()
        except Exception as exc:
            return f"UNKNOWN: {exc}"

    commit = _git("rev-parse", "HEAD")
    # DID THE PLACEMENT LOGIC MOVE? The tracer commits are Python under tools/ttnn-jit/; anything under
    # lib/ or include/ would be a different optimizer and a different experiment.
    changed_optimizer = _git("diff", "--name-only", f"{OPTIMIZER_PIN}..HEAD", "--", "lib", "include")
    # HOST FINGERPRINT. A contaminated v2 cell produced documentation citing a TTMLIR_ADVISOR_HOME that
    # does not exist on the machine that wrote it, with plausible hashes and op counts -- fabricated
    # provenance for a command that could not have run, and only byte-identity comparison caught it. The
    # gate verifies the tool exists at this path ON THIS HOST, which turns that into a hard failure.
    tool = shutil.which("ttnn-advise") or os.path.join(advisor_home, "build/bin/ttnn-advise")

    def _sha(path):
        try:
            with open(path, "rb") as fh:
                return hashlib.sha256(fh.read()).hexdigest()
        except Exception as exc:
            return f"UNKNOWN: {exc}"

    tool_sha = _sha(os.path.realpath(tool))
    # GIT PROVENANCE IS NOT TOOLCHAIN PROVENANCE, and this stage depends on the difference.
    #
    # advisor_commit above describes the CHECKOUT. It does not describe the code that runs: `ttnn_jit` is
    # installed into the toolchain venv as a plain directory, not as an editable install, so `ttnn-advise`
    # imports site-packages and a `git checkout` of another branch would change advisor_commit while the
    # module that traces stays exactly as it was. There is also a build/lib.../ttnn_jit copy inside the
    # checkout that predates the current tracer.
    #
    # The tracer is what decides whether a layer is visible to the advisor at all -- it is the difference
    # between a real zero and a coverage zero -- so record the file that will actually be imported and
    # compare it with the checkout. A mismatch means the advice describes a graph a different tracer saw.
    try:
        import ttnn_jit._src.ttnn_emit_tracer as _tracer
        tracer_path = _tracer.__file__
    except Exception as exc:
        tracer_path = f"UNIMPORTABLE: {exc}"
    tracer_sha = _sha(tracer_path) if not tracer_path.startswith("UNIMPORTABLE") else tracer_path
    checkout_tracer = os.path.join(advisor_home, "tools/ttnn-jit/_src/ttnn_emit_tracer.py")
    checkout_sha = _sha(checkout_tracer)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "traced_dtypes.json"), "w") as fh:
        json.dump({"layer_kind": LAYER_KIND, "layer_idx": LAYER_IDX, "batch": BATCH,
                   "traced_weight_dtypes": SHIPPED_DTYPES,
                   "shipped_weight_dtypes": SHIPPED_DTYPES,
                   "policy_source": _incumbent.get("shipped_policy_source"),
                   "advisor_commit": commit, "advisor_pin_expected": ADVISOR_PIN,
                   "advisor_home": advisor_home,
                   "optimizer_pin_expected": OPTIMIZER_PIN,
                   "optimizer_files_changed_since_pin": (
                       [f for f in changed_optimizer.splitlines() if f]
                       if not changed_optimizer.startswith("UNKNOWN") else changed_optimizer),
                   "host": socket.gethostname(),
                   "tool_path": tool, "tool_realpath": os.path.realpath(tool), "tool_sha256": tool_sha,
                   "tracer_imported_from": tracer_path, "tracer_sha256": tracer_sha,
                   "tracer_checkout_path": checkout_tracer, "tracer_checkout_sha256": checkout_sha,
                   "tracer_matches_checkout": (tracer_sha == checkout_sha
                                               and not str(tracer_sha).startswith(("UNKNOWN", "UNIMPORT"))),
                   "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
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
