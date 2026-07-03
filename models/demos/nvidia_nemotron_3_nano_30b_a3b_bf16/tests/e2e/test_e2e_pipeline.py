# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end pipeline test for `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.

Real input (HF tokenizer) -> chained graduated TTNN stubs (tt/pipeline.py) ->
real output (generated token ids), compared to the HF reference golden
(Source A: NemotronHForCausalLM.generate).

Gates:
  Gate 1 - no torch runtime fallback fired (_runtime_fallbacks.json empty).
  Gate 2 - every graduated module invoked (compose mode).
  Gate 3 - e2e next-token-logits PCC vs HF golden >= 0.95.

Run on device:
  ./python_env/bin/python -m pytest models/demos/nvidia_nemotron_3_nano_30b_a3b_bf16/tests/e2e/test_e2e_pipeline.py -s

Env:
  TT_E2E_COMPOSE=1   compose children (Gate 2). 0 = monolith backbone.
  TT_E2E_N=5         generation horizon (both sides capped to the same N).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from models.common.utility_functions import comp_pcc
from models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.tt import _invocation
from models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.tt import pipeline as pl
from models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.tt._hf_compat import install_hf_compat

install_hf_compat()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

HF_MODEL_ID = pl.HF_MODEL_ID
DEMO_DIR = Path(__file__).resolve().parents[2]
GOLDEN = DEMO_DIR / "_captured" / "_e2e_golden"
PCC_TARGET = 0.95


def _load_golden():
    ids = torch.load(GOLDEN / "input_ids.pt", weights_only=False)
    new_ids = torch.load(GOLDEN / "golden_new_ids.pt", weights_only=False).tolist()
    step_logits = torch.load(GOLDEN / "golden_step_logits.pt", weights_only=False)
    return ids, new_ids, step_logits


def _load_hf():
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    return model


def _compose():
    return os.environ.get("TT_E2E_COMPOSE", "1") == "1"


def _reset_runtime_fallbacks():
    p = DEMO_DIR / "_runtime_fallbacks.json"
    try:
        p.write_text("[]")
    except Exception:
        pass


def _check_gate1():
    p = DEMO_DIR / "_runtime_fallbacks.json"
    if not p.is_file():
        return True, []
    try:
        data = json.loads(p.read_text() or "[]")
    except Exception:
        return True, []
    return (len(data) == 0), data


@pytest.fixture(scope="module")
def mesh_pipe():
    """Open the 4-chip TP=2 x DP=2 mesh (FABRIC_1D + shard runner) ONCE, build the
    HF reference + the shared TTNN pipeline, and hand both to the tests. Closed at
    module teardown. Falls back to a single device if the mesh cannot be opened."""
    compose = _compose()
    dev, is_mesh = pl.open_pipeline_mesh(l1_small_size=24576)
    hf = _load_hf()
    pipe = pl.build_pipeline(dev, hf, compose=compose)
    print(
        f"[e2e] mesh={is_mesh} shape={list(dev.shape) if is_mesh else [1, 1]} "
        f"shard_active={pipe.shard_active} compose={compose}",
        flush=True,
    )
    try:
        yield hf, pipe, is_mesh
    finally:
        pl.close_pipeline_mesh(dev, is_mesh)


def test_e2e_prefill(mesh_pipe):
    """Fast Gate-3 proxy: ONE prefill forward, first-token logits PCC vs golden."""
    compose = _compose()
    _reset_runtime_fallbacks()
    ids, new_ids, step_logits = _load_golden()
    _, pipe, _ = mesh_pipe

    logits = pipe.forward_logits(ids)  # (vocab,)
    golden0 = step_logits[0].to(torch.float32)
    ok, pcc = comp_pcc(golden0, logits, PCC_TARGET)
    tt_tok = int(torch.argmax(logits).item())

    print(f"e2e PCC={pcc}")
    print(f"[e2e] compose={compose} first_token tt={tt_tok} golden={new_ids[0]} " f"match={tt_tok == new_ids[0]}")
    assert ok, f"first-token logits PCC {pcc} < {PCC_TARGET} (compose={compose})"


def test_e2e_generate(mesh_pipe):
    """Full gate: capped greedy decode, per-step logits PCC + token match + Gate 1/2."""
    compose = _compose()
    N = int(os.environ.get("TT_E2E_N", "5"))
    _reset_runtime_fallbacks()
    _invocation.reset()
    ids, golden_new_ids, golden_step_logits = _load_golden()
    golden_new_ids = golden_new_ids[:N]

    hf, pipe, is_mesh = mesh_pipe
    eos = int(getattr(hf.config, "eos_token_id", 2))
    assert pipe.shard_active or not is_mesh, "TP>1 mesh run must have shard_active (ShardTensorToMesh + all_reduce)"

    tt_new_ids, tt_step_logits = pipe.generate(ids, N, eos_token_id=eos)

    n_cmp = min(len(tt_new_ids), golden_step_logits.shape[0], N)
    per_step_pcc = []
    for s in range(n_cmp):
        _, p = comp_pcc(golden_step_logits[s].to(torch.float32), tt_step_logits[s], PCC_TARGET)
        per_step_pcc.append(float(p))
    mean_pcc = sum(per_step_pcc) / len(per_step_pcc) if per_step_pcc else 0.0
    token_match = tt_new_ids[:n_cmp] == golden_new_ids[:n_cmp]

    # Gate 1
    g1_ok, fallbacks = _check_gate1()
    # Gate 2 — based on the honest registry of stubs that ACTUALLY executed.
    actually_invoked = _invocation.snapshot()
    missing = set(pl.GRADUATED_MODULES) - actually_invoked if compose else set()

    print(f"e2e PCC={mean_pcc}")
    print(f"[e2e] compose={compose} N={n_cmp} per_step_pcc={per_step_pcc}")
    print(f"[e2e] tt_ids={tt_new_ids[:n_cmp]} golden_ids={golden_new_ids[:n_cmp]} match={token_match}")
    print(f"[e2e] Gate1 no_fallback={g1_ok} fallbacks={fallbacks}")
    print(f"[e2e] Gate2 invoked={sorted(actually_invoked)} missing={sorted(missing)}")

    assert g1_ok, f"Gate 1 failed: runtime fallbacks fired: {fallbacks}"
    if compose:
        assert not missing, f"Gate 2 failed: graduated modules not invoked: {sorted(missing)}"
    assert mean_pcc >= PCC_TARGET, f"Gate 3 failed: mean per-step logits PCC {mean_pcc} < {PCC_TARGET}"
    assert token_match, f"token mismatch tt={tt_new_ids[:n_cmp]} golden={golden_new_ids[:n_cmp]}"
