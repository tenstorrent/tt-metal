# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M1 (host only): the reference against the REAL checkpoint — ground truth, not self-consistency.

The golden trace is the output of *upstream* ``transformers`` Qwen3_5TextModel (fp32 compute) on the
real checkpoint. Because prefill is causal, the reference run on the trace's first N tokens must
reproduce the trace's final hidden states and attention K/V for positions [0, N). All 64 layers,
full width, real weights, streamed one layer at a time (fp32, like the trace).
(pattern: minimax_m3/tests/golden_hf_first_token.py)
"""

import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from models.demos.qwen_3_8_27b.config import QWEN38
from models.demos.qwen_3_8_27b.reference import golden
from models.demos.qwen_3_8_27b.reference.checkpoint import CheckpointReader
from models.demos.qwen_3_8_27b.tests.torch.test_reference_model import pcc

N = 128


@pytest.fixture(scope="module")
def trace_dir():
    d = os.environ.get("PREFILL_TRACE_DIR")
    ck = os.environ.get("PREFILL_HF_MODEL") or os.environ.get("HF_MODEL")
    if not d or not ck or not Path(d, "metadata.json").exists():
        pytest.skip("needs PREFILL_TRACE_DIR and PREFILL_HF_MODEL")
    return Path(d)


def test_reference_matches_upstream_on_real_checkpoint(trace_dir):
    ids = golden.trace_token_ids(trace_dir)[:N][None]
    h, states = golden.stream_forward(QWEN38, ids, reader=CheckpointReader(), dtype=torch.float32)
    with safe_open(str(trace_dir / "final_hidden.safetensors"), "pt") as f:
        want_h = f.get_slice("final_hidden")[:, :N].float()
    p = pcc(h, want_h)
    print(f"final_hidden[:{N}] PCC vs upstream trace = {p:.6f}")
    assert p > 0.999
    for L in QWEN38.full_attention_layers[::5]:  # 3, 23, 43, 63
        with safe_open(str(trace_dir / f"kv_cache/layer_{L}.safetensors"), "pt") as f:
            wk = f.get_slice(f"key_cache_layer_{L}")[:, :, :N].float()
            wv = f.get_slice(f"value_cache_layer_{L}")[:, :, :N].float()
        pk, pv = pcc(states[L]["k"], wk), pcc(states[L]["v"], wv)
        print(f"layer {L} K/V[:{N}] PCC vs upstream trace = {pk:.6f} / {pv:.6f}")
        assert pk > 0.999 and pv > 0.999
