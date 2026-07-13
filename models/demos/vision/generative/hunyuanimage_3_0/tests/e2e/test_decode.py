# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Decode-path test for HunyuanImage-3.0: real incremental-KV autoregressive
decode (KV cache + causal single-token attention).

  * CORRECTNESS — greedy-decode N tokens with the TTNN incremental-KV loop and
    compare token-for-token against a CAUSAL HF reference (growing-prefix causal
    forward = the ground truth for cached decode), at the SAME layer count and
    bf16. Decode tokens are confident, so exact/near-exact token agreement is the
    valid functional bar (unlike accumulated prefill positions).
  * PERF — median per-step transformer decode time -> per-user decode t/s/u
    (batch=1), the metric directly comparable to Llama-70B-galaxy's 71 t/s/u.

Run:  HUNYUAN_E2E_NUM_LAYERS=32 ./python_env/bin/python -m pytest -o timeout=0 \
        models/demos/vision/generative/hunyuanimage_3_0/tests/e2e/test_decode.py -s
"""
from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.demos.vision.generative.hunyuanimage_3_0.tt import pipeline as pl

try:
    _MESH = tuple(int(x) for x in ttnn._ttnn.multi_device.SystemMeshDescriptor().shape())
except Exception:
    _MESH = (1, 8)

PROMPT = "A serene mountain lake at sunrise, photorealistic."
NUM_LAYERS = int(os.environ.get("HUNYUAN_E2E_NUM_LAYERS", "32"))
SEQ_LEN = int(os.environ.get("HUNYUAN_E2E_SEQ_LEN", "128"))
N_NEW = int(os.environ.get("HUNYUAN_DECODE_NEW", "16"))
# min token agreement (longest-common-prefix fraction) vs the causal HF reference.
MIN_MATCH_FRAC = float(os.environ.get("HUNYUAN_DECODE_MIN_MATCH", "0.9"))

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = pl.load_reference_model()
    return _MODEL


def _hf_causal_greedy(model, prompt_ids, num_layers, n_new, head_dim):
    """Ground truth: bf16 causal growing-prefix greedy decode."""
    seq = list(prompt_ids)
    gen = []
    with torch.no_grad():
        for _ in range(n_new):
            S = len(seq)
            hid = model.model.wte(torch.tensor([seq], dtype=torch.long)).to(torch.bfloat16)
            cos, sin = pl.build_2d_rope_text(model, S, head_dim)
            pos = (cos.to(torch.bfloat16), sin.to(torch.bfloat16))
            mask = torch.triu(torch.full((S, S), float("-inf")), diagonal=1)[None, None].to(torch.bfloat16)
            for i in range(num_layers):
                hid = model.model.layers[i].to(torch.bfloat16)(
                    hid,
                    attention_mask=mask,
                    position_ids=None,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                    custom_pos_emb=pos,
                )[0]
            logits = model.lm_head(model.model.ln_f(hid.to(torch.bfloat16)))
            nxt = int(logits[0, -1].argmax().item())
            gen.append(nxt)
            seq.append(nxt)
    return gen


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [_MESH], indirect=True)
def test_decode(device_params, mesh_device):
    torch.manual_seed(0)
    model = _get_model()
    pipe = pl.build_pipeline(mesh_device, model, num_layers=NUM_LAYERS, seq_len=SEQ_LEN)

    out = pipe.run_decode(PROMPT, n_new_tokens=N_NEW)
    tt_tokens = out["generated"]
    step_times = out["step_times"]
    hf_tokens = _hf_causal_greedy(model, out["prompt_ids"], NUM_LAYERS, N_NEW, pipe.head_dim)

    lcp = 0
    for a, b in zip(tt_tokens, hf_tokens):
        if a == b:
            lcp += 1
        else:
            break
    match_frac = lcp / max(len(hf_tokens), 1)

    import statistics

    med = statistics.median(step_times) if step_times else float("nan")
    tsu = (1.0 / med) if step_times else float("nan")

    print("\n================ HunyuanImage-3.0 decode ================")
    print(f"config: num_layers={NUM_LAYERS} seq_len={SEQ_LEN} n_new={N_NEW} (batch=1)")
    print(f"[correctness] TT tokens: {tt_tokens}")
    print(f"[correctness] HF tokens: {hf_tokens}")
    print(f"[correctness] longest-common-prefix = {lcp}/{N_NEW} ({match_frac:.0%})  (>= {MIN_MATCH_FRAC:.0%})")
    print(f"[perf] median step = {med*1000:.1f} ms -> transformer decode t/s/u = {tsu:.2f}")
    print("         (host lm_head excluded; on-device head is the perf-phase optimization)")
    print("=========================================================")

    assert match_frac >= MIN_MATCH_FRAC, (
        f"decode correctness FAIL — only {lcp}/{N_NEW} tokens match the causal HF reference "
        f"(TT={tt_tokens} HF={hf_tokens})"
    )
