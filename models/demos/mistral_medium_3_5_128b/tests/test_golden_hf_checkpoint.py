# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host only, real checkpoint (``PREFILL_HF_MODEL`` / ``HF_MODEL``): the torch reference against ground
truth for the whole model.

* sampled decoder layers, the embedding, the final norm and the lm_head on the real fp8-dequantized
  weights against the upstream HF Ministral3 modules;
* the full 88-layer reference forward over a prefix of the golden trace's prompt reproduces every
  layer's K/V in the trace (``PREFILL_TRACE_DIR``) — K/V of positions [0, n) depend only on tokens
  [0, n), so the prefix of a 10240-token trace is an exact target.
"""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from safetensors import safe_open
from transformers.models.ministral3.modeling_ministral3 import (
    Ministral3DecoderLayer,
    Ministral3RMSNorm,
    Ministral3RotaryEmbedding,
)

from models.common.utility_functions import comp_pcc
from models.demos.mistral_medium_3_5_128b.config import MistralMediumConfig
from models.demos.mistral_medium_3_5_128b.reference.checkpoint import CheckpointReader
from models.demos.mistral_medium_3_5_128b.reference.model import ReferenceDecoderLayer, rms_norm, rope_cos_sin

from .test_config import hf_text_config

CKPT = os.environ.get("PREFILL_HF_MODEL") or os.environ.get("HF_MODEL")
TRACE = os.environ.get("PREFILL_TRACE_DIR")
requires_checkpoint = pytest.mark.skipif(
    not (CKPT and (Path(CKPT) / "model.safetensors.index.json").is_file()),
    reason="needs the real checkpoint in PREFILL_HF_MODEL / HF_MODEL",
)
requires_trace = pytest.mark.skipif(
    not (TRACE and (Path(TRACE) / "metadata.json").is_file()), reason="needs PREFILL_TRACE_DIR"
)
CFG = MistralMediumConfig.from_json()


def trace_tokens(n):
    with open(Path(TRACE) / "metadata.json") as f:
        return torch.tensor(json.load(f)["token_ids"][:n])


def _on_meta(module_cls, *args, state):
    with torch.device("meta"):
        module = module_cls(*args)
    module.load_state_dict(state, assign=True)
    return module.eval()


@requires_checkpoint
@requires_trace
@pytest.mark.timeout(1800)
@torch.no_grad()
def test_reference_matches_hf_modules_on_real_weights():
    reader = CheckpointReader(CKPT)
    hf_cfg = hf_text_config(CFG)
    embed = reader.embedding()
    tokens = trace_tokens(512)
    x = embed[tokens][None]
    assert torch.equal(x, torch.nn.functional.embedding(tokens, embed)[None])

    pos = torch.arange(x.shape[1])
    cos, sin = rope_cos_sin(CFG, pos)
    cos_hf, sin_hf = Ministral3RotaryEmbedding(hf_cfg)(x, pos[None])
    for i, sd in reader.iter_layers([0, 43, 87]):
        ref = _on_meta(ReferenceDecoderLayer, CFG, state=sd)
        hf = _on_meta(Ministral3DecoderLayer, hf_cfg, i, state=sd)
        out_ref, _, _ = ref(x, cos, sin, pos)
        out_hf = hf(x, attention_mask=None, position_ids=pos[None], position_embeddings=(cos_hf, sin_hf))
        out_hf = out_hf[0] if isinstance(out_hf, tuple) else out_hf
        passing, pcc = comp_pcc(out_hf.float(), out_ref.float(), 0.9999)
        logger.info(f"real-weight layer {i}: reference vs HF {pcc}")
        assert passing, f"layer {i}: reference vs HF {pcc}"

    norm_w, lm_head = reader.final_norm(), reader.lm_head()
    hf_norm = Ministral3RMSNorm(CFG.hidden_size, eps=CFG.rms_norm_eps).to(torch.bfloat16)
    hf_norm.weight.copy_(norm_w)
    h = x * 4.0
    assert torch.equal(hf_norm(h), rms_norm(h, norm_w, CFG.rms_norm_eps))
    logits = rms_norm(h, norm_w, CFG.rms_norm_eps) @ lm_head.t()
    assert torch.equal(logits, torch.nn.functional.linear(hf_norm(h), lm_head))
    assert lm_head.shape == (CFG.vocab_size, CFG.hidden_size) and embed.shape == (CFG.vocab_size, CFG.hidden_size)


@requires_checkpoint
@requires_trace
@pytest.mark.timeout(3600)
@torch.no_grad()
def test_full_depth_reference_reproduces_golden_trace_prefix():
    n = 256
    reader = CheckpointReader(CKPT)
    tokens = trace_tokens(n)
    h = reader.embedding()[tokens][None]
    pos = torch.arange(n)
    cos, sin = rope_cos_sin(CFG, pos)
    worst = {"k": 1.0, "v": 1.0}
    for i, sd in reader.iter_layers(range(CFG.num_hidden_layers)):
        layer = _on_meta(ReferenceDecoderLayer, CFG, state=sd)
        h, k, v = layer(h, cos, sin, pos)
        with safe_open(str(Path(TRACE) / "kv_cache" / f"layer_{i}.safetensors"), framework="pt") as f:
            gk = f.get_slice(f"key_cache_layer_{i}")[:, :, :n, :]
            gv = f.get_slice(f"value_cache_layer_{i}")[:, :, :n, :]
        for name, got, want in (("k", k, gk), ("v", v, gv)):
            _, pcc = comp_pcc(want.float(), got.float(), 0.0)
            worst[name] = min(worst[name], pcc)
            assert pcc >= 0.999, f"layer {i} {name}: reference vs golden trace prefix {pcc}"
        del layer, sd
    logger.info(f"full-depth reference vs golden prefix ({n} tokens): worst K {worst['k']:.6f} V {worst['v']:.6f}")
