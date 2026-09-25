# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only: the functional reference matches HF ``Gemma4TextModel`` and is chunk-invariant."""

import pytest
import torch

from models.demos.gemma4_26b_d_p.reference.blocks import Gemma4TextModel
from models.demos.gemma4_26b_d_p.reference.config import DEFAULT_CKPT_DIR, Gemma4TextConfig
from models.demos.gemma4_26b_d_p.reference.weights import CheckpointReader, load_text_model

N_LAYERS = 6  # 5 sliding + 1 full
SEQ = 1152  # > sliding_window so the window mask is exercised


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@pytest.fixture(scope="module")
def reader():
    return CheckpointReader(DEFAULT_CKPT_DIR)


@pytest.fixture(scope="module")
def ref_model(reader):
    torch.manual_seed(0)
    cfg = Gemma4TextConfig.from_json().with_layers(N_LAYERS)
    return load_text_model(Gemma4TextModel(cfg), reader).eval()


@pytest.fixture(scope="module")
def ref_model_fp32(reader):
    cfg = Gemma4TextConfig.from_json().with_layers(N_LAYERS)
    return load_text_model(Gemma4TextModel(cfg), reader, dtype=torch.float32).eval()


@pytest.fixture(scope="module")
def ids():
    torch.manual_seed(0)
    return torch.randint(10, 200000, (1, SEQ))


@torch.no_grad()
def test_matches_hf(reader, ref_model, ids):
    from transformers import AutoConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel as HFText

    hf_cfg = AutoConfig.from_pretrained(DEFAULT_CKPT_DIR).text_config
    hf_cfg.num_hidden_layers = N_LAYERS
    hf_cfg.layer_types = hf_cfg.layer_types[:N_LAYERS]
    hf_cfg._attn_implementation = "eager"
    hf = HFText(hf_cfg).to(torch.bfloat16).eval()
    sd = {k: v for k, v in reader.substate("").items() if not k.startswith("layers.") or int(k.split(".")[1]) < N_LAYERS}
    missing, unexpected = hf.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    hf_out = hf(input_ids=ids, use_cache=False).last_hidden_state

    x = ref_model.prefill(ids, return_logits=False)[0]
    ours = ref_model.norm(x)
    p = pcc(ours, hf_out)
    print(f"PCC vs HF ({N_LAYERS} layers, S={SEQ}): {p:.6f}")
    assert p > 0.999


@torch.no_grad()
@pytest.mark.parametrize("chunk", [384, 576])
def test_chunk_invariant(ref_model_fp32, ids, chunk):
    """fp32 so the check is about logic, not bf16 rounding/top-k flips (bf16 gives ~0.9994 at layer 6)."""
    full, kv_full = ref_model_fp32.prefill(ids, return_logits=False)
    chunked, kv_chunked = ref_model_fp32.prefill(ids, chunk_size=chunk, return_logits=False)
    p = pcc(full, chunked)
    print(f"chunk={chunk}: hidden PCC {p:.7f}")
    assert p > 0.99999
    for i, ((kf, vf), (kc, vc)) in enumerate(zip(kv_full, kv_chunked)):
        assert pcc(kf, kc) > 0.99999 and pcc(vf, vc) > 0.99999, f"layer {i} KV mismatch"
