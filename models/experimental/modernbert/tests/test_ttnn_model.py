# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The full 22-layer TTNN encoder vs the HuggingFace reference."""

import pytest
import torch

import ttnn
from models.experimental.modernbert.common import (
    SAMPLE_TEXT,
    build_inputs,
    load_config,
    load_tokenizer,
    load_torch_model,
)
from models.experimental.modernbert.reference.modernbert import ModernBertModel
from models.experimental.modernbert.tests.pcc_utils import outlier_report, pcc
from models.experimental.modernbert.tt.modernbert_model import TtnnModernBertModel
from models.experimental.modernbert.tt.weights import prepare_weights

MODEL_PCC = 0.99


@pytest.fixture(scope="module")
def torch_ref():
    config = load_config()
    hf = load_torch_model()
    ref = ModernBertModel(config)
    ref.load_state_dict(hf.state_dict(), strict=True)
    ref.eval()
    return config, hf, ref


def _run_ttnn(config, ref, device, ids, attention_mask, seq_len):
    params = prepare_weights(ref, device)
    # masks and rotary caches are built once, at construction, since they are
    # fixed for a given sequence length and shared across all 22 layers
    model = TtnnModernBertModel(params, config, device, seq_len, attention_mask=attention_mask)
    tt_ids = ttnn.from_torch(ids.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out = model(tt_ids)
    return ttnn.to_torch(out)


@pytest.mark.parametrize("seq_len", [256, 512])
def test_ttnn_model_matches_hf(device, torch_ref, seq_len):
    config, hf, ref = torch_ref
    ids, attention_mask = build_inputs(seq_len=seq_len)

    with torch.no_grad():
        expected_fp32 = hf(input_ids=ids, attention_mask=attention_mask).last_hidden_state

    got = _run_ttnn(config, ref, device, ids, attention_mask, seq_len).reshape(expected_fp32.shape)

    p = pcc(expected_fp32, got.float())
    print(f"\n[model seq={seq_len}] TTNN vs HF fp32 PCC={p:.8f}")
    print(f"  reference outliers: {outlier_report(expected_fp32)}")
    assert p >= MODEL_PCC, f"model PCC {p:.8f} < {MODEL_PCC}"


def test_ttnn_model_vs_bf16_reference(device, torch_ref):
    """Isolates TTNN error from the bf16 penalty."""
    config, _, ref = torch_ref
    seq_len = 256
    ids, attention_mask = build_inputs(seq_len=seq_len)

    hf_bf16 = load_torch_model(dtype=torch.bfloat16, attn_implementation="eager")
    with torch.no_grad():
        expected = hf_bf16(input_ids=ids, attention_mask=attention_mask).last_hidden_state

    got = _run_ttnn(config, ref, device, ids, attention_mask, seq_len).reshape(expected.shape)

    p = pcc(expected.float(), got.float())
    print(f"\n[model seq={seq_len}] TTNN vs HF bf16-eager PCC={p:.8f}")
    assert p >= MODEL_PCC, f"model vs bf16 reference PCC {p:.8f} < {MODEL_PCC}"


@pytest.mark.parametrize("batch_size", [2, 4])
def test_ttnn_model_batched(device, torch_ref, batch_size):
    """Batch > 1 through the full encoder."""
    config, hf, ref = torch_ref
    seq_len = 256
    ids, attention_mask = build_inputs(seq_len=seq_len, batch_size=batch_size)
    assert ids.shape[0] == batch_size

    with torch.no_grad():
        expected = hf(input_ids=ids, attention_mask=attention_mask).last_hidden_state

    got = _run_ttnn(config, ref, device, ids, attention_mask, seq_len).reshape(expected.shape)
    p = pcc(expected, got.float())
    print(f"\n[model batch={batch_size} seq={seq_len}] PCC={p:.8f}")
    assert p >= MODEL_PCC, f"batched model PCC {p:.8f} < {MODEL_PCC}"


def test_ttnn_model_with_padding(device, torch_ref):
    """Padded batch through the full encoder."""
    config, hf, ref = torch_ref
    seq_len, real_len = 256, 200
    tok = load_tokenizer()
    full = tok(SAMPLE_TEXT, return_tensors="pt")["input_ids"]

    ids = torch.full((1, seq_len), config.pad_token_id, dtype=torch.long)
    ids[:, :real_len] = full[:, :real_len]
    attention_mask = torch.zeros((1, seq_len), dtype=torch.long)
    attention_mask[:, :real_len] = 1
    assert not torch.all(attention_mask == 1), "padding branch would not be exercised"

    with torch.no_grad():
        expected = hf(input_ids=ids, attention_mask=attention_mask).last_hidden_state

    got = _run_ttnn(config, ref, device, ids, attention_mask, seq_len).reshape(expected.shape)
    p = pcc(expected[:, :real_len], got.float()[:, :real_len])
    print(f"\n[model padded {real_len}/{seq_len}] unpadded-region PCC={p:.8f}")
    assert p >= MODEL_PCC, f"padded model PCC {p:.8f} < {MODEL_PCC}"


def test_ttnn_model_longer_sequence(device, torch_ref):
    """seq 768 - more of the sequence lies outside the 129-wide local band."""
    config, hf, ref = torch_ref
    seq_len = 768
    ids, attention_mask = build_inputs(seq_len=seq_len)
    with torch.no_grad():
        expected = hf(input_ids=ids, attention_mask=attention_mask).last_hidden_state
    got = _run_ttnn(config, ref, device, ids, attention_mask, seq_len).reshape(expected.shape)
    p = pcc(expected, got.float())
    print(f"\n[model seq={seq_len}] PCC={p:.8f}")
    assert p >= MODEL_PCC, f"model PCC {p:.8f} < {MODEL_PCC}"


def test_layer_type_pattern_is_read_not_recomputed(torch_ref):
    """The global/sliding pattern comes from config.layer_types so it cannot drift."""
    config, _, _ = torch_ref
    expected = [
        "full_attention" if i % config.global_attn_every_n_layers == 0 else "sliding_attention"
        for i in range(config.num_hidden_layers)
    ]
    assert list(config.layer_types) == expected
    assert config.layer_types[0] == "full_attention"
    assert config.layer_types[1] == "sliding_attention"
    assert len(config.layer_types) == 22
