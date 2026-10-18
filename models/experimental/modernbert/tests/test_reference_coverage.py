# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Reference coverage for paths test_reference_parity does not reach."""

import pytest
import torch

from models.experimental.modernbert.common import (
    SAMPLE_TEXT,
    build_inputs,
    load_config,
    load_tokenizer,
    load_torch_model,
)
from models.experimental.modernbert.reference.modernbert import ModernBertModel
from models.experimental.modernbert.tests.pcc_utils import pcc

PARITY_PCC = 0.9999


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def hf_model():
    return load_torch_model()


def make_ref(config, hf_model, dtype=torch.float32):
    ref = ModernBertModel(config)
    ref.load_state_dict(hf_model.state_dict(), strict=True)
    ref.eval()
    return ref.to(dtype)


@pytest.mark.parametrize("batch_size", [2, 4])
def test_batch_greater_than_one(config, hf_model, batch_size):
    ref = make_ref(config, hf_model)
    ids, mask = build_inputs(seq_len=256, batch_size=batch_size)
    assert ids.shape[0] == batch_size

    with torch.no_grad():
        hf_out = hf_model(input_ids=ids, attention_mask=mask).last_hidden_state
        ref_out = ref(ids, mask)

    p = pcc(hf_out, ref_out)
    print(f"\n[batch={batch_size}] PCC={p:.10f}")
    assert p >= PARITY_PCC


def test_padding_mask_path(config, hf_model):
    """Exercises the padding branch of build_masks against HF."""
    cfg = config
    tok = load_tokenizer()
    full = tok(SAMPLE_TEXT, return_tensors="pt")["input_ids"]

    seq_len, real_len = 256, 200
    ids = torch.full((1, seq_len), cfg.pad_token_id, dtype=torch.long)
    ids[:, :real_len] = full[:, :real_len]
    mask = torch.zeros((1, seq_len), dtype=torch.long)
    mask[:, :real_len] = 1

    assert not torch.all(mask == 1), "padding branch would not be exercised"

    ref = make_ref(cfg, hf_model)
    with torch.no_grad():
        hf_out = hf_model(input_ids=ids, attention_mask=mask).last_hidden_state
        ref_out = ref(ids, mask)

    p_real = pcc(hf_out[:, :real_len], ref_out[:, :real_len])
    print(f"\n[padding] unpadded-region PCC={p_real:.10f} (real_len={real_len}/{seq_len})")
    assert p_real >= PARITY_PCC


def test_padding_actually_changes_output(config, hf_model):
    """Negative control: if masking padded positions had no effect the padding
    branch would be dead code and the test above would prove nothing."""
    cfg = config
    tok = load_tokenizer()
    full = tok(SAMPLE_TEXT, return_tensors="pt")["input_ids"]

    seq_len, real_len = 256, 200
    ids = torch.full((1, seq_len), cfg.pad_token_id, dtype=torch.long)
    ids[:, :real_len] = full[:, :real_len]
    masked = torch.zeros((1, seq_len), dtype=torch.long)
    masked[:, :real_len] = 1
    unmasked = torch.ones((1, seq_len), dtype=torch.long)

    ref = make_ref(cfg, hf_model)
    with torch.no_grad():
        out_masked = ref(ids, masked)
        out_unmasked = ref(ids, unmasked)

    p = pcc(out_masked[:, :real_len], out_unmasked[:, :real_len])
    print(f"\n[padding-control] masked vs unmasked PCC={p:.10f} (must be < {PARITY_PCC})")
    assert p < PARITY_PCC, "attention mask had no effect - padding branch is dead code"


def test_bfloat16_reference(config, hf_model):
    """Reference must run in bf16 and match an equivalently-cast HF model."""
    ref_bf = make_ref(config, hf_model, dtype=torch.bfloat16)
    hf_bf = load_torch_model(dtype=torch.bfloat16, attn_implementation="eager")

    ids, mask = build_inputs(seq_len=256)
    with torch.no_grad():
        hf_out = hf_bf(input_ids=ids, attention_mask=mask).last_hidden_state
        ref_out = ref_bf(ids, mask)

    assert ref_out.dtype == torch.bfloat16, f"expected bf16 output, got {ref_out.dtype}"
    p = pcc(hf_out.float(), ref_out.float())
    print(f"\n[bf16 vs eager] reference vs HF-bf16 PCC={p:.10f}")
    assert p >= PARITY_PCC, f"bf16 reference diverges from bf16 HF-eager: {p:.10f}"


def test_torch_sdpa_eager_spread():
    """Records torch's own fused-vs-eager spread in bf16, with no reference code
    involved.
    """
    ids, mask = build_inputs(seq_len=256)
    sdpa = load_torch_model(dtype=torch.bfloat16, attn_implementation="sdpa")
    eager = load_torch_model(dtype=torch.bfloat16, attn_implementation="eager")
    with torch.no_grad():
        a = sdpa(input_ids=ids, attention_mask=mask).last_hidden_state
        b = eager(input_ids=ids, attention_mask=mask).last_hidden_state
    p = pcc(a.float(), b.float())
    print(f"\n[torch bf16 kernel spread] sdpa vs eager PCC={p:.10f}")
    # fp32 agrees exactly; bf16 does not. Assert the gap is real but bounded.
    assert p < PARITY_PCC, "expected a measurable sdpa/eager gap in bf16"
    assert p > 0.99, f"sdpa/eager spread larger than expected: {p:.10f}"


@pytest.mark.parametrize("seq_len", [256, 512, 768])
def test_longer_sequences(config, hf_model, seq_len):
    """Corpus is 862 tokens, so 768 is reachable. Longer sequences put more
    tokens outside the 129-wide local band, stressing the window logic harder."""
    ref = make_ref(config, hf_model)
    ids, mask = build_inputs(seq_len=seq_len)
    with torch.no_grad():
        hf_out = hf_model(input_ids=ids, attention_mask=mask).last_hidden_state
        ref_out = ref(ids, mask)
    p = pcc(hf_out, ref_out)
    print(f"\n[seq={seq_len}] PCC={p:.10f}")
    assert p >= PARITY_PCC
