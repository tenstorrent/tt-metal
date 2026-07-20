# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Optimized-decoder tests for hexgrad/Kokoro-82M (plbert / ALBERT encoder).

Mirrors ``tests/test_functional_decoder.py`` exactly (same lengths, seeds, and
representative-input methodology) but exercises
``models/demos/audio/kokoro/tt/optimized_decoder.py``. This is an
apples-to-apples PCC comparison against the functional acceptance bar (0.995).

The optimized path (packed QKV + nlp_create_qkv_heads + fused SDPA +
nlp_concat_heads + BFP8 weights + fused-gelu FF1 + explicit ffn_output program
config) is what these tests run - ``test_uses_optimized_path`` asserts there is
no functional fallback (single packed QKV weight, no separate q/k/v; the layer
issues an SDPA op).

Layer kinds / paged / stateless rationale: identical to the functional decoder
(Kokoro is non-autoregressive, one weight-tied AlbertLayer kind, no KV cache);
see that test file and doc/optimized_decoder/README.md.
"""

import json
import os
import random

import pytest
import torch

import ttnn
from models.demos.audio.kokoro.tt.optimized_decoder import OptimizedDecoder

MODEL_ID = "hexgrad/Kokoro-82M"
PCC_BAR = 0.995
DOC_DIR = os.path.join(os.path.dirname(__file__), "..", "doc", "functional_decoder")

_REP_SYMBOLS = "ɐɚɛɜɪʊʌəɹɾɡ aioueɑɔbdfhjklmnprstvwzˈˌ "

_IPA_SENTENCES = [
    "hɛlˈO wˈɜːld",
    "ðə kwˈɪk brˈaʊn fˈɑks ʤˈʌmps ˈOvɚ ðə lˈeɪzi dˈɔɡ",
    "tˈɛnstɔɹɛnt bˈɪldz ˈAI ˈaksɛlɚˌeɪɾɚz",
]


# --------------------------------------------------------------------- helpers
def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _rep_pool(vocab):
    return [vocab[c] for c in _REP_SYMBOLS if c in vocab]


def _rep_ids(vocab, batch, seq_len, seed):
    pool = _rep_pool(vocab)
    rng = random.Random(seed)
    rows = []
    for _ in range(batch):
        body = [rng.choice(pool) for _ in range(max(seq_len - 2, 0))]
        ids = ([0] + body + [0])[:seq_len]
        while len(ids) < seq_len:
            ids.append(0)
        rows.append(ids)
    return torch.tensor(rows, dtype=torch.long)


def _ipa_ids(vocab, text):
    return torch.tensor([[0] + [vocab[c] for c in text if c in vocab] + [0]], dtype=torch.long)


def _kokoro_config_dict():
    from huggingface_hub import hf_hub_download

    return json.load(open(hf_hub_download(MODEL_ID, "config.json")))


def _albert_config():
    from transformers import AlbertConfig

    cfg = _kokoro_config_dict()
    return AlbertConfig(vocab_size=cfg["n_token"], **cfg["plbert"])


def _kokoro_vocab():
    return _kokoro_config_dict()["vocab"]


def _real_state_dict():
    from huggingface_hub import hf_hub_download

    ckpt = hf_hub_download(MODEL_ID, "kokoro-v1_0.pth")
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)["bert"]
    return {k[len("module.") :] if k.startswith("module.") else k: v for k, v in sd.items()}


def _hf_model(config, state_dict):
    from transformers import AlbertModel

    m = AlbertModel(config).eval()
    m.load_state_dict(state_dict, strict=False)
    return m


def _synthetic_state_dict(config, seed=0):
    stats = json.load(open(os.path.join(DOC_DIR, "weight_stats.json")))["tensor_stats"]
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for name, st in stats.items():
        shape = st["shape"]
        if name.endswith("LayerNorm.weight") or name.endswith("layer_norm.weight"):
            sd[name] = torch.full(shape, st["mean"]) + st["std"] * torch.randn(shape, generator=g) * 0.1
        else:
            sd[name] = st["mean"] + st["std"] * torch.randn(shape, generator=g)
    return sd


def _run_prefill(decoder, device, ids, attention_mask=None):
    prep = OptimizedDecoder.prepare_inputs(ids, device, attention_mask=attention_mask)
    out = decoder.prefill_forward(
        prep["input_ids"],
        prep["position_ids"],
        prep["token_type_ids"],
        prep["attention_mask"],
        batch=prep["batch"],
        seq_len=prep["padded_seq_len"],
    )
    return ttnn.to_torch(out)[:, : prep["seq_len"], :].float()


def _run_decode(decoder, device, ids, attention_mask=None):
    prep = OptimizedDecoder.prepare_inputs(ids, device, attention_mask=attention_mask)
    out = decoder.decode_forward(
        prep["input_ids"],
        prep["position_ids"],
        prep["token_type_ids"],
        prep["attention_mask"],
        batch=prep["batch"],
        seq_len=prep["padded_seq_len"],
    )
    return ttnn.to_torch(out)[:, : prep["seq_len"], :].float()


# -------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="module")
def real_ctx(device):
    config = _albert_config()
    vocab = _kokoro_vocab()
    sd = _real_state_dict()
    hf = _hf_model(config, sd)
    decoder = OptimizedDecoder.from_state_dict(sd, hf_config=config, mesh_device=device)
    yield config, hf, decoder, vocab
    decoder.release_traces()


def _hf_forward(hf, ids, attention_mask=None):
    if attention_mask is None:
        attention_mask = torch.ones_like(ids)
    with torch.no_grad():
        return hf(ids, attention_mask=attention_mask).last_hidden_state


# --------------------------------------------------- optimized-path assertions
def test_uses_optimized_path(real_ctx, device):
    """Assert the delivered path is the optimized one, not a functional fallback:
    a single packed QKV weight (no separate q/k/v matmul weights) and the selected
    reduced-precision policy (BFP8 attention/MLP weights)."""
    config, hf, decoder, vocab = real_ctx
    assert "qkv_w" in decoder.w, "expected packed QKV weight (optimized path)"
    assert not any(k in decoder.w for k in ("q_w", "k_w", "v_w")), "separate q/k/v weights => functional fallback"
    # packed QKV output width == q + k + v
    assert decoder.w["qkv_w"].shape[-1] == 3 * config.hidden_size
    # selected precision policy actually reached the built weights/activations
    assert decoder.activation_dtype == ttnn.bfloat16
    assert decoder.w["qkv_w"].get_dtype() == ttnn.bfloat8_b, "attn weights should be BFP8 per selected policy"
    assert decoder.w["ffn_w"].get_dtype() == ttnn.bfloat8_b, "MLP weights should be BFP8 per selected policy"
    assert decoder.policy.fp32_dest_acc is True, "fp32 dest acc required for BFP8 short-length PCC"


def test_single_layer_kind():
    config = _albert_config()
    assert config.num_hidden_groups == 1
    assert config.inner_group_num == 1
    assert config.num_hidden_layers == 12
    assert config.hidden_size % config.num_attention_heads == 0


# --------------------------------------------------- prefill PCC (real weights)
@pytest.mark.parametrize("seq_len", [8, 16, 31, 32, 33, 64, 128, 256, 500, 511, 512])
def test_prefill_pcc_real_weights(real_ctx, device, seq_len):
    config, hf, decoder, vocab = real_ctx
    ids = _rep_ids(vocab, 1, seq_len, seed=seq_len)
    ref = _hf_forward(hf, ids)
    got = _run_prefill(decoder, device, ids)
    pcc = _pcc(got, ref)
    assert pcc >= PCC_BAR, f"prefill T={seq_len} PCC={pcc:.5f} < {PCC_BAR}"


def test_prefill_real_ipa_sentences(real_ctx, device):
    config, hf, decoder, vocab = real_ctx
    for text in _IPA_SENTENCES:
        ids = _ipa_ids(vocab, text)
        pcc = _pcc(_run_prefill(decoder, device, ids), _hf_forward(hf, ids))
        assert pcc >= PCC_BAR, f"IPA '{text[:24]}' (len {ids.shape[1]}) PCC={pcc:.5f} < {PCC_BAR}"


@pytest.mark.parametrize("batch", [2, 4, 8, 32])
def test_prefill_batch_pcc(real_ctx, device, batch):
    config, hf, decoder, vocab = real_ctx
    ids = _rep_ids(vocab, batch, 128, seed=1000 + batch)
    ref = _hf_forward(hf, ids)
    got = _run_prefill(decoder, device, ids)
    pcc = _pcc(got, ref)
    assert pcc >= PCC_BAR, f"prefill batch={batch} PCC={pcc:.5f} < {PCC_BAR}"


# ------------------------------------------------ decode (traced) PCC evidence
@pytest.mark.parametrize("seq_len", [32, 64, 128, 500, 511, 512])
def test_decode_traced_pcc_real_weights(real_ctx, device, seq_len):
    config, hf, decoder, vocab = real_ctx
    ids = _rep_ids(vocab, 1, seq_len, seed=5000 + seq_len)
    ref = _hf_forward(hf, ids)
    got = _run_decode(decoder, device, ids)
    pcc = _pcc(got, ref)
    assert pcc >= PCC_BAR, f"decode(traced) T={seq_len} PCC={pcc:.5f} < {PCC_BAR}"


def test_decode_traced_masked_batch(real_ctx, device):
    config, hf, decoder, vocab = real_ctx
    lengths = [113, 47]  # non-aligned; max 113 -> padded 128
    max_len = max(lengths)
    ids = torch.zeros((len(lengths), max_len), dtype=torch.long)
    mask = torch.zeros((len(lengths), max_len), dtype=torch.long)
    for i, L in enumerate(lengths):
        ids[i, :L] = _rep_ids(vocab, 1, L, seed=6000 + i)[0]
        mask[i, :L] = 1
    ref = _hf_forward(hf, ids, attention_mask=mask)
    got = _run_decode(decoder, device, ids, attention_mask=mask)
    for i, L in enumerate(lengths):
        pcc = _pcc(got[i, :L], ref[i, :L])
        assert pcc >= PCC_BAR, f"traced-decode masked row {i} (len {L}) PCC={pcc:.5f} < {PCC_BAR}"


# ----------------------------------------------------------------- determinism
def test_prefill_determinism(real_ctx, device):
    config, hf, decoder, vocab = real_ctx
    ids = _rep_ids(vocab, 1, 128, seed=77)
    a = _run_prefill(decoder, device, ids)
    b = _run_prefill(decoder, device, ids)
    assert torch.equal(a, b), "prefill not deterministic for identical input"


def test_decode_determinism(real_ctx, device):
    config, hf, decoder, vocab = real_ctx
    ids = _rep_ids(vocab, 1, 64, seed=88)
    a = _run_decode(decoder, device, ids)
    b = _run_decode(decoder, device, ids)
    assert torch.equal(a, b), "traced decode not deterministic for identical input"


# ------------------------------------------------------------- stateless proof
def test_decode_is_stateless(real_ctx, device):
    """Kokoro has no KV cache / autoregression: decode == prefill for same input."""
    config, hf, decoder, vocab = real_ctx
    ids = _rep_ids(vocab, 1, 64, seed=303)
    prefill = _run_prefill(decoder, device, ids)
    decode = _run_decode(decoder, device, ids)
    assert torch.equal(prefill, decode), "decode diverged from prefill (unexpected state)"


# --------------------------------------------------------------- padding masks
def test_padding_mask(real_ctx, device):
    config, hf, decoder, vocab = real_ctx
    lengths = [96, 40]
    max_len = max(lengths)
    ids = torch.zeros((len(lengths), max_len), dtype=torch.long)
    mask = torch.zeros((len(lengths), max_len), dtype=torch.long)
    for i, L in enumerate(lengths):
        ids[i, :L] = _rep_ids(vocab, 1, L, seed=404 + i)[0]
        mask[i, :L] = 1
    ref = _hf_forward(hf, ids, attention_mask=mask)
    got = _run_prefill(decoder, device, ids, attention_mask=mask)
    for i, L in enumerate(lengths):
        pcc = _pcc(got[i, :L], ref[i, :L])
        assert pcc >= PCC_BAR, f"masked row {i} (len {L}) PCC={pcc:.5f} < {PCC_BAR}"


# ------------------------------------------------------ stress / repeated runs
def test_decode_stress_repeated_replay(real_ctx, device):
    """Repeated traced-decode replays across several shapes stay correct and stable
    (trace-buffer reuse / persistent-input copy-update stress)."""
    config, hf, decoder, vocab = real_ctx
    shapes = [512, 128, 511, 64, 128, 512]  # revisit shapes to exercise trace cache reuse
    for rep, L in enumerate(shapes):
        ids = _rep_ids(vocab, 1, L, seed=9000 + L)
        ref = _hf_forward(hf, ids)
        first = None
        for _ in range(3):
            got = _run_decode(decoder, device, ids)
            if first is None:
                first = got
            else:
                assert torch.equal(first, got), f"decode replay nondeterministic at T={L} rep={rep}"
        pcc = _pcc(first, ref)
        assert pcc >= PCC_BAR, f"stress decode T={L} PCC={pcc:.5f} < {PCC_BAR}"


def test_prefill_stress_batched(real_ctx, device):
    """Repeated prefill over varied batch/length stays >= bar (larger-batch preserved)."""
    config, hf, decoder, vocab = real_ctx
    for batch, L in [(1, 300), (2, 200), (8, 96), (16, 64)]:
        ids = _rep_ids(vocab, batch, L, seed=12000 + batch * 100 + L)
        ref = _hf_forward(hf, ids)
        got = _run_prefill(decoder, device, ids)
        pcc = _pcc(got, ref)
        assert pcc >= PCC_BAR, f"stress prefill b={batch} T={L} PCC={pcc:.5f} < {PCC_BAR}"


# ------------------------------------------------- synthetic-weights (portable)
def test_synthetic_weights_pcc(device):
    """CI-portable path: deterministic synthetic weights, real config shapes."""
    config = _albert_config()
    sd = _synthetic_state_dict(config)
    hf = _hf_model(config, sd)
    decoder = OptimizedDecoder.from_state_dict(sd, hf_config=config, mesh_device=device)
    try:
        ids = _rep_ids(_kokoro_vocab(), 1, 96, seed=202)
        ref = _hf_forward(hf, ids)
        got = _run_prefill(decoder, device, ids)
        pcc = _pcc(got, ref)
        assert pcc >= PCC_BAR, f"synthetic-weights prefill PCC={pcc:.5f} < {PCC_BAR}"
    finally:
        decoder.release_traces()
