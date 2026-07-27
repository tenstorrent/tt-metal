# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared reference layer — chiefly the hand-written safetensors reader.

The reader replaces the `safetensors` package, so nothing else in the suite exercises it: the
block tests all build weights from the vendored manifest. If it is wrong, every block silently
loads garbage the first time a real checkpoint is used. These tests write safetensors files BY
HAND (stdlib struct + json only, no safetensors dependency) and read them back.

    pytest -svv models/experimental/voxtral_tts/tests/test_common_ref.py
"""

import json
import struct

import pytest
import torch

from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
    ACOUSTIC_CODEBOOK_SIZE,
    N_ACOUSTIC_CODEBOOK,
    NUM_CODEBOOKS,
    SEMANTIC_CODEBOOK_SIZE,
    SafeTensors,
    codebook_offsets,
    codebook_sizes,
    gqa_attention,
    load_manifest,
    pcc,
    repeat_kv,
    rms_norm,
    swiglu,
)

_ST_NAMES = {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16", torch.int64: "I64"}


def _write_safetensors(path, tensors):
    """Minimal writer: u64 header length | JSON header | contiguous tensor bytes."""
    header, blobs, off = {}, [], 0
    for name, t in tensors.items():
        t = t.contiguous()
        raw = t.view(torch.uint8).reshape(-1).numpy().tobytes() if t.dtype == torch.bfloat16 else None
        if raw is None:
            raw = t.flatten().numpy().tobytes()
        header[name] = {"dtype": _ST_NAMES[t.dtype], "shape": list(t.shape),
                        "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    blob = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for b in blobs:
            f.write(b)


def test_reader_round_trips_every_dtype(tmp_path):
    """bf16 is the one that matters — the whole checkpoint is bf16 and torch cannot memcpy it
    through numpy, so a wrong dtype mapping would give plausible-looking garbage."""
    p = tmp_path / "rt.safetensors"
    ref = {
        "a.weight": torch.randn(4, 7).to(torch.bfloat16),
        "b": torch.randn(3, 3, 2),
        "c": torch.arange(5, dtype=torch.int64),
        "d.weight": torch.randn(240, 1024, 7).to(torch.bfloat16),  # a real output_proj shape
    }
    _write_safetensors(p, ref)
    st = SafeTensors(str(p))
    assert sorted(st.keys()) == sorted(ref)
    for k, v in ref.items():
        got = st.get(k, torch.float32)
        assert got.shape == v.shape, k
        assert torch.equal(got, v.to(torch.float32)), f"{k} not bit-exact"


def test_reader_seeks_and_does_not_depend_on_order(tmp_path):
    """Tensors must be addressed by data_offsets, not by position, and reading one must not
    require reading its neighbours (that is the point of seeking per tensor)."""
    p = tmp_path / "s.safetensors"
    ref = {"x": torch.randn(64, 64), "y": torch.arange(9, dtype=torch.int64), "z": torch.randn(8)}
    _write_safetensors(p, ref)
    st = SafeTensors(str(p))
    for k in ("z", "x", "y"):  # deliberately out of file order
        assert torch.equal(st.get(k, torch.float32), ref[k].to(torch.float32)), k
    assert st.shape("x") == (64, 64)
    assert "nope" not in st and "x" in st


def test_prefixed_strips_and_filters(tmp_path):
    p = tmp_path / "p.safetensors"
    _write_safetensors(p, {"enc.a": torch.randn(2), "enc.b": torch.randn(2), "dec.a": torch.randn(2)})
    st = SafeTensors(str(p))
    assert sorted(st.prefixed("enc.").keys()) == ["a", "b"]
    assert sorted(st.prefixed("enc.", strip=False).keys()) == ["enc.a", "enc.b"]


def test_missing_checkpoint_raises_with_download_hint():
    with pytest.raises(FileNotFoundError, match="hf download"):
        SafeTensors("/nonexistent/consolidated.safetensors")


# ---------------------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------------------
def test_rms_norm_matches_torch():
    x = torch.randn(2, 5, 64)
    wgt = torch.rand(64) + 0.5
    for eps in (1e-5, 1e-6, 1e-2):  # 1e-2 is the codec's
        ours = rms_norm(x, wgt, eps)
        ref = torch.nn.functional.rms_norm(x, (64,), wgt, eps)
        assert pcc(ours, ref) > 0.99999
        assert torch.allclose(ours, ref, atol=1e-6), f"eps={eps}"


def test_swiglu_matches_module_form():
    x = torch.randn(3, 4, 16)
    w1, w3, w2 = torch.randn(32, 16), torch.randn(32, 16), torch.randn(16, 32)
    ref = (torch.nn.functional.silu(x @ w1.t()) * (x @ w3.t())) @ w2.t()
    assert torch.allclose(swiglu(x, w1, w2, w3), ref, atol=1e-5)


def test_repeat_kv_is_interleaved():
    """GQA grouping must repeat each KV head consecutively (head i serves queries 4i..4i+3),
    not tile the whole set. Getting this wrong scrambles which query attends to which key."""
    k = torch.arange(2 * 3).float().view(1, 2, 3, 1)  # 2 kv heads
    out = repeat_kv(k, 3)
    assert out.shape == (1, 6, 3, 1)
    assert torch.equal(out[0, 0], out[0, 1]) and torch.equal(out[0, 1], out[0, 2])
    assert not torch.equal(out[0, 2], out[0, 3]), "second kv head must start at index 3"


def test_gqa_attention_matches_sdpa():
    q = torch.randn(1, 8, 6, 16)
    k = torch.randn(1, 2, 6, 16)
    v = torch.randn(1, 2, 6, 16)
    ref = torch.nn.functional.scaled_dot_product_attention(q, repeat_kv(k, 4), repeat_kv(v, 4))
    assert torch.allclose(gqa_attention(q, k, v), ref, atol=1e-5)


def test_gqa_attention_honours_additive_bias():
    q = torch.randn(1, 8, 5, 16)
    k, v = torch.randn(1, 8, 5, 16), torch.randn(1, 8, 5, 16)
    bias = torch.zeros(1, 1, 5, 5)
    bias[..., 1:] = float("-inf")  # only key 0 survives
    out = gqa_attention(q, k, v, bias)
    assert torch.allclose(out, v[:, :, :1].expand_as(out), atol=1e-5), "bias not applied pre-softmax"


# ---------------------------------------------------------------------------------------
# Codebook bookkeeping
# ---------------------------------------------------------------------------------------
def test_codebook_sizes_and_offsets():
    sizes = codebook_sizes()
    assert len(sizes) == NUM_CODEBOOKS == 37
    assert sizes[0] == SEMANTIC_CODEBOOK_SIZE + 2
    assert all(s == ACOUSTIC_CODEBOOK_SIZE + 2 for s in sizes[1:])
    off = codebook_offsets()
    assert off[0] == 0 and off[1] == SEMANTIC_CODEBOOK_SIZE + 2
    total = int(off[-1]) + sizes[-1]
    assert total == 9022
    table = load_manifest()["mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight"]["shape"][0]
    assert table == 9088 == 128 * ((total + 127) // 128), "table must be the 128-padded total"


def test_codebook_sizes_without_specials():
    assert codebook_sizes(include_special=False) == [SEMANTIC_CODEBOOK_SIZE] + \
        [ACOUSTIC_CODEBOOK_SIZE] * N_ACOUSTIC_CODEBOOK


def test_pcc_edges():
    x = torch.randn(50)
    assert pcc(x, x) == pytest.approx(1.0, abs=1e-6)
    assert pcc(x, -x) == pytest.approx(-1.0, abs=1e-6)
    assert pcc(torch.zeros(5), torch.zeros(5)) == 1.0  # degenerate: defined as 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-svv", __file__]))
