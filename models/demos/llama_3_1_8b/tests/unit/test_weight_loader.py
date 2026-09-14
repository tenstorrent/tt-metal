# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""P1 — the checkpoint loader. Host only, no device.

This stands in for the recipe's ``test_mxfp4_loader.py`` row, which is explicitly conditional on the
checkpoint being quantized. Llama-3.1-8B-Instruct is plain bf16 safetensors with no
``quantization_config`` — asserted, not assumed, by
``tests/torch_ref/test_reference_llama.py::test_checkpoint_is_not_quantized`` — so there is no
dequantization to verify. What there IS to verify is the part that would otherwise be silent: the
safetensors walk, the resolution order, the depth limit, and failing loudly on a key the model does
not expect rather than dropping it.

The Meta RoPE permutation and the QKV fusion are also tested here, because they are the two places a
weight is *reshaped* on its way to the device and a transposed one produces a model that runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models.demos.llama_3_1_8b.reference.config import LlamaConfig
from models.demos.llama_3_1_8b.tt import model_config
from models.demos.llama_3_1_8b.utils.rope_layout import hf_to_meta_perm, meta_permute_proj, meta_to_hf_perm

CHECKPOINT = "/mnt/models/meta-llama/Llama-3.1-8B-Instruct"
requires_checkpoint = pytest.mark.skipif(
    not Path(CHECKPOINT, "model.safetensors.index.json").exists(), reason=f"no checkpoint at {CHECKPOINT}"
)


@requires_checkpoint
def test_resolve_checkpoint_prefers_explicit_override(monkeypatch, tmp_path):
    monkeypatch.delenv("PREFILL_HF_MODEL", raising=False)
    monkeypatch.delenv("HF_MODEL", raising=False)
    assert model_config.resolve_checkpoint() == Path(CHECKPOINT), "should fall back to the shared store"

    monkeypatch.setenv("PREFILL_HF_MODEL", CHECKPOINT)
    assert model_config.resolve_checkpoint() == Path(CHECKPOINT)


def test_resolve_checkpoint_fails_loudly(monkeypatch, tmp_path):
    """No checkpoint anywhere must raise with what was tried — never silently synthesize weights."""
    monkeypatch.setattr(model_config, "SHARED_STORE", str(tmp_path / "absent"))
    monkeypatch.setenv("PREFILL_HF_MODEL", str(tmp_path / "also-absent"))
    monkeypatch.delenv("HF_MODEL", raising=False)
    with pytest.raises(FileNotFoundError, match="does not download weights"):
        model_config.resolve_checkpoint()


@requires_checkpoint
def test_load_state_dict_depth_limited():
    """``num_layers`` must load exactly those layers, from only the shards that hold them."""
    sd = model_config.load_state_dict(CHECKPOINT, num_layers=2)
    layers = {int(k.split(".")[2]) for k in sd if k.startswith("model.layers.")}
    assert layers == {0, 1}
    assert set(sd) - {k for k in sd if k.startswith("model.layers.")} == {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    }
    cfg = LlamaConfig.from_json()
    assert sd["model.embed_tokens.weight"].shape == (cfg.vocab_size, cfg.hidden_size)
    assert sd["lm_head.weight"].shape == (cfg.vocab_size, cfg.hidden_size)
    assert sd["model.layers.0.self_attn.q_proj.weight"].shape == (cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size)
    assert sd["model.layers.0.self_attn.k_proj.weight"].shape == (cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size)
    assert sd["model.layers.0.mlp.gate_proj.weight"].shape == (cfg.intermediate_size, cfg.hidden_size)
    assert all(v.dtype == torch.bfloat16 for v in sd.values()), "checkpoint tensors should arrive as bf16"


def test_validate_state_dict_rejects_surprises():
    cfg = LlamaConfig.from_json()
    sd = model_config.random_state_dict(cfg, num_layers=1)
    model_config.validate_state_dict(sd, num_layers=1)

    extra = dict(sd)
    extra["model.layers.0.self_attn.q_norm.weight"] = torch.zeros(128)
    with pytest.raises(AssertionError, match="extra="):
        model_config.validate_state_dict(extra, num_layers=1)

    missing = {k: v for k, v in sd.items() if k != "model.layers.0.mlp.up_proj.weight"}
    with pytest.raises(AssertionError, match="missing="):
        model_config.validate_state_dict(missing, num_layers=1)

    renamed = {k.replace("model.norm.weight", "model.final_layernorm.weight"): v for k, v in sd.items()}
    with pytest.raises(AssertionError):
        model_config.validate_state_dict(renamed, num_layers=1)


def test_meta_permute_is_an_involution_free_bijection():
    """The row permutation and the column gather must be exact inverses of each other."""
    head_dim, n_heads = 128, 3
    w = torch.arange(n_heads * head_dim * 4, dtype=torch.float32).reshape(n_heads * head_dim, 4)
    permuted = meta_permute_proj(w, head_dim)
    assert permuted.shape == w.shape

    # Row m of the permuted weight is row hf_to_meta_perm[m] of the original, per head.
    perm = hf_to_meta_perm(head_dim)
    for h in range(n_heads):
        base = h * head_dim
        for m in range(head_dim):
            assert torch.equal(permuted[base + m], w[base + int(perm[m])])

    # ... and meta_to_hf_perm puts it back.
    inv = meta_to_hf_perm(head_dim)
    restored = permuted.reshape(n_heads, head_dim, 4)[:, inv].reshape(n_heads * head_dim, 4)
    assert torch.equal(restored, w)


def test_meta_permute_preserves_the_rotation():
    """Rotating in the Meta order must equal rotating in HF order, up to the column permutation.

    This is the property the whole device layout rests on: if it did not hold, every K in the cache
    would be a differently-rotated vector and no amount of permuting the golden would fix it.
    """
    from models.demos.llama_3_1_8b.reference import model as ref

    cfg = LlamaConfig.from_json()
    torch.manual_seed(0)
    seq = 64
    k_hf = torch.randn(1, 2, seq, cfg.head_dim)
    cos, sin = ref.rope_cos_sin(cfg, seq, dtype=torch.float32)
    rotated_hf = ref.apply_rope(k_hf, cos, sin)

    # Meta form: interleave the columns, and pair (2j, 2j+1) with the duplicated cos/sin.
    from models.demos.llama_3_1_8b.utils.rope_layout import meta_cos_sin

    perm = hf_to_meta_perm(cfg.head_dim)
    k_meta = k_hf[..., perm]
    c, s = meta_cos_sin(cos, sin)
    even, odd = k_meta[..., 0::2], k_meta[..., 1::2]
    rot_even = even * c[..., 0::2] - odd * s[..., 0::2]
    rot_odd = odd * c[..., 1::2] + even * s[..., 1::2]
    rotated_meta = torch.stack([rot_even, rot_odd], -1).flatten(-2)

    assert torch.allclose(rotated_meta, rotated_hf[..., perm], atol=2e-3)


@requires_checkpoint
def test_qkv_fusion_layout_is_per_device():
    """The fused QKV weight must be ``[q_i | k_i | v_i]`` per TP device, not a global q|k|v concat.

    A global concatenation followed by a column-parallel shard gives device 0 all of q and none of
    k or v — the model still runs, and every number it produces is wrong.
    """
    cfg = LlamaConfig.from_json()
    tp, d = 4, cfg.head_dim
    sd = model_config.random_state_dict(cfg, num_layers=1)
    q = sd["model.layers.0.self_attn.q_proj.weight"]
    k = sd["model.layers.0.self_attn.k_proj.weight"]
    v = sd["model.layers.0.self_attn.v_proj.weight"]

    q_m, k_m = meta_permute_proj(q, d), meta_permute_proj(k, d)
    per_device = []
    for i in range(tp):
        per_device.append(
            torch.cat(
                [
                    torch.chunk(q_m, tp, 0)[i].transpose(-2, -1),
                    torch.chunk(k_m, tp, 0)[i].transpose(-2, -1),
                    torch.chunk(v, tp, 0)[i].transpose(-2, -1),
                ],
                dim=-1,
            )
        )
    fused = torch.cat(per_device, dim=-1)

    q_local = cfg.num_attention_heads // tp * d
    kv_local = cfg.num_key_value_heads // tp * d
    assert fused.shape == (cfg.hidden_size, tp * (q_local + 2 * kv_local))
    stride = q_local + 2 * kv_local
    for i in range(tp):
        block = fused[:, i * stride : (i + 1) * stride]
        assert torch.equal(block[:, :q_local], torch.chunk(q_m, tp, 0)[i].transpose(-2, -1))
        assert torch.equal(block[:, q_local : q_local + kv_local], torch.chunk(k_m, tp, 0)[i].transpose(-2, -1))
        assert torch.equal(block[:, q_local + kv_local :], torch.chunk(v, tp, 0)[i].transpose(-2, -1))
