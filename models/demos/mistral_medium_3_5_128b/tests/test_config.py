# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""D1 (host only): every config constant against the vendored config.json, the binding spec, and the
vendored torch reference against the upstream HF Ministral3 math (transformers 5.12.1)."""

import json
import os
from pathlib import Path

import pytest
import torch
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config
from transformers.models.ministral3.modeling_ministral3 import (
    Ministral3DecoderLayer,
    Ministral3ForCausalLM,
    Ministral3RotaryEmbedding,
)

from models.common.utility_functions import comp_pcc
from models.demos.mistral_medium_3_5_128b.config import (
    VENDORED_CONFIG,
    VENDORED_SPEC,
    MistralMediumConfig,
    load_prefill_spec,
    pcc_thresholds,
    resolve_dataformats,
)
from models.demos.mistral_medium_3_5_128b.reference.model import (
    ReferenceDecoderLayer,
    build_reference_model,
    random_state_dict,
    rope_cos_sin,
    yarn_inv_freq,
)

# Reduced width for the HF-parity checks: real head_dim (YaRN depends on it) and the real GQA ratio shape.
REDUCED = dict(hidden_size=512, intermediate_size=1024, num_attention_heads=8, num_key_value_heads=2, vocab_size=1024)


def _text_config():
    with open(VENDORED_CONFIG) as f:
        return json.load(f)["text_config"]


def hf_text_config(cfg: MistralMediumConfig) -> Ministral3Config:
    hf = Ministral3Config(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        rms_norm_eps=cfg.rms_norm_eps,
        max_position_embeddings=cfg.max_position_embeddings,
        rope_parameters=dict(_text_config()["rope_parameters"]),
        tie_word_embeddings=False,
        sliding_window=None,
    )
    hf._attn_implementation = "sdpa"
    return hf


def test_constants_match_vendored_config():
    tc = _text_config()
    cfg = MistralMediumConfig.from_json()
    # The class defaults ARE the checkpoint's constants.
    assert cfg == MistralMediumConfig()
    for field, key in [
        ("vocab_size", "vocab_size"),
        ("hidden_size", "hidden_size"),
        ("intermediate_size", "intermediate_size"),
        ("num_hidden_layers", "num_hidden_layers"),
        ("num_attention_heads", "num_attention_heads"),
        ("num_key_value_heads", "num_key_value_heads"),
        ("head_dim", "head_dim"),
        ("rms_norm_eps", "rms_norm_eps"),
        ("max_position_embeddings", "max_position_embeddings"),
        ("hidden_act", "hidden_act"),
        ("tie_word_embeddings", "tie_word_embeddings"),
        ("bos_token_id", "bos_token_id"),
        ("eos_token_id", "eos_token_id"),
        ("pad_token_id", "pad_token_id"),
    ]:
        assert getattr(cfg, field) == tc[key], field
    rp = tc["rope_parameters"]
    assert (cfg.rope_type, cfg.rope_theta, cfg.rope_factor) == (rp["rope_type"], rp["rope_theta"], rp["factor"])
    assert cfg.original_max_position_embeddings == rp["original_max_position_embeddings"]
    assert (cfg.beta_fast, cfg.beta_slow) == (rp["beta_fast"], rp["beta_slow"])
    assert (cfg.mscale, cfg.mscale_all_dim, cfg.llama_4_scaling_beta) == (
        rp["mscale"],
        rp["mscale_all_dim"],
        rp["llama_4_scaling_beta"],
    )
    assert tc["sliding_window"] is None and tc["model_type"] == "ministral3"
    with open(VENDORED_CONFIG) as f:
        qc = json.load(f)["quantization_config"]
    assert qc["quant_method"] == "fp8" and qc["weight_block_size"] is None and "lm_head" in qc["modules_to_not_convert"]
    assert cfg.num_kv_groups == 12


def test_vendored_config_is_the_checkpoint_config():
    ckpt = os.environ.get("PREFILL_HF_MODEL") or os.environ.get("HF_MODEL")
    if not ckpt or not (Path(ckpt) / "config.json").is_file():
        pytest.skip("no checkpoint dir (PREFILL_HF_MODEL / HF_MODEL) to compare against")
    with open(Path(ckpt) / "config.json") as f, open(VENDORED_CONFIG) as g:
        assert json.load(f) == json.load(g)


def test_spec_is_binding_and_resolves():
    spec = load_prefill_spec()
    with open(VENDORED_SPEC) as f:
        assert json.load(f) == spec, "vendored spec snapshot diverged from PREFILL_SPEC"
    assert spec["model_name"] == "mistral_medium_3_5_128b" and spec["target_hw"] == "bh_galaxy"
    sp, tp = spec["parallelism"]["sp"], spec["parallelism"]["tp"]
    assert (sp, tp) == (8, 4)
    for key in ("max_seq_len", "chunk_size"):
        assert spec["shapes"][key] % (32 * sp) == 0, key
    df = resolve_dataformats(spec)
    assert df == {
        "activations": "bfloat16",
        "kv_cache": "bfloat8_b",
        "weights_default": "bfloat8_b",
        "attention": "bfloat8_b",
        "mlp_gate": "bfloat8_b",
        "mlp_up": "bfloat8_b",
        "mlp_down": "bfloat8_b",
    }
    assert pcc_thresholds(spec) == (0.99, 0.85)
    cfg = MistralMediumConfig.from_json()
    assert cfg.num_attention_heads % tp == 0 and cfg.num_key_value_heads % tp == 0
    assert cfg.hidden_size % (32 * tp) == 0 and cfg.intermediate_size % (32 * tp) == 0


def test_yarn_rope_matches_hf():
    cfg = MistralMediumConfig.from_json()
    hf = hf_text_config(cfg)
    inv_hf, scale_hf = ROPE_INIT_FUNCTIONS["yarn"](hf, None)
    inv, scale = yarn_inv_freq(cfg)
    assert torch.equal(inv, inv_hf.float()), (inv - inv_hf).abs().max()
    assert scale == pytest.approx(scale_hf, abs=0.0) and scale == pytest.approx(
        0.1 * torch.log(torch.tensor(64.0)).item() + 1
    )

    positions = torch.cat([torch.arange(10240), torch.tensor([131071, 262143])])
    rotary = Ministral3RotaryEmbedding(hf)
    cos_hf, sin_hf = rotary(torch.zeros(1, dtype=torch.bfloat16), positions[None])
    cos, sin = rope_cos_sin(cfg, positions)
    assert torch.equal(cos, cos_hf[0]) and torch.equal(sin, sin_hf[0])


def _hf_layer_state(ref_sd, i):
    prefix = f"layers.{i}."
    return {k[len(prefix) :]: v for k, v in ref_sd.items() if k.startswith(prefix)}


@torch.no_grad()
def test_decoder_layer_matches_hf():
    cfg = MistralMediumConfig().reduced(num_hidden_layers=1, **REDUCED)
    hf_cfg = hf_text_config(cfg)
    sd = random_state_dict(cfg, seed=3)
    layer_sd = _hf_layer_state(sd, 0)

    ref = ReferenceDecoderLayer(cfg).to(torch.bfloat16).eval()
    ref.load_state_dict(layer_sd)
    hf = Ministral3DecoderLayer(hf_cfg, 0).to(torch.bfloat16).eval()
    hf.load_state_dict(layer_sd)

    s = 256
    x = torch.randn(1, s, cfg.hidden_size, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    positions = torch.arange(s)
    cos, sin = rope_cos_sin(cfg, positions)
    out_ref, _, _ = ref(x, cos, sin, positions)
    cos_hf, sin_hf = Ministral3RotaryEmbedding(hf_cfg)(x, positions[None])
    out_hf = hf(x, attention_mask=None, position_ids=positions[None], position_embeddings=(cos_hf, sin_hf))
    out_hf = out_hf[0] if isinstance(out_hf, tuple) else out_hf
    passing, pcc = comp_pcc(out_hf.float(), out_ref.float(), 0.9999)
    assert passing, f"decoder layer vs HF: {pcc}"


@torch.no_grad()
def test_model_and_kv_match_hf():
    cfg = MistralMediumConfig().reduced(num_hidden_layers=2, **REDUCED)
    hf_cfg = hf_text_config(cfg)
    sd = random_state_dict(cfg, seed=5)
    ref = build_reference_model(cfg, sd)
    hf = Ministral3ForCausalLM(hf_cfg).to(torch.bfloat16).eval()
    hf.load_state_dict({(k if k == "lm_head.weight" else "model." + k): v for k, v in sd.items()})
    # .to(bf16) also rounds the rotary inv_freq buffer; from_pretrained keeps it fp32 (and so does the
    # golden trace: layer-0 K matches it bit-exactly only with fp32 angles). Rebuild it in fp32.
    hf.model.rotary_emb = Ministral3RotaryEmbedding(hf_cfg)

    tokens = torch.randint(0, cfg.vocab_size, (1, 192), generator=torch.Generator().manual_seed(7))
    logits_ref, _, kvs = ref(tokens)
    out = hf(input_ids=tokens, use_cache=True)
    passing, pcc = comp_pcc(out.logits.float(), logits_ref.float(), 0.9999)
    assert passing, f"logits vs HF: {pcc}"
    for i, (k, v) in enumerate(kvs):
        layer = out.past_key_values.layers[i]
        for name, got, want in (("k", k, layer.keys), ("v", v, layer.values)):
            passing, pcc = comp_pcc(want.float(), got.float(), 0.9999)
            assert passing, f"layer {i} {name} vs HF cache: {pcc}"
