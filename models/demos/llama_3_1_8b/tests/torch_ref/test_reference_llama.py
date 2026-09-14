# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""D1 — the torch reference, pinned two ways. Host only: no ttnn, no device, no checkpoint.

1. Every constant in :class:`LlamaConfig` against the vendored ``configs/config.json``, and that
   vendored file against the real checkpoint's when one is reachable — so the dims the whole
   bring-up is built on cannot drift from the weights.
2. Every block of ``reference/model.py`` against the upstream HuggingFace math
   (``transformers.models.llama.modeling_llama``) at a reduced config with **identical** random
   weights, plus the whole-model forward. Two oracles that cannot drift apart.

Reduced dims here are a *host-side diagnostic* (recipe §4): they test the math, not the model. The
graded whole-model number is the acceptance run at full depth and width.
"""

import json
from pathlib import Path

import pytest
import torch

from models.demos.llama_3_1_8b.reference import model as ref
from models.demos.llama_3_1_8b.reference.config import CONFIG_JSON, LlamaConfig

CHECKPOINT_DIR = "/mnt/models/meta-llama/Llama-3.1-8B-Instruct"

pytestmark = pytest.mark.parametrize("seed", [0])


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    x, y = a.to(torch.float64).flatten(), b.to(torch.float64).flatten()
    x, y = x - x.mean(), y - y.mean()
    denom = (x.norm() * y.norm()).item()
    return 1.0 if denom == 0 else float((x @ y).item() / denom)


def _small_cfg() -> LlamaConfig:
    """Reduced-dimension config (2 layers, hidden 256, vocab 512) — diagnostics only."""
    return LlamaConfig.from_json().reduced(num_hidden_layers=2, hidden_size=256, vocab_size=512)


# ---------------------------------------------------------------------------------------------
# 1. config constants
# ---------------------------------------------------------------------------------------------
def test_config_constants_match_vendored_json(seed):
    """Every field of the constants class equals the vendored config.json."""
    with open(CONFIG_JSON) as f:
        raw = json.load(f)
    cfg = LlamaConfig()  # the hand-written defaults, NOT from_json
    for key in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "rms_norm_eps",
        "rope_theta",
        "max_position_embeddings",
        "hidden_act",
        "attention_bias",
        "mlp_bias",
        "tie_word_embeddings",
    ):
        assert getattr(cfg, key) == raw[key], f"{key}: class {getattr(cfg, key)} != config.json {raw[key]}"
    assert cfg.rope_scaling == raw["rope_scaling"]
    assert cfg.head_dim == 128 and cfg.num_key_value_groups == 4
    assert raw["architectures"] == ["LlamaForCausalLM"]


def test_vendored_config_matches_checkpoint(seed):
    """The vendored config.json is a copy of the real checkpoint's."""
    ckpt = Path(CHECKPOINT_DIR) / "config.json"
    if not ckpt.exists():
        pytest.skip(f"checkpoint not reachable at {ckpt}")
    assert json.load(open(CONFIG_JSON)) == json.load(open(ckpt))


def test_checkpoint_is_not_quantized(seed):
    """P1's dequant row is dropped only because the checkpoint really is plain bf16 — assert it."""
    ckpt = Path(CHECKPOINT_DIR) / "config.json"
    if not ckpt.exists():
        pytest.skip(f"checkpoint not reachable at {ckpt}")
    raw = json.load(open(ckpt))
    assert "quantization_config" not in raw, "checkpoint is quantized; the dequant recipe row is NOT droppable"
    assert raw["torch_dtype"] == "bfloat16"


# ---------------------------------------------------------------------------------------------
# 2. reference vs upstream HuggingFace
# ---------------------------------------------------------------------------------------------
def _hf_config(cfg: LlamaConfig):
    from transformers.models.llama.configuration_llama import LlamaConfig as HFConfig

    return HFConfig(
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        vocab_size=cfg.vocab_size,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        max_position_embeddings=cfg.max_position_embeddings,
        rope_scaling=dict(cfg.rope_scaling),
        attention_bias=cfg.attention_bias,
        mlp_bias=cfg.mlp_bias,
        tie_word_embeddings=cfg.tie_word_embeddings,
        attn_implementation="eager",
        dtype=torch.float16,
    )


def test_rope_inv_freq_vs_hf(seed):
    """The llama3 piecewise rescaling, against HF's own rope-init function at FULL head_dim."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    cfg = LlamaConfig.from_json()
    hf_inv, attn_scale = ROPE_INIT_FUNCTIONS["llama3"](_hf_config(cfg), device=torch.device("cpu"))
    assert attn_scale == 1.0, "llama3 rope has no attention scaling"
    ours = ref.llama3_inv_freq(cfg)
    # HF builds inv_freq in fp32 and we build it in fp64, so the agreement bound is fp32's relative
    # resolution (~1.2e-7), not an absolute one: the smallest frequency here is 3e-7 and the largest 1.0.
    assert torch.allclose(ours.to(torch.float32), hf_inv.to(torch.float32), rtol=1e-6, atol=0)


def test_rope_freqs_match_the_device_rope_source(seed):
    """The reference's llama3 rescaling equals ``tt_transformers``' — the code the DEVICE cos/sin is
    built from. Without this the two halves of every RoPE comparison could drift together.

    The *rescaled inverse frequencies* must agree to fp32 resolution. The *angles* do not: the
    reference forms ``pos * inv_freq`` in fp64 and ``precompute_freqs`` does it in fp32, and at
    position 4095 on the unscaled j=1 frequency the angle is ~3300 rad, where fp32's ulp is 2.4e-4.
    So cos/sin agree only to ~3e-4, entirely in the FAST (unscaled, wavelen < 2048) columns — the
    slow columns, whose angles stay small, agree exactly. That is a full order of magnitude below
    the bf16 the device stores cos/sin in (ulp ~4e-3 near 1.0), so it never reaches a PCC number;
    the test pins the shape of the disagreement so a real scaling change cannot hide inside it.
    """
    from models.tt_transformers.tt.common import apply_scaling, precompute_freqs

    cfg = LlamaConfig.from_json()
    half = cfg.head_dim // 2

    unscaled = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2)[:half].float() / cfg.head_dim))
    tt_inv = apply_scaling(
        unscaled.clone(),
        cfg.rope_scaling["factor"],
        cfg.rope_scaling["original_max_position_embeddings"],
        rope_type="llama3",
    )
    assert torch.allclose(ref.llama3_inv_freq(cfg).to(torch.float32), tt_inv, rtol=1e-6, atol=0)

    seq = 4096
    tt_cos, tt_sin = precompute_freqs(
        cfg.head_dim,
        seq,
        theta=cfg.rope_theta,
        scale_factor=cfg.rope_scaling["factor"],
        orig_context_len=cfg.rope_scaling["original_max_position_embeddings"],
        rope_type="llama3",
    )
    cos, sin = ref.rope_cos_sin(cfg, seq, dtype=torch.float32)
    dcos = (cos[:, :half] - tt_cos).abs()
    dsin = (sin[:, :half] - tt_sin).abs()
    fp32_ulp_bound = torch.outer(torch.arange(seq).float(), tt_inv).abs().max(0).values * 2.0**-23 * 4
    assert (dcos.max(0).values <= fp32_ulp_bound + 1e-6).all(), "cos diff exceeds the fp32 angle ulp"
    assert (dsin.max(0).values <= fp32_ulp_bound + 1e-6).all(), "sin diff exceeds the fp32 angle ulp"
    assert dcos.max() < 1e-3 and dsin.max() < 1e-3


def test_rope_cos_sin_vs_hf_full_context(seed):
    """cos/sin over the full 10240-token acceptance range, against HF's LlamaRotaryEmbedding."""
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    cfg = LlamaConfig.from_json()
    seq = 10240
    rot = LlamaRotaryEmbedding(_hf_config(cfg))
    pos = torch.arange(seq).unsqueeze(0)
    hf_cos, hf_sin = rot(torch.zeros(1, seq, cfg.hidden_size, dtype=torch.float16), pos)
    cos, sin = ref.rope_cos_sin(cfg, seq)
    assert _pcc(cos, hf_cos[0]) > 0.999999 and _pcc(sin, hf_sin[0]) > 0.999999
    assert (cos.float() - hf_cos[0].float()).abs().max() < 1e-3  # fp16 storage granularity
    assert (sin.float() - hf_sin[0].float()).abs().max() < 1e-3


def test_rms_norm_vs_hf(seed):
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    torch.manual_seed(seed)
    cfg = _small_cfg()
    ours = ref.RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
    w = torch.randn(cfg.hidden_size, dtype=torch.float16)
    ours.weight.data = w.clone()
    hf = LlamaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps).to(torch.float16)
    hf.weight.data = w.clone()
    x = torch.randn(1, 64, cfg.hidden_size, dtype=torch.float16)
    assert _pcc(ours(x), hf(x)) > 0.9999


def test_mlp_vs_hf(seed):
    from transformers.models.llama.modeling_llama import LlamaMLP

    torch.manual_seed(seed)
    cfg = _small_cfg()
    ours, hf = ref.MLP(cfg), LlamaMLP(_hf_config(cfg)).to(torch.float16)
    hf.load_state_dict({k: v.clone() for k, v in ours.state_dict().items()})
    x = torch.randn(1, 64, cfg.hidden_size, dtype=torch.float16)
    assert _pcc(ours(x), hf(x)) > 0.999


def test_attention_vs_hf(seed):
    """Whole attention block, and the K it dumps, against HF at identical weights."""
    from transformers.models.llama.modeling_llama import LlamaAttention

    torch.manual_seed(seed)
    cfg = _small_cfg()
    hf_cfg = _hf_config(cfg)
    ours = ref.Attention(cfg)
    hf = LlamaAttention(hf_cfg, layer_idx=0).to(torch.float16)
    hf.load_state_dict({k: v.clone() for k, v in ours.state_dict().items()})

    s = 128
    x = torch.randn(1, s, cfg.hidden_size, dtype=torch.float16)
    cos, sin = ref.rope_cos_sin(cfg, s)
    out, k, v = ours(x, cos, sin)

    mask = torch.full((1, 1, s, s), float("-inf"), dtype=torch.float32).triu(1).to(torch.float16)
    hf_out, _ = hf(x, position_embeddings=(cos.unsqueeze(0), sin.unsqueeze(0)), attention_mask=mask)
    assert _pcc(out, hf_out) > 0.999

    # and the dumped K is the post-RoPE K HF would have cached
    _, k_hf, v_hf = ours.project(x)
    assert _pcc(k, ref.apply_rope(k_hf, cos, sin)) > 0.9999
    assert torch.equal(v, v_hf)


def test_decoder_layer_vs_hf(seed):
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    torch.manual_seed(seed)
    cfg = _small_cfg()
    ours = ref.DecoderLayer(cfg)
    hf = LlamaDecoderLayer(_hf_config(cfg), layer_idx=0).to(torch.float16)
    hf.load_state_dict({k: v.clone() for k, v in ours.state_dict().items()})

    s = 128
    x = torch.randn(1, s, cfg.hidden_size, dtype=torch.float16)
    cos, sin = ref.rope_cos_sin(cfg, s)
    out, _, _ = ours(x, cos, sin)
    mask = torch.full((1, 1, s, s), float("-inf"), dtype=torch.float32).triu(1).to(torch.float16)
    hf_out = hf(x, attention_mask=mask, position_embeddings=(cos.unsqueeze(0), sin.unsqueeze(0)))
    hf_out = hf_out[0] if isinstance(hf_out, (tuple, list)) else hf_out
    assert _pcc(out, hf_out) > 0.999


def test_whole_model_vs_hf(seed):
    """M1's ground-truth shape at reduced dims: the full reference forward against HF LlamaModel."""
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = _small_cfg()
    ours = ref.LlamaReference(cfg)
    hf = LlamaForCausalLM(_hf_config(cfg)).to(torch.float16)
    hf.load_state_dict(
        {("model." + k if not k.startswith("lm_head") else k): v.clone() for k, v in ours.state_dict().items()}
    )
    hf.eval()

    s = 128
    ids = torch.randint(0, cfg.vocab_size, (1, s))
    logits, kv = ours(ids)
    with torch.no_grad():
        hf_out = hf(ids, use_cache=True)
    assert _pcc(logits, hf_out.logits) > 0.999
    assert len(kv) == cfg.num_hidden_layers
    for i, (k, v) in enumerate(kv):
        assert _pcc(k, hf_out.past_key_values.layers[i].keys) > 0.999
        assert _pcc(v, hf_out.past_key_values.layers[i].values) > 0.999


def test_chunked_reference_matches_one_shot(seed):
    """The host chunked path equals the host one-shot path — the property P2 asserts on device."""
    torch.manual_seed(seed)
    cfg = _small_cfg()
    model = ref.LlamaReference(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 256))
    logits_1, kv_1 = model(ids)
    logits_c, kv_c = model.forward_chunked(ids, chunk_size=128)
    # one-shot logits cover the whole sequence; the chunked run returns the last chunk's
    assert _pcc(logits_1[:, -128:], logits_c) > 0.999
    for (k1, v1), (kc, vc) in zip(kv_1, kv_c):
        assert _pcc(k1, kc) > 0.9999 and _pcc(v1, vc) > 0.9999


def test_reference_is_torch_only(seed):
    """Reference purity: importing it must not drag in ttnn or transformers."""
    import subprocess
    import sys

    code = (
        "import sys; import models.demos.llama_3_1_8b.reference.model as m; "
        "assert 'ttnn' not in sys.modules, 'reference imported ttnn'; "
        "assert 'transformers' not in sys.modules, 'reference imported transformers'; print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(Path.cwd()))
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr
