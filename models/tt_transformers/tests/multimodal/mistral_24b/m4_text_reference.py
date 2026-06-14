# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reference-parity harness for the Mistral-Small-4 (mistral4) TEXT core.

Builds an N-layer HF Mistral4ForCausalLM with REAL weights for the loaded layers
(filtered from the checkpoint + fp8-dequantized via dequantize_fp8_state_dict) — never
the full 226 GB model — and captures per-block golden activations (input/output of the
MLA attention and the MoE, plus final logits) for PCC-gating the TTNN blocks.

The text keys in the checkpoint are prefixed `language_model.` (the VLM wrapper); we strip
it to match a standalone Mistral4ForCausalLM. fp8 weights (MLA + experts + shared experts)
become bf16; router gate + norms are already bf16.
"""
import torch
from loguru import logger
from transformers import AutoConfig
from transformers.models.mistral4.modeling_mistral4 import Mistral4ForCausalLM

from models.tt_transformers.tt.load_checkpoints import dequantize_fp8_state_dict, load_hf_state_dict_filtered

_LM_PREFIX = "language_model."


def load_m4_weights(ckpt_dir, n_layers):
    """Load TT-side text weights DIRECTLY from the checkpoint (no HF model build/forward).

    Returns a state_dict with standalone keys (model.layers.N.*, model.embed_tokens.weight,
    model.norm.weight, lm_head.weight), fp8-dequantized to bf16. Fast — just the filtered
    safetensors read + dequant — so the TT test never waits on the ~40-min HF reference forward.
    """
    prefixes = [
        f"{_LM_PREFIX}model.embed_tokens.",
        f"{_LM_PREFIX}model.norm.",
        f"{_LM_PREFIX}lm_head.",
    ] + [f"{_LM_PREFIX}model.layers.{i}." for i in range(n_layers)]
    raw = load_hf_state_dict_filtered(ckpt_dir, prefixes)
    raw = {k[len(_LM_PREFIX) :]: v for k, v in raw.items()}
    return dequantize_fp8_state_dict(raw)


def get_cached_golden(ckpt_dir, n_layers, seed, seq_len, cache_dir="/tmp"):
    """Golden activations/logits, cached to disk — the HF reference (load + CPU forward) is built
    ONCE per (n_layers, seed, seq_len) and reused. Avoids paying the ~40-min 36-layer reference on
    every TT-side iteration. Bump CACHE_VER if capture_golden's contents change."""
    import os

    CACHE_VER = 1
    path = os.path.join(cache_dir, f"m4_golden_v{CACHE_VER}_{n_layers}L_s{seed}_q{seq_len}.pt")
    if os.path.exists(path):
        logger.info(f"golden cache HIT: {path}")
        return torch.load(path)
    logger.info(f"golden cache MISS: building {n_layers}-layer reference (slow, one-time)")
    model, cfg, _ = load_m4_text_reference(ckpt_dir, n_layers=n_layers)
    torch.manual_seed(seed)
    ids = torch.randint(0, cfg.vocab_size, (1, seq_len))
    g = capture_golden(model, ids)
    g["input_ids"] = ids
    torch.save(g, path)
    logger.info(f"golden cached -> {path}")
    return g


def load_m4_text_reference(ckpt_dir, n_layers=1, dtype=torch.bfloat16):
    """Return an n_layer Mistral4ForCausalLM with real weights for layers [0, n_layers).

    Loads only embed_tokens + layers[0:n_layers] + norm + lm_head (filtered), dequantizes
    fp8 to bf16, and loads into a config-truncated model. Returns (model, text_config).
    """
    text_cfg = AutoConfig.from_pretrained(ckpt_dir).text_config
    text_cfg.num_hidden_layers = n_layers

    prefixes = [
        f"{_LM_PREFIX}model.embed_tokens.",
        f"{_LM_PREFIX}model.norm.",
        f"{_LM_PREFIX}lm_head.",
    ] + [f"{_LM_PREFIX}model.layers.{i}." for i in range(n_layers)]
    raw = load_hf_state_dict_filtered(ckpt_dir, prefixes)
    # strip the VLM `language_model.` prefix to match standalone Mistral4ForCausalLM keys
    raw = {k[len(_LM_PREFIX) :]: v for k, v in raw.items()}
    sd = dequantize_fp8_state_dict(raw)
    logger.info(f"m4 text reference: {len(sd)} tensors after dequant (n_layers={n_layers})")

    # Construct in bf16 (default fp32 would allocate ~476 GB for the 36-layer reference and is
    # slow); we overwrite params with the loaded bf16 weights anyway and feed the reference's own
    # cos/sin to the TT side, so this is self-consistent.
    _prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = Mistral4ForCausalLM(text_cfg)
    finally:
        torch.set_default_dtype(_prev)
    model = model.eval()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # `missing` is expected to be empty for the loaded layers; tied lm_head may show up
    real_missing = [m for m in missing if "rotary_emb" not in m]
    logger.info(f"m4 text reference load: {len(real_missing)} missing, {len(unexpected)} unexpected")
    return model, text_cfg, (real_missing, unexpected)


def capture_golden(model, input_ids, layer_idx=0):
    """Run the reference and capture golden activations around layer `layer_idx`.

    Returns a dict with: hidden_in (into the decoder layer), mla_in/mla_out (around self_attn),
    moe_in/moe_out (around mlp), and logits. All on CPU, detached.
    """
    layer = model.model.layers[layer_idx]
    acts = {}

    def _hidden(args, kwargs):
        # hidden_states may be positional or kw depending on the call site
        if args:
            return args[0]
        return kwargs["hidden_states"]

    def grab_in(name, capture_pos_emb=False):
        def hook(_mod, args, kwargs, _out):
            acts[name] = _hidden(args, kwargs).detach().float().cpu()
            if capture_pos_emb and "position_embeddings" in kwargs:
                cos, sin = kwargs["position_embeddings"]
                acts["rope_cos"] = cos.detach().float().cpu()
                acts["rope_sin"] = sin.detach().float().cpu()

        return hook

    def grab_out(name):
        def hook(_mod, _args, _kwargs, out):
            t = out[0] if isinstance(out, tuple) else out
            acts[name] = t.detach().float().cpu()

        return hook

    # Capture the post-RoPE query/key/value states (locals inside Mistral4Attention) by wrapping
    # the module's attention interface — lets the TT SDPA+o_proj be PCC'd independently of RoPE.
    import transformers.models.mistral4.modeling_mistral4 as _mm

    _orig_eager = _mm.eager_attention_forward

    def _cap_attn(module, q, k, v, *a, **kw):
        if module is model.model.layers[layer_idx].self_attn:
            acts["q_states"] = q.detach().float().cpu()
            acts["k_states"] = k.detach().float().cpu()
            acts["value_states"] = v.detach().float().cpu()
        return _orig_eager(module, q, k, v, *a, **kw)

    _mm.eager_attention_forward = _cap_attn
    model.config._attn_implementation = "eager"
    for _l in model.model.layers:
        _l.self_attn.config._attn_implementation = "eager"

    sa = layer.self_attn
    handles = [
        layer.register_forward_hook(grab_in("hidden_in"), with_kwargs=True),
        layer.register_forward_hook(grab_out("layer_out"), with_kwargs=True),
        sa.register_forward_hook(grab_in("mla_in", capture_pos_emb=True), with_kwargs=True),
        sa.register_forward_hook(grab_out("mla_out"), with_kwargs=True),
        layer.mlp.register_forward_hook(grab_in("moe_in"), with_kwargs=True),
        layer.mlp.register_forward_hook(grab_out("moe_out"), with_kwargs=True),
        # MoE-internal boundaries: router logits, routed-experts output, shared-expert output
        layer.mlp.gate.register_forward_hook(grab_out("router_logits"), with_kwargs=True),
        layer.mlp.experts.register_forward_hook(grab_out("experts_out"), with_kwargs=True),
        layer.mlp.shared_experts.register_forward_hook(grab_out("shared_out"), with_kwargs=True),
        # MLA-internal module boundaries — let the TT MLA be PCC-gated sub-block by sub-block:
        # q projection chain, kv compress, kv expand, and the output projection (in = post-SDPA).
        sa.q_b_proj.register_forward_hook(grab_out("q_b_out"), with_kwargs=True),
        sa.kv_a_proj_with_mqa.register_forward_hook(grab_out("kv_a_out"), with_kwargs=True),
        sa.kv_b_proj.register_forward_hook(grab_out("kv_b_out"), with_kwargs=True),
        sa.o_proj.register_forward_hook(grab_in("o_proj_in"), with_kwargs=True),
    ]
    with torch.no_grad():
        out = model(input_ids)
    for h in handles:
        h.remove()
    _mm.eager_attention_forward = _orig_eager  # restore
    acts["logits"] = out.logits.detach().float().cpu()
    return acts


if __name__ == "__main__":
    import os

    ckpt = os.environ["HF_MODEL"]
    torch.manual_seed(0)
    model, cfg, (missing, unexpected) = load_m4_text_reference(ckpt, n_layers=1)
    assert not missing, f"unexpected missing keys: {missing[:10]}"
    ids = torch.randint(0, cfg.vocab_size, (1, 32))
    g = capture_golden(model, ids)
    for k, v in g.items():
        logger.info(f"  golden {k}: shape={tuple(v.shape)} finite={bool(torch.isfinite(v).all())}")
    torch.save({"input_ids": ids, **g}, "/tmp/m4_text_golden_L0.pt")
    logger.info("saved golden activations -> /tmp/m4_text_golden_L0.pt")
