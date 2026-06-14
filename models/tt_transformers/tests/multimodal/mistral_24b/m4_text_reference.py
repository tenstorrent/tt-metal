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

    model = Mistral4ForCausalLM(text_cfg).to(dtype).eval()
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

    handles = [
        layer.register_forward_hook(grab_in("hidden_in"), with_kwargs=True),
        layer.self_attn.register_forward_hook(grab_in("mla_in", capture_pos_emb=True), with_kwargs=True),
        layer.self_attn.register_forward_hook(grab_out("mla_out"), with_kwargs=True),
        layer.mlp.register_forward_hook(grab_in("moe_in"), with_kwargs=True),
        layer.mlp.register_forward_hook(grab_out("moe_out"), with_kwargs=True),
    ]
    with torch.no_grad():
        out = model(input_ids)
    for h in handles:
        h.remove()
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
