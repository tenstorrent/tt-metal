# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Export Voxtral-TTS's Block 1 backbone as a HuggingFace-format model directory.

WHY THIS EXISTS: `tt_transformers` only accepts HF checkpoints -- `model_config.py:588` raises
unless `HF_MODEL` is set, and the Meta `consolidated.00.pth` path is vestigial. Our checkpoint is
Mistral-native (`consolidated.safetensors` + `params.json`), so bridging it needs both a
`config.json` and a rename of every tensor. Nothing here changes numbers except the deliberate
RoPE permute below.

    python models/experimental/voxtral_tts/scripts/export_backbone_hf.py --out <dir>
    HF_MODEL=<dir> python ...            # then tt_transformers can load it

THE ROPE PERMUTE IS THE ONLY SUBTLE PART. Mistral-native ("meta") weights store q/k for
interleaved-pair RoPE (r1,i1,r2,i2,...). HF permutes them so `rotate_half` works on split halves
(r1,r2,...,i1,i2,...). The two are a matched pair: meta weights need meta RoPE, HF weights need HF
RoPE, and mixing them is silently wrong rather than an error.

`tt_transformers` always wants meta weights internally and picks its RoPE to match
(`model_config.py:3127`):

    use_hf_rope=False (DEFAULT) -> convert_hf_to_meta            -> APPLIES the un-permute
    use_hf_rope=True            -> convert_hf_to_meta_no_qkv_permute -> no permute

So if we wrote our already-meta weights out unpermuted, the default path would un-permute them
again and corrupt them. We therefore apply the meta->HF permute here, which the loader's
un-permute then exactly reverses. `test_roundtrip` asserts that, because getting it wrong produces
plausible-looking garbage rather than a crash.

Two structural notes about this model:
  * `n_heads * head_dim` (4096) != `dim` (3072), so wq/wo are NOT square. `head_dim` must be
    stated explicitly in config.json -- HF would otherwise derive dim // n_heads = 96.
  * Embeddings are tied and live under `mm_audio_embeddings.tok_embeddings.weight`. We emit them
    as `model.embed_tokens.weight` and set `tie_word_embeddings`, and deliberately emit NO
    `lm_head` -- Block 2 consumes the post-final-norm hidden state and we never produce text
    tokens, so that 402M-parameter tied head is pure overhead.
"""

import argparse
import json
import os

import torch

from models.experimental.voxtral_tts.reference.voxtral_common_ref import DEFAULT_CKPT, SafeTensors

DEFAULT_PARAMS = os.path.join(os.path.dirname(DEFAULT_CKPT), "params.json")

# ours (Mistral-native) -> HF, per tt_transformers' own table (load_checkpoints.py:256-268)
LAYER_MAP = {
    "attention.wq.weight": "self_attn.q_proj.weight",
    "attention.wk.weight": "self_attn.k_proj.weight",
    "attention.wv.weight": "self_attn.v_proj.weight",
    "attention.wo.weight": "self_attn.o_proj.weight",
    "attention_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "feed_forward.w1.weight": "mlp.gate_proj.weight",
    "feed_forward.w2.weight": "mlp.down_proj.weight",
    "feed_forward.w3.weight": "mlp.up_proj.weight",
}


def meta_to_hf_permute(t: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Inverse of tt_transformers' `reverse_permute` (load_checkpoints.py:891).

    Their un-permute is
        view(n_heads, 2, dim1//n_heads//2, dim2).transpose(1, 2).reshape(dim1, dim2)
    so ours must be
        view(n_heads, dim1//n_heads//2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)
    which is exactly their `permute`. Applied to wq and wk ONLY -- wv carries no RoPE.
    """
    d1, d2 = t.shape
    return t.view(n_heads, d1 // n_heads // 2, 2, d2).transpose(1, 2).reshape(d1, d2)


def hf_to_meta_permute(t: torch.Tensor, n_heads: int) -> torch.Tensor:
    """tt_transformers' `reverse_permute`, replicated here so the round-trip can be asserted."""
    d1, d2 = t.shape
    return t.view(n_heads, 2, d1 // n_heads // 2, d2).transpose(1, 2).reshape(d1, d2)


def build_config(params: dict) -> dict:
    """params.json -> HF Mistral config.json. Schema copied from
    models/tt_transformers/model_params/Mistral-7B-Instruct-v0.3/config.json."""
    return {
        "architectures": ["MistralForCausalLM"],
        "model_type": "mistral",
        "hidden_size": params["dim"],
        "num_hidden_layers": params["n_layers"],
        "num_attention_heads": params["n_heads"],
        "num_key_value_heads": params["n_kv_heads"],
        # MUST be explicit: 32 * 128 = 4096 != dim 3072, so the usual dim // n_heads = 96 is wrong.
        # tt_transformers honours this (model_config.py:2678).
        "head_dim": params["head_dim"],
        "intermediate_size": params["hidden_dim"],
        "rms_norm_eps": params["norm_eps"],
        "rope_theta": params["rope_theta"],
        "vocab_size": params["vocab_size"],
        "max_position_embeddings": params["max_position_embeddings"],
        "tie_word_embeddings": params["tied_embeddings"],
        "sliding_window": None,  # this model is fully causal; no window
        "hidden_act": "silu",
        "attention_dropout": 0.0,
        "use_cache": True,
        "torch_dtype": "bfloat16",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "transformers_version": "4.38.0",
        "_name_or_path": "voxtral-tts-backbone",
    }


def export(ckpt: str, params_path: str, out_dir: str, dtype=torch.bfloat16) -> dict:
    with open(params_path) as f:
        params = json.load(f)
    n_layers, n_heads, n_kv = params["n_layers"], params["n_heads"], params["n_kv_heads"]
    st = SafeTensors(ckpt)
    os.makedirs(out_dir, exist_ok=True)

    out = {}
    out["model.embed_tokens.weight"] = st.get("mm_audio_embeddings.tok_embeddings.weight", torch.float32)
    out["model.norm.weight"] = st.get("norm.weight", torch.float32)
    for i in range(n_layers):
        for ours, hf in LAYER_MAP.items():
            t = st.get(f"layers.{i}.{ours}", torch.float32)
            if ours == "attention.wq.weight":
                t = meta_to_hf_permute(t, n_heads)
            elif ours == "attention.wk.weight":
                t = meta_to_hf_permute(t, n_kv)  # k has n_kv_heads rows, not n_heads
            out[f"model.layers.{i}.{hf}"] = t
    # No lm_head on purpose -- see the module docstring.

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(build_config(params), f, indent=2)
    from safetensors.torch import save_file

    save_file({k: v.to(dtype).contiguous() for k, v in out.items()},
              os.path.join(out_dir, "model.safetensors"))
    return out


def test_roundtrip(ckpt: str = DEFAULT_CKPT, params_path: str = DEFAULT_PARAMS) -> None:
    """The permute must be EXACTLY reversed by tt_transformers' un-permute.

    Checked bit-exactly in fp32 on layer 0's wq and wk, because a wrong permute does not crash --
    it produces a model that runs and generates plausible-sounding nonsense.
    """
    with open(params_path) as f:
        params = json.load(f)
    st = SafeTensors(ckpt)
    for name, heads in (("wq", params["n_heads"]), ("wk", params["n_kv_heads"])):
        meta = st.get(f"layers.0.attention.{name}.weight", torch.float32)
        back = hf_to_meta_permute(meta_to_hf_permute(meta, heads), heads)
        assert torch.equal(meta, back), f"{name}: permute round-trip is not identity"
        # and the permute must actually DO something, or we would be testing nothing
        assert not torch.equal(meta, meta_to_hf_permute(meta, heads)), f"{name}: permute is a no-op"
    print("  permute round-trip: bit-exact for wq and wk, and non-trivial")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/experimental/voxtral_tts/reference/weights/hf_backbone")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--params", default=DEFAULT_PARAMS)
    ap.add_argument("--test-only", action="store_true")
    a = ap.parse_args()
    test_roundtrip(a.ckpt, a.params)
    if not a.test_only:
        d = export(a.ckpt, a.params, a.out)
        print(f"  wrote {len(d)} tensors + config.json to {a.out}")
