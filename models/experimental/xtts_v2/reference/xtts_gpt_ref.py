# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
CPU reference for the XTTS-v2 GPT transformer core (Block 3), plus a golden-tensor
generator for the TTNN PCC test.

Block boundary under test (see CLAUDE_XTTS_GPT.md):

    inputs_embeds [1, S, 1024]
        -> GPT2 stack (30 blocks, causal)      (gpt.gpt.h.*)
        -> ln_f                                (gpt.gpt.ln_f)      [GPT2's final LN]
        -> final_norm                          (gpt.final_norm)   [XTTS's extra LN]
        = latents [1, S, 1024]

The transformer core is HuggingFace GPT2Model with the built-in positional embedding
(wpe) nulled to zeros, so hidden = inputs_embeds (no positional addition inside the
transformer). For a single unpadded sequence (batch=1) the attention mask is all-ones,
so the effective attention is standard GPT2 causal.

Reference weights: the real coqui/XTTS-v2 checkpoint (model.pth). We only touch the
transformer-core tensors (gpt.gpt.h.*, gpt.gpt.ln_f.*, gpt.final_norm.*).

Run to (re)generate goldens:
    python models/experimental/xtts_v2/reference/xtts_gpt_ref.py --ckpt /localdev/acicovic/xtts_ref/model.pth
"""

import argparse
import os
import sys
import types

import torch

# ---- XTTS-v2 GPT core config (from coqui config.json + build_hf_gpt_transformer) ----
GPT_CONFIG = dict(
    n_embd=1024,
    n_layer=30,
    n_head=16,
    n_inner=4096,  # HF GPT2 default = 4*n_embd; matches mlp.c_fc [1024,4096]
    activation_function="gelu_new",  # HF GPT2 default (tanh-approx GELU)
    layer_norm_epsilon=1e-5,  # HF GPT2 default
    n_positions=2048,  # only bounds the (nulled) wpe; irrelevant to the forward
    vocab_size=256,  # unused (wte bypassed via inputs_embeds)
)

DEFAULT_CKPT = "/localdev/acicovic/xtts_ref/model.pth"
GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "..", "golden", "gpt")


# --------------------------------------------------------------------------------------
# Checkpoint loading (stub the TTS package so pickle can resolve config globals)
# --------------------------------------------------------------------------------------
def _install_tts_stub():
    class _Any:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, s):
            try:
                self.__dict__.update(s)
            except Exception:
                pass

        def __call__(self, *a, **k):
            return self

    class _StubMod(types.ModuleType):
        def __getattr__(self, name):
            return _Any

    for name in [
        "TTS",
        "TTS.tts",
        "TTS.tts.models",
        "TTS.tts.models.xtts",
        "TTS.tts.configs",
        "TTS.tts.configs.xtts_config",
        "TTS.config",
        "TTS.config.shared_configs",
        "TTS.tts.layers",
        "TTS.utils",
    ]:
        sys.modules.setdefault(name, _StubMod(name))


def load_full_state(ckpt_path):
    _install_tts_stub()
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    return {k: v for k, v in state.items() if hasattr(v, "shape")}


def load_gpt_core_state(ckpt_path=DEFAULT_CKPT):
    """Return the transformer-core weights with 'gpt.gpt.'/'gpt.' prefixes stripped.

    Keys returned:
      - HF GPT2Model names: h.{i}.*, ln_f.*
      - final_norm.weight / final_norm.bias   (XTTS's extra LayerNorm)
    """
    full = load_full_state(ckpt_path)
    core = {}
    for k, v in full.items():
        if k.startswith("gpt.gpt."):  # the HF GPT2Model submodule
            core[k[len("gpt.gpt.") :]] = v
        elif k in ("gpt.final_norm.weight", "gpt.final_norm.bias"):
            core[k[len("gpt.") :]] = v
    return core


# --------------------------------------------------------------------------------------
# Reference model
# --------------------------------------------------------------------------------------
class _NullPositionEmbedding(torch.nn.Module):
    """Replaces GPT2's wpe: returns zeros, matching coqui null_position_embeddings."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, position_ids):
        return torch.zeros(*position_ids.shape, self.dim)


def build_reference(ckpt_path=DEFAULT_CKPT):
    """Build the ground-truth HF GPT2 core + XTTS final_norm with real weights."""
    from transformers import GPT2Config, GPT2Model

    cfg = GPT2Config(
        n_embd=GPT_CONFIG["n_embd"],
        n_layer=GPT_CONFIG["n_layer"],
        n_head=GPT_CONFIG["n_head"],
        n_inner=GPT_CONFIG["n_inner"],
        activation_function=GPT_CONFIG["activation_function"],
        layer_norm_epsilon=GPT_CONFIG["layer_norm_epsilon"],
        n_positions=GPT_CONFIG["n_positions"],
        n_ctx=GPT_CONFIG["n_positions"],
        vocab_size=GPT_CONFIG["vocab_size"],
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
    )
    gpt = GPT2Model(cfg)
    gpt.wpe = _NullPositionEmbedding(GPT_CONFIG["n_embd"])  # null positional embedding

    core = load_gpt_core_state(ckpt_path)
    final_norm = torch.nn.LayerNorm(GPT_CONFIG["n_embd"], eps=GPT_CONFIG["layer_norm_epsilon"])
    final_norm.weight.data = core["final_norm.weight"].float()
    final_norm.bias.data = core["final_norm.bias"].float()

    gpt_sd = {k: v for k, v in core.items() if not k.startswith("final_norm.")}
    missing, unexpected = gpt.load_state_dict(gpt_sd, strict=False)
    # wte/wpe are expected-missing (bypassed / nulled); nothing else should be missing.
    real_missing = [m for m in missing if not (m.startswith("wte") or m.startswith("wpe"))]
    assert not real_missing, f"unexpected missing keys: {real_missing[:8]}"
    assert not unexpected, f"unexpected keys: {unexpected[:8]}"

    gpt.eval()
    final_norm.eval()
    return gpt, final_norm


def reference_forward(gpt, final_norm, inputs_embeds):
    """inputs_embeds [1,S,1024] -> (last_hidden_state, latents)."""
    with torch.no_grad():
        out = gpt(inputs_embeds=inputs_embeds, use_cache=False)
        last_hidden = out.last_hidden_state  # after ln_f
        latents = final_norm(last_hidden)
    return last_hidden, latents


def make_golden_input(ckpt_path=DEFAULT_CKPT, n_text=16, n_mel=48, seed=0):
    """Construct a realistic seeded inputs_embeds from the real embedding tables:
    [text_emb + text_pos] ++ [mel_emb + mel_pos]  ->  [1, n_text+n_mel, 1024]."""
    full = load_full_state(ckpt_path)
    text_emb_w = full["gpt.text_embedding.weight"].float()  # (6681,1024)
    mel_emb_w = full["gpt.mel_embedding.weight"].float()  # (1026,1024)
    text_pos = full["gpt.text_pos_embedding.emb.weight"].float()  # (404,1024)
    mel_pos = full["gpt.mel_pos_embedding.emb.weight"].float()  # (608,1024)

    g = torch.Generator().manual_seed(seed)
    text_ids = torch.randint(0, text_emb_w.shape[0], (n_text,), generator=g)
    mel_ids = torch.randint(0, mel_emb_w.shape[0], (n_mel,), generator=g)

    text_e = text_emb_w[text_ids] + text_pos[:n_text]
    mel_e = mel_emb_w[mel_ids] + mel_pos[:n_mel]
    inputs_embeds = torch.cat([text_e, mel_e], dim=0).unsqueeze(0)  # [1,S,1024]
    return inputs_embeds.contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=GOLDEN_DIR)
    ap.add_argument("--n-text", type=int, default=16)
    ap.add_argument("--n-mel", type=int, default=48)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"[ref] loading reference from {args.ckpt}")
    gpt, final_norm = build_reference(args.ckpt)
    print("[ref] building golden input")
    inputs_embeds = make_golden_input(args.ckpt, args.n_text, args.n_mel)
    print(f"[ref] inputs_embeds shape={tuple(inputs_embeds.shape)} " f"std={inputs_embeds.std().item():.4f}")
    last_hidden, latents = reference_forward(gpt, final_norm, inputs_embeds)

    torch.save(inputs_embeds, os.path.join(args.out, "inputs_embeds.pt"))
    torch.save(last_hidden, os.path.join(args.out, "last_hidden_state.pt"))
    torch.save(latents, os.path.join(args.out, "latents.pt"))
    torch.save({"n_text": args.n_text, "n_mel": args.n_mel, "config": GPT_CONFIG}, os.path.join(args.out, "meta.pt"))
    print(
        f"[ref] latents shape={tuple(latents.shape)} "
        f"mean={latents.mean().item():.5f} std={latents.std().item():.5f}"
    )
    print(f"[ref] wrote goldens to {args.out}")


if __name__ == "__main__":
    main()
