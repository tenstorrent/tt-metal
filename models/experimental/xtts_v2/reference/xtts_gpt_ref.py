# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
CPU reference for the XTTS-v2 GPT transformer core (Block 3).

This is a self-contained (torch + transformers only) op-for-op reference, written from
the coqui source in ../reference/TTS/ (gpt.py + tortoise/autoregressive.py). It is NOT
the coqui model itself: it isolates exactly the sub-computation we will port to TTNN and
gives us a golden-tensor generator + a PCC check.

Block boundary (the tensor in -> out that the TTNN port must match):

    inputs_embeds [1, S, 1024]
        -> HF GPT2Model stack   (30 causal blocks, wpe nulled)   ckpt keys: gpt.gpt.h.*
        -> ln_f                 (GPT2's own final LayerNorm)      ckpt keys: gpt.gpt.ln_f.*
        -> final_norm           (XTTS's EXTRA LayerNorm)          ckpt keys: gpt.final_norm.*
        = latents [1, S, 1024]

Everything that BUILDS inputs_embeds (token + positional embeddings, concat with the
conditioning latent) and everything AFTER latents (mel_head -> logits) lives outside this
block and runs on the host. We reproduce just enough of it (make_synthetic_inputs_embeds)
to feed the block a realistic input without needing the conditioning block (Block 1) yet.

Why the config below and not gpt.py's defaults: the RELEASED coqui/XTTS-v2 config.json
overrides gpt.py's constructor defaults. Real values (config.json -> model_args):
    gpt_layers=30, gpt_n_model_channels=1024, gpt_n_heads=16,
    gpt_num_audio_tokens=1026 (start=1024, stop=1025),  gpt_number_text_tokens=6681,
    gpt_max_audio_tokens=605, gpt_max_text_tokens=402, gpt_max_prompt_tokens=70.
(gpt.py's 8194/8192/8193 audio-token defaults are for a different config and are WRONG here.)

Run (once weights + env exist) to regenerate goldens:
    python models/experimental/xtts_v2/reference/xtts_gpt_ref.py --ckpt /path/to/model.pth
"""

import argparse
import os
import sys
import types

import torch

# ----------------------------------------------------------------------------------------
# Config (from coqui/XTTS-v2 config.json + transformers GPT2 defaults)
# ----------------------------------------------------------------------------------------
N_EMBD = 1024  # gpt_n_model_channels
N_LAYER = 30  # gpt_layers
N_HEAD = 16  # gpt_n_heads
N_INNER = 4 * N_EMBD  # HF GPT2 default (mlp.c_fc is [1024, 4096]); build_hf_gpt_transformer leaves it default
ACT_FN = "gelu_new"  # HF GPT2 default (tanh-approx GELU)
LN_EPS = 1e-5  # HF GPT2 default layer_norm_epsilon

# Vocab / token ids (from config.json). Only used by the input/head helpers, not the block itself.
NUM_AUDIO_TOKENS = 1026
START_AUDIO_TOKEN = 1024
STOP_AUDIO_TOKEN = 1025
NUMBER_TEXT_TOKENS = 6681

# Positional-table sizes coqui derives in GPT.__init__ (used to bound synthetic inputs):
#   max_mel_tokens  = gpt_max_audio_tokens + 2 + max_conditioning_inputs(1) = 605 + 2 + 1 = 608
#   max_text_tokens = gpt_max_text_tokens  + 2                              = 402 + 2     = 404
MEL_POS_LEN = 608
TEXT_POS_LEN = 404
MAX_PROMPT_LEN = 70
# wpe is nulled, so n_positions only bounds a module we never use; keep it >= any real seq.
N_POSITIONS = MEL_POS_LEN + TEXT_POS_LEN + MAX_PROMPT_LEN  # 1082

DEFAULT_CKPT = os.environ.get(
    "XTTS_CKPT",
    os.path.join(os.path.dirname(__file__), "weights", "model.pth"),
)
GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "..", "golden", "gpt")


# ----------------------------------------------------------------------------------------
# Checkpoint loading
# ----------------------------------------------------------------------------------------
def _install_tts_stub():
    """The coqui model.pth is a full pickle that references TTS.* config classes as globals.
    Unpickling it normally requires the whole `coqui-tts` package installed. Instead we register
    lightweight stub modules so pickle can resolve those globals to harmless placeholders; we
    then keep only the tensor entries. This avoids the heavy dependency entirely."""

    class _Placeholder:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, state):
            # pickle may call this to restore a config object's __dict__; accept and ignore.
            try:
                self.__dict__.update(state)
            except Exception:
                pass

        def __call__(self, *a, **k):
            return self

    class _StubModule(types.ModuleType):
        def __getattr__(self, name):
            # Dunders must behave like a normal (empty) module, or libraries that inspect
            # sys.modules (e.g. transformers' lazy import of GPT2Model) choke on the placeholder.
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return _Placeholder

    for name in (
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
    ):
        sys.modules.setdefault(name, _StubModule(name))


def load_full_state(ckpt_path=DEFAULT_CKPT):
    """Read the checkpoint and return {name: tensor} for every tensor entry.

    Note: weights_only=False is required because the checkpoint pickles non-tensor config
    objects; only load checkpoints you trust. The TTS stub makes those objects resolvable."""
    _install_tts_stub()
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    return {k: v for k, v in state.items() if hasattr(v, "shape")}


def load_gpt_core_state(ckpt_path=DEFAULT_CKPT):
    """Slice out just the transformer-core tensors and re-key them to match a fresh
    transformers.GPT2Model (h.*, ln_f.*) plus XTTS's extra final_norm.*.

    Checkpoint prefixes:
        gpt.gpt.*          -> HF GPT2Model submodule  (strip 'gpt.gpt.')
        gpt.final_norm.*   -> XTTS extra LayerNorm     (strip 'gpt.')
    """
    full = load_full_state(ckpt_path)
    core = {}
    for k, v in full.items():
        if k.startswith("gpt.gpt."):
            core[k[len("gpt.gpt.") :]] = v
        elif k in ("gpt.final_norm.weight", "gpt.final_norm.bias"):
            core[k[len("gpt.") :]] = v
    return core


# ----------------------------------------------------------------------------------------
# Reference model
# ----------------------------------------------------------------------------------------
class _NullPositionEmbedding(torch.nn.Module):
    """Replacement for GPT2's wpe. XTTS adds its own learned positions OUTSIDE the
    transformer (text_pos_embedding / mel_pos_embedding), so coqui nulls GPT2's internal
    positional embedding (see tortoise.autoregressive.null_position_embeddings). Returning
    zeros makes `hidden = inputs_embeds + wpe(pos) = inputs_embeds`."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, position_ids):
        return torch.zeros(*position_ids.shape, self.dim)


def build_reference(ckpt_path=DEFAULT_CKPT):
    """Build the ground-truth GPT2 core + XTTS final_norm with the REAL weights."""
    from transformers import GPT2Config, GPT2Model

    cfg = GPT2Config(
        vocab_size=256,  # unused: we feed inputs_embeds, so wte is bypassed
        n_positions=N_POSITIONS,
        n_ctx=N_POSITIONS,
        n_embd=N_EMBD,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_inner=N_INNER,
        activation_function=ACT_FN,
        layer_norm_epsilon=LN_EPS,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
    )
    gpt = GPT2Model(cfg)
    gpt.wpe = _NullPositionEmbedding(N_EMBD)  # null GPT2's positional embedding (see above)

    core = load_gpt_core_state(ckpt_path)

    # XTTS's extra LayerNorm applied after GPT2's own ln_f.
    final_norm = torch.nn.LayerNorm(N_EMBD, eps=LN_EPS)
    final_norm.weight.data = core["final_norm.weight"].float()
    final_norm.bias.data = core["final_norm.bias"].float()

    # Drop final_norm (loaded separately) and any wte/wpe the checkpoint may carry: wte is
    # bypassed (we feed inputs_embeds) and wpe is nulled above, so neither should be loaded.
    skip = ("final_norm.", "wte", "wpe")
    gpt_sd = {k: v.float() for k, v in core.items() if not k.startswith(skip)}
    missing, unexpected = gpt.load_state_dict(gpt_sd, strict=False)
    # wte/wpe are legitimately absent from the checkpoint (bypassed / nulled); anything else missing is a bug.
    real_missing = [m for m in missing if not (m.startswith("wte") or m.startswith("wpe"))]
    assert not real_missing, f"unexpected missing keys: {real_missing[:8]}"
    assert not unexpected, f"unexpected keys in checkpoint: {unexpected[:8]}"

    gpt.eval()
    final_norm.eval()
    return gpt, final_norm


@torch.no_grad()
def reference_forward(gpt, final_norm, inputs_embeds):
    """inputs_embeds [1, S, 1024] -> (last_hidden [1,S,1024] post-ln_f, latents [1,S,1024]).

    For a single unpadded batch=1 sequence the attention mask is all-ones, so GPT2's effective
    attention is plain causal (no explicit mask needed)."""
    out = gpt(inputs_embeds=inputs_embeds, use_cache=False)
    last_hidden = out.last_hidden_state  # already includes GPT2's ln_f
    latents = final_norm(last_hidden)  # XTTS's extra normalization
    return last_hidden, latents


@torch.no_grad()
def make_synthetic_inputs_embeds(ckpt_path=DEFAULT_CKPT, n_text=16, n_mel=48, seed=0):
    """Build a deterministic, realistic inputs_embeds from the REAL embedding tables, so the
    GPT core can be tested in isolation (before the conditioning block exists). Mirrors coqui's
    construction: [text_emb + text_pos] ++ [mel_emb + mel_pos] -> [1, n_text+n_mel, 1024]."""
    full = load_full_state(ckpt_path)
    text_emb_w = full["gpt.text_embedding.weight"].float()  # (6681, 1024)
    mel_emb_w = full["gpt.mel_embedding.weight"].float()  # (1026, 1024)
    text_pos = full["gpt.text_pos_embedding.emb.weight"].float()  # (404, 1024)
    mel_pos = full["gpt.mel_pos_embedding.emb.weight"].float()  # (608, 1024)

    g = torch.Generator().manual_seed(seed)
    text_ids = torch.randint(0, text_emb_w.shape[0], (n_text,), generator=g)
    mel_ids = torch.randint(0, mel_emb_w.shape[0], (n_mel,), generator=g)

    text_e = text_emb_w[text_ids] + text_pos[:n_text]
    mel_e = mel_emb_w[mel_ids] + mel_pos[:n_mel]
    return torch.cat([text_e, mel_e], dim=0).unsqueeze(0).contiguous()  # [1, S, 1024]


def pcc(a, b):
    """Pearson correlation over flattened tensors — the standard tt-metal accuracy metric."""
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


# ----------------------------------------------------------------------------------------
# Decode / autoregressive generation (Block 3, decode path)
# ----------------------------------------------------------------------------------------
def load_gen_head(ckpt_path=DEFAULT_CKPT):
    """The generation-head tensors that live OUTSIDE the transformer block: they embed tokens
    on the way in (mel_embedding + mel_pos_embedding) and project latents to logits (mel_head).
    In the real pipeline these run on host; the transformer block is everything between them."""
    full = load_full_state(ckpt_path)
    return {
        "mel_emb": full["gpt.mel_embedding.weight"].float(),  # (1026, 1024)
        "mel_pos": full["gpt.mel_pos_embedding.emb.weight"].float(),  # (608, 1024)
        "mel_head_w": full["gpt.mel_head.weight"].float(),  # (1026, 1024)
        "mel_head_b": full["gpt.mel_head.bias"].float(),  # (1026,)
        "text_emb": full["gpt.text_embedding.weight"].float(),  # (6681, 1024)
        "text_pos": full["gpt.text_pos_embedding.emb.weight"].float(),  # (404, 1024)
    }


@torch.no_grad()
def make_synthetic_prefix(heads, n_text=8, seed=0):
    """A deterministic text-only prefix [1, n_text, 1024] from the real text tables. Stands in
    for the (conditioning-latent ++ text) prefix until Block 1 exists — the only requirement for
    validating decode is that reference and coqui start from the IDENTICAL prefix."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, heads["text_emb"].shape[0], (n_text,), generator=g)
    return (heads["text_emb"][ids] + heads["text_pos"][:n_text]).unsqueeze(0).contiguous()


@torch.no_grad()
def reference_generate(
    gpt, final_norm, heads, prefix_emb, max_new=24, start_token=START_AUDIO_TOKEN, stop_token=STOP_AUDIO_TOKEN
):
    """Greedy autoregressive decode with KV-cache, mirroring coqui's GPT2InferenceModel:

        prefill:  [prefix_emb, mel_emb[start] + mel_pos[0]]  -> latent -> mel_head -> code_0
        step m:   mel_emb[code_{m-1}] + mel_pos[m]  (+ KV cache) -> latent -> code_m
        stop when code == stop_token or max_new steps reached.

    Positions come ONLY from mel_pos (GPT2's wpe is nulled), so mel_pos is the sole position
    signal: the start token is position 0 and the m-th forward uses position m."""
    mel_emb, mel_pos = heads["mel_emb"], heads["mel_pos"]
    mh_w, mh_b = heads["mel_head_w"], heads["mel_head_b"]

    def head(latent):  # [1,1,1024] -> [1,1,1026]
        return latent @ mh_w.t() + mh_b

    start_emb = (mel_emb[start_token] + mel_pos[0]).view(1, 1, -1)
    out = gpt(inputs_embeds=torch.cat([prefix_emb, start_emb], dim=1), use_cache=True)
    past = out.past_key_values
    latent = final_norm(out.last_hidden_state[:, -1:])
    logits = head(latent)
    code = int(logits.argmax(-1))

    codes, latents, all_logits = [code], [latent], [logits]
    for m in range(1, max_new):
        if code == stop_token:
            break
        emb = (mel_emb[code] + mel_pos[m]).view(1, 1, -1)
        out = gpt(inputs_embeds=emb, past_key_values=past, use_cache=True)
        past = out.past_key_values
        latent = final_norm(out.last_hidden_state[:, -1:])
        logits = head(latent)
        code = int(logits.argmax(-1))
        codes.append(code)
        latents.append(latent)
        all_logits.append(logits)

    return {
        "codes": torch.tensor(codes),  # [T]
        "latents": torch.cat(latents, dim=1),  # [1, T, 1024]
        "logits": torch.cat(all_logits, dim=1),  # [1, T, 1026]
        "prefix_emb": prefix_emb,  # [1, P, 1024]
    }


def main():
    """Regenerate goldens in the real inference order: DECODE (generate codes), then PREFILL
    over those codes to get the latents that feed the vocoder — mirroring Xtts.inference(),
    which calls gpt.generate() then gpt(..., return_latent=True) over the generated codes."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=GOLDEN_DIR)
    ap.add_argument("--n-text", type=int, default=8, help="prefix (text) length for the decode->prefill chain")
    ap.add_argument("--max-new", type=int, default=24, help="max audio codes to generate")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"[ref] loading GPT core from {args.ckpt}")
    gpt, final_norm = build_reference(args.ckpt)
    heads = load_gen_head(args.ckpt)

    # ---- 1) DECODE: generate audio codes from a prefix (mirrors Xtts: gpt.generate) ----
    prefix = make_synthetic_prefix(heads, n_text=args.n_text)
    gen = reference_generate(gpt, final_norm, heads, prefix, max_new=args.max_new)
    codes = gen["codes"]
    print(f"[ref] decode: generated {codes.numel()} codes: {codes.tolist()}")

    gen_dir = os.path.join(args.out, "generate")
    os.makedirs(gen_dir, exist_ok=True)
    for k in ("prefix_emb", "codes", "logits", "latents"):
        torch.save(gen[k], os.path.join(gen_dir, f"{k}.pt"))
    torch.save(
        {"max_new": args.max_new, "n_text": args.n_text, "start": START_AUDIO_TOKEN, "stop": STOP_AUDIO_TOKEN},
        os.path.join(gen_dir, "meta.pt"),
    )
    print(f"[ref] wrote decode goldens to {gen_dir}")

    # ---- 2) PREFILL over the generated codes (mirrors Xtts: gpt(..., return_latent=True)) ----
    # Rebuild the exact mel sequence decode fed through the transformer:
    #   tokens = [start, code_0, ..., code_{T-2}]  at mel positions 0..T-1
    # A single cache-free forward then yields the latents; the mel-position slice is what the
    # HiFi-GAN vocoder (Block 4) consumes. GPT2's wpe is nulled, so mel_pos is the only position.
    mel_emb, mel_pos = heads["mel_emb"], heads["mel_pos"]
    T = codes.numel()
    mel_ids = torch.cat([torch.tensor([START_AUDIO_TOKEN], dtype=codes.dtype), codes[:-1]])  # [start, code_0..code_{T-2}]
    mel_seq = mel_emb[mel_ids] + mel_pos[:T]  # [T, 1024]
    inputs_embeds = torch.cat([prefix, mel_seq.unsqueeze(0)], dim=1)  # [1, P+T, 1024]
    last_hidden, latents = reference_forward(gpt, final_norm, inputs_embeds)
    vocoder_latents = latents[:, prefix.shape[1] :]  # [1, T, 1024] -> feeds Block 4

    # Bonus: single prefill vs incremental decode must agree at the mel positions (KV-cache check).
    consistency = pcc(vocoder_latents, gen["latents"])
    print(f"[ref] prefill over codes: latents {tuple(latents.shape)}; decode-vs-prefill latent PCC = {consistency:.6f}")

    torch.save(inputs_embeds, os.path.join(args.out, "inputs_embeds.pt"))
    torch.save(last_hidden, os.path.join(args.out, "last_hidden_state.pt"))
    torch.save(latents, os.path.join(args.out, "latents.pt"))
    torch.save(vocoder_latents, os.path.join(args.out, "vocoder_latents.pt"))
    torch.save(
        {"prefix_len": prefix.shape[1], "n_codes": int(T), "n_embd": N_EMBD, "n_layer": N_LAYER, "n_head": N_HEAD},
        os.path.join(args.out, "meta.pt"),
    )
    print(f"[ref] wrote prefill goldens to {args.out}")


if __name__ == "__main__":
    main()
