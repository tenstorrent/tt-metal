# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Helpers to build the coqui XTTS-v2 GPT as ground truth for the reference PCC tests.

These drive coqui's REAL modules (from the vendored source under ../reference/TTS/) so our
self-contained reference can be checked against them. Not a test module (underscore-prefixed).

Requires:
  - the checkpoint (see reference/PROVENANCE.md) — location via $XTTS_CKPT or reference/weights/model.pth
  - the vendored coqui source at reference/TTS/ (gitignored; re-fetch per PROVENANCE.md)
"""

import os
import sys
import types

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
XTTS_DIR = os.path.dirname(_HERE)
REF_DIR = os.path.join(XTTS_DIR, "reference")
VENDORED_TTS = os.path.join(REF_DIR, "TTS")

# XTTS-v2 released config (see reference/xtts_gpt_ref.py)
START_AUDIO_TOKEN = 1024
STOP_AUDIO_TOKEN = 1025


def checkpoint_path():
    return os.environ.get("XTTS_CKPT", os.path.join(REF_DIR, "weights", "model.pth"))


def have_checkpoint():
    return os.path.isfile(checkpoint_path())


def have_vendored_coqui():
    return os.path.isdir(VENDORED_TTS)


def _install_tts_stub():
    class _Placeholder:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, state):
            try:
                self.__dict__.update(state)
            except Exception:
                pass

        def __call__(self, *a, **k):
            return self

    class _StubModule(types.ModuleType):
        def __getattr__(self, name):
            return _Placeholder

    for name in (
        "TTS", "TTS.tts", "TTS.tts.models", "TTS.tts.models.xtts",
        "TTS.tts.configs", "TTS.tts.configs.xtts_config", "TTS.tts.configs.shared_configs",
        "TTS.config", "TTS.config.shared_configs", "TTS.tts.layers", "TTS.utils",
    ):
        sys.modules.setdefault(name, _StubModule(name))


def _clear_tts_modules():
    for k in list(sys.modules):
        if k == "TTS" or k.startswith("TTS."):
            del sys.modules[k]


def load_gpt_weights(ckpt_path=None):
    """Extract gpt.* tensors (prefix stripped) with a throwaway TTS stub, then clear the stub."""
    ckpt_path = ckpt_path or checkpoint_path()
    _install_tts_stub()
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    weights = {k[len("gpt."):]: v for k, v in state.items() if k.startswith("gpt.") and hasattr(v, "shape")}
    _clear_tts_modules()
    return weights


def build_coqui_gpt(gpt_weights):
    """Instantiate coqui's real GPT (from vendored TTS/) and load the gpt.* weights into it."""
    _clear_tts_modules()  # ensure the real package imports, not a leftover stub

    # transformers 5.x removed isin_mps_friendly, which coqui's autoregressive.py imports.
    import transformers.pytorch_utils as ptu

    if not hasattr(ptu, "isin_mps_friendly"):
        ptu.isin_mps_friendly = lambda elements, test_elements: torch.isin(elements, test_elements)

    if REF_DIR not in sys.path:
        sys.path.insert(0, REF_DIR)
    from TTS.tts.layers.xtts.gpt import GPT

    gpt = GPT(
        layers=30, model_dim=1024, heads=16,
        max_text_tokens=402, max_mel_tokens=605, max_prompt_tokens=70,
        number_text_tokens=6681, num_audio_tokens=1026,
        start_audio_token=START_AUDIO_TOKEN, stop_audio_token=STOP_AUDIO_TOKEN,
        use_perceiver_resampler=True, code_stride_len=1024,
    )
    gpt.eval()
    gpt.load_state_dict(gpt_weights, strict=False)
    return gpt


@torch.no_grad()
def coqui_prefill_latents(gpt, inputs_embeds):
    """coqui's GPT-core forward: inputs_embeds -> GPT2 stack -> ln_f -> final_norm = latents."""
    out = gpt.gpt(inputs_embeds=inputs_embeds, return_dict=True, use_cache=False)
    return gpt.final_norm(out.last_hidden_state)


@torch.no_grad()
def coqui_cond(gpt, mel):
    """Block 1 ground truth: conditioning encoder output and the full style embedding.

    Returns (enc [b,1024,T], style [b,1024,32]). style = get_style_emb (perceiver output
    transposed), i.e. the gpt_cond_latent transposed — what feeds the GPT prefix."""
    enc = gpt.conditioning_encoder(mel)
    style = gpt.get_style_emb(mel)
    return enc, style


@torch.no_grad()
def coqui_decode(gpt, prefix_emb, n_steps, start=START_AUDIO_TOKEN, stop=STOP_AUDIO_TOKEN):
    """Greedy decode via coqui's real GPT2InferenceModel.forward (its own position logic)."""
    gpt.init_gpt_for_inference(kv_cache=True)
    gpt.gpt_inference.eval()
    infer = gpt.gpt_inference
    infer.store_prefix_emb(prefix_emb)
    P = prefix_emb.shape[1]

    gpt_inputs = torch.full((1, P + 1), 1, dtype=torch.long)
    gpt_inputs[:, -1] = start
    attn = torch.ones(1, P + 1, dtype=torch.long)
    out = infer(input_ids=gpt_inputs, past_key_values=None, attention_mask=attn, use_cache=True, return_dict=True)
    past = out.past_key_values
    logits = out.logits[:, -1:]
    code = int(logits.argmax(-1))
    codes, logits_list = [code], [logits]
    for m in range(1, n_steps):
        if code == stop:
            break
        inp = torch.tensor([[code]], dtype=torch.long)
        attn = torch.ones(1, P + 1 + m, dtype=torch.long)
        out = infer(input_ids=inp, past_key_values=past, attention_mask=attn, use_cache=True, return_dict=True)
        past = out.past_key_values
        logits = out.logits[:, -1:]
        code = int(logits.argmax(-1))
        codes.append(code)
        logits_list.append(logits)
    return torch.tensor(codes), torch.cat(logits_list, dim=1)
