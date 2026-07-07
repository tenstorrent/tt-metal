# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Reference (pure-PyTorch) implementation of a single XTTS-v2 GPT block.

XTTS-v2's autoregressive core is a HuggingFace GPT-2 transformer (``gpt.gpt``)
built with a custom config: 30 identical decoder blocks, 1024 model channels,
16 heads, 4096 FFN. A single one of those repeating blocks is exactly a
``transformers`` ``GPT2Block`` (causal self-attention + MLP with two residual
LayerNorms).

The reference weights come straight from the upstream checkpoint published at
https://huggingface.co/coqui/XTTS-v2 (``model.pth``). We deliberately avoid
depending on the ``coqui-tts`` package (which pins an older ``transformers`` and
would break other tt-metal models) — instead we download the checkpoint with
``huggingface_hub`` and unpickle only the tensor state dict, stubbing out the
coqui config objects that are also pickled inside it.
"""

import importlib.abc
import importlib.machinery
import sys
import types

import torch
from torch import nn
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

HF_REPO_ID = "coqui/XTTS-v2"
CHECKPOINT_FILE = "model.pth"

# GPT-2 backbone hyper-parameters, read from coqui/XTTS-v2 config.json
# (model_args.gpt_layers / gpt_n_model_channels / gpt_n_heads).
NUM_LAYERS = 30
HIDDEN_SIZE = 1024
NUM_HEADS = 16
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS  # 64
FFN_SIZE = 4 * HIDDEN_SIZE  # 4096 (GPT2 n_inner default)
LAYER_NORM_EPS = 1e-5

# Real sequence-length limits, read off the checkpoint's learned position
# embeddings (gpt.text_pos_embedding=404, gpt.mel_pos_embedding=608). At inference
# the GPT runs on the concatenated [text] + [mel] stream, so coqui sizes the GPT-2
# causal backbone to n_positions = text + mel.
MAX_TEXT_POS = 404  # gpt_max_text_tokens (402) + 2
MAX_MEL_POS = 608  # gpt_max_audio_tokens (605) + 3
MAX_GPT_SEQ_LEN = MAX_TEXT_POS + MAX_MEL_POS  # 1012 — full GPT context
MAX_POSITIONS = MAX_GPT_SEQ_LEN  # sizes the causal mask; must cover any tested seq_len


# ---------------------------------------------------------------------------
# checkpoint loading (no coqui-tts dependency)
# ---------------------------------------------------------------------------
class _StubObject:
    """Placeholder for any coqui-TTS object pickled inside the checkpoint.

    We only care about the tensor state dict, so config/optimizer objects are
    reconstructed as inert stubs (their state is captured but never used).
    """

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class _FakeTTSFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Fabricate ``TTS.*`` modules on demand so unpickling resolves classes."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "TTS" or fullname.startswith("TTS."):
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__path__ = []  # mark as a package so submodules can be imported

        def __getattr__(name, _module=spec.name):
            return type(name, (_StubObject,), {"__module__": _module})

        module.__getattr__ = __getattr__
        return module

    def exec_module(self, module):  # nothing to execute for a stub module
        pass


def _install_tts_stub():
    if not any(isinstance(f, _FakeTTSFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _FakeTTSFinder())


# Non-block GPT weights the model uses (block weights match ``gpt.gpt.h.``).
_GPT_SCALAR_WEIGHTS = frozenset(
    {
        "gpt.gpt.ln_f.weight",
        "gpt.gpt.ln_f.bias",
        "gpt.text_embedding.weight",
        "gpt.mel_embedding.weight",
        "gpt.text_pos_embedding.emb.weight",
        "gpt.mel_pos_embedding.emb.weight",
        "gpt.final_norm.weight",
        "gpt.final_norm.bias",
        "gpt.text_head.weight",
        "gpt.text_head.bias",
        "gpt.mel_head.weight",
        "gpt.mel_head.bias",
    }
)


def _is_gpt_weight(key):
    """True for the checkpoint keys the XTTS GPT decoder model actually loads."""
    return key.startswith("gpt.gpt.h.") or key in _GPT_SCALAR_WEIGHTS


def load_xtts_state_dict():
    """Download coqui/XTTS-v2 ``model.pth`` and return its tensor state dict.

    The checkpoint is a dict ``{"config", "model", "optimizer", ...}``; the GPT
    weights live under ``model`` with keys like ``gpt.gpt.h.{i}.attn.c_attn.weight``.
    First call downloads ~1.9 GB and caches it under the HF hub cache.
    """
    from huggingface_hub import hf_hub_download

    _install_tts_stub()
    checkpoint_path = hf_hub_download(repo_id=HF_REPO_ID, filename=CHECKPOINT_FILE)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    tensors = {k: v for k, v in state_dict.items() if torch.is_tensor(v)}

    # Verify only the weights the GPT model actually consumes (the 30 decoder
    # blocks + ln_f, the text/mel token + position embeddings, final_norm, and
    # the two heads) — not the whole ~1.9 GB checkpoint. Print name, shape,
    # dtype and size for each.
    total_params, total_bytes = 0, 0
    used = {k: v for k, v in tensors.items() if _is_gpt_weight(k)}
    print(f"[load_xtts_state_dict] {len(used)} GPT weights used (of {len(tensors)} tensors in checkpoint):")
    for name, t in used.items():
        n_elem = t.numel()
        n_bytes = n_elem * t.element_size()
        total_params += n_elem
        total_bytes += n_bytes
        print(f"  {name:<55} shape={tuple(t.shape)} dtype={t.dtype} " f"params={n_elem:,} size={n_bytes / 1e6:.2f} MB")
    print(f"[load_xtts_state_dict] total GPT: {total_params:,} params, {total_bytes / 1e6:.2f} MB")

    return tensors


# ---------------------------------------------------------------------------
# reference module
# ---------------------------------------------------------------------------
def build_gpt2_config():
    """GPT2Config matching the XTTS-v2 GPT backbone."""
    return GPT2Config(
        n_positions=MAX_POSITIONS,
        n_embd=HIDDEN_SIZE,
        n_layer=NUM_LAYERS,
        n_head=NUM_HEADS,
        n_inner=FFN_SIZE,
        activation_function="gelu_new",
        layer_norm_epsilon=LAYER_NORM_EPS,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        # XTTS's GPT is a causal autoregressive decoder. A standalone GPT2Block
        # does NOT build a causal mask by itself (GPT2Model normally does), so we
        # use the eager attention path and pass an explicit causal mask in
        # forward — see build_causal_mask.
        attn_implementation="eager",
    )


def build_causal_mask(seq_len, dtype=torch.float32):
    """Additive causal attention mask of shape ``[1, 1, seq_len, seq_len]``.

    0 on/below the diagonal, a large negative value above it — so each position
    attends only to itself and earlier positions.
    """
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype)
    return torch.triu(mask, diagonal=1).view(1, 1, seq_len, seq_len)


class XttsReferenceGptBlock(nn.Module):
    """One XTTS GPT decoder block, wrapping a HF ``GPT2Block``.

    The wrapper (a) applies causal masking — the XTTS GPT is an autoregressive
    decoder — and (b) normalizes the return value to a bare hidden-states tensor
    of shape ``[batch, seq, hidden]``, shielding callers from the transformers
    API drift where ``GPT2Block.forward`` returns a tuple on some versions and a
    plain tensor on others.
    """

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.block = GPT2Block(config, layer_idx=layer_idx)

    def forward(self, hidden_states):
        mask = build_causal_mask(hidden_states.shape[1], hidden_states.dtype)
        out = self.block(hidden_states, attention_mask=mask)
        return out[0] if isinstance(out, tuple) else out


def reference_gpt_block(state_dict, layer_idx=0):
    """Build one XTTS GPT decoder block with real weights, in eval mode.

    Args:
        state_dict: full checkpoint state dict from :func:`load_xtts_state_dict`.
        layer_idx: which of the 30 repeating blocks to instantiate.
    """
    config = build_gpt2_config()
    module = XttsReferenceGptBlock(config, layer_idx=layer_idx)

    prefix = f"gpt.gpt.h.{layer_idx}."
    block_state = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
    _, unexpected = module.block.load_state_dict(block_state, strict=False)
    # GPT2Attention registers non-persistent causal-mask buffers that are absent
    # from the checkpoint and from state_dict(); every real parameter, though,
    # must be present in the slice we loaded.
    truly_missing = set(module.block.state_dict().keys()) - set(block_state.keys())
    assert not unexpected, f"unexpected keys loading GPT block: {unexpected}"
    assert not truly_missing, f"missing keys loading GPT block: {sorted(truly_missing)}"

    module.eval()
    return module
