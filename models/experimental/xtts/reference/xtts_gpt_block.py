# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import importlib.abc
import importlib.machinery
import sys
import types

import torch
from torch import nn
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from models.experimental.xtts.config import (  # noqa: F401
    CHECKPOINT_FILE,
    FFN_SIZE,
    HEAD_DIM,
    HF_REPO_ID,
    HF_REVISION,
    HIDDEN_SIZE,
    LAYER_NORM_EPS,
    MAX_GPT_SEQ_LEN,
    MAX_MEL_POS,
    MAX_POSITIONS,
    MAX_TEXT_POS,
    NUM_HEADS,
    NUM_LAYERS,
)


class _StubObject:
    def __setstate__(self, state):
        """Restore stub attributes from an unpickle state dict."""
        if isinstance(state, dict):
            self.__dict__.update(state)


class _FakeTTSFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        """Return a fake ModuleSpec for TTS imports if matched."""
        if fullname == "TTS" or fullname.startswith("TTS."):
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        """Create a stub package module with dynamic attribute factories."""
        module = types.ModuleType(spec.name)
        module.__path__ = []  # mark as a package so submodules can be imported

        def __getattr__(name, _module=spec.name):
            """Return a stub class for any missing TTS attribute."""
            return type(name, (_StubObject,), {"__module__": _module})

        module.__getattr__ = __getattr__
        return module

    def exec_module(self, module):
        """No-op module execution for the TTS stub loader."""


def _install_tts_stub():
    """Install the TTS import stub on sys.meta_path if missing."""
    if not any(isinstance(f, _FakeTTSFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _FakeTTSFinder())


def load_xtts_state_dict():
    """Download the XTTS checkpoint and return tensor weights only."""
    from huggingface_hub import hf_hub_download

    _install_tts_stub()
    checkpoint_path = hf_hub_download(repo_id=HF_REPO_ID, filename=CHECKPOINT_FILE, revision=HF_REVISION)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    tensors = {k: v for k, v in state_dict.items() if torch.is_tensor(v)}

    return tensors


def build_gpt2_config():
    """Build the GPT-2 config used by standalone XTTS GPT blocks."""
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
        # Standalone GPT2Block needs eager attn + explicit causal mask (see build_causal_mask).
        attn_implementation="eager",
    )


def build_causal_mask(seq_len, dtype=torch.float32):
    """Build an upper-triangular causal attention mask."""
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype)
    return torch.triu(mask, diagonal=1).view(1, 1, seq_len, seq_len)


class XttsReferenceGptBlock(nn.Module):
    def __init__(self, config, layer_idx=0):
        """Wrap a single HuggingFace GPT2Block."""
        super().__init__()
        self.block = GPT2Block(config, layer_idx=layer_idx)

    def forward(self, hidden_states):
        """Run one GPT-2 block with an explicit causal mask."""
        mask = build_causal_mask(hidden_states.shape[1], hidden_states.dtype)
        out = self.block(hidden_states, attention_mask=mask)
        return out[0] if isinstance(out, tuple) else out


def reference_gpt_block(state_dict, layer_idx=0):
    """Load one GPT block from checkpoint weights into eval mode."""
    config = build_gpt2_config()
    module = XttsReferenceGptBlock(config, layer_idx=layer_idx)

    prefix = f"gpt.gpt.h.{layer_idx}."
    block_state = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
    _, unexpected = module.block.load_state_dict(block_state, strict=False)
    # GPT2Attention non-persistent causal-mask buffers are absent from the checkpoint.
    truly_missing = set(module.block.state_dict().keys()) - set(block_state.keys())
    assert not unexpected, f"unexpected keys loading GPT block: {unexpected}"
    assert not truly_missing, f"missing keys loading GPT block: {sorted(truly_missing)}"

    module.eval()
    return module
