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
MAX_POSITIONS = 1024  # >= any prefill length we test; only sizes the causal mask


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
    return {k: v for k, v in state_dict.items() if torch.is_tensor(v)}


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
    )


class XttsReferenceGptBlock(nn.Module):
    """One XTTS GPT decoder block, wrapping a HF ``GPT2Block``.

    The wrapper normalizes the return value to a bare hidden-states tensor of
    shape ``[batch, seq, hidden]``. This shields callers from the transformers
    API drift where ``GPT2Block.forward`` returns a tuple on some versions and a
    plain tensor on others.
    """

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.block = GPT2Block(config, layer_idx=layer_idx)

    def forward(self, hidden_states):
        out = self.block(hidden_states)
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
