# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Load Molmo2 weights / processor from the HF Hub checkpoint.

The allenai/Molmo2-8B remote code targets transformers==4.57.1 (see the model card),
while tt-metal pins transformers 5.x. Under 5.x:

  * ``from_pretrained(<local snapshot dir>)`` fails: the dynamic-module hasher follows the
    snapshot symlinks into ``blobs/`` and cannot find the relative imports.
  * ``Molmo2Processor`` passes its config flags (``image_use_col_tokens`` ...) to
    ``ProcessorMixin.__init__``, which now rejects anything that isn't a sub-processor.
  * The remote modeling code itself is incompatible (removed ``ROPE_INIT_FUNCTIONS["default"]``,
    ``create_causal_mask`` signature change, ``generate()`` no longer passes ``cache_position``).
    Under transformers 5 remote-code models can also silently keep randomly initialised weights.

The TT model only needs the raw checkpoint tensors and the processor, so weights are read
directly from safetensors (no remote modeling code involved) and the processor is loaded
through a small compatibility shim. Processor outputs were verified bit-exact against
transformers 4.57.1 for text / single image / multi image / video inputs.

The HF reference model (golden generation, reference/test_functional.py) must run under
transformers==4.57.1.
"""

import json
import os
from pathlib import Path

import torch

MOLMO2_HF_ID = "allenai/Molmo2-8B"
# Pin the revision the TT implementation was validated against.
MOLMO2_HF_REVISION = "e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"


def _hf_model_id() -> str:
    """HF repo id, or a local checkpoint dir passed via HF_MODEL (e.g. by tt-inference-server)."""
    return os.environ.get("HF_MODEL", MOLMO2_HF_ID)


def _revision(model_id: str):
    return MOLMO2_HF_REVISION if model_id == MOLMO2_HF_ID else None


def molmo2_checkpoint_dir(model_id: str = None) -> Path:
    """Local directory containing the safetensors / config / sample files (downloads if needed)."""
    model_id = model_id or _hf_model_id()
    if os.path.isdir(model_id):
        return Path(model_id)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_id, revision=_revision(model_id)))


def load_molmo2_state_dict(model_id: str = None, dtype: torch.dtype = torch.bfloat16) -> dict:
    """Read every checkpoint tensor directly from safetensors.

    Keys match ``AutoModelForImageTextToText.from_pretrained(...).state_dict()``.
    """
    from safetensors import safe_open

    ckpt_dir = molmo2_checkpoint_dir(model_id)
    weight_map = json.loads((ckpt_dir / "model.safetensors.index.json").read_text())["weight_map"]
    state_dict = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(str(ckpt_dir / shard), framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key).to(dtype)
    assert len(state_dict) == len(weight_map), f"loaded {len(state_dict)} of {len(weight_map)} tensors"
    return state_dict


def _transformers_major() -> int:
    import transformers

    return int(transformers.__version__.split(".")[0])


def _remote_code_id(model_id: str) -> str:
    # A HF cache snapshot dir (symlinks into blobs/) breaks the transformers 5 dynamic-module
    # loader; load by repo id instead (resolves from the local cache, works with HF_HUB_OFFLINE=1).
    if os.path.isdir(model_id) and "models--allenai--Molmo2-8B" in str(Path(model_id).resolve()):
        return MOLMO2_HF_ID
    return model_id


def _patch_processor_class(model_id: str):
    if _transformers_major() < 5:
        return
    from transformers import ProcessorMixin
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cls = get_class_from_dynamic_module(
        "processing_molmo2.Molmo2Processor", model_id, revision=_revision(model_id), trust_remote_code=True
    )
    if getattr(cls, "_tt_compat_patched", False):
        return
    config_attrs = [a for a in cls.optional_attributes if a != "chat_template"]

    class _ProcessorMixinCompat(ProcessorMixin):
        # transformers 4.x stored optional_attributes passed to ProcessorMixin.__init__ (None if absent).
        def __init__(self, *args, **kwargs):
            config = {k: kwargs.pop(k) for k in list(kwargs) if k in config_attrs}
            super().__init__(*args, **kwargs)
            for k in config_attrs:
                setattr(self, k, config.get(k))

    cls.__bases__ = (_ProcessorMixinCompat,)
    cls._tt_compat_patched = True


def load_molmo2_processor(model_id: str = None):
    from transformers import AutoProcessor

    model_id = _remote_code_id(model_id or _hf_model_id())
    _patch_processor_class(model_id)
    return AutoProcessor.from_pretrained(model_id, revision=_revision(model_id), trust_remote_code=True)
