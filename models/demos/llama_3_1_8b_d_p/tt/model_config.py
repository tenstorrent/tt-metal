# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Weight resolution, checkpoint loading, and the tilized weight-cache path.

Borrowed structurally from `minimax_m3/tt/model_config.py`. The loader body itself is written fresh:
the donor loads a `trust_remote_code` MoE checkpoint and converts to Meta naming, while this one
walks a plain `LlamaForCausalLM` safetensors set whose keys already match the reference's
`state_dict` (pinned by `tests/torch_ref/test_llama_reference.py::test_state_dict_keys_match_upstream`).

**There is no dequantization step.** The checkpoint is unquantized bf16, so the single dtype exit is
the whole conversion — the packed-block / scale-tensor machinery every quantized donor carries has
no counterpart here.

Weight resolution follows recipe §4 step 0, in order: `PREFILL_HF_MODEL` / `HF_MODEL` if set, then
the shared store at `/mnt/models/meta-llama/Llama-3.1-8B-Instruct`. The weights for this bring-up
are REAL and already staged there; nothing here falls back to synthetic values, because a synthetic
run must be a loud, deliberate choice rather than something a missing env var causes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from loguru import logger

DEFAULT_WEIGHTS_PATH = "/mnt/models/meta-llama/Llama-3.1-8B-Instruct"
HF_REPO = "meta-llama/Llama-3.1-8B-Instruct"


def resolve_weights_path(required: bool = True) -> str | None:
    """Locate the checkpoint. Recipe §4 step 0 order: explicit override, then the shared store."""
    for env in ("PREFILL_HF_MODEL", "HF_MODEL"):
        value = os.getenv(env)
        if value:
            if not Path(value).is_dir():
                raise FileNotFoundError(f"{env}={value} is not a directory")
            return value
    if Path(DEFAULT_WEIGHTS_PATH).is_dir():
        return DEFAULT_WEIGHTS_PATH
    if required:
        raise FileNotFoundError(
            f"no checkpoint: set HF_MODEL, or stage {HF_REPO} at {DEFAULT_WEIGHTS_PATH}. "
            "Synthetic weights are a deliberate, logged choice - never a silent fallback."
        )
    return None


class ModelArgs:
    """Checkpoint paths, the HF config, and the tilized weight cache location."""

    def __init__(self, mesh_device=None, dummy_weights: bool = False, weights_path: str | None = None):
        self.mesh_device = mesh_device
        self.dummy_weights = dummy_weights
        self.model_path = weights_path or resolve_weights_path(required=not dummy_weights)
        if self.model_path:
            logger.info(f"Llama-3.1-8B weights: {self.model_path}{' (DUMMY - random values)' if dummy_weights else ''}")

    @property
    def hf_config(self):
        """The live HF config, built from the VENDORED config.json rather than the checkpoint's.

        Deliberate: the vendored copy is what `tests/torch_ref/test_llama_reference.py` asserts every
        constant against, so the device side and the oracle cannot be reading different dims. It also
        fixes up the 4.x/5.x `rope_scaling` vs `rope_parameters` split in one place.
        """
        from models.demos.llama_3_1_8b_d_p.reference.config import LlamaConfigConstants

        return LlamaConfigConstants.from_json().to_hf_config()

    @staticmethod
    def load_state_dict(weights_path: str, dummy_weights: bool = False) -> dict:
        """Full `state_dict` from the safetensors shards, cast to a single dtype.

        Keys come back in HF naming (`model.layers.N.self_attn.q_proj.weight`, ...), unchanged —
        `substate()` slices per layer and per block from them.
        """
        if dummy_weights:
            return {}
        from safetensors.torch import load_file

        weights_dir = Path(weights_path)
        index_path = weights_dir / "model.safetensors.index.json"
        if index_path.exists():
            shards = sorted({v for v in json.loads(index_path.read_text())["weight_map"].values()})
        else:
            shards = sorted(p.name for p in weights_dir.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"no safetensors under {weights_dir}")

        state_dict: dict[str, torch.Tensor] = {}
        for shard in shards:
            logger.info(f"loading {shard}")
            state_dict.update(load_file(str(weights_dir / shard)))
        logger.info(f"loaded {len(state_dict)} tensors from {len(shards)} shard(s)")
        return state_dict

    def weight_cache_path(self, dtype) -> str | None:
        """Directory for ttnn's tilized tensor cache.

        The mesh shape is part of the directory name because a tilized tensor is sharded for a
        specific mesh. Note that a name like `tensor_cache_bfp8_MeshShape([8, 4])` contains
        `[8, 4]`, which glob reads as a character class — see
        `utils/general_utils.cache_file_exists`, which uses listdir for exactly that reason.
        """
        if self.dummy_weights or not self.model_path:
            return None
        shape = "" if self.mesh_device is None else f"_{self.mesh_device.shape}"
        path = Path(self.model_path) / f"tensor_cache_{dtype}{shape}"
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            # The shared store is often owned by another user; a read-only cache dir is not fatal.
            logger.warning(f"weight cache {path} is not writable ({e}); tensors will be re-tilized each run")
            return str(path) if path.is_dir() else None
        return str(path)
