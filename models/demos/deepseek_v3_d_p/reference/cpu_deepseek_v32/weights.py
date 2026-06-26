# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Weight initialization and loading for the CPU reference layers (``IndexerCPU`` /
``MLACPU`` in ``model.py``).

Two entry points, both via the free function ``initialize_weights(module, ...)``:
  * random init (default) — normal(0, 0.02) on every ``Linear``; norms keep ones.
  * pretrained load — read a layer's attention tensors from DeepSeek-V3.2-Exp
    safetensors shard file(s), map HF (transformers-style) names to our module
    names (mirrors ``inference/convert.py``), and dequantize fp8 → bf16.

Pretrained shards are fetched just-in-time (and cached) by ``load_pretrained_hf`` /
``resolve_layer_shards``. This module depends on ``model.py`` (for the module/layer
types); ``model.py`` does not import it, so there is no circular import.
"""

import torch
import torch.nn as nn
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32.model import BLOCK_SIZE, IndexerCPU, Linear

DEFAULT_REPO = "deepseek-ai/DeepSeek-V3.2-Exp"
_INDEX_FILE = "model.safetensors.index.json"

# HF (transformers-style) submodule name -> our parameter prefix within MLACPU.
# Mirrors inference/convert.py. fp8 weights carry a companion ``*.weight_scale_inv``.
_HF_TO_MLA = {
    "q_a_proj": "wq_a",
    "q_a_layernorm": "q_norm",
    "q_b_proj": "wq_b",
    "kv_a_proj_with_mqa": "wkv_a",
    "kv_a_layernorm": "kv_norm",
    "kv_b_proj": "wkv_b",
    "o_proj": "wo",
    "indexer.wq_b": "indexer.wq_b",
    "indexer.wk": "indexer.wk",
    "indexer.k_norm": "indexer.k_norm",
    "indexer.weights_proj": "indexer.weights_proj",
}


def _dequant_fp8(weight: torch.Tensor, scale_inv: torch.Tensor, block: int = BLOCK_SIZE) -> torch.Tensor:
    """
    Reconstruct a bf16 weight from a blockwise fp8 (e4m3) checkpoint tensor and
    its per-(block x block) scale, matching model.py:weight_dequant. ``scale_inv``
    has shape ``[ceil(O/block), ceil(I/block)]``.
    """
    w = weight.float()
    s = scale_inv.repeat_interleave(block, 0)[: w.shape[0]].repeat_interleave(block, 1)[:, : w.shape[1]]
    return (w * s).to(torch.bfloat16)


def load_attention_state_dict(shard_paths, layer: int) -> dict:
    """
    Read one transformer layer's attention (MLA + indexer) tensors from the given
    DeepSeek-V3.2-Exp safetensors shard file(s) and return a state dict keyed to
    ``MLACPU``'s parameter names (indexer params keep the ``indexer.`` prefix).

    fp8 weights are dequantized to bf16; norms and the indexer's ``weights_proj``
    are cast to fp32 to match the module dtypes. Shard paths come from
    ``resolve_layer_shards`` (which downloads on demand).
    """
    from safetensors import safe_open

    if isinstance(shard_paths, str):
        shard_paths = [shard_paths]
    prefix = f"model.layers.{layer}.self_attn."
    state = {}
    for path in shard_paths:
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in (k for k in f.keys() if k.startswith(prefix)):
                if k.endswith(".weight_scale_inv"):
                    continue
                sub, field = k[len(prefix) :].rsplit(".", 1)  # ("q_a_proj","weight") / ("indexer.wk","weight")
                if sub not in _HF_TO_MLA:
                    continue
                t = f.get_tensor(k)
                if t.dtype == torch.float8_e4m3fn:
                    t = _dequant_fp8(t, f.get_tensor(k.replace(".weight", ".weight_scale_inv")))
                key = f"{_HF_TO_MLA[sub]}.{field}"
                # Match module dtypes: norms + weights_proj fp32, linear weights bf16.
                t = t.float() if ("norm" in key or "weights_proj" in key) else t.to(torch.bfloat16)
                state[key] = t
    if not state:
        raise ValueError(f"No layer-{layer} attention tensors found in {shard_paths}.")
    return state


def _check_pretrained_load(missing, unexpected, what: str):
    """Validate a strict=False load_state_dict result for a pretrained load."""
    if missing:
        raise RuntimeError(f"{what}: parameters missing from checkpoint: {missing}")
    if unexpected:
        logger.warning(f"{what}: ignoring unexpected checkpoint keys: {unexpected}")


def init_random(module: nn.Module):
    """
    Random-init every ``Linear`` in ``module`` (normal 0, 0.02); norms keep their
    default (ones) value. Works for both ``IndexerCPU`` and ``MLACPU`` (whose walk
    also covers the nested indexer, in registration order).
    """
    name = type(module).__name__
    logger.info(f"Initializing {name} weights randomly for testing")
    for _, m in module.named_modules():
        if isinstance(m, Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.0, std=0.02)
    logger.info(f"✓ {name} random weight initialization complete")


def load_pretrained(module: nn.Module, shard_paths, layer: int = 0):
    """
    Load pretrained weights for ``layer`` into ``module`` from DeepSeek-V3.2-Exp
    safetensors shard file(s). ``MLACPU`` gets MLA + nested indexer; a standalone
    ``IndexerCPU`` gets just the indexer tensors (``indexer.`` prefix stripped).
    """
    name = type(module).__name__
    logger.info(f"Loading pretrained {name} weights (layer {layer}) from {shard_paths}")
    sd = load_attention_state_dict(shard_paths, layer)
    if isinstance(module, IndexerCPU):
        sd = {k[len("indexer.") :]: v for k, v in sd.items() if k.startswith("indexer.")}
    missing, unexpected = module.load_state_dict(sd, strict=False)
    _check_pretrained_load(missing, unexpected, f"{name} layer {layer}")
    logger.info(f"✓ Loaded pretrained {name} weights")


def initialize_weights(
    module: nn.Module,
    layer: int = None,
    checkpoint_path=None,
    repo: str = DEFAULT_REPO,
    token: str = None,
    local_files_only: bool = False,
):
    """
    The single weight-init entry point. Modes:

    * **random** (default) — neither ``layer`` nor ``checkpoint_path`` given.
    * **pretrained from HF** — ``layer`` given: resolve + (JIT) download that layer's
      shard(s) and load them. ``local_files_only=True`` uses the cache only and raises
      ``LocalEntryNotFoundError`` if a shard is absent (so callers can skip).
    * **pretrained from a local shard** — ``checkpoint_path`` (a shard path or list)
      for ``layer`` (default 0); takes precedence over HF resolution.
    """
    if checkpoint_path is not None:
        load_pretrained(module, checkpoint_path, layer or 0)
    elif layer is not None:
        load_pretrained_hf(module, layer, repo, token, local_files_only)
    else:
        init_random(module)


def resolve_layer_shards(layer: int = 0, repo: str = DEFAULT_REPO, token: str = None, local_files_only: bool = False):
    """
    Return local path(s) to the safetensors shard(s) that hold ``layer``'s attention
    (MLA + indexer) weights. Missing files are downloaded just-in-time, in full, via
    ``hf_hub_download`` (cached, atomic) — a warning is emitted first since the shards
    are multi-GB. With ``local_files_only=True`` only cached files are used and
    ``LocalEntryNotFoundError`` is raised if a shard is absent.
    """
    import json

    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    index = hf_hub_download(repo, _INDEX_FILE, token=token, local_files_only=local_files_only)
    weight_map = json.load(open(index))["weight_map"]
    prefix = f"model.layers.{layer}.self_attn."
    shards = sorted({weight_map[k] for k in weight_map if k.startswith(prefix)})
    if not shards:
        raise ValueError(f"No attention tensors for layer {layer} in {repo} index.")

    if not local_files_only:
        missing = [s for s in shards if not isinstance(try_to_load_from_cache(repo, s), str)]
        if missing:
            logger.warning(
                f"Downloading {len(missing)} shard(s) for layer {layer} from {repo} "
                f"({', '.join(missing)}); this can take a while — the shards are multi-GB."
            )
    return [hf_hub_download(repo, s, token=token, local_files_only=local_files_only) for s in shards]


def load_pretrained_hf(
    module: nn.Module, layer: int = 0, repo: str = DEFAULT_REPO, token: str = None, local_files_only: bool = False
):
    """
    Resolve (downloading if needed) and load ``layer``'s pretrained weights from the
    DeepSeek-V3.2-Exp HF repo into ``module``. Convenience wrapper over
    ``resolve_layer_shards`` + ``load_pretrained``.
    """
    load_pretrained(module, resolve_layer_shards(layer, repo, token, local_files_only), layer)
