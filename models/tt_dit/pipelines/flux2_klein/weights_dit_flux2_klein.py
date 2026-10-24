# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX.2 Klein DiT weights: the diffusers checkpoint as a torch state and its load through the tt_dit weight cache.

The checkpoint is ``transformer/diffusion_pytorch_model.safetensors`` of a diffusers snapshot (169 bf16 tensors for
4B, no biases; the BFL-layout twin ``flux-2-klein-*.safetensors`` is not read). ``Flux2KleinTransformer.
_prepare_torch_state`` and its children understand the diffusers names, so the state goes in as loaded. The cache
convention is ``utils/cache.py`` unchanged: ``TT_DIT_CACHE_DIR/<model_name>/<subfolder>/<config_id>mesh<shape>_bf16/``
with ``config_id`` from the ``DiTGParallelConfigNoCFG`` (``TP4_0_SP8_1`` for the default layout) and the completeness
marker ``cache_dict.json``. Mirrors ``pipelines/flux3_klein/pipeline_flux3_klein.py:421-489``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from loguru import logger

import ttnn

from ...utils import cache

if TYPE_CHECKING:
    from ...models.transformers.flux2_klein.transformer_flux2_klein import Flux2KleinTransformer
    from ...parallel.config import DiTGParallelConfigNoCFG

FLUX2_KLEIN_DIT_CHECKPOINT = "transformer/diffusion_pytorch_model.safetensors"


def transformer_cache_key(variant: str, size: str, snapshot: Path, model: Flux2KleinTransformer) -> tuple[str, str]:
    """``(model_name, subfolder)`` of the weight cache: the variant, size and snapshot hash name the model, the block
    counts the subfolder."""
    model_name = f"flux2_klein_{variant}_{size}_{snapshot.name}"
    subfolder = f"transformer_D{model.config.num_layers}_S{model.config.num_single_layers}"
    return model_name, subfolder


def load_dit_torch_state(snapshot: Path) -> dict[str, torch.Tensor]:
    """The diffusers transformer checkpoint with every floating tensor as bf16 (the checkpoint's own dtype)."""
    from safetensors.torch import load_file

    state = load_file(str(Path(snapshot) / FLUX2_KLEIN_DIT_CHECKPOINT))
    return {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in state.items()}


def load_time_torch_state(snapshot: Path) -> dict[str, torch.Tensor]:
    """The six tensors of the time path (``time_flux2_klein.FLUX2_KLEIN_TIME_WEIGHT_KEYS``) read directly from the
    checkpoint (a few hundred MB for 4B), bf16: the host-exact modulation needs them even on a weight-cache hit."""
    from safetensors import safe_open

    from ...models.transformers.flux2_klein.time_flux2_klein import FLUX2_KLEIN_TIME_WEIGHT_KEYS

    with safe_open(str(Path(snapshot) / FLUX2_KLEIN_DIT_CHECKPOINT), framework="pt") as f:
        return {k: f.get_tensor(k).to(torch.bfloat16) for k in FLUX2_KLEIN_TIME_WEIGHT_KEYS}


def load_flux2_klein_transformer_weights(
    model: Flux2KleinTransformer,
    variant: str,
    size: str,
    snapshot: Path,
    *,
    parallel_config: DiTGParallelConfigNoCFG,
    mesh_device: ttnn.MeshDevice,
) -> str:
    """Load ``model`` through the weight cache (``cache.load_model``: a complete cache is read as is, else the torch
    state is converted and the cache written). Returns ``"HIT"``, ``"MISS"`` or ``"NO CACHE DIR"``."""
    snapshot = Path(snapshot)
    model_name, subfolder = transformer_cache_key(variant, size, snapshot, model)
    cache_dir = (
        cache.model_cache_dir(
            model_name=model_name,
            subfolder=subfolder,
            parallel_config=parallel_config,
            mesh_shape=tuple(mesh_device.shape),
            mesh_device=mesh_device,
            required=False,
        )
        if cache.cache_dir_is_set()
        else None
    )
    state = (
        "HIT"
        if cache_dir is not None and (Path(cache_dir) / cache.CACHE_DICT_FILE).is_file()
        else ("MISS" if cache_dir is not None else "NO CACHE DIR")
    )
    t0 = time.time()
    cache.load_model(
        model,
        model_name=model_name,
        subfolder=subfolder,
        parallel_config=parallel_config,
        mesh_shape=tuple(mesh_device.shape),
        mesh_device=mesh_device,
        get_torch_state_dict=lambda: load_dit_torch_state(snapshot),
    )
    ttnn.synchronize_device(mesh_device)
    from ...models.transformers.flux2_klein.time_flux2_klein import Flux2KleinHostTime

    model.host_time = Flux2KleinHostTime(load_time_torch_state(snapshot), model.config)
    model.host_time(1000.0)  # warm-up: pages the weights in and primes the thread pool (the value is memoized)
    logger.info(
        f"RECORD transformer[{variant}/{size}] weights: cache {state} ({cache_dir}) loaded in {time.time() - t0:.1f}s; "
        f"host time path attached"
    )
    return state
