# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline factory for tt_dit serving and one-shot runners.

Only constructs pipelines via each model's ``create_pipeline``. Warmup is the
caller's responsibility (see ``pipeline_runner.py``).

External models use ``model_id="external:<module.name>"`` where the module
is importable via ``PYTHONPATH`` and defines one pipeline class with
``create_pipeline``.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from functools import partial
from typing import Any

import ttnn

_EXTERNAL_PREFIX = "external:"

# Model IDs match inference-server ModelRunners values without the ``tt-`` prefix.
# Flux / Qwen: one pipeline class serves multiple model_ids — map id → checkpoint.
_FLUX_CHECKPOINTS = {
    "flux.1-dev": "black-forest-labs/FLUX.1-dev",
    "flux.1-schnell": "black-forest-labs/FLUX.1-schnell",
}
_QWEN_CHECKPOINTS = {
    "qwen-image": "Qwen/Qwen-Image",
    "qwen-image-2512": "Qwen/Qwen-Image-2512",
}


def _wan_create_pipeline(*, mesh_device: ttnn.MeshDevice, **pipeline_params: Any):
    from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline

    return WanPipeline.create_pipeline(mesh_device=mesh_device, **pipeline_params)


def _wan_i2v_create_pipeline(*, mesh_device: ttnn.MeshDevice, **pipeline_params: Any):
    from models.tt_dit.pipelines.wan.pipeline_wan_i2v import WanPipelineI2V

    return WanPipelineI2V.create_pipeline(mesh_device=mesh_device, **pipeline_params)


def _mochi_create_pipeline(*, mesh_device: ttnn.MeshDevice, **pipeline_params: Any):
    from models.tt_dit.pipelines.mochi.pipeline_mochi import MochiPipeline

    return MochiPipeline.create_pipeline(mesh_device=mesh_device, **pipeline_params)


def _sd35_create_pipeline(*, mesh_device: ttnn.MeshDevice, **pipeline_params: Any):
    from models.tt_dit.pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large import (
        StableDiffusion3Pipeline,
    )

    return StableDiffusion3Pipeline.create_pipeline(mesh_device=mesh_device, **pipeline_params)


def _flux_create_pipeline(model_id: str, *, mesh_device: ttnn.MeshDevice, **pipeline_params: Any):
    from models.tt_dit.pipelines.flux1.pipeline_flux1 import Flux1Pipeline

    params = dict(pipeline_params)
    params.setdefault("checkpoint_name", _FLUX_CHECKPOINTS[model_id])
    return Flux1Pipeline.create_pipeline(mesh_device=mesh_device, **params)


def _motif_create_pipeline(*, mesh_device: ttnn.MeshDevice, **pipeline_params: Any):
    from models.tt_dit.pipelines.motif.pipeline_motif import MotifPipeline

    return MotifPipeline.create_pipeline(mesh_device=mesh_device, **pipeline_params)


def _qwen_create_pipeline(model_id: str, *, mesh_device: ttnn.MeshDevice, **pipeline_params: Any):
    from models.tt_dit.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline

    params = dict(pipeline_params)
    params.setdefault("checkpoint_name", _QWEN_CHECKPOINTS[model_id])
    return QwenImagePipeline.create_pipeline(mesh_device=mesh_device, **params)


_PIPELINE_CREATORS: dict[str, Callable[..., Any]] = {
    "wan2.2": _wan_create_pipeline,
    "wan2.2-i2v": _wan_i2v_create_pipeline,
    "mochi-1": _mochi_create_pipeline,
    "sd3.5": _sd35_create_pipeline,
    "flux.1-dev": partial(_flux_create_pipeline, "flux.1-dev"),
    "flux.1-schnell": partial(_flux_create_pipeline, "flux.1-schnell"),
    "motif-image-6b-preview": _motif_create_pipeline,
    "qwen-image": partial(_qwen_create_pipeline, "qwen-image"),
    "qwen-image-2512": partial(_qwen_create_pipeline, "qwen-image-2512"),
}


def _warmup_args_for(model_id: str, pipeline) -> dict[str, Any]:
    """Extra kwargs unpacked into the runner's traced warmup ``__call__``."""
    if model_id == "wan2.2-i2v":
        from PIL import Image

        width = int(getattr(pipeline, "_width", 832))
        height = int(getattr(pipeline, "_height", 480))
        return {"image_prompt": Image.new("RGB", (width, height))}
    return {}


def _load_module(module_name: str):
    """Load a module via ``importlib.import_module`` (dotted module name)."""
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"Cannot import external pipeline module {module_name!r}: {e}. "
            f"Check that {module_name!r} is a valid dotted module name and that: "
            f"1. the package root is on PYTHONPATH, e.g. "
            f"2. you have updated PYTHONPATH - e.g. PYTHONPATH=/path/to/package/parent:$PYTHONPATH "
            f"3. the parent of the top-level package for {module_name!r} is on PYTHONPATH."
        ) from e


def _pipeline_class_from_module(module) -> type:
    """Return the single pipeline class defined in ``module`` (has ``create_pipeline``)."""
    candidates = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if obj.__module__ == module.__name__ and callable(getattr(obj, "create_pipeline", None))
    ]
    if len(candidates) != 1:
        names = ", ".join(sorted(c.__name__ for c in candidates)) or "(none)"
        raise ImportError(
            f"External module {module.__name__!r} must define exactly one pipeline "
            f"class with create_pipeline; found {len(candidates)}: {names}"
        )
    return candidates[0]


def _resolve_creator(model_id: str) -> Callable[..., Any]:
    if model_id.startswith(_EXTERNAL_PREFIX):
        module_name = model_id[len(_EXTERNAL_PREFIX) :]
        if not module_name:
            raise ValueError(f"Invalid model_id={model_id!r}; expected external:<module.name>")
        pipeline_cls = _pipeline_class_from_module(_load_module(module_name))
        return pipeline_cls.create_pipeline

    creator = _PIPELINE_CREATORS.get(model_id)
    if creator is None:
        supported = ", ".join(sorted(_PIPELINE_CREATORS))
        raise ValueError(
            f"Unknown model_id={model_id!r}. Registered ids: {supported}. " f"External: external:<module.name>"
        )
    return creator


class PipelineFactory:
    """Create tt_dit pipelines by model_id."""

    @staticmethod
    def create(
        model_id: str,
        mesh_device: ttnn.MeshDevice,
        **pipeline_params: Any,
    ) -> tuple[Any, dict[str, Any]]:
        """Construct a pipeline via ``create_pipeline`` only (no warmup).

        Args:
            model_id: Registered id (e.g. ``wan2.2``), or
                ``external:<module.name>`` for an importable module that
                defines one pipeline class with ``create_pipeline``.
            mesh_device: Open mesh device.
            **pipeline_params: Forwarded as kwargs to ``create_pipeline``
                (e.g. ``height=720``, ``width=1280``).

        Returns:
            ``(pipeline, warmup_args)`` — ``warmup_args`` are unpacked into the
            runner's traced warmup only.
        """
        creator = _resolve_creator(model_id)
        pipeline = creator(mesh_device=mesh_device, **pipeline_params)
        return pipeline, _warmup_args_for(model_id, pipeline)
