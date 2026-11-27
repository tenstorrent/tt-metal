# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The L1 a pipeline leaves to its matmuls, the `l1_limit` of `models.tt_dit.utils.matmul_config_search`.

Each case builds a pipeline with its preset for the mesh, with its warmup, and runs a request
traced where serving traces, the state it serves in. Before every single-chip matmul of the
transformer it reads the lowest allocated L1 address of the matmul's device; the matmuls of the
text encoders and the VAE are left out. A trace replay runs no Python, so the lowest address at
each replay is read too and reported apart (`lowest_at_replay`). The allocator fills L1 top-down, so a matmul's static circular buffers must end at
or below the lowest address seen. Pick cases with -k (the model and mesh, e.g. -k "klein and 2x4").

    pytest models/tt_dit/tests/test_l1_probe.py -k "klein and 2x4"

Results go to `generated/outputs/l1_probe/<case>.json` and to a `L1_PROBE` log line.
"""

from __future__ import annotations

import functools
import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from loguru import logger
from PIL import Image

import ttnn
from models.tt_dit.utils.test import (
    line_params_req_exact_devices,
    ring_params_8k_req_exact_devices,
    ring_params_req_exact_devices,
)

from .models.flux2.device_params import line_params_flux2_klein
from .models.qwenimage.device_params import EDIT_MESH_DEVICE_PARAMS

_MATMULS = ("minimal_matmul", "minimal_matmul_split", "dit_minimal_matmul_addcmul_fused")
_IMAGE = Path(__file__).resolve().parents[2] / "sample_data" / "demo.jpeg"
_QWEN_EDIT_PARAMS = {p.values[0]: p.values[1] for p in EDIT_MESH_DEVICE_PARAMS}
_IDEOGRAM = {"l1_small_size": 32768, "trace_region_size": 60000000}


def _run_klein(mesh_device: ttnn.MeshDevice, model_location_generator: Callable[[str], str]) -> None:
    from models.tt_dit.pipelines.flux2.pipeline_flux2 import Flux2Pipeline

    pipeline = Flux2Pipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=model_location_generator("black-forest-labs/FLUX.2-klein-4B"),
    )
    pipeline(prompts=["a watercolor"], images=[Image.open(_IMAGE)], num_inference_steps=4, seed=0, traced=True)


def _run_qwen(mesh_device: ttnn.MeshDevice, model_location_generator: Callable[[str], str]) -> None:
    from models.tt_dit.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline

    pipeline = QwenImagePipeline.create_pipeline(mesh_device=mesh_device, checkpoint_name="Qwen/Qwen-Image-Edit-2511")
    pipeline(
        prompts=["a watercolor"],
        num_inference_steps=4,
        seed=0,
        image=Image.open(_IMAGE).convert("RGB").resize((1024, 1024)),
    )


def _run_wan(mesh_device: ttnn.MeshDevice, model_location_generator: Callable[[str], str]) -> None:
    from models.tt_dit.pipelines.wan.pipeline_wan_i2v import ImagePrompt, WanPipelineI2V

    # Every size the mesh serves, warmed untraced by the pipeline, then one traced request at each,
    # as serving runs them.
    pipeline = WanPipelineI2V.create_pipeline(
        mesh_device=mesh_device, checkpoint_name="Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    )
    for width, height in pipeline._resolutions:
        image = Image.open(_IMAGE).convert("RGB").resize((width, height))
        # Two steps, so that both experts run.
        pipeline(
            prompts=["a watercolor"],
            image_prompt=[ImagePrompt(image=image, frame_pos=0)],
            num_inference_steps=2,
            width=width,
            height=height,
            traced=True,
        )


def _fake_load_torch_state_dict(self: Any, state_dict: Any, *, strict: bool = True) -> Any:
    """Module.load_torch_state_dict without weights: each parameter gets an unfilled tensor of its shape.

    The L1 a pipeline leaves to its matmuls depends on what it allocates, not on the weight values.
    """
    import torch

    from models.tt_dit.layers.module import IncompatibleKeys
    from models.tt_dit.utils import tensor

    for _, child in self.named_children():
        _fake_load_torch_state_dict(child, {})
    for _, parameter in self.named_parameters():
        if parameter.on_host:
            parameter.data = tensor.from_torch(
                torch.zeros(parameter.total_shape),
                device=parameter.device,
                layout=parameter.layout,
                dtype=parameter.dtype,
                mesh_axes=parameter.mesh_axes,
                on_host=True,
            )
        else:
            parameter.data = ttnn.allocate_tensor_on_device(
                ttnn.Shape(parameter.local_shape),
                parameter.dtype,
                parameter.layout,
                parameter.device,
                parameter.memory_config,
            )
    self._mark_loaded()
    return IncompatibleKeys([], [])


def _run_ideogram(mesh_device: ttnn.MeshDevice, model_location_generator: Callable[[str], str]) -> None:
    """With fake weights: the fp8 checkpoint is gated and on none of the CI runners' stores.

    The configs and the tokenizer come from the public Qwen3-VL checkpoint the pipeline reads its
    encoder config from.
    """
    import tempfile

    import transformers
    from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

    from models.tt_dit.layers.module import Module
    from models.tt_dit.pipelines.ideogram4 import pipeline_ideogram4 as pi

    class RandomAutoencoderKL(AutoencoderKL):
        def load_state_dict(self, state_dict: Any, strict: bool = True, assign: bool = False) -> Any:
            return super().load_state_dict(self.state_dict(), strict=strict, assign=assign)

    tokenizer = transformers.AutoTokenizer.from_pretrained(pi._DEFAULT_QWEN_REPO)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Module, "load_torch_state_dict", _fake_load_torch_state_dict)
        mp.setattr(pi, "_dq", lambda path: {})
        mp.setattr(pi, "_load_file", lambda path: {})
        mp.setattr(pi, "_AutoencoderKL", RandomAutoencoderKL)
        mp.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
        with tempfile.TemporaryDirectory() as weights_dir:
            pipeline = pi.Ideogram4Pipeline.create_pipeline(
                mesh_device=mesh_device, height=2048, width=2048, weights_dir=weights_dir
            )
            pipeline(prompts=["a watercolor"], height=2048, width=2048, seed=0)


# Per model: the runner, the transformer class, its methods that run matmuls, and its submodules
# the pipeline calls directly.
_MODELS = {
    "klein": (
        _run_klein,
        "models.tt_dit.models.transformers.transformer_flux2",
        "Flux2Transformer",
        ("forward",),
        ("context_embedder",),  # once per request, outside forward
    ),
    "qwen": (
        _run_qwen,
        "models.tt_dit.models.transformers.transformer_qwenimage",
        "QwenImageTransformer",
        ("forward", "prologue", "run_blocks", "epilogue"),  # DBCache calls the three parts directly
        (),
    ),
    "wan": (
        _run_wan,
        "models.tt_dit.models.transformers.wan2_2.transformer_wan",
        "WanTransformer3DModel",
        # combined_step runs inner_step; DBCache splits a step in three; the prompt's projection
        # runs once per request.
        ("forward", "inner_step", "cache_head", "cache_body", "cache_tail", "prepare_text_conditioning"),
        (),
    ),
    "ideogram": (
        _run_ideogram,
        "models.tt_dit.models.transformers.transformer_ideogram4",
        "Ideogram4Transformer",
        ("forward",),
        (),
    ),
}

# Per spatial bucket each Wan expert keeps its traces, as in test_pipeline_wan_resolutions.
_WAN_TRACE_REGION_PER_BUCKET = 256 * 1024 * 1024

_CASES = [
    ("klein", (2, 2), line_params_flux2_klein),
    ("klein", (2, 4), line_params_flux2_klein),
    ("klein", (4, 8), line_params_flux2_klein),
    *(("qwen", shape, _QWEN_EDIT_PARAMS[shape]) for shape in ((2, 2), (2, 4), (4, 8))),
    ("wan", (1, 4), {**line_params_req_exact_devices, "trace_region_size": _WAN_TRACE_REGION_PER_BUCKET}),
    ("wan", (2, 2), {**line_params_req_exact_devices, "trace_region_size": _WAN_TRACE_REGION_PER_BUCKET}),
    ("wan", (2, 4), {**line_params_req_exact_devices, "trace_region_size": _WAN_TRACE_REGION_PER_BUCKET}),
    # 480p and 720p
    ("wan", (4, 8), {**ring_params_req_exact_devices, "trace_region_size": 2 * _WAN_TRACE_REGION_PER_BUCKET}),
    ("wan", (4, 32), {**ring_params_8k_req_exact_devices, "trace_region_size": 2 * _WAN_TRACE_REGION_PER_BUCKET}),
    ("ideogram", (2, 4), {**line_params_req_exact_devices, **_IDEOGRAM}),
    ("ideogram", (4, 8), {**ring_params_req_exact_devices, **_IDEOGRAM}),
]


def _lowest_allocated(device: ttnn.MeshDevice) -> int | None:
    """The lowest allocated L1 address of the device's allocator, or None if nothing is allocated.

    The block table holds addresses relative to the allocator's base. Each mesh device, a submesh
    too, has an allocator of its own.
    """
    view = ttnn.device.get_memory_view(device, ttnn.BufferType.L1)
    addresses = [int(b["address"]) for b in view.block_table if b.get("allocated") == "yes"]
    if not addresses:
        return None
    return ttnn.get_allocator_base_address(device, ttnn.BufferType.L1) + min(addresses)


@pytest.mark.parametrize(
    ("model", "mesh_device", "device_params"),
    [pytest.param(m, shape, params, id=f"{m}-{shape[0]}x{shape[1]}") for m, shape, params in _CASES],
    indirect=["mesh_device", "device_params"],
)
def test_l1_probe(
    *,
    model: str,
    mesh_device: ttnn.MeshDevice,
    model_location_generator: Callable[[str], str],
    is_ci_env: bool,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    import importlib

    if is_ci_env:
        monkeypatch.setenv("TT_DIT_CACHE_DIR", "/tmp/TT_DIT_CACHE")

    run, module, cls, methods, submodules = _MODELS[model]
    lowest: dict[tuple[int, int], int] = {}
    calls: dict[tuple[int, int], int] = defaultdict(int)
    replays: list[int] = []
    forwards = 0  # transformer calls running, so that only their matmuls are recorded

    def wrap_matmul(fn: Any) -> Any:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            w = args[1] if len(args) > 1 else kwargs.get("weight_tensor")
            assert w is not None
            if forwards:
                key = (int(w.padded_shape[-2]), int(w.padded_shape[-1]))
                calls[key] += 1
                address = _lowest_allocated(w.device())
                if address is not None:
                    lowest[key] = min(lowest.get(key, address), address)
            return fn(*args, **kwargs)

        return wrapper

    def wrap_execute_trace(fn: Any) -> Any:
        @functools.wraps(fn)
        def wrapper(device: ttnn.MeshDevice, *args: Any, **kwargs: Any) -> Any:
            address = _lowest_allocated(device)
            if address is not None:
                replays.append(address)
            return fn(device, *args, **kwargs)

        return wrapper

    def wrap_forward(fn: Any) -> Any:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal forwards
            forwards += 1
            try:
                return fn(*args, **kwargs)
            finally:
                forwards -= 1

        return wrapper

    def wrap_init(fn: Any) -> Any:
        @functools.wraps(fn)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> None:
            fn(self, *args, **kwargs)
            for name in submodules:
                child = getattr(self, name)
                object.__setattr__(child, "forward", wrap_forward(child.forward))

        return wrapper

    for name in _MATMULS:
        monkeypatch.setattr(ttnn.experimental, name, wrap_matmul(getattr(ttnn.experimental, name)))
    monkeypatch.setattr(ttnn, "execute_trace", wrap_execute_trace(ttnn.execute_trace))
    klass = getattr(importlib.import_module(module), cls)
    for name in methods:
        monkeypatch.setattr(klass, name, wrap_forward(getattr(klass, name)))
    if submodules:
        monkeypatch.setattr(klass, "__init__", wrap_init(klass.__init__))

    run(mesh_device, model_location_generator)

    assert calls, "no single-chip matmul ran in a transformer forward"
    assert lowest, "no L1 was allocated during any transformer matmul"
    result = {
        "case": request.node.callspec.id,
        "arch": "blackhole" if ttnn.device.is_blackhole() else "wormhole_b0",
        "mesh": list(mesh_device.shape),
        "grid": str(mesh_device.compute_with_storage_grid_size()),
        "l1_limit": min(lowest.values()),
        # Over every replay, the encoders' and the VAE's too. A transformer trace replayed with less
        # L1 free than at its capture would overwrite what sits below.
        "lowest_at_replay": min(replays, default=None),
        "per_matmul": [
            {"K": k, "N": n, "lowest_allocated": lowest.get((k, n)), "calls": calls[(k, n)]} for k, n in sorted(calls)
        ],
    }
    out = Path("generated/outputs/l1_probe")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{result['case']}.json").write_text(json.dumps(result, indent=2))
    logger.info(f"L1_PROBE {json.dumps(result)}")
    if replays and min(replays) < result["l1_limit"]:
        logger.warning(f"a trace replayed with L1 allocated below l1_limit, down to {min(replays)}")
