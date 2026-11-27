# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Scratch: FLUX.2 klein at sizes whose tokens per device are not whole tiles. Per case of
KLEIN_CASES (HxW or HxW:edit, comma-separated) one eager request of 4 steps; saves the image to
KLEIN_OUT_DIR. Where the pipeline keeps one padded token sequence, also checks that its reference
and padding rows are bit-identical after the steps to what was uploaded.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import ttnn
from models.tt_dit.utils import tensor

from ....pipelines.flux2 import pipeline_flux2
from ....pipelines.flux2.pipeline_flux2 import Flux2Pipeline
from .device_params import line_params_flux2_klein


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [[(2, 4), line_params_flux2_klein], [(2, 2), line_params_flux2_klein], [(4, 8), line_params_flux2_klein]],
    ids=["2x4", "2x2", "4x8"],
    indirect=["mesh_device", "device_params"],
)
def test_klein_padding(*, mesh_device: ttnn.MeshDevice, model_location_generator) -> None:
    out_dir = Path(os.environ.get("KLEIN_OUT_DIR", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for c in os.environ.get("KLEIN_CASES", "1024x1024,1024x1024:edit").split(","):
        size, _, kind = c.partition(":")
        h, w = (int(v) for v in size.split("x"))
        cases.append((h, w, kind == "edit"))
    pipeline = Flux2Pipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=model_location_generator("black-forest-labs/FLUX.2-klein-4B"),
        resolutions=sorted({(1024, 1024), *((w, h) for h, w, _ in cases)}),
    )
    reference = Image.effect_noise((1536, 1152), 64).convert("RGB")
    sp_axis = pipeline._parallel_config.sequence_parallel.mesh_axis

    states, uploads = [], []
    make_state = pipeline._denoise_state
    to_device = pipeline._to_device

    def capture_state(*args, **kwargs):
        states.append(make_state(*args, **kwargs))
        return states[-1]

    def capture_upload(x: torch.Tensor, *, mesh_axes):
        uploads.append(x)
        return to_device(x, mesh_axes=mesh_axes)

    pipeline._denoise_state = capture_state
    pipeline._to_device = capture_upload

    failures = []
    for h, w, edit in cases:
        name = f"{h}x{w}_{'edit' if edit else 't2i'}"
        states.clear()
        uploads.clear()
        try:
            (image,) = pipeline(
                prompts=["a lighthouse on a cliff, golden hour"],
                num_inference_steps=4,
                seed=0,
                width=w,
                height=h,
                images=[reference] if edit else None,
            )
        except Exception as e:  # noqa: BLE001
            print(f"CASE {name}: failed: {str(e).splitlines()[0][:200]}", flush=True)
            failures.append(name)
            continue
        image.save(out_dir / f"{name}.png")
        line = f"CASE {name}: ok"
        state = states[-1]
        if hasattr(state, "latent_mask"):
            spatial = (h // 16) * (w // 16)
            sequence = next(u for u in uploads if u.ndim == 3 and u.shape[1] >= spatial and u.shape[2] == 128)
            after = tensor.to_torch(state.latents.data, mesh_axes=(None, sp_axis, None)).float()
            before = sequence.to(torch.bfloat16).float()
            same = torch.equal(after[:, spatial:], before[:, spatial:])
            line += f"; {sequence.shape[1]} rows, {sequence.shape[1] - spatial} reference and padding rows unchanged: {same}"
            if not same:
                failures.append(name)
        print(line, flush=True)
    assert not failures, failures


def test_klein_padding_psnr() -> None:
    """Compares the PNGs of KLEIN_OUT_DIR against those of KLEIN_BASE_DIR, case by case."""
    out_dir, base_dir = Path(os.environ["KLEIN_OUT_DIR"]), Path(os.environ["KLEIN_BASE_DIR"])
    for path in sorted(base_dir.glob("*.png")):
        other = out_dir / path.name
        if not other.exists():
            print(f"PSNR {path.stem}: missing", flush=True)
            continue
        a = np.asarray(Image.open(path)).astype(np.float64)
        b = np.asarray(Image.open(other)).astype(np.float64)
        mse = np.mean((a - b) ** 2)
        print(f"PSNR {path.stem}: {10 * np.log10(255**2 / mse) if mse else float('inf'):.1f} dB", flush=True)


_ = pipeline_flux2
