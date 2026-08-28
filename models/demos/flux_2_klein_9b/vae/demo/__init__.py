# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Runnable demo surface for the FLUX.2 Klein 9B VAE (diffusers ``AutoencoderKLFlux2``).

Three entrypoints live in this package, one per Call in ``e2e_plan.json``::

    python -m models.demos.flux_2_klein_9b.vae.demo.demo_encode
    python -m models.demos.flux_2_klein_9b.vae.demo.demo_decode
    python -m models.demos.flux_2_klein_9b.vae.demo.demo_reconstruct

CARDINAL RULE: the chained forward pass lives ONLY in ``tt/pipeline.py``. Every demo here
obtains the resident pipeline from ``build_pipeline(mesh_device, layers=...)`` and calls
``run_encode`` / ``run_decode`` / ``run_reconstruct`` on it. Nothing in this package
re-implements the wiring; this module only holds argument parsing, mesh open/close, and
the reporting metrics that all three demos print.
"""
from __future__ import annotations

import argparse
import math
import os
import pathlib

import torch
from PIL import Image

import ttnn
from models.common.utility_functions import comp_pcc

# --- device conventions (must match the bring-up's sharded PCC tests) ----------------
L1_SMALL_SIZE = 24576
DEFAULT_TP = int(os.environ.get("TT_HW_PLANNER_SHARD_TP", "8"))
DEFAULT_OUTPUT_DIR = "flux2_vae_demo_out"
PCC_TARGET = 0.95

# Images produced/consumed by VaeImageProcessor live in [-1, 1]; PSNR peak-to-peak is 2.0.
IMAGE_DATA_RANGE = 2.0

TAG = "[flux2-vae-demo]"


def build_arg_parser(task: str) -> argparse.ArgumentParser:
    """The flag set shared by all three demos."""
    parser = argparse.ArgumentParser(
        prog=f"demo_{task}",
        description=f"FLUX.2 Klein 9B VAE — {task} demo (TT device vs the HF golden).",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an input image (default: reference.load_input_image()'s built-in).",
    )
    parser.add_argument("--size", type=int, default=224, help="Pinned square image side; the latent side is size/8.")
    parser.add_argument("--tp", type=int, default=DEFAULT_TP, help="Tensor-parallel degree; mesh is 1 x TP.")
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory for the emitted artifacts."
    )
    parser.add_argument("--layers", type=int, default=None, help="Cap resnets-per-repeated-block (None = every layer).")
    return parser


def ensure_output_dir(path: str) -> pathlib.Path:
    out = pathlib.Path(path).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def open_mesh(tp: int):
    """Open a 1 x TP mesh with the fabric enabled.

    ``ttnn.open_mesh_device`` in this checkout takes no ``fabric_config`` argument, so the
    fabric is enabled the way conftest's ``set_fabric`` does it: a ``set_fabric_config``
    call BEFORE the mesh is opened. Without it every CCL in the sharded stubs raises
    ``TT_FATAL ... fabric_context_ != nullptr``.
    """
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, int(tp)), l1_small_size=L1_SMALL_SIZE)


def close_mesh(mesh_device) -> None:
    try:
        for submesh in mesh_device.get_submeshes():
            ttnn.close_mesh_device(submesh)
    except Exception:
        pass
    ttnn.close_mesh_device(mesh_device)
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    except Exception:
        pass


# --- metrics -------------------------------------------------------------------------
def max_abs_err(golden: torch.Tensor, tt: torch.Tensor) -> float:
    return float((golden.detach().float() - tt.detach().float()).abs().max().item())


def psnr(a: torch.Tensor, b: torch.Tensor, data_range: float = IMAGE_DATA_RANGE) -> float:
    """Peak signal-to-noise ratio in dB between two [-1, 1] image tensors."""
    mse = float(((a.detach().float() - b.detach().float()) ** 2).mean().item())
    if mse <= 0.0:
        return float("inf")
    return 10.0 * math.log10((data_range * data_range) / mse)


def report_pcc(label: str, golden: torch.Tensor, tt: torch.Tensor, target: float = PCC_TARGET) -> float:
    """Print the achieved PCC (on its own line) and the max abs err. Returns the PCC."""
    golden32 = golden.detach().float()
    tt32 = tt.detach().float()
    ok, pcc = comp_pcc(golden32, tt32, target)
    print(f"{TAG} {label}: shape={tuple(tt32.shape)} target={target}", flush=True)
    print(f"e2e PCC={pcc}", flush=True)
    print(f"{TAG} {label}: max_abs_err={max_abs_err(golden32, tt32)} pass={bool(ok)}", flush=True)
    return float(pcc)


def print_ledger(pipeline) -> dict:
    """Print the passive invocation ledger of graduated stubs (Gate 2 evidence)."""
    ledger = dict(pipeline.invoked_modules())
    print(f"{TAG} graduated modules invoked ({len(ledger)}):", flush=True)
    for name in sorted(ledger):
        print(f"{TAG}   {name} x{ledger[name]}", flush=True)
    return ledger


def save_image(sample: torch.Tensor, path: pathlib.Path, reference_module) -> pathlib.Path:
    """Write one [1,3,H,W] sample in [-1,1] through VaeImageProcessor.postprocess."""
    reference_module.postprocess_image(sample.detach().float())[0].save(path)
    print(f"{TAG} wrote {path}", flush=True)
    return path


def save_latent_preview(latent: torch.Tensor, path: pathlib.Path, size: int) -> pathlib.Path:
    """Per-channel-mean of the latent, min-max normalised to an 8-bit grayscale PNG."""
    plane = latent.detach().float()[0].mean(dim=0)
    lo = float(plane.min().item())
    hi = float(plane.max().item())
    plane = (plane - lo) / (hi - lo + 1e-8)
    array = (plane.clamp(0.0, 1.0).numpy() * 255.0).round().astype("uint8")
    Image.fromarray(array, mode="L").resize((int(size), int(size)), Image.NEAREST).save(path)
    print(f"{TAG} wrote {path}", flush=True)
    return path


def describe_latent(latent: torch.Tensor) -> None:
    lat = latent.detach().float()
    print(
        f"{TAG} latent: shape={tuple(lat.shape)} dtype={lat.dtype} "
        f"mean={float(lat.mean().item()):.6f} std={float(lat.std().item()):.6f} "
        f"min={float(lat.min().item()):.6f} max={float(lat.max().item()):.6f}",
        flush=True,
    )
