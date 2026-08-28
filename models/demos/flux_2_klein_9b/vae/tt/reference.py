# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Source-A (HuggingFace) reference side of the FLUX.2 Klein 9B VAE bring-up.

This module owns *everything that is not TTNN*: loading the golden fp32
reference model, building its inputs the way HF builds them, turning its
outputs back into images, reading the bring-up capture, and computing the
three goldens the TT pipeline is scored against.

Why the loader is diffusers-native and not `AutoModel`
------------------------------------------------------
The checkpoint at ``HF_MODEL_ID`` ships real weights
(``diffusion_pytorch_model.safetensors``, 251 tensors, 84.0M params) but it is
**not** a transformers checkpoint. Its ``config.json`` has no ``model_type``
key -- it carries::

    {"_class_name": "AutoencoderKLFlux2", "_diffusers_version": "0.37.0.dev0",
     "_name_or_path": "black-forest-labs/FLUX.2-dev", ...}

``transformers.AutoConfig``/``AutoModel`` dispatch on ``model_type`` against
the transformers model registry, which has no entry for ``AutoencoderKLFlux2``
and never will: the architecture lives in *diffusers*, in that package's own
``ModelMixin`` registry. Calling ``AutoModel.from_pretrained`` here fails with
"Unrecognized model ... Should have a `model_type` key", which reads like
"no weights" but really means "wrong registry". The correct loader is the
model package's native one::

    diffusers.AutoencoderKLFlux2.from_pretrained(path, torch_dtype=torch.float32)

``AutoencoderKLFlux2`` first appears in diffusers **0.37**; older releases
cannot represent it at all. Rather than re-implement that version handling,
this module delegates to the bring-up's loader,
``models/tt_dit/pipelines/flux_2_klein_9b_vae/tests/pcc/_reference_loader.py``,
which already resolves ``_class_name`` from the config and stages a private,
version-matched diffusers 0.37.1 (``pip install --no-deps --target``) only when
the ambient one is too old. The ambient diffusers in ``./python_env`` is 0.38.0
and already exposes the class, so on this box nothing is staged and nothing is
shadowed. That loader is reached by *file path* (not ``import
models.tt_dit....``) because there is no ``__init__.py`` at
``pipelines/flux_2_klein_9b_vae/`` or below, and because importing through
``models.tt_dit.pipelines`` would execute that package's ``__init__``, dragging
in the TT-DiT pipeline machinery for what is a pure-CPU, pure-torch loader.

There is no tokenizer and no feature extractor: this is an image autoencoder.
Its HF-side input construction is
``diffusers.image_processor.VaeImageProcessor.preprocess(image, height, width)``
-> float32 ``[1, 3, H, W]`` in ``[-1, 1]``, and its HF-side output construction
is ``VaeImageProcessor.postprocess(sample)`` -> PIL. Both are used verbatim
here; the normalisation is never hand-rolled.

Task heads (all deterministic -- no RNG, no ``generate()``)
-----------------------------------------------------------
====================  ==========================================
``hf_reference_encode``       ``model.encode(x).latent_dist.mode()``
``hf_reference_decode``       ``model.decode(z).sample``
``hf_reference_reconstruct``  ``model(x).sample``  (``sample_posterior=False``)
====================  ==========================================

``AutoencoderKLFlux2.forward`` defaults to ``sample_posterior=False``, i.e. it
takes the posterior *mode* (== the mean == the first 32 channels of
``quant_conv(encoder(x))``), so a top-level forward draws no random numbers and
is bit-reproducible across runs.

Import safety
-------------
Importing this module has no side effects: no weight loading, no network
access, no ``sys.path`` mutation, no device acquisition. ``torch``,
``diffusers`` and ``PIL`` are imported lazily inside the functions that need
them, and the reference model and image processor are built on first use and
cached at module level.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from pathlib import Path

__all__ = [
    "HF_MODEL_ID",
    "BRINGUP_DIR",
    "IMAGE_SIZE",
    "LATENT_SIZE",
    "VAE_SCALE_FACTOR",
    "SAMPLE_IMAGE_CANDIDATES",
    "load_reference_model",
    "image_processor",
    "load_input_image",
    "preprocess_image",
    "postprocess_image",
    "captured_tensor",
    "hf_reference_encode",
    "hf_reference_decode",
    "hf_reference_reconstruct",
]

# --------------------------------------------------------------------------
# Frozen constants (other agents code against these)
# --------------------------------------------------------------------------

#: The diffusers checkpoint under test.
# Resolved from the HuggingFace repo rather than a machine-local path.
# Override with TT_FLUX2_KLEIN_VAE to point at an existing local snapshot
# (skips the Hub round-trip and makes runs deterministic).
_HF_REPO = "black-forest-labs/FLUX.2-klein-9B"
_HF_SUBFOLDER = "vae"


def _resolve_checkpoint() -> str:
    override = os.environ.get("TT_FLUX2_KLEIN_VAE")
    if override:
        return override
    from huggingface_hub import snapshot_download

    root = snapshot_download(_HF_REPO, allow_patterns=[f"{_HF_SUBFOLDER}/*"])
    return os.path.join(root, _HF_SUBFOLDER)


HF_MODEL_ID: str = _resolve_checkpoint()

# .../models/demos/flux_2_klein_9b_vae/tt/reference.py -> .../models
_MODELS_ROOT: Path = Path(__file__).resolve().parents[3]

#: The bring-up package: graduated TTNN stubs, captured goldens, run report.
BRINGUP_DIR: Path = _MODELS_ROOT / "tt_dit" / "pipelines" / "flux_2_klein_9b_vae"

#: Image side the bring-up capture used (``_captured/encoder/args.pt`` is
#: ``[1, 3, 224, 224]``). Everything downstream is pinned to it.
IMAGE_SIZE: int = 224

#: Spatial compression of this VAE (``spatial_compression: 8`` in config.json).
VAE_SCALE_FACTOR: int = 8

#: Latent side at the pinned capacity (``_captured/decoder/args.pt`` is
#: ``[1, 32, 28, 28]``).
LATENT_SIZE: int = IMAGE_SIZE // VAE_SCALE_FACTOR

#: Real photographs already in this repo, best first. ``load_input_image()``
#: takes the first one that exists. These live in the shared
#: ``models/sample_data/`` pool, not in another model's demo directory.
SAMPLE_IMAGE_CANDIDATES: tuple[Path, ...] = (
    _MODELS_ROOT / "sample_data" / "demo.jpeg",  # 2048x1365 photo: woman + dog on a beach
    _MODELS_ROOT / "sample_data" / "huggingface_cat_image.jpg",  # 640x480 photo: two cats
    _MODELS_ROOT / "sample_data" / "ILSVRC2012_val_00048736.JPEG",  # 500x425 ImageNet photo
    _MODELS_ROOT / "sample_data" / "house_in_field_1080p.jpg",  # 1600x900 photo: house in a field
)

_BRINGUP_LOADER_PATH: Path = BRINGUP_DIR / "tests" / "pcc" / "_reference_loader.py"
_BRINGUP_LOADER_MODNAME = "_flux_2_klein_9b_vae_reference_loader"
_CAPTURED_DIR: Path = BRINGUP_DIR / "_captured"

_lock = threading.Lock()
_cache: dict = {}


# --------------------------------------------------------------------------
# Reference model
# --------------------------------------------------------------------------


def _bringup_loader():
    """Import the bring-up's `_reference_loader.py` by file path.

    There is no ``__init__.py`` at ``models/tt_dit/pipelines/flux_2_klein_9b_vae``
    or below it, and its nearest packaged ancestor
    (``models.tt_dit.pipelines``) has an ``__init__`` that pulls in the TT-DiT
    pipeline machinery. A file-path import gets the loader without either
    problem, and without any dependency on how ``PYTHONPATH`` is set.
    """
    cached = sys.modules.get(_BRINGUP_LOADER_MODNAME)
    if cached is not None:
        return cached

    if not _BRINGUP_LOADER_PATH.is_file():
        raise FileNotFoundError(
            f"bring-up reference loader not found at {_BRINGUP_LOADER_PATH}. "
            "It is the shared diffusers-native loader for "
            f"{HF_MODEL_ID}; models/demos/flux_2_klein_9b_vae reuses it rather "
            "than duplicating the diffusers-version handling."
        )

    spec = importlib.util.spec_from_file_location(_BRINGUP_LOADER_MODNAME, _BRINGUP_LOADER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build an import spec for {_BRINGUP_LOADER_PATH}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so a re-entrant import sees the same object.
    sys.modules[_BRINGUP_LOADER_MODNAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_BRINGUP_LOADER_MODNAME, None)
        raise
    return module


def load_reference_model():
    """Return the fp32 HF golden `AutoencoderKLFlux2`, cached at module level.

    ``eval()``, ``requires_grad_(False)``, float32, on CPU. Loaded through the
    bring-up's ``_reference_loader.load_reference_model`` (see the module
    docstring for why that is a diffusers call and not ``AutoModel``).
    """
    with _lock:
        model = _cache.get("model")
        if model is not None:
            return model

        model = _bringup_loader().load_reference_model(HF_MODEL_ID)
        model.eval()
        model.requires_grad_(False)

        _cache["model"] = model
        return model


# --------------------------------------------------------------------------
# HF-side input / output construction
# --------------------------------------------------------------------------


def image_processor():
    """Return the cached `VaeImageProcessor` for this VAE (scale factor 8).

    This is the model's processor: an image autoencoder has no tokenizer and no
    feature extractor, and ``VaeImageProcessor`` is what the FLUX.2 pipelines
    use to turn a PIL image into the VAE's input tensor and back again.
    """
    with _lock:
        proc = _cache.get("image_processor")
        if proc is not None:
            return proc

        from diffusers.image_processor import VaeImageProcessor

        proc = VaeImageProcessor(vae_scale_factor=VAE_SCALE_FACTOR)
        _cache["image_processor"] = proc
        return proc


def _synthetic_image(size: int):
    """Deterministic, *structured* RGB image used only if the repo has none.

    Smooth colour gradients plus a handful of geometric shapes (discs, bars, a
    checker patch) -- deliberately not white noise, because a VAE round trip
    over noise is visually meaningless and PCC over noise hides real wiring
    bugs. Shape placement is drawn from a seeded generator, so the image is
    identical on every call and every machine.
    """
    import numpy as np
    import torch
    from PIL import Image

    gen = torch.Generator().manual_seed(20260828)

    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, size),
        torch.linspace(0.0, 1.0, size),
        indexing="ij",
    )

    # Base: three smooth, mutually different gradients.
    r = 0.5 + 0.45 * torch.sin(3.0 * torch.pi * xx) * torch.cos(1.5 * torch.pi * yy)
    g = 0.5 + 0.45 * (xx * 0.6 + yy * 0.4 - 0.5)
    b = 0.5 + 0.45 * torch.cos(2.0 * torch.pi * (xx + yy))
    img = torch.stack([r, g, b], dim=0).clamp(0.0, 1.0)

    # A few filled discs.
    for _ in range(5):
        cx, cy = torch.rand(2, generator=gen).tolist()
        rad = 0.06 + 0.10 * float(torch.rand(1, generator=gen))
        colour = torch.rand(3, generator=gen)
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= rad**2
        img = torch.where(mask.unsqueeze(0), colour.view(3, 1, 1).expand_as(img), img)

    # Two hard-edged bars (high-frequency edges the VAE must preserve).
    img[:, int(0.12 * size) : int(0.18 * size), :] = 0.05
    img[:, :, int(0.72 * size) : int(0.76 * size)] = 0.95

    # A checkerboard patch in the lower-left quadrant.
    cell = max(size // 28, 1)
    checker = (((yy * size).long() // cell + (xx * size).long() // cell) % 2).float()
    patch = (yy > 0.60) & (yy < 0.92) & (xx > 0.08) & (xx < 0.40)
    img = torch.where(patch.unsqueeze(0), checker.unsqueeze(0).expand_as(img), img)

    array = (img.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(np.ascontiguousarray(array), mode="RGB")


def load_input_image(path: str | Path | None = None, size: int = IMAGE_SIZE):
    """Return a `size x size` RGB PIL image to run the VAE round trip on.

    ``path`` -- if given, that file is opened. Otherwise the first existing
    entry of ``SAMPLE_IMAGE_CANDIDATES`` is used; on this checkout that is
    ``models/sample_data/demo.jpeg`` (a 2048x1365 photograph of a woman and a
    dog on a beach, from the repo's shared sample-data pool -- a real photo
    with skin, fur, sand and sky texture, not an icon or a diagram; it
    round-trips through the fp32 reference at ~35 dB PSNR at 224x224). If no
    repo image exists at all, a deterministic structured image is synthesised
    instead (see ``_synthetic_image``).

    Note that this is *not* the tensor the bring-up capture used: the golden
    under ``_captured/encoder/args.pt`` is a standard-normal tensor, not a
    photograph. Use ``captured_tensor`` when you want to reproduce the capture
    exactly and this function when you want the behavioural round trip.

    The image is converted to RGB and centre-cropped to a square before being
    resized to ``size x size``, so a wide photo is not squashed -- an
    aspect-distorted input makes the reconstruction hard to judge by eye.
    """
    from PIL import Image

    if path is not None:
        image = Image.open(path)
    else:
        chosen = next((p for p in SAMPLE_IMAGE_CANDIDATES if p.is_file()), None)
        image = Image.open(chosen) if chosen is not None else _synthetic_image(size)

    image = image.convert("RGB")

    # Centre crop to square, then resize to the pinned capacity.
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    if image.size != (size, size):
        image = image.resize((size, size), Image.LANCZOS)
    return image


def preprocess_image(image, size: int = IMAGE_SIZE):
    """PIL image -> float32 `[1, 3, size, size]` in `[-1, 1]`.

    Built by ``VaeImageProcessor.preprocess(image, height=size, width=size)``
    -- the HF-side input construction for this model. The normalisation is
    never hand-rolled here; whatever the processor does is by definition what
    the golden sees.
    """
    import torch

    pixel_values = image_processor().preprocess(image, height=size, width=size)
    return pixel_values.to(dtype=torch.float32)


def postprocess_image(sample):
    """VAE output tensor `[N, 3, H, W]` -> `list[PIL.Image.Image]`.

    ``VaeImageProcessor.postprocess(sample, output_type="pil")`` -- the HF-side
    output construction (denormalise from ``[-1, 1]``, clamp, to PIL).
    """
    import torch

    if not torch.is_tensor(sample):
        raise TypeError(f"postprocess_image expects a torch.Tensor, got {type(sample).__name__}")
    tensor = sample.detach().to(device="cpu", dtype=torch.float32)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return image_processor().postprocess(tensor, output_type="pil")


# --------------------------------------------------------------------------
# Bring-up capture
# --------------------------------------------------------------------------


def captured_tensor(component: str, which: str = "args", index: int = 0):
    """Load a golden tensor from `BRINGUP_DIR/_captured/<component>/<which>.pt`.

    ``which="args"`` files hold a tuple/list of positional inputs -- element
    ``index`` is returned. ``which="output"`` files hold a bare tensor, but a
    tuple is accepted too (some modules were captured returning one). The
    result is always a float32 CPU tensor.
    """
    import torch

    path = _CAPTURED_DIR / component / f"{which}.pt"
    if not path.is_file():
        available = sorted(p.name for p in _CAPTURED_DIR.iterdir() if p.is_dir()) if _CAPTURED_DIR.is_dir() else []
        raise FileNotFoundError(
            f"captured tensor not found: {path}. " f"Captured components under {_CAPTURED_DIR}: {available or '<none>'}"
        )

    obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, dict):
        keys = list(obj)
        key = index if isinstance(index, str) else (keys[index] if index < len(keys) else None)
        if key not in obj:
            raise KeyError(f"{path} has no entry {index!r}; keys are {keys}")
        obj = obj[key]
    elif isinstance(obj, (tuple, list)):
        if not (-len(obj) <= index < len(obj)):
            raise IndexError(f"{path} holds {len(obj)} item(s); index {index} is out of range")
        obj = obj[index]

    if not torch.is_tensor(obj):
        raise TypeError(f"{path}[{index}] is a {type(obj).__name__}, not a torch.Tensor")
    return obj.detach().to(device="cpu", dtype=torch.float32)


# --------------------------------------------------------------------------
# Goldens
# --------------------------------------------------------------------------


def _as_fp32_cpu(tensor, name: str):
    import torch

    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    return tensor.detach().to(device="cpu", dtype=torch.float32)


def hf_reference_encode(pixel_values):
    """`[1, 3, H, W]` -> latent `[1, 32, H/8, W/8]` == `encode(x).latent_dist.mode()`.

    The posterior *mode*, not a sample: deterministic, no RNG. For a
    ``DiagonalGaussianDistribution`` the mode is the mean, i.e. the first 32
    channels of ``quant_conv(encoder(x))``.
    """
    import torch

    model = load_reference_model()
    x = _as_fp32_cpu(pixel_values, "pixel_values")
    with torch.no_grad():
        latent = model.encode(x).latent_dist.mode()
    return latent.detach().to(device="cpu", dtype=torch.float32)


def hf_reference_decode(latent):
    """Latent `[1, 32, h, w]` -> image tensor `[1, 3, 8h, 8w]` == `decode(z).sample`."""
    import torch

    model = load_reference_model()
    z = _as_fp32_cpu(latent, "latent")
    with torch.no_grad():
        sample = model.decode(z).sample
    return sample.detach().to(device="cpu", dtype=torch.float32)


def hf_reference_reconstruct(pixel_values):
    """`[1, 3, H, W]` -> reconstruction `[1, 3, H, W]` == `model(x).sample`.

    The model's own forward. ``sample_posterior`` defaults to ``False``, so
    this is ``decode(encode(x).mode())`` with no random draw anywhere.
    """
    import torch

    model = load_reference_model()
    x = _as_fp32_cpu(pixel_values, "pixel_values")
    with torch.no_grad():
        sample = model(x).sample
    return sample.detach().to(device="cpu", dtype=torch.float32)


if __name__ == "__main__":
    import torch

    _model = load_reference_model()
    print(
        f"{type(_model).__name__}  params={sum(p.numel() for p in _model.parameters()):,}  training={_model.training}"
    )
    _image = load_input_image()
    _px = preprocess_image(_image)
    print(f"input {_image.size} {_image.mode} -> {tuple(_px.shape)} {_px.dtype} [{_px.min():.3f}, {_px.max():.3f}]")
    _z = hf_reference_encode(_px)
    print(f"latent {tuple(_z.shape)} mean={_z.mean():.4f} std={_z.std():.4f}")
    _recon = hf_reference_reconstruct(_px)
    _mse = torch.mean((_recon.clamp(-1, 1) - _px) ** 2)
    print(f"recon  {tuple(_recon.shape)}  PSNR vs input = {10 * torch.log10(4.0 / _mse):.2f} dB")
