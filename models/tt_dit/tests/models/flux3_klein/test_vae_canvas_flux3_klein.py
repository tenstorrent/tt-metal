# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Klein reference encoder on fixed canvases. Host: every prepared reference fits a canvas, and the masked
GroupNorm and mid-block attention on a canvas equal torch's on the image alone (ttnn ops run as torch ops). Device: after
one encode per canvas, references of other shapes compile nothing and match the torch encoder on the reference alone."""

from __future__ import annotations

import time
import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image

import ttnn

from ....models.vae import canvas_flux3_klein as cv
from ....models.vae.canvas_flux3_klein import REFERENCE_CANVASES, canvas_for
from ....models.vae.vae_flux3_klein import (
    BLOCK_OUT_CHANNELS,
    ENCODER_SLICE_PARAMS,
    FULL_CANVAS,
    PATCH,
    SCALE,
    SUBFOLDER,
    Z_CHANNELS,
    Flux3KleinVae,
)
from ....parallel.manager import CCLManager
from ....pipelines.flux3_klein.packing import DOWNSAMPLING_RATIO
from ....pipelines.flux3_klein.pipeline_flux3_klein import MATCH_MAX_PIXELS, MAX_SIDE, MIN_SIDE, prepare_reference
from ....pipelines.flux3_klein.weights_flux3_klein import resolve_weights_dir
from ....utils.test import ring_params_8k

# (width, height) before prepare_reference: every canvas, inside and at its edges, the smallest reference, odd sizes
REFERENCE_SIZES = (
    (64, 64),
    (1000, 750),
    (1024, 1024),
    (1280, 960),
    (720, 1280),
    (2048, 2048),
    (2720, 1536),
    (1530, 2720),
    (2560, 1600),
)


def test_every_prepared_reference_fits_a_canvas():
    sides = range(MIN_SIDE, MAX_SIDE + 1, DOWNSAMPLING_RATIO)
    for w in sides:
        for h in sides:
            if w * h <= MATCH_MAX_PIXELS:
                cw, ch = canvas_for(w, h)
                assert w <= cw and h <= ch
    assert all(cw % 256 == 0 and ch % 256 == 0 for cw, ch in REFERENCE_CANVASES)


@pytest.mark.parametrize("size, canvas", [((2720, 1536), (2816, 1536)), ((1530, 2720), (1536, 2816))])
def test_4mp_wide_and_tall_references_take_the_tight_canvases(size, canvas):
    image = prepare_reference(Image.new("RGB", size))
    assert canvas_for(*image.size) == canvas
    assert canvas[0] * canvas[1] < 1.05 * image.width * image.height


@pytest.fixture
def torch_ttnn(monkeypatch):
    """canvas_flux3_klein's ttnn and uploads as torch: tensors stay torch tensors (fp32 for both dtypes)."""
    silu, square = object(), object()
    unary = {silu: F.silu, square: torch.square}

    def binary(op):
        def run(a, b, *, activations=()):
            out = op(a, b)
            for f in activations:
                out = unary[f](out)
            return out

        return run

    def inplace(op):
        return lambda a, b, **kwargs: a.copy_(binary(op)(a, b, **kwargs))

    stub = types.SimpleNamespace(
        float32=torch.float32,
        bfloat16=torch.float32,
        TILE_LAYOUT=torch.strided,
        UnaryWithParam=lambda op: op,
        UnaryOpType=types.SimpleNamespace(SILU=silu, SQUARE=square),
        typecast=lambda x, dtype: x.to(dtype),
        multiply=binary(torch.mul),
        subtract=binary(torch.sub),
        add=binary(torch.add),
        multiply_=inplace(torch.mul),
        add_=inplace(torch.add),
        deallocate=lambda t: None,
        rsqrt=torch.rsqrt,
        sum=lambda x, dim, keepdim, compute_kernel_config=None: x.sum(dim=dim, keepdim=keepdim),
        reshape=lambda x, shape: x.reshape(shape),
        matmul=lambda a, b, compute_kernel_config=None: a @ b,
        concat=lambda ts, dim: torch.cat(ts, dim=dim),
        full_like=lambda t, value: torch.full_like(t, value),
        zeros_like=torch.zeros_like,
        slice=lambda x, start, end: x[tuple(slice(a, b) for a, b in zip(start, end))],
        transformer=types.SimpleNamespace(
            scaled_dot_product_attention=lambda q, k, v, *, scale, **kwargs: F.scaled_dot_product_attention(
                q, k, v, scale=scale
            )
        ),
    )
    monkeypatch.setattr(cv, "ttnn", stub)
    monkeypatch.setattr(cv.tensor, "from_torch", lambda t, **kwargs: t.float().clone())


CANVAS, IMAGE = (64, 48), (40, 24)  # (width, height); the image is 5 x 3 latent pixels at level 3


def _norm(channels: int, silu: bool) -> tuple[cv.MaskedGroupNorm, dict]:
    norm = object.__new__(cv.MaskedGroupNorm)
    norm.__dict__.update(num_channels=channels, _local=channels, _silu=silu, _fp32_config=None)
    state = {"weight": torch.randn(channels), "bias": torch.randn(channels)}
    norm._prepare_torch_state(state)
    norm.__dict__.update({k: types.SimpleNamespace(data=v) for k, v in state.items()})
    return norm, state


@pytest.mark.parametrize("silu", [True, False])
def test_masked_group_norm_is_group_norm_on_the_image(torch_ttnn, silu):
    torch.manual_seed(0)
    (cw, ch), (iw, ih), c = CANVAS, IMAGE, 64
    norm, state = _norm(c, silu)
    masks = cv.CanvasMasks(None, canvas=CANVAS, image=IMAGE)
    x = torch.randn(1, ch, cw, c) * 3 + 1
    out = norm.forward(x, masks(x))
    crop = x[:, :ih, :iw].permute(0, 3, 1, 2)
    ref = F.group_norm(crop, cv.GROUPS, state["weight"].flatten(), state["bias"].flatten(), eps=cv.GROUP_NORM_EPS)
    ref = (F.silu(ref) if silu else ref).permute(0, 2, 3, 1)
    torch.testing.assert_close(out[:, :ih, :iw], ref, atol=1e-4, rtol=1e-4)
    assert not out[:, ih:].any() and not out[:, :, iw:].any()


def test_masks_free_every_tensor_once(torch_ttnn):
    freed = []
    cv.ttnn.deallocate = freed.append
    masks = cv.CanvasMasks(None, canvas=CANVAS, image=IMAGE)
    masks.deallocate()
    assert len(freed) == len({id(t) for t in freed}) == cv.LEVELS * 4 + 1


def test_masked_attention_is_attention_on_the_image(torch_ttnn):
    torch.manual_seed(0)
    (cw, ch), (iw, ih), c = CANVAS, IMAGE, 64
    w, h, vw, vh = cw >> 3, ch >> 3, iw >> 3, ih >> 3
    wq, wk, wv, wo = (torch.randn(c, c) / c**0.5 for _ in range(4))
    attn = types.SimpleNamespace(
        norm=types.SimpleNamespace(forward=lambda x, mask: x),
        to_qkv=types.SimpleNamespace(forward=lambda x, **kwargs: (x @ wq, x @ wk, x @ wv)),
        to_out=types.SimpleNamespace(forward=lambda x, **kwargs: x @ wo),
        _ccl_manager=None,
        _tp_axis=None,
        _sdpa_program_config=None,
        _sdpa_compute_kernel_config=None,
        _mm_compute_kernel_config=None,
    )
    masks = cv.CanvasMasks(None, canvas=CANVAS, image=IMAGE)
    x = torch.randn(1, h, w, c)
    out = cv.CanvasVaeEncoder._attention(None, attn, x, masks)
    tokens = x[:, :vh, :vw].reshape(1, 1, vh * vw, c)
    ref = F.scaled_dot_product_attention(tokens @ wq, tokens @ wk, tokens @ wv) @ wo
    torch.testing.assert_close(out[:, :vh, :vw], ref.reshape(1, vh, vw, c) + x[:, :vh, :vw], atol=1e-4, rtol=1e-4)


def test_banded_is_whole(torch_ttnn, monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setattr(cv, "WHOLE_ELEMENTS", 0)
    monkeypatch.setattr(cv, "BAND_ELEMENTS", 64 * 32 * 8)
    weight = torch.randn(8, 8, 3, 3)
    conv = types.SimpleNamespace(
        forward=lambda x: F.conv2d(x.permute(0, 3, 1, 2), weight, padding=1).permute(0, 2, 3, 1).contiguous()
    )
    x = torch.randn(1, 80, 64, 8)  # bands of 32, 32 and 16 rows
    torch.testing.assert_close(cv.CanvasVaeEncoder._banded(None, conv.forward, x, 8), conv.forward(x))


def _encoder_conv_keys(canvas: tuple[int, int]) -> set[tuple[int, int, int, int, int]]:
    """(h, w, in, out, stride) of every conv forward_canvas runs on a (width, height) canvas, row bands included."""
    keys = set()

    def conv(cin, cout, stride=1):
        def forward(x):
            _, h, w, _ = x.shape
            keys.add((h, w, cin, cout, stride))
            return torch.empty(1, h // stride, w // stride, cout, device="meta")

        return types.SimpleNamespace(forward=forward, inner=types.SimpleNamespace(out_channels=cout))

    def resnet(cin, cout):
        def shortcut(x, **kwargs):
            return torch.empty(*x.shape[:3], cout, device="meta")

        identity = types.SimpleNamespace(forward=lambda x, mask: x)
        return types.SimpleNamespace(
            norm1=identity,
            conv1=conv(cin, cout),
            norm2=identity,
            conv2=conv(cout, cout),
            conv_shortcut=types.SimpleNamespace(forward=shortcut) if cin != cout else None,
        )

    blocks, cin = [], BLOCK_OUT_CHANNELS[0]
    for level, cout in enumerate(BLOCK_OUT_CHANNELS):
        downsampler = conv(cout, cout, stride=2) if level < len(BLOCK_OUT_CHANNELS) - 1 else None
        blocks.append(types.SimpleNamespace(resnets=[resnet(cin, cout), resnet(cout, cout)], downsampler=downsampler))
        cin = cout
    encoder = types.SimpleNamespace(
        conv_in=conv(3, BLOCK_OUT_CHANNELS[0]),
        down_blocks=blocks,
        mid_block=types.SimpleNamespace(resnets=[resnet(cin, cin), resnet(cin, cin)], attentions=[None]),
        conv_norm_out=types.SimpleNamespace(forward=lambda x, mask: x),
        conv_out=conv(cin, Z_CHANNELS),
        _ctx=types.SimpleNamespace(ccl_manager=None),
    )
    encoder._banded = lambda *args: cv.CanvasVaeEncoder._banded(encoder, *args)
    encoder._resnet = lambda *args: cv.CanvasVaeEncoder._resnet(encoder, *args)
    encoder._attention = lambda attn, x, masks: x
    cw, ch = canvas
    image = torch.empty(1, ch, cw, 3, device="meta")
    cv.CanvasVaeEncoder.forward_canvas(encoder, image, lambda x: types.SimpleNamespace(apply=lambda t: t))
    return keys


def test_pinned_encoder_slicing_is_on_canvas_convs(torch_ttnn):
    """Every pinned key is a stride-1 conv of some canvas, so no downsampler sharing the key changes its slicing."""
    convs = set().union(*(_encoder_conv_keys(canvas) for canvas in REFERENCE_CANVASES))
    for table in ENCODER_SLICE_PARAMS.values():
        for key in table:
            assert {stride for *k, stride in convs if tuple(k) == key} == {1}, key


def _pattern(width: int, height: int) -> Image.Image:
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    rgb = np.stack([x / width, y / height, 0.5 + 0.5 * np.sin(x / 7.0) * np.cos(y / 11.0)], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def _torch_tokens(torch_vae, vae: Flux3KleinVae, image: Image.Image) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(image)).float().permute(2, 0, 1)[None] / 127.5 - 1.0
    with torch.no_grad():
        mean = torch_vae.encode(x).latent_dist.mean.permute(0, 2, 3, 1)
    b, h, w, c = mean.shape
    tokens = mean.reshape(b, h // PATCH, PATCH, w // PATCH, PATCH, c).permute(0, 1, 3, 5, 2, 4).flatten(3).flatten(1, 2)
    return (tokens[0] - vae._bn_mean) / vae._bn_std


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{**ring_params_8k, "l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_vae_canvas(*, mesh_device: ttnn.MeshDevice) -> None:
    from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2

    snapshot = resolve_weights_dir("distilled")
    ccl = CCLManager(mesh_device=mesh_device, num_links=2, topology=ttnn.Topology.Ring)
    vae = Flux3KleinVae(
        snapshot, device=mesh_device, ccl_manager=ccl, dit_ccl_manager=ccl, dit_sp_axis=1, sizes=(1024,)
    )
    for width, height in REFERENCE_CANVASES:
        vae.encode(Image.new("RGB", (width - 16, height)))
    vae.encode(Image.new("RGB", FULL_CANVAS))
    torch_vae = AutoencoderKLFlux2.from_pretrained(str(snapshot), subfolder=SUBFOLDER).eval()

    failures = []
    for size in REFERENCE_SIZES:
        image = prepare_reference(_pattern(*size))
        ttnn.synchronize_device(mesh_device)
        programs = mesh_device.num_program_cache_entries()
        t0 = time.perf_counter()
        tokens = vae.encode(image)
        ttnn.synchronize_device(mesh_device)
        seconds = time.perf_counter() - t0
        new_programs = mesh_device.num_program_cache_entries() - programs
        expected = _torch_tokens(torch_vae, vae, image)
        assert (
            tokens.shape == expected.shape == ((image.height // SCALE // PATCH) * (image.width // SCALE // PATCH), 128)
        )
        pcc = torch.corrcoef(torch.stack([tokens.float().flatten(), expected.flatten()]))[0, 1].item()
        tag = f"{size} -> {image.size} on canvas {canvas_for(*image.size)}"
        logger.info(f"CANVAS {tag}: new programs {new_programs}, pcc {pcc:.5f}, encode {seconds:.3f} s")
        if new_programs:
            failures.append(f"{tag}: {new_programs} new programs")
        if pcc < 0.995:
            failures.append(f"{tag}: pcc {pcc:.5f} vs the torch encoder")
    vae.deallocate_weights()
    assert not failures, "\n".join(failures)
