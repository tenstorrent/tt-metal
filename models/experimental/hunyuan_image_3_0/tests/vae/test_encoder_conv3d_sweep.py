# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Brute-force conv3d blocking sweep for Hunyuan VAE encoder hot paths.

The encoder runs on a replicated 2×2 mesh (``h_factor=1, w_factor=1``). Large
spatial convs are H-chunked inside ``HunyuanSymmetricConv3d``; each chunk calls
``get_conv3d_config`` with its slice ``(T, H, W)``. This sweep benchmarks every
priority shape (full tensor + representative chunk strips) and writes JSON with
``best_blocking`` for hand-off into ``tt/vae/conv3d_blockings.py`` ``HUNYUAN_EXACT_BLOCKINGS``.

Run on BH 2×2 (priority — conv_in + early down at 1024²):

    python_env/bin/python -m pytest \\
      models/experimental/hunyuan_image_3_0/tests/vae/test_encoder_conv3d_sweep.py \\
      -k "bh_2x2 and enc_1024 and conv_in" -s --timeout=0

Full 1024² encoder sweep (all layers + chunk strips):

    python_env/bin/python -m pytest \\
      models/experimental/hunyuan_image_3_0/tests/vae/test_encoder_conv3d_sweep.py \\
      -k "bh_2x2 and enc_1024" -s --timeout=0

512² early-down shapes:

    python_env/bin/python -m pytest \\
      models/experimental/hunyuan_image_3_0/tests/vae/test_encoder_conv3d_sweep.py \\
      -k "bh_2x2 and enc_512" -s --timeout=0

Environment overrides (same names as ``bruteforce_conv3d_sweep.py``):

    HY_CONV3D_SWEEP_H=1024 HY_CONV3D_SWEEP_W=1024   # pixel spatial (default 1024²)
    HY_CONV3D_SWEEP_SIZES=512,1024                  # comma-separated squares (overrides H/W)
    HY_CONV3D_SWEEP_PRESETS=priority                # priority|all|ratio_idx|HxW list
    HY_CONV3D_SWEEP_CHUNKS=1                        # 0 = full shapes only
    HY_CONV3D_SWEEP_CHUNK_MODE=representative       # representative | all
    CONV3D_SWEEP_HW_PRODUCT=none                    # widen spatial block search
    CONV3D_SWEEP_MAX_T_BLOCK=4
    CONV3D_SWEEP_MAX_COMBOS=800
    CONV3D_SWEEP_TRACE_ITERS=10

Install winners into ``tt/vae/conv3d_blockings.py`` ``HUNYUAN_EXACT_BLOCKINGS`` as::

    (1, 1, Cin, Cout, (3, 3, 3), T, H, W): (Cin_blk, Cout_blk, T_blk, H_blk, W_blk)

Use the exact ``(T, H, W)`` from the sweep row (input spatial dims passed to
``get_conv3d_config``).
"""

from __future__ import annotations

import os
from typing import NamedTuple

import pytest
import ttnn

from models.experimental.hunyuan_image_3_0.ref.tokenizer.resolution import Resolution, ResolutionGroup
from models.experimental.hunyuan_image_3_0.ref.vae.encoder import (
    BLOCK_OUT_CHANNELS,
    IN_CHANNELS,
    PIXEL_T,
    encoder_down_level_specs,
    encoder_head_shape,
)
from models.tt_dit.tests.models.wan2_2.bruteforce_conv3d_sweep import (
    TRACE_REGION_SIZE,
    run_sweep,
)
from models.tt_dit.utils.conv3d import aligned_channels
from models.tt_dit.utils.test import line_params

_KERNEL = (3, 3, 3)
_STRIDE = (1, 1, 1)
_PADDING = (1, 1, 1)
_CHUNK_ELEMS = int(os.environ.get("HY_CONV_CHUNK_ELEMS", str(1024**3)))
_KVOL = _KERNEL[0] * _KERNEL[1] * _KERNEL[2]


class SweepLayer(NamedTuple):
    name: str
    c_in: int
    c_out: int
    t: int
    h: int
    w: int


def _representative_chunk_heights(heights: list[int]) -> list[int]:
    """Pick min / mid / max strip heights — enough to tune without sweeping every strip."""
    if len(heights) <= 3:
        return heights
    mid = heights[len(heights) // 2]
    return [heights[0], mid, heights[-1]]


def _chunk_input_heights(c_in: int, t: int, h: int, w: int, *, mode: str) -> list[int]:
    """Mirror HunyuanSymmetricConv3d.forward H-chunk slice heights (padded strips)."""
    ac = aligned_channels(c_in)
    im2col = ac * t * h * w * _KVOL
    if im2col <= _CHUNK_ELEMS or h <= 1:
        return []
    n_chunks = (im2col + _CHUNK_ELEMS - 1) // _CHUNK_ELEMS
    hc = (h + n_chunks - 1) // n_chunks
    pH = _PADDING[1]
    h_pad = h + 2 * pH
    heights: list[int] = []
    for o in range(0, h, hc):
        oe = min(h, o + hc)
        in_h = min(oe + 2 * pH, h_pad)
        if in_h not in heights:
            heights.append(in_h)
    if mode == "all":
        return heights
    return _representative_chunk_heights(heights)


def _add_conv(
    layers: list[SweepLayer],
    *,
    name: str,
    c_in: int,
    c_out: int,
    t: int,
    h: int,
    w: int,
    include_chunks: bool,
    chunk_mode: str,
) -> None:
    layers.append(SweepLayer(name, aligned_channels(c_in), c_out, t, h, w))
    if not include_chunks:
        return
    for in_h in _chunk_input_heights(c_in, t, h, w, mode=chunk_mode):
        if in_h >= h + 2 * _PADDING[1]:
            continue
        layers.append(SweepLayer(f"{name}_H{in_h}", aligned_channels(c_in), c_out, t, in_h, w))


def encoder_sweep_layers(
    pixel_t: int = PIXEL_T,
    pixel_h: int = 1024,
    pixel_w: int = 1024,
    *,
    include_chunks: bool = True,
    include_head: bool = True,
    chunk_mode: str = "representative",
) -> list[SweepLayer]:
    """Build sweep rows for conv_in, down blocks, and encoder head at ``pixel_h×pixel_w``."""
    layers: list[SweepLayer] = []

    _add_conv(
        layers,
        name="enc_conv_in",
        c_in=IN_CHANNELS,
        c_out=BLOCK_OUT_CHANNELS[0],
        t=pixel_t,
        h=pixel_h,
        w=pixel_w,
        include_chunks=include_chunks,
        chunk_mode=chunk_mode,
    )

    for spec in encoder_down_level_specs(pixel_t, pixel_h, pixel_w):
        lvl = spec.level
        in_ch = spec.in_channels
        for bi in range(2):
            _add_conv(
                layers,
                name=f"enc_d{lvl}_res{bi}",
                c_in=in_ch,
                c_out=spec.block_channels,
                t=spec.t,
                h=spec.h,
                w=spec.w,
                include_chunks=include_chunks,
                chunk_mode=chunk_mode,
            )
            in_ch = spec.block_channels

        if spec.has_downsample and spec.downsample_out_channels is not None:
            factor = 8 if spec.add_temporal_downsample else 4
            out_conv = spec.downsample_out_channels // factor
            _add_conv(
                layers,
                name=f"enc_d{lvl}_down",
                c_in=in_ch,
                c_out=out_conv,
                t=spec.t,
                h=spec.h,
                w=spec.w,
                include_chunks=include_chunks,
                chunk_mode=chunk_mode,
            )

    if include_head:
        t, h, w, c = encoder_head_shape(pixel_t, pixel_h, pixel_w)
        for name in ("enc_mid_res0", "enc_mid_res1"):
            _add_conv(
                layers,
                name=name,
                c_in=c,
                c_out=c,
                t=t,
                h=h,
                w=w,
                include_chunks=include_chunks,
                chunk_mode=chunk_mode,
            )
        _add_conv(
            layers,
            name="enc_head_out",
            c_in=c,
            c_out=64,
            t=t,
            h=h,
            w=w,
            include_chunks=include_chunks,
            chunk_mode=chunk_mode,
        )

    return layers


def _sweep_env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None else default


def _sweep_env_opt(name: str, default):
    v = os.environ.get(name)
    if v is None:
        return default
    v = v.strip().lower()
    if v in ("none", "off", ""):
        return None
    return int(v)


def _layer_parametrize_id(layer: SweepLayer, pixel_h: int, pixel_w: int) -> str:
    return f"enc_{pixel_h}x{pixel_w}_{layer.name}_{layer.c_in}x{layer.c_out}_T{layer.t}_H{layer.h}_W{layer.w}"


def _vae_resolution_group() -> ResolutionGroup:
    """Match ``HunyuanImage3ImageProcessor`` / ``build_gen_image_info`` presets."""
    return ResolutionGroup(
        base_size=_sweep_env_int("HY_CONV3D_SWEEP_BASE_SIZE", 1024),
        extra_resolutions=[
            Resolution("1024x768"),
            Resolution("1280x720"),
            Resolution("768x1024"),
            Resolution("720x1280"),
        ],
    )


# Common non-square presets (ratio indices) for a faster sweep tier.
_PRIORITY_RATIO_IDX = (13, 19, 20, 33, 34, 35, 36)  # 832x1216 … 720x1280


def sweep_result_dir(pixel_h: int, pixel_w: int) -> str:
    """Sweep JSON directory for a pixel resolution (``enc_{H}x{W}``)."""
    return f"models/experimental/hunyuan_image_3_0/sweep_results/enc_{pixel_h}x{pixel_w}"


def _parse_hw_token(token: str) -> tuple[int, int]:
    token = token.strip().lower()
    if "x" in token:
        a, b = token.split("x", 1)
        return int(a), int(b)
    s = int(token)
    return s, s


def _sweep_resolutions() -> list[tuple[int, int]]:
    """Return ``(pixel_h, pixel_w)`` list to sweep.

    Env (first match wins):
      HY_CONV3D_SWEEP_PRESETS=all|priority|<idx,...>|<HxW,...>
      HY_CONV3D_SWEEP_SIZES=512,1024          # squares only
      HY_CONV3D_SWEEP_H / HY_CONV3D_SWEEP_W   # single resolution
    """
    presets = os.environ.get("HY_CONV3D_SWEEP_PRESETS")
    if presets:
        mode = presets.strip().lower()
        rg = _vae_resolution_group()
        if mode == "all":
            return [(r.height, r.width) for r in rg.data]
        if mode == "priority":
            return [(rg.data[i].height, rg.data[i].width) for i in _PRIORITY_RATIO_IDX]
        out: list[tuple[int, int]] = []
        for part in presets.split(","):
            part = part.strip()
            if not part:
                continue
            if part.isdigit():
                idx = int(part)
                out.append((rg.data[idx].height, rg.data[idx].width))
            else:
                out.append(_parse_hw_token(part))
        return out

    sizes = os.environ.get("HY_CONV3D_SWEEP_SIZES")
    if sizes:
        squares = [int(s.strip()) for s in sizes.split(",") if s.strip()]
        return [(s, s) for s in squares]
    h = _sweep_env_int("HY_CONV3D_SWEEP_H", 1024)
    w = _sweep_env_int("HY_CONV3D_SWEEP_W", h)
    return [(h, w)]


def _build_sweep_cases() -> tuple[list, list[str]]:
    """(pytest param tuples, ids) for configured encoder resolutions."""
    include_chunks = os.environ.get("HY_CONV3D_SWEEP_CHUNKS", "1").strip().lower() not in ("0", "false", "no")
    chunk_mode = os.environ.get("HY_CONV3D_SWEEP_CHUNK_MODE", "representative").strip().lower()
    cases: list[tuple] = []
    ids: list[str] = []

    for pixel_h, pixel_w in _sweep_resolutions():
        for layer in encoder_sweep_layers(
            pixel_h=pixel_h,
            pixel_w=pixel_w,
            include_chunks=include_chunks,
            chunk_mode=chunk_mode,
        ):
            cases.append(
                (
                    pixel_h,
                    pixel_w,
                    layer.name,
                    layer.c_in,
                    layer.c_out,
                    _KERNEL,
                    _STRIDE,
                    _PADDING,
                    layer.t,
                    layer.h,
                    layer.w,
                    1,
                    1,
                )
            )
            ids.append(_layer_parametrize_id(layer, pixel_h, pixel_w))
    return cases, ids


_SWEEP_CASES, _SWEEP_IDS = _build_sweep_cases()


@pytest.fixture(scope="function")
def device_params(request):
    return {**line_params, "trace_region_size": TRACE_REGION_SIZE}


@pytest.mark.parametrize(
    "mesh_device, mesh_shape",
    [[(2, 2), (1, 1)]],
    ids=["bh_2x2"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "pixel_h, pixel_w, layer_name, C_in, C_out, kernel, stride, padding, T, H, W, h_factor, w_factor",
    _SWEEP_CASES,
    ids=_SWEEP_IDS,
)
def test_encoder_conv3d_blocking_sweep(
    mesh_device,
    mesh_shape,
    pixel_h,
    pixel_w,
    layer_name,
    C_in,
    C_out,
    kernel,
    stride,
    padding,
    T,
    H,
    W,
    h_factor,
    w_factor,
):
    device = mesh_device.create_submesh(ttnn.MeshShape(*mesh_shape))
    out_dir = sweep_result_dir(pixel_h, pixel_w)
    output = f"{out_dir}/{layer_name}_{C_in}x{C_out}_T{T}_H{H}_W{W}.json"
    if os.environ.get("HY_CONV3D_SWEEP_SKIP_EXISTING", "1").strip().lower() not in ("0", "false", "no"):
        import json
        import pathlib

        p = pathlib.Path(output)
        if p.is_file():
            data = json.loads(p.read_text())
            if data.get("best_blocking"):
                pytest.skip(f"existing result {output}")
    run_sweep(
        device,
        C_in,
        C_out,
        kernel,
        T,
        H,
        W,
        output,
        stride=stride,
        padding=padding,
        h_factor=h_factor,
        w_factor=w_factor,
        max_combos=_sweep_env_opt("CONV3D_SWEEP_MAX_COMBOS", 800),
        max_t_block=_sweep_env_opt("CONV3D_SWEEP_MAX_T_BLOCK", 4),
        hw_product=_sweep_env_opt("CONV3D_SWEEP_HW_PRODUCT", None),
        trace_iters=_sweep_env_opt("CONV3D_SWEEP_TRACE_ITERS", 10),
    )
