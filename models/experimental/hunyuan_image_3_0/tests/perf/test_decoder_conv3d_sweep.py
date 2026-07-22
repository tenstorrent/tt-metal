# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Brute-force conv3d blocking sweep for Hunyuan VAE decoder hot paths.

The decode perf path runs H/W-spatial on a 2×2 mesh (``h_mesh_axis=0``,
``w_mesh_axis=1``). Convs see **local** spatial tiles (global H/W ÷ 2) plus a
1-voxel neighbor-pad halo on sharded axes. This sweep uses those post-pad
dimensions and representative H-chunk strip heights (same schedule as
``HunyuanSymmetricConv3d._conv_valid_h``).

Install winners::

    python models/experimental/hunyuan_image_3_0/scripts/apply_decoder_sweep_blockings.py

Run on BH 2×2 (priority — upsample + tail at 64×64 latent → 1024² output):

    python_env/bin/python -m pytest \\
      models/experimental/hunyuan_image_3_0/tests/perf/test_decoder_conv3d_sweep.py \\
      -k "bh_2x2 and dec_64x64 and upsample" -s --timeout=0

Full 64×64-latent decoder sweep:

    python_env/bin/python -m pytest \\
      models/experimental/hunyuan_image_3_0/tests/perf/test_decoder_conv3d_sweep.py \\
      -k "bh_2x2 and dec_64x64" -s --timeout=0

Environment overrides (same names as ``test_encoder_conv3d_sweep.py``):

    HY_CONV3D_SWEEP_SPATIAL=2          # mesh H/W sharding factor (default 2)
    HY_CONV3D_SWEEP_CHUNKS=1
    HY_CONV3D_SWEEP_CHUNK_MODE=representative
    HY_CONV3D_SWEEP_SKIP_EXISTING=1
    CONV3D_SWEEP_MAX_COMBOS=800
"""

from __future__ import annotations

import os
from typing import NamedTuple

import pytest
import ttnn

from models.experimental.hunyuan_image_3_0.ref.vae.decoder import (
    BLOCK_IN_CHANNELS,
    LATENT_H,
    LATENT_T,
    LATENT_W,
    MID_CHANNELS,
    NUM_RES_BLOCKS,
    OUT_CHANNELS,
    Z_CHANNELS,
    decoder_tail_shape,
    decoder_up_level_specs,
)
from models.tt_dit.tests.models.wan2_2.bruteforce_conv3d_sweep import (
    TRACE_REGION_SIZE,
    run_sweep,
)
from models.tt_dit.utils.conv3d import aligned_channels
from models.tt_dit.utils.test import line_params

_KERNEL3 = (3, 3, 3)
_KERNEL1 = (1, 1, 1)
_STRIDE = (1, 1, 1)
_PADDING3 = (1, 1, 1)
_PADDING0 = (0, 0, 0)
from models.experimental.hunyuan_image_3_0.tt.vae.conv3d import _CONV3D_CHUNK_ELEMS

_KVOL = _KERNEL3[0] * _KERNEL3[1] * _KERNEL3[2]
_KH = _KERNEL3[1]


class SweepLayer(NamedTuple):
    name: str
    c_in: int
    c_out: int
    kernel: tuple[int, int, int]
    padding: tuple[int, int, int]
    t: int
    h: int
    w: int


def _representative_chunk_heights(heights: list[int]) -> list[int]:
    if len(heights) <= 3:
        return heights
    mid = heights[len(heights) // 2]
    return [heights[0], mid, heights[-1]]


def _valid_conv_chunk_heights(
    c_in: int,
    t: int,
    h_padded: int,
    w: int,
    *,
    mode: str,
) -> list[int]:
    """Strip input heights for sharded valid-conv H chunking (post neighbor-pad)."""
    ac = aligned_channels(c_in)
    im2col = ac * t * h_padded * w * _KVOL
    if im2col <= _CONV3D_CHUNK_ELEMS or h_padded <= _KH:
        return []
    h_out = h_padded - (_KH - 1)
    n_chunks = (im2col + _CONV3D_CHUNK_ELEMS - 1) // _CONV3D_CHUNK_ELEMS
    hc = (h_out + n_chunks - 1) // n_chunks
    heights: list[int] = []
    for o in range(0, h_out, hc):
        oe = min(h_out, o + hc)
        in_h = oe + (_KH - 1)
        if in_h not in heights:
            heights.append(in_h)
    if mode == "all":
        return heights
    return _representative_chunk_heights(heights)


def _replicated_chunk_heights(c_in: int, t: int, h: int, w: int, *, mode: str) -> list[int]:
    """Strip heights for replicated conv H-chunking (symmetric pad, padding_h=0 strips)."""
    ac = aligned_channels(c_in)
    im2col = ac * t * h * w * _KVOL
    if im2col <= _CONV3D_CHUNK_ELEMS or h <= 1:
        return []
    n_chunks = (im2col + _CONV3D_CHUNK_ELEMS - 1) // _CONV3D_CHUNK_ELEMS
    hc = (h + n_chunks - 1) // n_chunks
    pH = _PADDING3[1]
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


def _sharded_padded_hw(global_h: int, global_w: int, *, spatial: int, pad: int = 1) -> tuple[int, int]:
    local_h = global_h // spatial
    local_w = global_w // spatial
    return local_h + 2 * pad, local_w + 2 * pad


def _add_conv(
    layers: list[SweepLayer],
    *,
    name: str,
    c_in: int,
    c_out: int,
    kernel: tuple[int, int, int],
    padding: tuple[int, int, int],
    t: int,
    h: int,
    w: int,
    include_chunks: bool,
    chunk_mode: str,
    chunk_fn,
) -> None:
    ac_in = aligned_channels(c_in)
    layers.append(SweepLayer(name, ac_in, c_out, kernel, padding, t, h, w))
    if not include_chunks:
        return
    for in_h in chunk_fn(ac_in, t, h, w, mode=chunk_mode):
        if in_h >= h:
            continue
        layers.append(SweepLayer(f"{name}_H{in_h}", ac_in, c_out, kernel, padding, t, in_h, w))


def decoder_sweep_layers(
    latent_t: int = LATENT_T,
    latent_h: int = LATENT_H,
    latent_w: int = LATENT_W,
    *,
    spatial_factor: int = 2,
    include_chunks: bool = True,
    chunk_mode: str = "representative",
) -> list[SweepLayer]:
    """Build sweep rows for spatial-sharded VAE decode (2×2 mesh, both H/W sharded)."""
    layers: list[SweepLayer] = []
    pad = _PADDING3[1]

    def sharded_add(**kwargs) -> None:
        gh, gw = kwargs.pop("global_h"), kwargs.pop("global_w")
        h, w = _sharded_padded_hw(gh, gw, spatial=spatial_factor, pad=pad)
        _add_conv(
            layers,
            **kwargs,
            h=h,
            w=w,
            include_chunks=include_chunks,
            chunk_mode=chunk_mode,
            chunk_fn=_valid_conv_chunk_heights,
        )

    def full_add(**kwargs) -> None:
        _add_conv(
            layers,
            **kwargs,
            include_chunks=include_chunks,
            chunk_mode=chunk_mode,
            chunk_fn=_replicated_chunk_heights,
        )

    sharded_add(
        name="dec_conv_in",
        c_in=Z_CHANNELS,
        c_out=BLOCK_IN_CHANNELS,
        kernel=_KERNEL3,
        padding=_PADDING3,
        t=latent_t,
        global_h=latent_h,
        global_w=latent_w,
    )

    for i in range(2):
        sharded_add(
            name=f"dec_mid_res{i}",
            c_in=MID_CHANNELS,
            c_out=MID_CHANNELS,
            kernel=_KERNEL3,
            padding=_PADDING3,
            t=latent_t,
            global_h=latent_h,
            global_w=latent_w,
        )

    full_add(
        name="dec_mid_attn",
        c_in=MID_CHANNELS,
        c_out=MID_CHANNELS,
        kernel=_KERNEL1,
        padding=_PADDING0,
        t=latent_t,
        h=latent_h,
        w=latent_w,
    )

    for spec in decoder_up_level_specs(latent_t, latent_h, latent_w):
        lvl = spec.level
        in_ch = spec.in_channels
        for bi in range(NUM_RES_BLOCKS + 1):
            sharded_add(
                name=f"dec_u{lvl}_res{bi}",
                c_in=in_ch,
                c_out=spec.block_channels,
                kernel=_KERNEL3,
                padding=_PADDING3,
                t=spec.t,
                global_h=spec.h,
                global_w=spec.w,
            )
            in_ch = spec.block_channels

        if spec.has_upsample and spec.upsample_out_channels is not None:
            factor = 8 if spec.add_temporal_upsample else 4
            out_conv = spec.upsample_out_channels * factor
            sharded_add(
                name=f"dec_u{lvl}_upsample",
                c_in=in_ch,
                c_out=out_conv,
                kernel=_KERNEL3,
                padding=_PADDING3,
                t=spec.t,
                global_h=spec.h,
                global_w=spec.w,
            )

    tail_t, tail_h, tail_w, tail_c = decoder_tail_shape(latent_t, latent_h, latent_w)
    sharded_add(
        name="dec_conv_out",
        c_in=tail_c,
        c_out=OUT_CHANNELS,
        kernel=_KERNEL3,
        padding=_PADDING3,
        t=tail_t,
        global_h=tail_h,
        global_w=tail_w,
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


def _layer_parametrize_id(layer: SweepLayer, latent_h: int, latent_w: int) -> str:
    return f"dec_{latent_h}x{latent_w}_{layer.name}_" f"{layer.c_in}x{layer.c_out}_T{layer.t}_H{layer.h}_W{layer.w}"


def sweep_result_dir(latent_h: int, latent_w: int) -> str:
    return f"models/experimental/hunyuan_image_3_0/sweep_results/dec_{latent_h}x{latent_w}"


def _parse_latent_token(token: str) -> tuple[int, int]:
    token = token.strip().lower()
    if "x" in token:
        a, b = token.split("x", 1)
        return int(a), int(b)
    s = int(token)
    return s, s


def _sweep_latent_shapes() -> list[tuple[int, int]]:
    presets = os.environ.get("HY_CONV3D_SWEEP_PRESETS")
    if presets:
        mode = presets.strip().lower()
        if mode in ("64", "64x64", "dec_64x64", "default"):
            return [(64, 64)]
        out: list[tuple[int, int]] = []
        for part in presets.split(","):
            part = part.strip()
            if part:
                out.append(_parse_latent_token(part))
        return out or [(64, 64)]

    h = _sweep_env_int("HY_CONV3D_SWEEP_H", 64)
    w = _sweep_env_int("HY_CONV3D_SWEEP_W", h)
    return [(h, w)]


def _build_sweep_cases() -> tuple[list, list[str]]:
    include_chunks = os.environ.get("HY_CONV3D_SWEEP_CHUNKS", "1").strip().lower() not in ("0", "false", "no")
    chunk_mode = os.environ.get("HY_CONV3D_SWEEP_CHUNK_MODE", "all").strip().lower()
    spatial = _sweep_env_int("HY_CONV3D_SWEEP_SPATIAL", 2)
    cases: list[tuple] = []
    ids: list[str] = []

    for latent_h, latent_w in _sweep_latent_shapes():
        for layer in decoder_sweep_layers(
            latent_h=latent_h,
            latent_w=latent_w,
            spatial_factor=spatial,
            include_chunks=include_chunks,
            chunk_mode=chunk_mode,
        ):
            cases.append(
                (
                    latent_h,
                    latent_w,
                    layer.name,
                    layer.c_in,
                    layer.c_out,
                    layer.kernel,
                    _STRIDE,
                    layer.padding,
                    layer.t,
                    layer.h,
                    layer.w,
                    1,
                    1,
                )
            )
            ids.append(_layer_parametrize_id(layer, latent_h, latent_w))
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
    "latent_h, latent_w, layer_name, C_in, C_out, kernel, stride, padding, T, H, W, h_factor, w_factor",
    _SWEEP_CASES,
    ids=_SWEEP_IDS,
)
def test_decoder_conv3d_blocking_sweep(
    mesh_device,
    mesh_shape,
    latent_h,
    latent_w,
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
    out_dir = sweep_result_dir(latent_h, latent_w)
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
