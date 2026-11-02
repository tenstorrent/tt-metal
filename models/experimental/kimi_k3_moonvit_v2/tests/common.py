# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Test constants and helpers shared across this folder."""

from __future__ import annotations

import dataclasses
import os

import pytest
import torch

import ttnn
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import MoonViTConfig
from models.common.utility_functions import comp_pcc
from models.experimental.kimi_k3_moonvit_v2.tt.ccl import ParallelFactor, TT_CCL

# Depth of the shallow model that the full-model tests run (`shallow_reference`, `shallow_config`).
TEST_LAYERS = 2

# Fixtures (conftest.py), each with random weights or the checkpoint at $KIMI_K3_HF_MODEL:
# `reference` has every layer, `shallow_reference` the first TEST_LAYERS.
WEIGHTS = ["random", "checkpoint"]
REFERENCE = pytest.mark.parametrize("reference", WEIGHTS, indirect=True)
SHALLOW_REFERENCE = pytest.mark.parametrize("shallow_reference", WEIGHTS, indirect=True)

# Layers the per-layer tests cover. With random weights every layer is alike, so one is enough;
# with the checkpoint, the first, middle and last. MOONVIT_TEST_LAYERS (e.g. "5" or "0,20") sets
# the layers for both.
LAYERS_ENV = "MOONVIT_TEST_LAYERS"


def _test_layers() -> dict[str, list[int]]:
    value = os.environ.get(LAYERS_ENV)
    if value:
        layers = [int(layer) for layer in value.split(",")]
        return {"random": layers, "checkpoint": layers}
    n = MoonViTConfig().num_hidden_layers
    return {"random": [0], "checkpoint": [0, n // 2, n - 1]}


REFERENCE_LAYERS = pytest.mark.parametrize(
    ("reference", "layer"),
    [
        pytest.param(weights, layer, id=f"{weights}-layer{layer}")
        for weights, layers in _test_layers().items()
        for layer in layers
    ],
    indirect=["reference"],
)

# (t, h, w) in patches. h and w are even, as the processor's 28 px padding guarantees.
GRIDS = {
    "square": [(1, 8, 8)],
    "wide": [(1, 6, 20)],
    "tall": [(1, 20, 6)],
    # Matches the 64x64 position table exactly, so it skips interpolation.
    "native_table": [(1, 64, 64)],
}

# Device grids for attention, the block and the full model. Their CPU golden builds L x L scores per
# head, which rules out 65,536 patches here.
ATTENTION_GRIDS = pytest.mark.parametrize("grid", [[(1, 64, 64)], [(1, 40, 100)]], ids=["64x64", "40x100"])

# Device grids for the parts that work on each row alone (everything but attention). Their CPU golden
# grows only with the row count, so it runs at the full padded length in seconds. Each size names one
# (h, w) per mesh, since one P150 runs a quarter of the ring's rows (`device_config`):
#   full:     fills the padded length exactly; real rows on every ring chip, no padding
#   max_side: also fills it, with the longest side the processor allows (512)
#   partial:  nearly full; the real rows end inside a tile (not a multiple of 32)
#   small:    a small image, almost all padding
ROW_GRID_SIZES = {
    "full": {"ring1x4": (256, 256), "p150": (128, 128)},
    "max_side": {"ring1x4": (512, 128), "p150": (512, 32)},
    "partial": {"ring1x4": (250, 250), "p150": (122, 130)},
    "small": {"ring1x4": (40, 100), "p150": (40, 100)},
}
ROW_GRIDS = pytest.mark.parametrize("size", list(ROW_GRID_SIZES))


def row_grid(size: str, sp_axis: int | None) -> list[tuple[int, int, int]]:
    """The `size` grid for the mesh under test: `ring1x4` when the sequence is split, else `p150`."""
    h, w = ROW_GRID_SIZES[size]["p150" if sp_axis is None else "ring1x4"]
    return [(1, h, w)]


# The 4 Blackhole chips of a QuietBox 2 in a ring along mesh axis 1 (the (1, 4) mesh other QB2 models
# open), 2 links between neighbours.
RING_SIZE = 4
RING4 = pytest.param((1, RING_SIZE), 1, 2, {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="ring1x4")
# One P150 standing in for one chip of that ring: it gets the first quarter of the padded sequence
# (`device_config`).
P150 = pytest.param((1, 1), None, 1, {}, id="p150")


def mesh_params(*params):
    return pytest.mark.parametrize(
        ("mesh_device", "sp_axis", "num_links", "device_params"),
        list(params),
        indirect=["mesh_device", "device_params"],
    )


# Attention needs the ring. Parts that work on each row alone also run on one P150.
RING = mesh_params(RING4)
P150_AND_RING = mesh_params(P150, RING4)

NOT_WRITTEN = pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="ttnn forward not written yet")

# ttsim (TT_METAL_SIMULATOR) simulates one chip at about 1/100 of silicon speed. The ROW_GRIDS cases pad
# to 16,384 rows per chip, so they skip there; each row-wise part also has a SIMULATOR_GRID case on one
# chip, padded to SIMULATOR_ROWS, with random weights.
ON_SIMULATOR = bool(os.environ.get("TT_METAL_SIMULATOR"))
NOT_ON_SIMULATOR = pytest.mark.skipif(ON_SIMULATOR, reason="full padded length is too slow on ttsim")
SINGLE_CHIP = mesh_params(P150)
SIMULATOR_ROWS = 128
# 120 rows: the real rows end inside the last tile, and the padding fills one more.
SIMULATOR_GRID = [(1, 6, 20)]


def device_config(config: MoonViTConfig, sp_axis: int | None) -> MoonViTConfig:
    """The config a test builds with. On one P150 (`sp_axis` None) the padded length is one ring
    chip's share, `max_patches / RING_SIZE` rows, so the chip sees the shapes a ring chip sees."""
    if sp_axis is not None:
        return config
    return dataclasses.replace(config, max_patches=config.max_patches // RING_SIZE)


def ring_parallel(mesh_device, sp_axis: int | None, num_links: int) -> tuple[ParallelFactor | None, TT_CCL | None]:
    """`(sequence_parallel, ccl)` for `resolve_parallel`; `(None, None)` on a single device."""
    if sp_axis is None:
        return None, None
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    return ParallelFactor(sp_factor, sp_axis), TT_CCL(mesh_device, num_links=num_links, topology=ttnn.Topology.Ring)


def assert_pcc(golden: torch.Tensor, actual: torch.Tensor, pcc: float = 0.99) -> None:
    passing, found = comp_pcc(golden.float(), actual.float(), pcc)
    assert passing, found


def random_patches(config: MoonViTConfig, grid_thws: list[tuple[int, int, int]], seed: int = 0):
    """Patches in the processor's output range ((x/255 - 0.5)/0.5 lies in [-1, 1])."""
    n = sum(t * h * w for t, h, w in grid_thws)
    g = torch.Generator().manual_seed(seed)
    pixels = torch.rand((n, config.in_channels, config.patch_size, config.patch_size), generator=g) * 2 - 1
    return pixels, torch.tensor(grid_thws, dtype=torch.int64)


def reference_rope(reference, grid_thws: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(cu_seqlens, max_seqlen, rope_freqs_cis)` as `MoonViT3dEncoder.forward` builds them."""
    rope_freqs_cis = reference.vision_tower.encoder.rope_2d.get_freqs_cis(
        grid_thws=grid_thws, device=torch.device("cpu")
    )
    lengths = torch.cat(
        (
            torch.zeros(1, dtype=grid_thws.dtype, device=grid_thws.device),
            grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2],
        )
    )
    max_seqlen = lengths.max()
    cu_seqlens = lengths.cumsum(dim=0, dtype=torch.int32)
    return cu_seqlens, max_seqlen, rope_freqs_cis


def sp_shard(x: torch.Tensor, mesh_device, sp_axis: int | None, rows: int) -> ttnn.Tensor:
    """(L, C) or (H, L, C) on the host -> [1, 1 or H, rows, C] bf16 on device, zero-padded to `rows`,
    split along `sp_axis`, or replicated when `sp_axis` is None."""
    x = x if x.ndim == 3 else x[None]
    if x.shape[-2] > rows:
        raise ValueError(f"{x.shape[-2]} rows do not fit the padded length {rows}")
    padded = torch.nn.functional.pad(x, (0, 0, 0, rows - x.shape[-2]))[None]
    if sp_axis is None:
        mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    else:
        dims = [None, None]
        dims[sp_axis] = 2
        mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims)
    return ttnn.from_torch(padded, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)


def sp_gather(x: ttnn.Tensor, mesh_device, sp_axis: int | None, logical_n: int) -> torch.Tensor:
    """The inverse of `sp_shard`: -> (logical_n, C), or (H, logical_n, C) when H > 1.

    Replicas are stacked on dim 0 and the first is kept.
    """
    if sp_axis is None:
        composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    else:
        dims = [0, 0]
        dims[sp_axis] = 2
        composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims)
    out = ttnn.to_torch(x, mesh_composer=composer)[0, :, :logical_n]
    return out[0] if out.shape[0] == 1 else out
