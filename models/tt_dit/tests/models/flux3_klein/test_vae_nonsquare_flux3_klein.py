# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of the FLUX 3 Klein VAE decoder at the non-square outputs (4096x2304 and 2304x4096) next to the
squares, on one Galaxy (8 H shards) and on a quad (32): each side divides into its mesh shards, the conv slice table
carries the partitioned keys of every level, decode()'s unpatchify inverts encode()'s packing on a grid whose height
and width differ, in whole token rows per device or (144 token rows over 32 devices) on the gathered tokens before the
split, and the conv fallback ladder keeps the shard layout the fused decode needs. No device."""

from __future__ import annotations

import pytest
import torch

from ....layers.conv2d import Conv2d, _level_slices, _slice_config, _stats_slices
from ....models.vae.vae import VAE_A_MAX_SLICES
from ....pipelines.flux3_klein.packing import latent_shape
from ....pipelines.flux3_klein.pipeline_flux3_klein import SHAPES
from ....pipelines.flux3_klein.vae_flux3_klein import (
    LATENT_CHANNELS,
    PARTITIONED_DECODER_CONVS_PER_LEVEL,
    SCALE,
    register_conv_slices,
    token_rows_split,
)

PATCH = 2
Z_CHANNELS = LATENT_CHANNELS // PATCH**2
W_SHARDS = 4
MESHES = pytest.mark.parametrize("h_shards", (8, 32))


def _unpatchify(tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """preprocess_and_unpatchify's reshapes: [n, h * w, C * P * P] -> [n, h * P, w * P, C]."""
    channels = tokens.shape[-1] // PATCH**2
    z = tokens.reshape(tokens.shape[0], h, w, channels, PATCH, PATCH).permute(0, 1, 4, 2, 5, 3)
    return z.reshape(tokens.shape[0], h * PATCH, w * PATCH, channels)


@MESHES
@pytest.mark.parametrize("width, height", SHAPES)
def test_shape_divides_the_mesh(width, height, h_shards):
    h, w = latent_shape(height, width)
    assert (h * w) % h_shards == 0  # the SP-sharded token rows
    lh, lw = height // SCALE, width // SCALE
    assert lh % h_shards == 0 and lw % (PATCH * W_SHARDS) == 0
    assert token_rows_split(h, h_shards) == (lh % (PATCH * h_shards) == 0)  # else decode() splits the latent rows
    assert (lh // h_shards) * (lw // W_SHARDS) % 32 == 0  # the local attention's query rows


def test_only_the_landscape_shape_on_a_quad_splits_latent_rows():
    split = {
        (width, height, n): token_rows_split(latent_shape(height, width)[0], n)
        for width, height in SHAPES
        for n in (8, 32)
    }
    assert [key for key, whole in split.items() if not whole] == [(4096, 2304, 32)]
    with pytest.raises(ValueError):
        token_rows_split(144, 64)


@MESHES
def test_slice_table_keys_every_level(h_shards):
    mesh = (W_SHARDS, h_shards)
    saved = Conv2d.slice_params.get(mesh)
    try:
        Conv2d.slice_params.pop(mesh, None)
        for width, height in SHAPES:
            register_conv_slices(mesh, height, width, h_factor=h_shards, w_factor=W_SHARDS)
        table = Conv2d.slice_params[mesh]
        for width, height in SHAPES:
            for level, partitioned in enumerate(PARTITIONED_DECODER_CONVS_PER_LEVEL):
                assert (height >> level) % h_shards == 0
                local = ((height >> level) // h_shards + 2, (width >> level) // W_SHARDS + 2)  # 1-px halo on each side
                for cin, cout in partitioned:
                    assert table[(*local, cin, cout)] == 0
    finally:
        Conv2d.slice_params.pop(mesh, None)
        if saved is not None:
            Conv2d.slice_params[mesh] = saved


def test_unpatchify_inverts_the_encoder_packing():
    h, w = latent_shape(2304, 4096)
    tokens = torch.randn(h * w, LATENT_CHANNELS)
    z = _unpatchify(tokens[None], h, w)
    # encode(): [1, H, W, C] -> [1, h, P, w, P, C] -> [1, h, w, C, P, P] -> [h * w, C * P * P]
    packed = z.reshape(1, h, PATCH, w, PATCH, Z_CHANNELS).permute(0, 1, 3, 5, 2, 4).flatten(3).flatten(1, 2)[0]
    assert torch.equal(packed, tokens)
    patch = tokens[1 * w + 2].reshape(Z_CHANNELS, PATCH, PATCH)  # token row 1, column 2
    assert torch.equal(z[0, 3, 5], patch[:, 1, 1])  # latent row 3 = token row 1, patch row 1


@MESHES
@pytest.mark.parametrize("width, height", SHAPES)
def test_both_decode_splits_give_the_same_latent_rows(width, height, h_shards):
    h, w = latent_shape(height, width)
    tokens = torch.randn(h * w, LATENT_CHANNELS)
    full = _unpatchify(tokens[None], h, w)
    # the gathered tokens unpatchified, then the latent rows split over the H axis (decode()'s second path)
    per_device = full.reshape(h_shards, h * PATCH // h_shards, w * PATCH, Z_CHANNELS)
    assert torch.equal(per_device.reshape(full.shape), full)
    if token_rows_split(h, h_shards):
        # whole token rows per device, unpatchified per device (decode()'s first path)
        local = _unpatchify(tokens.reshape(h_shards, h * w // h_shards, LATENT_CHANNELS), h // h_shards, w)
        assert torch.equal(local, per_device)


def test_fallback_levels_double_the_slices():
    # levels 0 and 1 keep the table's slicing (0 = ttnn picks, None = no slice config); from level 2 the DRAM width
    # slices double per level, from one slice where ttnn picked, so a conv whose program does not fit next to its
    # activations stays height-sharded (the external halo, prologue and epilogue statistics need that) and shrinks
    assert [_level_slices(0, level) for level in range(5)] == [0, 0, 2, 4, 8]
    assert [_level_slices(None, level) for level in range(4)] == [None, None, 2, 4]
    assert [_level_slices(4, level) for level in range(4)] == [4, 4, 8, 16]


def test_slice_config_per_level():
    import ttnn

    halo = object()
    assert _slice_config(None, None) is None
    assert _slice_config(None, halo) is ttnn.Conv2dL1FullSliceConfig
    assert _slice_config(1, halo) is ttnn.Conv2dL1FullSliceConfig
    for slices in (0, 2):
        config = _slice_config(slices, halo)
        assert config.num_slices == slices and config.slice_type == ttnn.Conv2dDRAMSliceWidth


def test_stats_pages_cap_the_ladder():
    stride = 120 * 2 * (128 // 32)  # cores x 2 pages x channel tiles, as the resnet sizes them
    pages = torch.zeros(1, 1, VAE_A_MAX_SLICES * stride * 32, 32)
    assert _stats_slices((pages, stride)) == VAE_A_MAX_SLICES
    assert _stats_slices([(pages, stride), (pages, stride)]) == VAE_A_MAX_SLICES
    assert _stats_slices(None) is None
    assert _level_slices(0, 4) == VAE_A_MAX_SLICES and _level_slices(0, 5) > VAE_A_MAX_SLICES  # level 5 is refused


class _MeshTensor:
    """A torch stand-in for a mesh tensor: one tensor per (tp, h) device."""

    def __init__(self, parts: dict, layout="tile"):
        self.parts, self.layout = parts, layout

    @property
    def shape(self):
        return tuple(next(iter(self.parts.values())).shape)

    def memory_config(self):
        return None


class _MeshOps:
    """The ttnn calls of Flux3KleinVae.decode on _MeshTensor, with ttnn's semantics per device."""

    bfloat16, ROW_MAJOR_LAYOUT, TILE_LAYOUT = "bf16", "row_major", "tile"

    @staticmethod
    def typecast(x, dtype):
        return x

    @staticmethod
    def deallocate(x):
        pass

    @staticmethod
    def to_layout(x, layout):
        return _MeshTensor(x.parts, layout)

    @staticmethod
    def slice(x, begin, end):
        index = tuple(slice(b, e) for b, e in zip(begin, end))
        return _MeshTensor({coord: part[index] for coord, part in x.parts.items()}, x.layout)

    @staticmethod
    def mesh_partition(x, dim, cluster_axis, memory_config=None):
        n = max(coord[cluster_axis] for coord in x.parts) + 1
        assert x.shape[dim] % n == 0
        return _MeshTensor(
            {coord: part.chunk(n, dim)[coord[cluster_axis]] for coord, part in x.parts.items()}, x.layout
        )

    @staticmethod
    def all_gather(x, dim, mesh_axis, use_hyperparams=True):
        n = max(coord[mesh_axis] for coord in x.parts) + 1

        lines = {}  # one concatenation per line of devices, shared by the devices on it

        def line(coord):
            key = tuple(c for axis, c in enumerate(coord) if axis != mesh_axis)
            if key not in lines:
                along = [tuple(i if axis == mesh_axis else c for axis, c in enumerate(coord)) for i in range(n)]
                lines[key] = torch.cat([x.parts[c] for c in along], dim)
            return lines[key]

        return _MeshTensor({coord: line(coord) for coord in x.parts}, x.layout)


class _Decoder:
    """Flux2VaeDecoder's unpatchify (the input's H shards: the decoder's own unless given) and an identity forward."""

    def __init__(self, h_shards: int):
        self.h_shards, self.calls = h_shards, []

    def preprocess_and_unpatchify(self, z, *, height, width, h_factor=None):
        h_factor = self.h_shards if h_factor is None else h_factor
        assert height % (PATCH * h_factor) == 0
        self.calls.append(h_factor)
        h, w = height // h_factor // PATCH, width // PATCH
        done = {}  # replicated inputs share one tensor
        for part in z.parts.values():
            if id(part) not in done:
                done[id(part)] = _unpatchify(part, h, w)
        return _MeshTensor({coord: done[id(part)] for coord, part in z.parts.items()})

    def forward(self, z, *, traced, gather_hw):
        return z


@MESHES
@pytest.mark.parametrize("width, height", SHAPES)
def test_decode_hands_every_device_its_latent_rows_and_columns(width, height, h_shards, monkeypatch):
    from types import SimpleNamespace

    from ....pipelines.flux3_klein import vae_flux3_klein as module

    h, w = latent_shape(height, width)
    tokens = torch.randn(h * w, PATCH**2)  # one latent channel: the split does not depend on the channels
    expected = _unpatchify(tokens[None], h, w)
    mesh = (W_SHARDS, h_shards)
    coords = [(r, c) for r in range(W_SHARDS) for c in range(h_shards)]
    decoder = _Decoder(h_shards)
    seen = {}
    decoder.forward = lambda z, *, traced, gather_hw: seen.setdefault("z", z)

    vae = object.__new__(module.Flux3KleinVae)
    vae._device = SimpleNamespace(shape=mesh)
    vae.h_axis, vae._tp_axis, vae._dit_sp_axis = 1, 0, 1
    vae._ccl = vae._dit_ccl = _MeshOps
    vae._decoders = {(width, height): decoder}
    monkeypatch.setattr(module, "ttnn", _MeshOps)
    monkeypatch.setitem(Conv2d.slice_params, mesh, {})

    state = _MeshTensor({(r, c): tokens[None].chunk(h_shards, 1)[c] for r, c in coords})
    out = vae.decode(state, size=(width, height))

    rows, cols = h * PATCH // h_shards, w * PATCH // W_SHARDS
    # booleans first: pytest formats the operands of every assert, and these are tensors per device
    wrong_inputs, wrong_outputs = [], []
    for r, c in coords:
        band = expected[:, c * rows : (c + 1) * rows]
        if not torch.equal(seen["z"].parts[(r, c)], band[:, :, r * cols : (r + 1) * cols]):
            wrong_inputs.append((r, c))  # the decoder's input: this device's latent rows and columns
        if not torch.equal(out.parts[(r, c)], band):
            wrong_outputs.append((r, c))  # H-sharded, whole columns
    assert not wrong_inputs and not wrong_outputs
    assert decoder.calls == [h_shards if token_rows_split(h, h_shards) else 1]
