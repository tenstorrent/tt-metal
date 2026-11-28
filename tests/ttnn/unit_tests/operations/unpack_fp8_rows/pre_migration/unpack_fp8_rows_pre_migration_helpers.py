# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side helpers of the unpack_fp8_rows pre-migration suite: the row encoding, the independent float64 reference
(e4m3 x fp32 scale, then round-to-nearest-even onto the bfloat8_b grid; the golden suite's own reference and its
exact tolerance), device upload/readback on a mesh (each chip its own data) and buffer addresses.

Nothing here imports the operation's planner or program-descriptor code."""

from __future__ import annotations

import torch
import ttnn

ROW = 192  # bytes a row: e4m3 x 128 | fp32 scale (little endian) | 60 zero bytes
VALUES = 128
TILE = 32
GRID_CAP = (11, 10)  # the golden suite's max_grid (a Galaxy's 110-core cap)


# --------------------------------------------------------------------------- reference (golden suite's, verbatim math)


def encode(k: torch.Tensor) -> torch.Tensor:
    """keys [.., 128] float -> 192-byte rows uint8 [.., 192] (scale = max(amax, 1e-4) / 448; e4m3 of k / scale; an
    all-zero row all zero)."""
    amax = k.abs().amax(-1, keepdim=True).float()
    scale = amax.clamp(min=1e-4) * (1.0 / 448.0)
    e = (k.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    out = torch.zeros(*k.shape[:-1], ROW, dtype=torch.uint8)
    out[..., :128] = e.view(torch.uint8)
    out[..., 128:132] = scale.float().contiguous().view(torch.uint8)
    zero = amax.squeeze(-1) == 0
    out[zero] = 0
    return out


def decode(rows: torch.Tensor) -> torch.Tensor:
    """The values the rows hold, fp32 (e4m3 x scale, one fp32 multiply)."""
    v = rows[..., :128].contiguous().view(torch.float8_e4m3fn).float()
    sc = rows[..., 128:132].contiguous().view(torch.float32)
    return v * sc


def bfp8(x: torch.Tensor) -> torch.Tensor:
    """x [.., 128] fp32 -> exactly rounded to nearest even onto the bfloat8_b grid (float64): per 16 values of a row
    E = the max exponent, step 2^(E - 6), q = |x| / step rounded half even, at most 127; an all-zero group is 0."""
    shape = x.shape
    g = x.double().reshape(-1, 16)
    _, e = torch.frexp(g.abs().amax(-1, keepdim=True))
    step = torch.ldexp(torch.ones_like(g[:, :1]), e - 7)
    q = torch.round(g.abs() / step).clamp(max=127)
    out = torch.where(g.abs().amax(-1, keepdim=True) == 0, torch.zeros_like(g), torch.sign(g) * q * step)
    return out.float().reshape(shape)


def reference(rows: torch.Tensor) -> torch.Tensor:
    """rows [.., 192] uint8 -> the expected decoded values [.., 128] float32 (chunked: bounded host memory)."""
    flat = rows.reshape(-1, ROW)
    out = torch.empty(flat.shape[0], VALUES, dtype=torch.float32)
    step = 1 << 18
    for i in range(0, flat.shape[0], step):
        out[i : i + step] = bfp8(decode(flat[i : i + step]))
    return out.reshape(*rows.shape[:-1], VALUES)


def keys(lead: tuple, r: int, seed: int) -> torch.Tensor:
    """Random keys [*lead, r, 128] with the golden suite's special rows: a large channel, a tiny row (3), a zero row
    (7), a half-zero row (9); plus a negative-zero row (11) and a row with one non-zero value (13)."""
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(*lead, r, VALUES, generator=g)
    k[..., :, 5] *= 11.0
    if r > 13:
        k[..., 3, :] *= 1e-5
        k[..., 7, :] = 0.0
        k[..., 9, :64] = 0.0
        k[..., 11, :] = -0.0
        k[..., 13, :] = 0.0
        k[..., 13, 77] = -2.5
    return k


def make_rows(chips: int, shape: tuple, seed: int) -> torch.Tensor:
    """Encoded rows [chips, *shape] (shape ends in 192), every chip and every row its own data."""
    assert shape[-1] == ROW
    return encode(keys((chips, *shape[:-2]), shape[-2], seed))


def fill(chips: int, shape: tuple, salt: int) -> torch.Tensor:
    """A prior output content [chips, *shape] that bfloat8_b holds exactly (a constant per row, multiples of 1/2 in
    [-2, 2]), different per chip, row and salt, so a stale or misplaced write is visible."""
    m = 1
    for d in shape[:-1]:
        m *= d
    r = torch.arange(m).reshape(1, m, 1)
    c = torch.arange(chips).reshape(chips, 1, 1)
    v = (((r * 3 + c * 5 + salt * 7) % 9) - 4).float() * 0.5
    return v.expand(chips, m, shape[-1]).reshape(chips, *shape).contiguous()


def expected_out(prior: torch.Tensor, rows: torch.Tensor, row_lo: int, row_hi: int, batch: int, out_row0: int):
    """The output after one call: ``prior`` [chips, .., M, 128] with rows [out_row0, + n) (flattened over the leading
    dims) replaced by the reference of rows [row_lo, row_hi) of batch ``batch`` (leading dims of rows flattened)."""
    chips = prior.shape[0]
    exp = prior.clone().reshape(chips, -1, VALUES)
    r = rows.shape[-2]
    src = rows.reshape(chips, -1, r, ROW)[:, batch, row_lo:row_hi]
    exp[:, out_row0 : out_row0 + (row_hi - row_lo)] = reference(src)
    return exp.reshape(prior.shape)


# --------------------------------------------------------------------------- device helpers


def _mapper(dev):
    n = dev.get_num_devices() if hasattr(dev, "get_num_devices") else 1
    return ttnn.ShardTensorToMesh(dev, dim=0) if n > 1 else None, n


def num_chips(dev) -> int:
    return dev.get_num_devices() if hasattr(dev, "get_num_devices") else 1


def nd_config(dev, shard_rows: int = 16, rank: int = 4) -> ttnn.MemoryConfig:
    """ND-sharded DRAM (the served caches' layout): one ``shard_rows``-row record a shard, round robin over the DRAM
    banks."""
    banks = dev.dram_grid_size().x
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(i, 0), ttnn.CoreCoord(i, 0)) for i in range(banks)])
    spec = ttnn.NdShardSpec(
        shard_shape=[1] * (rank - 2) + [shard_rows, ROW],
        grid=grid,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    return ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=spec)


def rows_to_device(dev, rows: torch.Tensor, mc=None) -> ttnn.Tensor:
    """rows [chips, *shape] uint8 -> device uint8 ROW_MAJOR of ``shape`` on each chip, chip i its own."""
    mapper, chips = _mapper(dev)
    assert rows.shape[0] == chips
    host = rows.reshape(chips * rows.shape[1], *rows.shape[2:]).contiguous()
    return ttnn.from_torch(
        host,
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=dev,
        memory_config=mc or ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


def out_to_device(dev, values: torch.Tensor, mc=None) -> ttnn.Tensor:
    """values [chips, *shape] float -> device bfloat8_b TILE of ``shape`` on each chip."""
    mapper, chips = _mapper(dev)
    assert values.shape[0] == chips
    host = values.reshape(chips * values.shape[1], *values.shape[2:]).contiguous()
    return ttnn.from_torch(
        host,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=mc or ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


def to_host(t: ttnn.Tensor) -> torch.Tensor:
    """device tensor -> [chips, *per-chip shape] (float32 for floats, the raw dtype otherwise)."""
    parts = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(t)]
    parts = [p.float() if p.is_floating_point() else p for p in parts]
    return torch.stack(parts)


def addr(t: ttnn.Tensor) -> int:
    """The device address of chip 0's buffer of ``t``."""
    return ttnn.get_device_tensors(t)[0].buffer_address()


def grid_cap(dev) -> ttnn.CoreCoord:
    g = dev.compute_with_storage_grid_size()
    return ttnn.CoreCoord(min(g.x, GRID_CAP[0]), min(g.y, GRID_CAP[1]))


def grid_cores(dev, max_grid=None) -> int:
    """Cores inside ``max_grid`` clipped to the device grid (the contract's meaning of max_grid)."""
    g = dev.compute_with_storage_grid_size()
    if max_grid is None:
        return g.x * g.y
    mx, my = (max_grid.x, max_grid.y) if hasattr(max_grid, "x") else tuple(max_grid)
    return max(1, min(g.x, mx)) * max(1, min(g.y, my))


def assert_exact(got: torch.Tensor, exp: torch.Tensor, what: str) -> None:
    if not torch.equal(got, exp):
        bad = (got != exp).reshape(got.shape[0], -1, got.shape[-1]).any(-1)
        idx = bad.nonzero()[:8].tolist()
        raise AssertionError(
            f"{what}: {int(bad.sum())} rows differ from the reference (first [chip, row] {idx}), "
            f"max |diff| {(got - exp).abs().max().item():.4g}"
        )
