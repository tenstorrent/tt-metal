# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Independent torch reference and tensor helpers of the ckv_bfp4_rows pre-migration suite.

The op's contract (README of ``ttnn/ttnn/bringup/ckv_bfp4_ttnn``): rows ``[src_row0, src_row0 + n)`` of a bf16
ROW_MAJOR ``[.., 512]`` tensor become 320-byte bfloat4_b row records at rows ``[row0, row0 + n)`` of a uint8 ROW_MAJOR
``[.., 320]`` table, written in place, byte-exact to tt-metal's host bfloat4_b packer rule:

* per 16-value group, the shared exponent is the group's max fp32 biased exponent;
* every value's 24-bit significand (implicit 1) is shifted right by ``shared - own exponent``, then rounded to 3
  magnitude bits, round to nearest even on the dropped 21 bits, clamped at 7; a zero/denormal exponent gives 0;
  a zero magnitude has no sign;
* record: the 32 shared exponents, the same 32 again, then 256 bytes of sign-magnitude nibbles (value 2i in the low
  nibble of byte i, value 2i + 1 in the high nibble).

Written from that rule, not from the op's own test reference.
"""

from __future__ import annotations

import torch
import ttnn

WIDTH, RECORD_BYTES, GROUP = 512, 320, 16
FILL = 0xAB  # the table's pre-fill: rows the call must not write keep it
MAG_BITS = 3


def encode_rows(x: torch.Tensor) -> torch.Tensor:
    """[n, 512] values (bf16 exact) -> [n, 320] uint8 bfloat4_b row records by the host rule."""
    x = x.reshape(-1, WIDTH).to(torch.bfloat16).float()
    n = x.shape[0]
    u = x.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    exp = (u >> 23) & 0xFF
    sign = (u >> 31) & 1
    sig = (u & 0x7FFFFF) | (1 << 23)
    exp_g = exp.view(n, WIDTH // GROUP, GROUP)
    shared = exp_g.amax(-1, keepdim=True)
    aligned = sig.view(n, WIDTH // GROUP, GROUP) >> (shared - exp_g).clamp(max=62)
    drop = 24 - MAG_BITS
    q = aligned >> drop
    rem = aligned & ((1 << drop) - 1)
    half = 1 << (drop - 1)
    up = (rem > half) | ((rem == half) & ((q & 1) == 1))
    q = (q + up.to(torch.int64)).clamp(max=(1 << MAG_BITS) - 1)
    q = torch.where(exp_g == 0, torch.zeros_like(q), q)
    s = torch.where(q == 0, torch.zeros_like(q), sign.view(n, WIDTH // GROUP, GROUP))
    codes = ((s << MAG_BITS) | q).view(n, WIDTH)
    nibbles = codes[:, 0::2] | (codes[:, 1::2] << 4)
    shared = shared.view(n, WIDTH // GROUP)
    return torch.cat([shared, shared, nibbles], dim=-1).to(torch.uint8)


def make_rows(total: int, seed: int) -> torch.Tensor:
    """[total, 512] bf16 rows whose records reveal scale, indexing, rounding and sign errors: per-row scales over
    2^-12 .. 2^5, a large first group, a zero group, an all-zero row, exact rounding ties and a clamp at the top code,
    and a per-row impulse so swapped or shifted rows differ."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(total, WIDTH, generator=g) * torch.pow(2.0, torch.randint(-12, 6, (total, 1), generator=g).float())
    x[:, :GROUP] *= 50.0
    if total > 2:
        x[1, 32:48] = 0.0
        x[2] = 0.0
    if total > 3:
        # group max 1.0 (step 0.25): 0.375 and 0.625 are ties (to even: 0.5, 0.5), 0.875 a tie (to 1.0),
        # 1.9375 rounds past the top code (clamped at 7 = 1.75), negatives mirror them
        ties = torch.tensor([1.0, 0.375, 0.625, 0.875, -0.375, -0.625, -0.875, 0.125] * 2)
        x[3, :GROUP] = ties
        x[3, GROUP : 2 * GROUP] = torch.tensor([1.0, 1.9375, -1.9375, 0.0625] * 4)
    idx = torch.arange(total)
    x[idx, 64 + (idx % 400)] = 3.0 + (idx % 7).float()
    return x.to(torch.bfloat16)


def interleaved_dram():
    return ttnn.DRAM_MEMORY_CONFIG


def nd_sharded_dram(device, shard_rows: int = 16):
    """ND-sharded DRAM memory config: [1, 1, shard_rows, 512] shards round-robin over the DRAM banks (a served cache
    slab's layout)."""
    banks = device.dram_grid_size().x
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(banks)])
    spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, shard_rows, WIDTH],
        grid=grid,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    return ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=spec)


def _mapper(device):
    return ttnn.ReplicateTensorToMesh(device) if device.get_num_devices() > 1 else None


def to_src(device, x: torch.Tensor, shape, memory_config=None):
    """bf16 ROW_MAJOR device source of shape ``shape`` from the [total, 512] rows ``x`` (replicated on a mesh)."""
    return ttnn.from_torch(
        x.reshape(shape),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_mapper(device),
    )


def to_table(device, shape, fill: int = FILL, memory_config=None):
    """uint8 ROW_MAJOR record table of shape ``shape`` (last dim 320), pre-filled with ``fill``."""
    return ttnn.from_torch(
        torch.full(tuple(shape), fill, dtype=torch.uint8),
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_mapper(device),
    )


def table_host(t) -> list[torch.Tensor]:
    """The table as [rows, 320] uint8, one per device (every chip of a mesh)."""
    return [ttnn.to_torch(d).view(torch.uint8).reshape(-1, RECORD_BYTES) for d in ttnn.get_device_tensors(t)]


def src_host(t) -> list[torch.Tensor]:
    return [ttnn.to_torch(d).reshape(-1, WIDTH) for d in ttnn.get_device_tensors(t)]


def check_table(table, x: torch.Tensor, src_row0: int, n: int, row0: int, fill: int = FILL, label: str = ""):
    """Every chip: rows [row0, row0 + n) == the host rule's records of x[src_row0 : src_row0 + n], byte for byte;
    every other row still ``fill``."""
    expected = encode_rows(x[src_row0 : src_row0 + n])
    for chip, got in enumerate(table_host(table)):
        written = got[row0 : row0 + n]
        bad = (written != expected).any(-1).nonzero().flatten()
        assert not len(bad), (
            f"{label} chip {chip} (src_row0={src_row0}, n={n}, row0={row0}): {len(bad)} records differ, "
            f"first rows {bad[:5].tolist()}; exponents equal {torch.equal(written[:, :64], expected[:, :64])}, "
            f"values equal {torch.equal(written[:, 64:], expected[:, 64:])}"
        )
        assert bool((got[:row0] == fill).all()), f"{label} chip {chip}: a row before row0 was written"
        assert bool((got[row0 + n :] == fill).all()), f"{label} chip {chip}: a row past row0 + n was written"
