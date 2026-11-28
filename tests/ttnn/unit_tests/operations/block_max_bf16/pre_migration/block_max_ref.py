# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Independent torch references and tensor helpers for the block_max_bf16 pre-migration suite.

Precision convention (the golden suite's ``check_block_select``): the op selects data only, so every
compared value is byte-exact (bf16 bit patterns compared as int16), never PCC or allclose.
"""

from __future__ import annotations

import torch
import ttnn

SENTINEL_FILL = 99.0  # exactly representable in bf16; never a max of the generated inputs


def round8(n: int) -> int:
    return -(-int(n) // 8) * 8


def written(n: int, width: int) -> int:
    """Columns the op writes for ``n``: n rounded up to 8, at most the output width."""
    return min(round8(n), width)


# ----------------------------------------------------------------------------------------- inputs


def fold_input(shape, seed: int) -> torch.Tensor:
    """fp32 tensor of bf16 values (the op keeps the high halves): random, ties, -inf runs, all-negative rows."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(*shape, generator=g)
    rows = x.view(-1, shape[-1])
    nrows = rows.shape[0]
    rows[:, ::7] = 0.5  # ties
    if nrows > 0:
        rows[0, :] = -rows[0, :].abs() - 1  # every value negative
    if nrows > 1:
        rows[1, ::3] = float("-inf")
    if nrows > 2:
        rows[2, :] = float("-inf")  # a whole row of -inf
    if nrows > 3:
        rows[3, shape[-1] // 2 :] = float("-inf")  # a causal tail
    return x.to(torch.bfloat16).float()


def fold_ref(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Column b = max of input columns [b factor, (b + 1) factor), as bf16 (exact)."""
    shape = list(x.shape)
    return x.reshape(*shape[:-1], shape[-1] // factor, factor).amax(-1).to(torch.bfloat16)


def strip_input(lead, S: int, NB: int, seed: int) -> torch.Tensor:
    """Strip-major bf16 maxes [*lead, S, NB * 32]: strip s, block b's 32 row values at [32 b, 32 b + 32)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(*lead, S, NB * 32, generator=g)
    x[..., 0, ::5] = float("-inf")
    if S > 1:
        x[..., 1, 64:96] = 0.25  # a tied block
    return x.to(torch.bfloat16)


def strip_device_view(x: torch.Tensor, S: int, NB: int) -> torch.Tensor:
    """The op's input layout: [*lead, S NB / 8, 256] (512-byte pages of 8 blocks)."""
    return x.reshape(*x.shape[:-2], S * NB // 8, 256)


def strip_ref(x: torch.Tensor, S: int, NB: int) -> torch.Tensor:
    """[*lead, S, NB * 32] -> [*lead, 32 S, NB]: each strip's [NB, 32] transposed (exact copies)."""
    lead = list(x.shape[:-2])
    return x.reshape(*lead, S, NB, 32).transpose(-1, -2).reshape(*lead, S * 32, NB)


# ----------------------------------------------------------------------------------------- device helpers


def _mapper(device):
    if hasattr(device, "get_num_devices") and device.get_num_devices() > 1:
        return ttnn.ReplicateTensorToMesh(device)
    return None


def up(t: torch.Tensor, device, dtype, memory_config=None, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_mapper(device),
    )


def up_n(n: int, device):
    """``n_tensor``: one uint32 value, ROW_MAJOR DRAM (the golden suite's [1, 1, 1, 1])."""
    return up(torch.tensor([int(n)], dtype=torch.int32).view(1, 1, 1, 1), device, ttnn.uint32)


def downs(t) -> list[torch.Tensor]:
    """Every device's copy of ``t`` (a replicated tensor), as torch."""
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def assert_bytes_equal(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    assert got.shape == want.shape, f"{what}: shape {tuple(got.shape)} != {tuple(want.shape)}"
    bad = (bits(got) != bits(want)).nonzero()
    assert len(bad) == 0, f"{what}: {len(bad)} mismatching values, first {bad[:4].tolist()}"


def check_written(out_t, want: torch.Tensor, m: int, prev: torch.Tensor | None, what: str) -> None:
    """The first ``m`` columns equal ``want`` byte-exactly on every device; with ``prev`` (the caller's output
    buffer's previous contents) the columns from ``m`` on are untouched."""
    for i, d in enumerate(downs(out_t)):
        d = d.reshape(want.shape)
        assert_bytes_equal(d[..., :m], want[..., :m], f"{what} dev{i} written columns")
        if prev is not None and m < want.shape[-1]:
            assert_bytes_equal(d[..., m:], prev[..., m:], f"{what} dev{i} columns past n (must be untouched)")
