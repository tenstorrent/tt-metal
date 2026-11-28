# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Input builders and independent host references for the pack_fp8_rows pre-migration suite.

Nothing here imports the operation's package: inputs are built with stock ttnn calls and the references are computed
on the host from the inputs' own raw bytes (read back from the device) or from torch."""

from __future__ import annotations

import torch
import ttnn

VALUES = 128  # e4m3 values a row (the op's contract)
ROW_BYTES = 192  # output bytes a row: 128 values | 4 scale bytes | 60 zero bytes
SCALE_OFF = 128
PAD_OFF = 132

E4M3_POS_ZERO = 0x00
E4M3_NEG_ZERO = 0x80


# --------------------------------------------------------------------------------------------------------- raw bytes
def device_count(t: ttnn.Tensor) -> int:
    return len(ttnn.get_device_tensors(t))


def raw_bytes(t: ttnn.Tensor) -> torch.Tensor:
    """The bytes of every chip's buffer of a device tensor, read from its host copy without any dtype conversion
    (to_torch of FP8_E4M3 is not available / flushes subnormals): [chips, bytes] uint8."""
    host = t.cpu()
    buf = host.host_buffer()
    mesh = t.device()
    out = []
    shape = getattr(mesh, "shape", None)
    if shape is not None and hasattr(buf, "get_shard"):
        rows, cols = tuple(shape)
        for r in range(rows):
            for c in range(cols):
                shard = buf.get_shard(ttnn.MeshCoordinate(r, c))
                out.append(torch.from_dlpack(shard).view(torch.uint8).flatten().clone())
    else:
        out.append(torch.from_dlpack(buf).view(torch.uint8).flatten().clone())
    return torch.stack(out)


def rows_of(shape) -> int:
    r = 1
    for d in list(shape)[:-1]:
        r *= int(d)
    return r


def e4m3_rows(t: ttnn.Tensor) -> torch.Tensor:
    """[chips, rows, 128] uint8: the raw e4m3 bytes of a ROW_MAJOR FP8_E4M3 tensor."""
    rows = rows_of(t.shape)
    return raw_bytes(t)[:, : rows * VALUES].reshape(-1, rows, VALUES)


def scale_rows(t: ttnn.Tensor) -> torch.Tensor:
    """[chips, rows, 4] uint8: the raw fp32 bytes of a ROW_MAJOR [.., M, 1] FLOAT32 tensor."""
    rows = rows_of(t.shape)
    return raw_bytes(t)[:, : rows * 4].reshape(-1, rows, 4)


def output_rows(t: ttnn.Tensor) -> torch.Tensor:
    """[chips, rows, 192] uint8 of a device uint8 ROW_MAJOR output, through the ordinary to_torch path."""
    rows = rows_of(t.shape)
    return torch.stack([ttnn.to_torch(x).reshape(rows, ROW_BYTES) for x in ttnn.get_device_tensors(t)])


# --------------------------------------------------------------------------------------------------------- reference
def reference_rows(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Independent reference of the op's contract: values [.., R, 128] uint8 (e4m3 bytes), scales [.., R, 4] uint8
    (fp32 little endian) -> [.., R, 192] uint8 = [values | scale | 60 zero bytes], all zero where every value is +-0."""
    out = torch.zeros((*values.shape[:-1], ROW_BYTES), dtype=torch.uint8)
    out[..., :SCALE_OFF] = values
    out[..., SCALE_OFF:PAD_OFF] = scales
    zero = ((values & 0x7F) == 0).all(dim=-1)
    out[zero] = 0
    return out


def assert_rows_equal(got: torch.Tensor, want: torch.Tensor, what: str = "output") -> None:
    assert got.shape == want.shape, f"{what}: shape {tuple(got.shape)} != {tuple(want.shape)}"
    bad = (got != want).any(dim=-1)
    if bad.any():
        idx = torch.nonzero(bad)[:4].tolist()
        first = tuple(idx[0])
        cols = torch.nonzero(got[first] != want[first]).flatten()[:8].tolist()
        raise AssertionError(
            f"{what}: {int(bad.sum())} of {bad.numel()} rows differ from the reference; first rows {idx}, "
            f"row {first} differs at bytes {cols}: got {got[first][cols].tolist()} want {want[first][cols].tolist()}"
        )


# --------------------------------------------------------------------------------------------------------- inputs
def random_e4m3_bytes(n_rows: int, gen: torch.Generator) -> torch.Tensor:
    """[n_rows, 128] uint8 of normal, finite e4m3 values (exponent field 1..15, no NaN 0x7F / 0xFF), both signs: exactly
    representable, so the host conversion to FP8_E4M3 cannot round them."""
    sign = torch.randint(0, 2, (n_rows, VALUES), generator=gen, dtype=torch.int32) << 7
    exp = torch.randint(1, 16, (n_rows, VALUES), generator=gen, dtype=torch.int32) << 3
    man = torch.randint(0, 8, (n_rows, VALUES), generator=gen, dtype=torch.int32)
    b = sign | exp | man
    b = torch.where((b & 0x7F) == 0x7F, b - 1, b)  # 0x7F / 0xFF are NaN: take 0x7E / 0xFE (+-448)
    return b.to(torch.uint8)


def e4m3_bytes_to_float(b: torch.Tensor) -> torch.Tensor:
    return b.contiguous().view(torch.float8_e4m3fn).to(torch.float32)


def make_case_bytes(shape, seed: int, special_rows: bool = True):
    """Distinct per-row data for an e4m3 shape [.., M, 128]: (values [.., M, 128] uint8, scales [.., M, 1] float32).
    With ``special_rows`` (rows permitting) some rows are all +0, all -0, half zero, or a single nonzero value, and a
    zero row carries a nonzero scale (the op must zero it), so index, scale-placement and zero-row errors show."""
    gen = torch.Generator().manual_seed(seed)
    rows = rows_of(shape)
    vals = random_e4m3_bytes(rows, gen)
    # distinct scales, varied sign/magnitude (fp32 bytes are copied verbatim)
    scales = (torch.rand(rows, 1, generator=gen) * 3.0 + 0.01) * (
        2.0 ** torch.randint(-20, 20, (rows, 1), generator=gen)
    )
    scales[torch.arange(rows) % 7 == 3] *= -1.0
    if special_rows and rows >= 8:
        vals[1] = E4M3_POS_ZERO  # all +0 (scale nonzero: still written all zero)
        vals[4] = E4M3_NEG_ZERO  # all -0
        vals[5, :64] = E4M3_POS_ZERO  # half zero: a normal row
        vals[6] = E4M3_NEG_ZERO
        vals[6, 127] = 0x08  # one nonzero value at the end: a normal row
        vals[7] = E4M3_POS_ZERO
        vals[7, 0] = 0xFE  # one nonzero value (-448) at the start
        vals[7, 1:64:2] = E4M3_NEG_ZERO  # mixed +0 / -0 around it
    if special_rows and rows >= 3:
        vals[rows - 1] = E4M3_NEG_ZERO  # the last row zero (a partial final block's tail)
    lead = list(shape)[:-1]
    return vals.reshape(*lead, VALUES), scales.reshape(*lead, 1).to(torch.float32)


def to_device_e4m3(vals_u8: torch.Tensor, device, memory_config=ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor:
    return ttnn.from_torch(
        e4m3_bytes_to_float(vals_u8),
        dtype=ttnn.fp8_e4m3,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def to_device_scale(scales: torch.Tensor, device, memory_config=ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor:
    return ttnn.from_torch(
        scales.contiguous(),
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def make_inputs(shape, device, seed: int, special_rows: bool = True):
    """(e4m3 tensor, scale tensor, expected [chips, rows, 192]) on ``device``. The expected rows come from the inputs'
    raw device bytes (what the op reads), packed on the host by reference_rows."""
    vals, scales = make_case_bytes(shape, seed, special_rows)
    e = to_device_e4m3(vals, device)
    s = to_device_scale(scales, device)
    expected = reference_rows(e4m3_rows(e), scale_rows(s))
    return e, s, expected


def preallocated_output(shape, device, fill: int = 0xA5, memory_config=ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor:
    """A uint8 ROW_MAJOR [.., M, 192] buffer filled with ``fill`` (dirty: the op must write every byte)."""
    out_shape = list(shape)[:-1] + [ROW_BYTES]
    return ttnn.from_torch(
        torch.full(out_shape, fill, dtype=torch.uint8),
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def assert_output_spec(out: ttnn.Tensor, e4m3: ttnn.Tensor) -> None:
    want = list(e4m3.shape)[:-1] + [ROW_BYTES]
    assert list(out.shape) == want, f"output shape {list(out.shape)} != {want}"
    assert list(out.padded_shape) == want, f"output padded shape {list(out.padded_shape)} != {want}"
    assert out.dtype == ttnn.uint8, out.dtype
    assert out.layout == ttnn.ROW_MAJOR_LAYOUT, out.layout
    mc = out.memory_config()
    assert mc.buffer_type == ttnn.BufferType.DRAM, mc
    assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, mc
    assert ttnn.is_tensor_storage_on_device(out)
    assert device_count(out) == device_count(e4m3)


# --------------------------------------------------------------------------------------------------------- golden
# The golden suite's own input generator and references (ttnn/ttnn/bringup/fp8_rows_ttnn/tests/unit/test_pack_fp8_rows.py),
# copied verbatim so the cut checks exactly what the golden suite checks.
def golden_keys(nd, m, seed):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(nd, 1, m, 128, generator=g)
    k[..., :, 3] *= 9.0  # a large channel
    k[..., 1, :] *= 1e-6  # a tiny row (values flush below 2^-6 x scale)
    k[..., 2, :] = 0.0  # a pad row
    k[..., 5, :] = -0.0  # a pad row of -0 (a masked negative key)
    k[..., 7, :64] = 0.0  # half a row zero
    return k.to(torch.bfloat16)


def golden_op_rows(e4m3, sc, nd, m):
    """The cast's own outputs packed on the host from their raw bytes: [nd, m, 192]."""
    v = raw_bytes(e4m3)[:, : m * 128].reshape(nd, m, 128)
    s = raw_bytes(sc)[:, : m * 4].reshape(nd, m, 4)
    raw = torch.zeros(nd, m, 192, dtype=torch.uint8)
    raw[..., :128] = v
    raw[..., 128:132] = s
    zero = ((v & 0x7F) == 0).all(dim=-1)
    raw[zero] = 0
    return raw


def golden_recipe(keys):
    """tt-blaze _quantize_scaled_keys (verbatim) + all-zero rows for all-zero keys."""
    values = keys.float().contiguous()
    scales = values.abs().amax(dim=-1, keepdim=True).clamp(1e-4, 3e38) * (1 / 448)
    inv = scales.reciprocal().contiguous()
    inv = (inv.view(torch.int32) & -8192).view(torch.float32)
    normalized = (values * inv).contiguous()
    truncated = (normalized.view(torch.int32) & ~((1 << 20) - 1)).view(torch.float32)
    truncated[normalized.abs() < 2**-6] = 0
    raw = torch.zeros((*values.shape[:-1], 192), dtype=torch.uint8)
    raw[..., :128] = truncated.to(torch.float8_e4m3fn).view(torch.uint8)
    raw[..., 128:132] = scales.contiguous().view(torch.uint8)
    raw[(values == 0).all(dim=-1)] = 0
    return raw


def golden_within_one_step(got, want):
    """The golden tolerance against Blaze's recipe: scales byte-equal, every value within one e4m3 step."""
    assert torch.equal(got[..., 128:132], want[..., 128:132]), "scales differ from Blaze's recipe"
    gv = got[..., :128].contiguous().view(torch.float8_e4m3fn).double()
    wv = want[..., :128].contiguous().view(torch.float8_e4m3fn).double()
    step = torch.where(
        wv.abs() >= 2**-6, 2.0 ** (torch.floor(torch.log2(wv.abs().clamp_min(2**-6))) - 3), 2.0**-6
    )
    far = (gv - wv).abs() > step
    assert not far.any(), f"{int(far.sum())} values beyond one e4m3 step of Blaze's recipe"
