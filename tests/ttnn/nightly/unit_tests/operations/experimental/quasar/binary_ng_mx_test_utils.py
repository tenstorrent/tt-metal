# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""MX helpers of test_binary_ng_mx.py: the raw-bit route, the exact MX oracle and the route check.

Not collected by pytest: the module has no `test_` prefix.

The oracle has three parts. Each part starts from the bytes that the device holds.
1. The reference decoder (mx_reference_codec.py, tt-regen-spec 4.2) reads each MX operand from the host buffer that
   goes to the device. The unpacker gives Float16_b. Every MX value has at most 8 significant bits, so the bfloat16
   of the decoded float32 is exact: it is the operand that the kernel sees. A bfloat16 operand is its own value.
2. The kernel's bfloat16 model (binary_ng_quasar_test_utils.py) computes the result from the operands.
3. For an MX output, the reference encoder packs the result (4.2 is the packer's rule). The reference decoder then
   reads the expected bytes and the device's bytes. For a bfloat16 output, the test compares the values.

No golden goes through the ttnn MX codec. ttnn.from_torch may build an MX input, and the test then reads its bytes.
ttnn.to_torch never reads an MX tensor here. The comparison is exact: 0 differing elements. As production's
allow_nonfinite does, a NaN matches a NaN of any sign and payload, and an Inf matches an Inf of its own sign. The
sign of a zero does not count.

A tensor here has any rank. Its last two dims are whole tiles, and its tiles lie in row-major tile order over the
rows of all the outer dims, as in a 2D tensor of (all rows, columns).
"""

import os
import pathlib
import time

import torch

import ttnn
from tests.ttnn.nightly.unit_tests.operations.experimental.quasar import mx_reference_codec as ref
from tt_py_test_utils_common.utils_for_testing import ulp_distance

# The reference codec's name of each ttnn MX dtype.
MX_FORMAT = {
    ttnn.mxfp8_e4m3: "mxfp8_e4m3",
    ttnn.mxfp8_e5m2: "mxfp8_e5m2",
    ttnn.mxfp6_e2m3: "mxfp6_e2m3",
    ttnn.mxfp6_e3m2: "mxfp6_e3m2",
    ttnn.mxfp4: "mxfp4",
    ttnn.mxint8: "mxint8",
    ttnn.mxint4: "mxint4",
    ttnn.mxint2: "mxint2",
}

_TILE = 32  # a tile is 32x32 elements
_BLOCKS = 32  # MX blocks in a tile, one scale byte each (4.3)


def tile_bytes(dtype):
    """The bytes of one 32x32 MX tile: the 32 scale bytes, then 1,024 elements in their storage width (4.3)."""
    return _BLOCKS + _TILE * _TILE * ref.FORMATS[MX_FORMAT[dtype]].storage_bits // 8


def element_bits(dtype):
    """The bits of one element code of an MX dtype."""
    return ref.FORMATS[MX_FORMAT[dtype]].element_bits


def code_values(dtype):
    """float32 [2^bits]: the value of each element code of an MX dtype at S = 127 (X = 1), by the reference decoder."""
    n = 1 << element_bits(dtype)
    codes = (torch.arange(max(n, _BLOCKS)) % n).reshape(-1, _BLOCKS)
    values = ref.decode_block(torch.full((codes.shape[0],), 127), codes, MX_FORMAT[dtype])
    return values.flatten()[:n]


def host_bytes(t):
    """A copy of the bytes of a host tensor's single shard. For an MX tensor, these are its tiles in row-major tile
    order, each tile as 4.3 lays it out."""
    shard = t.host_buffer().get_shard(ttnn.MeshCoordinate([0, 0]))
    return torch.from_dlpack(shard).clone()


def mx_host_tensor(raw, dtype, shape):
    """A host MX tensor of `shape` in TILE layout that holds exactly the bytes `raw`."""
    t = ttnn.from_torch(torch.zeros(shape, dtype=torch.float32), dtype=dtype, layout=ttnn.TILE_LAYOUT)
    shard = t.host_buffer().get_shard(ttnn.MeshCoordinate([0, 0]))
    view = torch.from_dlpack(shard)
    assert view.numel() == raw.numel(), f"a {dtype} tensor of {shape} holds {view.numel()} bytes, not {raw.numel()}"
    view.copy_(raw)
    assert torch.equal(host_bytes(t), raw), "the host buffer did not take the bytes"
    return t


def _rows_cols(shape):
    """(all rows, columns) of a tensor whose last two dims are whole tiles."""
    *outer, h, w = shape
    assert h % _TILE == 0 and w % _TILE == 0, f"{tuple(shape)} is not a whole number of tiles"
    rows = h
    for d in outer:
        rows *= d
    return rows, w


def _tiles(values):
    """[..., H, W] -> [tiles, 32, 32] in row-major tile order."""
    h, w = _rows_cols(values.shape)
    return values.reshape(h // _TILE, _TILE, w // _TILE, _TILE).permute(0, 2, 1, 3).reshape(-1, _TILE, _TILE)


def _untile(tiles, shape):
    """[tiles, 32, 32] in row-major tile order -> a tensor of `shape`."""
    h, w = _rows_cols(shape)
    return tiles.reshape(h // _TILE, w // _TILE, _TILE, _TILE).permute(0, 2, 1, 3).reshape(shape)


def decode(raw, dtype, shape):
    """float32 of `shape`: the reference decoding of the MX bytes of a tensor of `shape`."""
    h, w = _rows_cols(shape)
    size = tile_bytes(dtype)
    count = (h // _TILE) * (w // _TILE)
    assert raw.numel() == count * size, f"{raw.numel()} bytes are not {count} {dtype} tiles of {size} bytes"
    fmt = MX_FORMAT[dtype]
    tiles = [ref.decode_tile(raw[k * size : (k + 1) * size], fmt) for k in range(count)]
    return _untile(torch.stack(tiles), tuple(shape))


def encode(values, dtype):
    """uint8: the reference encoding of float32 values as MX tiles in row-major tile order."""
    fmt = MX_FORMAT[dtype]
    return torch.cat([ref.encode_tile(tile.contiguous(), fmt) for tile in _tiles(values.float())])


def scale_bytes(raw, dtype):
    """int64 [tiles, 32]: the scale bytes of each tile of the MX bytes `raw`."""
    return raw.reshape(-1, tile_bytes(dtype))[:, :_BLOCKS].long()


def as_bf16(values):
    """The bfloat16 of decoded MX values. The conversion is exact; the assertion checks that it is."""
    out = values.to(torch.bfloat16)
    exact = (out.float() == values) | values.isnan()
    assert bool(exact.all()), "a decoded MX value is not exact in bfloat16"
    return out


def mx_operand(device, host, memory_config):
    """(device tensor, bfloat16 values) of an MX host tensor: the device gets its bytes, and the values are the
    reference decoding of the same bytes, as the unpacker gives them."""
    raw = host_bytes(host)
    values = as_bf16(decode(raw, host.dtype, tuple(host.shape)))
    return ttnn.to_device(host, device, memory_config=memory_config), values


def expected_mx(golden, dtype):
    """float32: the bfloat16 result `golden` packed by the reference encoder, then read back by the reference
    decoder."""
    return decode(encode(golden.float(), dtype), dtype, tuple(golden.shape))


def device_bytes(t):
    """The bytes of an MX device tensor, read back with ttnn.from_device: its tiles in row-major tile order."""
    return host_bytes(ttnn.from_device(t))


def device_mx(t):
    """float32: the bytes of an MX device tensor, decoded by the reference decoder."""
    return decode(device_bytes(t), t.dtype, tuple(t.shape))


def wrong_elements(got, golden):
    """The sorted flat indices where `got` differs from `golden` at 0 ULP. As production's allow_nonfinite does, a
    NaN matches a NaN whatever its sign and payload, and an Inf matches only an Inf of the same sign; ulp_distance
    counts +0 and -0 as equal."""
    assert got.shape == golden.shape, f"{tuple(got.shape)} vs {tuple(golden.shape)}"
    far = ulp_distance(got, golden) > 0
    inf_differs = (got.isinf() | golden.isinf()) & (got != golden)
    differ = (far | inf_differs) & ~(got.isnan() & golden.isnan())
    return torch.nonzero(differ.flatten()).flatten().tolist()


def mx_output_wrong(t, golden):
    """The elements of the MX device tensor `t` that differ from the bfloat16 result `golden` packed by 4.2."""
    return wrong_elements(device_mx(t), expected_mx(golden, t.dtype))


def bf16_output_wrong(t, golden):
    """The elements of the bfloat16 device tensor `t` that differ from the bfloat16 result `golden`."""
    return wrong_elements(ttnn.to_torch(t), golden)


def block_cells(b):
    """(rows, cols) of the 32 elements of MX block b in a 32x32 tile. 4.3: a block is two consecutive rows of one
    16x16 face, and the faces are in row-major order."""
    face, pair = divmod(b, 8)
    rows = [16 * (face // 2) + 2 * pair + i // 16 for i in range(32)]
    cols = [16 * (face % 2) + i % 16 for i in range(32)]
    return rows, cols


def pack_tile(scales, codes, dtype):
    """The bytes of one MX tile from its 32 scale bytes and its codes [32 blocks, 32 elements]. 4.3: the scale bytes
    come first. Element 32 b + i follows in its storage width, from the low bits of each byte up. A 6-bit code sits in
    the high bits of its 8-bit slot."""
    f = ref.FORMATS[MX_FORMAT[dtype]]
    out = torch.zeros(tile_bytes(dtype), dtype=torch.int64)
    out[:_BLOCKS] = torch.as_tensor(scales, dtype=torch.int64)
    for k, code in enumerate(torch.as_tensor(codes).flatten().tolist()):
        unit = code << (f.storage_bits - f.element_bits)
        bit = k * f.storage_bits
        out[_BLOCKS + bit // 8] |= unit << (bit % 8)
    return out.to(torch.uint8)


# The inspector's kernels.yaml, where the runtime resolves it: <logs dir>/generated/inspector, with the logs dir
# TT_METAL_LOGS_PATH if set, else TT_METAL_HOME, else the working directory (as test_binary_ng_quasar_native.py does).
_LOGS_ROOT = pathlib.Path(os.environ.get("TT_METAL_LOGS_PATH") or os.environ.get("TT_METAL_HOME") or os.getcwd())
_KERNELS_YAML = _LOGS_ROOT / "generated" / "inspector" / "kernels.yaml"
# pytest imports this module at collection, before any device opens. So a kernels.yaml of this process is newer.
_IMPORT_TIME = time.time()


def kernel_sources():
    """The kernel sources that this process bound, in the order of kernels.yaml. The file grows as ops build
    programs, so the sources after a recorded length are those of the ops that ran since."""
    mtime = _KERNELS_YAML.stat().st_mtime
    assert mtime >= _IMPORT_TIME, f"{_KERNELS_YAML} predates this process: a stale file proves nothing"
    return [ln.split("source:", 1)[1].strip() for ln in _KERNELS_YAML.read_text().splitlines() if "source:" in ln]


def _binary_ng_since(before):
    return [s for s in kernel_sources()[len(before) :] if "/binary_ng/device/kernels" in s]


def assert_native(before, what):
    """The op that ran since `before` bound binary_ng kernels from kernels_qsr/ only: the native factory ran it.
    The deprecated Metal 2.0 factory (kernels_dfb/) is not a valid path for MX."""
    bound = _binary_ng_since(before)
    qsr = [s for s in bound if "/kernels_qsr/" in s]
    assert qsr and len(qsr) == len(bound), f"{what}: not on the native factory, the op bound {bound}"


def assert_not_native(before, what):
    """The op that ran since `before` bound no kernels_qsr/ source. Returns the binary_ng sources that it bound."""
    bound = _binary_ng_since(before)
    assert not [s for s in bound if "/kernels_qsr/" in s], f"{what}: the native factory ran it, the op bound {bound}"
    return bound
