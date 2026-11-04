# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone test: ttnn.typecast to an unsigned-int output on Quasar.

`ttnn.typecast` binds the mainline copy/typecast op. Casting to ttnn.uint32 / ttnn.uint16 is NOT supported on
Quasar:

  * UInt32 / UInt16 are rejected as DFB formats by `is_supported_quasar` (tt_backend_api_types.cpp) — it
    supports Int32 and the byte-size-equal RawUInt32 / RawUInt16, not UInt32 / UInt16. So the op FATAL'd at
    program build ("DFB 'out' has data format 'UInt32' which is not supported on architecture QUASAR").
  * Remapping the output DFB to RawUInt32 / RawUInt16 lets the program BUILD, but the generic Quasar typecast
    LLK path has no uint32/uint16 store mode, so the SFPU->pack conversion produces WRONG values (verified on
    the emulator: all-zero for uint32, bit-garbage for uint16). So the remap was reverted.

Current behaviour: the typecast device op (validate_on_program_cache_miss) now REJECTS a UINT32/UINT16 output
on Quasar with a clear message ("... not supported on Quasar ...; cast to INT32 instead"), rather than silently
returning wrong data. INT32 / FLOAT32 / BFLOAT16 outputs are unaffected and correct.

Where llama hits it: the sampling / log-probs path (sampling/tt_log_probs.py: typecast(..., ttnn.uint32) for
topk indices / remainder). That path is NOT exercised by the e2e teacher-forcing accuracy test (next-token
selection is host torch.argmax). To run it on Quasar, either route those casts through INT32 (indices are
non-negative and fit int32) or add a uint32/uint16 SFPU store mode to the Quasar LLK.

Run (Quasar sim, 2-node emulator, SLOW dispatch):
    TTSIM_QSR_TC_LEGACY_TRUNCATION_ALIAS=0 TT_METAL_SIMULATOR=~/sim/libttsim.so TT_METAL_SLOW_DISPATCH_MODE=1 \
        TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="3,2" MESH_DEVICE=N150 \
        pytest models/experimental/llama32_1b_quasar/tests/debug_ops/test_quasar_typecast_uint32.py
"""

import pytest
import torch
from loguru import logger

import ttnn

H, W = 32, 64  # tile-aligned small shape (2 tiles wide)


def _is_quasar():
    try:
        return "quasar" in ttnn.get_arch_name()
    except Exception:
        return False


def _to_device_tile(t, dtype, mesh_device):
    """Host-tilize then to_device so the test isolates typecast (no device tilize in the path)."""
    host = ttnn.from_torch(
        t,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )
    return ttnn.to_device(host, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "in_dtype, out_dtype, out_label",
    [
        (ttnn.int32, ttnn.uint32, "int32->uint32"),  # llama sampling path (topk indices / remainder)
        (ttnn.bfloat16, ttnn.uint32, "bf16->uint32"),
        (ttnn.bfloat16, ttnn.uint16, "bf16->uint16"),
    ],
)
def test_typecast_to_unsigned_int_rejected_on_quasar(mesh_device, in_dtype, out_dtype, out_label, expect_error):
    """typecast X -> uint32/uint16 is rejected on Quasar with a clear message (no uint32/uint16 LLK store mode).
    On WH/BH these casts are supported, so this reject-assertion is Quasar-only."""
    if not _is_quasar():
        pytest.skip("uint32/uint16 typecast reject is Quasar-only; WH/BH support these outputs natively")

    torch.manual_seed(0)
    vals = torch.randint(0, 200, (1, 1, H, W))
    torch_in = vals.to(torch.int32 if in_dtype == ttnn.int32 else torch.float32)
    x = _to_device_tile(torch_in, in_dtype, mesh_device)

    # The FATAL fires in the device op's validate at program build (when the op is invoked).
    with expect_error(RuntimeError, "not supported on Quasar"):
        out = ttnn.typecast(x, out_dtype)
        ttnn.synchronize_device(mesh_device)
        _ = ttnn.to_torch(out)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "in_dtype, in_label",
    [(ttnn.bfloat16, "bf16"), (ttnn.float32, "fp32")],
)
def test_typecast_to_int32(mesh_device, in_dtype, in_label):
    """Positive control / recommended Quasar alternative: typecast float -> INT32 (a supported Quasar DFB
    format) must return exact integer values. This is the path llama's sampling should use instead of UINT32."""
    torch.manual_seed(0)
    vals = torch.randint(0, 200, (1, 1, H, W))  # non-negative, exact in bf16 (<256)
    torch_in = vals.to(torch.float32)

    x = _to_device_tile(torch_in, in_dtype, mesh_device)
    out = ttnn.typecast(x, ttnn.int32)
    ttnn.synchronize_device(mesh_device)

    got = ttnn.to_torch(out).reshape(-1).to(torch.int64)
    ref = vals.reshape(-1).to(torch.int64)
    match = (got == ref).float().mean().item()
    logger.info(
        f"[typecast {in_label}->int32] match={match * 100:.1f}% got[:6]={got[:6].tolist()} ref[:6]={ref[:6].tolist()}"
    )
    assert got.numel() == ref.numel(), f"element count mismatch: {got.numel()} vs {ref.numel()}"
    assert (
        match == 1.0
    ), f"typecast {in_label}->int32 mismatched: {match * 100:.1f}% (got {got[:16].tolist()} vs {ref[:16].tolist()})"
