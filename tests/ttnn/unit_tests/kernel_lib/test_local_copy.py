# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Unit test for the local L1 -> L1 copy helpers
# (ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp):
#
#   local_addr / set_read_state / read_with_state (all four overloads) /
#   set_read_trid / async_read_barrier_with_trid
#
# One reader kernel per program: it stages DRAM -> cb_src, copies cb_src -> cb_dst using ONLY the
# helpers, then writes cb_dst -> DRAM. Bit-exact output therefore proves the L1 -> L1 copy landed.
# There is no write-side equivalent to test against: Noc::async_write resolves its destination as
# AddressType::NOC, which a CircularBuffer/DataflowBuffer destination static_asserts against, so
# the self-aimed READ is the only way to express this copy at all.
#
# The `noc` parametrization is the correctness gate on the helper's per-NoC coordinate indexing:
# NOC 0 and NOC 1 have DIFFERENT coordinate spaces, so a helper that indexed my_x[0]/my_y[0]
# instead of my_x[noc_id]/my_y[noc_id] would aim the NOC 1 read at a DIFFERENT core's L1 and the
# noc=1 cases would fail (or trip the NoC sanitizer), while the noc=0 cases stayed green.
#
import torch
import pytest
import ttnn
from loguru import logger

TILE_BYTES = 32 * 32 * 2  # bf16 tile
KERNEL_DIR = "tests/ttnn/unit_tests/kernel_lib/kernels"

# Must match the MODE_* constants in kernels/local_copy_reader.cpp.
MODES = {
    "gather_raw": 0,  # per-page set_read_state + read_with_state(raw dst addr)
    "gather_typed": 1,  # per-page set_read_state + read_with_state(cb, {.offset_bytes})
    "bcast_raw": 2,  # ONE set_read_state, N reads -> raw dst addrs
    "bcast_typed": 3,  # ONE set_read_state, N reads -> cb offsets
    "trid": 4,  # rotating trid + async_read_barrier_with_trid
    "direct": 5,  # local_addr() fed to a plain noc.async_read (stateless)
    "typed_noargs": 6,  # read_with_state(cb, src) with no dst_args
}

# Modes that copy source page 0 into EVERY destination page (the amortised-state shape, where the
# source is programmed once and only the destination moves).
BCAST_MODES = {"bcast_raw", "bcast_typed", "typed_noargs"}


def _run_local_copy(device, mode, payload_tiles, noc, num_trids=2, core=(0, 0)):
    mode_id = MODES[mode]
    page_bytes = TILE_BYTES
    payload_pages = payload_tiles
    cx, cy = core

    # ---- tensors: one row of tiles, so each 32-column block is exactly one tile page ----
    shape = [1, 1, 32, 32 * payload_tiles]
    payload = torch.arange(0, payload_tiles * 1024, dtype=torch.float32).reshape(shape).to(torch.bfloat16)
    input_tensor = ttnn.from_torch(
        payload, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    io_tensors = [input_tensor, output_tensor]

    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy))])

    cb_src, cb_dst = 0, 1
    cbs = [
        ttnn.CBDescriptor(
            total_size=payload_pages * page_bytes,
            core_ranges=crs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.bfloat16, page_size=page_bytes)
            ],
        )
        for idx in (cb_src, cb_dst)
    ]

    ct = [cb_src, cb_dst, payload_pages, page_bytes, mode_id, num_trids]
    ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    rt = ttnn.RuntimeArgs()
    rt[cx][cy] = [input_tensor.buffer_address(), output_tensor.buffer_address()]

    # ReaderConfigDescriptor -> NCRISC/NOC0, WriterConfigDescriptor -> BRISC/NOC1. The kernel builds
    # a default `Noc noc;`, so the RISC-V's own NoC is the one under test.
    config = ttnn.WriterConfigDescriptor() if noc == 1 else ttnn.ReaderConfigDescriptor()

    kernel = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/local_copy_reader.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=crs,
        compile_time_args=ct,
        runtime_args=rt,
        config=config,
    )

    pd = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)
    output = ttnn.generic_op(io_tensors, pd)

    torch_out = ttnn.to_torch(output).reshape(shape).to(torch.float32)
    torch_in = payload.to(torch.float32)

    ok = True
    for i in range(payload_tiles):
        got = torch_out[..., i * 32 : (i + 1) * 32]
        src_page = 0 if mode in BCAST_MODES else i
        want = torch_in[..., src_page * 32 : (src_page + 1) * 32]
        if not torch.equal(got, want):
            logger.error(f"mode={mode} noc={noc}: page {i} mismatch (expected source page {src_page})")
            ok = False
    assert ok, f"mode={mode} noc={noc} tiles={payload_tiles}: local L1->L1 copy not bit-exact"
    logger.info(f"mode={mode} noc={noc} tiles={payload_tiles}: PASS")


# ---------- SMOKE: the plain amortised-state loop on the default NoC ----------
def test_smoke(device):
    _run_local_copy(device, mode="bcast_typed", payload_tiles=1, noc=0)


# ---------- coverage: every helper overload x both NoCs x a few payload sizes ----------
@pytest.mark.parametrize("mode", list(MODES.keys()))
@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("payload_tiles", [1, 4])
def test_overloads(device, mode, noc, payload_tiles):
    _run_local_copy(device, mode=mode, payload_tiles=payload_tiles, noc=noc)


# ---------- trid: the rotating-slot drain must be correct for depth 1..4 ----------
@pytest.mark.parametrize("num_trids", [1, 2, 4])
@pytest.mark.parametrize("payload_tiles", [1, 3, 8])
def test_trid_depth(device, num_trids, payload_tiles):
    _run_local_copy(device, mode="trid", payload_tiles=payload_tiles, noc=0, num_trids=num_trids)


# ---------- off-origin core: my_x/my_y must come from the core, not be assumed to be (0,0) ----------
@pytest.mark.parametrize("noc", [0, 1])
def test_off_origin_core(device, noc):
    _run_local_copy(device, mode="gather_typed", payload_tiles=4, noc=noc, core=(3, 2))
