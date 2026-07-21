#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_circular_buffers [--cb-bytes=<n>] [--cb-offset=<n>]

Options:
    --cb-bytes=<n>    Bytes of CB fifo L1 data to show per CB at -vv [default: 64].
    --cb-offset=<n>   Byte offset from the fifo start to read the -vv data from [default: 0].

Description:
    Dumps the live circular-buffer state for every active tensix core, model-agnostically.
    Circular buffers are allocated by the op/model (CreateCircularBuffer); firmware fills the
    fixed cb_interface[] storage from that per-program config at launch, so this reflects
    whatever the running model set up.

    Geometry + fifo pointers come from the per-RISC cb_interface global (read from the running
    kernel ELF via the RISC debug interface, non-intrusive). Live tile counts (occupancy) come
    from the tensix overlay/stream scratch registers, which the dataflow API uses for CB flow
    control on both Wormhole and Blackhole. occupancy = recv - acked; the State column reads it
    against the fifo capacity: "full" (consumer stuck), "empty" (producer never filled), or
    "partial". Remote (global) CBs read pages_sent/pages_acked from the L1 flow-control counters.

    ncrisc is skipped (its private memory is unreadable on WH and reading it can break BH cores,
    tt-exalens#895) and trisc1/math has no cb_interface; local occupancy still comes from the
    tensix-wide stream registers, so it is unaffected. trisc pointers/sizes are normalized to
    bytes (compute stores them in 16B words). The raw remote role-specific words (w5/w6/w7) are
    shown at -v.

Owner:
    onenezicTT
"""

import struct
from dataclasses import dataclass

from triage import ScriptConfig, hex_serializer, triage_field, run_script, log_check
from run_checks import run as get_run_checks
from dispatcher_data import run as get_dispatcher_data
from elfs_cache import run as get_elfs_cache
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.context import Context
from ttexalens.memory_access import RiscDebugMemoryAccess
from ttexalens.tt_exalens_lib import read_word_from_device, read_from_device

script_config = ScriptConfig(
    depends=["run_checks", "dispatcher_data", "elfs_cache"],
)

# CB live tile counts live in the tensix overlay (stream) scratch registers on WH & BH.
# STREAM_REG_ADDR(stream_id, reg) = base + stream_id*space + (reg<<2); stream_id == cb index.
STREAM_REG_BASE = 0xFFB40000
STREAM_REG_SPACE = 0x1000
# (tiles_received_reg, tiles_acked_reg) per arch (BUF_SIZE_REG_INDEX, BUF_START_REG_INDEX).
STREAM_TILE_REGS = {
    "wormhole": (4, 3),
    "blackhole": (10, 8),
}

CB_ENTRY_SIZE = 32  # sizeof(CBInterface)
TRISC_ADDR_SHIFT = 4  # CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT: compute stores addrs in 16B words
L1_ALIGNMENT = 16
# brisc = DM reader/writer, trisc0 = unpacker (consumer), trisc2 = packer (producer).
CB_RISCS = ("brisc", "trisc0", "trisc2")
NO_REMOTE = 1 << 30  # sentinel: nothing is a remote CB


@dataclass
class CircularBufferRow:
    # Dev + Loc are injected by run_per_block_check's result wrapper.
    risc: str = triage_field("RISC")
    cb: int = triage_field("CB")
    kind: str = triage_field("Kind")
    state: str = triage_field("State")
    fifo_addr: int = triage_field("fifo_addr", hex_serializer)
    fifo_size: int = triage_field("fifo_size")
    page_size: int = triage_field("page_size")
    num_pages: int | None = triage_field("pages")
    rd_ptr: int = triage_field("rd_ptr", hex_serializer)
    wr_ptr: int = triage_field("wr_ptr", hex_serializer)
    tiles_received: int | None = triage_field("recv")
    tiles_acked: int | None = triage_field("acked")
    occupancy: int | None = triage_field("occ")
    wr_tile_ptr: int | None = triage_field("wr_tile_ptr", verbose=1)
    remote_words: str | None = triage_field("remote_w5_w6_w7", verbose=1)
    data: str | None = triage_field("data", verbose=2)


def _classify(occ: int | None, capacity: int) -> str:
    if occ is None:
        return "?"
    if occ == 0:
        return "empty"
    if capacity and occ >= capacity:
        return "full"
    return "partial"


def _stream_tile_counts(location: OnChipCoordinate, cb: int, tile_regs):
    """Live (received, acked) tile counts for a CB from the overlay stream registers."""
    if tile_regs is None:
        return None, None
    recv_reg, acked_reg = tile_regs
    base = STREAM_REG_BASE + cb * STREAM_REG_SPACE
    recv = read_word_from_device(location, base + (recv_reg << 2))
    acked = read_word_from_device(location, base + (acked_reg << 2))
    return recv, acked


def _fifo_data(location: OnChipCoordinate, fifo_addr: int, fifo_size: int, offset: int, nbytes: int) -> str | None:
    if offset < 0 or (fifo_size and offset >= fifo_size):
        return None
    avail = (fifo_size - offset) if fifo_size else nbytes
    n = min(nbytes, avail)
    if n <= 0:
        return None
    try:
        return bytes(read_from_device(location, fifo_addr + offset, num_bytes=n)).hex()
    except Exception:
        return None


def _decode_local(
    entry: bytes, cb: int, risc: str, location, tiles, dump_data: bool, cb_offset: int, cb_bytes: int
) -> CircularBufferRow | None:
    fifo_size, fifo_limit, fifo_page_size, _num_pages, fifo_rd_ptr, fifo_wr_ptr, _tiles_init, wr_tile_ptr = (
        struct.unpack("<8I", entry)
    )
    if fifo_size == 0:
        return None  # not configured on this RISC
    shift = TRISC_ADDR_SHIFT if risc.lower().startswith("trisc") else 0

    def to_bytes(x: int) -> int:
        return x << shift

    fifo_addr = to_bytes(fifo_limit - fifo_size)
    fifo_size_b = to_bytes(fifo_size)
    page_size_b = to_bytes(fifo_page_size)
    capacity = fifo_size_b // page_size_b if page_size_b else 0
    recv, acked = tiles
    occ = (recv - acked) if (recv is not None and acked is not None) else None
    return CircularBufferRow(
        risc=risc,
        cb=cb,
        kind="local",
        state=_classify(occ, capacity),
        fifo_addr=fifo_addr,
        fifo_size=fifo_size_b,
        page_size=page_size_b,
        num_pages=_num_pages,  # 0 on read-only RISCs (init only sets it for writers)
        rd_ptr=to_bytes(fifo_rd_ptr),
        wr_ptr=to_bytes(fifo_wr_ptr),
        tiles_received=recv,
        tiles_acked=acked,
        occupancy=occ,
        wr_tile_ptr=wr_tile_ptr,
        remote_words=None,
        data=_fifo_data(location, fifo_addr, fifo_size_b, cb_offset, cb_bytes) if dump_data else None,
    )


def _decode_remote(
    entry: bytes, cb: int, risc: str, location, dump_data: bool, cb_offset: int, cb_bytes: int
) -> CircularBufferRow | None:
    # Shared prefix of RemoteSender/RemoteReceiverCBInterface; word4 is fifo_wr_ptr (sender) or
    # fifo_rd_ptr (receiver). Word5 distinguishes them: on a receiver it packs the two small
    # sender NoC coords; on a sender it is the receiver_noc_xy_ptr (an L1 pointer).
    config_ptr, fifo_start_addr, fifo_limit_pa, fifo_page_size, ptr4, w5, w6, w7 = struct.unpack("<8I", entry)
    if config_ptr == 0 and fifo_start_addr == 0:
        return None
    capacity = (
        (fifo_limit_pa - fifo_start_addr) // fifo_page_size
        if (fifo_page_size and fifo_limit_pa >= fifo_start_addr)
        else 0
    )

    pages_sent = pages_acked = None
    w5_lo, w5_hi = w5 & 0xFFFF, w5 >> 16
    if w5 != 0 and w5_lo < 0x100 and w5_hi < 0x100:
        # receiver: pages_acked=*aligned_pages_acked_ptr(w6), pages_sent=*(w6 - L1_ALIGNMENT)
        kind = "remote-recv"
        try:
            pages_acked = read_word_from_device(location, w6)
            pages_sent = read_word_from_device(location, w6 - L1_ALIGNMENT)
        except Exception:
            pass
    elif w5 >= 0x1000:
        # sender (per-spec, not HW-validated here): receiver-0 pair at aligned_pages_sent_ptr(w6):
        # pages_sent @ +0, pages_acked @ +L1_ALIGNMENT (per-receiver stride REMOTE_CB_LOCAL_PAGES_STRIDE).
        kind = "remote-send"
        try:
            pages_sent = read_word_from_device(location, w6)
            pages_acked = read_word_from_device(location, w6 + L1_ALIGNMENT)
        except Exception:
            pass
    else:
        kind = "remote?"

    # Remote flow-control counters are in REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE (== L1_ALIGNMENT)
    # units; normalize to CB pages so recv/acked/occ match the local rows.
    per_page = (fifo_page_size // L1_ALIGNMENT) if (fifo_page_size and fifo_page_size % L1_ALIGNMENT == 0) else 1
    if per_page > 1:
        if pages_sent is not None:
            pages_sent //= per_page
        if pages_acked is not None:
            pages_acked //= per_page

    occ = (pages_sent - pages_acked) if (pages_sent is not None and pages_acked is not None) else None
    fifo_size = (fifo_limit_pa - fifo_start_addr) if fifo_limit_pa >= fifo_start_addr else 0
    return CircularBufferRow(
        risc=risc,
        cb=cb,
        kind=kind,
        state=_classify(occ, capacity),
        fifo_addr=fifo_start_addr,
        fifo_size=fifo_size,
        page_size=fifo_page_size,
        num_pages=capacity or None,
        rd_ptr=ptr4,
        wr_ptr=ptr4,
        tiles_received=pages_sent,
        tiles_acked=pages_acked,
        occupancy=occ,
        wr_tile_ptr=None,
        remote_words=f"{w5:#x} {w6:#x} {w7:#x}",
        data=_fifo_data(location, fifo_start_addr, fifo_size, cb_offset, cb_bytes) if dump_data else None,
    )


def _read_min_remote(location: OnChipCoordinate, dispatcher_data) -> int:
    """First CB index that is a remote (global) CB, from this core's launch message."""
    try:
        mailboxes = dispatcher_data.read_mailboxes(location)
        rd = int(mailboxes.launch_msg_rd_ptr)
        return int(mailboxes.launch[rd].kernel_config.min_remote_cb_start_index)
    except Exception:
        return NO_REMOTE


def read_core(
    location: OnChipCoordinate, dispatcher_data, elfs_cache, dump_data: bool, cb_offset: int, cb_bytes: int
) -> list[CircularBufferRow] | None:
    device = location.device
    arch = "wormhole" if device.is_wormhole() else ("blackhole" if device.is_blackhole() else None)
    tile_regs = STREAM_TILE_REGS.get(arch) if arch else None
    noc_block = location.noc_block

    tile_cache: dict[int, tuple] = {}

    def tiles_for(cb: int):
        if cb not in tile_cache:
            tile_cache[cb] = _stream_tile_counts(location, cb, tile_regs)
        return tile_cache[cb]

    min_remote = _read_min_remote(location, dispatcher_data)

    rows: list[CircularBufferRow] = []
    for risc_name in noc_block.risc_names:
        if risc_name.lower() not in CB_RISCS:
            continue
        core_data = dispatcher_data.get_cached_core_data(location, risc_name)
        if core_data.kernel_path is None:
            continue  # no kernel loaded on this RISC
        kernel_elf = elfs_cache[core_data.kernel_path]
        try:
            sym = kernel_elf.find_symbol_by_name("cb_interface")
        except Exception:
            sym = None
        if sym is None or sym.size < CB_ENTRY_SIZE:
            continue
        num = sym.size // CB_ENTRY_SIZE
        mem = RiscDebugMemoryAccess(noc_block.get_risc_debug(risc_name), ensure_halted_access=False)
        buf = bytearray(sym.size)
        try:
            mem.read(sym.value, buf)
        except Exception as e:
            log_check(False, f"Failed to read cb_interface at {location} {risc_name}: {e}")
            continue
        for cb in range(num):
            entry = bytes(buf[cb * CB_ENTRY_SIZE : (cb + 1) * CB_ENTRY_SIZE])
            if cb >= min_remote:
                row = _decode_remote(entry, cb, risc_name, location, dump_data, cb_offset, cb_bytes)
            else:
                row = _decode_local(entry, cb, risc_name, location, tiles_for(cb), dump_data, cb_offset, cb_bytes)
            if row is not None:
                rows.append(row)
    return rows or None


def run(args, context: Context):
    from triage import set_verbose_level

    verbose_level = args["-v"] or 0
    set_verbose_level(verbose_level)
    run_checks = get_run_checks(args, context)
    dispatcher_data = get_dispatcher_data(args, context)
    elfs_cache = get_elfs_cache(args, context)

    # CB fifo L1 data is a -vv column; only pay the extra L1 reads when it will be shown.
    dump_data = verbose_level >= 2
    cb_bytes = int(args["--cb-bytes"])
    cb_offset = int(args["--cb-offset"])

    return run_checks.run_per_block_check(
        lambda location: read_core(location, dispatcher_data, elfs_cache, dump_data, cb_offset, cb_bytes),
        block_filter=["tensix"],
    )


if __name__ == "__main__":
    run_script()
