# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Drop-in replacements for the ttexalens transfer helpers that split large
requests into small ones.

ttexalens only takes its DMA path when ``can_use_dma`` is true, which excludes
Blackhole, the simulator, and non-MMIO devices. Without DMA the request goes to
UMD's register access, which has a hard cliff: anything up to ~2112 bytes is
moved in bulk, anything larger degrades to one 4-byte word at a time. Measured
on Blackhole, a 128 KB result buffer takes 94 ms to read as a single request but
3.4 ms split into 2 KB requests; the same write goes from 47 ms to 1.5 ms.

Splitting is skipped where DMA is available, since there a large transfer is a
single descriptor and chunking would only multiply the setup cost.

These wrappers mirror the signatures of the ttexalens functions they replace, so
call sites only need to change their import.
"""

import struct
from functools import lru_cache

from ttexalens.tt_exalens_lib import convert_coordinate
from ttexalens.tt_exalens_lib import read_from_device as _read_from_device
from ttexalens.tt_exalens_lib import read_words_from_device as _read_words_from_device
from ttexalens.tt_exalens_lib import write_to_device as _write_to_device
from ttexalens.tt_exalens_lib import write_words_to_device as _write_words_to_device

# Comfortably below the measured ~2112 byte cliff, and a natural alignment.
MAX_TRANSFER_BYTES = 2048


@lru_cache(maxsize=None)
def _splitting_helps(device_id: int) -> bool:
    from ttexalens import check_context

    try:
        return not check_context().devices[device_id]._umd_device.can_use_dma
    except Exception:
        # Rather than guess at a ttexalens internal that moved, keep splitting:
        # it is a large win without DMA and only a small loss with it.
        return True


def read_from_device(
    location,
    addr,
    device_id=0,
    num_bytes=4,
    context=None,
    noc_id=None,
    use_4B_mode=None,
    safe_mode=None,
):
    if num_bytes <= MAX_TRANSFER_BYTES or not _splitting_helps(device_id):
        return _read_from_device(
            location,
            addr,
            device_id,
            num_bytes,
            context,
            noc_id,
            use_4B_mode,
            safe_mode,
        )

    # Resolve once so each chunk skips re-parsing the "x,y" location string.
    coordinate = convert_coordinate(location, device_id, context)
    return b"".join(
        _read_from_device(
            coordinate,
            addr + offset,
            device_id,
            min(MAX_TRANSFER_BYTES, num_bytes - offset),
            context,
            noc_id,
            use_4B_mode,
            safe_mode,
        )
        for offset in range(0, num_bytes, MAX_TRANSFER_BYTES)
    )


def write_to_device(
    location,
    addr,
    data,
    device_id=0,
    context=None,
    noc_id=None,
    use_4B_mode=None,
    safe_mode=None,
):
    if isinstance(data, list):
        data = bytes(data)

    if len(data) <= MAX_TRANSFER_BYTES or not _splitting_helps(device_id):
        return _write_to_device(
            location, addr, data, device_id, context, noc_id, use_4B_mode, safe_mode
        )

    coordinate = convert_coordinate(location, device_id, context)
    for offset in range(0, len(data), MAX_TRANSFER_BYTES):
        _write_to_device(
            coordinate,
            addr + offset,
            data[offset : offset + MAX_TRANSFER_BYTES],
            device_id,
            context,
            noc_id,
            use_4B_mode,
            safe_mode,
        )


def read_words_from_device(
    location,
    addr,
    device_id=0,
    word_count=1,
    context=None,
    noc_id=None,
    use_4B_mode=None,
    safe_mode=None,
):
    num_bytes = 4 * word_count
    if num_bytes <= MAX_TRANSFER_BYTES or not _splitting_helps(device_id):
        return _read_words_from_device(
            location,
            addr,
            device_id,
            word_count,
            context,
            noc_id,
            use_4B_mode,
            safe_mode,
        )

    raw = read_from_device(
        location, addr, device_id, num_bytes, context, noc_id, use_4B_mode, safe_mode
    )
    return list(struct.unpack(f"<{word_count}I", raw))


def write_words_to_device(
    location,
    addr,
    data,
    device_id=0,
    context=None,
    noc_id=None,
    use_4B_mode=None,
    safe_mode=None,
):
    if isinstance(data, int) or len(data) * 4 <= MAX_TRANSFER_BYTES:
        return _write_words_to_device(
            location, addr, data, device_id, context, noc_id, use_4B_mode, safe_mode
        )

    return write_to_device(
        location,
        addr,
        b"".join(word.to_bytes(4, "little") for word in data),
        device_id,
        context,
        noc_id,
        use_4B_mode,
        safe_mode,
    )
