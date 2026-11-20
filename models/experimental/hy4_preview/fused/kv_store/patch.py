# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The frame-table patch of the three kernels that address the KV storage, as data: for each file of the
sparse-MLA and scan pieces a list of (old text, new text). ``PATCH.md`` explains every replacement; this module
keeps them exact: ``patched_sources()`` applies them to the owners' CURRENT files (each old text must occur exactly
once) and writes patched COPIES under ``kv_store/patched/`` for ``tests/test_patch.py``; the owners' files are not
touched. No ttnn.
"""

from __future__ import annotations

from pathlib import Path

FUSED = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "patched"

_INCLUDE = ('#include "sm_common.hpp"\n', '#include "sm_common.hpp"\n#include "kv_frames.hpp"\n')
_NAMED = (
    """    constexpr uint32_t ROWS_PER_SLOT = get_named_compile_time_arg_val("rows_per_slot");
    constexpr uint32_t SLOT_STRIDE = get_named_compile_time_arg_val("slot_stride");
    constexpr uint32_t BANKS = get_named_compile_time_arg_val("banks");
    constexpr uint32_t SHARD_ROWS = get_named_compile_time_arg_val("shard_rows");
""",
    """    constexpr uint32_t cb_frames = get_named_compile_time_arg_val("cb_frames");    // the frame tables (L1 tensor)
    constexpr uint32_t SLOT_BYTES = get_named_compile_time_arg_val("slot_bytes");  // bank bytes per layer step
    static_assert(get_named_compile_time_arg_val("banks") == 8 && get_named_compile_time_arg_val("shard_rows") == 32);
""",
)
_TABLES = """    volatile tt_l1_ptr uint32_t* frames = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_frames));
    const uint32_t layer_bytes = layer * SLOT_BYTES;
"""

PATCHES = {
    "sparse_mla/kernels/sm_reader.cpp": [
        _INCLUDE,
        _NAMED,
        (
            "    const uint32_t layer = get_common_arg_val<uint32_t>(2);\n",
            "    const uint32_t layer = get_common_arg_val<uint32_t>(2);\n" + _TABLES,
        ),
        (
            "        const uint32_t page_base = (u * SLOT_STRIDE + layer) * ROWS_PER_SLOT;\n",
            "        volatile tt_l1_ptr uint32_t* table = frames + u * kv_frames::TABLE_FRAMES;\n",
        ),
        (
            """                const uint32_t page = page_base + idx[first + r];
                const uint64_t src = get_noc_addr_from_bank_id<true>(
                    sm::kv_row_bank<BANKS, SHARD_ROWS>(page),
                    kv_addr + sm::kv_row_offset<BANKS, SHARD_ROWS>(page, STRIDE));
""",
            """                const kv_frames::Loc at = kv_frames::locate(table, layer_bytes, idx[first + r], STRIDE);
                const uint64_t src = get_noc_addr_from_bank_id<true>(at.bank, kv_addr + at.offset);
""",
        ),
    ],
    "sparse_mla/kernels/sm_append.cpp": [
        _INCLUDE,
        _NAMED,
        (
            "    const uint32_t layer = get_common_arg_val<uint32_t>(3);\n",
            "    const uint32_t layer = get_common_arg_val<uint32_t>(3);\n" + _TABLES,
        ),
        ("    uint32_t page_of[sm::MAX_USERS];\n", "    kv_frames::Loc at_of[sm::MAX_USERS];\n"),
        (
            """        if ((position / BLOCK) % CHIPS != chip || row >= ROWS_PER_SLOT) {
            continue;
        }
        noc_async_read(rows.get_noc_addr(u), scratch + mine * STRIDE, ROW_BYTES);
        page_of[mine++] = (u * SLOT_STRIDE + layer) * ROWS_PER_SLOT + row;
""",
            """        volatile tt_l1_ptr uint32_t* table = frames + u * kv_frames::TABLE_FRAMES;
        if ((position / BLOCK) % CHIPS != chip || !kv_frames::mapped(table, row)) {
            continue;  // another chip's position, or beyond the conversation's frames
        }
        noc_async_read(rows.get_noc_addr(u), scratch + mine * STRIDE, ROW_BYTES);
        at_of[mine++] = kv_frames::locate(table, layer_bytes, row, STRIDE);
""",
        ),
        (
            """        const uint64_t dst = get_noc_addr_from_bank_id<true>(
            sm::kv_row_bank<BANKS, SHARD_ROWS>(page_of[i]),
            kv_addr + sm::kv_row_offset<BANKS, SHARD_ROWS>(page_of[i], STRIDE));
""",
            "        const uint64_t dst = get_noc_addr_from_bank_id<true>(at_of[i].bank, kv_addr + at_of[i].offset);\n",
        ),
    ],
    "indexer_scan/kernels/scan_reader.cpp": [
        ('#include "scan_common.hpp"\n', '#include "scan_common.hpp"\n#include "kv_frames.hpp"\n'),
        (
            "constexpr uint32_t k_capacity_blocks = get_compile_time_arg_val(9);  // key blocks cb_k holds\n",
            "constexpr uint32_t k_capacity_blocks = get_compile_time_arg_val(9);  // key blocks cb_k holds\n"
            "constexpr uint32_t cb_frames = 7;  // the frame tables of fused/kv_store (an L1 tensor); free in scan_common.hpp\n"
            'static_assert(stream_depth > 0, "the paged key stream is the pipelined one");\n',
        ),
        (
            "    const uint32_t slot_offset = get_arg_val<uint32_t>(2);  // bytes from the buffer address to this slot, in a bank\n",
            "    const uint32_t slot_offset = get_arg_val<uint32_t>(2);  // this layer's slot step in a bank (Geometry.slot_bytes(rank))\n",
        ),
        (
            "    const uint32_t stream_vc = get_arg_val<uint32_t>(8);  // response virtual channel of the pipelined key stream\n",
            "    const uint32_t stream_vc = get_arg_val<uint32_t>(8);  // response virtual channel of the pipelined key stream\n"
            "    // frame bases of this core's user: block j of this bank's stream is the bank's one shard of frame j\n"
            "    volatile tt_l1_ptr uint32_t* table =\n"
            "        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_frames)) + user * kv_frames::TABLE_FRAMES;\n",
        ),
        (
            "        const uint32_t blocks = min_u32(blocks_per_read, padded_blocks - issued);\n",
            """        // a packet holds frames that lie back to back in this bank, and does not run over the end of cb_k
        uint32_t blocks = min_u32(blocks_per_read, padded_blocks - issued);
        blocks = min_u32(blocks, (k_end - k_write) / key_block_bytes);
        if (issued < num_blocks) {
            uint32_t run = 1;
            while (run < blocks && issued + run < num_blocks &&
                   table[issued + run] == table[issued] + run * key_block_bytes) {
                ++run;
            }
            blocks = run;
        }
""",
        ),
        (
            "                static_cast<uint32_t>(bank_base), issued * key_block_bytes, k_write, slot + 1);\n",
            "                static_cast<uint32_t>(bank_base), table[issued], k_write, slot + 1);\n",
        ),
    ],
}


def apply(relative: str) -> str:
    """The owner's current file with the patch applied."""
    text = (FUSED / relative).read_text()
    for old, new in PATCHES[relative]:
        assert (
            text.count(old) == 1
        ), f"{relative}: this text occurs {text.count(old)} times, the patch expects once:\n{old}"
        text = text.replace(old, new)
    return text


def patched_sources() -> dict:
    """Write the patched copies; returns {file name: path relative to the tt-metal tree}."""
    OUT.mkdir(exist_ok=True)
    out = {}
    for relative in PATCHES:
        name = Path(relative).name
        target = OUT / name
        text = apply(relative)
        if not target.is_file() or target.read_text() != text:
            target.write_text(text)
        out[name] = "models/experimental/hy4_preview/fused/kv_store/patched/" + name
    return out


def size() -> dict:
    """{file: (lines removed, lines added)}"""
    return {
        relative: (sum(old.count("\n") for old, _ in reps), sum(new.count("\n") for _, new in reps))
        for relative, reps in PATCHES.items()
    }


def scan_packets(
    table, num_blocks: int, padded_blocks: int, blocks_per_read: int, capacity_blocks: int, block_bytes: int = 4352
):
    """Host mirror of the patched ``issue_packet`` of scan_reader.cpp: the packets one (user, bank) core issues, as
    (first block, blocks pushed, blocks read, bank offset read at or None). What the compute kernel waits for is
    the sum of the blocks pushed: it must be ``padded_blocks`` whatever the table."""
    out, issued, k_write = [], 0, 0
    while issued < padded_blocks:
        blocks = min(blocks_per_read, padded_blocks - issued, capacity_blocks - k_write)
        if issued < num_blocks:
            run = 1
            while (
                run < blocks and issued + run < num_blocks and table[issued + run] == table[issued] + run * block_bytes
            ):
                run += 1
            blocks = run
        assert blocks >= 1
        read = min(blocks, num_blocks - issued) if issued < num_blocks else 0
        out.append((issued, blocks, read, table[issued] if read else None))
        k_write += blocks
        assert k_write <= capacity_blocks, "a packet runs over the end of cb_k"
        k_write %= capacity_blocks
        issued += blocks
    return out
