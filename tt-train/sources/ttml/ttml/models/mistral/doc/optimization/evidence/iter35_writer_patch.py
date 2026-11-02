# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Iteration 35: ttnn matmul writer kernels (Metal 2.0 forks bound by the mcast 1D / 2D factories) keep an output
block's DRAM writes in flight under the next block's K loop: the writes of a block are issued back to back from a
fixed read pointer (offset = tiles already issued, the output buffer holds a whole block: out_block_tiles, no double
buffer) with no barrier, and the barrier + pop happen right before the next block's drain (or at the kernel end).
Before, every output subblock ended with a write barrier + pop, so the whole block's write-back (DRAM-write bound,
~210 GB/s) sat exposed after the last K block. The padded subblocks the compute pushes are counted into the same
pop. Applied once to the two kernels below; kept as evidence.

    python iter35_writer_patch.py <tt-metal root>
"""

import re
import sys

root = sys.argv[1]
D = root + "/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
FILES = [
    D + "reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp",
    D + "reader_bmm_tile_layout_in1_receiver_writer_padding_metal2.cpp",
]


def sub1(src, pattern, repl, flags=re.M):
    new, n = re.subn(pattern, repl, src, flags=flags)
    assert n == 1, (pattern, n)
    return new


for path in FILES:
    src = open(path).read()
    assert "out_tiles_pending" not in src, path + " already patched"
    # 1. a counter of output tiles waited on and issued to the NoC but not yet popped (declared once); the previous
    #    block's writes are completed and popped only when the next block's drain starts
    src = sub1(
        src,
        r"^( *)for \(uint32_t b = 0; b < batch; \+\+b\) \{\n",
        r"\1// Output tiles issued to the NoC but not yet completed and popped. A block's writes stay in flight under\n"
        r"\1// the next block's K loop (the output buffer holds a whole block, so the next block's packs need the\n"
        r"\1// space only at its own drain); a barrier + pop per subblock left the write-back (DRAM-write bound)\n"
        r"\1// exposed after the last K block.\n"
        r"\1uint32_t out_tiles_pending = 0;\n"
        r"\1for (uint32_t b = 0; b < batch; ++b) {\n",
    )
    src = sub1(
        src,
        r"^( *)uint32_t out_tensor_sbh_start_tile_id = out_tensor_current_w_dim_block_tile_id;\n",
        r"\1if (out_tiles_pending > 0) {  // the previous block's writes\n"
        r"\1    noc.async_write_barrier();\n"
        r"\1    dfb_out.pop_front(static_cast<uint16_t>(out_tiles_pending));\n"
        r"\1    out_tiles_pending = 0;\n"
        r"\1}\n"
        r"\1uint32_t out_tensor_sbh_start_tile_id = out_tensor_current_w_dim_block_tile_id;\n",
    )
    # 2. cumulative wait, read offset past the tiles already issued
    src = sub1(
        src,
        r"^( *)dfb_out\.wait_front\(out_subblock_tile_count\);\n *uint32_t out_read_offset = 0;\n",
        r"\1dfb_out.wait_front(static_cast<uint16_t>(out_tiles_pending + out_subblock_tile_count));\n"
        r"\1uint32_t out_read_offset = out_tiles_pending * output_single_tile_size_bytes;\n",
    )
    # 3. no barrier / pop per subblock
    src = sub1(
        src,
        r"^( *)noc\.async_write_barrier\(\);\n\n? *dfb_out\.pop_front\(out_subblock_tile_count\);\n",
        r"\1out_tiles_pending += out_subblock_tile_count;\n",
    )
    # 4. padded subblocks along the row: counted, popped with the block
    src = sub1(
        src,
        r"^( *)dfb_out\.wait_front\(static_cast<uint16_t>\(padded_block_tiles_w_skip\)\);\n *dfb_out\.pop_front\(static_cast<uint16_t>\(padded_block_tiles_w_skip\)\);\n",
        r"\1out_tiles_pending += padded_block_tiles_w_skip;\n"
        r"\1dfb_out.wait_front(static_cast<uint16_t>(out_tiles_pending));\n",
    )
    # 5. padded rows: counted and waited on (the compute pushes them), popped with the block later
    src = sub1(
        src,
        r"^( *)dfb_out\.wait_front\(static_cast<uint16_t>\(padded_block_tiles_h_skip\)\);\n *dfb_out\.pop_front\(static_cast<uint16_t>\(padded_block_tiles_h_skip\)\);\n",
        r"\1out_tiles_pending += padded_block_tiles_h_skip;\n"
        r"\1dfb_out.wait_front(static_cast<uint16_t>(out_tiles_pending));\n",
    )
    # 6. the last block's writes: completed and popped after the batch loop
    src = sub1(
        src,
        r"^( *)\}\n\n(#ifdef OUT_SHARDED\n)",
        r"\1}\n"
        r"\1if (out_tiles_pending > 0) {\n"
        r"\1    noc.async_write_barrier();\n"
        r"\1    dfb_out.pop_front(static_cast<uint16_t>(out_tiles_pending));\n"
        r"\1}\n\n\2",
    )
    open(path, "w").write(src)
    print("patched", path)
