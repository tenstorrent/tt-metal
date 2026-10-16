# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Patched copies of the legacy 2D-multicast matmul kernels (the bs1 QKV / WO / FF1 / FF3 / FF2 path) for ablations:
one tree per variant under <out>/<variant>/, used as TT_METAL_KERNEL_PATH=<out>/<variant> (run from a directory with no
ttnn/ tree, with its own TT_METAL_CACHE; see sdpa_kernel_variants.py). Each data transfer of the dataflow kernels is
guarded by a switch; semaphores and CB handshakes are kept, so every variant runs the full protocol.

    python mm_legacy_variants.py <out> [variant ...]      (default: all)

  base         unmodified copy (the control)
  conly        compute only: no in0 / in1 reads or multicasts, no output writes
  no_in1_read  in1 senders skip the DRAM weight reads (the multicast still sends the CB)
  no_in1       no in1 DRAM reads and no in1 multicast
  no_in0       no in0 reads (interleaved / shard extraction) and no in0 multicast
  no_out       no output writes
"""

import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
K = "ttnn/cpp/ttnn/operations/matmul/device/kernels"
DF = "dataflow/"
FLAGS = ("ABL_IN0_READ", "ABL_IN0_MCAST", "ABL_IN1_READ", "ABL_IN1_MCAST", "ABL_OUT")

# file -> [(regex for the call's opening, flag, expected count)]
SITES = {
    "reader_bmm_tile_layout_in1_sender_writer_padding.cpp": [
        (r"noc\.async_read\(\n\s+dram_src,", "ABL_IN1_READ", 1),
        (r"noc\.async_read\(\n\s+s1,", "ABL_IN1_READ", 1),
        (r"noc\.async_write_multicast\(\n\s+CoreLocalMem<uint32_t>\(static_cast<uint32_t>\(in1_start_address\)\)", "ABL_IN1_MCAST", 1),
        (r"noc\.async_write\(\n\s+dfb_out,", "ABL_OUT", 1),
    ],
    "reader_bmm_tile_layout_in1_receiver_writer_padding.cpp": [
        (r"noc\.async_write\(\n\s+dfb_out,", "ABL_OUT", 1),
    ],
    "reader_bmm_tile_layout_in0_sender_padding.cpp": [
        (r"noc\.async_read\(\n\s+s0,", "ABL_IN0_READ", 1),
        (r"noc\.async_read\(\n\s+self_ep,", "ABL_IN0_READ", 1),
        (r"noc\.async_write_multicast\(\n\s+CoreLocalMem<uint32_t>\(in0_start_address\)", "ABL_IN0_MCAST", 1),
    ],
    "reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp": [
        (r"noc\.async_read\(\n\s+self_ep,", "ABL_IN0_READ", 1),
        (r"noc\.async_write_multicast(<NocOptions::MCAST_INCL_SRC>)?\(\n\s+CoreLocalMem<uint32_t>\(in0_tensor_read_addr\)", "ABL_IN0_MCAST", 3),
        (r"noc\.async_write\(\n\s+CoreLocalMem<uint32_t>\(in0_tensor_read_addr\)", "ABL_IN0_MCAST", 1),
    ],
}  # fmt: skip


def patch(d, off):
    for f, sites in SITES.items():
        p = d / DF / f
        s = p.read_text()
        for rx, flag, n in sites:
            s, k = re.subn(rx, lambda m, flag=flag: f"if ({flag}) " + m.group(0), s)
            assert k == n, (f, rx, k)
        defs = "".join(f"#define {fl} {0 if fl in off else 1}\n" for fl in FLAGS)
        p.write_text(defs + s)


VARIANTS = {
    "base": (),
    "conly": FLAGS,
    "no_in1_read": ("ABL_IN1_READ",),
    "no_in1": ("ABL_IN1_READ", "ABL_IN1_MCAST"),
    "no_in0": ("ABL_IN0_READ", "ABL_IN0_MCAST"),
    "no_out": ("ABL_OUT",),
}

if __name__ == "__main__":
    out = Path(sys.argv[1]).resolve()
    for v in sys.argv[2:] or VARIANTS:
        d = out / v / K
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(REPO / K, d)
        patch(d, VARIANTS[v])
        print(f"{v}: {out / v}")
