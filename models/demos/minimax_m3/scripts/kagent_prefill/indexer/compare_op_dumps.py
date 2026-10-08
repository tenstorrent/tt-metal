# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bit-exact compare of kagent_indexer_op_bench.py --dump files: block scores (bf16, valid columns) + top-16 ids.

    python compare_op_dumps.py <a.pt> <b.pt> [<c.pt> ...]   # every file vs the first
"""
import sys

import torch


def main(paths):
    ref = torch.load(paths[0])
    ok = True
    for p in paths[1:]:
        d = torch.load(p)
        same_s = torch.equal(ref["scores"].view(torch.int16), d["scores"].view(torch.int16))
        same_i = torch.equal(ref["ids"], d["ids"])
        diff = int((ref["scores"].view(torch.int16) != d["scores"].view(torch.int16)).sum())
        print(f"OPDUMP {paths[0]} vs {p}: scores_bitexact={same_s} ({diff} differing) ids_equal={same_i}")
        ok &= same_s and same_i
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
