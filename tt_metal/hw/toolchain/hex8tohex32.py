# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3

import itertools
import sys


def write_data(outf, data, ptr):
    if len(data) != 0:
        outf.write("@%08x\n" % (ptr >> 2))
        while len(data) % 4 != 0:
            data.append(0)
        for word_bytes in zip(*([iter(data)] * 4)):
            outf.write("".join(["%02x" % b for b in reversed(word_bytes)]) + "\n")


assert len(sys.argv) > 2
in_lines = open(sys.argv[1]).readlines()
data = []
ptr = 0
outf = open(sys.argv[2], "wt")
for line in in_lines:
    if line.startswith("@"):
        addr = int(line[1:], 16)
        if addr > ptr + 4:
            write_data(outf, data, ptr)
            ptr = addr
            data = []
            while ptr % 4 != 0:
                data.append(0)
                ptr -= 1
        else:
            while ptr + len(data) < addr:
                data.append(0)
    else:
        data += [int(tok, 16) for tok in line.split()]

write_data(outf, data, ptr)
