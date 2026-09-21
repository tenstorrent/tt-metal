# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Keep the top 890 PNG rows without resampling or changing their pixels."""

import struct
import zlib
from pathlib import Path

root = Path(__file__).resolve().parent
source = (root / "sdpa-ring-pareto.svg.png").read_bytes()
assert source[:8] == b"\x89PNG\r\n\x1a\n"
position = 8
compressed = []
while position < len(source):
    length = struct.unpack(">I", source[position : position + 4])[0]
    kind = source[position + 4 : position + 8]
    data = source[position + 8 : position + 8 + length]
    if kind == b"IHDR":
        header = data
    elif kind == b"IDAT":
        compressed.append(data)
    position += length + 12
width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", header)
assert (width, height, depth, interlace) == (1440, 1440, 8, 0)
assert color in (2, 6)
channels = 3 if color == 2 else 4
rows = zlib.decompress(b"".join(compressed))
assert len(rows) == height * (1 + channels * width)


def chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


new_header = struct.pack(">II", width, 890) + header[8:]
result = source[:8] + chunk(b"IHDR", new_header)
result += chunk(b"IDAT", zlib.compress(rows[: 890 * (1 + channels * width)]))
result += chunk(b"IEND", b"")
(root / "sdpa-ring-pareto.png").write_bytes(result)
