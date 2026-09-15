"""Trim blank space below the verified plot without changing its pixels."""

import struct
import zlib
from pathlib import Path

root = Path(__file__).parent
source = (root / "light-736.png").read_bytes()
assert source[:8] == b"\x89PNG\r\n\x1a\n"
pos = 8
idat = []
while pos < len(source):
    length = struct.unpack(">I", source[pos : pos + 4])[0]
    kind = source[pos + 4 : pos + 8]
    data = source[pos + 8 : pos + 8 + length]
    if kind == b"IHDR":
        header = data
    elif kind == b"IDAT":
        idat.append(data)
    pos += 12 + length
width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", header)
assert depth == 8 and color in (2, 6) and interlace == 0
channels = 3 if color == 2 else 4
rows = zlib.decompress(b"".join(idat))
assert len(rows) == height * (1 + channels * width)
new_height = 740


def chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


new_header = struct.pack(">II", width, new_height) + header[8:]
result = source[:8] + chunk(b"IHDR", new_header)
result += chunk(b"IDAT", zlib.compress(rows[: new_height * (1 + channels * width)]))
result += chunk(b"IEND", b"")
(root / "sdpa-pareto-static.png").write_bytes(result)
