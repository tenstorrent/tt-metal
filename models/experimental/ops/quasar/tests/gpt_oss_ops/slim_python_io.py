# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Shrink a ``graph_capture.python_io.json`` to the fields the generator reads.

    python models/experimental/ops/quasar/tests/gpt_oss_ops/slim_python_io.py \
        generated/ttnn/reports/<report>/graph_capture.python_io.json \
        generated/ttnn/reports/<report>/graph_capture.python_io.slim.json

Why this exists: a full Qwen3-VL demo run (200 generated tokens x 2 repeat batches x
36 decoder layers) records ~750k python-level calls, and every record carries its own
``captured_graph`` — the C++ node list for that one op. That field is ~97% of the
bytes and ``generate_from_graph_capture.py`` never reads it (it uses ``name``,
``arguments``, ``input_tensor_ids``, ``output_tensor_ids``). Dropping it took the
Qwen3-VL capture from 5.0 GB to 1.1 GB, which is what makes the generator's
read-it-all approach viable here.

Every record is kept, so call counts in the generated cases stay exact.

Streaming: the input is decoded one record at a time over a sliding buffer, so peak
memory is the chunk size, not the file size. The llama suite's captures are ~100 MB
and skip this step entirely -- the generator reads those directly.
"""

import json
import sys

# Fields generate_from_graph_capture.py reads (build_cases / iter_records).
KEEP = ("name", "arguments", "input_tensor_ids", "output_tensor_ids")
CHUNK = 64 << 20


def stream_records(path):
    """Yield each record of the top-level JSON array without holding the whole file.

    Every way of running out of input is an error, never a quiet stop: a capture that
    was truncated mid-write would otherwise slim down to a syntactically valid array of
    fewer records, and the suite generated from it would silently be missing model calls
    with no sign that anything went wrong.
    """
    decoder = json.JSONDecoder()
    buf = ""
    with open(path) as f:
        while "[" not in buf:
            chunk = f.read(1 << 16)
            if not chunk:
                raise ValueError(f"{path}: no JSON array found; not a python_io capture?")
            buf += chunk
        i = buf.index("[") + 1
        while True:
            while True:
                while i < len(buf) and buf[i] in " \n\r\t,":
                    i += 1
                if i < len(buf) and buf[i] == "]":
                    return
                try:
                    record, i = decoder.raw_decode(buf, i)
                    break
                except ValueError as exc:  # incomplete record: more input, or a bad file
                    more = f.read(CHUNK)
                    if not more:
                        raise ValueError(f"{path}: input ended inside a record (truncated capture): {exc}") from exc
                    buf += more
            yield record
            if i > CHUNK:  # drop the consumed prefix so the buffer stays bounded
                buf = buf[i:]
                i = 0


def main(src, dst):
    written = 0
    with open(dst, "w") as out:
        out.write("[")
        for record in stream_records(src):
            if written:
                out.write(",\n")
            out.write(json.dumps({k: record.get(k) for k in KEEP}))
            written += 1
            if written % 100000 == 0:
                print(f"  {written} records", flush=True)
        out.write("]")
    if not written:
        raise ValueError(f"{src}: no records; an empty capture would generate an empty suite")
    print(f"wrote {written} records -> {dst}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
