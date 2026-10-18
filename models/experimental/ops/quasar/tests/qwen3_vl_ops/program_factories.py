# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Summarize which C++ program factory each ttnn op actually selected in a capture.

    python models/experimental/ops/quasar/tests/qwen3_vl_ops/program_factories.py \
        generated/ttnn/reports/<report>/graph_capture.json > PROGRAM_FACTORIES.md

Reads the *C++* graph (``graph_capture.json``), not the python_io sidecar: the factory
identity lives on each device op's ``function_start`` params, which the per-op case data
does not carry. That field is not recorded by the graph tracer in this tree, so this
prints an empty table for a capture taken from it.

Why it is worth having next to the generated tests: a case here reproduces one op with
one captured config, and this table says which factory that config routes to — so a
hang or a fault in ``test_linear.py::04_...`` points at a named C++ factory instead of
"one of the matmul factories". ``program_cache_hit`` additionally separates the
compile path from the cached-run path.

Streaming: the file is read in overlapping chunks and scanned with a regex, so a 4.8 GB
capture costs a few minutes and a few MB of memory. Each ``params`` object is flat, so
``[^{}]*`` cannot run past the end of one op's params into the next.
"""

import collections
import re
import sys

CHUNK = 8 << 20
OVERLAP = 1 << 13  # params objects are far smaller than this

# The capture is pretty-printed and its params keys are emitted in alphabetical order:
#   {"inputs", "name", "program_cache_hit", "program_factory_index", "program_factory_type"}
# `name` here is the device-operation class (TilizeDeviceOperation), which is what the
# factory belongs to. [^{}] keeps a match inside one params object.
_FACTORY_RE = re.compile(
    r'"name":\s*"(?P<op>[^"]+)"[^{}]*?"program_cache_hit":\s*(?P<hit>true|false)'
    r'[^{}]*?"program_factory_type":\s*"(?P<factory>[^"]+)"'
)


def scan(path):
    """(op, factory) -> [total, cache_hits], streaming over the capture."""
    counts = collections.defaultdict(lambda: [0, 0])
    tail = ""
    with open(path) as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            text = tail + chunk
            carried = len(tail)
            for m in _FACTORY_RE.finditer(text):
                # The carried tail was already scanned last round; only a match that
                # reaches into the new chunk is new. Without this every match inside the
                # overlap is counted twice, inflating calls and hit rates.
                if m.end() <= carried:
                    continue
                entry = counts[(m.group("op"), m.group("factory"))]
                entry[0] += 1
                entry[1] += m.group("hit") == "true"
            tail = text[-OVERLAP:]
    return counts


def main(path):
    counts = scan(path)
    by_op = collections.defaultdict(list)
    for (op, factory), (total, hits) in counts.items():
        by_op[op].append((total, factory, hits))

    print("# Program factories selected in the captured run\n")
    print(f"Source capture: `{path}`\n")
    print("| op | calls | program factory | cache hits |")
    print("| --- | ---: | --- | ---: |")
    for op in sorted(by_op, key=lambda o: -sum(t for t, _, _ in by_op[o])):
        for total, factory, hits in sorted(by_op[op], reverse=True):
            print(f"| `{op}` | {total} | `{factory}` | {hits} ({100 * hits / total:.0f}%) |")
    ops = len(by_op)
    print(
        f"\n{ops} op(s), {len(counts)} distinct (op, factory) pair(s), "
        f"{sum(t for _, (t, _) in counts.items())} device-op launch(es)."
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
