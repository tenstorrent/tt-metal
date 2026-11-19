# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Record host allocator metadata around the first public PLE chunk's expert allocations.

Reuses unchanged HF/reset/replay capacity gates. Diagnostic timing is not ranked.
No tensor is retained by instrumentation and no tensor data is read to host here.
"""
import argparse
import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import ttnn


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-output", type=Path, required=True)
    p.add_argument("arguments", nargs=argparse.REMAINDER)
    args = p.parse_args()
    original_empty, original_sparse = ttnn.empty, ttnn.sparse_matmul
    events = []
    active = False
    finished = False

    def view(device):
        info = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
        return dict(
            banks=info.num_banks,
            free=info.num_banks * info.total_bytes_free_per_bank,
            largest_per_bank=info.largest_contiguous_bytes_free_per_bank,
        )

    def event(name, device, tensor=None, requested=None):
        row = dict(event=name, memory=view(device))
        if tensor is not None:
            row.update(
                shape=list(tensor.shape),
                padded_shape=list(tensor.padded_shape),
                address=tensor.buffer_address(),
                dtype=str(tensor.dtype),
            )
        if requested is not None:
            row["requested_shape"] = list(requested)
        events.append(row)

    def empty(shape, *positional, **kwargs):
        nonlocal active
        shape = list(shape)
        if len(shape) == 4 and shape[-2] > 1000 and shape[-1] == 10240:
            active = True
        record = active and not finished and (shape[-1] == 10240 or (shape[-1] == 640 and math.prod(shape[:-2]) == 512))
        if record:
            event("empty_before", kwargs["device"], requested=shape)
        result = original_empty(shape, *positional, **kwargs)
        if record:
            event("empty_after", result.device(), result)
        return result

    def sparse(lhs, rhs, *positional, **kwargs):
        nonlocal finished
        record = active and not finished
        if record:
            event("packed_before" if rhs.shape[-1] == 1280 else "down_before", lhs.device(), lhs)
        result = original_sparse(lhs, rhs, *positional, **kwargs)
        if record:
            event("packed_after" if rhs.shape[-1] == 1280 else "down_after", result.device(), result)
            if rhs.shape[-1] == 2560:
                finished = True
        return result

    try:
        from models.autoports.qwen_qwen3_8_flash_next.tests import probe_ple_decoder

        sys.argv = [probe_ple_decoder.__file__] + (
            args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
        )
        with patch.object(ttnn, "empty", empty), patch.object(ttnn, "sparse_matmul", sparse):
            probe_ple_decoder.main()
    finally:
        args.trace_output.write_text(json.dumps(dict(events=events, first_down_completed=finished), indent=2) + "\n")


if __name__ == "__main__":
    main()
