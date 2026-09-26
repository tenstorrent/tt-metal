# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Verify native paged-prefill read bounds against the logical page table."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests import run_optimized_contract


def main():
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    native = ttnn.transformer.chunked_scaled_dot_product_attention
    calls = []

    def checked(q, k, v, table, **kwargs):
        program = kwargs["program_config"]
        offset = kwargs["chunk_start_idx"]
        end = offset + q.shape[-2]
        bound = end + (-end) % program.k_chunk_size
        capacity = table.shape[-1] * k.shape[-2]
        row = dict(
            offset=offset,
            query_rows=q.shape[-2],
            q_chunk=program.q_chunk_size,
            k_chunk=program.k_chunk_size,
            read_end=bound,
            cache_capacity=capacity,
            table_pages=table.shape[-1],
            passed=bound <= capacity,
        )
        calls.append(row)
        assert row["passed"], row
        return native(q, k, v, table, **kwargs)

    with patch.object(ttnn.transformer, "chunked_scaled_dot_product_attention", checked):
        run_optimized_contract.main()
    report = json.loads(output.read_text())
    report["paged_prefill_capacity_checks"] = calls
    assert calls and all(row["passed"] for row in calls)
    report["paged_prefill_capacity_passed"] = True
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
