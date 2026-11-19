# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Run the unchanged numerical harness and observe first-prefill row dispatch.

Only tensor shape metadata is inspected. No tensor conversion or operation is
added; native-call interception distinguishes executed fusion from a predicted
branch. These watcher runs are not used for performance claims.
"""

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.run_multichip_decoder import main
from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import MultichipDecoder


def run():
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    linear = MultichipDecoder._linear
    native = ttnn.experimental.matmul_reduce_scatter_prefill
    records, seen = [], set()
    active = None

    def observed_native(value, *args, **kwargs):
        if active is not None:
            active.append(list(value.shape))
        return native(value, *args, **kwargs)

    def observed_linear(model, value, key):
        nonlocal active
        if key not in model.row_parallel_weights or key in seen:
            return linear(model, value, key)
        seen.add(key)
        pieces = []
        active = pieces
        try:
            result = linear(model, value, key)
        finally:
            active = None
        records.append(
            {
                "weight": key,
                "input_shape": list(value.shape),
                "flattened_rows": value.shape[0] * value.shape[1],
                "output_shape": list(result.shape),
                "native_piece_shapes": pieces,
                "path": "native_mmrs" if pieces else "separate_matmul",
            }
        )
        return result

    with patch.object(MultichipDecoder, "_linear", observed_linear), patch.object(
        ttnn.experimental, "matmul_reduce_scatter_prefill", observed_native
    ):
        try:
            main()
        finally:
            if output.exists():
                report = json.loads(output.read_text())
                report["prefill_row_dispatch"] = records
                report["dispatch_observer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    run()
