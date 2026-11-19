# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Select fused score reductions under the unchanged full-capacity score gate."""
import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path
from unittest.mock import patch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import probe_qsa_score_runtime
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import fused_indexer_scores
from models.autoports.qwen_qwen3_8_flash_next.tt.qsa import QSA


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--merge-heads", action="store_true")
    parser.add_argument("--accurate-sum", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    for name in ("--merge-heads", "--accurate-sum"):
        if name in sys.argv:
            sys.argv.remove(name)
    calls = []

    def selected(owner, query, keys):
        owner.compute = ttnn.init_device_compute_kernel_config(
            query.device().arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            math_approx_mode=False,
            packer_l1_acc=False,
        )
        calls.append(query.shape[-2])
        return fused_indexer_scores(owner, query, keys, merge_heads=args.merge_heads, accurate_sum=args.accurate_sum)

    try:
        with patch.object(QSA, "indexer_scores", selected):
            probe_qsa_score_runtime.main()
        assert calls
    finally:
        if args.output.exists():
            report = json.loads(args.output.read_text())
            report["fused_score_selection"] = dict(
                merge_heads=args.merge_heads,
                accurate_sum=args.accurate_sum,
                logical_call_counts=calls,
                source_sha256=hashlib.sha256(Path(inspect.getfile(fused_indexer_scores)).read_bytes()).hexdigest(),
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
