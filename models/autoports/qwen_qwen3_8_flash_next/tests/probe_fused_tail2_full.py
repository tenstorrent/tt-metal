# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-decoder experiment for the already-exact two-row compact MoE union.

The production runtime remains unchanged. Only best_tail2 substitutes the
experimental two-row MoE implementation; every other logical count dispatches
the selected production FusedSparseMoe. The shared benchmark preserves its
functional reference, state checks, trace repetitions and timing regime.
"""

import hashlib
import inspect
import json
import sys
from pathlib import Path

from models.autoports.qwen_qwen3_8_flash_next.tests import probe_fused_candidates as benchmark
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_fused_prefill_compact import UnionMoe
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import FusedDecoder, FusedSparseMoe


class Tail2Moe(FusedSparseMoe):
    def __call__(self, x):
        if x.shape[-2] == 2:
            return UnionMoe.__call__(self, x)
        return super().__call__(x)


class CandidateFactory(FusedDecoder):
    @classmethod
    def from_state_dict(cls, *args, experimental_tail2=False, **kwargs):
        model = super().from_state_dict(*args, **kwargs)
        if experimental_tail2:
            model.mlp.__class__ = Tail2Moe
        return model


def main():
    benchmark.FusedDecoder = CandidateFactory
    benchmark.POLICIES = dict(benchmark.POLICIES)
    benchmark.POLICIES["best_tail2"] = dict(benchmark.POLICIES["best_exact_qkv_cleanup"], experimental_tail2=True)
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    try:
        benchmark.main()
    finally:
        if output.exists():
            report = json.loads(output.read_text())
            report["experimental_tail2"] = {
                "scope": "Only logical two-row MoE calls use UnionMoe; other counts use production FusedSparseMoe",
                "source_sha256": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (Path(__file__), Path(inspect.getfile(UnionMoe)))
                },
            }
            output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
