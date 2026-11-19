# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Adapt isolated output-projection winners to whole-layer L1 residency."""
import argparse
import math
import sys
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_long_prefill_geometry as original


class WholeLayerGeometry(original.LongPrefillGeometry):
    output_k_block = 8
    output_input = "l1"

    def linear(self, x, name):
        if x.shape[2] < 512 or "output" not in self.selected or name not in ("linear_attn.out_proj", "self_attn.o_proj"):
            return super().linear(x, name)
        cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8), in0_block_w=self.output_k_block,
            out_subblock_h=1, out_subblock_w=2, per_core_M=math.ceil(x.shape[2] / 256),
            per_core_N=16, transpose_mcast=False, fused_activation=None)
        memory = ttnn.L1_MEMORY_CONFIG if self.output_input == "l1" else ttnn.DRAM_MEMORY_CONFIG
        return ttnn.linear(ttnn.to_memory_config(x, memory), self.weights[name + ".weight"],
                           program_config=cfg, compute_kernel_config=self.matmul_compute(name),
                           memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)


def main():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--output-k-block", type=int, default=8)
    ap.add_argument("--output-input", choices=["l1", "dram"], default="l1")
    args, rest = ap.parse_known_args()
    WholeLayerGeometry.output_k_block = args.output_k_block
    WholeLayerGeometry.output_input = args.output_input
    original.LongPrefillGeometry = WholeLayerGeometry
    sys.argv = [sys.argv[0], *rest]
    original.main()


if __name__ == "__main__":
    main()
