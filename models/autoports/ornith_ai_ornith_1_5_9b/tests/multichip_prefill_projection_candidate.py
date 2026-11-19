# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Short-prefill input-projection geometry selected by the real-weight sweep."""

import argparse
import math
import sys

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import MultichipDecoder
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_prefill_kernel_candidate import PrefillKernelCandidate


class PrefillProjectionCandidate(MultichipDecoder):
    def linear(self, x, name):
        n = x.shape[2]
        if name != "packed_input" or n == 1 or n > 256:
            return super().linear(x, name)
        weight = self.weights[name + ".weight"]
        if n <= 64:
            cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 8), in0_block_w=4,
                out_subblock_h=1, out_subblock_w=1,
                per_core_M=math.ceil(n / 32), per_core_N=math.ceil(weight.shape[-1] / (32 * 64)),
                fused_activation=None, fuse_batch=True, mcast_in0=True,
            )
        else:
            cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(11, 10), in0_block_w=16,
                out_subblock_h=1, out_subblock_w=1,
                per_core_M=math.ceil(n / (32 * 10)), per_core_N=math.ceil(weight.shape[-1] / (32 * 11)),
                fused_activation=None, transpose_mcast=False,
            )
        return ttnn.linear(
            ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG), weight,
            program_config=cfg, compute_kernel_config=self.matmul_compute(name),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


class PrefillCombinedCandidate(PrefillProjectionCandidate, PrefillKernelCandidate):
    pass


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--short-mlp", action="store_true")
    args, rest = parser.parse_known_args()
    from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_probe
    multichip_probe.MultichipDecoder = PrefillCombinedCandidate if args.short_mlp else PrefillProjectionCandidate
    sys.argv = [sys.argv[0], *rest]
    multichip_probe.main()


if __name__ == "__main__":
    main()
