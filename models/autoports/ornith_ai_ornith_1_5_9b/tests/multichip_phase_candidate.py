# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Cross measured decode fusion with short-MLP and long-prefill precision policies."""

import argparse
import importlib
import sys

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_pre_fusion_reference import MultichipDecoder
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_local_fusion_candidate import LocalFusionCandidate
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_prefill_kernel_candidate import PrefillKernelCandidate
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_prefill_precision_candidate import PrefillPrecisionCandidate


class PhaseCandidate(LocalFusionCandidate):
    fused_conv = True
    fused_pad = True
    decode_wave_heads = 48
    gdn_depth = None

    def norm(self, x, name):
        result = MultichipDecoder.norm(self, x, name)
        if x.shape[2] >= 512 and name in ("input_layernorm", "post_attention_layernorm"):
            result = ttnn.typecast(result, ttnn.bfloat8_b)
        return result

    def linear(self, x, name):
        if x.shape[2] >= 512:
            return PrefillPrecisionCandidate.linear(self, x, name)
        return MultichipDecoder.linear(self, x, name)

    def reduce(self, x):
        if x.shape[2] >= 512:
            return ttnn.typecast(MultichipDecoder.reduce(self, ttnn.typecast(x, ttnn.bfloat8_b)), ttnn.bfloat16)
        return MultichipDecoder.reduce(self, x)

    def finish(self, x, mixed):
        if 1 < x.shape[2] <= 128:
            return PrefillKernelCandidate.finish(self, x, mixed)
        if x.shape[2] >= 512:
            return PrefillPrecisionCandidate.finish(self, x, mixed)
        return MultichipDecoder.finish(self, x, mixed)


class PrefillLoFiCandidate(PhaseCandidate):
    selected_kind = None

    def linear(self, x, name):
        if (x.shape[2] >= 512 and name == "packed_input"
                and self.selected_kind in (None, self.kind)):
            return ttnn.experimental.minimal_matmul(
                ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG), self.weights[name + ".weight"],
                config=self.prefill_program, compute_kernel_config=self.mlp_compute,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16,
            )
        return super().linear(x, name)


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--harness", choices=["packed_stack", "probe", "long"], required=True)
    ap.add_argument("--prefill-input-lofi", action="store_true")
    ap.add_argument("--lofi-kind", choices=["linear_attention", "full_attention"])
    args, rest = ap.parse_known_args()
    PrefillLoFiCandidate.selected_kind = args.lofi_kind
    harness = importlib.import_module("models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_" + args.harness)
    harness.MultichipDecoder = PrefillLoFiCandidate if args.prefill_input_lofi else PhaseCandidate
    sys.argv = [sys.argv[0], *rest]
    harness.main()


if __name__ == "__main__":
    main()
