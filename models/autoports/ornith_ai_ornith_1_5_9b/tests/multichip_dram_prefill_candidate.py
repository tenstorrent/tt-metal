# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated large-prefill DRAM residency / minimal-K-block discriminator.

The frozen reference owns alignment, recurrence masking, cache padding and output
slicing. This subclass changes only norm staging and large projection residency.
It inherits the validated early-release prefill loop without editing that module.
"""

import argparse
import sys

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_large_chunk_geometry as geometry
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_prefill_lifetime_candidate import PrefillLifetimeCandidate

# Capture before main replaces geometry.LargeChunkGeometry process-locally.
OriginalGeometry = geometry.LargeChunkGeometry


class DramPrefillCandidate(PrefillLifetimeCandidate):
    minimal_k_block = 16
    dram_threshold = 8192

    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        layer = super().from_state_dict(*args, **kwargs)
        assert layer.residual_layout == "replicated", "This discriminator holds the replicated residual contract fixed"
        assert not layer.fused_prefill, "This discriminator measures separate minimal matmul with DRAM input"
        assert tuple(layer.selected) == ("control",), "Use --roles control; do not mix multicast geometries"
        print(
            "DRAM_PREFILL_POLICY",
            dict(norm_dram_first_above=4096, keep_dram_above=cls.dram_threshold, minimal_k_block=cls.minimal_k_block),
            flush=True,
        )
        return layer

    def norm(self, x, name):
        if x.shape[2] <= 4096 or name not in ("input_layernorm", "post_attention_layernorm"):
            return OriginalGeometry.norm(self, x, name)
        out = ttnn.rms_norm(
            x,
            weight=self.weights[name + ".weight"],
            epsilon=self.config.rms_norm_eps,
            compute_kernel_config=self.compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.typecast(out, ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if x.shape[2] <= self.dram_threshold:
            return ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG)
        return out

    def linear(self, x, name):
        if x.shape[2] <= self.dram_threshold:
            return super().linear(x, name)
        # The frozen reference unconditionally copies every prefill projection
        # activation to L1. Avoid that copy for BOTH normalized BF8 producers and
        # the BF16 attention/down-projection inputs; no dtype change here.
        return ttnn.experimental.minimal_matmul(
            ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG),
            self.weights[name + ".weight"],
            config=self.prefill_program,
            compute_kernel_config=self.matmul_compute(name),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--minimal-k-block", type=int, choices=(8, 16), default=16)
    parser.add_argument("--dram-threshold", type=int, default=8192)
    args, remaining = parser.parse_known_args()
    assert args.dram_threshold >= 4096
    DramPrefillCandidate.minimal_k_block = args.minimal_k_block
    DramPrefillCandidate.dram_threshold = args.dram_threshold
    original = geometry.LargeChunkGeometry
    original_argv = sys.argv[:]
    try:
        geometry.LargeChunkGeometry = DramPrefillCandidate
        sys.argv = [sys.argv[0], *remaining]
        geometry.main()
    finally:
        geometry.LargeChunkGeometry = original
        sys.argv[:] = original_argv


if __name__ == "__main__":
    main()
