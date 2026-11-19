# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare cumulative decode changes on the same packed TP4 stack workload."""

import argparse
import json
import sys
from pathlib import Path

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt import multichip_decoder as production
from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_fast_zero_kernel as zero
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_bf8_ar_candidate import BF8AR, BF8Collectives
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_pooled_one_full_candidate import PooledOneFull
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_fast_zero_conv_gates_fp32_softplus_candidate import FastZeroConvGatesCandidate
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_sdpa_precision_candidate import wrap_sdpa


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--combined-family", choices=("zero", "zero_bf8", "zero_bf8_sdpa", "conv_bf8", "conv_bf8_sdpa"), required=True)
    options, remaining = parser.parse_known_args()
    bf8 = options.combined_family != "zero"
    fused_conv = options.combined_family.startswith("conv")
    sdpa = options.combined_family.endswith("sdpa")
    production.flat_ordered_post_loop = zero.flat_ordered_post_loop

    class Combined(PooledOneFull):
        output_bf8 = False

        def decode_reduce(self, x):
            return BF8AR.decode_reduce(self, x) if bf8 else super().decode_reduce(x)

        def pooled_linear_attention_decode_batch(self, packed, states):
            if fused_conv:
                return FastZeroConvGatesCandidate.pooled_linear_attention_decode_batch(self, packed, states)
            return super().pooled_linear_attention_decode_batch(packed, states)

    from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_packed_stack as stack
    stack.MultichipDecoder = Combined
    stack.MultichipCollectives = BF8Collectives if bf8 else production.MultichipCollectives
    observations = {}
    original = ttnn.transformer.paged_scaled_dot_product_attention_decode
    if sdpa:
        def chunk64(*operands, **kwargs):
            kwargs["program_config"].k_chunk_size = 64
            # Native batch production already owns four-page table padding.
            # Also own the scalar continuation's potentially odd table width.
            if operands[3].shape[-1] % 2:
                table = ttnn.pad(operands[3], [(0, 0), (0, 1)], 0)
                operands = (*operands[:3], table, *operands[4:])
            return original(*operands, **kwargs)
        ttnn.transformer.paged_scaled_dot_product_attention_decode = wrap_sdpa(chunk64, "hifi2_exact_fp32", observations)
    sys.argv = [sys.argv[0], *remaining]
    try:
        stack.main()
    finally:
        ttnn.transformer.paged_scaled_dot_product_attention_decode = original
        output = Path(remaining[remaining.index("--output") + 1])
        output.with_name(output.stem + "_combined_policy.json").write_text(json.dumps(dict(
            family=options.combined_family, pooled_linear_B1=True, packed_full_B1=True,
            zero_fill=True, bf8_collective_payload=bf8, bf16_collective_output=True,
            fp32_softplus_conv_fusion=fused_conv, sdpa_hifi2_k64=sdpa,
            sdpa_observations=list(observations.values()),
        ), indent=2) + "\n")


if __name__ == "__main__":
    main()
