# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolate fused projection reductions with the unchanged public batch gates."""
import argparse
import sys
from unittest.mock import patch

from models.autoports.qwen_qwen3_8_flash_next.tests import run_fused_probe
from models.autoports.qwen_qwen3_8_flash_next.tt import fused_decoder
from models.autoports.qwen_qwen3_8_flash_next.tt.precise_gdn_projection import PreciseGDNProjection
from models.autoports.qwen_qwen3_8_flash_next.tt.precise_projection import PreciseRouterProjection
from models.autoports.qwen_qwen3_8_flash_next.tt.precise_residual_projection import PreciseResidualProjection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-reduction", choices=("gdn", "qkv", "out", "router", "gr", "all", "qkv_sum_group1"), required=True
    )
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    classes = {"gdn": PreciseGDNProjection, "router": PreciseRouterProjection, "gr": PreciseResidualProjection}
    selected = (
        tuple(classes.values())
        if args.original_reduction == "all"
        else (classes.get(args.original_reduction, PreciseGDNProjection),)
    )
    fused = fused_decoder.FusedPreciseProjection
    calls = []

    def factory(original, compute, decode_group=1, **kwargs):
        if args.original_reduction == "qkv_sum_group1":
            group = 1 if isinstance(original, PreciseGDNProjection) and original.output_width == 10240 else decode_group
            calls.append((type(original).__name__, getattr(original, "output_width", None), "sum", group))
            return fused(original, compute, group, **{**kwargs, "balanced_tree": False})
        restore = isinstance(original, selected)
        if args.original_reduction in ("qkv", "out"):
            restore = restore and original.output_width == (10240 if args.original_reduction == "qkv" else 2560)
        calls.append((type(original).__name__, getattr(original, "output_width", None), restore))
        return original if restore else fused(original, compute, decode_group, **kwargs)

    sys.argv = [__file__] + (args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments)
    try:
        with patch.object(fused_decoder, "FusedPreciseProjection", factory):
            run_fused_probe.main()
    finally:
        print("PROJECTION_CONTROL", args.original_reduction, calls, flush=True)


if __name__ == "__main__":
    main()
