# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Combined decode fusion and wave geometry, validated before production selection."""

import argparse
import importlib
import sys

from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_local_fusion_candidate import LocalFusionCandidate
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_prefill_precision_candidate import PrefillPrecisionCandidate


class CumulativeCandidate(LocalFusionCandidate):
    fused_conv = True
    fused_pad = True
    decode_wave_heads = 52
    gdn_depth = 1


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--harness", choices=["stack", "packed_stack", "long", "probe", "pool_contracts"], required=True)
    ap.add_argument("--prefill-precision", action="store_true")
    args, rest = ap.parse_known_args()
    harness = importlib.import_module("models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_" + args.harness)
    harness.MultichipDecoder = PrefillPrecisionCandidate if args.prefill_precision else CumulativeCandidate
    sys.argv = [sys.argv[0], *rest]
    harness.main()


if __name__ == "__main__":
    main()
