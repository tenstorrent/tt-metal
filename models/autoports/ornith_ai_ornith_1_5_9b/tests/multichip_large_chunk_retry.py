# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Smaller K blocks for larger prefill chunks after measured L1 conflicts."""
import argparse
import sys
from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_large_chunk_geometry as base

if __name__ == "__main__":
    ap=argparse.ArgumentParser(add_help=False)
    ap.add_argument("--large-k-block",type=int,required=True)
    args,rest=ap.parse_known_args();sys.argv=[sys.argv[0],*rest]
    base.LargeChunkGeometry.large_k_block=args.large_k_block
    base.main()
