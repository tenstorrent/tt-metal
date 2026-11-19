# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Replicated AR precision families on recorded HF activations, same gates."""
import argparse
import sys
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_bf8_ar_candidate import BF8AR, BF8Collectives
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_bf8_row_ar_candidate import BF8RowAR
from models.autoports.ornith_ai_ornith_1_5_9b.tt import multichip_decoder as production
from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_recorded_batch as harness

if __name__=='__main__':
    ap=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    ap.add_argument('--ar-policy',choices=['control','payload','output','row'],required=True)
    args,rest=ap.parse_known_args()
    if args.ar_policy!='control':
        production.MultichipCollectives=BF8Collectives
    parent=production.MultichipDecoder if args.ar_policy=='control' else BF8RowAR if args.ar_policy=='row' else BF8AR
    class RecordedAR(parent):
        output_bf8=args.ar_policy=='output'
    harness.GdnFidelityCandidate=RecordedAR
    sys.argv=[sys.argv[0],*rest]
    harness.main()
