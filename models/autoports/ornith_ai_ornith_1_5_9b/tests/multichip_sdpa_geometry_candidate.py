# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Cross SDPA precision with K chunks, owning any attention-table alignment."""
import argparse
import json
import sys
from pathlib import Path
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_sdpa_precision_candidate as precision

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--sdpa-k-chunk', type=int, choices=[32,64,128,256,512,1024], required=True)
    args, remaining = parser.parse_known_args()
    original_wrap = precision.wrap_sdpa
    observations = []
    def wrap_geometry(operation):
        def geometry(*operands, **kwargs):
            config = kwargs['program_config']
            config.k_chunk_size = args.sdpa_k_chunk
            table = operands[3]
            block = operands[1].shape[2]
            assert block == 32
            page_multiple = args.sdpa_k_chunk // block
            width = table.shape[-1]
            aligned_width = ((width + page_multiple - 1) // page_multiple) * page_multiple
            if width != aligned_width:
                # Page zero is in bounds. Added columns are beyond every valid
                # logical position and SDPA masks them; cache-update tables stay
                # unchanged because this adaptation is local to attention.
                table = ttnn.pad(table, [(0,0), (0,aligned_width-width)], 0)
                operands = (*operands[:3], table, *operands[4:])
            key = dict(query=list(operands[0].shape), table_width=width,
                       aligned_width=aligned_width, program=str(config))
            if key not in observations: observations.append(key)
            return operation(*operands, **kwargs)
        return geometry
    precision.wrap_sdpa = lambda operation, policy, observed: original_wrap(wrap_geometry(operation), policy, observed)
    sys.argv = [sys.argv[0], *remaining]
    try:
        precision.main()
    finally:
        output = Path(remaining[remaining.index('--output')+1])
        output.with_name(output.stem+'_sdpa_geometry.json').write_text(json.dumps(
            dict(k_chunk=args.sdpa_k_chunk, observed=observations,
                 public_alignment_requirement=None, cache_update_table_unchanged=True), indent=2)+'\n')
