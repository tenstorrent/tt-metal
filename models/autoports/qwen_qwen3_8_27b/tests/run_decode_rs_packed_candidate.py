# SPDX-License-Identifier: Apache-2.0
"""RS role geometry on the winning shared packed-RoPE decode family."""
import hashlib
import json
from pathlib import Path
import sys
from . import run_multichip_decoder as harness
from .run_packed_rope_candidate import PackedRopeCandidate
from .run_decode_rs_grid_candidate import projection
from ..tt import fused_decode

if __name__=='__main__':
    provenance={f'tests/{name}':hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('run_decode_rs_packed_candidate.py','run_decode_rs_grid_candidate.py','run_packed_rope_candidate.py')}
    fused_decode.native_rs_projection=projection
    harness.MultichipDecoder=PackedRopeCandidate
    output=Path(sys.argv[sys.argv.index('--output')+1])
    try:harness.main()
    finally:
        if output.exists():
            r=json.loads(output.read_text());r['provenance'].update(provenance)
            r['candidate_adapter']='Native decodeRS grid and role-isolated K with the selected shared-table packed RoPE enabled by policy. Same persistent owner and residual.'
            output.write_text(json.dumps(r,indent=2)+'\n')
