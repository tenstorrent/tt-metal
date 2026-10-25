# SPDX-License-Identifier: Apache-2.0
"""Selected-precision native RS grid comparison; unchanged owner and residual."""
import hashlib
import json
from pathlib import Path
import sys
import ttnn
from . import run_multichip_decoder as harness
from ..tt import fused_prefill

_IMPORTED_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_original=fused_prefill.prefill_rs_projection


def projection(model,x,role):
    geometry=model.policy.get('candidate_rs_grid')
    if geometry is None:return _original(model,x,role)
    gx,gy=geometry['grid']
    pn=(160+gx-1)//gx
    constructor=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig
    def configured(*args,**kwargs):
        kwargs.update(compute_with_storage_grid_size=(gx,gy),
            allowed_worker_cores=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0,0),ttnn.CoreCoord(gx-1,gy-1))]),
            per_core_N=pn,out_block_w=pn,
            out_subblock_h=geometry['subh'],out_subblock_w=geometry['subw'])
        if role in geometry.get('role_h',{}):kwargs['out_block_h']=geometry['role_h'][role]
        if role in geometry.get('role_subh',{}):kwargs['out_subblock_h']=geometry['role_subh'][role]
        if role in geometry.get('role_subw',{}):kwargs['out_subblock_w']=geometry['role_subw'][role]
        return constructor(*args,**kwargs)
    ttnn.MatmulMultiCoreReuseMultiCastProgramConfig=configured
    try:return _original(model,x,role)
    finally:ttnn.MatmulMultiCoreReuseMultiCastProgramConfig=constructor


if __name__=='__main__':
    fused_prefill.prefill_rs_projection=projection
    output=Path(sys.argv[sys.argv.index('--output')+1])
    try:harness.main()
    finally:
        if output.exists():
            r=json.loads(output.read_text());r['provenance']['tests/run_prefill_rs_grid_candidate.py']=_IMPORTED_SHA
            r['candidate_adapter']='Only native prefillRS grid/N/subblock; full-width native reader, same selectedK48/17,H4,BF16input/BFP4LoFi/BFP8output, persistent owner/CCLoffset/residual/decode unchanged.'
            output.write_text(json.dumps(r,indent=2)+'\n')
