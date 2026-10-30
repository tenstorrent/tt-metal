# SPDX-License-Identifier: Apache-2.0
"""Tile-aligned prefill users folded into sequence rows, candidate only."""
import hashlib
import json
from pathlib import Path
import sys
import ttnn
from . import run_multichip_decoder as harness
_IMPORTED_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def packed_prefill_rope(model,value,tables,batch,length):
    heads=value.shape[1]
    rot,tail=value[...,:64],value[...,64:]
    packed=ttnn.reshape(ttnn.permute(rot,(1,0,2,3)),[1,heads,batch*length,64])
    y=ttnn.experimental.rotary_embedding(packed,*tables,compute_kernel_config=model.ckc)
    y=ttnn.permute(ttnn.reshape(y,[heads,batch,length,64]),(1,0,2,3))
    return ttnn.concat([y,tail],dim=-1,memory_config=ttnn.DRAM_MEMORY_CONFIG)

class PrefillRopeCandidate(harness.MultichipDecoder):
    def _attention(self,x,state,*,page_table,cos,sin,current_pos=None,chunk_start=0):
        if not self.policy.get('prefill_packed_rope') or current_pos is not None or x.shape[0]<8 or x.shape[1]%32:
            return super()._attention(x,state,page_table=page_table,cos=cos,sin=sin,current_pos=current_pos,chunk_start=chunk_start)
        batch,length,_=x.shape
        tables=tuple(ttnn.reshape(t,[1,1,batch*length,64]) for t in (cos,sin))
        original=self._rope
        def apply(value,c,s):return packed_prefill_rope(self,value,tables,batch,length)
        self._rope=apply
        try:return super()._attention(x,state,page_table=page_table,cos=cos,sin=sin,current_pos=current_pos,chunk_start=chunk_start)
        finally:self._rope=original

if __name__=='__main__':
    harness.MultichipDecoder=PrefillRopeCandidate
    output=Path(sys.argv[sys.argv.index('--output')+1])
    try:harness.main()
    finally:
        if output.exists():
            r=json.loads(output.read_text());r['provenance']['tests/run_prefill_rope_candidate.py']=_IMPORTED_SHA
            r['candidate_adapter']='Tile-aligned prefill only: fold users into independent sequence rows with matching cos/sin; unchanged DRAM boundary and unchanged decode.'
            output.write_text(json.dumps(r,indent=2)+'\n')
