# SPDX-License-Identifier: Apache-2.0
"""Decode users as independent RoPE sequence rows; candidate-only adapter."""
import hashlib
import json
from pathlib import Path
import sys
import ttnn
from . import run_multichip_decoder as harness

_IMPORTED_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def packed_tables(cos,sin,batch):
    return tuple(ttnn.to_layout(ttnn.reshape(ttnn.to_layout(t,ttnn.ROW_MAJOR_LAYOUT),[1,1,batch,64]),
        ttnn.TILE_LAYOUT,memory_config=ttnn.L1_MEMORY_CONFIG) for t in (cos,sin))


def packed_rope(model,x,tables):
    batch=x.shape[0]
    assert x.shape[2]==1 and 1<batch<=32
    rot=ttnn.to_layout(x[...,:64],ttnn.ROW_MAJOR_LAYOUT)
    packed=ttnn.to_layout(ttnn.permute(rot,(2,1,0,3),memory_config=ttnn.L1_MEMORY_CONFIG),
        ttnn.TILE_LAYOUT,memory_config=ttnn.L1_MEMORY_CONFIG)
    y=ttnn.experimental.rotary_embedding(packed,*tables,compute_kernel_config=model.ckc,
        memory_config=ttnn.L1_MEMORY_CONFIG)
    y=ttnn.to_layout(y,ttnn.ROW_MAJOR_LAYOUT)[:,:,:batch,:]
    y=ttnn.to_layout(ttnn.permute(y,(2,1,0,3),memory_config=ttnn.L1_MEMORY_CONFIG),
        ttnn.TILE_LAYOUT,memory_config=ttnn.L1_MEMORY_CONFIG)
    return ttnn.concat([y,x[...,64:]],dim=-1,memory_config=ttnn.DRAM_MEMORY_CONFIG)


class PackedRopeCandidate(harness.MultichipDecoder):
    def _attention(self,x,state,*,page_table,cos,sin,current_pos=None,chunk_start=0):
        mode=self.policy.get('packed_rope_candidate')
        if not mode or current_pos is None or x.shape[0]<self.policy.get('packed_rope_min_batch',8):
            return super()._attention(x,state,page_table=page_table,cos=cos,sin=sin,current_pos=current_pos,chunk_start=chunk_start)
        original=self._rope
        tables=None
        def apply(value,c,s):
            nonlocal tables
            if tables is None or mode=='separate_tables':tables=packed_tables(c,s,value.shape[0])
            return packed_rope(self,value,tables)
        self._rope=apply
        try:return super()._attention(x,state,page_table=page_table,cos=cos,sin=sin,current_pos=current_pos,chunk_start=chunk_start)
        finally:self._rope=original


if __name__=='__main__':
    harness.MultichipDecoder=PackedRopeCandidate
    output=Path(sys.argv[sys.argv.index('--output')+1])
    try:harness.main()
    finally:
        if output.exists():
            r=json.loads(output.read_text());r['provenance']['tests/run_packed_rope_candidate.py']=_IMPORTED_SHA
            r['candidate_adapter']='Decode-only row-major pack/unpack; one RoPE per Q/K, distinct user position rows, shared or separate transient cos/sin tables. No persistent state/cache or B1/prefill change.'
            output.write_text(json.dumps(r,indent=2)+'\n')
