# SPDX-License-Identifier: Apache-2.0
"""Per-projection L1 input placement screen without changing production defaults."""
import hashlib
import json
from pathlib import Path
import sys
from . import run_multichip_decoder as harness

class PrefillL1Candidate(harness.MultichipDecoder):
    def _linear(self,x,w,activation=None,**kwargs):
        role=self.weight_roles[id(w)]
        old_l1=self.policy.get('prefill_l1',False)
        sentinel=object()
        old_block=self.policy.get('prefill_out_block_h',sentinel)
        self.policy['prefill_l1']=role in self.policy.get('prefill_l1_roles',[])
        block=self.policy.get('prefill_l1_role_block_h',{}).get(role)
        if block is not None:self.policy['prefill_out_block_h']=block
        try:return super()._linear(x,w,activation,**kwargs)
        finally:
            self.policy['prefill_l1']=old_l1
            if old_block is sentinel:self.policy.pop('prefill_out_block_h',None)
            else:self.policy['prefill_out_block_h']=old_block

if __name__=='__main__':
    harness.MultichipDecoder=PrefillL1Candidate
    output=Path(sys.argv[sys.argv.index('--output')+1])
    try:harness.main()
    finally:
        if output.exists():
            r=json.loads(output.read_text())
            source=Path(__file__)
            r['provenance']['tests/run_prefill_l1_candidate.py']=hashlib.sha256(source.read_bytes()).hexdigest()
            r['candidate_adapter']='Per-role prefill L1 selection; same production decoder/mesh/residual/weights, baseline unchanged.'
            output.write_text(json.dumps(r,indent=2)+'\n')
