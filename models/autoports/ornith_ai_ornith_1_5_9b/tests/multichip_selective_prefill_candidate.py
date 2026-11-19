# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""8192-token chunks with DRAM-first norm staging only above4096 rows."""
import sys
from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_prefill_lifetime_candidate as base
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_large_chunk_geometry import LargeChunkGeometry as OriginalGeometry

class SelectivePrefill(base.PrefillLifetimeCandidate):
    norm_quantize_first=True
    @classmethod
    def from_state_dict(cls,*args,**kwargs):
        kwargs['chunk_size']=8192
        return super().from_state_dict(*args,**kwargs)
    def norm(self,x,name):
        if x.shape[2] <= 4096:
            return OriginalGeometry.norm(self,x,name)
        return super().norm(x,name)

if __name__=='__main__':
    if '--long-context' in sys.argv:
        sys.argv.remove('--long-context')
        from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_sdpa_reference as reference
        from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import CCLManager
        reference.CCLManager=CCLManager
        from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_long as runner
        runner.MultichipDecoder=SelectivePrefill
        runner.main()
    else:
        base.PrefillLifetimeCandidate=SelectivePrefill
        sys.argv+=['--norm-quantize-first']
        base.main()
