# SPDX-License-Identifier: Apache-2.0
"""Test non-power-of-two output subblocks without changing frozen defaults."""
import os
import ttnn
from .run_optimized_decoder import main
original=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig
width=int(os.environ['QWEN_SUBBLOCK_WIDTH'])
def candidate(**kwargs):
    if kwargs['out_subblock_w']==1 and kwargs['per_core_N'] % width==0:
        kwargs['out_subblock_w']=width
        while kwargs['out_subblock_h']*width>8 or kwargs['per_core_M']%kwargs['out_subblock_h']:
            kwargs['out_subblock_h']-=1
    return original(**kwargs)
ttnn.MatmulMultiCoreReuseMultiCastProgramConfig=candidate
print('PREFILL_SUBBLOCK_EXPERIMENT',width,flush=True)
main()
import hashlib,json,sys
from pathlib import Path
output=Path(sys.argv[sys.argv.index('--output')+1])
result=json.loads(output.read_text())
result['experimental_override']={'prefill_subblock_width_where_original_is_one':width,'source':__file__,'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
output.write_text(json.dumps(result,indent=2)+'\n')
