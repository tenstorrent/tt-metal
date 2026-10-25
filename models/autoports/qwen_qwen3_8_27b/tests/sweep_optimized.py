# SPDX-License-Identifier: Apache-2.0
"""Serial policy screens on real weights, followed by explicit profile finalists."""
import json
import os
import subprocess
from pathlib import Path
base = Path('models/autoports/qwen_qwen3_8_27b')
base_policy={'dram_sharded':True,'fidelity':'LoFi','mlp_dtype':'bfloat4_b','attention_dtype':'bfloat4_b','down_dtype':'bfloat4_b','block':4,'sharded_norm':True,'carry_residual':True,'kv_dtype':'bfloat8_b','packed_mlp':True,'prefill_chunk':512,'prefill_grid':[8,8],'prefill_block':4,'inplace_recurrence':True}
layouts={
 'g80':{'mix':{'cores':80,'block':2},'gateup':{'cores':80,'block':2},'out':{'cores':64,'block':3},'down':{'cores':68,'block':8}},
 'g40':{'mix':{'cores':40,'block':4},'gateup':{'cores':40,'block':4},'out':{'cores':48,'block':4},'down':{'cores':68,'block':8}},
 'g32':{'mix':{'cores':32,'block':5},'gateup':{'cores':32,'block':5},'out':{'cores':32,'block':6},'down':{'cores':34,'block':16}},
 'g16':{'mix':{'cores':16,'block':10},'gateup':{'cores':16,'block':5},'out':{'cores':24,'block':8},'down':{'cores':34,'block':16}},
}
policies={f'{name}_r{r}':{**base_policy,**layout,'readers':r} for name,layout in layouts.items() for r in (1,2,3)}
for name, policy in policies.items():
    for layer in (0,3):
        label=f'{name}_l{layer}'
        with (base/'doc/optimized_decoder'/f'{label}.log').open('w') as log:
            result=subprocess.run(['bash',str(base/'tests/optimized_run.sh'),'check',label,'--layer',str(layer),'--seq',str(1025 if name.startswith('prefill') else 129),'--timing','--recorded'], env={**os.environ,'QWEN_OPT_POLICY':json.dumps(policy)},stdout=log,stderr=subprocess.STDOUT)
        print(label, result.returncode,flush=True)
        (base/'doc/optimized_decoder'/f'{label}.policy.json').write_text(json.dumps(policy,indent=2)+'\n')
