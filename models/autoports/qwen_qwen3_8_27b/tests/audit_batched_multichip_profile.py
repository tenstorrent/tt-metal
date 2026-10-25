# SPDX-License-Identifier: Apache-2.0
"""Audit the selected batched native-RS family from actual device profile rows."""
import csv
import json
from pathlib import Path
from models.autoports.qwen_qwen3_8_27b.tests.profile_fidelity_evidence import fused_matmul_fidelity

base=Path(__file__).resolve().parents[1]/'doc/optimized_multichip_decoder'
result={}
for layer in (0,3):
    root=base/f'tracy/final_b32_profile_l{layer}'
    timing=json.loads((root/'whole_layer_timing.json').read_text())
    run=json.loads((base/f'final_b32_profile_l{layer}.json').read_text())
    rows=list(csv.DictReader((root/'decode_ops.csv').open()))
    assert {r['DEVICE ID'] for r in rows}=={'0','1','2','3'}
    devices={}
    for device in ('0','1','2','3'):
        local=[r for r in rows if r['DEVICE ID']==device]
        fused=[r for r in local if 'MatmulReduceScatterAsync' in r['OP CODE']]
        assert len(fused)==2,(device,[r['OP CODE'] for r in local])
        for r in fused:
            fused_matmul_fidelity(r)
            assert int(r['CORE COUNT'])==22,r
            assert r['INPUT_0_DATATYPE']=='BFLOAT16' and r['INPUT_1_DATATYPE']=='BFLOAT4_B',r
            geometry=run['multichip']['effective_decode_policy']['decode_rs_geometry']
            gx,gy=geometry['grid'];pn=(160+gx-1)//gx
            k=int(r['INPUT_1_Y_PAD[LOGICAL]'].split('[')[0]);role='out' if k==1536 else 'down'
            for field,value in [('out_block_w',pn),('per_core_N',pn),('out_subblock_w',geometry['subw']),('in0_block_w',geometry[role+'_k'])]:
                assert f'{field}={value}' in r['ATTRIBUTES'],r
            assert f'compute_with_storage_grid_size={gx}-{gy}' in r['ATTRIBUTES'],r
        conv=[r for r in local if 'conv' in r['OP CODE'].lower()]
        assert len(conv)==(1 if layer==0 else 0),(device,[r['OP CODE'] for r in conv])
        rotary=[r for r in local if 'RotaryEmbedding' in r['OP CODE']]
        assert len(rotary)==(2 if layer==3 else 0),(device,len(rotary))
        envelope=timing['phases']['decode']['per_device'][device]['device_us']
        fields=['OP CODE','MATH FIDELITY','CORE COUNT','ATTRIBUTES','INPUT_0_DATATYPE','INPUT_0_MEMORY','INPUT_1_DATATYPE','INPUT_1_MEMORY','OUTPUT_0_MEMORY','OUTPUT_1_MEMORY']
        contracts=[]
        for r in local:
            item={k:r.get(k) for k in fields}
            if 'MatmulReduceScatterAsync' in r['OP CODE']:item['per_kernel_fidelity']=fused_matmul_fidelity(r)
            item['kernel_us']=float(r['DEVICE KERNEL DURATION [ns]'])/1000
            item['whole_layer_pct']=100*item['kernel_us']/envelope
            contracts.append(item)
        devices[device]={'whole_layer_device_us':envelope,'native_fused_rs_count':len(fused),'convolution_count':len(conv),'rotary_count':len(rotary),'sequence':contracts}
    prefill=list(csv.DictReader((root/'prefill_ops.csv').open()))
    for device in ('0','1','2','3'):
        count=sum('RotaryEmbedding' in r['OP CODE'] for r in prefill if r['DEVICE ID']==device)
        assert count==(2 if layer==3 else 0),(device,count)
    assert run['multichip']['decode_residual_layout']=='sharded'
    result[run['layer_type']]={'batch':32,'mesh':[1,4],'residual':'1280 hidden channels/chip; no boundary gather','devices':devices,'source':str(root/'decode_ops.csv'),'decoder_sha256':run['provenance']['tt/multichip_decoder.py']}
(base/'batched_runtime_contract.json').write_text(json.dumps(result,indent=2)+'\n')
