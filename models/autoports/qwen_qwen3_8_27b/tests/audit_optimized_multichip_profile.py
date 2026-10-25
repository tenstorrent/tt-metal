# SPDX-License-Identifier: Apache-2.0
"""Verify final runtime precision and retain boundary/config evidence from raw rows."""
import csv
import json
import re
from models.autoports.qwen_qwen3_8_27b.tests.profile_fidelity_evidence import fused_matmul_fidelity
from pathlib import Path

base=Path(__file__).resolve().parents[1]/'doc/optimized_multichip_decoder'
result={}
for layer in (0,3):
    root=base/f'tracy/final_profile_l{layer}'
    timing=json.loads((root/'whole_layer_timing.json').read_text())
    run=json.loads((base/f'final_profile_l{layer}.json').read_text())
    policy=run['multichip']['effective_policy']
    rows=list(csv.DictReader((root/'decode_ops.csv').open()))
    assert {r['DEVICE ID'] for r in rows}=={'0','1','2','3'}
    contracts=[]
    for r in rows:
        if r['DEVICE ID']!='0':continue
        item={k:r[k] for k in ['OP CODE','GLOBAL CALL COUNT','MATH FIDELITY','CORE COUNT','ATTRIBUTES','INPUT_0_DATATYPE','INPUT_0_MEMORY','INPUT_1_DATATYPE','INPUT_1_MEMORY','OUTPUT_0_MEMORY']}
        item['kernel_us']=float(r['DEVICE KERNEL DURATION [ns]'])/1000
        item['whole_layer_pct']=100*item['kernel_us']/timing['phases']['decode']['device_us']
        if r['OP CODE']=='MatmulDeviceOperation' and r['INPUT_1_DATATYPE']=='BFLOAT4_B':
            k=int(r['INPUT_1_Y_PAD[LOGICAL]'].split('[')[0]);n=int(r['INPUT_1_X_PAD[LOGICAL]'].split('[')[0])
            role='out' if k==1536 else 'down' if k==4352 else 'gateup' if n>=8704 else 'mix'
            assert r['MATH FIDELITY']=='LoFi'
            assert f"num_workers_per_dram_bank={policy[role]['readers']}" in r['ATTRIBUTES']
            assert f"in0_block_w={policy[role]['block']}" in r['ATTRIBUTES']
            item.update(role=role,physical_MKN=[32,k,n],input_storage_cores=policy[role]['cores'],input_shard_K_tiles=k//32//policy[role]['cores'],classification='DRAM-sharded weight streaming; physical bandwidth diagnostics in reader microbenchmark')
        contracts.append(item)
    roles=[r['role'] for r in contracts if 'role' in r]
    assert sorted(roles)==['down','gateup','mix','out'],roles
    per_device={}
    for device in ('0','1','2','3'):
        local=[r for r in rows if r['DEVICE ID']==device]
        kernel=sum(float(r['DEVICE KERNEL DURATION [ns]'])/1000 for r in local)
        firmware=sum(float(r['DEVICE FW DURATION [ns]'])/1000 for r in local)
        envelope=timing['phases']['decode']['per_device'][device]['device_us']
        per_device[device]={'whole_layer_us':envelope,'summed_kernel_us':kernel,'summed_firmware_us':firmware,'envelope_minus_kernel_us':envelope-kernel,'inter_firmware_gaps_us':envelope-firmware}
    prefill_contracts=[]
    separate_columns=0
    for r in csv.DictReader((root/'prefill_ops.csv').open()):
        if r['DEVICE ID']!='0' or (r['OP CODE']!='MatmulDeviceOperation' and 'MatmulReduceScatterAsync' not in r['OP CODE']) or r['INPUT_1_DATATYPE']!='BFLOAT4_B':continue
        k=int(r['INPUT_1_Y_PAD[LOGICAL]'].split('[')[0]);n=int(r['INPUT_1_X_PAD[LOGICAL]'].split('[')[0])
        role='out' if k==1536 else 'down' if k==4352 else 'gateup' if n>=8704 else 'mix'
        if k==5120 and n==4352 and policy.get('separate_prefill'):
            role=('gate','up')[separate_columns]
            separate_columns+=1
        fused='MatmulReduceScatterAsync' in r['OP CODE']
        grid=policy['prefill_rs_geometry'].get('grid',[8,8]) if fused else policy[role].get('prefill_grid',policy.get('prefill_grid',[11,10]))
        assert f"compute_with_storage_grid_size={grid[0]}-{grid[1]}" in r['ATTRIBUTES']
        block=policy['prefill_rs_geometry'][role+'_k'] if fused else policy[role]['prefill_block']
        assert f"in0_block_w={block}" in r['ATTRIBUTES']
        fidelity_evidence=fused_matmul_fidelity(r) if fused else None
        if not fused:assert r['MATH FIDELITY']=='LoFi'
        if fused:
            geometry=policy['prefill_rs_geometry']
            assert int(r['CORE COUNT'])==108,r
            assert role in ('out','down') and f'out_block_w={(160+grid[0]-1)//grid[0]}' in r['ATTRIBUTES']
            for key,field in [('block_h','out_block_h'),('subh','out_subblock_h'),('subw','out_subblock_w')]:
                assert f"{field}={geometry[key]}" in r['ATTRIBUTES']
            assert r['OUTPUT_0_DATATYPE']=='BFLOAT8_B'
        if layer==3 and role=='mix':
            assert 'out_subblock_h=4' in r['ATTRIBUTES'] and 'out_subblock_w=1' in r['ATTRIBUTES']
        expected_dtype=policy[role].get('prefill_activation','bfloat16').upper()
        assert r['INPUT_0_DATATYPE']==expected_dtype,(role,r['INPUT_0_DATATYPE'],expected_dtype)
        expected_memory='L1' if policy[role].get('prefill_l1') else 'DRAM'
        assert f'_{expected_memory}_INTERLEAVED' in r['INPUT_0_MEMORY'],(role,r['INPUT_0_MEMORY'])
        prefill_contracts.append(dict(role=role,operation=r['OP CODE'],attributes=r['ATTRIBUTES'],input_dtype=r['INPUT_0_DATATYPE'],input_memory=r['INPUT_0_MEMORY'],weight_dtype=r['INPUT_1_DATATYPE'],fidelity='LoFi',scalar_profiler_fidelity=r['MATH FIDELITY'],per_kernel_fidelity=fidelity_evidence))
    expected_roles=['down','gate','mix','out','up'] if policy.get('separate_prefill') else ['down','gateup','mix','out']
    assert sorted(r['role'] for r in prefill_contracts)==expected_roles
    if policy.get('separate_prefill'):
        assert separate_columns==2
        for item in prefill_contracts:
            if item['role'] in ('gate','up'):
                if item['role']=='gate':assert 'op_type=UnaryOpType::SILU' in item['attributes']
                else:assert 'fused_activation=std::nullopt' in item['attributes']
                for field,value in [('out_block_h',8),('out_subblock_h',8),('out_subblock_w',1)]:
                    assert f'{field}={value}' in item['attributes']
    ccl_contracts=[]
    for r in csv.DictReader((root/'prefill_ops.csv').open()):
        if r['DEVICE ID']!='0' or (r['OP CODE'] not in ('AllGatherAsyncDeviceOperation','ReduceScatterMinimalAsyncDeviceOperation') and 'MatmulReduceScatterAsync' not in r['OP CODE']):continue
        width=int(r['INPUT_0_X_PAD[LOGICAL]'].split('[')[0])
        fused='MatmulReduceScatterAsync' in r['OP CODE']
        expected='BFLOAT16' if width==32 or fused else policy['prefill_ccl_dtype'].upper()
        assert r['INPUT_0_DATATYPE']==expected,(r['OP CODE'],width,r['INPUT_0_DATATYPE'])
        if not fused:assert f"'num_workers_per_link': '{policy['prefill_ccl_workers']}'" in r['ATTRIBUTES']
        ccl_contracts.append(dict(operation=r['OP CODE'],input_width=width,input_dtype=r['INPUT_0_DATATYPE'],output_memory=r['OUTPUT_0_MEMORY'],attributes=r['ATTRIBUTES']))
    assert len(ccl_contracts)==6
    assert sum(x['operation']=='AllGatherAsyncDeviceOperation' and x['input_width']==1280 for x in ccl_contracts)==2
    result[run['layer_type']]={'all_four_devices_present':True,'four_packed_projection_roles_verified':roles,'dtype_fidelity_and_reader_policy_verified':True,'prefill_projection_contracts':prefill_contracts,'prefill_ccl_contracts':ccl_contracts,'device0_sequence':contracts,'per_device_accounting':per_device,'source':str(root/'decode_ops.csv'),'policy':policy}
(base/'runtime_contract.json').write_text(json.dumps(result,indent=2)+'\n')
