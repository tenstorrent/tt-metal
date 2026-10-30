# SPDX-License-Identifier: Apache-2.0
"""CPU-only current reduced-stack attribution; never a full-model device result."""
import csv
import hashlib
import json
import re
from pathlib import Path

root=Path(__file__).resolve().parents[1]/'doc'
out=root/'optimized_full_model'
records=[]
policies=[]
for batch,linear_start,full_start,terminal_start in [(1,7,74,118),(32,6,172,250)]:
    base=out/f'tracy/b{batch}'
    timing=json.loads((base/'reduced_stack_timing.json').read_text())
    path=base/'decode_ops.csv'
    rows=list(csv.DictReader(path.open()))
    old=list(csv.DictReader((root/f'full_model/tracy/b{batch}/decode_ops.csv').open()))
    old=[r for r in old if r['DEVICE ID']==old[0]['DEVICE ID']]
    segments={name:[] for name in ['embedding_and_rope','linear_attention','full_attention','terminal_sampling_collector']}
    by_device={}
    for device in sorted({r['DEVICE ID'] for r in rows}):
        local=[r for r in rows if r['DEVICE ID']==device]
        # Boundaries were mapped from the source's residual-return -> next norm
        # and final residual -> terminal gather. Verify every preceding opcode,
        # not just a few labels, before reusing that structural mapping.
        assert [r['OP CODE'] for r in local[:terminal_start]]==[r['OP CODE'] for r in old[:terminal_start]]
        bounds=[0,linear_start,full_start,terminal_start,len(local)]
        parts={}
        for name,first,last in zip(segments,bounds,bounds[1:]):
            window=local[first:last]
            us=(max(int(r['DEVICE FW END CYCLE']) for r in window)-min(int(r['DEVICE FW START CYCLE']) for r in window))/timing['clock_mhz']
            segments[name].append(us)
            parts[name]=dict(device_us=us,first_op_index=first,last_op_index=last-1,first_op=window[0]['OP CODE'],last_op=window[-1]['OP CODE'])
        by_device[device]=parts
    maxima={k:max(v) for k,v in segments.items()}
    stack=48*maxima['linear_attention']+16*maxima['full_attention']
    records.append(dict(batch=batch,per_device=by_device,max_device_us=maxima,
        derived_64layer_stack_us=stack,derived_stack_plus_terminal_us=stack+maxima['embedding_and_rope']+maxima['terminal_sampling_collector'],
        source=str(path.relative_to(root)),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        scope='Derived attribution estimate from two real layers at decode position4223, not all64-layer device timing. Whole-layer windows include every operation and internal gap; profile overhead/context differences remain.'))
    for phase in ('prefill','decode'):
        p=base/f'{phase}_ops.csv'
        rs=list(csv.DictReader(p.open()));rs=[r for r in rs if r['DEVICE ID']==rs[0]['DEVICE ID']]
        for index,row in enumerate(rs):
            if row['OP CODE']=='MatmulReduceScatterAsyncDeviceOperation':
                attrs=row['ATTRIBUTES']
                match=re.search(r'in0_block_w[=:](\d+)',attrs)
                assert 'LoFi' in attrs and row['INPUT_1_DATATYPE']=='BFLOAT4_B'
                policies.append(dict(batch=batch,phase=phase,op_index=index,operation=row['OP CODE'],
                    k_block_tiles=int(match.group(1)) if match else None,lofi='LoFi' in attrs,
                    weight_dtype=row.get('INPUT_1_DATATYPE'),output_dtype=row.get('OUTPUT_0_DATATYPE'),
                    source=str(p.relative_to(root)),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        ordinary=[r for r in rs if r['OP CODE']=='MatmulDeviceOperation' and r['INPUT_1_DATATYPE']=='BFLOAT4_B']
        assert all(r['MATH FIDELITY']=='LoFi' for r in ordinary)
        policies.append(dict(batch=batch,phase=phase,operation='ordinary_BFP4_matmuls',count=len(ordinary),fidelity='LoFi',source=str(p.relative_to(root))))
        if phase=='decode':
            head=next(r for r in reversed(rs) if r['OP CODE']=='MatmulDeviceOperation')
            assert 'BFLOAT16' in head['INPUT_1_DATATYPE'].upper(),head['INPUT_1_DATATYPE']
            assert 'HiFi4' in head['MATH FIDELITY'],head['MATH FIDELITY']
            policies.append(dict(batch=batch,phase=phase,operation='terminal_head',weight_dtype=head['INPUT_1_DATATYPE'],fidelity=head['MATH FIDELITY'],source=str(p.relative_to(root))))
            names=[r['OP CODE'] for r in rs]
            assert 'TopKDeviceOperation' not in names and 'TopkDeviceOperation' not in names
            assert 'TopkLargeIndicesDeviceOperation' in names and 'SamplingDeviceOperation' in names
            assert 'IndexedFillDeviceOperation' in names
(out/'reduced_decode_attribution.json').write_text(json.dumps(dict(rows=records,
    warning='Never populate full-model device-time/roofline telemetry from this reduced profile.'),indent=2)+'\n')
for batch,phase,expected in [(1,'prefill',[4,17,4,17]),(32,'decode',[6,17,24,17])]:
    actual=[r['k_block_tiles'] for r in policies if r['batch']==batch and r['phase']==phase and r['operation']=='MatmulReduceScatterAsyncDeviceOperation']
    assert actual==expected,(batch,phase,actual,expected)
(out/'kernel_policy_audit.json').write_text(json.dumps(dict(scope='Selected fused matmul/RS runtime attributes in reduced real layers0/3',records=policies),indent=2)+'\n')
print(json.dumps(records,indent=2))
