# SPDX-License-Identifier: Apache-2.0
"""Audit measured runtime policy and summarize kernel costs (not rooflines)."""
import csv
import json
from collections import defaultdict
from pathlib import Path
base=Path(__file__).resolve().parents[1]/'doc/multichip_decoder'
results={}
lines=['# Final device-profile inspection','',
       'Times below are per-device sums of kernel durations, averaged across four devices. They diagnose cost categories; they are not critical-path times or roofline denominators. Whole-layer firmware envelopes in performance.json include all device gaps.','',
       '| Layer | Phase | Projection | CCL | Movement | Other | Whole layer |',
       '|---|---|---:|---:|---:|---:|---:|']
for layer,kind in [(0,'linear_attention'),(3,'full_attention')]:
    root=base/f'tracy/final_profile_l{layer}'
    whole=json.loads((root/'whole_layer_timing.json').read_text())
    results[kind]={}
    for phase in ('prefill','decode'):
        rows=list(csv.DictReader((root/f'{phase}_ops.csv').open()))
        devices={r['DEVICE ID'] for r in rows};assert len(devices)==4
        grouped=defaultdict(float);ops=defaultdict(float);projection_rows=[]
        for r in rows:
            op=r['OP CODE'];t=float(r['DEVICE KERNEL DURATION [ns]'])/1000/4
            if op=='MatmulDeviceOperation' and r['INPUT_1_DATATYPE']=='BFLOAT4_B':
                assert r['MATH FIDELITY']=='LoFi'
                projection_rows.append(r);group='projection'
            elif op.startswith(('AllGather','ReduceScatter')):group='CCL'
            elif any(n in op for n in ('ShardedTo','InterleavedTo','Copy','Slice','Concat','Reshape','Transpose','Typecast','Pad','Untilize','Repeat')):group='movement'
            else:group='other'
            grouped[group]+=t;ops[op]+=t
        assert len(projection_rows)==(16 if phase=='decode' else 64),(kind,phase,len(projection_rows))
        assert any(r['INPUT_1_X_PAD[LOGICAL]'].split('[')[0]=='8704' for r in projection_rows)
        results[kind][phase]={'category_mean_sum_kernel_us':dict(grouped),
            'top_operations_mean_sum_kernel_us':sorted(ops.items(),key=lambda kv:-kv[1])[:10],
            'projection_rows':len(projection_rows),'projection_policy':'BFP4/LoFi',
            'whole_layer_device_us':whole['phases'][phase]['device_us'],
            'evidence':str(root.relative_to(base)/f'{phase}_ops.csv')}
        vals=[grouped[k] for k in ('projection','CCL','movement','other')]+[whole['phases'][phase]['device_us']]
        lines.append(f'| {kind} | {phase} | '+' | '.join(f'{v:.2f} µs' for v in vals)+' |')
(base/'profile_audit.json').write_text(json.dumps(results,indent=2)+'\n')
(base/'profile_analysis.md').write_text('\n'.join(lines)+'\n')
print('Runtime BFP4/LoFi and packed8704 projection audits passed on both phases and layer kinds.')
