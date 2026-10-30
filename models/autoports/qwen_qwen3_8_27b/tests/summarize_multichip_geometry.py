# SPDX-License-Identifier: Apache-2.0
"""Per-role search tables with paired complete-layer timings and final-role shares."""
import csv
import json
import math
from pathlib import Path

b=Path(__file__).resolve().parents[1]/'doc/optimized_multichip_decoder'
selected=json.loads((b/'selected_overrides.json').read_text())
runtime=json.loads((b/'runtime_contract.json').read_text()) if (b/'runtime_contract.json').exists() else {}
families=('geometry','final_geometry','geometry_confirm','large_k_final','multishard_k','readers_integrated','reader_padding')
def evidence_path(name):
    path=b/name
    if path.exists():return path
    archived=list(b.glob('integration*/'+name))
    assert len(archived)<=1, (name,archived)
    return archived[0] if archived else path

rows=[]
for family in families:
    trials=json.loads((b/f'{family}_results.json').read_text())
    for layer in (0,3):
        local=[r for r in trials if r['layer']==layer]
        control=local[0]
        assert 'control' in control['candidate']
        cp=Path(control['command'][control['command'].index('--output')+1]).name
        control_run=json.loads(evidence_path(cp).read_text())
        original=json.loads((b/f'before_policy_l{layer}.json').read_text())
        control_policy=control_run['multichip'].get('effective_policy',{**original,**control_run['policy']})
        control_us=control_run['multichip']['median_decode_host_us']
        for r in local:
            file=Path(r['command'][r['command'].index('--output')+1]).name
            path=evidence_path(file)
            run=json.loads(path.read_text()) if path.exists() else None
            target=path if path.exists() else evidence_path(Path(file).with_suffix('.log').name)
            assert target.exists(),target
            policy=run['multichip'].get('effective_policy',{**original,**run['policy']}) if run else {**control_policy,**r['policy']}
            for role in ('mix','out','gateup','down'):
                cfg=policy[role]
                if r is not control and cfg==control_policy[role]:continue
                kind='linear_attention' if layer==0 else 'full_attention'
                pct=next((x['whole_layer_pct'] for x in runtime.get(kind,{}).get('device0_sequence',[]) if x.get('role')==role),None)
                scores=([run['prefill_pcc'],*run['decode_pcc'],*run.get('state_pcc',{}).values()] if run else [])
                us=run['multichip']['median_decode_host_us'] if run else None
                final=selected[role]
                same=all(cfg.get(k,default)==final.get(k,default) for k,default in [('cores',8),('block',4),('readers',1)]) and math.lcm(cfg.get('readers',1),cfg.get('weight_pad_readers',cfg.get('readers',1)))==math.lcm(final['readers'],final.get('weight_pad_readers',final['readers']))
                if r['exit_code']:decision='rejected: failed numerical or physical-admission gate; exact reason in candidate_summary.csv'
                elif r is control:decision='paired family control; comparison baseline'
                elif same:decision='selected role geometry; final combined default independently measured'
                elif us>=control_us:decision='rejected: slower complete layer than paired family control'
                else:decision='improved family control, then superseded by later integrated geometry/reader comparison'
                rows.append(dict(family=family,layer_type=kind,role=role,final_role_whole_layer_pct=pct,
                    candidate=r['candidate'],config=json.dumps(cfg,sort_keys=True),
                    weight_dtype=policy.get(('mlp' if role=='gateup' else 'down' if role=='down' else 'attention')+'_dtype'),
                    fidelity=cfg.get('fidelity',policy.get('fidelity')),paired_control=control['candidate'],
                    paired_control_traced_host_us=control_us,traced_whole_layer_host_us=us,
                    minimum_pcc=min(scores) if scores and all(math.isfinite(x) for x in scores) else None,
                    exit_code=r['exit_code'],decision=decision,evidence=str(target.relative_to(b))))
with (b/'matmul_search.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(len(rows),'per-role search rows')
