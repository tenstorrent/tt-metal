# SPDX-License-Identifier: Apache-2.0
"""Verify completed stage04 evidence without opening hardware."""
import hashlib
import json
import math
from pathlib import Path
root=Path(__file__).resolve().parents[1]
base=root/'doc/multichip_decoder'
implementation=hashlib.sha256((root/'tt/multichip_decoder.py').read_bytes()).hexdigest()
summary={'decoder_sha256':implementation,'suites':{},'layer_types':{}}
for suite,count in [('batch',6),('boundary',25),('continuation',4),('stack',2),('context',2),('watcher',4)]:
    rows=json.loads((base/f'{suite}_validation.json').read_text())
    assert len(rows)==count,(suite,len(rows),count)
    metrics=[]
    for row in rows:
        assert row['exit_code']==0,row
        data=json.loads((base/(row['case']+'.json')).read_text())
        assert data['provenance']['tt/multichip_decoder.py']==implementation,row['case']
        scores=[data['prefill_pcc'],*data['decode_pcc'],*data.get('state_pcc',{}).values()]
        assert all(math.isfinite(v) and v>=.995 for v in scores),(row['case'],scores)
        assert all(data[mode]['runtime_fallback_audit'] and data[mode]['restored_replays_bitwise']==3 for mode in ('baseline','multichip'))
        metrics.append({'case':row['case'],'minimum_output_pcc':min([data['prefill_pcc'],*data['decode_pcc']]),
                        'minimum_state_pcc':min(data['state_pcc'].values()) if 'state_pcc' in data else None})
    summary['suites'][suite]={'passed_cases':len(rows),'cases':metrics}
for layer,kind in [(0,'linear_attention'),(3,'full_attention')]:
    data=json.loads((base/f'watcher_l{layer}.json').read_text())
    summary['layer_types'][kind]={'prefill_pcc':data['prefill_pcc'],'minimum_decode_pcc':min(data['decode_pcc']),
        'completed_decode_steps':len(data['decode_pcc']),'state_pcc':data['state_pcc'],
        'evidence':f'watcher_l{layer}.json'}
(base/'validation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v['passed_cases'] for k,v in summary['suites'].items()}))
