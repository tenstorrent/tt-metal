# SPDX-License-Identifier: Apache-2.0
"""Compact prefill family evidence; candidate host trace timing is not device time."""
import csv
import json
from pathlib import Path
b=Path(__file__).resolve().parents[1]/'doc/optimized_multichip_decoder'
rows=[]
for path in sorted(b.glob('prefill_*_results.json')):
    for r in json.loads(path.read_text()):
        label=path.name.removesuffix('_results.json')+'_'+r['candidate']+'_l'+str(r['layer'])
        artifact=b/(label+'.json');log=b/(label+'.log')
        record=json.loads(artifact.read_text()) if artifact.exists() else {}
        failure=''
        if r['exit_code'] and log.exists():
            lines=log.read_text(errors='replace').splitlines()
            failure=next((x.strip() for x in lines if 'critical' in x and 'TT_THROW' in x),'')
            if not failure:failure=next((x.strip() for x in lines if x.startswith(('RuntimeError:','TypeError:','AssertionError:'))),'No result; inspect preserved log/triage')
        rows.append(dict(case=label,layer_type='linear_attention' if r['layer']==0 else 'full_attention',
            exit_code=r['exit_code'],prefill_trace_host_us=r.get('prefill_host_us'),decode_trace_host_us=r.get('host_us'),minimum_pcc=r.get('minimum_pcc'),
            policy=json.dumps(r['policy'],sort_keys=True),adapter_sha256=next((record.get('provenance',{}).get(name) for name in ('tests/run_prefill_rope_candidate.py','tests/run_prefill_rs_grid_candidate.py','tests/run_prefill_shared_gate_candidate.py','tests/run_prefill_projection_candidate.py','tests/run_prefill_column_padded_candidate.py','tests/run_prefill_column_interleaved_candidate.py','tests/run_prefill_column_candidate.py','tests/run_prefill_ccl_candidate.py') if record.get('provenance',{}).get(name)),''),
            adapter_provenance=json.dumps({k:v for k,v in record.get('provenance',{}).items() if k.startswith('tests/run_prefill_')},sort_keys=True),
            column_weight_padding=json.dumps(record.get('column_weight_padding',[]),sort_keys=True),
            evidence=artifact.name if artifact.exists() else log.name,failure=failure))
with (b/'prefill_ccl_candidates.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(len(rows),'prefill candidates; host trace timings only; final device profiles reported separately')
