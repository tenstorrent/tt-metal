# SPDX-License-Identifier: Apache-2.0
"""Regenerate a compact policy/correctness/timing table from actual screen artifacts."""
import csv
import json
import math
from pathlib import Path
base=Path(__file__).resolve().parents[1]/'doc/optimized_decoder'
rows=[]
for p in sorted(base.glob('*.json')):
 d=json.loads(p.read_text())
 if 'screen_host_traced_us' not in d:continue
 policy=d.get('policy')
 if policy is None and p.with_suffix('.policy.json').exists():policy=json.loads(p.with_suffix('.policy.json').read_text())
 vals=[d.get(k) for k in ('prefill_pcc','decode_pcc','changed_input_decode_pcc')]
 rows.append({'artifact':p.name,'layer_type':d['layer_type'],'sequence':d['sequence'],'batch':d['batch'],
  'activations':d.get('activations','synthetic'),'experimental_override':json.dumps(d.get('experimental_override')),'policy':json.dumps(policy,sort_keys=True),
  'prefill_pcc':vals[0],'decode_pcc':vals[1],'next_decode_pcc':vals[2],
  'host_traced_us':d['screen_host_traced_us'],'prefill_host_us':d.get('prefill_host_us'),
  'correct':all(v is not None and math.isfinite(v) and v>=.995 for v in vals) and d['deterministic']})
with (base/'screen_results.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
print(len(rows),'screen rows; these are host timing, not device headline measurements')
