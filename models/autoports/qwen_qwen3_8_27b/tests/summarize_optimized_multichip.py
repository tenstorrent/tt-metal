# SPDX-License-Identifier: Apache-2.0
"""Compact candidate ledger; failed gates retain measured PCC and timing."""
import csv
import json
import math
from pathlib import Path

base=Path(__file__).resolve().parents[1]/'doc/optimized_multichip_decoder'
def evidence_path(name):
    path=base/name
    if path.exists():return path
    archived=list(base.glob('integration*/'+name))
    assert len(archived)<=1, (name,archived)
    return archived[0] if archived else path

rows=[]
for manifest in sorted(base.glob('*_results.json')):
    for result in json.loads(manifest.read_text()):
        command=result['command']
        name=Path(command[command.index('--output')+1]).name
        path=evidence_path(name)
        row={'family':manifest.stem.removesuffix('_results'),'layer':result['layer'],
             'candidate':result['candidate'],'exit_code':result['exit_code'],'evidence':name,
             'policy':json.dumps(result['policy'],sort_keys=True)}
        target=path if path.exists() else evidence_path(Path(name).with_suffix('.log').name)
        row['evidence']=str(target.relative_to(base))
        assert target.exists(),target
        if path.exists():
            value=json.loads(path.read_text())
            row.update(batch=value['batch'],seq=value['seq'],steps=value['steps'],layout=value['layout'],
                prefill_host_us=value['multichip']['prefill_host_us'],
                traced_decode_host_us=value['multichip']['median_decode_host_us'],
                prefill_pcc=value.get('prefill_pcc'),minimum_decode_pcc=min(value.get('decode_pcc',[float('nan')])),
                minimum_state_pcc=min(value.get('state_pcc',{'unknown':float('nan')}).values()),
                decoder_sha256=value['provenance']['tt/multichip_decoder.py'])
        if result['exit_code']:
            if path.exists():
                failed={k:v for k,v in {'prefill':value.get('prefill_pcc'), 'decode':min(value.get('decode_pcc',[1])), **value.get('state_pcc',{})}.items() if v is not None and (not math.isfinite(v) or v<.995)}
                row['failure_reason']='Numerical gate: '+json.dumps({k:(v if math.isfinite(v) else str(v)) for k,v in failed.items()},sort_keys=True)
            else:
                log=evidence_path(Path(name).with_suffix('.log').name)
                lines=log.read_text(errors='replace').splitlines() if log.exists() else []
                errors=[line for line in lines if any(t in line for t in ('RuntimeError:', 'TT_FATAL', 'AssertionError', 'Error Message:'))]
                row['failure_reason']=' | '.join(errors[-3:])[:1200] or 'See preserved log and recovery record'
        rows.append(row)
fields=['family','layer','candidate','batch','seq','steps','layout','exit_code','prefill_host_us','traced_decode_host_us',
        'prefill_pcc','minimum_decode_pcc','minimum_state_pcc','failure_reason','evidence','decoder_sha256','policy']
with (base/'candidate_summary.csv').open('w') as output:
    writer=csv.DictWriter(output,fieldnames=fields);writer.writeheader();writer.writerows(rows)
print(f'{len(rows)} candidate rows written')
