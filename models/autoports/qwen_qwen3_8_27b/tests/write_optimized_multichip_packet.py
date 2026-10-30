# SPDX-License-Identifier: Apache-2.0
"""Compact local stage packet. Never substitute host latency for device time."""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--template',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--commit')
a=p.parse_args()
b=Path(__file__).resolve().parents[1]/'doc/optimized_multichip_decoder'
packet=json.loads(a.template.read_text())
assert packet['stage']=='05-optimized-multichip-decoder' and packet['kind']=='layer'
assert packet['workload']==dict(input_tokens=4096,output_tokens=128,batch=1,concurrency=1)
packet['selected_mesh']=dict(shape=[1,4],devices=4,architecture='Blackhole',sku='p300c',physical_cards=2)
packet['accuracy']=[]
packet['layer_types']={}
packet['performance']={}
packet['missing_reasons']={'accuracy.dataset_samples':'Recorded-activation layer PCC probes have no benchmark-dataset population; counts denote probe sequences/steps.'}
fields=('prefill_device_us','decode_device_us','prefill_flops_pct','decode_dram_pct',
        'prefill_useful_flops','decode_dram_bytes','peak_flops_per_s','peak_dram_bytes_per_s','peak_basis','source','evidence')
performance=json.loads((b/'performance.json').read_text()) if (b/'performance.json').exists() else {}
for layer,kind in ((0,'linear_attention'),(3,'full_attention')):
    path=b/f'final_accepted_l{layer}.json'
    if not path.exists():
        packet['missing_reasons'][kind]='Final-default target workload has not completed.'
        continue
    run=json.loads(path.read_text())
    assert (run['seq'],run['steps'],run['batch'])==(4096,128,1)
    current_sha = hashlib.sha256((b.parents[1]/'tt/multichip_decoder.py').read_bytes()).hexdigest()
    if any(hashlib.sha256((b.parents[1]/name).read_bytes()).hexdigest() != digest
           for name,digest in run['provenance'].items()):
        packet['missing_reasons'][kind] = 'Prior acceptance source differs from the current default; fresh acceptance pending.'
        continue
    if kind in performance and performance[kind].get('decoder_sha256') == current_sha:
        packet['layer_types'][kind]={k:performance[kind][k] for k in fields}
    else:
        packet['layer_types'][kind]={'prefill_device_us':None,'decode_device_us':None}
        packet['missing_reasons'][kind+'.performance']='Final-default whole-layer device profile pending; host timing is deliberately excluded.'
    for name,value,phase,count in (
        ('prefill_output_pcc',run['prefill_pcc'],'prefill',1),
        ('minimum_decode_output_pcc',min(run['decode_pcc']),'decode',len(run['decode_pcc']))):
        packet['accuracy'].append(dict(name=name,value=value,unit='PCC',phase=phase,layer_type=kind,
            completed_samples=count,dataset_samples=None,scope='unknown',evidence=path.name))
review=b/'stage_review.md'
clean=review.exists() and 'Verdict: clean-pass' in review.read_text()
packet['notes']=[
    'TP4 single decoder layers with recorded hidden activations; no full-model generation or serving claim.',
    'PCC compares the accepted single-chip TTNN reference; sample counts denote one prefill sequence and128 advancing decode steps, not a dataset.',
    'Rooflines use complete layer device envelopes including gaps; useful dense work, BFP4/LoFi weights, BFP8 KV, modeled compulsory DRAM traffic excluding spills/CCL. Not hardware utilization counters.',
    'Independent clean-pass recorded.' if clean else 'Acceptance/review in progress; this packet does not declare stage completion.'
]
if a.commit:packet['notes'].append('Local stage commit '+a.commit+'; never pushed.')
if not clean:packet['missing_reasons']['stage_completion']='Final acceptance and independent clean-pass are required.'
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(packet,indent=2,allow_nan=False)+'\n')
