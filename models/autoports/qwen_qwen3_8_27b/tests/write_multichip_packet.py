# SPDX-License-Identifier: Apache-2.0
"""Compact local telemetry from accepted artifacts; never launches hardware work."""
import argparse
import json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--template',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--commit')
p.add_argument('--review-pending',action='store_true')
args=p.parse_args()
root=Path(__file__).resolve().parents[1]
base=root/'doc/multichip_decoder'
packet=json.loads(args.template.read_text())
assert packet['stage']=='04-multichip-decoder'
assert packet['workload']==dict(input_tokens=4096,output_tokens=128,batch=1,concurrency=1)
perf=json.loads((base/'performance.json').read_text())
validation=json.loads((base/'validation_summary.json').read_text())
if not args.review_pending:
    assert args.commit and 'Verdict: clean-pass' in (base/'stage_review.md').read_text()
assert all(row['decoder_sha256']==validation['decoder_sha256'] for row in perf.values())
packet['selected_mesh']={'shape':[1,4],'devices':4,'hardware':'Blackhole p300c','topology':'FABRIC_1D_RING','links_per_edge':2}
keys=['prefill_device_us','decode_device_us','prefill_flops_pct','decode_dram_pct',
      'prefill_useful_flops','decode_dram_bytes','peak_flops_per_s','peak_dram_bytes_per_s','peak_basis','source']
packet['layer_types']={kind:{**{k:row[k] for k in keys},
    'evidence':[f'models/autoports/qwen_qwen3_8_27b/doc/multichip_decoder/performance.json:{kind}',*([args.commit] if args.commit else [])]}
    for kind,row in perf.items()}
packet['accuracy']=[]
for kind,row in validation['layer_types'].items():
    for name,value,phase,count in [('prefill_tensor_pcc',row['prefill_pcc'],'prefill',1),
                                  ('minimum_decode_step_pcc',row['minimum_decode_pcc'],'decode',row['completed_decode_steps'])]:
        packet['accuracy'].append(dict(name=name,value=value,unit='PCC',phase=phase,layer_type=kind,
            completed_samples=count,dataset_samples=None,scope='unknown',evidence=[f'multichip_decoder/{row["evidence"]}']))
packet['missing_reasons']={'accuracy_dataset_samples':'PCC fixtures compare complete layer tensors/steps; there is no evaluation dataset size for these comparisons.'}
if args.review_pending: packet['missing_reasons']['completion']='Independent final review and local checkpoint pending.'
packet['notes']=[
    'Single-layer supplied-hidden-input workload,4096 prefill positions+128 advancing decode steps,B1/one request; not token-out generation.',
    'Completed samples count tensor/step comparisons against single-chip optimized TTNN; dataset size/coverage is unknown. Every compared tensor includes all its logical elements.',
    'Whole-layer device envelopes include all operations and device gaps; decode is the final warmed step at context4224. DRAM traffic is a compulsory-transfer lower-bound estimate; assumptions and precision are in performance.json.',
    'Native context262144 preserved;43 correctness/watcher cases pass. Full-model weights/KV fit is a calculated ledger, not full-model execution.',
    'Independent final acceptance/checkpoint pending.' if args.review_pending else f'Local checkpoint {args.commit}; stage_review.md clean-pass.']
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(packet,indent=2)+'\n')
print(args.output)
