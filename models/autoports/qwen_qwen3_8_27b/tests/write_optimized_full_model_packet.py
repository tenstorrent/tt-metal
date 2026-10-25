# SPDX-License-Identifier: Apache-2.0
"""Write compact telemetry from existing final evidence; never run hardware."""
import argparse
import json
import statistics
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--template',required=True,type=Path)
parser.add_argument('--output',required=True,type=Path)
parser.add_argument('--commit')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
evidence=root/'doc/optimized_full_model'
packet=json.loads(args.template.read_text())
assert packet['stage']=='07-optimized-full-model'
assert packet['workload']==dict(input_tokens=4096,output_tokens=128,batch=1,concurrency=1)
packet['selected_mesh']=[1,4]
packet['performance']={'source':'autoregressive'}
packet['accuracy']=[]
packet['missing_reasons']={}
path=evidence/'benchmark_b1_perf.json'
if path.exists():
    data=json.loads(path.read_text())
    rows=[r for r in data['records'] if r.get('source')=='autoregressive' and
          r.get('input_tokens')==[4096] and r.get('output_tokens')==128 and
          r.get('batch')==1 and r.get('concurrency')==1]
    if rows:
        packet['performance'].update(ttft_ms=statistics.median(r['ttft_ms'] for r in rows),
            decode_tps=statistics.median(r['decode_tps_per_user'] for r in rows),
            evidence='doc/optimized_full_model/benchmark_b1_perf.json; all64 layers, warmed4096/128,B1,one request; median of '+str(len(rows))+' repeats')
if 'ttft_ms' not in packet['performance']:
    packet['missing_reasons']['performance']='No completed matching final4096/128,B1,one-request autoregressive measurement.'
for key in ('prefill_device_us','decode_device_us','prefill_flops_pct','decode_dram_pct'):
    packet['performance'][key]=None
packet['missing_reasons']['device_time_and_rooflines']='Only reduced layers0/3 plus terminal were profiled; no all64-layer device time. Host TTFT/token-out timings are not device times or device rooflines.'
for phase,filename in [('prefill','prefill_accuracy.json'),('teacher_forcing_decode','teacher_forcing_accuracy.json')]:
    path=evidence/filename
    if not path.exists():
        packet['missing_reasons'][phase]='Final accuracy artifact unavailable.'
        continue
    for row in json.loads(path.read_text()):
        for metric in ('top1','top5','top100'):
            packet['accuracy'].append(dict(name='AIME24 chat-template '+metric,value=row[metric],unit='fraction',phase=phase,
                completed_samples=1,dataset_samples=None,scope='subset',
                evidence='doc/optimized_full_model/'+filename+'; one AIME24 problem, '+str(row.get('total'))+' compared reference token positions'))
packet['missing_reasons']['accuracy_dataset_samples']='Reference covers100 generated token positions from one AIME24 problem; dataset-wide sample count not measured by this stage.'
packet['notes']=['TP4 on four Blackhole p300c ASICs; native262144 context preserved.',
    'Teacher-forcing timings and reduced profiler values excluded from headline performance.',
    'Fixed128-token throughput disables EOS; EOS latency controls are separate. No vLLM work.']
if args.commit:
    packet['notes'].append('Local stage checkpoint: '+args.commit)
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(packet,indent=2,allow_nan=False)+'\n')
print(args.output)
