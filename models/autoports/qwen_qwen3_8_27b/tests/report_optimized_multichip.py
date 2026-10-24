# SPDX-License-Identifier: Apache-2.0
"""Whole-layer device rooflines, with all participating ASICs in the peak."""
import json
import csv
from pathlib import Path
root=Path(__file__).resolve().parents[1]
base=root/'doc/optimized_multichip_decoder'
baseline=json.loads((root/'doc/optimized_decoder/performance.json').read_text())
results={}
for layer,kind,mix_n in [(0,'linear_attention',4608),(3,'full_attention',3584)]:
    timing=json.loads((base/f'tracy/final_profile_l{layer}/whole_layer_timing.json').read_text())
    run=json.loads((base/f'final_profile_l{layer}.json').read_text())
    pre,dec=[timing['phases'][p]['device_us'] for p in ('prefill','decode')]
    policy=run['multichip']['effective_policy']
    logical_mix=4160 if layer==0 else 3584
    import math
    def physical_n(role,n):
        readers=policy[role]['readers']
        readers=math.lcm(readers,policy[role].get('weight_pad_readers',readers))
        return math.ceil(n/(32*8*readers))*32*8*readers
    physical_weights=4*(5120*physical_n('mix',logical_mix)+1536*physical_n('out',5120)+5120*physical_n('gateup',8704)+4352*physical_n('down',5120))*576/1024
    before=json.loads((base/f'tracy/before_profile_l{layer}/whole_layer_timing.json').read_text())
    before_pre,before_dec=[before['phases'][p]['device_us'] for p in ('prefill','decode')]
    state=2*3207168 if layer==0 else (4224+1)*2176
    traffic=physical_weights+state+4*2*5120*2
    flops=baseline[kind]['prefill_useful_flops']
    # 4096 LoFi FLOP/cycle/core: MatmulDeviceOperation performance model
    # (matmul_device_operation.cpp); physical cores/clock/bandwidth: official
    # https://docs.tenstorrent.com/aibs/blackhole/p300.html (two ASICs/card).
    peak=4*120*1.35e9*4096
    bandwidth=4*512e9
    results[kind]={'status':'selected_default_measured','decoder_sha256':run['provenance']['tt/multichip_decoder.py'],'prefill_device_us':pre,'decode_device_us':dec,
        'prefill_useful_flops':flops,'decode_dram_bytes':traffic,'peak_flops_per_s':peak,'peak_dram_bytes_per_s':bandwidth,
        'prefill_flops_pct':100*flops/(peak*pre/1e6),'decode_dram_pct':100*traffic/(bandwidth*dec/1e6),
        'peak_basis':'Four participating p300c ASICs;120 physical cores/ASIC,nominal1.35GHz,4096 LoFi FLOP/cycle/core;512GB/s/ASIC. Whole-layer maximum per-device firmware envelope includes attention, norms, CCL, movement and gaps.',
        'flops_assumptions':'Useful dense projection, logical causal attention/recurrence and scalar work; no MoE or padding counted useful. BFP4/LoFi weights, BFP8 column inputs/CCL, BF16 row inputs/residuals, FP32 sensitive recurrence. Denominator is the four-ASIC LoFi peak, not an achievable mixed-kernel ceiling.',
        'traffic_assumptions':'BFP4 tile storage576 bytes/1024 elements, actual local weight padding, compulsory decode projection reads; recurrent/history read+write or4224 prefix KV reads+one token write including BFP8 headers; replicated residual input/output. Excludes intermediate spills, rereads, CCL staging and small constants. Lower-bound estimate, not bus counters.',
        'source':'single-layer Tracy','evidence':[f'tracy/final_profile_l{layer}/whole_layer_timing.json',f'tracy/final_profile_l{layer}/prefill_perf_report.csv',f'tracy/final_profile_l{layer}/decode_perf_report.csv',f'final_profile_l{layer}.json','tests/report_optimized_multichip.py'],
        'workload':{'input_tokens':4096,'output_tokens':128,'batch':1,'concurrency':1,'scope':'single-layer supplied-hidden-activation workload','decode_measurement':'Final warmed step at context4224 after128 advancing decode steps'},
        'before_prefill_device_us':before_pre,'before_decode_device_us':before_dec,
        'prefill_speedup':before_pre/pre,'decode_speedup':before_dec/dec,
        'prefill_host_us':run['multichip']['prefill_host_us'],'decode_host_us':run['multichip']['median_decode_host_us']}
(base/'performance.json').write_text(json.dumps(results,indent=2)+'\n')
for k,r in results.items():print(k,{key:r[key] for key in ('prefill_device_us','decode_device_us','prefill_flops_pct','decode_dram_pct','prefill_speedup','decode_speedup')})

fields=["layer_type","before_prefill_device_us","prefill_device_us","before_decode_device_us","decode_device_us","prefill_speedup","decode_speedup","prefill_flops_pct","decode_dram_pct","decoder_sha256"]
with (base/"performance.csv").open("w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    for kind,row in results.items():writer.writerow({"layer_type":kind,**{k:row[k] for k in fields[1:]}})
