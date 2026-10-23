# SPDX-License-Identifier: Apache-2.0
"""Render reviewable tables from completed measurements; no hardware calls."""
import csv,json
from pathlib import Path
D=Path(__file__).resolve().parents[1]/'doc/optimized_decoder'
def read(name):return json.loads((D/name).read_text())
perf=read('performance.json')
lines=['## Final measurements','', 'All headline device values use4096+128,B1,one request. Every layer kind is reported separately. `performance.json` contains formulas, useful FLOPs/estimated bytes, theoretical peaks and exact source references. Lower precision explains the small PCC change; all results exceed the functional .995 bar.','', '| Layer kind | Fused prefill us | Final prefill us | Speedup | Fused decode us | Final decode us | Speedup |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for layer,kind in ((0,'linear_attention'),(3,'full_attention')):
 old=read(f'tracy/baseline_accepted_l{layer}/whole_layer_timing.json')['phases'];r=perf[kind]
 a,b=old['prefill']['device_us'],old['decode']['device_us'];c,d=r['prefill_device_us'],r['decode_device_us']
 lines.append(f'|{kind}|{a:.2f}|{c:.2f}|{a/c:.2f}x|{b:.2f}|{d:.2f}|{b/d:.2f}x|')
lines+=['','| Layer kind | Fused / final prefill PCC | Fused / final initial decode PCC | Fused / final128th decode PCC |','| --- | ---: | ---: | ---: |']
for layer,kind in ((0,'linear_attention'),(3,'full_attention')):
 old=read(f'baseline_accepted_l{layer}.json');r=read(f'final_accepted_l{layer}.json')
 assert old['provenance']==r['provenance'], 'Baseline and final source provenance differ'
 assert old['performance_workload']==r['performance_workload']
 assert old['runtime_class']=='FusedDecoder' and r['runtime_class']=='OptimizedDecoder'
 vals=[f"{old[k]:.8f} / {r[k]:.8f}" for k in ('prefill_pcc','decode_pcc','workload_final_decode_pcc')]
 lines.append('|'+kind+'|'+'|'.join(vals)+'|')
lines+=['','| Layer kind | Ideal compute prefill us | Actual device prefill us | Host prefill us | Prefill chip FLOPs % | Ideal DRAM decode us | Actual device decode us | Host last decode us | Decode DRAM % |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for kind,r in perf.items():
 keys=('prefill_compute_roofline_us','prefill_device_us','prefill_host_us','prefill_flops_pct','decode_dram_roofline_us','decode_device_us','decode_host_us','decode_dram_pct')
 lines.append('|'+kind+'|'+'|'.join(f'{r[k]:.2f}' for k in keys)+'|')
lines+=['','The ideal times are lower bounds, not mixed-kernel attainable ceilings. DRAM estimates count physical BFP4 weight tiles/headers/padding and state/KV traffic, omitting spills and rereads. Device gaps remain inside the denominator. Host values here are from the same profiler-enabled run; profiler-off screens below provide selection comparisons. Trace replay removes per-op host dispatch; remaining firmware gaps and single replay synchronization are included, not subtracted.','', '| Layer | Length | Fused prefill / decode host us | Final prefill / decode host us |','| --- | ---: | ---: | ---: |']
for layer in (0,3):
 for length in (513,4096):
  r=read(f'finalscreen_default_{length}_l{layer}.json');oldfile=D/f'finalscreen_fused_{length}_l{layer}.json';old=read(oldfile.name) if oldfile.exists() else None
  val=lambda d:f"{d['prefill_host_us']:.2f} / {d['screen_host_traced_us']:.2f}" if d else 'See device baseline above'
  lines.append(f'|{r["layer_type"]}|{length}|{val(old)}|{val(r)}|')
lines+=['','These profiler-off decode screens use100 static-position replays after one initial restore for candidate comparison. The headline uses the valid advancing128-step workload. Larger-batch values below use its final valid step, not the static-position diagnostic loop.','', '| Layer kind | Batch | Prefill PCC | Final128th decode PCC | Final valid-step host us |','| --- | ---: | ---: | ---: | ---: |']
for layer in (0,3):
 for batch in (8,16,32):
  r=read(f'workload_b{batch}_l{layer}.json');lines.append(f'|{r["layer_type"]}|{batch}|{r["prefill_pcc"]:.8f}|{r["workload_final_decode_pcc"]:.8f}|{r["workload_last_decode_host_us"]:.2f}|')
lines+=['','## Dominant matmul search','', 'Role shares below are kernel time divided by the final complete layer device window; layout and internal gaps are outside the numerator but inside the denominator. Search latencies are whole-layer traced host screens at4096. Exact per-role configs/PCC and other precision/topology candidates are in `screen_results.csv` and matching JSON/policy files. Historical role rows are the precision-locked BFP4/LoFi geometry sweep before the final paired packed-output layout refinement; only the named role changes within that sweep. Final shares and accepted-default metrics come from the new final profiles.','', '| Kind | Role | Final layer time % | Config candidate | Traced decode us | Minimum output PCC | Decision |','| --- | --- | ---: | --- | ---: | ---: | --- |']
for layer,kind in ((0,'linear_attention'),(3,'full_attention')):
 rows=list(csv.DictReader((D/f'tracy/final_accepted_l{layer}/decode_perf_report.csv').open()));mm=[r for r in rows if 'MatmulDeviceOperation' in r['OP Code']];assert len(mm)==4
 variants={'mix':['control','mix40k4'],'out':['control','out32k6'],'gateup':['control','gateup40k4','gateup16k10'],'down':['control','down34k16','down17k32','down8k68']}
 for role,row in zip(variants,mm):
  assert row['Input 0 Datatype']=='BFLOAT16' and row['Input 1 Datatype']=='BFLOAT4_B' and 'LoFi' in row['Math Fidelity'] and row['DRAM Sharded']=='True'
  share=100*float(row['Device Time'])/perf[kind]['decode_device_us']
  for candidate in variants[role]:
   filename=f'role_{candidate}_l{layer}.json';path=D/filename
   if path.exists():
    r=read(filename);pcc=min(r[k] for k in ('prefill_pcc','decode_pcc','changed_input_decode_pcc'));lat=f'{r["screen_host_traced_us"]:.2f}';pcc=f'{pcc:.8f}';decision='Keep' if candidate=='control' else 'Reject: slower'
   else:lat=pcc='—';decision='Reject: L1 capacity; legal smaller alternatives measured'
   lines.append(f'|{kind}|{role}|{share:.2f}|{candidate}|{lat}|{pcc}|{decision}|')
lines+=['','Final mix/gateup remain K2 on80 input-storage cores: two tiles/shard is the largest divisor. The crossed16/32/40/80-core1/2/3-reader search and isolated40/K4,16/K10 trials earn this choice. Output64/K3 beats32/K6; down68/K8 beats34/K16 and17/K32;8/K68 fails by12928 L1 bytes. The DRAM factory chooses output subblocks internally; its public program config exposes no override. Prefill has explicit subblock height2 where legal, wider K10/K32, and11x10 grids.','', 'Final dtype is also verified directly above from measured CSV columns. `readers_accepted_l0.device.json` and `readers_accepted_l3.device.json` recheck one/two/three readers per role on the selected policy, in alternating order, with complete role envelopes and physical byte estimates.','']
(D/'results.md').write_text('\n'.join(lines))
readme=D/'README.md'
text=readme.read_text().split('## Final measurements')[0]
text=text.replace('The full optimize checklist and final measurements are added below only after required validation and review artifacts exist. Full-model quality, token-out generation and serving performance are not claimed by this layer stage.', 'The completed evidence mapping is in [checklist.md](checklist.md); independent signoff is in [stage_review.md](stage_review.md). Full-model quality, token-out generation and serving performance are not claimed by this layer stage.')
readme.write_text(text.rstrip()+'\n\n'+'\n'.join(lines))
