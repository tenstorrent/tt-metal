# SPDX-License-Identifier: Apache-2.0
"""Reconcile full-model candidate rows and draw measured Pareto frontiers."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--selected');a=p.parse_args()
root=Path(__file__).resolve().parents[1]/'doc/datatype_sweep'
rows=[]
for folder in ('results','failures'):
 for path in sorted((root/folder).glob('*.json')):
  row=json.loads(path.read_text())
  if 'dtype_policy' not in row:continue  # Reduced diagnostic probes are not full-model rows.
  row=dict(row,attempt_id=path.stem if folder=='failures' else 'evaluated',evidence=str(path.relative_to(root)))
  if row.get('prefill_accuracy'):
   row.update({f'prefill_{k}':min(x[k] for x in row['prefill_accuracy']) for k in ('top1','top5','top100')})
  speeds=[run['metrics']['decode_tps_per_user'] for run in row.get('runs',[])]
  if speeds: row.update(decode_tps_min=min(speeds),decode_tps_max=max(speeds))
  if row.get('command','').startswith('/workspace'):
   row['recorded_argv']=row['command']
   row['command']='docker exec qwen38-functional-build bash models/autoports/qwen_qwen3_8_27b/tests/optimized_multichip_env.sh python -m models.autoports.qwen_qwen3_8_27b.tests.run_datatype_candidate '+row['command'].split(' ',1)[1]
   row['command_provenance']='Wrapper from work_log.md; argument string preserved in recorded_argv'
  rows.append(row)
(root/'sweep_results.json').write_text(json.dumps(rows,indent=2)+'\n')
fields=['config_id','attempt_id','evidence','precision_config','dtype_policy','compute_fidelity_policy','top1','top5','top100','prefill_top1','prefill_top5','prefill_top100','ttft_ms','decode_tps','decode_tps_min','decode_tps_max','measurement_regime','trace_verified','command','hardware','mesh','status','accuracy_pass','quality_pass','selection_eligible','selection_rejection','quality_evidence']
with (root/'sweep_results.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for row in rows:
  w.writerow({k:json.dumps(row[k],sort_keys=True) if isinstance(row.get(k),(dict,list)) else row.get(k) for k in fields})
valid=[r for r in rows if r.get('trace_verified') and r.get('decode_tps') is not None]
for metric,threshold in [('top1',.90),('top5',.98)]:
 fig,ax=plt.subplots(figsize=(13,7))
 fig.subplots_adjust(left=.08,right=.68,bottom=.13,top=.86)
 fig.set_facecolor('#f8fafc');ax.set_facecolor('#f8fafc')
 frontier=[r for r in valid if not any(s[metric]>=r[metric] and s['decode_tps']>=r['decode_tps'] and
              (s[metric]>r[metric] or s['decode_tps']>r['decode_tps']) for s in valid)]
 frontier.sort(key=lambda r:r[metric])
 ax.plot([100*r[metric] for r in frontier],[r['decode_tps'] for r in frontier],color='#0f766e',marker='D',markersize=14,markerfacecolor='none',markeredgewidth=1.8,alpha=.8,label='Pareto frontier')
 for i,r in enumerate(sorted(valid,key=lambda row:row['decode_tps'])):
  selected=r['config_id']==a.selected
  ax.scatter(100*r[metric],r['decode_tps'],s=115 if selected else 65,c='#dc2626' if selected else '#2563eb',zorder=6 if selected else 4)
  ax.annotate(r['config_id']+(' (accuracy failed)' if not r.get('accuracy_pass',False) else ' (quality failed)' if r.get('quality_pass') is False else ' (selected)' if selected else ''),(100*r[metric],r['decode_tps']),
              xytext=(1.04,.04+.92*i/max(1,len(valid)-1)),textcoords=ax.transAxes,fontsize=8,
              color='#dc2626' if selected else '#334155',
              arrowprops=dict(arrowstyle='-',color='#94a3b8',linewidth=.6),annotation_clip=False)
 ax.axvline(100*threshold,color='#475569',linestyle=':',label=f'Minimum {threshold:.0%}')
 ax.set(xlabel=f'{metric.replace("top","Top-")} agreement (%)',ylabel='Traced teacher-forcing decode tokens/s/user',
        title='Qwen3.8-27B · full 64-layer precision sweep\nAIME24 chat · 203 input / 100 generated · B1 · TP4 Blackhole')
 ax.set_xlim(min([100*threshold,*[100*r[metric] for r in valid]])-1,101.5)
 ax.grid(alpha=.15);ax.legend(loc='lower left');fig.savefig(root/f'{metric}_perf_pareto.png',dpi=180);plt.close(fig)
print('Reconciled',len(rows),'candidate rows;',len(valid),'trace-verified points')
