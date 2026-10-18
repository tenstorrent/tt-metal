"""Aggregate real completed runs and draw accuracy/throughput frontiers."""
import argparse
import csv
import json
import statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('models/autoports/qwen_qwen3_8_flash_next')
OUT=ROOT/'doc/datatype_sweep'

def collect():
    rows=[]
    for path in sorted(OUT.glob('*.json')):
        d=json.loads(path.read_text())
        if not isinstance(d,dict) or 'args' not in d or d.get('args',{}).get('smoke') or d.get('args',{}).get('final'):
            continue
        reps=d.get('teacher_forcing_repetitions',[])
        if d.get('status') == 'runtime_fail':
            policy=d.get('runtime',{}).get('policy')
            if policy is None:
                policy=json.loads(Path(d['args']['config']).read_text())
            rows.append(dict(config_id=d['config_id'],precision_config_path=d['args']['config'],
                dtype_policy=policy,compute_fidelity_policy=policy['compute_fidelities'],
                top1=None,top5=None,top100=None,prefill=d.get('prefill'),token_count=100,reference_path=d['reference'],
                ttft_ms=None,traced_teacher_forcing_tps=None,repetitions_tps=[],measurement_regime='full-model runtime failure; not ranked',
                command=d['command'],hardware=d['hardware'],mesh=d['mesh'],status=d['status'],trace_evidence=str(path),
                runtime_policy_evidence=str(path) if 'runtime' in d else None,commit=d['commit'],rejection_reason=d.get('error')))
            continue
        if len(reps)!=3 or d.get('status') not in ('pass','accuracy_fail'):
            continue
        assert all(r['counter_delta']['decode_replays']==99 for r in reps)
        metrics=[r['metrics'][0] for r in reps]
        row=dict(config_id=d['config_id'],precision_config_path=d['args']['config'],
                 dtype_policy=d['runtime']['policy'],compute_fidelity_policy=d['runtime']['policy']['compute_fidelities'],
                 top1=min(r['top1'] for r in metrics),top5=min(r['top5'] for r in metrics),top100=min(r['top100'] for r in metrics),
                 prefill=d['prefill'],token_count=100,reference_path=d['reference'],
                 ttft_ms=statistics.median(r['ttft_ms'] for r in metrics),
                 traced_teacher_forcing_tps=statistics.median(r['decode_t/s/u'] for r in metrics),
                 repetitions_tps=[r['decode_t/s/u'] for r in metrics],
                 measurement_regime=d['measurement_regime'],command=d['command'],hardware=d['hardware'],mesh=d['mesh'],
                 status=d['status'],trace_evidence=str(path),runtime_policy_evidence=str(path),commit=d['commit'],rejection_reason=None)
        rows.append(row)
    rejections=OUT/'rejected_candidates.json'
    if rejections.exists():
        rows.extend(json.loads(rejections.read_text()))
    (OUT/'sweep_results.json').write_text(json.dumps(rows,indent=2)+'\n')
    if rows:
        with (OUT/'sweep_results.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
            for r in rows: writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    return rows

def charts(rows,selected):
    rows=[r for r in rows if r.get('traced_teacher_forcing_tps') is not None]
    for metric,threshold in [('top1',.90),('top5',.98)]:
        fig,ax=plt.subplots(figsize=(12,7),layout='constrained')
        fig.get_layout_engine().set(rect=(0,.035,1,.965))
        fig.set_facecolor('#f6f8fc');ax.set_facecolor('#f6f8fc')
        frontier=[r for r in rows if not any(q[metric]>=r[metric] and q['traced_teacher_forcing_tps']>=r['traced_teacher_forcing_tps'] and (q[metric]>r[metric] or q['traced_teacher_forcing_tps']>r['traced_teacher_forcing_tps']) for q in rows)]
        frontier.sort(key=lambda r:r[metric])
        ax.plot([r[metric]*100 for r in frontier],[r['traced_teacher_forcing_tps'] for r in frontier],color='#4069a8',lw=2,marker='o',ms=10,alpha=.6,label='Pareto frontier')
        for i,r in enumerate(rows):
            chosen=r['config_id']==selected
            color='#db3038' if chosen else '#2d728f' if r['status']=='pass' else '#9b9fa8'
            samples=r['repetitions_tps']
            ax.plot([r[metric]*100]*2,[min(samples),max(samples)],color=color,lw=1,alpha=.45,zorder=2)
            ax.scatter(r[metric]*100,r['traced_teacher_forcing_tps'],s=125 if chosen else 60,c=color,zorder=4,edgecolors='white')
            if chosen or r['config_id']=='baseline_policy' or r in frontier:
                offset=(-12,18) if chosen else (-12,-24) if r['config_id']=='baseline_policy' else (-12,32)
                ax.annotate(r['config_id'],(r[metric]*100,r['traced_teacher_forcing_tps']),xytext=offset,textcoords='offset points',ha='right',fontsize=9,color=color,
                            arrowprops=dict(arrowstyle='-',color=color,lw=.6))
        ax.axvline(threshold*100,color='#5a6070',ls=':',lw=1.8,label=f'Minimum {metric}: {threshold:.0%}')
        ax.set(xlabel=f'Teacher-forcing {metric.replace("top","top-")} accuracy (%)',ylabel='Traced teacher-forcing decode (tokens/s/user)',title='Qwen3.8-Flash-Next · precision frontier\nAIME24 chat · 100 generated tokens · 4× Blackhole')
        ax.scatter([],[],s=60,c='#2d728f',label='Accuracy gate passed')
        ax.scatter([],[],s=60,c='#9b9fa8',label='Accuracy gate failed')
        ax.scatter([],[],s=125,c='#db3038',label='Selected')
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15);ax.legend(loc='best')
        ax.margins(x=.12,y=.20)
        fig.text(.01,.005,f'{len(rows)} evaluated full-model configurations · median of 3 traced runs · see sweep_results.csv for every point',fontsize=8,color='#5a6070')
        fig.savefig(OUT/f'{metric}_perf_pareto.png',dpi=180);plt.close(fig)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--select',action='store_true');a=ap.parse_args()
    rows=collect()
    passing=[r for r in rows if r['status']=='pass']
    if passing:
        winner=max(passing,key=lambda r:r['traced_teacher_forcing_tps'])
        if a.select:
            policy=json.loads(Path(winner['precision_config_path']).read_text())
            recorded=winner['dtype_policy']
            # Early runs predate the explicit representation of inherited QSA
            # exceptions. No numerical setting may change during selection.
            comparable=dict(policy)
            if 'qsa_execution' not in recorded:
                comparable.pop('qsa_execution')
            assert comparable==recorded, 'Candidate config changed after measurement'
            (OUT/'selected_precision_config.json').write_text(json.dumps(policy,indent=2)+'\n')
            (OUT/'selection.json').write_text(json.dumps(dict(
                selected_config_id=winner['config_id'],selection_metric='median of three uninstrumented trace-verified teacher-forcing runs',
                user_rule='fastest evaluated config satisfying both full-model readiness gates',
                noise_note='Observed ranges are reported; raw median ordering is not a statistical significance claim.',
                qsa_rule_schema_explicit='qsa_execution' not in recorded,
                selected_candidate=winner,required_default_verification='selected_default.json'),indent=2)+'\n')
        charts(rows,winner['config_id'])
        print(json.dumps({k:winner[k] for k in ('config_id','top1','top5','top100','ttft_ms','traced_teacher_forcing_tps')},indent=2))
