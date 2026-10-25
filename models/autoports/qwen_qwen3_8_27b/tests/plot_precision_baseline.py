# SPDX-License-Identifier: Apache-2.0
"""Read-only view of prior full-model host phases and memory checkpoints."""
import json
import statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]/'doc'
fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
for i,b in enumerate([1,8,16,32]):
    d=json.loads((root/f'optimized_full_model/benchmark_b{b}_perf.json').read_text())
    rows=[r for r in d['records'] if r['input_tokens'][0]==4096]
    first=statistics.median(r['ttft_ms']/1000 for r in rows)
    decode=statistics.median(r['decode_seconds'] for r in rows)
    axes[0].barh(str(b),first,color='#14b8a6',label='TTFT' if i==0 else None)
    axes[0].barh(str(b),decode,left=first,color='#3b82f6',label='127 decode steps' if i==0 else None)
    axes[1].plot(['Before preparation','After preparation'],[d[k]['allocated_bytes_per_device']/1e9 for k in ['memory_before_preparation','memory_after_preparation']],marker='o',label=f'B{b}')
axes[0].set(xlabel='End-to-end host seconds (not device activity)',ylabel='Batch',title='Stage07 warmed 4096/128 request phases');axes[0].legend()
axes[1].set(ylabel='Allocated GB/device',title='Stage07 persistent memory checkpoints');axes[1].legend()
fig.savefig(root/'datatype_sweep/baseline_run_view.png',dpi=150)
