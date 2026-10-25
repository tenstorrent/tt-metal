# SPDX-License-Identifier: Apache-2.0
"""Draw host submission intervals, not device profiling, from serving events."""
import json
from pathlib import Path
root=Path('models/autoports/qwen_qwen3_8_27b/doc/optimized_vllm')
events=[json.loads(line) for line in (root/'after_b32_events.jsonl').read_text().splitlines()]
work=[e for e in events if e['event'] not in ('prepared','shutdown_release')]
start=work[0]['time'];duration=work[-1]['time']-start
scale=1000/duration
svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="260" viewBox="0 0 1200 260">',
     '<rect width="1200" height="260" fill="white"/>',
     '<g font-family="sans-serif" font-size="14" fill="#17212b">',
     '<text x="20" y="25">B32 mixed traffic: host submissions and prefill blocking intervals</text>',
     '<text x="20" y="50">Device activity is not measured; no profiler collection.</text>',
     '<text x="20" y="90">Prefill</text><text x="20" y="135">Decode</text>']
intervals=[]
for event in work:
    x=150+(event['time']-start)*scale
    if event['event']=='prefill_complete':
        seconds=event['duration_s'];left=x-seconds*scale
        svg.append(f'<rect x="{left:.2f}" y="70" width="{max(.5,seconds*scale):.2f}" height="25" fill="#356ab6"/>')
        intervals.append(seconds)
    elif event['event']=='decode_submit':
        svg.append(f'<path d="M{x:.2f},112 v25" stroke="#26845b" stroke-width="1"/>')
for sec in range(0,int(duration)+1,25):
    x=150+sec*scale
    svg.append(f'<text x="{x:.2f}" y="165">{sec}s</text>')
svg.extend(['<text x="20" y="200">Blue: eager prefill host intervals; green: nonblocking decode submissions. Gaps include HTTP/scheduler waits.</text>',
            '<text x="20" y="225">Memory: unchanged prepared pool; prefix hits 131072 / 132160 prior tokens. Detailed checkpoints in prepared event.</text>',
            '</g></svg>'])
(root/'mixed_traffic_timeline.svg').write_text('\n'.join(svg)+'\n')
(root/'timeline_summary.json').write_text(json.dumps(dict(source='after_b32_events.jsonl',
    measurement='host event timestamps, not device time',duration_s=duration,
    prefill_intervals=len(intervals),sum_prefill_host_s=sum(intervals),max_prefill_host_s=max(intervals)),indent=2)+'\n')
