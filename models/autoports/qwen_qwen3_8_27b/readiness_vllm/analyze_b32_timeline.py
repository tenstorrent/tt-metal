"""Offline host event analysis only; no device imports or runtime measurements."""
import collections
import argparse
import datetime
import json
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--prefix', default='full_b32', help='Artifact filename prefix within readiness_vllm')
args = parser.parse_args()
prefix = args.prefix
if Path(prefix).name != prefix:
    parser.error('--prefix must be a filename prefix, not a path')
s = json.loads((p/f'{prefix}_lifecycle.json').read_text())
r = json.loads((p/f'{prefix}_lifecycle.requests.json').read_text())
all_events = [json.loads(line) for line in (p/f'{prefix}_events.jsonl').read_text().splitlines()]
events = all_events[slice(*s['event_line_offsets'])]
if not events:
    raise ValueError('Lifecycle event slice is empty; cannot draw an observed timeline')
t0 = events[0]['time']
spans, pending = [], None
for e in events:
    if e['event'] == 'prefill_submit':
        pending = e
    elif e['event'] == 'prefill_complete':
        spans.append(dict(start_s=e['time']-e['duration_s']-t0,
                          duration_s=e['duration_s'], rows=len(pending['slots']),
                          logical_tokens=sum(b-a for a,b in zip(pending['starts'],pending['ends'])),
                          starts=pending['starts'], ends=pending['ends'],
                          active_decode_slots=pending['active_decode_slots']))
queue = []
server_log = p/f'{prefix}_server.log'
year = datetime.datetime.fromtimestamp(t0, datetime.timezone.utc).year
for line in (server_log.read_text().splitlines() if server_log.exists() else []):
    m = re.search(r'(\d\d-\d\d) (\d\d:\d\d:\d\d).*Running: (\d+) reqs, Waiting: (\d+) reqs',line)
    if m:
        stamp = datetime.datetime.fromisoformat(f'{year}-{m[1]}T{m[2]}').replace(tzinfo=datetime.timezone.utc).timestamp()
        if t0 <= stamp <= events[-1]['time']:
            queue.append(dict(time_s=stamp-t0,running=int(m[3]),waiting=int(m[4])))
dec = [e['time']-t0 for e in events if e['event']=='decode_submit']
phase_data = {}
phase_offsets = s.get('agent_turns', {}).get('event_line_offsets', [])
for i, bounds in enumerate(phase_offsets):
    subset = all_events[slice(*bounds)]
    if not subset:
        phase_data[str(i)] = dict(missing_reason='Empty phase event slice')
        continue
    phase_data[str(i)] = dict(wall_s=subset[-1]['time']-subset[0]['time'],
        prefill_host_s=sum(e.get('duration_s',0) for e in subset if e['event']=='prefill_complete'),
        prefill_calls=sum(e['event']=='prefill_submit' for e in subset),
        chunks=sum(e['event']=='prefill_chunk' for e in subset),
        decode_submissions=sum(e['event']=='decode_submit' for e in subset))
summary = dict(basis='host wall timestamps; NOT device profiling', artifact_prefix=prefix,
    event_offsets=s['event_line_offsets'],
    host_prefill_spans=spans, decode_submission_times_s=dec, queue_samples=queue,
    phase_data=phase_data, prefill_host_total_s=sum(x['duration_s'] for x in spans),
    max_prefill_host_s=max((x['duration_s'] for x in spans), default=None),
    prefill_shape_counts={str(k):v for k,v in collections.Counter((x['rows'],x['logical_tokens']) for x in spans).items()},
    captures=sorted(set(e['counters']['captures'] for e in events)),
    retirements=sorted(set(e['retirement_count'] for e in events)),
    max_waiting=max((x['waiting'] for x in queue), default=None),
    queue_missing_reason=None if queue else 'No matching running/waiting log samples in lifecycle interval',
    largest_sse_pause_ms=max((g for req in r for g in req['sse_inter_chunk_ms']), default=None),
    agent_turns=s.get('agent_turns'),
    prepared_allocation=next(e['allocation'] for e in reversed(all_events[:s['event_line_offsets'][0]])
                             if e['event']=='prepared' and e['pid']==events[0]['pid']))
(p/f'{prefix}_host_timeline.json').write_text(json.dumps(summary,indent=2)+'\n')
fig, axes = plt.subplots(4,1,figsize=(12,7),sharex=True,gridspec_kw={'height_ratios':[2,1,1,1]})
for x in spans:
    axes[0].broken_barh([(x['start_s'],x['duration_s'])],(0,1),facecolors='#cc8033',alpha=.7)
axes[0].set_ylabel('Prefill host\nphase spans')
axes[0].set_yticks([])
axes[1].eventplot(dec,lineoffsets=0,linelengths=1,color='#386cb0')
axes[1].set_ylabel('Decode\nsubmissions'); axes[1].set_yticks([])
if queue:
    for name in ('running','waiting'):
        axes[2].step([x['time_s'] for x in queue],[x[name] for x in queue],where='post',label=name)
    axes[2].legend(loc='upper right')
else:
    axes[2].text(.5,.5,'Queue observations unavailable',transform=axes[2].transAxes,ha='center')
axes[2].set_ylabel('Requests')
for key, values in [('captures',[e['counters']['captures'] for e in events]),
                    ('retirements',[e['retirement_count'] for e in events])]:
    axes[3].step([e['time']-t0 for e in events],values,label=key)
axes[3].set_ylabel('Trace count'); axes[3].legend(loc='upper right')
axes[3].set_xlabel('Seconds since first lifecycle event (host wall clock)')
for ax in axes:
    ax.grid(axis='x',alpha=.2)
    for i, bounds in enumerate(phase_offsets):
        x = all_events[bounds[0]]['time']-t0
        ax.axvline(x,ls=':',color='black',alpha=.4)
turns = s.get('agent_turns') or {}
axes[0].set_title(f"{prefix}: observed prefill phases, decode submissions and trace counts\n"
                 f"Synthetic mixed shapes plus {s['max_concurrency']} × {turns.get('prompt_tokens', 'unknown')}-token turns; not device time")
fig.tight_layout()
fig.savefig(p/f'{prefix}_host_timeline.png',dpi=160)
print(json.dumps({k:v for k,v in summary.items() if k not in ('host_prefill_spans','decode_submission_times_s','queue_samples','agent_turns','prepared_allocation')}))
