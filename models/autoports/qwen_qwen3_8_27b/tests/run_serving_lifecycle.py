#!/usr/bin/env python3
"""HTTP-only synthetic shape/trace probe; token fixtures are NOT quality evidence."""
import argparse
import concurrent.futures
import json
import pathlib
import re
import threading
import time
import urllib.request


def events(path):
    lines = pathlib.Path(path).read_text().splitlines()
    return lines, [json.loads(line) for line in lines if line.strip()]


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-url', required=True)
    parser.add_argument('--model', default='/workspace/.cache/qwen38-full')
    parser.add_argument('--max-concurrency', type=int, default=2)
    parser.add_argument('--events-path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reduced', default=None, help='Label reduced-model evidence explicitly')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--agent-turn-length', type=int, default=None)
    parser.add_argument('--skip-agent-turns', action='store_true',
                        help='Debug only: omit multi-turn pool-pressure traffic')
    args = parser.parse_args()
    if args.max_concurrency < 2:
        parser.error('--max-concurrency must be at least 2 for concurrent decode coverage')
    turn_length = args.agent_turn_length or 4130
    if turn_length < 1:
        parser.error('--agent-turn-length must be positive')
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output.with_suffix('.requests.json')
    records, failures = [], []
    agent_turns = dict(skipped=args.skip_agent_turns, prompt_tokens=turn_length,
                       appended_tokens=33, requested_concurrency=args.max_concurrency)

    def metrics():
        try:
            with urllib.request.urlopen(args.server_url.rstrip('/') + '/metrics', timeout=30) as response:
                lines = response.read().decode().splitlines()
            return dict(samples=[line for line in lines if not line.startswith('#') and
                                 any(key in line for key in ('prefix_cache', 'kv_cache_usage',
                                                            'preempt', 'cache_config_info',
                                                            'num_requests_running'))])
        except Exception as exc:
            return dict(samples=[], missing_reason=f'{type(exc).__name__}: {exc}')
    before_lines, initial = events(args.events_path)
    prepared = [event for event in initial if event.get('event') == 'prepared']
    baseline = prepared[-1] if prepared else {}
    pid = baseline.get('pid')
    signatures = baseline.get('execution_signatures', {})
    capture_count = baseline.get('counters', {}).get('captures')
    retirements = baseline.get('retirement_count')
    if not signatures or capture_count is None or retirements is None or pid is None:
        failures.append('Missing prepared PID, execution signatures, capture or retirement counters')

    # Valid ordinary vocabulary IDs, intentionally synthetic and language-neutral.
    def fixture(length, family=0):
        return [100 + (i + family * 37) % 251 for i in range(length)]

    def request(name, prompt, generated=8, sampled=False, first=None):
        started = time.perf_counter()
        record = dict(name=name, prompt_tokens=len(prompt), requested_output_tokens=generated,
                      sampling=dict(temperature=0.7 if sampled else 0.0,
                                    top_k=4 if sampled else 1, top_p=1.0),
                      chunks=[], complete=False)
        payload = dict(model=args.model, prompt=prompt, max_tokens=generated, stream=True,
                       stream_options={'include_usage': True}, ignore_eos=True,
                       **record['sampling'])
        arrival_times, text_parts = [], []
        try:
            req = urllib.request.Request(args.server_url.rstrip('/') + '/v1/completions',
                                         json.dumps(payload).encode(),
                                         {'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=1800) as response:
                for line in response:
                    line = line.decode().strip()
                    if not line.startswith('data:'):
                        continue
                    data = line[5:].strip()
                    if data == '[DONE]':
                        record['complete'] = True
                        break
                    chunk = json.loads(data)
                    elapsed = (time.perf_counter() - started) * 1000
                    record['chunks'].append(dict(elapsed_ms=elapsed, data=chunk))
                    if chunk.get('usage'):
                        record['usage'] = chunk['usage']
                    for choice in chunk.get('choices', []):
                        text = choice.get('text', '')
                        if text:
                            arrival_times.append(elapsed)
                            text_parts.append(text)
                            if first is not None:
                                first.set()
                        if choice.get('finish_reason'):
                            record['finish_reason'] = choice['finish_reason']
            record['text'] = ''.join(text_parts)
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
        finally:
            record['completion_ms'] = (time.perf_counter() - started) * 1000
            record['ttft_ms'] = arrival_times[0] if arrival_times else None
            # SSE chunks can contain multiple tokens. Do not mislabel this as token ITL.
            gaps = [b - a for a, b in zip(arrival_times, arrival_times[1:])]
            record['sse_inter_chunk_ms'] = gaps
            record['itl_ms'] = gaps if record.get('usage', {}).get('completion_tokens') == len(arrival_times) else None
            records.append(record)
        return record

    workload_started = time.perf_counter()
    try:
        if failures:
            raise RuntimeError('Cannot validate lifecycle without baseline instrumentation')
        known = {}
        for length in (137, 138, 139):
            known[length] = request(f'first_{length}', fixture(length))
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_concurrency) as pool:
            wave = [pool.submit(request, f'concurrent_slot_{i}', fixture(137 + i % 3, i + 1))
                    for i in range(args.max_concurrency)]
            for future in wave:
                future.result()
            for length in (8195, 12291):
                first = threading.Event()
                anchor = pool.submit(request, f'anchor_{length}', fixture(139, length),
                                     256, False, first)
                if not first.wait(600):
                    failures.append(f'Anchor {length} never produced a first streamed token')
                long = pool.submit(request, f'chunked_{length}', fixture(length, length), 8)
                long.result()
                anchor.result()
        for length in (139, 137, 138):
            repeated = request(f'reordered_{length}', fixture(length))
            if repeated.get('text') != known[length].get('text'):
                failures.append(f'Greedy output changed for repeated length {length}')
        request('sampled_k4', fixture(138), sampled=True)
        repeated = request('greedy_after_sampled', fixture(138))
        if repeated.get('text') != known[138].get('text'):
            failures.append('Greedy output changed after sampled-k4 switch')
        prefix = fixture(4096, 91)
        cold = request('prefix_cold_4130', prefix + fixture(34, 92))
        hit_offset = len(events(args.events_path)[0])
        hit = request('prefix_hit_4130', prefix + fixture(34, 92))
        hit_events = events(args.events_path)[1][hit_offset:]
        if not any(e.get('event') == 'prefill_chunk' and e.get('start') == 4096
                   for e in hit_events):
            failures.append('Repeated 4130-token prefix did not resume at cached boundary 4096')
        request('prefix_fork_4130', prefix + fixture(34, 93))
        if cold.get('text') != hit.get('text'):
            failures.append('Greedy cached-prefix output differs from cold output')
        request('decode_boundary_4080_64', fixture(4080, 94), 64)
        if not args.skip_agent_turns:
            agent_turns['metrics_before'] = metrics()
            wave_offsets = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_concurrency) as pool:
                for turn in range(2):
                    begin = len(events(args.events_path)[0])
                    order = list(range(args.max_concurrency))
                    if turn:
                        order.reverse()
                    futures = []
                    for conversation in order:
                        # Each conversation differs from token zero; no cross-request shared prefix.
                        prompt = [1000 + conversation] * turn_length
                        if turn:
                            prompt += [2000 + conversation] * 33
                        futures.append(pool.submit(request, f'agent_{conversation}_turn_{turn}', prompt))
                    for future in futures:
                        future.result()
                    end = len(events(args.events_path)[0])
                    wave_offsets.append([begin, end])
                    agent_turns[f'metrics_after_turn_{turn}'] = metrics()
            all_events = events(args.events_path)[1]
            second = all_events[slice(*wave_offsets[1])]
            owned, admissions = set(), []
            observed_active = 0
            for event in second:
                if event.get('event') == 'release_request':
                    owned.discard(event['slot'])
                elif event.get('event') == 'prefill_submit':
                    for slot, start in zip(event['slots'], event['starts']):
                        if slot not in owned:
                            admissions.append(dict(slot=slot, cached_start=start))
                            owned.add(slot)
                    observed_active = max(observed_active,
                                          len(owned | set(event.get('active_decode_slots', []))))
            previous_tokens = turn_length * args.max_concurrency
            cached_tokens = sum(item['cached_start'] for item in admissions)
            agent_turns.update(event_line_offsets=wave_offsets,
                               previous_sent_tokens_ceiling=previous_tokens,
                               previous_sent_tokens_per_conversation=turn_length,
                               actual_cachehit_admissions=admissions,
                               observed_active_slots_max=observed_active,
                               observed_cached_tokens=cached_tokens,
                               reuse_fraction_of_previous_sent=cached_tokens / previous_tokens,
                               completed_admissions_observed=len(admissions),
                               pool_bound=None,
                               pool_bound_missing_reason='No numeric scheduler pool capacity found in metrics')
            for sample in agent_turns['metrics_after_turn_1']['samples']:
                if 'cache_config_info' in sample:
                    match = re.search(r'num_gpu_blocks="(\d+)"', sample)
                    if match:
                        agent_turns['pool_bound'] = dict(scheduler_blocks=int(match.group(1)),
                                                        source='vLLM cache_config_info metric')
                        agent_turns.pop('pool_bound_missing_reason', None)
            if len(admissions) != args.max_concurrency:
                failures.append('Multi-turn admission count cannot be reconciled with requested conversations')
            if cached_tokens / previous_tokens < .9:
                agent_turns['pool_shortfall_observed'] = True
                agent_turns['pool_shortfall_reason'] = (
                    'Observed reuse below 0.9 under requested working set; inspect recorded prefix-cache '
                    'and preemption metrics. Capacity causality is not established by the ratio alone.')
                def numeric_samples(snapshot):
                    values = {}
                    for sample in snapshot['samples']:
                        try:
                            key, value = sample.rsplit(' ', 1)
                            values[key] = float(value)
                        except ValueError:
                            continue
                    return values
                old = numeric_samples(agent_turns['metrics_before'])
                new = numeric_samples(agent_turns['metrics_after_turn_1'])
                deltas = {key: value - old[key] for key, value in new.items()
                          if key in old and ('preempt' in key or 'prefix_cache' in key)}
                agent_turns['metric_deltas'] = deltas
                if agent_turns['pool_bound'] is None and not any(
                        'preempt' in key and value > 0 for key, value in deltas.items()):
                    failures.append('Reuse below 0.9 without measured pool bound or preemption evidence')
    except Exception as exc:
        failures.append(f'{type(exc).__name__}: {exc}')

    after_lines, _ = events(args.events_path)
    workload_seconds = time.perf_counter() - workload_started
    fresh = [json.loads(line) for line in after_lines[len(before_lines):] if line.strip()]
    if after_lines[:len(before_lines)] != before_lines:
        failures.append('Event file was truncated or rewritten during the probe')
    for event in fresh:
        if event.get('pid') != pid:
            failures.append('Worker PID changed during the probe')
        if event.get('event') == 'shutdown_release':
            failures.append('Server shut down during lifecycle traffic')
        if event.get('counters', {}).get('captures') != capture_count:
            failures.append('Capture count missing or changed during lifecycle traffic')
        if event.get('retirement_count') != retirements:
            failures.append('Retirement count missing or changed during lifecycle traffic')
        if event.get('event') == 'decode_submit' and event.get('trace_id') not in signatures.values():
            failures.append('Decode used a trace absent from prepared execution signatures')
    chunks = [dict(start=e['start'], length=e['length'], slots=e.get('slots'))
              for e in fresh if e.get('event') == 'prefill_chunk']
    concurrent_prefills = [dict(starts=e.get('starts'), ends=e.get('ends'),
                       active_decode_slots=e['active_decode_slots'])
                  for e in fresh if e.get('event') == 'prefill_submit' and e.get('active_decode_slots')]
    if not any(c['start'] > 0 for c in chunks):
        failures.append('No nonzero chunk start observed')
    if not any(c['start'] == 8192 for c in chunks):
        failures.append('No chunk at actual start 8192 observed for long prompt fixture')
    if not concurrent_prefills:
        failures.append('No prefill with concurrent decode slots observed')
    if not fresh:
        failures.append('No lifecycle events observed')
    host_read_delta = (fresh[-1]['host_logits_reads']-json.loads(before_lines[-1])['host_logits_reads']) if fresh else None
    if host_read_delta != 0:
        failures.append('Lifecycle token-out traffic used host logits or lacks counters')
    for record in records:
        if not record['complete'] or record.get('finish_reason') is None:
            failures.append(f"Incomplete request: {record['name']}")
        usage = record.get('usage', {})
        if usage.get('prompt_tokens') != record['prompt_tokens']:
            failures.append(f"Prompt token count missing or incorrect: {record['name']}")
        if usage.get('completion_tokens') != record['requested_output_tokens']:
            failures.append(f"Generated token count missing or incorrect: {record['name']}")
    raw_path.write_text(json.dumps(records, indent=2) + '\n')
    summary = dict(status='fail' if failures else 'pass', failures=sorted(set(failures)),
                   fixture='synthetic token IDs; NOT qualitative or headline performance evidence',
                   model=args.model, reduced=args.reduced, quick=args.quick,
                   max_concurrency=args.max_concurrency, server_url=args.server_url,
                   agent_turns=agent_turns,
                   events_path=args.events_path, event_line_offsets=[len(before_lines), len(after_lines)],
                   pid=pid, execution_signatures=signatures, captures=capture_count,
                   captures_final=fresh[-1].get('counters', {}).get('captures') if fresh else None,
                   retirement_final=fresh[-1].get('retirement_count') if fresh else None,
                   state_pool_final=fresh[-1].get('state_pool') if fresh else None,
                   retirement_baseline=retirements, actual_chunks=chunks, concurrent_prefill=concurrent_prefills,
                   host_logits_read_delta=host_read_delta,
                   requests=len(records), raw_requests=str(raw_path),
                   workload_seconds=workload_seconds,
                   completed_requests_per_second=sum(r['complete'] for r in records)/workload_seconds,
                   ttft_ms_p50=percentile([r['ttft_ms'] for r in records if r['ttft_ms'] is not None], .5),
                   ttft_ms_p95=percentile([r['ttft_ms'] for r in records if r['ttft_ms'] is not None], .95),
                   ttft_ms_p99=percentile([r['ttft_ms'] for r in records if r['ttft_ms'] is not None], .99),
                   sse_pause_ms_p99=percentile([gap for r in records for gap in r['sse_inter_chunk_ms']], .99),
                   sse_pause_ms_max=max((gap for r in records for gap in r['sse_inter_chunk_ms']),default=None),
                   completion_ms_p99=percentile([r['completion_ms'] for r in records], .99))
    output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(dict(status=summary['status'], output=str(output), failures=summary['failures'])))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
