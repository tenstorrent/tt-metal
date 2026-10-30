#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
"""Unscored ten-conversation paired probe; HTTP only, no process/device lifecycle.

Run through the goal12 queue against a coordinator-owned, warmed server. Changing
batch is the coordinator's job. Exactness uses native returned token IDs without
logprobs, preserving device sampling. Missing evidence never constitutes a pass.
"""
import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import urllib.request

MODEL = 'Qwen/Qwen3.8-27B'


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent,
                                         prefix='.'+path.name+'.', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(value, indent=2) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def event_snapshot(path):
    last = None
    with path.open() as stream:
        for line in stream:
            try:
                last = json.loads(line)
            except ValueError:
                continue
    return last


def metric_snapshot(base_url):
    try:
        with urllib.request.urlopen(base_url + '/metrics', timeout=15) as response:
            text = response.read().decode()
        return [line for line in text.splitlines() if not line.startswith('#') and
                any(word in line for word in ('cache', 'preempt', 'evict', 'running'))]
    except Exception as error:
        return {'unavailable': repr(error)}


def make_cohort():
    notes = '\n'.join(f'Reference record {i:04d}: the stored count is seven apples.' for i in range(384))
    return [{'id': f'conversation-{i:02d}', 'messages': [
        {'role': 'system', 'content': f'Conversation identifier {i:02d}. Use only the requested output format.'},
        {'role': 'user', 'content': notes + '\n\nAlice has 7 apples and gives Bob 3. '
         'Return only a JSON object with keys alice_apples and bob_apples containing the resulting counts.'}
    ]} for i in range(10)]


def request(base_url, item, cap, timeout, barrier=None):
    payload = dict(model=MODEL, messages=copy.deepcopy(item['messages']), max_tokens=cap,
                   temperature=0.0, top_p=1.0, top_k=1, seed=1234,
                   return_token_ids=True,
                   chat_template_kwargs=dict(enable_thinking=True, reasoning_effort='xhigh'))
    if barrier:
        barrier.wait(timeout=30)
    started = time.monotonic()
    try:
        req = urllib.request.Request(base_url + '/v1/chat/completions',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            value = json.load(response)
        choice = value['choices'][0]
        return dict(id=item['id'], request=payload, request_sha256=sha(payload), response=value,
                    token_ids=choice.get('token_ids'), finish_reason=choice.get('finish_reason'),
                    usage=value.get('usage'), started_monotonic=started,
                    elapsed_seconds=time.monotonic()-started, status='completed')
    except Exception as error:
        return dict(id=item['id'], request=payload, request_sha256=sha(payload),
                    started_monotonic=started, elapsed_seconds=time.monotonic()-started,
                    status='error', error=repr(error))


def run_round(args, cohort, turn):
    """Persist each completed HTTP result before waiting for the cohort tail."""
    directory = args.output / 'turns' / f'turn-{turn}'
    directory.mkdir(parents=True, exist_ok=True)
    manifest = dict(round=turn, status='running', expected_ids=[item['id'] for item in cohort],
                    cohort_sha256=sha(cohort), completed=[], pending_ids=[item['id'] for item in cohort])
    write(directory/'cohort.json', cohort)
    write(directory/'manifest.json', manifest)
    barrier = threading.Barrier(10)
    started = time.monotonic()
    completed = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(request, args.base_url, item, args.output_cap,
                               args.request_timeout, barrier): item for item in cohort}
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                row = future.result()
            except Exception as error:
                row = dict(id=item['id'], status='error', error=repr(error),
                           request_sha256=None, elapsed_seconds=time.monotonic()-started)
            # IDs are generated by make_cohort, not supplied by the endpoint.
            path = directory / (item['id']+'.json')
            write(path, row)
            completed[item['id']] = row
            manifest['completed'].append(dict(id=item['id'], status=row['status'], file=path.name))
            manifest['pending_ids'] = [x['id'] for x in cohort if x['id'] not in completed]
            manifest['elapsed_seconds'] = time.monotonic()-started
            write(directory/'manifest.json', manifest)
    manifest.update(status='complete', elapsed_seconds=time.monotonic()-started)
    write(directory/'manifest.json', manifest)
    return dict(round=turn, elapsed_seconds=manifest['elapsed_seconds'],
                responses=[completed[item['id']] for item in cohort],
                manifest=str(directory/'manifest.json'))


def counter_delta(before, after, key, section=None):
    if before is None or after is None:
        return None
    if section:
        before, after = before.get(section, {}), after.get(section, {})
    left, right = before.get(key), after.get(key)
    return right-left if isinstance(left, (int, float)) and isinstance(right, (int, float)) else None


def compare_pair(reference, candidate):
    comparisons = []
    for previous, current in zip(reference['rounds'], candidate['rounds']):
        old = {r['id']: r for r in previous['responses']}
        for row in current['responses']:
            baseline = old.get(row['id'], {})
            tokens = row.get('token_ids')
            comparisons.append(dict(round=current['round'], id=row['id'],
                requests_equal=row['request_sha256'] == baseline.get('request_sha256'),
                tokens_equal=isinstance(tokens, list) and bool(tokens) and tokens == baseline.get('token_ids')))
    expected = 20
    return {'comparisons': comparisons, 'exact': len(reference['rounds']) == len(candidate['rounds']) == 2 and len(comparisons) == expected and
            all(r['requests_equal'] and r['tokens_equal'] for r in comparisons)}


def main(args):
    if args.output_cap < 81920 or args.request_timeout < 12000:
        raise ValueError('Probe retains81920 output tokens and12000s timeout to cover all release floors and B32 prediction')
    identity = json.loads(args.server_config.read_text())
    if identity.get('max_num_seqs') != args.batch or identity.get('layer_count') != 64:
        raise ValueError('Observed server identity must confirm requested batch and64layers')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'result.json').exists():
        raise ValueError('Refusing to overwrite an existing arm')
    cohort = make_cohort()
    result = dict(model=MODEL, batch=args.batch, concurrency=10, scored=False,
                  scope='synthetic batch-fit discriminator; not headline performance or isolated eval',
                  prompt_mode='native structured chat', server_identity=identity,
                  output_cap=args.output_cap, request_timeout=args.request_timeout,
                  control='B32 paired arm plus stage10 structured control semantics; HF comparison unavailable',
                  cohort_sha256=sha(cohort), rounds=[])
    write(args.output / 'cohort.json', cohort)
    write(args.output / 'result.json', result)
    warm = request(args.base_url, {'id': 'warmup', 'messages': [{'role': 'user', 'content':
        'Return only the JSON object {"ready":true}.'}]}, args.output_cap, args.request_timeout)
    write(args.output / 'warmup.json', warm)
    if warm['status'] != 'completed' or warm['finish_reason'] != 'stop':
        result.update(status='warmup_failed', warmup=warm)
        write(args.output / 'result.json', result)
        return 2
    before = event_snapshot(args.events)
    result['metrics_before'] = metric_snapshot(args.base_url)
    for turn in range(2):
        observed_round = run_round(args, cohort, turn)
        rows = observed_round['responses']
        result['rounds'].append(observed_round)
        write(args.output / 'result.json', result)
        if any(row['status'] != 'completed' or row['finish_reason'] != 'stop' for row in rows):
            result['status'] = 'cutoff_or_error_requires_review'
            break
        if turn == 0:
            for item, row in zip(cohort, rows):
                content = row['response']['choices'][0]['message'].get('content')
                if not isinstance(content, str) or not content.strip():
                    raise ValueError('Empty final content; inspect retained first-turn responses')
                item['messages'] += [{'role': 'assistant', 'content': content},
                    {'role': 'user', 'content': 'Bob gives one apple back. Return only the updated JSON object with the same keys.'}]
    after = event_snapshot(args.events)
    result['events_before'], result['events_after'] = before, after
    result['metrics_after'] = metric_snapshot(args.base_url)
    result['trace_capture_delta'] = counter_delta(before, after, 'captures', 'counters')
    result['trace_retirement_delta'] = counter_delta(before, after, 'retirement_count')
    result['host_logits_read_delta'] = counter_delta(before, after, 'host_logits_reads')
    result['state_restore_delta'] = counter_delta(before, after, 'restores', 'state_pool')
    result['long_prefix_confirmed'] = bool(result['rounds']) and all(
        (row.get('usage') or {}).get('prompt_tokens', 0) > 4096
        for row in result['rounds'][0]['responses'])
    result['measured_wall_seconds'] = sum(r['elapsed_seconds'] for r in result['rounds'])
    result['all_native_token_ids_present'] = all(isinstance(r.get('token_ids'), list) and r['token_ids']
                                               for turn in result['rounds'] for r in turn['responses'])
    result['invariant_evidence_complete'] = result['long_prefix_confirmed'] and all(result[k] == 0 for k in
        ['trace_capture_delta', 'trace_retirement_delta', 'host_logits_read_delta']) and (
            result['state_restore_delta'] is not None and result['state_restore_delta'] > 0)
    if args.reference:
        result['pair'] = compare_pair(json.loads(args.reference.read_text()), result)
    result.setdefault('status', 'review_required')
    result['acceptance'] = 'Coordinator must review active-row/cache metrics, cutoffs, exact pair and timing; no automatic selection.'
    write(args.output / 'result.json', result)
    return 0 if result['status'] == 'review_required' else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--batch', type=int, choices=[8, 16, 32], required=True)
    parser.add_argument('--server-config', type=Path, required=True)
    parser.add_argument('--events', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output-cap', type=int, default=81920)
    parser.add_argument('--request-timeout', type=int, default=12000)
    raise SystemExit(main(parser.parse_args()))
