#!/usr/bin/env python3
"""HTTP cache-boundary correctness probe; synthetic tokens are not quality evidence."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', required=True)
    p.add_argument('--model', default='Qwen/Qwen3.8-27B')
    p.add_argument('--block-size', type=int, required=True)
    p.add_argument('--events', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    before = args.events.read_text().splitlines()
    prepared = next(json.loads(x) for x in reversed(before) if json.loads(x).get('event') == 'prepared')
    failures, rows, comparisons = [], [], []
    nonce = str(time.time_ns())

    def request(name, length, family, generated=16):
        prompt = [100 + (i + family * 37) % 251 for i in range(length)]
        payload = dict(model=args.model, prompt=prompt, max_tokens=generated,
                       temperature=0.0, top_k=1, top_p=1.0, ignore_eos=True,
                       return_token_ids=True, stream=False,
                       cache_salt=f'goal12-boundary-{nonce}-{family}')
        record = dict(name=name, prompt_tokens=length, requested_output_tokens=generated,
                      prompt_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest())
        start = time.monotonic()
        try:
            req = urllib.request.Request(args.url.rstrip('/') + '/v1/completions',
                                         json.dumps(payload).encode(), {'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=1800) as response:
                body = json.load(response)
            record.update(usage=body.get('usage'), token_ids=body['choices'][0].get('token_ids'),
                          finish_reason=body['choices'][0].get('finish_reason'))
            if not isinstance(record['token_ids'], list) or len(record['token_ids']) != generated:
                record['error'] = 'Missing or incomplete generated token IDs'
            if (record['usage'] or {}).get('prompt_tokens') != length:
                record['error'] = 'Prompt token count mismatch'
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
        record['host_elapsed_s'] = time.monotonic() - start
        return record

    b = args.block_size
    for index, length in enumerate([b-1, b, b+1, 2*b-1, 2*b, 2*b+1]):
        cold = request(f'cold-{length}', length, index)
        warm = request(f'warm-{length}', length, index)
        rows.extend([cold, warm])
        equal = cold.get('token_ids') is not None and cold.get('token_ids') == warm.get('token_ids')
        comparisons.append(dict(length=length, cold_warm_token_exact=equal))
        if not equal:
            failures.append(f'Cold/warm token mismatch at {length}')
        cached = ((warm.get('usage') or {}).get('prompt_tokens_details') or {}).get('cached_tokens')
        expected = ((length-1)//b)*b
        if cached is None or cached < expected:
            failures.append(f'Warm {length}: cached={cached}, expected at least {expected}')

    # Concurrent requests cross checkpoint boundaries during decode, then reuse
    # the same slots in reverse order. Independent salts avoid accidental reuse.
    controls = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for wave, order in [('forward', [0, 1]), ('reverse', [1, 0])]:
            futures = {i: pool.submit(request, f'{wave}-{i}', (i+1)*b-16, 20+i, 64) for i in order}
            for i, future in futures.items():
                record = future.result(); rows.append(record)
                if wave == 'forward':
                    controls[i] = record.get('token_ids')
                elif controls[i] is None or controls[i] != record.get('token_ids'):
                    failures.append(f'Concurrent slot reuse changed greedy tokens for {i}')

    final = args.events.read_text().splitlines()
    if final[:len(before)] != before:
        failures.append('Event history changed during probe')
    for line in final[len(before):]:
        event = json.loads(line)
        if event.get('event') == 'shutdown_release':
            continue
        if event.get('pid') != prepared.get('pid'):
            failures.append('Engine PID changed')
        if event.get('counters', {}).get('captures') != prepared.get('counters', {}).get('captures'):
            failures.append('Trace captures changed after warmup')
        if event.get('retirement_count') != prepared.get('retirement_count'):
            failures.append('Trace retirements changed after warmup')
    failures.extend(f"{r['name']}: {r['error']}" for r in rows if r.get('error'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(status='fail' if failures else 'pass', block_size=b,
        model=args.model, source='vllm', scope='full-model synthetic correctness, not quality or headline performance',
        allocation=prepared.get('allocation'), comparisons=comparisons,
        failures=sorted(set(failures)), requests=rows), indent=2)+'\n')
    print(json.dumps(dict(status='fail' if failures else 'pass', failures=sorted(set(failures)))))
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
