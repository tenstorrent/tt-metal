#!/usr/bin/env python3
"""Allocator-driven decode growth versus a non-overlapped, token-exact control.

Run --mode control against a server launched with --no-async-scheduling, then
--mode verify against the normal async server using the saved --control JSON.
Synthetic token prompts are not qualitative or performance evidence.
"""
import argparse
import concurrent.futures
import hashlib
import json
import shlex
from pathlib import Path
import urllib.request


def read_events(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-url', required=True)
    parser.add_argument('--model', default='/workspace/.cache/qwen38-full')
    parser.add_argument('--events-path', required=True)
    parser.add_argument('--server-command-file', required=True,
                        help='Exact server launch command; control must contain --no-async-scheduling')
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['control', 'verify'], required=True)
    parser.add_argument('--control')
    parser.add_argument('--prompts-file',required=True,
                        help='Prompt-correct distinct token fixtures, long enough to exercise allocated pages')
    parser.add_argument('--concurrency', type=int, default=3)
    parser.add_argument('--generated-tokens', type=int, default=4160)
    parser.add_argument('--reduced')
    args = parser.parse_args()
    if not 2 <= args.concurrency <= 32 or args.generated_tokens < 4160:
        parser.error('Require concurrency 2..32 and >=4160 output tokens to cross two scheduler boundaries')
    if args.mode == 'verify' and not args.control:
        parser.error('--control is required for verification')
    output = Path(args.output)
    prompt_fixture = json.loads(Path(args.prompts_file).read_text())
    if len(prompt_fixture['prompts']) < args.concurrency:
        parser.error('Prompt fixture has fewer distinct requests than concurrency')
    output.parent.mkdir(parents=True, exist_ok=True)
    initial = read_events(args.events_path)
    prepared = next((e for e in reversed(initial) if e.get('event') == 'prepared'), {})
    failures, results = [], []
    expected_async = args.mode == 'verify'
    command = Path(args.server_command_file).read_text()
    command_tokens = shlex.split(command.replace('\\\n', ' '))
    disabled = '--no-async-scheduling' in command_tokens
    if disabled == expected_async:
        failures.append('Server launch command async policy does not match selected mode')
    if expected_async and '--async-scheduling' not in command_tokens:
        failures.append('Verification launch command must explicitly contain --async-scheduling')
    signatures = prepared.get('execution_signatures')
    captures = prepared.get('counters', {}).get('captures')
    retirements = prepared.get('retirement_count')
    if not signatures or captures is None or retirements is None:
        failures.append('Prepared trace lifecycle counters/signatures missing')
    def command_value(flag):
        for index, token in enumerate(command_tokens):
            normalized = token.replace('_','-')
            if normalized.startswith(flag + '='):
                return token.split('=', 1)[1]
            if normalized == flag and index + 1 < len(command_tokens):
                return command_tokens[index + 1]
        return None

    allocation = prepared.get('allocation') or {}
    shape_fields = ('scheduler_blocks', 'scheduler_block_size', 'group_count',
                    'recurrent_snapshot_bytes_per_device', 'attention_bytes_per_device',
                    'active_recurrent_bytes_per_device', 'selected_layers', 'precision_config')
    provenance = dict(batch=prepared.get('batch'),
                      prompts_sha256=hashlib.sha256(Path(args.prompts_file).read_bytes()).hexdigest(),
                      execution_signature_names=sorted(signatures or {}),
                      prepared_buckets=prepared.get('prepared_buckets'),
                      max_model_len=command_value('--max-model-len'),
                      allocation={key: allocation.get(key) for key in shape_fields})
    if (provenance['batch'] is None or provenance['max_model_len'] is None or
            provenance['prepared_buckets'] is None or
            any(value is None for value in provenance['allocation'].values())):
        failures.append('Missing prepared batch/layers/precision/cache shape or explicit launch --max-model-len')
    control = None
    if args.mode == 'verify':
        control = json.loads(Path(args.control).read_text())
        if (control.get('status') != 'pass' or control.get('mode') != 'control' or
                control.get('model') != args.model or control.get('reduced') != args.reduced or
                control.get('generated_tokens') != args.generated_tokens or
                control.get('concurrency') != args.concurrency):
            failures.append('Control provenance/configuration does not match')
        if control.get('prepared_provenance') != provenance:
            failures.append('Control prepared batch, layers, precision, signatures, context or cache shape differs')

    def request(index, wave):
        prompt = prompt_fixture['prompts'][index]['token_ids']
        if len(prompt) != 4080-index:
            raise ValueError('Page-growth prompt must start near the required allocator boundary')
        payload = dict(model=args.model, prompt=prompt, max_tokens=args.generated_tokens,
                       temperature=0.0, top_k=1, top_p=1.0, ignore_eos=True,
                       return_token_ids=True, stream=False)
        record = dict(index=index, wave=wave, prompt_tokens=len(prompt),
                      prompt_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest())
        try:
            req = urllib.request.Request(args.server_url.rstrip('/') + '/v1/completions',
                                         json.dumps(payload).encode(), {'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=14400) as response:
                body = json.load(response)
            record.update(response=body, token_ids=body['choices'][0].get('token_ids'))
            if body.get('usage', {}).get('completion_tokens') != args.generated_tokens:
                record['error'] = 'Incomplete generation'
            if not isinstance(record['token_ids'], list) or len(record['token_ids']) != args.generated_tokens:
                record['error'] = 'Missing exact generated token IDs'
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
        return record

    if not failures:
        if args.mode == 'control':
            # Requests remain concurrent; only step submission overlap is off
            # (--no-async-scheduling). Keep the same fixed batch and occupancy
            # as verification so this isolates asynchronous state handling.
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = [pool.submit(request,index,'synchronous') for index in range(args.concurrency)]
                results = [future.result() for future in futures]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                for wave, order in [('forward', range(args.concurrency)),
                                    ('reverse', reversed(range(args.concurrency)))]:
                    futures = [pool.submit(request, index, wave) for index in order]
                    results.extend(future.result() for future in futures)
    final = read_events(args.events_path)
    fresh = final[len(initial):]
    host_read_delta = final[-1]['host_logits_reads']-initial[-1]['host_logits_reads'] if final and initial else None
    if host_read_delta != 0:
        failures.append('Greedy page-growth traffic used host logits or lacks counters')
    if final[:len(initial)] != initial:
        failures.append('Event history changed during run')
    for event in fresh:
        if (event.get('pid') != prepared.get('pid') or
                event.get('counters', {}).get('captures') != captures or
                event.get('retirement_count') != retirements):
            failures.append('PID, capture count or retirement count changed/missing')
    for record in results:
        if 'error' in record:
            failures.append(f"{record['index']}/{record['wave']}: {record['error']}")
    comparisons = []
    if args.mode == 'verify':
        controls = {r['index']: r for r in json.loads(Path(control['raw_requests']).read_text())}
        for record in results:
            reference = controls.get(record['index'], {})
            equal = (record.get('token_ids') is not None and
                     record.get('token_ids') == reference.get('token_ids') and
                     record['prompt_sha256'] == reference.get('prompt_sha256'))
            comparisons.append(dict(index=record['index'], wave=record['wave'], exact_tokens_equal=equal))
            if not equal:
                failures.append(f"Token mismatch: {record['index']}/{record['wave']}")

    # Instrumentation is host scheduler metadata, never host-read TT positions.
    # Row fingerprints cover raw scheduler IDs, including newly allocated pages.
    last_rows, crossings, growth = {}, [], []
    unchanged, table_only, concurrent_slots = 0, 0, 0
    for event in fresh:
        if event.get('event') == 'release_request':
            last_rows.pop(str(event['slot']), None)
        if event.get('event') != 'decode_submit':
            continue
        if event.get('slot_remap') is not None:
            last_rows = {str(new): last_rows[str(old)]
                         for new, old in enumerate(event['slot_remap']) if str(old) in last_rows}
        if event.get('trace_id') not in (signatures or {}).values():
            failures.append('Decode trace is absent from prepared execution signatures')
        if any(key not in event for key in ('active_slots', 'slot_ends', 'scheduler_table_rows')):
            failures.append('Decode event lacks allocator-growth instrumentation')
            continue
        active = event['active_slots']
        concurrent_slots = max(concurrent_slots, len(active))
        if event.get('reload_page_table') and not event.get('reload_inputs'):
            table_only += 1
        if not event.get('reload_page_table') and not event.get('reload_inputs'):
            unchanged += 1
        for slot in active:
            key = str(slot)
            ends = event['slot_ends']
            end = ends.get(key) if isinstance(ends, dict) else ends[slot]
            row = event['scheduler_table_rows'].get(key)
            fingerprint = hashlib.sha256(json.dumps(row).encode()).hexdigest()
            if row is None:
                failures.append('Active slot has no scheduler table row')
            old = last_rows.get(key)
            if old and end is not None and old[0] is not None:
                if fingerprint != old[1]:
                    growth.append(dict(slot=slot, end=end,
                                       changed_columns=[i for i, (a, b) in enumerate(zip(old[2], row or []))
                                                        if a != b],
                                       reload_inputs=event.get('reload_inputs'),
                                       reload_page_table=event.get('reload_page_table')))
                if fingerprint != old[1] and not (event.get('reload_page_table') or event.get('reload_inputs')):
                    failures.append('Changed allocator row without table/input refresh')
                for boundary in (4096, 8192):
                    if old[0] <= boundary < end:
                        crossings.append(dict(slot=slot, boundary=boundary,
                                              before_end=old[0], after_end=end,
                                              row_changed=fingerprint != old[1],
                                              reload_inputs=event.get('reload_inputs'),
                                              reload_page_table=event.get('reload_page_table')))
            last_rows[key] = (end, fingerprint, row or [])
    if args.mode == 'verify':
        if concurrent_slots < 2 or not table_only or not unchanged or not growth:
            failures.append('Missing concurrent decode, table-only refresh, or unchanged-table execution')
        if len([c for c in crossings if c['boundary'] == 8192]) < args.concurrency:
            failures.append('Insufficient observed request crossings at second scheduler boundary')
    raw = output.with_suffix('.requests.json')
    raw.write_text(json.dumps(results) + '\n')
    summary = dict(status='fail' if failures else 'pass', failures=sorted(set(failures)),
                   mode=args.mode, model=args.model, reduced=args.reduced,
                   generated_tokens=args.generated_tokens, concurrency=args.concurrency,
                   control=args.control, raw_requests=str(raw.resolve()),
                   prompts_file=args.prompts_file,
                   pid=prepared.get('pid'), async_scheduling=not disabled,
                   async_policy_source='explicit supplied server launch command',
                   server_command_file=args.server_command_file, server_command=command,
                   prepared_provenance=provenance, prepared_allocation=allocation,
                   execution_signatures=signatures, captures=captures, retirements=retirements,
                   event_offsets=[len(initial), len(final)], events_path=args.events_path,
                   crossings=crossings, allocator_row_changes=growth, table_only_refreshes=table_only,
                   unchanged_table_steps=unchanged, max_active_decode_slots=concurrent_slots,
                   host_logits_read_delta=host_read_delta,
                   comparisons=comparisons, synthetic_not_quality=True)
    output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(dict(status=summary['status'], output=str(output), failures=summary['failures'])))
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
