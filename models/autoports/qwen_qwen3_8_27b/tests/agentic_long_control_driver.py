#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One cached-HF native long-chat replay; coordinator queue owns the lifecycle.

No HF model load, profiler, allocator tracking, or automatic configuration choice.
Streaming response evidence survives a request error or outer effort-box cutoff.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

REPO = Path(__file__).resolve().parents[4]
REL = Path('models/autoports/qwen_qwen3_8_27b')
TESTS = REPO / REL / 'tests'
CASE = REL / 'doc/datatype_sweep/agent_long_cases.json'
HF = REL / 'doc/datatype_sweep/agent_long_hf.json'
CASE_SHA = 'e50b91ee5611f7adc6cf57de89c8993ad217d9b2b35c886298ed8eb7efbf96cd'
HF_SHA = '1fd11f9926efca5921ae93b07d5b191eeaf2c8f2403739e628e4bec023a44e9c'
REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
MODEL = 'Qwen/Qwen3.8-27B'


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, TESTS / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


probe = module('long_control_probe_helpers', 'agentic_batch_probe.py')
write = probe.write


def reference():
    for path, expected in ((CASE, CASE_SHA), (HF, HF_SHA)):
        if hashlib.sha256((REPO/path).read_bytes()).hexdigest() != expected:
            raise ValueError('Cached reference bytes changed: '+str(path))
    cases = json.loads((REPO/CASE).read_text())
    hf = json.loads((REPO/HF).read_text())
    case = next(c for c in cases['cases'] if c['name'] == 'agent_history_32769')
    answer = next(c for c in hf['cases'] if c['name'] == case['name'])
    if (case['enable_thinking'] is not False or answer['enable_thinking'] is not False
            or len(case['input_tokens']) != 32769 or len(answer['tokens']) != 33
            or answer['tokens'][-1] not in hf['eos_ids'] or hf['status'] != 'complete'
            or not hf['stop_on_eos'] or not hf['greedy']
            or answer['input_sha256'] != case['input_sha256']
            or hashlib.sha256(json.dumps(case['input_tokens']).encode()).hexdigest() != case['input_sha256']):
        raise ValueError('Cached native prompt/completion contract differs')
    return case, answer, hf, cases


def verify_reference(output):
    # Tokenizer-only, in the established serving container; never load HF weights.
    from transformers import AutoTokenizer
    case, answer, hf, cases = reference()
    checkpoint = REPO / '.cache/qwen38-full'
    for name, expected in hf['checkpoint_metadata_sha256'].items():
        if hashlib.sha256((checkpoint/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Checkpoint metadata differs from cached HF: '+name)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    if hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest() != cases['chat_template_sha256']:
        raise ValueError('Chat template differs from cached HF')
    ids = tokenizer.apply_chat_template(case['messages'], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_dict=False)
    if ids != case['input_tokens']:
        raise ValueError('Native rendering differs from cached HF IDs')
    receipt = dict(ok=True, model=MODEL, expected_revision=REVISION, tokenizer=type(tokenizer).__name__,
        prompt_tokens=len(ids), input_sha256=case['input_sha256'], case_sha256=CASE_SHA,
        hf_sha256=HF_SHA, chat_template_sha256=cases['chat_template_sha256'],
        checkpoint_metadata_sha256=hf['checkpoint_metadata_sha256'],
        cached_hf_output_tokens=len(answer['tokens']), cached_hf_natural_eos=True,
        scope='Tokenizer/checkpoint metadata only; no HF model load or new HF computation')
    write(output, receipt)
    return 0


def stream_request(base_url, messages, output):
    output.mkdir(parents=True, exist_ok=True)
    payload = dict(model=MODEL, messages=copy.deepcopy(messages), max_tokens=81920,
        temperature=0.0, top_p=1.0, top_k=1, seed=1234, return_token_ids=True,
        chat_template_kwargs={'enable_thinking': False}, stream=True,
        stream_options={'include_usage': True})
    result = dict(status='running', request=payload, request_sha256=probe.sha(payload),
        token_ids=[], prompt_token_ids=None, content='', reasoning='', usage=None,
        finish_reason=None, stop_reason=None, request_timeout_seconds=12000,
        chunks=0, stream_done=False, started_unix=time.time())
    write(output/'response.json', result)
    start = time.monotonic()
    try:
        req = urllib.request.Request(base_url+'/v1/chat/completions', data=json.dumps(payload).encode(),
                                     headers={'Content-Type':'application/json'})
        with (output/'chunks.jsonl').open('w') as chunks, urllib.request.urlopen(req, timeout=12000) as response:
            for line in response:
                if not line.startswith(b'data:'):
                    continue
                data = line[5:].strip()
                if data == b'[DONE]':
                    result['stream_done'] = True
                    break
                chunk = json.loads(data)
                chunks.write(json.dumps(chunk)+'\n'); chunks.flush()
                result['chunks'] += 1
                if chunk.get('error'):
                    raise RuntimeError('Serving stream returned error: '+str(chunk['error']))
                if chunk.get('prompt_token_ids') is not None:
                    result['prompt_token_ids'] = chunk['prompt_token_ids']
                if chunk.get('usage') is not None:
                    result['usage'] = chunk['usage']
                for choice in chunk.get('choices', []):
                    if choice.get('index', 0) != 0:
                        raise ValueError('Unexpected multiple-choice response')
                    delta = choice.get('delta') or {}
                    result['content'] += delta.get('content') or ''
                    result['reasoning'] += delta.get('reasoning') or delta.get('reasoning_content') or ''
                    if choice.get('token_ids') is not None:
                        result['token_ids'].extend(choice['token_ids'])
                    if choice.get('finish_reason') is not None:
                        result['finish_reason'] = choice['finish_reason']
                        result['stop_reason'] = choice.get('stop_reason')
                result['elapsed_seconds'] = time.monotonic()-start
                write(output/'response.json', result)
        if not result['stream_done'] or result['finish_reason'] is None or result['usage'] is None:
            raise ValueError('Incomplete stream termination or usage evidence')
        if len(result['token_ids']) != result['usage']['completion_tokens']:
            raise ValueError('Native streaming token count differs from completion usage')
        result['status'] = 'completed'
    except BaseException as error:
        result.update(status='error', error=repr(error))
    finally:
        result['elapsed_seconds'] = time.monotonic()-start
        write(output/'response.json', result)
    return result


def compare_response(response, case, answer, eos_ids):
    observed = response['token_ids']
    expected = answer['tokens']
    raw_exact = observed == expected
    normalized = None
    normalization = 'none'
    # Do not silently trim EOS: an omitted EOS needs an explicit API stop ID.
    if (not raw_exact and response.get('finish_reason') == 'stop'
            and response.get('stop_reason') in eos_ids
            and expected[-1] == response['stop_reason'] and observed == expected[:-1]):
        normalized = True
        normalization = 'HF terminal EOS omitted by transport; API stop_reason explicitly identifies that same EOS'
    try:
        parsed = json.loads(response['content'])
    except (ValueError, TypeError):
        parsed = None
    return dict(raw_native_tokens_exact=raw_exact, eos_normalized_exact=normalized,
        eos_normalization=normalization, native_token_count=len(observed), hf_token_count=len(expected),
        actual_prompt_tokens=(response.get('usage') or {}).get('prompt_tokens'),
        native_prompt_ids_exact=response.get('prompt_token_ids') == case['input_tokens'],
        semantic_exact=parsed == case['expected_facts'], parsed_json=parsed,
        expected_json=case['expected_facts'], normal_stop=response.get('finish_reason') == 'stop',
        reasoning_disabled_observed=not response.get('reasoning'),
        possible_unproven_eos_transport_difference=not raw_exact and normalized is None and observed == expected[:-1])


def run_probe(args):
    case, answer, hf, _ = reference()
    identity = json.loads(args.server_config.read_text())
    if (identity.get('layer_count') != 64 or identity.get('configured_layer_count') != 64
            or identity.get('max_num_seqs') != 32 or identity.get('model_revision') != REVISION
            or identity.get('tokenizer_revision') != REVISION):
        raise ValueError('Observed serving model identity differs')
    argv = identity['server_command']
    for option in ('--block-size', '--mamba-block-size'):
        if argv[argv.index(option)+1] != '2048':
            raise ValueError('Replay requires B32/block2048')
    result = dict(status='starting', scored=False, scope='One cached-HF long-context retrieval control; not thinking-loop diagnosis or performance benchmark',
        server_identity=identity, case_sha256=CASE_SHA, hf_sha256=HF_SHA,
        limits={'max_tokens':81920,'request_timeout_seconds':12000},
        remaining_causal_question='Thinking-enabled long-context numerical drift and loop cause remain unresolved')
    write(args.probe/'result.json', result)
    warm = stream_request(args.base_url, [{'role':'user','content':'Return only {"ready":true}.'}], args.probe/'warmup')
    if warm['status'] != 'completed' or warm['finish_reason'] != 'stop':
        result.update(status='warmup_failed', warmup_status=warm['status'])
        write(args.probe/'result.json', result)
        return 2
    before = probe.event_snapshot(args.events)
    result.update(status='running', events_before=before, metrics_before=probe.metric_snapshot(args.base_url))
    write(args.probe/'result.json', result)
    response = stream_request(args.base_url, case['messages'], args.probe/'long_control')
    after = probe.event_snapshot(args.events)
    prepared = []
    for line in args.events.read_text().splitlines():
        try:
            prepared.append(json.loads(line))
        except ValueError:
            continue  # Concurrent final event may still be an incomplete line.
    allocation = next((e.get('allocation') for e in reversed(prepared) if e.get('event') == 'prepared'), None)
    result.update(events_after=after, metrics_after=probe.metric_snapshot(args.base_url),
        comparison=compare_response(response, case, answer, hf['eos_ids']),
        allocation_checkpoint=allocation, allocation_scope='Allocator checkpoint at preparation, NOT peak; no allocator tracking',
        postwarmup_trace_captures=probe.counter_delta(before,after,'captures','counters'),
        postwarmup_trace_retirements=probe.counter_delta(before,after,'retirement_count'),
        host_logits_read_delta=probe.counter_delta(before,after,'host_logits_reads'),
        request_status=response['status'], finish_reason=response['finish_reason'],
        status='review_required' if response['status']=='completed' else 'incomplete')
    c = result['comparison']
    result['confirmation_pass'] = (response['status']=='completed' and c['normal_stop']
        and c['actual_prompt_tokens'] == 32769 and c['native_prompt_ids_exact'] and c['semantic_exact']
        and c['reasoning_disabled_observed'] and (c['raw_native_tokens_exact'] or c['eos_normalized_exact'] is True)
        and all(result[k] == 0 for k in ('postwarmup_trace_captures','postwarmup_trace_retirements','host_logits_read_delta'))
        and allocation is not None)
    result['requires_cutoff_progress_review'] = response['status'] != 'completed' or response['finish_reason'] != 'stop'
    result['cutoff_policy'] = 'No score assigned; a progressing cutoff requires coordinator rerun with higher relevant limits, never below release floors.'
    result['acceptance'] = 'Coordinator reviews exactness and semantic evidence; no automatic configuration selection'
    write(args.probe/'result.json', result)
    return 0 if result['status']=='review_required' else 2


def lifecycle(args):
    reference()  # Immutable artifacts verified before any server lifecycle.
    root = args.output.resolve(); root.relative_to(REPO)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Fresh output directory required')
    if not 0 < args.box_hours <= .5:
        raise ValueError('Coordinator box must be positive and at most0.5mesh-hour')
    root.mkdir(parents=True, exist_ok=True)
    owned = module('long_control_owned_server','benchmark_server.py')
    queue_helpers = module('long_control_lifecycle_helpers','agentic_batch_driver.py')
    started=time.monotonic(); deadline=started+args.box_hours*3600
    observed=root/'b32/observed'; observed.mkdir(parents=True)
    state=root/'b32/owned_server.json'; relay=None
    summary=dict(status='starting',scored=False,batch=32,scheduler_block_size=2048,
        box_hours=args.box_hours,device_hour_estimate=.15,started_unix=time.time(),errors=[],
        source_sha256={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),TESTS/'agentic_batch_probe.py',TESTS/'agentic_batch_driver.py',
             TESTS/'benchmark_server.py',TESTS/'benchmark_launch.sh',REPO/REL/'tt/generator_vllm.py']})
    def interrupted(signum, frame):
        raise InterruptedError('Coordinator interrupted long-control driver: '+str(signum))
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted)
    try:
        proof=root/'tokenizer-proof.json'
        queue_helpers.checked_run(['docker','exec','-w','/workspace',owned.CONTAINER,
            '/opt/venv/bin/python','/workspace/'+str(REL/'tests/agentic_long_control_driver.py'),
            '--verify-reference',owned.in_container(proof)],deadline,root/'tokenizer-proof.log')
        queue_helpers.check_port_available()
        with (root/'relay.log').open('w') as log:
            relay=subprocess.Popen([sys.executable,str(TESTS/'benchmark_proxy.py'),str(root/'relay-owner.json')],stdout=log,stderr=subprocess.STDOUT)
        for _ in range(50):
            if relay.poll() is not None:raise RuntimeError('Owned relay exited before readiness')
            if (root/'relay-owner.json').exists():break
            time.sleep(.1)
        else:raise TimeoutError('Owned relay readiness missing')
        write(root/'summary.json',summary)
        startup=time.monotonic()
        queue_helpers.checked_run([sys.executable,str(TESTS/'benchmark_server.py'),'--max-num-seqs','32',
            '--output',str(observed/'server.json'),'--serving-events',str(observed/'events.jsonl'),
            '--shutdown-timeout','30','--scheduler-block-size','2048'],deadline,observed/'launch.log')
        summary.update(startup_seconds=time.monotonic()-startup,status='probing')
        write(root/'summary.json',summary)
        queue_helpers.checked_run([sys.executable,str(Path(__file__)),'--probe',str(root/'b32/probe'),
            '--server-config',str(observed/'server.json'),'--events',str(observed/'events.jsonl')],deadline,observed/'probe.log')
        result=json.loads((root/'b32/probe/result.json').read_text())
        summary.update(status='review_required',confirmation_pass=result['confirmation_pass'])
    except BaseException as error:
        summary.update(status='incomplete',missing_reason=repr(error))
        summary['errors'].append(repr(error))
        summary['cutoff_policy']='Retain streaming chunks/partial response. No score assigned. If still progressing, coordinator re-estimates a larger effort box without lowering request/output floors.'
    finally:
        if state.exists():
            try:owned.helper('inspect',state);owned.helper('stop',state)
            except BaseException as error:summary['errors'].append('owned cleanup: '+repr(error))
        if relay is not None and relay.poll() is None:
            relay.terminate()
            try:relay.wait(timeout=10)
            except subprocess.TimeoutExpired:relay.kill();relay.wait()
        summary.update(elapsed_seconds=time.monotonic()-started,used_mesh_hours=(time.monotonic()-started)/3600,
                       owned_server_state_remaining=state.exists(),relay_exit=None if relay is None else relay.returncode)
        write(root/'summary.json',summary)
    return 0 if summary['status']=='review_required' and not summary['errors'] else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output',type=Path)
    mode.add_argument('--verify-reference',type=Path)
    mode.add_argument('--probe',type=Path)
    parser.add_argument('--server-config',type=Path)
    parser.add_argument('--events',type=Path)
    parser.add_argument('--base-url',default='http://127.0.0.1:8000')
    parser.add_argument('--box-hours',type=float,default=.5)
    args=parser.parse_args()
    raise SystemExit(verify_reference(args.verify_reference) if args.verify_reference else run_probe(args) if args.probe else lifecycle(args))
