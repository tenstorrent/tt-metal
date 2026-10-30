#!/usr/bin/env python3
"""Unscored, owned B32 smoke. Invoke only via goal12 device_queue/run.py."""
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

REPO = Path(__file__).resolve().parents[4]
MODEL_DIR = Path('models/autoports/qwen_qwen3_8_27b')
TESTS = REPO / MODEL_DIR / 'tests'
ROOT = REPO / MODEL_DIR / 'doc/agentic_evals/runs/local-smoke'
RUN = ROOT / 'observed'


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def request(item):
    payload = dict(model='Qwen/Qwen3.8-27B', messages=item['messages'], max_tokens=16384,
                   temperature=0.0, top_p=1.0, top_k=1, seed=1234,
                   chat_template_kwargs=dict(enable_thinking=True, reasoning_effort='xhigh'))
    started = time.monotonic()
    try:
        req = urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',
              data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=3600) as response:
            result = json.load(response)
        return dict(id=item['id'], request=payload, response=result,
                    elapsed_seconds=time.monotonic()-started, control=item['response'],
                    status='completed', review='Compare final content to retained stage10 control; not scored.')
    except Exception as error:
        return dict(id=item['id'], request=payload, elapsed_seconds=time.monotonic()-started,
                    status='error', error=repr(error))


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    RUN.mkdir(exist_ok=True)
    if (ROOT/'owned_server.json').exists():
        raise RuntimeError('Existing local-smoke ownership file; coordinator must inspect before resuming')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',8000))
    controls=json.loads((REPO/MODEL_DIR/'doc/optimized_vllm/shared_qualitative_outputs.json').read_text())
    items=[next(x for x in controls['outputs'] if x['id']==ident and x['generation']['temperature']==0)
           for ident in ['explain','structured']]
    # Tokenizer-only helper: no TTNN import or device open.
    render_code='''import json,sys\nfrom transformers import AutoTokenizer\nt=AutoTokenizer.from_pretrained('/workspace/.cache/qwen38-full',trust_remote_code=True)\na=json.load(sys.stdin)\nprint(json.dumps(dict(tokenizer_class=type(t).__name__,chat_template_present=bool(t.chat_template),outputs=[dict(id=x['id'],messages=x['messages'],rendered_prompt=t.apply_chat_template(x['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=True,reasoning_effort='xhigh'),prompt_token_ids=t.apply_chat_template(x['messages'],tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=True,reasoning_effort='xhigh')) for x in a])))'''
    rendered=subprocess.run(['docker','exec','-i','qwen38-functional-build','/opt/venv/bin/python','-c',render_code],
                            input=json.dumps(items),text=True,capture_output=True,check=False)
    (RUN/'tokenizer.stderr.log').write_text(rendered.stderr)
    rendered.check_returncode()
    prompt_format=json.loads(rendered.stdout)
    prompt_format.update(hf_model=controls['hf_model'],revision=controls['revision'],prompt_mode='chat',
                         endpoint='/v1/chat/completions',source='doc/optimized_vllm/shared_qualitative_outputs.json',
                         request_timeout_seconds=3600,output_cap=16384,batch=32,max_concurrency=2,
                         isolation='Local model smoke only; not isolated agent sandbox or release route.')
    write(RUN/'prompt-format.json',prompt_format)
    spec=importlib.util.spec_from_file_location('owned_benchmark_server',TESTS/'benchmark_server.py')
    server=importlib.util.module_from_spec(spec);spec.loader.exec_module(server)
    relay=None; rows=[]; errors=[]; started=time.monotonic()
    try:
        with (RUN/'relay.log').open('w') as log:
            relay=subprocess.Popen([sys.executable,str(TESTS/'benchmark_proxy.py'),str(RUN/'relay-owner.json')],
                                   stdout=log,stderr=subprocess.STDOUT)
        for _ in range(50):
            if relay.poll() is not None: raise RuntimeError('Owned relay failed; see relay.log')
            if (RUN/'relay-owner.json').exists():break
            time.sleep(.1)
        else: raise RuntimeError('Relay readiness marker absent')
        launch_started=time.monotonic()
        subprocess.run([sys.executable,str(TESTS/'benchmark_server.py'),'--max-num-seqs','32',
                        '--output',str(RUN/'server.json')],check=True)
        startup=time.monotonic()-launch_started
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            rows=list(pool.map(request,items))
        write(RUN/'responses.json',rows)
        write(RUN/'summary.json',dict(status='review_required' if all(x['status']=='completed' for x in rows) else 'failed',
              startup_seconds=startup,batch=32,max_concurrency=2,request_timeout_seconds=3600,output_cap=16384,
              requests=[dict(id=x['id'],status=x['status'],elapsed_seconds=x['elapsed_seconds'],
                  usage=x.get('response',{}).get('usage'),finish_reason=x.get('response',{}).get('choices',[{}])[0].get('finish_reason')) for x in rows],
              scored=False,isolated_eval_smoke=False,source='local vLLM native chat',
              limitation='No sustained agent workload, no headline performance, no device timing.'))
    except BaseException as error:
        errors.append(repr(error));write(RUN/'error.json',{'error':repr(error)})
    finally:
        try:
            state=ROOT/'owned_server.json'
            if state.exists():server.helper('stop',state)
        except BaseException as error:errors.append('server cleanup: '+repr(error))
        if relay is not None and relay.poll() is None:
            relay.terminate()
            try:relay.wait(timeout=10)
            except subprocess.TimeoutExpired:relay.kill();relay.wait()
        write(RUN/'cleanup.json',dict(owned_server_state_remaining=(ROOT/'owned_server.json').exists(),
              relay_exit=None if relay is None else relay.returncode,errors=errors,
              elapsed_seconds=time.monotonic()-started,reservation_device_hours=(time.monotonic()-started)/3600))
    return 1 if errors or any(x['status']!='completed' for x in rows) else 0


if __name__=='__main__':
    sys.exit(main())
