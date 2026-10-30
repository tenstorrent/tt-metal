#!/usr/bin/env python3
"""Stage 11 server hook. Host entry point; container helpers never open devices.

Only processes recorded by this hook with matching Linux start times may be
stopped. Startup/configuration evidence is observed, not inferred from arguments.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.request
from urllib.parse import urlparse

MODEL = 'Qwen/Qwen3.8-27B'
REL = Path('models/autoports/qwen_qwen3_8_27b')
REPO = Path(__file__).resolve().parents[4]
CONTAINER = 'qwen38-functional-build'
SCRIPT = '/workspace/' + str(REL / 'tests/benchmark_server.py')


def write(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, indent=2) + '\n')


def proc(pid):
    try:
        base = Path('/proc') / str(pid)
        stat = (base/'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': int(pid), 'start_ticks': stat[19], 'state': stat[0],
                'pgid': int(stat[2]), 'argv': (base/'cmdline').read_bytes().decode().strip('\0').split('\0')}
    except (OSError, ValueError):
        return None


def same(observed, saved):
    return observed and observed['start_ticks'] == saved['start_ticks'] and observed['state'] != 'Z'


def launch_environment(data):
    env = dict(os.environ, QWEN38_BENCHMARK_PHASES=data['phase_path'])
    block_size = data.get('scheduler_block_size', 4096)
    if block_size not in (2048, 4096):
        raise ValueError('Scheduler block size must be2048 or4096')
    for option in ('--block-size', '--mamba-block-size'):
        if data['args'][data['args'].index(option)+1] != str(block_size):
            raise ValueError('Launch cache arguments disagree with worker block setting')
    override = data.get('num_gpu_blocks_override')
    option = '--num-gpu-blocks-override'
    if override is not None and (type(override) is not int or override not in (353, 385) or block_size != 2048 or data.get('capacity') != 32):
        raise ValueError('Pool experiment requires B32/block2048 and353 or385 IDs')
    if (override is None and option in data['args']) or (override is not None and
            (option not in data['args'] or data['args'][data['args'].index(option)+1] != str(override))):
        raise ValueError('Launch pool override disagrees with owned manifest')
    env['QWEN38_SCHEDULER_BLOCK_SIZE'] = str(block_size)
    # Optional agentic observations are explicit; ambient values cannot enable them.
    env.pop('QWEN38_AGENTIC_EVENTS', None)
    if data.get('serving_events'):
        env['QWEN38_AGENTIC_EVENTS'] = data['serving_events']
    return env



def verify_pool_observation(requested, metrics, events_path):
    """Fail closed on actual planner/allocation evidence, never argv alone."""
    if requested is None:
        return None
    counts = re.findall(r'num_gpu_blocks="(\d+)"', metrics)
    if not counts or set(counts) != {str(requested)}:
        raise RuntimeError('Observed vLLM pool count does not match requested allocation')
    allocation = None
    for line in Path(events_path).read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get('event') == 'prepared':
            allocation = event.get('allocation')
    if not allocation or allocation.get('scheduler_blocks') != requested or allocation.get('scheduler_block_size') != 2048 or allocation.get('group_count') != 4:
        raise RuntimeError('Actual four-group autoport allocation does not match requested pool')
    return allocation


def internal(action, state_path, payload=None):
    state_path = Path(state_path)
    if action == 'start':
        conflicts = []
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            item = proc(int(entry.name))
            if item and item['state'] != 'Z' and any(
                    'vllm.entrypoints' in arg or 'EngineCore' in arg for arg in item['argv']):
                conflicts.append(item)
        if conflicts:
            raise RuntimeError('Existing serving processes are not owned by this hook: '+str(conflicts))
        data = json.loads(Path(payload).read_text())
        env = launch_environment(data)
        with open(data['log'], 'wb') as log:
            p = subprocess.Popen(['bash', '/workspace/'+str(REL/'tests/benchmark_launch.sh'), *data['args']],
                                 cwd='/workspace', env=env, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        observed = proc(p.pid)
        write(state_path, {**data, 'owner': observed, 'started_unix': time.time(), 'members': [observed]})
    elif action == 'inspect':
        data = json.loads(state_path.read_text())
        owner = proc(data['owner']['pid'])
        members = [v for p in Path('/proc').iterdir() if p.name.isdigit()
                   if (v := proc(int(p.name))) and v['pgid'] == data['owner']['pid']]
        # Persist only identities whose group is still anchored by a known member.
        if same(owner, data['owner']) or any(same(v, s) for v in members for s in data['members']):
            data['members'] = members
            write(state_path, data)
        print(json.dumps({'owner': owner, 'members': members}))
    elif action == 'stop':
        data = json.loads(state_path.read_text())
        def owned_alive():
            return [v for s in data['members'] if same(v := proc(s['pid']), s)]
        live = owned_alive()
        if live:
            if any(p['pgid'] != data['owner']['pid'] for p in live):
                raise RuntimeError('Recorded process left its owned process group; refusing group cleanup')
            os.killpg(data['owner']['pid'], signal.SIGTERM)
            until = time.monotonic()+40
            while owned_alive() and time.monotonic()<until:
                time.sleep(.5)
            if owned_alive():
                os.killpg(data['owner']['pid'], signal.SIGKILL)
                time.sleep(2)
            if owned_alive():
                raise RuntimeError('Owned server did not stop')
        write(state_path.with_suffix('.stopped.json'), {**data, 'stopped_unix': time.time()})
        state_path.unlink()


def in_container(path):
    return '/workspace/' + str(Path(path).resolve().relative_to(REPO))


def helper(action, state, payload=None):
    argv = ['docker', 'exec', '-w', '/workspace', CONTAINER, '/opt/venv/bin/python', SCRIPT,
            '--internal', action, '--state', in_container(state)]
    if payload:
        argv += ['--payload', in_container(payload)]
    result = subprocess.run(argv, text=True, capture_output=True, check=True)
    return json.loads(result.stdout) if result.stdout.strip() else None


def git(path):
    return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()


def main(args):
    output = Path(args.output).resolve()
    run_dir = output.parent
    run_dir.mkdir(parents=True,exist_ok=True)
    state = run_dir.parent/'owned_server.json'
    def interrupted(signum, frame):
        if state.exists():
            helper('stop', state)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if args.stop:
        if state.exists():
            helper('stop',state)
        return
    capacity = args.max_num_seqs
    block_size = getattr(args, 'scheduler_block_size', 4096)
    if block_size not in (2048, 4096):
        raise ValueError('Scheduler block size must be2048 or4096')
    pool_override = getattr(args, 'num_gpu_blocks_override', None)
    if pool_override is not None and (type(pool_override) is not int or pool_override not in (353, 385) or block_size != 2048 or capacity != 32):
        raise ValueError('Pool experiment requires B32/block2048 and353 or385 IDs')
    shutdown_timeout = args.shutdown_timeout
    if shutdown_timeout < 0 or shutdown_timeout > 30:
        raise ValueError('Shutdown timeout must be0..30s within the40s owned cleanup window')
    serving_events = in_container(args.serving_events) if args.serving_events else None
    if args.serving_events:
        Path(args.serving_events).resolve().parent.mkdir(parents=True, exist_ok=True)
    config = json.loads((REPO/REL/'doc/optimized_vllm/batch_configurations.json').read_text())['configurations'][str(capacity)]
    if state.exists():
        old = json.loads(state.read_text())
        observed = helper('inspect',state)
        if (old['capacity'] == capacity and old.get('scheduler_block_size',4096) == block_size
                and old.get('num_gpu_blocks_override') == pool_override
                and old.get('serving_events') == serving_events
                and old.get('shutdown_timeout', 0) == shutdown_timeout
                and same(observed['owner'],old['owner'])):
            data = old
        else:
            helper('stop',state)
            data = None
    else:
        data = None
    if data is None:
        # Reject an unowned live endpoint, even when it would happen to match.
        try:
            urllib.request.urlopen(args.base_url+'/health',timeout=2)
        except Exception:
            pass
        else:
            raise RuntimeError('A ready endpoint is not owned by this hook; refusing to replace it')
        url = urlparse(args.base_url)
        if url.hostname not in ('localhost','127.0.0.1'):
            raise ValueError('This local Docker hook requires a loopback endpoint')
        argv = ['--model','/workspace/.cache/qwen38-full','--served-model-name',MODEL,
                '--block-size',str(block_size),'--max-num-seqs',str(capacity),'--port',str(url.port or 8000),
                '--max-model-len','262144','--additional-config',json.dumps({'tt':{
                    'sample_on_device_mode':'all','trace_region_size':134217728,
                    'fabric_config':'FABRIC_1D_RING','trace_mode':'decode_only'}}),
                '--language-model-only','--mamba-cache-mode','align','--mamba-block-size',str(block_size),
                '--max-num-batched-tokens',str(config['max_num_batched_tokens']),
                '--max-logprobs','-1','--async-scheduling','--enable-prefix-caching','--reasoning-parser','qwen3']
        if pool_override is not None:
            argv += ['--num-gpu-blocks-override', str(pool_override)]
        if shutdown_timeout:
            argv += ['--shutdown-timeout', str(shutdown_timeout)]
        if config.get('long_prefill_token_threshold'):
            argv += ['--long-prefill-token-threshold',str(config['long_prefill_token_threshold'])]
        log = run_dir/f'server-b{capacity}.log'
        data = {'capacity':capacity,'scheduler_block_size':block_size,'args':argv,'log':in_container(log),
                'phase_path':in_container(run_dir/f'phases-b{capacity}.jsonl')}
        if pool_override is not None:
            data['num_gpu_blocks_override'] = pool_override
        if shutdown_timeout:
            data['shutdown_timeout'] = shutdown_timeout
        if serving_events:
            data['serving_events'] = serving_events
        launch = run_dir/f'launch-b{capacity}.json'
        write(launch,data)
        helper('start',state,launch)
        data = json.loads(state.read_text())
    started = time.monotonic()
    try:
        while True:
            observed = helper('inspect',state)
            if not same(observed['owner'],data['owner']):
                raise RuntimeError('Owned server exited during startup; inspect '+data['log'])
            try:
                with urllib.request.urlopen(args.base_url+'/v1/models',timeout=3) as response:
                    models = json.load(response)
                if MODEL not in [m['id'] for m in models['data']]:
                    raise RuntimeError('Serving endpoint has the wrong model')
                break
            except (OSError, ValueError):
                if time.monotonic()-started > 1800:
                    raise TimeoutError('Server readiness exceeded 1800 seconds')
                time.sleep(3)
        pool_allocation = None
        if pool_override is not None:
            if not args.serving_events:
                raise RuntimeError('Pool experiment needs actual allocation events before probe')
            with urllib.request.urlopen(args.base_url+'/metrics',timeout=3) as response:
                metrics = response.read().decode()
            pool_allocation = verify_pool_observation(pool_override,metrics,args.serving_events)
        log = REPO/Path(data['log']).relative_to('/workspace')
        text = log.read_text(errors='replace')
        marker = re.search(r'AUTOPORT_ADAPTER (\S+) layers=(\[[^\n]+?\]) precision=(\{[^\n]+\})',text)
        if not marker:
            raise RuntimeError('Missing actual imported autoport/layer/precision marker')
        layers, precision = ast.literal_eval(marker[2]), ast.literal_eval(marker[3])
        if layers != list(range(64)) or 'models/autoports/qwen_qwen3_8_27b/tt/generator_vllm.py' not in marker[1]:
            raise RuntimeError('Server did not import the full generated autoport')
        if precision.get('config_id') != 'head_bfp8_hifi2':
            raise RuntimeError('Unexpected precision policy')
        if f"'max_num_seqs': {capacity}" not in text or f"'max_num_batched_tokens': {config['max_num_batched_tokens']}" not in text:
            raise RuntimeError('Observed effective server capacity/scheduler budget mismatch')
        if 'max_seq_len=262144' not in text or 'enable_prefix_caching=True' not in text:
            raise RuntimeError('Observed context/prefix cache mismatch')
        if 'Opening local chip ids/PCIe ids: {0, 1, 2, 3}' not in text:
            raise RuntimeError('Observed four-chip hardware topology missing')
        actual_argv = observed['owner']['argv']
        if actual_argv[actual_argv.index('--max-num-seqs')+1] != str(capacity):
            raise RuntimeError('Process argv capacity mismatch')
        provenance = json.loads((REPO/REL/'doc/functional_decoder/weights_provenance.json').read_text())
        revision = provenance['hf_revision']
        checkpoint_proof = {}
        source_checkpoint = Path(provenance['source_checkpoint'])
        for name in ('config.json','tokenizer_config.json','tokenizer.json','chat_template.jinja'):
            local = REPO/'.cache/qwen38-full'/name
            source = source_checkpoint/name
            metadata = (source_checkpoint/'.cache/huggingface/download'/(name+'.metadata')).read_text().splitlines()
            local_hash = hashlib.sha256(local.read_bytes()).hexdigest()
            if metadata[0] != revision or local_hash != hashlib.sha256(source.read_bytes()).hexdigest():
                raise RuntimeError('Checkpoint/tokenizer revision proof mismatch: '+name)
            checkpoint_proof[name] = {'sha256':local_hash,'hf_revision':metadata[0],
                                      'etag':metadata[1],'source':str(source)}
        snapshot = run_dir/f'configuration-b{capacity}.log'
        snapshot.write_text(json.dumps({'observed_processes':observed,'models_endpoint':models,
              'checkpoint_tokenizer_proof':checkpoint_proof, 'pool_allocation':pool_allocation,
              'source_sha256':{str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((REPO/REL/'tt').glob('*.py'))},'phase_path':data['phase_path']},indent=2)+'\n'+text)
        identity = {'model':MODEL,'implementation':str(REL),
            'generator_module':'models.autoports.qwen_qwen3_8_27b.tt.generator_vllm',
            'imported_module_file':marker[1],'model_revision':revision,'tokenizer_revision':revision,
            'revision_evidence':snapshot.name+' checkpoint_tokenizer_proof; model weights inherited from Stage10',
            'precision':precision,'layer_count':len(layers),'configured_layer_count':64,
            'source_commits':{'tt-metal':git(REPO),'vllm-tt-plugin':git(REPO/'bringup/dependencies/vllm-tt-plugin'),
                              'vllm':None},
            'vllm_source_limitation':'Installed 0.30.0+empty wheel has no direct_url.json; source commit unestablished.',
            'hardware':'4 Blackhole p300c ASICs; P300x2; TP mesh 1x4; physical IDs 0,1,2,3',
            'server_command':actual_argv,'prefix_caching':True,'max_num_seqs':capacity,'max_model_len':262144,
            'base_url':args.base_url,'configuration_evidence':snapshot.name,
            'configuration_evidence_sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            'server_started_unix':data['started_unix'],'selection_wait_seconds':time.monotonic()-started,
            'phase_path':data['phase_path'],'owned_state':str(state)}
        if data.get('serving_events'):
            identity['serving_events'] = data['serving_events']
        write(output,identity)
        if not (run_dir.parent/'identity.json').exists():
            write(run_dir.parent/'identity.json',identity)
    except BaseException:
        helper('stop',state)
        raise


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--max-num-seqs',type=int,choices=[1,8,16,32])
    p.add_argument('--scheduler-block-size', type=int, choices=[2048,4096], default=4096)
    p.add_argument('--num-gpu-blocks-override', type=int, choices=[353,385], help='Opt-in B32/block2048 capacity experiment; default allocation unchanged')
    p.add_argument('--base-url',default='http://127.0.0.1:8000')
    p.add_argument('--output')
    p.add_argument('--stop',action='store_true')
    p.add_argument('--shutdown-timeout', type=int, default=0, help='Opt-in vLLM graceful shutdown seconds,0..30')
    p.add_argument('--serving-events', help='Optional host path inside checkout for agentic event JSONL')
    p.add_argument('--internal',choices=['start','inspect','stop'])
    p.add_argument('--state')
    p.add_argument('--payload')
    a=p.parse_args()
    if a.internal:
        internal(a.internal,a.state,a.payload)
    else:
        main(a)
