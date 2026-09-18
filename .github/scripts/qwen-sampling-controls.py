import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def identity(name, sha, plugin, sfpi):
    Path('output/identity-attempt.json').write_text(json.dumps(dict(arm=name, expected_metal=sha, expected_plugin=plugin, expected_sfpi=sfpi))+'\n')
    assert command('git', 'rev-parse', 'HEAD') == sha
    assert command('git', '-C', 'vllm-tt-plugin', 'rev-parse', 'HEAD') == plugin
    assert not command('git', 'diff', '--name-only')
    cache = Path(os.environ['TT_METAL_CACHE'])
    assert not cache.exists(), str(cache)
    cache.mkdir(parents=True)
    assert 'TT_METAL_CCACHE_KERNEL_SUPPORT' not in os.environ
    compiler = Path('/work/runtime/sfpi/compiler/bin/riscv-tt-elf-g++')
    assert compiler.is_file(), 'Paired build must supply SFPI without a container fallback'
    compiler_version = command(str(compiler), '--version')
    assert sfpi in compiler_version, compiler_version
    ttnn_version = importlib.metadata.version('ttnn')
    assert sha[:7] in ttnn_version, (sha, ttnn_version)
    assert importlib.metadata.version('vllm') == '0.26.0+empty'
    model_cache = Path('/mnt/MLPerf/huggingface/hub/models--Qwen--Qwen3-32B')
    model_revision = (model_cache/'refs/main').read_text().strip()
    assert model_revision == '9216db5781bf21249d130ec9da846c4624c16137', model_revision
    import ttnn
    loaded = [line for line in Path('/proc/self/maps').read_text().splitlines()
              if any(part in line for part in ('libtt_', 'libttnn', '_ttnn', 'libdevice'))]
    data = dict(arm=name, metal_sha=sha, plugin_sha=plugin,
                runner=os.environ['RUNNER_NAME'], container_hostname=command('hostname'),
                ttnn_version=ttnn_version, vllm_version=importlib.metadata.version('vllm'),
                ttnn_path=ttnn.__file__, native_path=ttnn._ttnn.__file__,
                runtime_root=os.environ['TT_METAL_RUNTIME_ROOT'], cache=str(cache),
                model_revision=model_revision,
                compiler_version=compiler_version,
                compiler_sha256=hashlib.sha256(compiler.read_bytes()).hexdigest(),
                loaded_native_mappings=loaded,
                packages=sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions()))
    Path('output/identity.json').write_text(json.dumps(data, indent=2)+'\n')
    print(json.dumps(data, indent=2))


def start():
    try:
        connection = socket.create_connection(('localhost', 8000), timeout=2)
    except OSError:
        pass
    else:
        connection.close()
        raise RuntimeError('A server is already listening before this arm')
    config = {'tt': {'dispatch_core_axis': 'col', 'sample_on_device_mode': 'all',
                     'fabric_config': 'FABRIC_1D_RING', 'worker_l1_size': 1344544,
                     'trace_region_size': 184915840}}
    args = [sys.executable, '/work/vllm-tt-plugin/examples/server_example_tt.py',
            '--model', 'Qwen/Qwen3-32B', '--data_parallel_size', '4', '--max_num_seqs', '8',
            '--async-scheduling', '--additional-config', json.dumps(config)]
    print(json.dumps(args))
    with Path('output/vllm_server.log').open('w') as log:
        server = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    Path('server.pid').write_text(str(server.pid)+'\n')


def group_members(pgid):
    members = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] != 'Z' and int(fields[2]) == pgid:
                members.append(int(proc.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return members


def wait():
    pgid = int(Path('server.pid').read_text())
    deadline = time.monotonic() + 35 * 60
    fatal = ('EngineCore failed to start', 'EngineCore encountered a fatal error',
             'EngineDeadError', 'Engine core initialization failed', 'Failed core proc')
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen('http://localhost:8000/health', timeout=5) as response:
                if response.status == 200:
                    print('Server is ready', flush=True)
                    return
        except OSError:
            pass
        assert group_members(pgid), 'Server process group exited before readiness'
        log = Path('output/vllm_server.log').read_text(errors='replace')
        assert not any(message in log for message in fatal), 'Fatal server startup message'
        print('Server is starting', flush=True)
        time.sleep(20)
    raise TimeoutError('Server did not become ready within 35 minutes')


def stop():
    pgid = int(Path('server.pid').read_text())
    assert pgid > 1 and pgid != os.getpgrp()
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    for _ in range(60):
        if not group_members(pgid):
            try:
                connection = socket.create_connection(('localhost', 8000), timeout=2)
            except OSError:
                pass
            else:
                connection.close()
                raise RuntimeError('Server port remains active after process-group exit')
            Path('output/cleanup.json').write_text(json.dumps({'pgid': pgid, 'remaining': []})+'\n')
            return
        time.sleep(1)
    remaining = group_members(pgid)
    Path('output/cleanup.json').write_text(json.dumps({'pgid': pgid, 'remaining': remaining})+'\n')
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    raise RuntimeError(f'Server processes did not exit; no later arm is valid: {remaining}')


def result(name):
    root = ET.parse('output/sampling.xml').getroot()
    cases = list(root.iter('testcase'))
    assert len(cases) == 78, f'Expected full suite, found {len(cases)} cases'
    assert not list(root.iter('error')), 'Collection or fixture errors invalidate comparison'
    cleanup = json.loads(Path('output/cleanup.json').read_text())
    assert not cleanup['remaining']
    record = json.loads(Path('output/identity.json').read_text())
    record.update(failed=[case.attrib for case in cases if case.find('failure') is not None],
                  skipped=sum(case.find('skipped') is not None for case in cases), cases=len(cases))
    assert record['skipped'] == 4, record['skipped']
    results = Path(os.environ['RUNNER_TEMP'])/'qwen-sampling-results'
    results.mkdir(exist_ok=True)
    (results/f'{name}.json').write_text(json.dumps(record, indent=2)+'\n')
    print(f'{name}: {len(record["failed"])} failed, {78 - 4 - len(record["failed"])} passed, 4 skipped')
    with Path(os.environ['GITHUB_STEP_SUMMARY']).open('a') as report:
        report.write(f'\n{name}: {len(record["failed"])} failed, {78 - 4 - len(record["failed"])} passed, 4 skipped. Metal {record["metal_sha"]}, runner {record["runner"]}.\n')


def summary():
    results = Path(os.environ['RUNNER_TEMP'])/'qwen-sampling-results'
    records = [json.loads((results/f'{name}.json').read_text()) for name in ('sep16','sep17','warmup','sfpi')]
    assert len({r['runner'] for r in records}) == 1
    assert len({r['plugin_sha'] for r in records}) == 1
    assert len({r['cache'] for r in records}) == 4
    versions = [{n.lower(): v for n, v in r['packages'] if n.lower() != 'ttnn'} for r in records]
    assert all(v == versions[0] for v in versions), 'Installed Python dependencies drifted across arms'
    print(json.dumps({r['arm']: [f['name'] for f in r['failed']] for r in records}, indent=2))
    if any(r['failed'] for r in records):
        raise SystemExit(1)


if __name__ == '__main__':
    functions = {'identity': identity, 'start': start, 'wait': wait, 'stop': stop, 'result': result, 'summary': summary}
    functions[sys.argv[1]](*sys.argv[2:])
