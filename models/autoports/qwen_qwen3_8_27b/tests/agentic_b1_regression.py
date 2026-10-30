#!/usr/bin/env python3
"""Owned headline-only B1 regression; invoke under goal12 device queue."""
import argparse
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

TESTS=Path(__file__).resolve().parent
ROOT=TESTS.parent
REPO=TESTS.parents[3]

def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--budget-seconds',type=int,default=2700)
    args=parser.parse_args();run=args.run_dir.resolve()
    run.mkdir(parents=True,exist_ok=False);observed=run/'observed';observed.mkdir()
    start=time.monotonic();deadline=start+args.budget_seconds-60
    spec=importlib.util.spec_from_file_location('owned_server',TESTS/'benchmark_server.py')
    server=importlib.util.module_from_spec(spec);spec.loader.exec_module(server)
    relay=None;errors=[];summary={'status':'incomplete','source':'vllm','scored':False}
    def interrupted(sig,frame):raise RuntimeError('Interrupted signal '+str(sig))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def execute(command,log):
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('B1 effort budget exhausted')
        with log.open('w') as stream:
            subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=remaining)
    try:
        with (run/'relay.log').open('w') as log:
            relay=subprocess.Popen([sys.executable,str(TESTS/'benchmark_proxy.py'),str(run/'relay-owner.json')],stdout=log,stderr=subprocess.STDOUT)
        for _ in range(50):
            if relay.poll() is not None:raise RuntimeError('Owned relay failed')
            if (run/'relay-owner.json').exists():break
            time.sleep(.1)
        else:raise RuntimeError('Relay readiness absent')
        execute([sys.executable,str(TESTS/'benchmark_server.py'),'--max-num-seqs','1','--scheduler-block-size','4096','--shutdown-timeout','30','--output',str(observed/'server.json')],run/'startup.log')
        identity=json.loads((observed/'server.json').read_text())
        assert identity['layer_count']==identity['configured_layer_count']==64
        assert identity['max_num_seqs']==1
        argv=identity['server_command'];assert argv[argv.index('--block-size')+1]=='4096'
        startup=time.monotonic()-start
        old=json.loads((ROOT/'doc/benchmark/run/perf-b1-client.json').read_text())['argv']
        tail=old[old.index('--backend'):]
        for name,count,seed in [('warmup',1,4100),('perf-b1',8,4101)]:
            cli=list(tail)
            for option,value in [('--num-prompts',count),('--seed',seed),('--result-dir',run),('--result-filename',name+'.json')]:cli[cli.index(option)+1]=str(value)
            execute([sys.executable,str(TESTS/'benchmark_client.py'),*cli],run/(name+'.log'))
            result=json.loads((run/(name+'.json')).read_text())
            assert result['completed']==count and result['failed']==0
            assert result['input_lens']==[4096]*count and result['output_lens']==[128]*count
            assert result['max_concurrency']==1
        result=json.loads((run/'perf-b1.json').read_text())
        baseline=json.loads((ROOT/'doc/benchmark/run/perf-b1.json').read_text())
        summary.update(status='measured',workload={'input_tokens':4096,'generated_tokens':128,'batch':1,'concurrent_requests':1},completed_requests=8,
            ttft_ms=result['mean_ttft_ms'],decode_tps=1000/result['mean_tpot_ms'],startup_seconds=startup,
            baseline={'ttft_ms':baseline['mean_ttft_ms'],'decode_tps':1000/baseline['mean_tpot_ms'],'source':'doc/benchmark/run/perf-b1.json'},
            policy='Same seeded exact random token-ID force-length performance protocol as stage11; ignore_eos, temperature0, no semantic accuracy score.',
            prefill_device_us=None,decode_device_us=None,missing_reasons={'device_times_rooflines':'No serving profiling; host stream metrics are not device measurements.'},evidence=['perf-b1.json','perf-b1-client.json','perf-b1-inputs.jsonl','observed/server.json'])
    except BaseException as error:
        errors.append(repr(error));summary['error']=repr(error)
    finally:
        try:
            if (run/'owned_server.json').exists():server.helper('stop',run/'owned_server.json')
        except BaseException as error:errors.append('server cleanup: '+repr(error))
        if relay is not None and relay.poll() is None:
            relay.terminate()
            try:relay.wait(timeout=10)
            except subprocess.TimeoutExpired:relay.kill();relay.wait()
        write(run/'summary.json',summary)
        write(run/'cleanup.json',{'errors':errors,'owned_state_remaining':(run/'owned_server.json').exists(),'elapsed_seconds':time.monotonic()-start,'mesh_hours':(time.monotonic()-start)/3600,'relay_exit':None if relay is None else relay.returncode})
    return 1 if errors else 0

if __name__=='__main__':sys.exit(main())
