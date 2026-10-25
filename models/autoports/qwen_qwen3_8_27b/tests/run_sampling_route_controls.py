# SPDX-License-Identifier: Apache-2.0
"""HTTP-only controls: canonical device RNG reorder, then explicit host logits.

Run only against the parent's already-running serialized server. No hardware
opening, profiling, benchmark claims, or server lifecycle changes.
"""
import argparse
import asyncio
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import sys
import time

from openai import AsyncOpenAI

utils_path=Path('bringup/dependencies/vllm-tt-plugin/tests/tt/utils.py')
spec=importlib.util.spec_from_file_location('qwen_sampling_controls_utils',utils_path)
utils=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=utils
spec.loader.exec_module(utils)
RequestConfig=utils.RequestConfig


def event_snapshot(path):
    if path is None:
        return None
    rows=Path(path).read_text().splitlines()
    return json.loads(rows[-1]) if rows else None


async def run(args,result):
    configs=[RequestConfig('Count: ',max_tokens=5,temperature=0,top_k=1),
        RequestConfig('Random: ',max_tokens=5,temperature=2,top_k=32,seed=42),
        RequestConfig('Word: ',max_tokens=3,temperature=2,top_k=1,seed=99),
        RequestConfig('Letter: ',max_tokens=3,temperature=.01,top_k=32,seed=42),
        RequestConfig('Number: ',max_tokens=3,temperature=2,top_k=5,seed=42),
        RequestConfig('TopP: ',max_tokens=8,temperature=1,top_k=32,top_p=.5,seed=7)]
    async with AsyncOpenAI(base_url=args.base_url,api_key='EMPTY',timeout=300) as client:
        model=args.model or (await client.models.list()).data[0].id
        result['model']=model
        result['configs']=[asdict(c) for c in configs]
        result['device']={'before':event_snapshot(args.events),'runs':[]}
        for order in (list(range(6)),list(reversed(range(6))),[2,4,1,5,0,3]):
            responses=await utils.send_batch_concurrent(client,model,[configs[i] for i in order],return_full_response=True)
            run_data=[None]*6
            for i,response in zip(order,responses):
                run_data[i]=response.choices[0].text
            result['device']['runs'].append({'order':order,'outputs':run_data})
        result['device']['after']=event_snapshot(args.events)
        reference=result['device']['runs'][0]['outputs']
        result['device']['reproduces']=all(r['outputs']==reference for r in result['device']['runs'])
        before=result['device']['before'];after=result['device']['after']
        result['device']['host_read_delta']=(after['host_logits_reads']-before['host_logits_reads']
                                            if before is not None and after is not None else None)
        # Logprobs intentionally force host compatibility. Compare distributions
        # at the identical first-token prefix, avoiding divergent continuation input.
        logits_configs=[replace(c,max_tokens=1,temperature=0,top_k=1,logprobs=20,
                                return_tokens_as_token_ids=True) for c in configs]
        result['host_logprobs']={'scope':'explicit compatibility, not device RNG or performance','runs':[]}
        for order in (list(range(6)),list(reversed(range(6)))):
            responses=await utils.send_batch_concurrent(client,model,[logits_configs[i] for i in order],return_full_response=True)
            run_data=[None]*6
            for i,response in zip(order,responses):
                run_data[i]=response.choices[0].model_dump()
            result['host_logprobs']['runs'].append(run_data)
        comparisons=[]
        a,b=result['host_logprobs']['runs']
        for i,(x,y) in enumerate(zip(a,b)):
            xp=x['logprobs'];yp=y['logprobs']
            left=xp['top_logprobs'][0];right=yp['top_logprobs'][0]
            common=set(left)&set(right)
            comparisons.append({'index':i,'tokens_equal':xp['tokens']==yp['tokens'],
                'top20_ids_equal':set(left)==set(right),'common_count':len(common),
                'max_abs_logprob_difference':max((abs(left[k]-right[k]) for k in common),default=None)})
        result['host_logprobs']['comparisons']=comparisons
        result['status']='pass' if (result['device']['reproduces'] and
            result['device']['host_read_delta']==0 and all(c['tokens_equal'] and c['common_count'] and
                c['max_abs_logprob_difference']<=.02 for c in comparisons)) else 'more-work-needed'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://localhost:8000/v1')
    parser.add_argument('--model')
    parser.add_argument('--events')
    parser.add_argument('--output',default='models/autoports/qwen_qwen3_8_27b/readiness_vllm/sampling_route_controls.json')
    args=parser.parse_args()
    result={'scope':'full-model HTTP correctness controls; no performance claim','started_at':time.time()}
    try:
        asyncio.run(run(args,result))
    except BaseException as error:
        result['status']='error';result['error']=repr(error)
        raise
    finally:
        Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    if result['status']!='pass':
        raise SystemExit('Controls need investigation; see '+args.output)
