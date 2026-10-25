# SPDX-License-Identifier: Apache-2.0
"""Semantic greedy comparison of both common Sampling1D strategies."""
import argparse,json,time,os
from pathlib import Path
import torch
import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator
p=argparse.ArgumentParser();p.add_argument('--persistent',action='store_true');p.add_argument('--async-ccl',action='store_true');p.add_argument('--output');p.add_argument('--batch',type=int,default=1);a=p.parse_args()
torch.set_num_threads(8)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
try:
    gen=build_generator('models/autoports/qwen_qwen3_8_27b',mesh,checkpoint='.cache/qwen38-probe',
        layer_indices=[0,3],context_length=512,batch_size=a.batch,prepare=False,persistent_sampling_ccl="async" if a.async_ccl else a.persistent)
    prompts=torch.arange(33).repeat(a.batch,1)+100
    output=gen.prefill_forward(prompts,page_table=gen.page_table,kv_cache=gen.kv_cache,prompt_lens=[33]*a.batch,return_device=True)
    ttnn.copy(ttnn.reshape(output,[1,1,a.batch,248320//4]),gen.logits)
    diagnostic_logits=gen.host_logits(gen.logits).reshape(a.batch,-1)
    expected=diagnostic_logits.argmax(-1).tolist()
    strategies={'topk_k1':lambda:gen._sample(gen.logits),
                'force_argmax':lambda:gen.sampler(gen.logits,tt_out_tok=gen.tokens)}
    for run in strategies.values():run()
    ttnn.synchronize_device(mesh)
    traces={}
    for name,run in strategies.items():
        trace=ttnn.begin_trace_capture(mesh,cq_id=0);run();ttnn.end_trace_capture(mesh,trace,cq_id=0)
        traces[name]=trace
    rows=[]
    for name,trace in traces.items():
        ttnn.execute_trace(mesh,trace,cq_id=0,blocking=True)
        actual=ttnn.to_torch(ttnn.get_device_tensors(gen.tokens)[0]).reshape(-1).tolist()
        assert all(diagnostic_logits[row,token]==diagnostic_logits[row].max() for row,token in enumerate(actual)),(name,actual,expected)
        start=time.monotonic()
        for _ in range(1000):ttnn.execute_trace(mesh,trace,cq_id=0,blocking=False)
        ttnn.synchronize_device(mesh)
        rows.append(dict(strategy=name,greedy_tokens=actual,matches_first_index_host_argmax=actual==expected,host_wall_us_per_replay=(time.monotonic()-start)*1000,iterations=1000))
    boundary_cases=[]
    for offset in (0,1):
        targets=[([7,62079,62080,124160,186240,248319][(row+offset)%6]) for row in range(a.batch)]
        # B1 explicitly visits every shard and the final valid vocabulary token.
        batches=[[target] for target in (7,62079,62080,124160,186240,248319)] if a.batch==1 else [targets]
        for expected_tokens in batches:
            logits=torch.zeros(1,1,a.batch,248320,dtype=torch.bfloat16)
            for row,token in enumerate(expected_tokens):logits[0,0,row,token]=100
            host=ttnn.from_torch(logits,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh,dim=-1))
            ttnn.copy_host_to_device_tensor(host,gen.logits)
            for name,trace in traces.items():
                ttnn.execute_trace(mesh,trace,cq_id=0,blocking=True)
                actual=ttnn.to_torch(ttnn.get_device_tensors(gen.tokens)[0]).reshape(-1).tolist()
                assert actual==expected_tokens,(name,actual,expected_tokens)
            boundary_cases.append(expected_tokens)
        if a.batch==1:break
    result=dict(scope='Reduced real layers0/3 and full terminal; sampler-only host-wall comparison, not full-model performance',
        synthetic_vocabulary_boundary_cases=boundary_cases,
        batch=a.batch,semantics='topk path uses k=1,p=0,temp=1; both select a diagnostic global maximum; first-index tie agreement recorded separately',rows=rows,
        fastest=min(rows,key=lambda r:r['host_wall_us_per_replay'])['strategy'],tracking=os.environ.get('TT_METAL_TRACE_ALLOC_TRACKING')=='1')
    Path(a.output or f'models/autoports/qwen_qwen3_8_27b/doc/full_model/sampling_b{a.batch}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
finally:ttnn.close_mesh_device(mesh)
