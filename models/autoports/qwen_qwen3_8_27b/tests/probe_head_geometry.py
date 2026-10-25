# SPDX-License-Identifier: Apache-2.0
"""Compare real hidden/weights with a same-precision common LM-head candidate."""
import argparse,json,time,os
from pathlib import Path
import torch,ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator
from models.autoports.qwen_qwen3_8_27b.tt.lm_head import build_head
p=argparse.ArgumentParser()
for key,default in [('batch',1),('chunk',8192),('readers',1),('block',5),('cores',8)]:p.add_argument('--'+key,type=int,default=default)
p.add_argument('--fidelity',choices=['HiFi4','HiFi2','LoFi'],default='HiFi4');p.add_argument('--generic-chunks',type=int,default=1);p.add_argument('--generic',action='store_true');p.add_argument('--grid-x',type=int,default=11);p.add_argument('--grid-y',type=int,default=10);p.add_argument('--subblock',type=int,default=3);p.add_argument('--output',required=True);a=p.parse_args()
torch.set_num_threads(8);ttnn.CONFIG.throw_exception_on_fallback=True
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
try:
    gen=build_generator('models/autoports/qwen_qwen3_8_27b',mesh,checkpoint='.cache/qwen38-probe',layer_indices=[0,3],batch_size=a.batch,context_length=512,prepare=False)
    terminal=gen.model.terminal;observed=[]
    def save(hidden,**kw):
        observed.append(hidden)
        return terminal(hidden,**kw)
    gen.model.terminal=save
    gen.prefill_forward(torch.full((a.batch,33),100,dtype=torch.int64),page_table=gen.page_table,kv_cache=gen.kv_cache,prompt_lens=[33]*a.batch,return_device=True)
    gen.model.terminal=terminal
    hidden=observed[-1]
    if hidden.shape[-1]==1280:hidden=gen.model.layers[0]._gather(hidden,('terminal',),compact_output=a.batch>1)
    hidden=ttnn.reshape(hidden,[1,1,a.batch,5120])
    hidden=ttnn.rms_norm(ttnn.to_memory_config(hidden,ttnn.DRAM_MEMORY_CONFIG),weight=gen.model.final_norm,epsilon=gen.model.config.rms_norm_eps,compute_kernel_config=gen.model.layers[0].ckc)
    candidate_compute=ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity,a.fidelity),
        math_approx_mode=False,fp32_dest_acc_en=True,packer_l1_acc=True)
    if a.generic:
        import math
        width=62080//a.generic_chunks
        assert 62080%a.generic_chunks==0 and width%32==0
        n=math.ceil(width/32/(a.grid_x*a.grid_y))
        pc=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=(a.grid_x,a.grid_y),
            in0_block_w=a.block,out_subblock_h=1,out_subblock_w=a.subblock,per_core_M=1,per_core_N=n,
            fuse_batch=False,fused_activation=None,mcast_in0=True)
        if a.generic_chunks==1:
            weights=[gen.model.head]
        else:
            raw=gen.model.checkpoint.tensor('lm_head.weight').T.contiguous()
            weights=[gen.model.put(torch.cat([raw[:,i*62080+offset:i*62080+offset+width] for i in range(4)],dim=-1).contiguous(),shard=1) for offset in range(0,62080,width)]
            del raw
        def candidate_fn():
            results=[ttnn.linear(hidden,w,compute_kernel_config=candidate_compute,program_config=pc,
                memory_config=ttnn.DRAM_MEMORY_CONFIG) for w in weights]
            return results[0] if len(results)==1 else ttnn.concat(results,dim=-1,memory_config=ttnn.DRAM_MEMORY_CONFIG)
    else:
        head=build_head(gen.model.checkpoint.tensor('lm_head.weight').T.contiguous(),mesh,candidate_compute,chunk=a.chunk,readers=a.readers,block=a.block,cores=a.cores)
        candidate_fn=lambda:head(ttnn.to_memory_config(hidden,head.config.input_memcfg))
    variants={'baseline':lambda:ttnn.linear(hidden,gen.model.head,compute_kernel_config=gen.model.layers[0].ckc,memory_config=ttnn.DRAM_MEMORY_CONFIG),'candidate':candidate_fn}
    outputs={name:gen.model.put(torch.zeros(1,1,a.batch,62080)) for name in variants}
    def run(name):ttnn.copy(variants[name](),outputs[name])
    for name in variants:run(name)
    ttnn.synchronize_device(mesh)
    baseline=gen.host_logits(outputs['baseline']).flatten();candidate=gen.host_logits(outputs['candidate']).flatten()
    pcc=float(torch.corrcoef(torch.stack([baseline,candidate]))[0,1]);assert pcc>.999,pcc
    traces={}
    for name in variants:
        trace=ttnn.begin_trace_capture(mesh,cq_id=0);run(name);ttnn.end_trace_capture(mesh,trace,cq_id=0);traces[name]=trace
    times={name:[] for name in variants}
    for order in [('baseline','candidate'),('candidate','baseline'),('baseline','candidate')]:
        for name in order:
            ttnn.execute_trace(mesh,traces[name],cq_id=0,blocking=True)
            started=time.monotonic()
            for _ in range(100):ttnn.execute_trace(mesh,traces[name],cq_id=0,blocking=False)
            ttnn.synchronize_device(mesh);times[name].append((time.monotonic()-started)*10000)
    result=dict(config=vars(a),pcc=pcc,argmax_equal=bool(torch.equal(baseline.reshape(a.batch,-1).argmax(-1),candidate.reshape(a.batch,-1).argmax(-1))),host_trace_us=times,scope='LM-head only, reduced real hidden, BF16 weights/output and FP32 accumulation; baseline HiFi4, candidate fidelity in config',tracking=os.getenv('TT_METAL_TRACE_ALLOC_TRACKING'))
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
finally:ttnn.close_mesh_device(mesh)
