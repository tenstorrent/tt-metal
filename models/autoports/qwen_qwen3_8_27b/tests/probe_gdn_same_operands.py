# SPDX-License-Identifier: Apache-2.0
"""Deferred eager GDN operator control on raw production operands; no semantic quality claim."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import torch
import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator
from models.autoports.qwen_qwen3_8_27b.tests.full_model_evidence import runtime_build

ROOT=Path('models/autoports/qwen_qwen3_8_27b')


def metrics(a,b,per_chunk=False):
    a=a.double();b=b.double()
    if per_chunk:
        assert a.shape[-2:]==(4096,128)
        a=a.reshape(-1,128,32,128).movedim(1,0).reshape(128,-1)
        b=b.reshape(-1,128,32,128).movedim(1,0).reshape(128,-1)
    else:
        a=a.reshape(1,-1);b=b.reshape(1,-1)
    delta=a-b
    ac=a-a.mean(-1,keepdim=True);bc=b-b.mean(-1,keepdim=True)
    denominator=(ac.square().sum(-1)*bc.square().sum(-1)).sqrt()
    pcc=(ac*bc).sum(-1)/denominator
    rows=[dict(max_abs=float(delta[i].abs().max()),rms=float(delta[i].square().mean().sqrt()),
               pcc=float(pcc[i]) if denominator[i]>0 else None,exact=bool(torch.equal(a[i],b[i])),
               finite=bool(torch.isfinite(a[i]).all() and torch.isfinite(b[i]).all())) for i in range(a.shape[0])]
    return rows if per_chunk else rows[0]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--layers',type=int,nargs='+',default=[0,3])
    p.add_argument('--checkpoint',default='.cache/qwen38-full')
    p.add_argument('--reference',default=str(ROOT/'doc/full_model/benchmark_hf_control_512.json'))
    p.add_argument('--output',default=str(ROOT/'doc/full_model/gdn_same_operands.json'))
    a=p.parse_args()
    assert a.layers in ([0,3],[0,1,2,3]),'Use the bounded reduced production prefix'
    reference=json.loads(Path(a.reference).read_text());prompt=reference['input_tokens']
    assert len(prompt)==4096
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
    source_files=[Path(__file__),*(ROOT/'tt').glob('*.py'),*Path('ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule').rglob('*.cpp'),
                  *Path('ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule').rglob('*.hpp')]
    result=dict(status='running',scope='Reduced production prefix/operator comparison, not full-model quality or performance',
        layers=a.layers,batch=1,context=262144,sequence=4096,chunk_size=32,num_chunks=128,
        input_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),runtime=runtime_build(),
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files},comparisons=[])
    def persist():target.write_text(json.dumps(result,indent=2)+'\n')
    torch.set_num_threads(8);ttnn.CONFIG.throw_exception_on_fallback=True
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
    original=ttnn.transformer.chunk_gated_delta_rule
    try:
        print('SAME_OPERAND_BUILD',a.layers,flush=True)
        gen=build_generator(ROOT,mesh,checkpoint=a.checkpoint,layer_indices=a.layers,
                            batch_size=1,context_length=262144,prepare=False)
        linear_indices=[i for i,layer in zip(gen.model.layer_indices,gen.model.layers) if layer.layer_type=='linear_attention']
        call=0
        def compare(*operands,**kwargs):
            nonlocal call
            layer=linear_indices[call];call+=1
            assert [list(t.shape) for t in operands]==[[1,4096,512],[1,4096,512],[1,4096,1536],[1,4096,12],[1,4096,12]]
            assert kwargs['chunk_size']==32 and kwargs['output_head_major'] and kwargs['output_final_state']
            assert kwargs.get('program_config') is None and kwargs['initial_state'].dtype==ttnn.float32
            # Flat Q/K selects internal L2 normalization in the production op;
            # never reshape to pre-normalized head tensors for this comparison.
            record=dict(layer=layer,operand_specs=[dict(shape=list(t.shape),dtype=str(t.dtype),layout=str(t.layout)) for t in operands],
                initial_state_shape=list(kwargs['initial_state'].shape),initial_state_dtype=str(kwargs['initial_state'].dtype),
                compute_config=str(kwargs['compute_kernel_config']),inverse=str(kwargs.get('wy_inverse','AUTO')),
                qk_normalization='Raw flat Q/K; production in-kernel L2 norm',runs={},pairs={})
            initial=ttnn.clone(kwargs['initial_state'])
            configs={'auto':None,'fused_nv2_np7':ttnn.ChunkGdnFusedProgramConfig(num_receivers=2,num_producers=7),
                     'phased':ttnn.ChunkGdnPhasedProgramConfig()}
            hosts={};production=None
            for name,config in configs.items():
                print('SAME_OPERAND_GDN',layer,name,flush=True)
                options={**kwargs,'initial_state':ttnn.clone(initial),'program_config':config}
                ttnn.synchronize_device(mesh)
                ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
                try:
                    output,state=original(*operands,**options)
                    ttnn.synchronize_device(mesh)
                finally:
                    graph=ttnn.graph.end_graph_capture()
                names=sorted({node.get('params',{}).get('name','') for node in graph
                              if node.get('node_type')=='function_start' and 'gdn' in node.get('params',{}).get('name','').lower()})
                hosts[name]=([ttnn.to_torch(t).clone() for t in ttnn.get_device_tensors(output)],
                             [ttnn.to_torch(t).clone() for t in ttnn.get_device_tensors(state)])
                assert len(hosts[name][0])==len(hosts[name][1])==4
                record['runs'][name]=dict(program_config='AUTO' if config is None else str(config),
                    observed_graph_gdn_functions=names,
                    identity_scope='Observed graph operation names; not a profiler kernel identity',
                    output_dtype=str(output.dtype),state_dtype=str(state.dtype))
                if name=='auto':production=(output,state)
                else:del output,state
                del options
            for left,right in itertools.combinations(configs,2):
                record['pairs'][left+'__'+right]=[
                    dict(rank=rank,output=metrics(hosts[left][0][rank],hosts[right][0][rank]),
                         output_per_chunk=metrics(hosts[left][0][rank],hosts[right][0][rank],True),
                         final_state=metrics(hosts[left][1][rank],hosts[right][1][rank])) for rank in range(4)]
            result['comparisons'].append(record);persist()
            return production
        ttnn.transformer.chunk_gated_delta_rule=compare
        print('SAME_OPERAND_PREFILL',flush=True)
        gen.reset()
        meta=gen._set_bucket(4096,torch.tensor([prompt]),[4096],0,[0])
        hidden=gen.model.prefill_forward(meta['tokens'],positions=meta['positions'],kv_cache=gen.kv_cache,
            page_table=gen.page_table,batch=1,length=4096,prepared_metadata=meta,return_hidden=True)
        ttnn.synchronize_device(mesh);del hidden
        assert call==len(linear_indices)
        result.update(status='complete',limitations=[
            'No autoregressive semantic conclusion: only selected reduced-prefix GDN operands are compared.',
            'No traces, device timing, profiler utilization, or production policy changes.',
            'Auto identity is recorded from graph operation names; physical fused geometry is explicit only for pinned control.'])
        print('SAME_OPERAND_COMPLETE',flush=True)
    except BaseException as error:
        result.update(status='fail',error_type=type(error).__name__,error=str(error)[:2000]);raise
    finally:
        ttnn.transformer.chunk_gated_delta_rule=original
        persist();ttnn.close_mesh_device(mesh)


if __name__=='__main__':main()
