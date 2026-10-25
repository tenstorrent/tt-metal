# SPDX-License-Identifier: Apache-2.0
"""Deferred same-history greedy decision and long-quality controls; never a benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
import ttnn
from models.autoports.qwen_qwen3_8_27b.tests.full_model_evidence import runtime_build

ROOT=Path('models/autoports/qwen_qwen3_8_27b')


def describe(logits,actual):
    values=logits.float().reshape(-1)
    maximum=values.max()
    ties=torch.where(values==maximum)[0].tolist()
    scores,indices=torch.topk(values,100)
    return dict(actual_sampler_token=int(actual),host_argmax=int(values.argmax()),
        maximum=float(maximum),maximum_tie_count=len(ties),maximum_tie_ids=ties[:256],
        maximum_tie_ids_truncated=len(ties)>256,actual_is_global_max=bool(values[actual]==maximum),
        tokens={str(token):dict(value=float(values[token]),rank_strict=1+int((values>values[token]).sum()),
                    rank_lowest_id_tiebreak=1+int((values>values[token]).sum())+int((values[:token]==values[token]).sum()),
                    margin_from_max=float(maximum-values[token])) for token in (1,279)},
        top100_ids=indices.tolist(),top100_values=scores.tolist())


def sampler_only(mesh,path):
    from models.common.modules.sampling.sampling_1d import Sampling1D
    from models.common.modules.tt_ccl import TT_CCL
    saved=torch.load(path,map_location='cpu',weights_only=True)
    host=(saved['logits'] if isinstance(saved,dict) else saved).to(torch.bfloat16).reshape(1,1,1,-1)
    assert host.shape[-1]==248320
    logits=ttnn.from_torch(host,device=mesh,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,
                          mesh_mapper=ttnn.ShardTensorToMesh(mesh,dim=-1))
    def put(value,dtype):
        return ttnn.from_torch(value,device=mesh,dtype=dtype,layout=ttnn.ROW_MAJOR_LAYOUT,
                              mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    tokens=put(torch.zeros(1,1,1,1,dtype=torch.int32),ttnn.uint32)
    k=put(torch.ones(1,dtype=torch.int32),ttnn.uint32)
    p=put(torch.zeros(1),ttnn.bfloat16);temp=put(torch.ones(1),ttnn.bfloat16)
    seeds=put(torch.arange(32,dtype=torch.int32)+1234,ttnn.uint32)
    sampler=Sampling1D(vocab_size=248320,mesh_device=mesh,tt_ccl=TT_CCL(mesh),
                       max_batch_size=32,max_top_k=32,allow_force_argmax=True,pad_to_power_of_2=True)
    sampler.load_device_buffers()
    strategies={'topk_k1':lambda:sampler(logits,k=k,p=p,temp=temp,seeds=seeds,tt_out_tok=tokens),
                'force_argmax':lambda:sampler(logits,tt_out_tok=tokens)}
    for run in strategies.values():
        run()
    ttnn.synchronize_device(mesh)
    traces={}
    try:
        for name,run in strategies.items():
            trace=ttnn.begin_trace_capture(mesh,cq_id=0)
            run()
            ttnn.end_trace_capture(mesh,trace,cq_id=0)
            traces[name]=trace
        rows={}
        for name,trace in traces.items():
            ttnn.execute_trace(mesh,trace,cq_id=0,blocking=True)
            token=int(ttnn.to_torch(ttnn.get_device_tensors(tokens)[0]).reshape(-1)[0])
            rows[name]=describe(host,token)
        return dict(scope='Same saved BF16 full-vocabulary logits; no model/checkpoint load; common samplers only',
                    artifact=str(path),artifact_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),strategies=rows)
    finally:
        for trace in traces.values():
            ttnn.release_trace(mesh,trace)


def install_diagnostic_variant(variant):
    """Process-local call guards; never modify production files or dtype policy."""
    from models.autoports.qwen_qwen3_8_27b.tt.multichip_decoder import MultichipDecoder
    original_gdn=ttnn.transformer.chunk_gated_delta_rule
    original_linear=MultichipDecoder._linear_impl
    record=dict(name=variant,diagnostic_only=True,explicit_phased_calls=0,separate_projection_calls={},
        invariants='All64, TP4, B1/context262144, identical HF4096 IDs, C32, AUTO inverse, original compute configs/dtypes/KV/residual/decode strategy',
        caveat='Separate projection changes matmul blocking and BF16-then-BF8 rounding versus native direct-BF8; not a pure fusion toggle')
    if variant in ('phased_gdn','combined'):
        def phased(*operands,**kwargs):
            if list(operands[0].shape)[:2]==[1,4096]:
                assert kwargs['chunk_size']==32 and kwargs.get('program_config') is None
                assert kwargs.get('output_head_major') is True
                assert kwargs['initial_state'].dtype==ttnn.float32
                kwargs['program_config']=ttnn.ChunkGdnPhasedProgramConfig()
                record['explicit_phased_calls']+=1
                if 'gdn_operands' not in record:
                    record['gdn_operands']=[dict(shape=list(t.shape),dtype=str(t.dtype),layout=str(t.layout)) for t in operands]
                    record['gdn_compute_config']=str(kwargs.get('compute_kernel_config'))
                    record['gdn_initial_state_dtype']=str(kwargs['initial_state'].dtype)
                    record['gdn_wy_inverse']=str(kwargs.get('wy_inverse','AUTO (unchanged default)'))
            return original_gdn(*operands,**kwargs)
        ttnn.transformer.chunk_gated_delta_rule=phased
    if variant in ('separate_prefill_rs','combined'):
        def separate(layer,x,w,activation=None,**kwargs):
            role=layer.weight_roles[id(w)]
            selected=(layer.policy.get('_prefill_phase') and layer.policy.get('prefill_native_rs')
                      and role in ('out','down') and list(x.shape)[:2]==[1,4096])
            if not selected:
                return original_linear(layer,x,w,activation,**kwargs)
            record['separate_projection_calls'][role]=record['separate_projection_calls'].get(role,0)+1
            record.setdefault('projection_operands',{}).setdefault(role,
                dict(input_shape=list(x.shape),input_dtype=str(x.dtype),weight_dtype=str(w.dtype),
                     compute_config=str(layer.matmul_ckc),prefill_ccl_dtype=layer.policy.get('prefill_ccl_dtype')))
            # _linear's workspace require/complete surrounds this call; restore
            # the flag before returning to that owner. Decode never enters here.
            accepted=layer.policy['prefill_native_rs']
            layer.policy['prefill_native_rs']=False
            try:
                return original_linear(layer,x,w,activation,**kwargs)
            finally:
                layer.policy['prefill_native_rs']=accepted
        MultichipDecoder._linear_impl=separate
    def restore():
        ttnn.transformer.chunk_gated_delta_rule=original_gdn
        MultichipDecoder._linear_impl=original_linear
    return record,restore


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',default=str(ROOT/'doc/full_model/benchmark_hf_control_512.json'))
    parser.add_argument('--checkpoint',default='.cache/qwen38-full')
    parser.add_argument('--output',default=str(ROOT/'doc/full_model/long_quality_decision.json'))
    parser.add_argument('--logits-output',default='.cache/qwen38-y56-logits.pt')
    parser.add_argument('--sampler-only-logits')
    parser.add_argument('--variant',choices=('production','phased_gdn','separate_prefill_rs','combined'),default='production')
    parser.add_argument('--skip-host-control',action='store_true')
    args=parser.parse_args()
    if args.variant!='production':
        if args.sampler_only_logits:
            parser.error('Math variants do not apply to sampler-only controls')
        if args.output==str(ROOT/'doc/full_model/long_quality_decision.json'):
            args.output=str(ROOT/'doc/full_model'/f'long_quality_{args.variant}.json')
        if args.logits_output=='.cache/qwen38-y56-logits.pt':
            args.logits_output=f'.cache/qwen38-y56-{args.variant}-logits.pt'
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__),*(ROOT/'tt').glob('*.py'),Path('models/common/modules/sampling/sampling_1d.py'),
             Path('models/common/modules/tt_ccl.py')]
    for directory in ('ttnn/cpp/ttnn/operations/reduction/topk','ttnn/cpp/ttnn/operations/reduction/sampling',
                      'ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding',
                      'ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule',
                      'ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_async',
                      'ttnn/cpp/ttnn/operations/experimental/matmul/ccl_fusion'):
        sources.extend(p for p in Path(directory).rglob('*') if p.suffix in ('.cpp','.hpp','.h'))
    result=dict(status='running',variant=args.variant,scope='Diagnostic only; no headline performance evidence',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        runtime=runtime_build(),tracking=os.environ.get('TT_METAL_TRACE_ALLOC_TRACKING'),
        skip_program_cache=os.environ.get('TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE'),
        torch_version=str(torch.__version__))
    def persist():
        target.write_text(json.dumps(result,indent=2)+'\n')
    torch.set_num_threads(8)
    ttnn.CONFIG.throw_exception_on_fallback=True
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
    restore_variant=lambda:None
    try:
        if args.sampler_only_logits:
            print('QUALITY_SAMPLER_ONLY',flush=True)
            result['sampler_only']=sampler_only(mesh,args.sampler_only_logits)
        else:
            from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator
            reference=json.loads(Path(args.reference).read_text())
            prompt=reference['input_tokens'];hf=reference['output_tokens']
            assert len(prompt)==4096 and len(hf)>=512
            assert hf[56]==1
            result.update(reference=args.reference,reference_sha256=hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),
                input_tokens_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                input_tokens=prompt,batch=1,context=262144,layers=64,
                preparation_scope='Default complete bucket domain, matching failing benchmark',sampling=dict(k=1,p=0,temperature=1),
                eos_policy='Normal stop_on_eos=True for both free-running controls; same generator/checkpoint EOS IDs')
            result['variant_config'],restore_variant=install_diagnostic_variant(args.variant)
            print('QUALITY_BUILD_ALL64',args.variant,flush=True)
            gen=build_generator(ROOT,mesh,checkpoint=args.checkpoint,batch_size=1,context_length=262144,prepare=False)
            assert len(gen.model.layers)==64
            gen.prepare_decode()
            print('QUALITY_SHARED_HISTORY_57',flush=True)
            observed=gen.generate(prompt,57,next_input=lambda step,token:hf[step],stop_on_eos=True)
            ttnn.synchronize_device(mesh)
            # Callback replaced tokens with HF y56, but output[-1] is actual TT y56.
            logits=gen.host_logits(gen.logits).reshape(-1).to(torch.bfloat16)
            positions=ttnn.to_torch(ttnn.get_device_tensors(gen.positions)[0]).reshape(-1).tolist()
            rope=ttnn.to_torch(ttnn.get_device_tensors(gen.rope_indices)[0]).reshape(-1).tolist()
            feedback=ttnn.to_torch(ttnn.get_device_tensors(gen.tokens)[0]).reshape(-1).tolist()
            decision=describe(logits,observed[-1])
            decision.update(output_index=56,input_position=4151,positions_after_replay=positions,
                rope_indices_after_replay=rope,feedback_after_callback=feedback,hf_expected_token=hf[56],
                consumed_token=hf[55],tt_output_tokens=observed,hf_teacher_tokens=hf[:57],
                prefix_first56_exact=observed[:56]==hf[:56],metrics_diagnostic_only=dict(gen.last_generation_metrics))
            result['shared_history_decision']=decision
            saved=Path(args.logits_output);saved.parent.mkdir(parents=True,exist_ok=True)
            torch.save(dict(logits=logits,input_tokens=torch.tensor(prompt),history=torch.tensor(hf[:56]),
                            output_index=56,actual_token=int(observed[-1]),hf_token=int(hf[56])),saved)
            result['logits_artifact']=dict(path=str(saved),sha256=hashlib.sha256(saved.read_bytes()).hexdigest(),dtype='BF16')
            persist()
            assert len(observed)==57
            if args.variant=='production':
                assert observed[:56]==hf[:56], 'Earlier divergence than the recorded shared prefix'
            # Math controls may differ earlier; their forced input history is still HF-exact.
            assert positions==[4152] and rope==[4152] and feedback==[hf[56]]
            controls=[('device_512',False)]
            if args.variant=='production' and not args.skip_host_control:
                controls.append(('host_512',True))
            for name,host_sampling in controls:
                print('QUALITY_'+name.upper(),flush=True)
                output=gen.generate(prompt,512,host_sampling=host_sampling,stop_on_eos=True)
                ttnn.synchronize_device(mesh)
                result[name]=dict(tokens=output,text=gen.tokenizer.decode(output,skip_special_tokens=False),
                    eos_ids=sorted(gen.eos_token_ids),metrics_diagnostic_only=dict(gen.last_generation_metrics))
                persist()
            if 'host_512' in result:
                left=result['device_512']['tokens'];right=result['host_512']['tokens']
                result['controls_first_difference']=next((i for i,(x,y) in enumerate(zip(left,right)) if x!=y),
                                                         min(len(left),len(right)) if len(left)!=len(right) else None)
        result['status']='complete'
        result['quality_verdict']='Not inferred from diagnostic completion; inspect saved controls and decision evidence'
        print('QUALITY_DIAGNOSTIC_COMPLETE',flush=True)
    except BaseException as error:
        result.update(status='fail',error_type=type(error).__name__,error=str(error)[:2000])
        raise
    finally:
        persist()
        try:
            ttnn.close_mesh_device(mesh)
        finally:
            restore_variant()


if __name__=='__main__':
    main()
