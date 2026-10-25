# SPDX-License-Identifier: Apache-2.0
"""Deferred all64 native-fused prefill K-block diagnostic; never automatic policy adoption."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator
from models.autoports.qwen_qwen3_8_27b.tests.full_model_evidence import runtime_build
from models.autoports.qwen_qwen3_8_27b.tests.probe_long_quality_decision import describe

ROOT=Path('models/autoports/qwen_qwen3_8_27b')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--variants',choices=('both','out','down','out16'),nargs='+',default=['both','out','down'])
    p.add_argument('--checkpoint',default='.cache/qwen38-full')
    p.add_argument('--reference',default=str(ROOT/'doc/full_model/benchmark_hf_control_512.json'))
    p.add_argument('--baseline-artifact',default=str(ROOT/'doc/full_model/long_quality_decision.json'))
    p.add_argument('--run-baseline',action='store_true')
    p.add_argument('--output',default=str(ROOT/'doc/full_model/native_prefill_geometry.json'))
    a=p.parse_args();assert len(set(a.variants))==len(a.variants)
    ref=json.loads(Path(a.reference).read_text());prompt=ref['input_tokens'];hf=ref['output_tokens']
    assert len(prompt)==4096 and len(hf)>=57
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__),ROOT/'tests/probe_long_quality_decision.py',*(ROOT/'tt').glob('*.py')]
    runtime=runtime_build()
    prompt_hash=hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
    result=dict(status='running',scope='All64 B1/context262144 native fused geometry quality diagnostic; no performance/policy adoption',
        input_sha256=prompt_hash,reference=a.reference,runtime=runtime,
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},variants={},
        invariant_contract='Only prefill_rs_geometry out_k/down_k change; accepted native fused op, BF4 weights, BF16 activations, LoFi/nonFP32 compute, BF8 MM/CCL, KV/cache/residual/decode strategy remain')
    def persist():target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    torch.set_num_threads(8);ttnn.CONFIG.throw_exception_on_fallback=True
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
    original_native=ttnn.experimental.matmul_reduce_scatter_async
    gen=None;original_geometries=[];phase='setup';calls={}
    try:
        print('GEOMETRY_BUILD_ALL64',flush=True)
        gen=build_generator(ROOT,mesh,checkpoint=a.checkpoint,batch_size=1,context_length=262144,prepare=False)
        assert len(gen.model.layers)==64
        views=[getattr(layer,'_prefill_impl',layer) for layer in gen.model.layers]
        original_geometries=[(view,dict(view.policy['prefill_rs_geometry'])) for view in views]
        for view,geometry in original_geometries:
            assert view.policy['prefill_native_rs'] and geometry['out_k']==48 and geometry['down_k']==17
        accepted_compute_configs={id(view.matmul_ckc) for view in views}
        def count_native(*args,**kwargs):
            assert list(args[0].shape)[:3]==[1,1,4096]
            assert args[0].dtype==ttnn.bfloat16 and args[1].dtype==ttnn.bfloat4_b
            assert id(kwargs['compute_kernel_config']) in accepted_compute_configs
            assert kwargs['dtype']==ttnn.bfloat8_b and kwargs['num_links']==2 and kwargs['topology']==ttnn.Topology.Ring
            role='out' if args[0].shape[-1]==1536 else 'down'
            assert args[0].shape[-1] in (1536,4352)
            calls.setdefault(phase,{'out':0,'down':0})[role]+=1
            return original_native(*args,**kwargs)
        ttnn.experimental.matmul_reduce_scatter_async=count_native
        def retire():
            ttnn.synchronize_device(mesh)
            for trace in {*gen.decode_traces.values(),gen.sample_trace,gen.buffered_sample_trace}:
                if trace is not None:ttnn.release_trace(mesh,trace)
            gen.decode_traces.clear();gen.model_trace=gen.sample_trace=gen.buffered_sample_trace=None;gen.prepared_buckets.clear()
        def decision():
            forced=[]
            def teacher(step,token):forced.append(hf[step]);return hf[step]
            tokens=gen.generate(prompt,57,next_input=teacher,stop_on_eos=True)
            ttnn.synchronize_device(mesh)
            logits=gen.host_logits(gen.logits).reshape(-1)
            positions=ttnn.to_torch(ttnn.get_device_tensors(gen.positions)[0]).reshape(-1).tolist()
            rope=ttnn.to_torch(ttnn.get_device_tensors(gen.rope_indices)[0]).reshape(-1).tolist()
            feedback=ttnn.to_torch(ttnn.get_device_tensors(gen.tokens)[0]).reshape(-1).tolist()
            assert len(tokens)==57 and forced==hf[:57] and positions==rope==[4152] and feedback==[hf[56]]
            row=describe(logits,tokens[-1]);row.update(tokens=tokens,forced_tokens=forced,positions=positions,
                rope_indices=rope,feedback=feedback,prefix_first56_matches_hf=tokens[:56]==hf[:56],
                metrics_diagnostic_only=dict(gen.last_generation_metrics))
            return row
        def completion():
            tokens=gen.generate(prompt,512,stop_on_eos=True)
            ttnn.synchronize_device(mesh)
            return dict(tokens=tokens,text=gen.tokenizer.decode(tokens,skip_special_tokens=False),
                eos_ids=sorted(gen.eos_token_ids),stop_on_eos=True,metrics_diagnostic_only=dict(gen.last_generation_metrics))
        print('GEOMETRY_PREPARE_PRODUCTION_COMPLETE_DOMAIN',flush=True)
        phase='production_prepare';gen.prepare_decode()
        result['production_prepared_buckets']=sorted(gen.prepared_buckets)
        if a.run_baseline:
            phase='production_y56';baseline_decision=decision()
            phase='production_512';baseline_output=completion()
            result['baseline']=dict(source='new diagnostic',decision=baseline_decision,completion=baseline_output)
        else:
            path=Path(a.baseline_artifact);baseline=json.loads(path.read_text())
            assert baseline['input_tokens_sha256']==prompt_hash and baseline['batch']==1 and baseline['context']==262144 and baseline['layers']==64
            mismatches=[str(f) for f in (ROOT/'tt').glob('*.py')
                        if baseline.get('source_sha256',{}).get(str(f))!=hashlib.sha256(f.read_bytes()).hexdigest()]
            assert not mismatches,('Baseline model-source mismatch: use --run-baseline',mismatches)
            assert baseline['runtime']['cpp_library_sha256']==runtime['cpp_library_sha256'], 'Backend differs: use --run-baseline'
            result['baseline']=dict(source='existing exact prompt/model/backend evidence',artifact=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),completion_key='device_512',decision_key='shared_history_decision')
        def policy_hash(view):
            value={k:v for k,v in view.policy.items() if k!='prefill_rs_geometry'}
            return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()
        accepted_policy_hashes=[policy_hash(view) for view in views]
        result['policy_except_geometry_sha256']=accepted_policy_hashes
        persist()
        for variant in a.variants:
            print('GEOMETRY_RETIRE_BEFORE_VARIANT',variant,flush=True);retire()
            changes={'both':{'out_k':4,'down_k':4},'out':{'out_k':4},'down':{'down_k':4},'out16':{'out_k':16}}[variant]
            for view,accepted in original_geometries:
                geometry={**accepted,**changes}
                view.policy['prefill_rs_geometry']=geometry
                assert {k:v for k,v in geometry.items() if k not in changes}=={k:v for k,v in accepted.items() if k not in changes}
                assert view.policy['prefill_native_rs']
            record=dict(geometry=views[0].policy['prefill_rs_geometry'],changed_keys=changes,
                out_k_blocks=48//views[0].policy['prefill_rs_geometry']['out_k'],
                down_k_blocks=136//views[0].policy['prefill_rs_geometry']['down_k'],
                native_fused=True,prepared_prefill_lengths=[4096],policy_adopted=False)
            result['variants'][variant]=record
            phase=variant+'_prepare';gen.prepare_decode(prefill_lengths=(4096,))
            trace_ids={str(k):str(v) for k,v in gen.decode_traces.items()};sample_id=str(gen.sample_trace)
            print('GEOMETRY_Y56',variant,flush=True)
            phase=variant+'_y56';record['decision']=decision();persist()
            print('GEOMETRY_512',variant,flush=True)
            phase=variant+'_512';record['completion']=completion()
            assert trace_ids=={str(k):str(v) for k,v in gen.decode_traces.items()} and sample_id==str(gen.sample_trace)
            assert calls[variant+'_y56']==calls[variant+'_512']=={'out':64,'down':64}
            assert [policy_hash(view) for view in views]==accepted_policy_hashes
            record['other_policy_fields_unchanged']=True
            record['native_call_counts']={key:dict(value) for key,value in calls.items() if key.startswith(variant+'_')}
            record['trace_ids']=dict(model=trace_ids,sampling=sample_id)
            persist()
        result.update(status='complete',native_call_counts=calls,quality_verdict='Read raw completions; diagnostic completion is not a stage pass or policy adoption')
        print('GEOMETRY_COMPLETE',flush=True)
    except BaseException as error:
        result.update(status='fail',error_type=type(error).__name__,error=str(error)[:2000],native_call_counts=calls);raise
    finally:
        ttnn.experimental.matmul_reduce_scatter_async=original_native
        if gen is not None:
            ttnn.synchronize_device(mesh)
            for trace in {*gen.decode_traces.values(),gen.sample_trace,gen.buffered_sample_trace}:
                if trace is not None:ttnn.release_trace(mesh,trace)
            for view,geometry in original_geometries:view.policy['prefill_rs_geometry']=geometry
        persist();ttnn.close_mesh_device(mesh)


if __name__=='__main__':main()
