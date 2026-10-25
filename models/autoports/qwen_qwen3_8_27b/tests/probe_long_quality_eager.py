# SPDX-License-Identifier: Apache-2.0
"""Deferred all64 same-history traced/eager decode decision control; not performance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator
from models.autoports.qwen_qwen3_8_27b.tests.full_model_evidence import runtime_build
from models.autoports.qwen_qwen3_8_27b.tests.probe_long_quality_decision import describe,install_diagnostic_variant
from models.autoports.qwen_qwen3_8_27b.tests.probe_prefill_rs_same_operands import metrics

ROOT=Path('models/autoports/qwen_qwen3_8_27b')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='.cache/qwen38-full')
    p.add_argument('--reference',default=str(ROOT/'doc/full_model/benchmark_hf_control_512.json'))
    p.add_argument('--output',default=str(ROOT/'doc/full_model/long_quality_eager.json'))
    p.add_argument('--logits-output',default='.cache/qwen38-y56-traced-eager.pt')
    p.add_argument('--separate-prefill',action='store_true')
    a=p.parse_args()
    ref=json.loads(Path(a.reference).read_text());prompt=ref['input_tokens'];hf=ref['output_tokens']
    assert len(prompt)==4096 and len(hf)>=57
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__),*(ROOT/'tt').glob('*.py'),ROOT/'tests/probe_long_quality_decision.py',
             ROOT/'tests/probe_prefill_rs_same_operands.py',ROOT/'tests/probe_gdn_same_operands.py',
             Path('models/common/modules/sampling/sampling_1d.py')]
    result=dict(status='running',scope='All64 production B1/context262144 same-HF-history decode dispatch control; not performance',
        input_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),reference=a.reference,
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},runtime=runtime_build(),
        tracking=os.environ.get('TT_METAL_TRACE_ALLOC_TRACKING'),skip_program_cache=os.environ.get('TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE'),
        invariants='Production dtype/fidelity/KV/CCL/residual/prefill and common sampling trace unchanged; only model trace dispatch bypassed in eager control')
    def persist():target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    torch.set_num_threads(8);ttnn.CONFIG.throw_exception_on_fallback=True
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
    original_execute=ttnn.execute_trace
    restore_variant=lambda:None
    try:
        print('EAGER_CONTROL_BUILD_ALL64',flush=True)
        gen=build_generator(ROOT,mesh,checkpoint=a.checkpoint,batch_size=1,context_length=262144,prepare=False)
        assert len(gen.model.layers)==64
        gen.prepare_decode()  # Complete production bucket domain, not a shortened model.
        def trace_ids():
            return dict(model=str(gen.model_trace),sampling=str(gen.sample_trace),
                        decode={str(k):str(v) for k,v in gen.decode_traces.items()})
        ids=trace_ids();captures=gen.counters['captures']
        def decision(label):
            forced=[]
            def teacher(step,token):
                forced.append(hf[step]);return hf[step]
            output=gen.generate(prompt,57,next_input=teacher,stop_on_eos=True)
            ttnn.synchronize_device(mesh)
            logits=gen.host_logits(gen.logits).reshape(-1).to(torch.bfloat16)
            def state(t):return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).reshape(-1).tolist()
            row=describe(logits,output[-1])
            row.update(tokens=output,forced_tokens=forced,positions=state(gen.positions),rope_indices=state(gen.rope_indices),
                       token_feedback=state(gen.tokens),trace_ids=trace_ids(),metrics_diagnostic_only=dict(gen.last_generation_metrics))
            result[label]=row;persist()
            assert len(output)==57 and forced==hf[:57]
            assert row['positions']==row['rope_indices']==[4152] and row['token_feedback']==[hf[56]]
            return logits
        print('EAGER_CONTROL_TRACED57',flush=True)
        traced=decision('traced')
        replaced=0
        def eager_model_only(device,trace,*args,**kwargs):
            nonlocal replaced
            if trace==gen.model_trace:
                assert device is mesh and not args and kwargs.get('cq_id',0)==0
                replaced+=1
                value=gen._decode(masked=False)
                if kwargs.get('blocking',False):ttnn.synchronize_device(mesh)
                return value
            return original_execute(device,trace,*args,**kwargs)
        print('EAGER_CONTROL_EAGER57_SAMPLER_TRACED',flush=True)
        ttnn.execute_trace=eager_model_only
        try:
            eager=decision('eager_model_traced_sampler')
        finally:
            ttnn.execute_trace=original_execute
        assert replaced==56 and trace_ids()==ids and gen.counters['captures']==captures
        assert result['traced']['forced_tokens']==result['eager_model_traced_sampler']['forced_tokens']==hf[:57]
        result['comparison']=dict(logits=metrics(traced,eager),model_dispatches_replaced=replaced,
            actual_outputs_exact=result['traced']['tokens']==result['eager_model_traced_sampler']['tokens'],
            traced_prefix_first56_matches_hf=result['traced']['tokens'][:56]==hf[:56],
            eager_prefix_first56_matches_hf=result['eager_model_traced_sampler']['tokens'][:56]==hf[:56],
            trace_ids_unchanged=True,capture_count_unchanged=True)
        saved=Path(a.logits_output);saved.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(traced=traced,eager=eager,prompt=torch.tensor(prompt),hf_history=torch.tensor(hf[:56])),saved)
        result['logits_artifact']=dict(path=str(saved),sha256=hashlib.sha256(saved.read_bytes()).hexdigest());persist()
        if a.separate_prefill:
            print('EAGER_CONTROL_RETIRE_TRACES_BEFORE_VARIANT',flush=True)
            ttnn.synchronize_device(mesh)
            for trace in {*gen.decode_traces.values(),gen.sample_trace,gen.buffered_sample_trace}:
                ttnn.release_trace(mesh,trace)
            gen.decode_traces.clear();gen.model_trace=gen.sample_trace=gen.buffered_sample_trace=None;gen.prepared_buckets.clear()
            result['separate_prefill_config'],restore_variant=install_diagnostic_variant('separate_prefill_rs')
            # All variant ops compile after retiring every previous trace and
            # before capturing the replacement model/sampling traces.
            gen.prepare_decode(prefill_lengths=(4096,))
            print('EAGER_CONTROL_SEPARATE_PREFILL57',flush=True)
            separate=decision('separate_prefill_traced57')
            result['separate_vs_production_y56']=metrics(traced,separate)
            print('EAGER_CONTROL_SEPARATE_PREFILL512',flush=True)
            output=gen.generate(prompt,512,stop_on_eos=True)
            ttnn.synchronize_device(mesh)
            result['separate_prefill_512']=dict(tokens=output,text=gen.tokenizer.decode(output,skip_special_tokens=False),
                eos_ids=sorted(gen.eos_token_ids),metrics_diagnostic_only=dict(gen.last_generation_metrics))
        result.update(status='complete',quality_verdict='Diagnostic completion does not establish a production quality pass')
        print('EAGER_CONTROL_COMPLETE',flush=True)
    except BaseException as error:
        result.update(status='fail',error_type=type(error).__name__,error=str(error)[:2000]);raise
    finally:
        ttnn.execute_trace=original_execute
        persist()
        try:ttnn.close_mesh_device(mesh)
        finally:restore_variant()


if __name__=='__main__':main()
