# SPDX-License-Identifier: Apache-2.0
"""Unprofiled all-layer autoregressive workload and allocator evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator
from models.autoports.qwen_qwen3_8_27b.tests.full_model_evidence import inventory,request_inventory,runtime_build,tensor_bytes

p=argparse.ArgumentParser()
p.add_argument('--buffered-output',action=argparse.BooleanOptionalAction,default=True)
p.add_argument('--compare-buffered',action='store_true')
p.add_argument('--batch',type=int,default=1)
p.add_argument('--context',type=int,default=262144)
p.add_argument('--lengths',type=int,nargs='+',default=[4096])
p.add_argument('--outputs',type=int,default=128)
p.add_argument('--repeats',type=int,default=3)
p.add_argument('--long-repeats',type=int,default=None,help='Optional repeat count for input lengths above4096')
p.add_argument('--output',required=True)
p.add_argument('--verify-native-context',action='store_true')
p.add_argument('--memory-audit',action='store_true')
p.add_argument('--eos-control',action='store_true')
p.add_argument('--qualitative-control-tokens',type=int,default=0)
p.add_argument('--teacher-forcing-control',action='store_true')
p.add_argument('--precision-evidence',action='store_true',help='Capture actual kernel dtype/fidelity arguments during untimed preparation only')
a=p.parse_args()
if a.teacher_forcing_control and a.batch!=1:
    p.error('--teacher-forcing-control requires batch1')
torch.set_num_threads(8)
ttnn.CONFIG.throw_exception_on_fallback=True
root=Path('models/autoports/qwen_qwen3_8_27b')
source_hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),root/'tests/full_model_memory.py',root/'tests/full_model_evidence.py',*(root/'tt').glob('*.py'),Path('ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/rotary_embedding_program_factory.cpp')]}
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
def memory():
    m=ttnn.get_memory_view(mesh,ttnn.BufferType.DRAM)
    return dict(num_banks=m.num_banks,allocated_bytes_per_device=m.total_bytes_allocated_per_bank*m.num_banks,
                free_bytes_per_device=m.total_bytes_free_per_bank*m.num_banks)
try:
    gen=build_generator(root,mesh,batch_size=a.batch,context_length=a.context,prepare=False)
    before=memory()
    weight_inventory=inventory(gen.model)
    request_buffers=request_inventory(gen)
    cache_inventory=[dict(layer=layer.layer_idx,field=field,**tensor_bytes(getattr(state,field)))
        for layer,state in zip(gen.model.layers,gen.kv_cache)
        for field in ('key','value','conv','recurrent') if hasattr(state,field)]
    precision_calls={}
    if a.precision_evidence:
        from .run_datatype_candidate import capture_precision_calls
        with capture_precision_calls(precision_calls):
            gen.prepare_decode()
    else:
        gen.prepare_decode()
    after=memory()
    owner=gen.model.layers[0].ar_workspace
    precision_runtime=dict(
        compute=[dict(layer=layer.layer_idx,norm=str(layer.ckc.math_fidelity),
                      sdpa=str(layer.sdpa_ckc.math_fidelity),
                      prefill_sdpa=str(layer.ckc.math_fidelity),
                      fused_prefill=str(layer.matmul_ckc.math_fidelity)) for layer in gen.model.layers],
        decode_ccl_dtype=str(owner.dtype),
        prefill_ccl_dtypes=sorted({str(t.dtype) for pair in owner.prefill_rs_buffers.values() for t in pair}),
        native_ccl_dtypes=sorted({str(t.dtype) for pair in owner.native_rs_buffers.get(a.batch,{}).values() for t in pair}),
        sampling_dtypes={name:str(getattr(gen,name).dtype) for name in ('p','temp','tokens','output_history')})
    words=gen.tokenizer.encode('The research team records each observation carefully and checks the evidence before drawing conclusions. ',add_special_tokens=False)
    def make_prompt(length):
        body_length=max(1,length-48)
        for _ in range(8):
            body=gen.tokenizer.decode((words*((body_length+len(words)-1)//len(words)))[:body_length])
            prompt=gen.tokenizer.apply_chat_template([{'role':'user','content':
                'Continue with a detailed, several-paragraph account of this research process. Context: '+body}],
                tokenize=True,add_generation_prompt=True,return_dict=False)
            if len(prompt)==length:return prompt
            body_length+=length-len(prompt)
        raise ValueError('Could not construct exact-length HF chat-template workload')
    records=[]
    packet=dict(scope='all64layers',source='autoregressive',prompt_mode='HF tokenizer chat template; repeated text throughput fixture; EOS disabled for fixed128-token measurement',mesh=[1,4],batch=a.batch,context=a.context,
        throw_exception_on_fallback=bool(ttnn.CONFIG.throw_exception_on_fallback),tracking=os.environ.get('TT_METAL_TRACE_ALLOC_TRACKING'),skip_program_cache=os.environ.get('TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE'),
        diagnostics={key:os.environ.get(key) for key in ('TT_METAL_WATCHER','TT_METAL_WATCHER_DEBUG_DELAY','TT_METAL_DEVICE_PROFILER','TT_METAL_TRACE_ALLOC_TRACEBACKS')},
        weight_inventory=weight_inventory,request_buffers=request_buffers,memory_before_preparation=before,memory_after_preparation=after,
        preparation_memory=getattr(gen,"last_preparation_memory",None),records=records,
        precision_config=gen.model.precision_config,
        cache_inventory=cache_inventory,
        prepared_kernel_precision_calls=precision_calls.get('prepared_kernel_precision_calls'),
        precision_runtime=precision_runtime,
        source_sha256=source_hashes,runtime_build=runtime_build())
    for length in a.lengths:
        prompt=make_prompt(length)
        prompts=[prompt[:] for _ in range(a.batch)]
        # Warm the exact request once; all timing below remains true token-out.
        gen.generate(prompts,a.outputs,stop_on_eos=False,buffered_output=a.buffered_output)
        repeats=a.long_repeats if length>4096 and a.long_repeats is not None else a.repeats
        for repeat in range(repeats):
            if a.compare_buffered:
                baseline_tokens=gen.generate(prompts,a.outputs,stop_on_eos=False,buffered_output=False)
                baseline=dict(gen.last_generation_metrics,repeat=repeat)
                packet.setdefault('streamed_baseline_records',[]).append(baseline)
            tokens=gen.generate(prompts,a.outputs,stop_on_eos=False,buffered_output=a.buffered_output)
            metrics=dict(gen.last_generation_metrics)
            if a.compare_buffered:
                assert tokens==baseline_tokens,'Buffered token sequence differs from streaming'
            assert metrics['source']=='autoregressive' and not metrics['host_sampling']
            for key in ('host_copies_total','token_refreshes','position_refreshes','page_table_refreshes','captures'):
                assert metrics['steady_state_counters'][key]==(max(0,metrics.get('output_windows',0)-1) if key=='host_copies_total' else 0),(key,metrics)
            assert all(len(t)==a.outputs for t in tokens)
            if not all(t==tokens[0] for t in tokens):
                failure=dict(scope='all64layer fixed-slot determinism failure',batch=a.batch,input_length=length,
                    outputs=tokens,metrics=metrics,source_sha256=source_hashes,
                    first_difference_by_slot={str(i):next((j for j,(x,y) in enumerate(zip(tokens[0],row)) if x!=y),None) for i,row in enumerate(tokens)})
                Path(a.output).with_suffix('.failure.json').write_text(json.dumps(failure,indent=2)+'\n')
                raise AssertionError('Identical prompts differ by fixed slot; see failure.json')
            metrics.update(repeat=repeat,prompt_tokens_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),completion_slot0=gen.tokenizer.decode(tokens[0],skip_special_tokens=False),
                tokens_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),memory=memory())
            records.append(metrics)
            Path(a.output).write_text(json.dumps(packet,indent=2)+'\n')
            print(json.dumps(metrics),flush=True)
    if a.batch>1:
        print('FULL_STACK_MIXED_TRACE_BEGIN',flush=True)
        trace_ids=[str(t) for t in gen.decode_traces.values()]+[str(gen.sample_trace),str(gen.buffered_sample_trace)]
        captures=gen.counters['captures']
        mixed_prompts=[[100]*33,[101]*65,[102]]
        mixed=gen.generate(mixed_prompts,4,stop_on_eos=False)
        mixed_metrics=dict(gen.last_generation_metrics)
        repeated=gen.generate(mixed_prompts,4,stop_on_eos=False)
        assert repeated==mixed
        all_active=gen.generate([[100]*33 for _ in range(a.batch)],4,stop_on_eos=False)
        assert len(all_active)==a.batch and all(row==all_active[0] for row in all_active)
        assert trace_ids==[str(t) for t in gen.decode_traces.values()]+[str(gen.sample_trace),str(gen.buffered_sample_trace)]
        assert captures==gen.counters['captures']
        for key in ('host_copies_total','token_refreshes','position_refreshes','page_table_refreshes','captures'):
            assert mixed_metrics['steady_state_counters'][key]==0
        packet['mixed_trace_validation']=dict(scope='all64layers; masked/repeated masked/all-active replay after full-batch measurements',
            trace_ids=trace_ids,captures=captures,mixed_prompt_lengths=[33,65,1],mixed_outputs=mixed,
            repeated_exact=True,all_active_slots_equal=True,masked_metrics=mixed_metrics)
        Path(a.output).write_text(json.dumps(packet,indent=2)+'\n')
        print('FULL_STACK_MIXED_TRACE_PASS',json.dumps(packet['mixed_trace_validation']),flush=True)
    if a.teacher_forcing_control:
        from readiness_check.schema import load_reference
        reference_path=root/'readiness_aime24_chat.refpt'
        reference=load_reference(reference_path)
        entry=reference.entries[0]
        tf_prompt=entry.prompt_tokens[0].tolist()
        expected=entry.generated_tokens[0].tolist()
        assert len(tf_prompt)==203 and len(expected)==100,('Unexpected AIME control dimensions',len(tf_prompt),len(expected))
        trace_ids=[str(gen.model_trace),str(gen.sample_trace),str(gen.buffered_sample_trace)]
        captures=gen.counters['captures']
        print('TEACHER_FORCING_CONTROL_WARMUP',flush=True)
        gen.generate(tf_prompt,100,next_input=lambda step,token:expected[step],stop_on_eos=False)
        print('TEACHER_FORCING_CONTROL_MEASURE',flush=True)
        predicted=gen.generate(tf_prompt,100,next_input=lambda step,token:expected[step],stop_on_eos=False)
        tf_metrics=dict(gen.last_generation_metrics)
        assert tf_metrics['source']=='teacher_forcing_token_out' and not tf_metrics['host_sampling']
        assert len(predicted)==len(expected)==100
        assert trace_ids==[str(gen.model_trace),str(gen.sample_trace),str(gen.buffered_sample_trace)] and captures==gen.counters['captures']
        correct=sum(actual==target for actual,target in zip(predicted,expected))
        packet['teacher_forcing_control']=dict(tf_metrics,
            scope='Clean high-level all64 B1 AIME teacher-forced token-out timing; canonical split sampling plus explicit reference-token callback; not autoregressive headline performance or run_teacher_forcing accuracy replacement',
            input_length=len(tf_prompt),generated_tokens=len(predicted),warmup_requests=1,
            prompt_tokens_sha256=hashlib.sha256(json.dumps(tf_prompt).encode()).hexdigest(),
            reference=str(reference_path),reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            predicted_tokens_sha256=hashlib.sha256(json.dumps(predicted).encode()).hexdigest(),
            predicted_top1_agreement=dict(correct_tokens=correct,compared_tokens=100,value=correct/100,
                unit='fraction',scope='All100 generated positions of reference entry0 under exact HF teacher forcing'),
            trace_ids=trace_ids,trace_ids_unchanged=True)
        Path(a.output).write_text(json.dumps(packet,indent=2)+'\n')
        print('TEACHER_FORCING_CONTROL',json.dumps(packet['teacher_forcing_control']),flush=True)
    if a.verify_native_context:
        assert a.batch==1 and a.context==gen.model.config.max_position_embeddings
        trace_ids=[str(gen.model_trace),str(gen.sample_trace),str(gen.buffered_sample_trace)]
        programs=mesh.num_program_cache_entries()
        length=a.context-3
        prompt=make_prompt(length)
        tokens=gen.generate(prompt,3,stop_on_eos=False)
        assert len(tokens)==3 and all(0<=t<gen.model.config.vocab_size for t in tokens)
        # Exercise the final physical cache position explicitly, without using
        # its next-token prediction as a request beyond the context contract.
        logits=gen.decode_forward(torch.tensor([[tokens[-1]]]),torch.tensor([a.context-1]),
            page_table=gen.page_table,kv_cache=gen.kv_cache,return_logits=True,return_device=True)
        assert torch.isfinite(gen.host_logits(logits)).all()
        del logits
        final_positions=ttnn.to_torch(ttnn.get_device_tensors(gen.positions)[0]).long().tolist()
        assert final_positions==[a.context]
        gen.reset()
        prompt=make_prompt(a.context)
        logits=gen.prefill_forward(torch.tensor([prompt]),page_table=gen.page_table,kv_cache=gen.kv_cache,
            prompt_lens=[a.context],return_device=True)
        assert torch.isfinite(gen.host_logits(logits)).all()
        del logits
        assert [str(gen.model_trace),str(gen.sample_trace),str(gen.buffered_sample_trace)]==trace_ids
        assert mesh.num_program_cache_entries()==programs
        packet['context_validation']=dict(nonaligned_prompt=length,generated_tokens=tokens,
            exact_max_prefill=a.context,final_decode_input_position=a.context-1,
            final_next_positions=final_positions,finite=True,programs=programs,trace_ids=trace_ids,memory=memory(),
            scope='All-layer capacity and finite-output check on repeated text; not HF accuracy or headline performance')
        Path(a.output).write_text(json.dumps(packet,indent=2)+'\n')
        print('NATIVE_CONTEXT_PASS',json.dumps(packet['context_validation']),flush=True)
    if a.qualitative_control_tokens:
        assert a.batch==1
        quality_tokens=gen.generate(make_prompt(4096),a.qualitative_control_tokens,stop_on_eos=True)
        packet['throughput_fixture_qualitative_control']=dict(input_tokens=4096,max_new_tokens=a.qualitative_control_tokens,
            tokens=quality_tokens,completion=gen.tokenizer.decode(quality_tokens,skip_special_tokens=False),
            scope='Longer qualitative continuation with normal EOS stopping; not a headline performance measurement',stop_on_eos=True)
        Path(a.output).write_text(json.dumps(packet,indent=2)+'\n')
        print('THROUGHPUT_QUALITY',packet['throughput_fixture_qualitative_control']['completion'],flush=True)
    if a.eos_control:
        assert a.batch==1
        cases=json.loads((root/'doc/full_model/long_quality_cases.json').read_text())['cases']
        case=next(c for c in cases if c['name']=='distinct_records_4096')
        prompt=case['input_tokens']
        assert len(prompt)==4096
        gen.generate(prompt,1024,stop_on_eos=True)
        eos_rows=[];expected=None;original_window=gen.eos_output_window
        try:
            for label,buffered,window in [('streamed',False,16),('buffered128',True,128),('buffered16',True,16)]:
                gen.eos_output_window=window
                for repeat in range(3):
                    tokens=gen.generate(prompt,1024,stop_on_eos=True,buffered_output=buffered)
                    if expected is None:expected=tokens
                    assert tokens==expected,'EOS control token sequence differs'
                    metrics=dict(gen.last_generation_metrics)
                    eos_rows.append(dict(mode=label,repeat=repeat,returned_tokens=len(tokens),
                        total_request_ms=metrics['ttft_ms']+1000*metrics['decode_seconds'],metrics=metrics))
        finally:
            gen.eos_output_window=original_window
        packet['eos_window_control']=dict(case='distinct_records_4096',rows=eos_rows,exact=True,
            completion=gen.tokenizer.decode(expected,skip_special_tokens=False),
            scope='All64layer paired warmed EOS latency control; 4096 input,1024 output cap; not headline4096/128 workload')
        Path(a.output).write_text(json.dumps(packet,indent=2)+'\n')
        print('EOS_WINDOW_CONTROL_PASS',json.dumps(packet['eos_window_control']),flush=True)
    if a.memory_audit:
        from .full_model_memory import allocation_peak,release_trace_bytes
        print('ALLOCATION_AUDIT_BEGIN',flush=True)
        packet['allocation_audit']=allocation_peak(gen,[make_prompt(4096) for _ in range(a.batch)],
            Path(a.output).with_suffix('.allocations.json.gz'))
        packet['trace_allocation']=release_trace_bytes(gen)
        Path(a.output).write_text(json.dumps(packet,indent=2)+'\n')
        print('ALLOCATION_AUDIT_PASS',json.dumps(packet['allocation_audit']),json.dumps(packet['trace_allocation']),flush=True)
finally:
    ttnn.close_mesh_device(mesh)
