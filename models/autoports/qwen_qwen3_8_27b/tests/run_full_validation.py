# SPDX-License-Identifier: Apache-2.0
"""Run unchanged readiness comparisons on one shared live all-layer generator.

The runner factories borrow the same generator to avoid repeated64-layer loading.
The autoregressive HF adapter reuses the freshly generated, exact-token AIME control;
its prompt and generation length must match. No comparator or TT output is replaced.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
from .full_model_evidence import runtime_build
import torch
import ttnn
from readiness_check import run_prefill_check as prefill,run_teacher_forcing as teacher
from models.autoports.qwen_qwen3_8_27b.tests import readiness_autoregressive as autoreg
from readiness_check.schema import load_reference
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator

@contextlib.contextmanager
def shared_factory(module,generator):
    original=module._import_build_generator
    module._import_build_generator=lambda directory: (lambda **kwargs: generator)
    try: yield
    finally: module._import_build_generator=original

parser=argparse.ArgumentParser()
parser.add_argument('--output-dir',default='models/autoports/qwen_qwen3_8_27b/doc/full_model')
parser.add_argument('--buffered-output',action='store_true')
parser.add_argument('--long-cases',action='store_true',help='Run three long closure cases on the same default generator')
parser.add_argument('--precision-long-cases',help='Additional prompt-correct cases for precision selection, with actual kernel-argument evidence')
args=parser.parse_args()

torch.set_num_threads(8)
root=Path('models/autoports/qwen_qwen3_8_27b')
out=Path(args.output_dir)
out.mkdir(parents=True,exist_ok=True)
reference_path=root/'readiness_aime24_chat.refpt'
reference=load_reference(reference_path)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
try:
    if args.precision_long_cases:
        from .run_datatype_candidate import capture_precision_calls
        precision_runtime={}
        with capture_precision_calls(precision_runtime):
            gen=build_generator(root,mesh,context_length=262144)
        precision_runtime['config']=gen.model.precision_config
        (out/'precision_runtime.json').write_text(json.dumps(precision_runtime,indent=2)+'\n')
    else:
        gen=build_generator(root,mesh,context_length=262144)
    print('FULL_READY',flush=True)
    provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*(root/'tt').glob('*.py'),Path('ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/rotary_embedding_program_factory.cpp')]}
    with shared_factory(prefill,gen):
        pre=prefill.run_prefill_check(model_dir=root,reference_path=reference_path,mesh_device=mesh)
    (out/'prefill_accuracy.json').write_text(json.dumps(pre,indent=2)+'\n')
    print('PREFILL_ACCURACY',pre,flush=True)
    with shared_factory(teacher,gen):
        dec=teacher.run_teacher_forcing(model_dir=root,reference_path=reference_path,mesh_device=mesh)
    (out/'teacher_forcing_accuracy.json').write_text(json.dumps(dec,indent=2)+'\n')
    print('DECODE_ACCURACY',dec,flush=True)
    (out/'teacher_forcing_performance.json').write_text(json.dumps(gen.last_generation_metrics,indent=2)+'\n')
    for rows in (pre,dec):
        for row in rows:
            assert row['top1']>=.90 and row['top5']>=.98 and row['top100']==1.0,row
    entry=reference.entries[0]
    prompt_file=out/'aime24_rendered_prompt.txt';prompt_file.write_text(entry.prompt_text)
    def hf_control(*,hf_model_id,prompt_token_ids,max_new_tokens,device):
        assert prompt_token_ids==entry.prompt_tokens[0].tolist()
        assert max_new_tokens==entry.generated_tokens.shape[1]
        return entry.generated_tokens[0].tolist()
    old_hf=autoreg._hf_generate_greedy
    autoreg._hf_generate_greedy=hf_control
    try:
        with shared_factory(autoreg,gen):
            autoreg.run_autoregressive(model_dir=root,hf_model_id='.cache/qwen38-full',
                prompt_file=prompt_file,mesh_device=mesh,output_dir=out/'autoregressive',max_new_tokens=100)
    finally: autoreg._hf_generate_greedy=old_hf
    (out/'aime_autoregressive_performance.json').write_text(json.dumps(gen.last_generation_metrics,indent=2)+'\n')
    suite=json.loads((root/'doc/qualitative_prompts.json').read_text())
    for item in suite['prompts']:
        rendered=gen.tokenizer.apply_chat_template(item['messages'],tokenize=False,add_generation_prompt=True)
        prompt=gen.tokenizer.encode(rendered,add_special_tokens=False)
        tokens=gen.generate(prompt,128)
        eos_positions=[i for i,t in enumerate(tokens) if t in gen.eos_token_ids]
        assert not eos_positions or eos_positions==[len(tokens)-1],(item['id'],eos_positions)
        record=dict(id=item['id'],messages=item['messages'],rendered_prompt=rendered,prompt_tokens=prompt,
            generated_tokens=tokens,completion=gen.tokenizer.decode(tokens,skip_special_tokens=False),
            metrics=gen.last_generation_metrics,eos_ids=sorted(gen.eos_token_ids),chat_template=True,do_sample=False,max_new_tokens=128)
        (out/'qualitative').mkdir(exist_ok=True)
        (out/'qualitative'/(item['id']+'_tt.json')).write_text(json.dumps(record,indent=2)+'\n')
        print('QUALITATIVE',item['id'],record['completion'],flush=True)
        if args.precision_long_cases and item['id']=='structured':
            final=record['completion'].split('</think>',1)[1] if '</think>' in record['completion'] else ''
            for eos in gen.eos_token_ids:
                final=final.replace(gen.tokenizer.decode([eos],skip_special_tokens=False),'')
            try: parsed=json.loads(final.strip())
            except (ValueError,TypeError): parsed=None
            if parsed!={'alice_apples':4,'bob_apples':3}:
                (out/'quality_gate_failure.json').write_text(json.dumps(dict(kind='quality_rejection',
                    case='structured',reason='No exact final JSON within shared128-token budget',
                    expected={'alice_apples':4,'bob_apples':3},parsed=parsed,
                    hf_control='doc/optimized_full_model/qualitative/structured_hf.json',
                    previous_control='doc/datatype_sweep/baseline/qualitative/structured_tt.json'),indent=2)+'\n')
                raise AssertionError('Precision quality gate: structured final JSON absent/incorrect at128')
    if args.precision_long_cases:
        cases=json.loads(Path(args.precision_long_cases).read_text())
        assert hashlib.sha256(gen.tokenizer.get_chat_template().encode()).hexdigest()==cases['chat_template_sha256']
        rows=[]
        trace_ids=[str(gen.model_trace),str(gen.sample_trace),str(gen.buffered_sample_trace)]
        captures=gen.counters['captures']
        for case in cases['cases']:
            prompt=gen.tokenizer.apply_chat_template(case['messages'],tokenize=True,add_generation_prompt=True,
                enable_thinking=case['enable_thinking'],return_dict=False)
            assert prompt==case['input_tokens'] and len(prompt)==case['target_tokens']
            tokens=gen.generate(prompt,1024,stop_on_eos=True)
            text=gen.tokenizer.decode(tokens,skip_special_tokens=False)
            plain=gen.tokenizer.decode(tokens,skip_special_tokens=True).strip()
            row=dict(name=case['name'],input_tokens=len(prompt),input_sha256=case['input_sha256'],
                     generated_tokens=tokens,completion=text,output_tokens=len(tokens),
                     eos_terminated=bool(tokens) and tokens[-1] in gen.eos_token_ids)
            if case['expected_facts'] is not None:
                try:parsed=json.loads(plain)
                except (ValueError,TypeError):parsed=None
                row.update(parsed_json=parsed,expected_facts=case['expected_facts'],facts_exact=parsed==case['expected_facts'])
            rows.append(row)
            (out/'precision_long_quality.json').write_text(json.dumps(dict(cases=rows,
                source=args.precision_long_cases,trace_ids=trace_ids,captures=captures),indent=2)+'\n')
            if case['expected_facts'] is None and case['enable_thinking'] and (
                '</think>' not in text or not text.split('</think>',1)[1].strip()):
                (out/'quality_gate_failure.json').write_text(json.dumps(dict(kind='quality_rejection',
                    case=case['name'],reason='No final-answer text within shared1024-token budget',
                    hf_control='doc/full_model/benchmark_hf_control_512.json' if case['target_tokens']==4096 else 'doc/optimized_full_model/long_quality_cases_hf.json',
                    previous_control='doc/optimized_full_model/default_long_quality_cases.json'),indent=2)+'\n')
                raise AssertionError('Precision quality gate: long reply never reaches final answer')
            assert trace_ids==[str(gen.model_trace),str(gen.sample_trace),str(gen.buffered_sample_trace)]
            assert captures==gen.counters['captures']
            if case['expected_facts'] is not None:
                assert row['facts_exact'] and row['eos_terminated'],row
            print('PRECISION_LONG_COMPLETION',case['name'],text,flush=True)
    if args.long_cases:
        case_path=root/'doc/full_model/long_quality_cases.json'
        cases=json.loads(case_path.read_text())
        assert {case['name'] for case in cases['cases']}=={
            'benchmark_question_4096','benchmark_question_4095','distinct_records_4096'}
        prefix_path=root/'doc/full_model/native_prefill_geometry.json'
        prefix_reference=json.loads(prefix_path.read_text())
        prefix_tokens=prefix_reference['variants']['out']['completion']['tokens']
        assert len(prefix_tokens)>=512
        assert hashlib.sha256(gen.tokenizer.get_chat_template().encode()).hexdigest()==cases['chat_template_sha256']
        trace_ids=dict(model=str(gen.model_trace),sampling=str(gen.sample_trace),buffered_sampling=str(gen.buffered_sample_trace))
        capture_count=gen.counters['captures']
        long_sources={**provenance,str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'models/common/modules/sampling/sampling_1d.py':hashlib.sha256(Path('models/common/modules/sampling/sampling_1d.py').read_bytes()).hexdigest()}
        geometries=sorted({json.dumps(getattr(layer,'_prefill_impl',layer).policy['prefill_rs_geometry'],sort_keys=True)
                           for layer in gen.model.layers})
        long_record=dict(status='running',scope='All64 default production source, shared live B1 generator, long qualitative closure; not a performance benchmark',
            batch=1,context=gen.context_length,layers=len(gen.model.layers),max_new_tokens=1024,
            stop_on_eos=True,host_sampling=False,geometry_override=False,
            default_prefill_geometries=[json.loads(value) for value in geometries],
            source_sha256=long_sources,runtime_build=runtime_build(),
            case_artifact=str(case_path),case_artifact_sha256=hashlib.sha256(case_path.read_bytes()).hexdigest(),
            prefix_artifact=str(prefix_path),prefix_artifact_sha256=hashlib.sha256(prefix_path.read_bytes()).hexdigest(),
            trace_ids=trace_ids,cases=[])
        long_path=out/'default_long_quality_cases.json'
        def save_long():
            long_path.write_text(json.dumps(long_record,indent=2,allow_nan=False)+'\n')
        try:
            assert len(gen.model.layers)==64 and gen.batch_size==1 and gen.context_length==262144
            for case in cases['cases']:
                prompt=gen.tokenizer.apply_chat_template(case['messages'],tokenize=True,
                    add_generation_prompt=True,enable_thinking=case['enable_thinking'],return_dict=False)
                prompt_hash=hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
                assert prompt==case['input_tokens'] and len(prompt)==case['target_tokens']
                assert prompt_hash==case['input_sha256']
                print('DEFAULT_LONG_CASE',case['name'],flush=True)
                tokens=gen.generate(prompt,1024,stop_on_eos=True)
                ttnn.synchronize_device(mesh)
                text=gen.tokenizer.decode(tokens,skip_special_tokens=False)
                plain=gen.tokenizer.decode(tokens,skip_special_tokens=True).strip()
                eos_positions=[i for i,token in enumerate(tokens) if token in gen.eos_token_ids]
                row=dict(name=case['name'],input_tokens=len(prompt),input_sha256=prompt_hash,
                    enable_thinking=case['enable_thinking'],generated_tokens=tokens,completion=text,
                    output_tokens=len(tokens),eos_ids=sorted(gen.eos_token_ids),eos_positions=eos_positions,
                    metrics_diagnostic_only=dict(gen.last_generation_metrics))
                if case['name']=='benchmark_question_4096':
                    assert prefix_reference['input_sha256']==prompt_hash
                    row['recorded_first512_exact']=tokens[:512]==prefix_tokens[:512]
                    row['extension_scope']='1024 cap is additional closure evidence; recorded candidate512 prefix must remain unchanged, not a longer-cap repair'
                if case['expected_facts'] is not None:
                    try:parsed=json.loads(plain)
                    except (ValueError,TypeError):parsed=None
                    row.update(expected_facts=case['expected_facts'],parsed_json=parsed,
                        strict_json=isinstance(parsed,dict),facts_exact=parsed==case['expected_facts'],
                        eos_terminated=bool(eos_positions) and eos_positions==[len(tokens)-1])
                long_record['cases'].append(row);save_long()
                assert not eos_positions or eos_positions==[len(tokens)-1],(case['name'],eos_positions)
                if case['name']=='benchmark_question_4096':
                    assert row['recorded_first512_exact'],'Default source changed recorded candidate512 trajectory'
                if case['expected_facts'] is not None:
                    assert row['strict_json'] and row['facts_exact'] and row['eos_terminated'],row
                assert trace_ids==dict(model=str(gen.model_trace),sampling=str(gen.sample_trace),buffered_sampling=str(gen.buffered_sample_trace))
                assert capture_count==gen.counters['captures']
                print('DEFAULT_LONG_COMPLETION',case['name'],text,flush=True)
            long_record.update(status='complete',trace_ids_unchanged=True,capture_count_unchanged=True,
                qualitative_verdict='Read original/nearby prose for coherence and repetition; machine gates prove prompt identity, prefix, factual JSON and factual EOS only')
        except BaseException as error:
            long_record.update(status='fail',error_type=type(error).__name__,error=str(error)[:2000])
            raise
        finally:
            save_long()
    (out/'full_validation_provenance.json').write_text(json.dumps(dict(source_sha256=provenance,runtime_build=runtime_build(),
        runner_prompt_fix='readiness_preserve_prompt.patch preserves trailing HF chat-template newline; comparators unchanged',
        reference=str(reference_path),reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        runner_adapter='Factory borrows one real all-layer generator; autoregressive HF control is the fresh exact-token100-step reference',
        tracking=True,skip_program_cache=False,counters=gen.counters),indent=2)+'\n')
finally:
    ttnn.close_mesh_device(mesh)
