# SPDX-License-Identifier: Apache-2.0
"""Paired HF/TT long-case closure runner. No timing here is performance evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

ROOT=Path('models/autoports/qwen_qwen3_8_27b')


def main():
    p=argparse.ArgumentParser();mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--hf',action='store_true');mode.add_argument('--tt',action='store_true')
    p.add_argument('--checkpoint',default='.cache/qwen38-full')
    p.add_argument('--cases',default=str(ROOT/'doc/full_model/long_quality_cases.json'))
    p.add_argument('--case',nargs='+',dest='case_names')
    p.add_argument('--prefix-reference',default=str(ROOT/'doc/full_model/native_prefill_geometry.json'))
    p.add_argument('--prefix-variant',default='out')
    p.add_argument('--output');p.add_argument('--max-new-tokens',type=int,default=512)
    p.add_argument('--geometry',choices=('production','both','out','down','out16'),default='production')
    a=p.parse_args()
    if a.hf and a.geometry!='production':p.error('Geometry override is TT diagnostic only')
    assert 1<=a.max_new_tokens<=1024
    data=json.loads(Path(a.cases).read_text())
    all_names={case['name'] for case in data['cases']}
    if a.case_names:
        assert set(a.case_names)<=all_names,('Unknown cases',set(a.case_names)-all_names)
        data['cases']=[case for case in data['cases'] if case['name'] in a.case_names]
    assert data['cases']
    label='hf' if a.hf else 'tt_'+a.geometry
    target=Path(a.output) if a.output else ROOT/'doc/full_model'/f'long_quality_cases_{label}.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    result=dict(status='running',scope='Long qualitative/factual closure, not a performance measurement',backend=label,
        case_artifact=a.cases,case_artifact_sha256=hashlib.sha256(Path(a.cases).read_bytes()).hexdigest(),
        thinking_policy='Per-case explicit setting, verified identically for HF and TT',stop_on_eos=True,greedy=True,max_new_tokens=a.max_new_tokens,cases=[],
        checkpoint=a.checkpoint,torch_version=str(torch.__version__),
        checkpoint_metadata_sha256={name:hashlib.sha256((Path(a.checkpoint)/name).read_bytes()).hexdigest()
            for name in ('config.json','generation_config.json','model.safetensors.index.json')},
        source_sha256={str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    def persist():target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    def verify(tok):
        assert hashlib.sha256(tok.get_chat_template().encode()).hexdigest()==data['chat_template_sha256']
        for case in data['cases']:
            ids=tok.apply_chat_template(case['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=case['enable_thinking'],return_dict=False)
            assert ids==case['input_tokens'] and len(ids)==case['target_tokens']
            assert hashlib.sha256(json.dumps(ids).encode()).hexdigest()==case['input_sha256']
    def save(case,tokens,tok):
        text=tok.decode(tokens,skip_special_tokens=False);plain=tok.decode(tokens,skip_special_tokens=True).strip()
        row=dict(name=case['name'],input_tokens=len(case['input_tokens']),input_sha256=case['input_sha256'],
                 enable_thinking=case['enable_thinking'],tokens=tokens,text=text)
        if a.tt and a.geometry=='out' and case['name']=='benchmark_question_4096':
            path=Path(a.prefix_reference);reference=json.loads(path.read_text())
            expected=reference['variants'][a.prefix_variant]['completion']['tokens']
            assert reference['input_sha256']==case['input_sha256']
            count=min(512,len(expected),a.max_new_tokens)
            row['recorded_prefix_check']=dict(artifact=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                variant=a.prefix_variant,compared_tokens=count,exact=tokens[:count]==expected[:count],
                scope='Longer cap is diagnostic extension only; original512 prefix must remain unchanged')
        if case['expected_facts'] is not None:
            try:parsed=json.loads(plain)
            except (ValueError,TypeError):parsed=None
            row.update(expected_facts=case['expected_facts'],parsed_json=parsed,strict_json=isinstance(parsed,dict),
                       facts_exact=parsed==case['expected_facts'])
        result['cases'].append(row);persist()
        if 'recorded_prefix_check' in row:
            assert row['recorded_prefix_check']['exact'],'Candidate512 prefix changed; extension is not the same trajectory'
    mesh=None;gen=None;saved_geometries=[]
    try:
        if a.hf:
            from .hf_text import load_model
            import transformers
            result['transformers_version']=transformers.__version__
            result['source_sha256'][str(ROOT/'tests/hf_text.py')]=hashlib.sha256((ROOT/'tests/hf_text.py').read_bytes()).hexdigest()
            torch.set_num_threads(16);model,tok=load_model(a.checkpoint);verify(tok)
            config=json.loads((Path(a.checkpoint)/'generation_config.json').read_text())
            eos=config['eos_token_id'];eos=set(eos if isinstance(eos,list) else [eos])
            result.update(eos_ids=sorted(eos),dtype=str(next(model.parameters()).dtype))
            for case in data['cases']:
                print('LONG_CASE_HF',case['name'],flush=True)
                cache=None;inp=torch.tensor([case['input_tokens']]);tokens=[]
                with torch.inference_mode():
                    for step in range(a.max_new_tokens):
                        output=model(input_ids=inp,past_key_values=cache,use_cache=True,logits_to_keep=1)
                        cache=output.past_key_values;token=int(output.logits[0,-1].float().argmax());tokens.append(token)
                        if token in eos:break
                        inp=torch.tensor([[token]])
                        if step%32==0:print('LONG_CASE_HF_STEP',step,flush=True)
                save(case,tokens,tok)
                del cache,output
        else:
            import ttnn
            from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator
            from models.autoports.qwen_qwen3_8_27b.tests.full_model_evidence import runtime_build
            torch.set_num_threads(8);ttnn.CONFIG.throw_exception_on_fallback=True
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
            mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
            gen=build_generator(ROOT,mesh,checkpoint=a.checkpoint,batch_size=1,context_length=262144,prepare=False)
            assert len(gen.model.layers)==64;verify(gen.tokenizer)
            if a.geometry!='production':
                changes={'both':{'out_k':4,'down_k':4},'out':{'out_k':4},'down':{'down_k':4},'out16':{'out_k':16}}[a.geometry]
                for layer in gen.model.layers:
                    view=getattr(layer,'_prefill_impl',layer);old=view.policy['prefill_rs_geometry']
                    saved_geometries.append((view,old));view.policy['prefill_rs_geometry']={**old,**changes}
            result.update(runtime=runtime_build(),context=262144,batch=1,layers=64,
                          geometry_override_diagnostic_only=a.geometry!='production',eos_ids=sorted(gen.eos_token_ids),
                          preparation_scope='Exact physical4096 bucket for logical4095/4096; not whole-domain preparation evidence')
            result['source_sha256'].update({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'tt').glob('*.py')})
            gen.prepare_decode(prefill_lengths=(4095,4096))
            traces=(str(gen.model_trace),str(gen.sample_trace));captures=gen.counters['captures']
            for case in data['cases']:
                print('LONG_CASE_TT',case['name'],flush=True)
                tokens=gen.generate(case['input_tokens'],a.max_new_tokens,stop_on_eos=True)
                ttnn.synchronize_device(mesh);save(case,tokens,gen.tokenizer)
                assert traces==(str(gen.model_trace),str(gen.sample_trace)) and captures==gen.counters['captures']
            result['trace_ids']=traces;result['trace_ids_unchanged']=True
        result.update(status='complete',qualitative_verdict='Read completions and paired HF evidence; completion alone is not a quality pass')
    except BaseException as error:
        result.update(status='fail',error_type=type(error).__name__,error=str(error)[:2000]);raise
    finally:
        persist()
        if mesh is not None:
            import ttnn
            ttnn.close_mesh_device(mesh)
        for view,geometry in saved_geometries:view.policy['prefill_rs_geometry']=geometry


if __name__=='__main__':main()
