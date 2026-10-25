# SPDX-License-Identifier: Apache-2.0
"""Verify prompt/control provenance; semantic verdict requires reading outputs."""
import hashlib
import json
from pathlib import Path
root=Path('models/autoports/qwen_qwen3_8_27b')
out=root/'doc/datatype_sweep';selected=out/'selected_validation'
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def token_hash(tokens):return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
short=[]
for name in ('structured','explain','code'):
    tt_path=selected/'qualitative'/f'{name}_tt.json'
    hf_path=root/'doc/optimized_full_model/qualitative'/f'{name}_hf.json'
    prior_path=out/'baseline/qualitative'/f'{name}_tt.json'
    tt,hf,prior=map(read,(tt_path,hf_path,prior_path))
    assert tt['prompt_tokens']==hf['prompt_tokens']==prior['prompt_tokens']
    assert tt['rendered_prompt']==hf['rendered_prompt']
    assert tt['chat_template'] and hf['chat_template'] and not tt['do_sample'] and not hf['do_sample']
    assert tt['max_new_tokens']==hf['max_new_tokens']==128
    short.append(dict(name=name,prompt_tokens_exact=True,prompt_sha256=token_hash(tt['prompt_tokens']),
        tt=str(tt_path),hf=str(hf_path),hf_sha256=sha(hf_path),previous_tt=str(prior_path),
        previous_tokens_exact=tt['generated_tokens']==prior['generated_tokens'],
        tt_output_tokens=len(tt['generated_tokens']),hf_output_tokens=len(hf['generated_tokens']),
        tt_eos=tt['generated_tokens'][-1] in tt['eos_ids'],hf_eos=hf['generated_tokens'][-1] in tt['eos_ids']))
cases={c['name']:c for c in read(out/'combined_long_cases.json')['cases']}
controls={}
p=root/'doc/full_model/benchmark_hf_control_512.json';d=read(p)
controls['benchmark_question_4096']=(p,dict(input_sha256=token_hash(d['input_tokens']),tokens=d['output_tokens'],max_new_tokens=d['max_new_tokens'],stop_on_eos=d['stop_on_eos']))
for p in (root/'doc/optimized_full_model/long_quality_cases_hf.json',out/'agent_long_hf.json'):
    d=read(p)
    for c in d['cases']:
        controls[c['name']]=(p,dict(c,max_new_tokens=d['max_new_tokens'],stop_on_eos=d['stop_on_eos']))
long=[]
for tt in read(selected/'precision_long_quality.json')['cases']:
    name=tt['name'];case=cases[name];p,hf=controls[name]
    assert tt['input_sha256']==case['input_sha256']==hf['input_sha256']
    assert tt['input_tokens']==case['target_tokens']
    if case['expected_facts'] is not None:
        assert tt['facts_exact'] and hf['facts_exact'] and tt['eos_terminated']
    long.append(dict(name=name,input_tokens=tt['input_tokens'],input_sha256=tt['input_sha256'],
        enable_thinking=case['enable_thinking'],tt=str(selected/'precision_long_quality.json'),
        hf=str(p),hf_sha256=sha(p),tt_output_tokens=tt['output_tokens'],hf_output_tokens=len(hf['tokens']),
        tt_cap=1024,hf_cap=hf['max_new_tokens'],hf_stop_on_eos=hf['stop_on_eos'],
        tt_eos=tt['eos_terminated'],facts_exact=tt.get('facts_exact'),
        control_scope='Same exact prompt and greedy decoding; cap/stop differences explicit. Read output for semantics; no equality gate for free-running text.'))
report=dict(status='provenance_and_factual_checks_passed_manual_semantic_review_required',short=short,long=long,
    scope='Prompt-correct qualitative evidence, not dataset accuracy or throughput')
(out/'qualitative_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print('Verified',len(short),'shared prompts and',len(long),'long prompt controls')
