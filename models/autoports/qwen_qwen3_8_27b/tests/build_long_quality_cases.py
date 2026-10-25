# SPDX-License-Identifier: Apache-2.0
"""Tokenizer-only builder; run in the configured container, no model/device execution."""
import argparse
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer

ROOT=Path('models/autoports/qwen_qwen3_8_27b')


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='.cache/qwen38-full')
    p.add_argument('--benchmark-reference',default=str(ROOT/'doc/full_model/benchmark_hf_control_512.json'))
    p.add_argument('--output',default=str(ROOT/'doc/full_model/long_quality_cases.json'));a=p.parse_args()
    tok=AutoTokenizer.from_pretrained(a.checkpoint)
    def render(content,thinking=False):
        messages=[{'role':'user','content':content}]
        text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=thinking)
        ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=thinking,return_dict=False)
        assert tok.encode(text,add_special_tokens=False)==ids
        return messages,text,ids
    def record(name,target,content,expected=None,thinking=False):
        messages,text,ids=render(content,thinking);assert len(ids)==target
        return dict(name=name,target_tokens=target,messages=messages,rendered_prompt=text,input_tokens=ids,
                    input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),enable_thinking=thinking,
                    add_generation_prompt=True,expected_facts=expected)
    tasks=['calibrate sensor','audit invoice','inspect bridge','translate notice','review access','count seedlings',
           'repair pump','verify shipment','survey shoreline','archive minutes','test battery','measure rainfall']
    statuses=['pending','approved','complete','held','scheduled']
    question=('Return ONLY one compact JSON object with keys "last_record_id", "R0007_amount", "R0023_status". '
              'Copy the last numbered RECORD ID, the integer amount in RECORD R0007, and the status in RECORD R0023. '
              'Use no explanation, markdown, or additional keys.')
    records=[]
    for i in range(1,180):
        line=f'RECORD R{i:04d}: task={tasks[(i*7)%len(tasks)]}; amount={1200+37*i}; status={statuses[(i*3)%len(statuses)]}; district=D{(i*11)%97:02d}; due_day={1+(i*13)%28:02d}.\n'
        if len(render('Distinct administrative records follow.\n'+''.join(records+[line])+'\n'+question)[2])>3750:break
        records.append(line)
    assert len(records)>=23
    expected=dict(last_record_id=f'R{len(records):04d}',R0007_amount=1200+37*7,R0023_status=statuses[(23*3)%len(statuses)])
    prefix='Distinct administrative records follow.\n'+''.join(records)+'\nSupplemental notes (not RECORD entries):\n'
    filler='\n'.join(f'Note N{i:04d}: route {i*19+3}, station {i*23+5}, parcel {i*29+7} verified by clerk C{i:04d}.' for i in range(1,150))
    filler_ids=tok.encode(filler,add_special_tokens=False)
    found=None
    guess=max(0,4096-len(render(prefix+'\n'+question)[2]))
    # Only supplemental notes vary; all answer-bearing records and the complete
    # user instruction/chat-template suffix remain intact.
    for n in range(max(0,guess-64),min(len(filler_ids),guess+128)):
        content=prefix+tok.decode(filler_ids[:n],skip_special_tokens=False)+'\n'+question
        if len(render(content)[2])==4096:found=content;break
    if found is None:raise RuntimeError('Could not render exact4096 case without touching factual records')
    cases=[record('distinct_records_4096',4096,found,expected)]
    words=tok.encode('The research team records each observation carefully and checks the evidence before drawing conclusions. ',add_special_tokens=False)
    def benchmark_case(length):
        budget=length-48
        for _ in range(16):
            body=tok.decode((words*((budget+len(words)-1)//len(words)))[:budget])
            content='Continue with a detailed, several-paragraph account of this research process. Context: '+body
            actual=len(render(content,True)[2])
            if actual==length:return record(f'benchmark_question_{length}',length,content,thinking=True)
            budget+=length-actual
        raise RuntimeError(f'Could not render exact{length} benchmark-question case')
    original=benchmark_case(4096)
    ref_path=Path(a.benchmark_reference);reference=json.loads(ref_path.read_text())
    assert original['input_tokens']==reference['input_tokens'],'Original reconstruction must match every saved HF token'
    original['existing_hf_reference']=dict(path=str(ref_path),sha256=hashlib.sha256(ref_path.read_bytes()).hexdigest(),
                                          output_tokens=len(reference['output_tokens']))
    nearby=benchmark_case(4095)
    nearby['scope_note']='Original benchmark question, repeated-body constructor and thinking-enabled HF template; adjacent logical length4095 pads internally to4096.'
    cases=[original,nearby,*cases]
    template=tok.get_chat_template()
    result=dict(scope='Paired exact-token long qualitative cases; tokenizer-only artifact',checkpoint=a.checkpoint,
                chat_template=template,chat_template_sha256=hashlib.sha256(template.encode()).hexdigest(),
                tokenizer_class=type(tok).__name__,builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                thinking_policy='Explicit per case: benchmark original/nearby True; independent factual records False',cases=cases)
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({c['name']:dict(tokens=len(c['input_tokens']),sha256=c['input_sha256'],expected_facts=c['expected_facts']) for c in cases}),flush=True)


if __name__=='__main__':main()
