# SPDX-License-Identifier: Apache-2.0
"""Full64/B32 standalone first-token logits against saved serving evidence.

Serialized correctness fixture only: short allocated cache512, native model
position contract262144 unchanged. Host logits are read solely for this numeric
comparison. No serving capacity claim, sampling path, profiling or performance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import ttnn
from transformers import AutoTokenizer
from models.autoports.qwen_qwen3_8_27b.tt.generator import Generator
from models.autoports.qwen_qwen3_8_27b.tt.model import Model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='.cache/qwen38-full')
    parser.add_argument('--serving-evidence',default='models/autoports/qwen_qwen3_8_27b/readiness_vllm/sampling_route_controls.json')
    parser.add_argument('--output',default='models/autoports/qwen_qwen3_8_27b/readiness_vllm/standalone_route_logits.json')
    args=parser.parse_args()
    if os.environ.get('TT_METAL_DEVICE_PROFILER') not in (None,'','0'):
        raise RuntimeError('Profiling is prohibited for this control')
    root=Path('models/autoports/qwen_qwen3_8_27b')
    evidence=json.loads(Path(args.serving_evidence).read_text())
    prompts=['Count: ','Random: ','Word: ','Letter: ','Number: ','TopP: ']
    assert [c['prompt'] for c in evidence['configs']]==prompts
    provenance_path=root/'doc/datatype_sweep/qualitative_prompt_format.json'
    provenance=json.loads(provenance_path.read_text())
    assert provenance['revision']=='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
    selected=root/'doc/datatype_sweep/selected_precision_config.json'
    files=[Path(__file__),Path(args.serving_evidence),provenance_path,selected,
           *sorted((root/'tt').glob('*.py')),
           Path(args.checkpoint)/'config.json',Path(args.checkpoint)/'tokenizer_config.json']
    result=dict(status='running',started_at=time.time(),hf_model=provenance['hf_model'],
        revision=provenance['revision'],checkpoint=args.checkpoint,
        scope='full64/B32 standalone prefill logits correctness; no performance claim',
        mesh=[1,4],allocated_cache_context=512,native_model_context=262144,
        precision_config=json.loads(selected.read_text()),
        prompt_mode='raw completion; encode(add_special_tokens=False)',
        logprobs='FP32 torch.log_softmax of assembled final BF16 logits; correctness-only host readback',
        gate_max_abs_logprob_difference=.02,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},runs=[])
    output=Path(args.output)
    def save():
        output.parent.mkdir(parents=True,exist_ok=True)
        tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(output)
    save()
    tokenizer=AutoTokenizer.from_pretrained(args.checkpoint)
    ids=[tokenizer.encode(p,add_special_tokens=False) for p in prompts]
    result['prompts']=[dict(prompt=p,token_ids=t) for p,t in zip(prompts,ids)]
    torch.set_num_threads(8)
    ttnn.CONFIG.throw_exception_on_fallback=True
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
    gen=None
    try:
        model=Model(args.checkpoint,mesh,precision_config=selected)
        assert model.layer_indices==list(range(64)) and model.config.max_position_embeddings==262144
        gen=Generator(model,args.checkpoint,batch_size=32,context_length=512,prefill_chunk_size=128)
        # No decode preparation is needed: this fixture evaluates only prefill.
        # In particular it does not warm irrelevant caches or create traces.
        for mapping in (list(range(6)),list(reversed(range(6)))):
            gen.reset()
            records=[None]*6
            # Match adapter's per-logical-length grouping exactly.
            for length in sorted(set(map(len,ids))):
                tokens=torch.zeros(32,length,dtype=torch.int64)
                lengths=[0]*32
                members=[i for i,t in enumerate(ids) if len(t)==length]
                for i in members:
                    slot=mapping[i];tokens[slot]=torch.tensor(ids[i]);lengths[slot]=length
                logits=gen.prefill_forward(tokens,page_table=gen.page_table,kv_cache=gen.kv_cache,
                                           prompt_lens=lengths,return_device=True)
                host=gen.host_logits(logits).reshape(32,-1)
                for i in members:
                    logp=torch.log_softmax(host[mapping[i]].float(),dim=-1)
                    values,indices=torch.topk(logp,20)
                    comparisons=[]
                    for run_number,serving_run in enumerate(evidence['host_logprobs']['runs']):
                        observed=serving_run[i]['logprobs']
                        top=observed['top_logprobs'][0]
                        deltas={key:float(logp[int(key.split(':')[1])])-value for key,value in top.items()}
                        maximum=max(map(abs,deltas.values()))
                        comparisons.append(dict(serving_run=run_number,
                            greedy_id_equal=observed['tokens']==[f'token_id:{int(logp.argmax())}'],
                            max_abs_logprob_difference=maximum,deltas=deltas,
                            pass_gate=maximum<=.02 and observed['tokens']==[f'token_id:{int(logp.argmax())}']))
                    records[i]=dict(prompt_index=i,slot=mapping[i],greedy_id=int(logp.argmax()),
                        top20={f'token_id:{int(k)}':float(v) for k,v in zip(indices,values)},
                        comparisons=comparisons)
                del host
            result['runs'].append(dict(slot_mapping=mapping,outputs=records));save()
        result['reordered_baseline_exact']=all(a['top20']==b['top20'] and a['greedy_id']==b['greedy_id']
            for a,b in zip(result['runs'][0]['outputs'],result['runs'][1]['outputs']))
        passed=all(c['pass_gate'] for run in result['runs'] for record in run['outputs'] for c in record['comparisons'])
        result['status']='pass' if passed and result['reordered_baseline_exact'] else 'more-work-needed'
    except BaseException as error:
        result['status']='error';result['error']=repr(error)
        raise
    finally:
        ttnn.synchronize_device(mesh)
        if gen is not None:
            traces=set(gen.decode_traces.values())
            traces.update(t for t in (gen.sample_trace,gen.buffered_sample_trace) if t is not None)
            for trace in traces:ttnn.release_trace(mesh,trace)
        ttnn.close_mesh_device(mesh)
        save()
    if result['status']!='pass':
        raise SystemExit('Standalone logit baseline differs; see '+str(output))


if __name__=='__main__':
    main()
