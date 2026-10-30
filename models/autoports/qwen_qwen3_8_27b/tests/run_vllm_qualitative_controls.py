# SPDX-License-Identifier: Apache-2.0
"""Prompt-matched full-model controls for serving quality, not a benchmark.

Run serially after stopping all serving workers. Uses all64 selected-precision
layers, native context262144, canonical on-device token-out sampling and EOS.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',default='.cache/qwen38-full')
    parser.add_argument('--output',default='models/autoports/qwen_qwen3_8_27b/readiness_vllm/full_model_controls.json')
    args=parser.parse_args()
    root=Path('models/autoports/qwen_qwen3_8_27b')
    package=os.environ.get('TT_MODEL_BRINGUP_ROOT')
    if not package:
        raise RuntimeError('Set TT_MODEL_BRINGUP_ROOT through the installed environment.py setup')
    prompt_source=Path(package)/'runtime/readiness_check/vllm_prompts.txt'
    prompts=[p.strip() for p in prompt_source.read_text().split('\n\n') if p.strip()]
    if len(prompts)!=6:
        raise ValueError(f'Expected the pinned six-prompt serving suite, found{len(prompts)}')
    if os.environ.get('TT_METAL_DEVICE_PROFILER') not in (None,'','0'):
        raise RuntimeError('This control runner must not enable device profiling')
    provenance_path=root/'doc/datatype_sweep/qualitative_prompt_format.json'
    provenance=json.loads(provenance_path.read_text())
    if provenance['hf_model']!='Qwen/Qwen3.8-27B' or provenance['revision']!='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0':
        raise ValueError('Control revision differs from the completed datatype-sweep checkpoint')
    shared_source=root/'doc/qualitative_prompts.json'
    shared=json.loads(shared_source.read_text())['prompts']
    suite=[dict(id=f'vllm_{i:02d}',suite='vllm',messages=[{'role':'user','content':prompt}])
           for i,prompt in enumerate(prompts,1)]
    suite += [dict(id=f'shared_{item["id"]}',suite='shared',messages=item['messages']) for item in shared]
    files=[Path(__file__),prompt_source,shared_source,provenance_path,
           root/'doc/datatype_sweep/selected_precision_config.json',
           *sorted((root/'tt').glob('*.py')),
           Path(args.checkpoint)/'config.json',Path(args.checkpoint)/'generation_config.json',
           Path(args.checkpoint)/'tokenizer_config.json']
    result=dict(hf_model=provenance['hf_model'],revision=provenance['revision'],
        revision_evidence=str(provenance_path),checkpoint=args.checkpoint,
        source='full-model autoregressive qualitative control',
        scope='all64 layers, batch1, native context262144; no vLLM or performance measurement',
        mesh=[1,4],prompt_mode='chat',endpoint_equivalence='/v1/chat/completions, one user message, no system message',
        rendering='AutoTokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True); encode(add_special_tokens=False)',
        prompt_sources=[str(prompt_source),str(shared_source)],
        generation=dict(do_sample=False,temperature=0,top_k=1,top_p=0,max_new_tokens=256,
                        stop_on_eos=True,host_sampling=False,buffered_output=True),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        diagnostics={key:os.environ.get(key) for key in ('TT_METAL_TRACE_ALLOC_TRACKING',
            'TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE','TT_METAL_DEVICE_PROFILER','TT_METAL_WATCHER')},
        status='running',outputs=[])
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        temporary=output.with_suffix(output.suffix+'.tmp')
        temporary.write_text(json.dumps(result,indent=2)+'\n')
        temporary.replace(output)
    save()
    torch.set_num_threads(8)
    ttnn.CONFIG.throw_exception_on_fallback=True
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
    gen=None
    try:
        gen=build_generator(root,mesh,checkpoint=args.checkpoint,batch_size=1,context_length=262144)
        if gen.model.layer_indices!=list(range(64)) or gen.batch_size!=1 or gen.context_length!=262144:
            raise ValueError('Control did not construct the required complete model/capacity')
        if not gen.tokenizer.chat_template:
            raise ValueError('Checkpoint must retain its HF-declared chat template')
        result.update(precision_config=gen.model.precision_config,
            tokenizer_class=type(gen.tokenizer).__name__,chat_template_present=True,
            eos_ids=sorted(gen.eos_token_ids),
            throw_exception_on_fallback=bool(ttnn.CONFIG.throw_exception_on_fallback),
            decode_trace_ids={str(key):str(value) for key,value in gen.decode_traces.items()},
            sampling_trace_id=str(gen.sample_trace),buffered_sampling_trace_id=str(gen.buffered_sample_trace))
        for item in suite:
            rendered=gen.tokenizer.apply_chat_template(item['messages'],tokenize=False,add_generation_prompt=True)
            prompt_tokens=gen.tokenizer.encode(rendered,add_special_tokens=False)
            gen.set_sampling_params(top_k=1,top_p=0,temperature=1,seed=1234)
            tokens=gen.generate(prompt_tokens,256,host_sampling=False,stop_on_eos=True,buffered_output=True)
            eos_positions=[i for i,token in enumerate(tokens) if token in gen.eos_token_ids]
            if eos_positions and eos_positions!=[len(tokens)-1]:
                raise AssertionError(f'EOS termination was not respected for{item["id"]}')
            record=dict(**item,rendered_prompt=rendered,prompt_tokens=prompt_tokens,
                input_tokens=len(prompt_tokens),generated_tokens=tokens,output_tokens=len(tokens),
                finish_reason='eos' if eos_positions else 'length',
                completion=gen.tokenizer.decode(tokens,skip_special_tokens=False),
                completion_without_special_tokens=gen.tokenizer.decode(tokens,skip_special_tokens=True))
            result['outputs'].append(record)
            save()
            print(f'FULL_MODEL_CONTROL {item["id"]} tokens={len(tokens)} finish={record["finish_reason"]}',flush=True)
        result['status']='complete'
    except BaseException as error:
        result['status']='failed'
        result['error']={'type':type(error).__name__,'message':str(error)}
        raise
    finally:
        save()
        if gen is not None:
            ttnn.synchronize_device(mesh)
            traces=set(gen.decode_traces.values())
            traces.update(t for t in (gen.sample_trace,gen.buffered_sample_trace) if t is not None)
            for trace in traces:ttnn.release_trace(mesh,trace)
        ttnn.close_mesh_device(mesh)


if __name__=='__main__':
    main()
