"""Serialized full-stack precision evaluation; all ranking uses traced teacher forcing."""
import argparse
import hashlib
import json
import subprocess
import shlex
import os
import sys
import time
from pathlib import Path
import torch
import ttnn
from readiness_check import run_prefill_check, run_teacher_forcing
from readiness_check.schema import load_reference
from models.autoports.qwen_qwen3_8_flash_next.tt.generator import build_generator
from models.autoports.qwen_qwen3_8_flash_next.tt.precision import runtime_summary
from models.autoports.qwen_qwen3_8_flash_next.tests.full_model_readiness import SessionView, prediction_details
from models.autoports.qwen_qwen3_8_flash_next.tests.multichip_capacity_probe import view

ROOT = Path('models/autoports/qwen_qwen3_8_flash_next').resolve()
OUT = ROOT / 'doc/datatype_sweep'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--checkpoint', default='/home/vkovacevic/qwen38-flash/hf-local/de4b8e4d43b917e7706784d8bb445c9af86a3540')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--final', action='store_true')
    ap.add_argument('--nonaligned', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(8)
    ttnn.CONFIG.throw_exception_on_fallback = True
    reference = ROOT / 'readiness_aime24_chat.refpt'
    entry = load_reference(reference).entries[0]
    assert entry.num_generated == 100
    report = dict(config_id=a.tag, args=vars(a), command=shlex.join([sys.executable, '-m', 'models.autoports.qwen_qwen3_8_flash_next.tests.datatype_sweep', *sys.argv[1:]]), reference=str(reference),
                  reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                  hardware='4 Blackhole p300c', mesh=[1,4], physical_order=[3,2,1,0],
                  commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'tt').glob('*.py')},
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  audit_sha256=hashlib.sha256((ROOT/'tests/datatype_runtime_audit.py').read_bytes()).hexdigest(),
                  measurement_regime='traced teacher forcing; B1 prompt203 gen100; forced input uploads and token readbacks included',
                  status='running')
    report['environment'] = {k: os.environ.get(k) for k in ('TT_METAL_TRACE_ALLOC_TRACKING', 'TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE', 'TT_METAL_WATCHER', 'TT_METAL_DEVICE_PROFILER', 'TT_METAL_HOME', 'PYTHONPATH', 'LD_LIBRARY_PATH')}
    native = Path('ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn')
    report['native_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in native.rglob('*') if p.suffix in ('.hpp', '.cpp')}
    report['binary_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path('build/lib/_ttnncpp.so'), Path('ttnn/ttnn/_ttnn.so'))}
    def save():
        destination=OUT / f'{a.tag}.json'
        temporary=destination.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report, indent=2, default=str)+'\n')
        temporary.replace(destination)
    save()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1,4), trace_region_size=768000000)
    gen = None
    try:
        tick = time.perf_counter()
        gen = build_generator(ROOT, mesh, checkpoint=a.checkpoint, precision_config=a.config,
                              layer_indices=[0,1,3] if a.smoke else None)
        report['runtime'] = runtime_summary(gen.model)
        report['generator_runtime'] = dict(
            logits_dtype=str(gen.tt_logits.dtype), token_ids_dtype=str(gen.tt_token.dtype),
            history_dtype=str(gen.tt_history.dtype), max_top_k=gen.sampler.config.max_top_k,
            parameter_dtypes={name:[str(t.dtype) for t in tensors] for name,tensors in gen.sample_params.items()})
        policy=gen.model.precision_policy
        assert gen.tt_logits.dtype==getattr(ttnn,policy['sampling']['logits'])
        assert gen.tt_token.dtype==getattr(ttnn,policy['sampling']['token_ids'])
        assert gen.sampler.config.max_top_k==policy['sampling']['max_top_k']
        report['load_seconds'] = time.perf_counter()-tick
        report['loaded_memory'] = view(mesh)
        save()
        from models.autoports.qwen_qwen3_8_flash_next.tests.datatype_runtime_audit import RuntimePolicyAudit
        with RuntimePolicyAudit(gen.model) as audit:
            gen.prepare()
        report["captured_native_arguments"] = audit.report()
        assert any(r["phase"] == "capture" and r["op"] == "linear" for r in report["captured_native_arguments"])
        report['prepared_memory'] = view(mesh)
        report['trace_memory'] = view(mesh,ttnn.BufferType.TRACE)
        ids = dict(gen.traces)
        report['trace_ids'] = {k:str(v) for k,v in ids.items()}
        report['smoke_tokens'] = gen.generate([100,101,102,103,104,105,106],4)
        save()
        if a.smoke:
            report['status']='smoke_pass'
            save()
            return
        borrowed = SessionView(gen)
        run_prefill_check._import_build_generator = lambda _: lambda **kwargs: borrowed
        run_teacher_forcing._import_build_generator = lambda _: lambda **kwargs: borrowed
        report['prefill'] = run_prefill_check.run_prefill_check(model_dir=ROOT, reference_path=reference, mesh_device=mesh)
        report['prefill_predictions'] = prediction_details(borrowed.prefill_predictions[entry.tf_prompt_len-1:entry.tf_prompt_len+99],entry)
        save()
        report['teacher_forcing_repetitions'] = []
        for _ in range(3):
            before = dict(gen.counters)
            result = run_teacher_forcing.run_teacher_forcing(model_dir=ROOT,reference_path=reference,mesh_device=mesh)
            delta = {k:gen.counters[k]-before.get(k,0) for k in gen.counters}
            assert ids == gen.traces and delta.get('captures',0)==0 and delta.get('releases',0)==0
            assert delta.get('decode_replays',0)==99, delta
            predictions=list(borrowed.last_tokens)
            if report['teacher_forcing_repetitions']:
                assert predictions==report['teacher_forcing_repetitions'][0]['predicted_tokens']
            report['teacher_forcing_repetitions'].append(dict(metrics=result,counter_delta=delta,predicted_tokens=predictions))
            save()
        report['teacher_forcing'] = result
        report['decode_predictions'] = prediction_details(borrowed.last_tokens,entry)
        if a.final:
            prompt = entry.prompt_tokens[0].tolist()
            primary=(prompt*2)[:128]
            report['post_selection_token_out_input_tokens'] = primary
            gen.generate(primary,128)
            report['post_selection_token_out'] = []
            for _ in range(3):
                before=dict(gen.counters)
                tokens=gen.generate(primary,128)
                delta={k:gen.counters[k]-before.get(k,0) for k in gen.counters}
                assert delta.get('decode_replays')==127 and delta.get('token_refreshes',0)==0, delta
                assert delta.get('token_readbacks')==1 and delta.get('history_readbacks')==1, delta
                if report['post_selection_token_out']:
                    assert tokens==report['post_selection_token_out'][0]['tokens']
                report['post_selection_token_out'].append(dict(perf=dict(gen.last_perf),tokens=tokens,counter_delta=delta))
                save()
            from models.autoports.qwen_qwen3_8_flash_next.tests.full_model_qualitative import run_tt
            dest=OUT/a.tag
            dest.mkdir(exist_ok=True)
            report['qualitative']=run_tt(gen,ROOT,output_dir=dest)
        if a.final or a.nonaligned:
            # Awkward tails compare generated tokens with the prepared eager path.
            prompt = entry.prompt_tokens[0].tolist()
            buckets=gen.prefill_trace_buckets.copy()
            report['nonaligned']=[]
            for n in (31,32,33,127,128,129,203,511,512,513,2053):
                inp=(prompt*(n//len(prompt)+1))[:n]
                traced=gen.generate(inp,4)
                gen.prefill_trace_buckets=[]
                eager=gen.generate(inp,4)
                gen.prefill_trace_buckets=buckets
                assert traced==eager,(n,traced,eager)
                report['nonaligned'].append(dict(length=n,traced=traced,eager=eager))
                save()
        assert ids==gen.traces
        report['counters']=dict(gen.counters)
        report['status']='pass' if all(r['top1']>=.90 and r['top5']>=.98 and r['top100']==1 for rows in [report['prefill'], *[x['metrics'] for x in report['teacher_forcing_repetitions']]] for r in rows) else 'accuracy_fail'
        save()
        print('DATATYPE_RESULT',a.tag,report['status'],report['teacher_forcing'],flush=True)
    except Exception as e:
        report.update(status='runtime_fail',error=repr(e))
        save()
        raise
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)

if __name__=='__main__':
    main()
