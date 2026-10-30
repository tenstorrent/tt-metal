# SPDX-License-Identifier: Apache-2.0
"""One serialized full-stack precision candidate, using normal construction."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import shlex
import torch
import ttnn
from readiness_check import run_prefill_check as prefill, run_teacher_forcing as teacher
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator
from models.autoports.qwen_qwen3_8_27b.tt.precision import load_precision

@contextlib.contextmanager
def borrow(module, gen):
    original = module._import_build_generator
    module._import_build_generator = lambda directory: lambda **kwargs: gen
    try:
        yield
    finally:
        module._import_build_generator = original


@contextlib.contextmanager
def capture_precision_calls(record):
    """Inspect actual kernel arguments during preparation, never during timing."""
    observed = {}
    originals = []
    def wrap(name, original):
        def call(*args, **kwargs):
            config = kwargs.get('compute_kernel_config')
            if len(args) >= 2 and config is not None:
                row = dict(op=name, input_dtype=str(args[0].dtype), weight_dtype=str(args[1].dtype),
                           input_shape=list(args[0].shape), weight_shape=list(args[1].shape),
                           math_fidelity=str(config.math_fidelity),
                           fp32_dest_acc_en=config.fp32_dest_acc_en)
                key = json.dumps(row, sort_keys=True)
                if key not in observed:
                    observed[key] = dict(row, calls=0)
                observed[key]['calls'] += 1
            return original(*args, **kwargs)
        return call
    for owner, name in ((ttnn, 'linear'), (ttnn.experimental, 'matmul_reduce_scatter_async')):
        original = getattr(owner, name)
        originals.append((owner, name, original))
        setattr(owner, name, wrap(name, original))
    try:
        yield
    finally:
        for owner, name, original in originals:
            setattr(owner, name, original)
        record['prepared_kernel_precision_calls'] = list(observed.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--wait-for-control', help='Untimed orchestration: wait for a completed CPU-control receipt before any measurement')
    args = parser.parse_args()
    torch.set_num_threads(8)
    root = Path('models/autoports/qwen_qwen3_8_27b')
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    config = load_precision(args.config)
    reference = root / 'readiness_aime24_chat.refpt'
    record = dict(config_id=config['config_id'], precision_config=args.config, dtype_policy=config,
                  compute_fidelity_policy=config['compute_fidelities'], status='running',
                  command='docker exec qwen38-functional-build bash '
                          'models/autoports/qwen_qwen3_8_27b/tests/optimized_multichip_env.sh '
                          'python -m models.autoports.qwen_qwen3_8_27b.tests.run_datatype_candidate '
                          + shlex.join(sys.argv[1:]),
                  hardware='4 x Blackhole p300c', mesh=[1,4],
                  commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
                  environment={name:os.environ.get(name) for name in
                               ('TT_METAL_TRACE_ALLOC_TRACKING','TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE',
                                'TT_METAL_WATCHER','TT_METAL_DEVICE_PROFILER')},
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  reference=str(reference),reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                  measurement_regime='traced_teacher_forcing', batch=1, input_tokens=203, generated_tokens=100,
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'tt').glob('*.py')})
    def save(): out.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    save()
    mesh = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
        started=time.monotonic()
        with capture_precision_calls(record):
            gen=build_generator(root,mesh,context_length=262144,precision_config=args.config)
        record['construction_seconds']=time.monotonic()-started
        record['runtime_policy']=gen.model.precision_config
        record['runtime_layers']=[]
        for layer,state in zip(gen.model.layers,gen.kv_cache):
            row=dict(index=layer.layer_idx,layer_type=layer.layer_type,
                     weights={role:str(getattr(layer,role).dtype) for role in ('mix','out','gate','up','gateup','down')},
                     policy=layer.policy,
                     fused_prefill_compute=dict(math_fidelity=str(layer.matmul_ckc.math_fidelity),
                                                fp32_dest_acc_en=layer.matmul_ckc.fp32_dest_acc_en),
                     kv_dtype=str(state.key.dtype) if hasattr(state,'key') else None,
                     recurrent_dtype=str(state.recurrent.dtype) if hasattr(state,'recurrent') else None)
            record['runtime_layers'].append(row)
        record['terminal_runtime']=dict(head_dtype=str(gen.model.head.dtype),embedding_dtype=str(gen.model.embedding.dtype),
                                        head_compute=dict(math_fidelity=str(gen.model.head_ckc.math_fidelity),
                                                          fp32_dest_acc_en=gen.model.head_ckc.fp32_dest_acc_en))
        save()
        if args.wait_for_control:
            receipt=Path(args.wait_for_control)
            print('WAIT_FOR_CPU_CONTROL',str(receipt),flush=True)
            while not receipt.exists():
                time.sleep(1)
            control=json.loads(receipt.read_text())
            assert control['exit_code']==0 and control['status']=='complete',control
            record['cpu_control_completed_before_measurement']=control
            record['cpu_control_receipt_sha256']=hashlib.sha256(receipt.read_bytes()).hexdigest()
            save()
        with borrow(prefill,gen):
            record['prefill_accuracy']=prefill.run_prefill_check(model_dir=root,reference_path=reference,mesh_device=mesh)
        with borrow(teacher,gen):
            record['warmup_accuracy']=teacher.run_teacher_forcing(model_dir=root,reference_path=reference,mesh_device=mesh)
            record['runs']=[]
            for _ in range(args.repeats):
                accuracy=teacher.run_teacher_forcing(model_dir=root,reference_path=reference,mesh_device=mesh)
                metrics=dict(gen.last_generation_metrics)
                assert metrics['decode_trace'] and metrics['steady_state_counters']['model_replays']==99
                assert metrics['steady_state_counters']['captures']==0
                record['runs'].append(dict(accuracy=accuracy,metrics=metrics)); save()
        accuracies=[r['accuracy'][0] for r in record['runs']]
        record.update({key:min(a[key] for a in accuracies) for key in ('top1','top5','top100')})
        record.update(ttft_ms=statistics.median(r['metrics']['ttft_ms'] for r in record['runs']),
                      decode_tps=statistics.median(r['metrics']['decode_tps_per_user'] for r in record['runs']),
                      trace_verified=True,compared_positions=100)
        record['accuracy_pass']=all(a['top1']>=.90 and a['top5']>=.98 and a['top100']==1
                                    for a in [*accuracies, *record['prefill_accuracy']])
        record['status']='pass' if record['accuracy_pass'] else 'fail_accuracy'
        record['selection_eligible']=False
        record['remaining_gates']='Long-prefix/free-running, memory, batch validation and post-selection token-out pending'
        save()
    except BaseException as error:
        record.update(status='runtime_failure', error_type=type(error).__name__, error=str(error)[:4000])
        save(); raise
    finally:
        if mesh is not None: ttnn.close_mesh_device(mesh)

if __name__ == '__main__': main()
