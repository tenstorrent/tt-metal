# SPDX-License-Identifier: Apache-2.0
"""Deferred native/separate prefill projection control; eager, identical operands."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator
from models.autoports.qwen_qwen3_8_27b.tt.multichip_decoder import MultichipDecoder
from models.autoports.qwen_qwen3_8_27b.tt.optimized_decoder import OptimizedDecoder
from models.autoports.qwen_qwen3_8_27b.tests.full_model_evidence import runtime_build
from models.autoports.qwen_qwen3_8_27b.tests.probe_gdn_same_operands import metrics as finite_metrics

ROOT=Path('models/autoports/qwen_qwen3_8_27b')


def metrics(left,right):
    left_finite=torch.isfinite(left);right_finite=torch.isfinite(right)
    if not bool(left_finite.all() and right_finite.all()):
        return dict(max_abs=None,rms=None,pcc=None,exact=bool(torch.equal(left,right)),finite=False,
                    nonfinite_left=int((~left_finite).sum()),nonfinite_right=int((~right_finite).sum()),
                    missing_reason='Nonfinite input; numerical comparison is undefined')
    return finite_metrics(left,right)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='.cache/qwen38-full')
    p.add_argument('--probe-layers',type=int,nargs='+',default=[0])
    p.add_argument('--matched-matmul',action='store_true')
    p.add_argument('--matched-reduction',action='store_true')
    p.add_argument('--reference',default=str(ROOT/'doc/full_model/benchmark_hf_control_512.json'))
    p.add_argument('--output',default=str(ROOT/'doc/full_model/prefill_rs_same_operands.json'))
    a=p.parse_args();assert set(a.probe_layers)<=set(range(4)) and a.probe_layers
    if a.matched_reduction:
        a.matched_matmul=True
        if a.output==str(ROOT/'doc/full_model/prefill_rs_same_operands.json'):
            a.output=str(ROOT/'doc/full_model/prefill_rs_matched_reduction.json')
    if a.matched_matmul and a.output==str(ROOT/'doc/full_model/prefill_rs_same_operands.json'):
        a.output=str(ROOT/'doc/full_model/prefill_rs_matched_matmul.json')
    prompt=json.loads(Path(a.reference).read_text())['input_tokens'];assert len(prompt)==4096
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
    source_files=[Path(__file__),ROOT/'tests/probe_gdn_same_operands.py',*(ROOT/'tt').glob('*.py')]
    for directory in ('ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_async',
                      'ttnn/cpp/ttnn/operations/experimental/matmul/ccl_fusion'):
        source_files.extend(f for f in Path(directory).rglob('*') if f.suffix in ('.cpp','.hpp','.h'))
    result=dict(status='running',scope='Real reduced layers0..3 prefix; operator comparison, not semantic quality/performance',
        layers=[0,1,2,3],probe_layers=a.probe_layers,batch=1,context=262144,sequence=4096,
        input_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),runtime=runtime_build(),
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files},comparisons=[])
    def persist():target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    ttnn.CONFIG.throw_exception_on_fallback=True
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
    original=MultichipDecoder._linear_impl
    original_fused_op=ttnn.experimental.matmul_reduce_scatter_async
    def host(tensor):
        return [ttnn.to_torch(local).clone() for local in ttnn.get_device_tensors(tensor)]
    try:
        print('RS_SAME_OPERAND_BUILD',flush=True)
        gen=build_generator(ROOT,mesh,checkpoint=a.checkpoint,layer_indices=[0,1,2,3],
                            batch_size=1,context_length=262144,prepare=False)
        indices={id(getattr(layer,'_prefill_impl',layer)):i
                 for i,layer in zip(gen.model.layer_indices,gen.model.layers)}
        def compare(layer,x,w,activation=None,**kwargs):
            role=layer.weight_roles[id(w)];index=indices.get(id(layer))
            selected=(index in a.probe_layers and layer.policy.get('_prefill_phase')
                      and layer.policy.get('prefill_native_rs') and role in ('out','down')
                      and list(x.shape)[:2]==[1,4096])
            if not selected:
                return original(layer,x,w,activation,**kwargs)
            assert activation is None and all(k=='decode_batch' and v is None for k,v in kwargs.items())
            owner=layer.ar_workspace
            owner.require_prefill_role(role)
            print('RS_SAME_OPERAND',index,role,flush=True)
            captured={}
            def capture_returned_mm(*op_args,**op_kwargs):
                outputs=original_fused_op(*op_args,**op_kwargs)
                ttnn.synchronize_device(mesh)
                # Return0 is the actual local matmul output. The supplied
                # persistent_intermediate_buffer is RS scratch, not a dense MM.
                captured['mm']=host(outputs[0])
                captured['dtype']=str(outputs[0].dtype)
                captured['shape']=list(outputs[0].shape)
                if a.matched_matmul:
                    assert len(op_args)==2
                    captured['operands']=op_args
                    captured['matmul_options']=dict(program_config=op_kwargs['program_config'],
                        compute_kernel_config=op_kwargs['compute_kernel_config'],dtype=op_kwargs['dtype'],
                        memory_config=op_kwargs['memory_config_mm'])
                return outputs
            ttnn.experimental.matmul_reduce_scatter_async=capture_returned_mm
            try:
                native=original(layer,x,w,activation,**kwargs)
            finally:
                ttnn.experimental.matmul_reduce_scatter_async=original_fused_op
            assert 'mm' in captured,'Native fused call was not intercepted'
            # A control may reuse CCL workspaces. Preserve the exact production
            # return before any additional operation; outer _linear still owns
            # the one require/complete transition for this role.
            production=ttnn.clone(native)
            ttnn.synchronize_device(mesh)
            native_output=host(native)
            native_mm=captured['mm']
            matched_host=None
            matched_rs_host=None
            if a.matched_matmul:
                print('RS_MATCHED_STANDALONE_MATMUL',index,role,flush=True)
                assert captured['matmul_options']['dtype']==ttnn.bfloat8_b
                matched=ttnn.matmul(*captured['operands'],**captured['matmul_options'])
                ttnn.synchronize_device(mesh)
                matched_host=host(matched)
                assert str(matched.dtype)==captured['dtype'] and list(matched.shape)==captured['shape']
                matched_spec=dict(shape=list(matched.shape),dtype=str(matched.dtype),
                    program_config=str(captured['matmul_options']['program_config']),
                    compute_kernel_config=str(captured['matmul_options']['compute_kernel_config']),
                    memory_config=str(captured['matmul_options']['memory_config']),
                    input_specs=[dict(shape=list(t.shape),dtype=str(t.dtype),layout=str(t.layout)) for t in captured['operands']],
                    scope='Eager standalone matmul, same actual operands/config objects and direct BF8 output as fused MM')
                if a.matched_reduction:
                    print('RS_MATCHED_REDUCTION',index,role,flush=True)
                    matched_rs=layer._reduce(ttnn.reshape(matched,[1,4096,5120]),role)
                    ttnn.synchronize_device(mesh)
                    matched_rs_host=host(matched_rs)
                    owner.require_prefill_role(role)
                    matched_spec['reduction_scope']='Original layer._reduce on identical direct-BF8 MM, unchanged policy; native fused RS versus standalone RS schedule'
                    matched_spec['reduction_output_dtype']=str(matched_rs.dtype)
                    del matched_rs
                del matched
            # This is exactly _linear_impl's separate path for row projections
            # at B1/S4096: base matmul then the original layer's _reduce. No
            # flags, weights, compute configs, residual layouts or dtypes change.
            separate_mm=OptimizedDecoder._linear(layer,x,w,activation,**kwargs)
            ttnn.synchronize_device(mesh)
            separate_bf16=host(separate_mm)
            ccl_dtype=getattr(ttnn,layer.policy['prefill_ccl_dtype'])
            separate_quantized=ttnn.typecast(separate_mm,ccl_dtype)
            separate_bf8=host(separate_quantized)
            separate=layer._reduce(separate_mm,role)
            ttnn.synchronize_device(mesh)
            separate_output=host(separate)
            owner.require_prefill_role(role)  # Controls must not advance owner order.
            assert len(native_mm)==len(separate_bf16)==len(native_output)==4
            # Compact CPU reference uses only final32 rows of the SAME TT input
            # and dequantized SAME BF4 weights. It is input-exact float32 matmul,
            # not HF-weight reference or a demand for bit-exact LoFi arithmetic.
            tail=ttnn.slice(x,[0,4064,0],[1,4096,x.shape[-1]],[1,1,1])
            input_tail=host(tail)
            local_refs=[]
            for rank,local_weight in enumerate(ttnn.get_device_tensors(w)):
                weight=ttnn.to_torch(local_weight).float().reshape(w.shape[-2],w.shape[-1])
                local_refs.append(input_tail[rank].float().reshape(32,-1) @ weight)
                del weight
            global_ref=torch.stack(local_refs).sum(0)
            row=dict(layer=index,role=role,input_shape=list(x.shape),input_dtype=str(x.dtype),weight_dtype=str(w.dtype),
                native_geometry=layer.policy['prefill_rs_geometry'],separate_role_policy=layer.policy.get(role,{}),
                native_mm_dtype=captured['dtype'],native_mm_shape=captured['shape'],
                native_mm_source='matmul_reduce_scatter_async return0, synchronized immediate readback; never persistent RS scratch',
                separate_mm_dtype=str(separate_mm.dtype),
                ccl_dtype=str(ccl_dtype),reference_rows=[4064,4096],
                reference_basis='CPU FP32 matmul of dequantized accepted BF4 weights and actual TT BF16 input, then FP32 rank sum/scatter',ranks=[])
            if a.matched_matmul:
                row['matched_matmul']=matched_spec
            for rank in range(4):
                nm=native_mm[rank].reshape(4096,5120)
                sm=separate_bf16[rank].reshape(4096,5120)
                sq=separate_bf8[rank].reshape(4096,5120)
                no=native_output[rank].reshape(4096,1280)
                so=separate_output[rank].reshape(4096,1280)
                ref_rs=global_ref[:,rank*1280:(rank+1)*1280]
                row['ranks'].append(dict(rank=rank,
                    native_mm_vs_separate_bf16=metrics(nm,sm),native_mm_vs_separate_bf8=metrics(nm,sq),
                    separate_quantization_delta=metrics(sm,sq),post_rs_native_vs_separate=metrics(no,so),
                    tail_native_mm_vs_fp32=metrics(nm[-32:],local_refs[rank]),
                    tail_separate_bf16_vs_fp32=metrics(sm[-32:],local_refs[rank]),
                    tail_separate_bf8_vs_fp32=metrics(sq[-32:],local_refs[rank]),
                    tail_native_rs_vs_fp32=metrics(no[-32:],ref_rs),tail_separate_rs_vs_fp32=metrics(so[-32:],ref_rs)))
                if a.matched_matmul:
                    matched_rows=matched_host[rank].reshape(4096,5120)
                    row['ranks'][-1].update(native_mm_vs_matched_standalone=metrics(nm,matched_rows),
                        tail_matched_standalone_vs_fp32=metrics(matched_rows[-32:],local_refs[rank]),
                        matched_standalone_vs_separate_bf8=metrics(matched_rows,sq))
                if a.matched_reduction:
                    matched_rs_rows=matched_rs_host[rank].reshape(4096,1280)
                    row['ranks'][-1].update(native_post_rs_vs_matched_reduction=metrics(no,matched_rs_rows),
                        tail_matched_reduction_vs_fp32=metrics(matched_rs_rows[-32:],ref_rs),
                        matched_reduction_vs_separate_post_rs=metrics(matched_rs_rows,so))
            result['comparisons'].append(row);persist()
            return production
        MultichipDecoder._linear_impl=compare
        gen.reset()
        meta=gen._set_bucket(4096,torch.tensor([prompt]),[4096],0,[0])
        hidden=gen.model.prefill_forward(meta['tokens'],positions=meta['positions'],kv_cache=gen.kv_cache,
            page_table=gen.page_table,batch=1,length=4096,prepared_metadata=meta,return_hidden=True)
        ttnn.synchronize_device(mesh);del hidden
        assert {(r['layer'],r['role']) for r in result['comparisons']}=={(i,role) for i in a.probe_layers for role in ('out','down')}
        result.update(status='complete',limitations=[
            'Native direct-BF8 and separate BF16-then-BF8 differ in blocking/rounding and RS schedule; disagreement alone is not a kernel bug.',
            'CPU FP32 reference covers last32 rows only and uses accepted dequantized inputs, not full-precision HF weights.',
            'No semantic autoregressive conclusion, timing, profiling, trace, or production policy change.'])
        print('RS_SAME_OPERAND_COMPLETE',flush=True)
    except BaseException as error:
        result.update(status='fail',error_type=type(error).__name__,error=str(error)[:2000]);raise
    finally:
        MultichipDecoder._linear_impl=original
        ttnn.experimental.matmul_reduce_scatter_async=original_fused_op
        persist();ttnn.close_mesh_device(mesh)


if __name__=='__main__':main()
