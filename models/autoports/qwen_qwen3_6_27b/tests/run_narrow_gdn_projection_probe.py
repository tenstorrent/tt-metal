"""Real-weight narrow-correction attribution and exact same-partial combine probe.

Diagnostic readbacks only; no perf claim or HF model. Run via serial device wrapper.
Changed-N projection deltas are reported, not waived; same-partial combine must be exact.
"""
import argparse,copy,gc,hashlib,importlib.util,json,sys
from pathlib import Path
from unittest.mock import patch
import torch
import ttnn
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from models.autoports.qwen_qwen3_6_27b.tests.evidence import decoder_kwargs
from models.autoports.qwen_qwen3_6_27b.tests.run_decoder import load_layer
from models.autoports.qwen_qwen3_6_27b.tests.run_projection_precision_probe import compare,metric

D=Path(__file__).resolve().parents[1]/'doc/optimized_decoder/candidates'

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate',type=Path,default=D/'narrow_gdn_correction.py')
    ap.add_argument('--baseline',type=Path,default=D/'zero_z_correction_on134.py')
    ap.add_argument('--capture',type=Path,default=D.parent/'capture41/layer0_step209.pt')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();torch.set_num_threads(8);torch.manual_seed(851)
    config=Qwen3_5TextConfig(**json.loads((D.parents[1]/'hf_config.json').read_text())['text_config'])
    weights=load_layer(0)
    modules=[load(args.baseline,'narrow_probe_baseline'),load(args.candidate,'narrow_probe_candidate')]
    options=copy.deepcopy(decoder_kwargs());base_policy=options.pop('policy',{})
    report={'diagnostic_only_no_perf_claim':True,'candidate':str(args.candidate),'baseline':str(args.baseline),
        'candidate_sha256':hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        'baseline_sha256':hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'seed':851,'base_policy':base_policy,'rows':[],'same_partial_combine':[]}
    mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,1),physical_device_ids=[0],trace_region_size=0)
    try:
        def read(t):
            if t.is_sharded():t=ttnn.to_memory_config(t,ttnn.DRAM_MEMORY_CONFIG)
            return ttnn.to_torch(t,mesh_composer=ttnn.ConcatMeshToTensor(mesh,dim=0)).clone()
        def cut(t,lo,hi):
            t=ttnn.to_memory_config(t,ttnn.DRAM_MEMORY_CONFIG)
            b,s,_=tuple(t.shape)
            return ttnn.slice(t,(0,0,lo),(b,s,hi),(1,1,1),memory_config=ttnn.DRAM_MEMORY_CONFIG)
        layers=[module.OptimizedDecoder.from_state_dict(weights,hf_config=config,layer_idx=0,mesh_device=mesh,
            **(options|{'policy':base_policy|{'zero_z_correction':True,'narrow_gdn_correction':bool(i),'narrow_gdn_verify_setup':bool(i)}}))
            for i,module in enumerate(modules)]
        key='packed_gdn';res=key+'_residual'
        p0,p1=(read(layer.weights[key]).float() for layer in layers)
        w0,w1=(read(layer.weights[res]).float() for layer in layers)
        expected=torch.cat([w0[:,:10240],w0[:,16384:16512]],-1)
        report['weights']={'primary_shapes':[list(x.shape) for x in (p0,p1)],'correction_shapes':[list(x.shape) for x in (w0,w1)],
            'primary_exact':torch.equal(p0,p1),'retained_correction_exact':torch.equal(expected,w1[:,:10368]),
            'baseline_z_zero':not bool(torch.count_nonzero(w0[:,10240:16384])),
            'narrow_padding_zero':not bool(torch.count_nonzero(w1[:,10368:]))}
        assert all(report['weights'][k] for k in ('primary_exact','retained_correction_exact','baseline_z_zero','narrow_padding_zero'))
        del p0,p1,w0,w1,expected
        captured=torch.load(args.capture,map_location='cpu',weights_only=False)['tt']['input_norm'][0].reshape(1,1,5120).bfloat16()
        cases=[('captured_decode',captured),('random_decode32',torch.randn(1,32,5120,dtype=torch.bfloat16)),
               ('prefill33',torch.randn(1,33,5120,dtype=torch.bfloat16)),('prefill_b3_s33',torch.randn(3,33,5120,dtype=torch.bfloat16)),
               ('prefill641_tail',torch.randn(1,641,5120,dtype=torch.bfloat16))]
        for label,host in cases:
            x=ttnn.from_torch(host,device=mesh,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,memory_config=ttnn.DRAM_MEMORY_CONFIG)
            saved=[]
            for index,layer in enumerate(layers):
                partials={key:[],res:[]};native=ttnn.linear
                def record(*a,**kw):
                    result=native(*a,**kw)
                    if len(a)>1:
                        for name in partials:
                            if a[1] is layer.weights[name]:partials[name].append(result)
                    return result
                with patch.object(ttnn,'linear',record):actual=layer._linear(x,key)
                assert len(partials[key])==len(partials[res])>=1
                logical_n=10368 if index else 16512
                primary_host=torch.cat([read(t).reshape(-1,t.shape[-1])[:,:16512] for t in partials[key]],0)
                residual_host=torch.cat([read(t).reshape(-1,t.shape[-1])[:,:logical_n] for t in partials[res]],0)
                saved.append((primary_host,residual_host,read(actual).reshape(-1,16512)))
                if index:
                    for chunk,(primary,correction) in enumerate(zip(partials[key],partials[res])):
                        qkv=cut(correction,0,10240);ba=cut(correction,10240,10368)
                        expanded=ttnn.concat([ttnn.pad(qkv,((0,0),(0,0),(0,6144)),value=0.0),ba],dim=-1,memory_config=ttnn.DRAM_MEMORY_CONFIG)
                        if primary.shape[-1]>16512:
                            expanded=ttnn.pad(expanded,((0,0),(0,0),(0,primary.shape[-1]-16512)),value=0.0)
                        expanded=ttnn.to_memory_config(expanded,primary.memory_config())
                        reference=ttnn.add(primary,expanded,dtype=ttnn.bfloat16)
                        combined=layer._combine_correction(primary,correction,key)
                        a=read(reference)[...,:16512];b=read(combined)
                        exact=torch.equal(a.contiguous().view(torch.int16),b.contiguous().view(torch.int16))
                        row={'case':label,'chunk':chunk,'primary_memory':str(primary.memory_config()),
                             'bit_exact':exact,'comparisons':compare(a,b,key)}
                        report['same_partial_combine'].append(row)
                        del qkv,ba,expanded,reference,combined,a,b
                del actual,partials
            a,b=saved
            residual_reference=torch.cat([a[1][:,:10240],a[1][:,16384:16512]],-1)
            row={'case':label,'shape':list(host.shape),
                'primary_partials':compare(a[0],b[0],key),'retained_residual_partials':metric(residual_reference,b[1]),
                'final_output':compare(a[2],b[2],key),
                'final_bit_equal':torch.equal(a[2].contiguous().view(torch.int16),b[2].contiguous().view(torch.int16))}
            report['rows'].append(row);print(label,row['final_output']['all_valid_channels'],flush=True)
            del x,saved,a,b;gc.collect()
    finally:
        ttnn.close_mesh_device(mesh);args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    assert len(report['rows'])==5
    assert all(r['bit_exact'] for r in report['same_partial_combine']), 'Same-input combine mismatch; inspect artifact before selecting candidate'

if __name__=='__main__':main()
