"""Diagnostic bit-equality runner; execute only under the serial device wrapper."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import torch
import ttnn

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--function',default='prefill_unflatten_bf16')
    ap.add_argument('--cores',type=int)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();torch.set_num_threads(8);torch.manual_seed(743)
    spec=importlib.util.spec_from_file_location('prefill_unflatten_candidate',args.candidate)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    operation=getattr(module,args.function)
    cases=[(1,1,16512),(1,17,16512),(3,17,16512),(3,33,16512),(3,65,16512),
        (32,31,16512),(32,33,16512),(3,513,16512),(32,32,16512),(3,33,5120)]
    report={'candidate':str(args.candidate),'candidate_sha256':hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seed':743,
        'threshold':'exact BF16 bits against CPU reshape and native TTNN reshape; repeats and address stable',
        'diagnostic_only_no_perf_claim':True,'rows':[]}
    mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,1),physical_device_ids=[0],trace_region_size=0)
    try:
        read=lambda t:ttnn.to_torch(t,mesh_composer=ttnn.ConcatMeshToTensor(mesh,dim=0)).clone()
        for batch,seq,width in cases:
            host=torch.randn(1,batch*seq,width,dtype=torch.bfloat16)
            x=ttnn.from_torch(host,device=mesh,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,memory_config=ttnn.DRAM_MEMORY_CONFIG)
            baseline=ttnn.reshape(x,(batch,seq,width));reference=read(baseline)
            output=operation(x,batch,seq,cores=args.cores);actual=read(output);address=output.buffer_address()
            output=operation(x,batch,seq,output=output,cores=args.cores);repeat=read(output)
            row={'batch':batch,'seq':seq,'width':width,'cpu_bit_equal':torch.equal(actual.view(torch.int16),host.reshape(batch,seq,width).view(torch.int16)),
                'native_bit_equal':torch.equal(actual.view(torch.int16),reference.view(torch.int16)),
                'repeat_bit_equal':torch.equal(actual.view(torch.int16),repeat.view(torch.int16)),
                'stable_output_address':output.buffer_address()==address,'logical_shape':list(output.shape),'padded_shape':list(output.padded_shape)}
            report['rows'].append(row);print(row,flush=True)
            del x,baseline,output,host,actual,reference,repeat
    finally:
        ttnn.close_mesh_device(mesh);args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    assert len(report['rows'])==len(cases)
    assert all(all(r[k] for k in ('cpu_bit_equal','native_bit_equal','repeat_bit_equal','stable_output_address')) for r in report['rows'])

if __name__=='__main__':main()
