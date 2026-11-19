# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Locate the first divergence between same-weight independent decoder instances."""
import argparse
import json
from pathlib import Path
import torch
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import MultichipDecoder, MultichipCollectives
from models.autoports.ornith_ai_ornith_1_5_9b.tests.run_decoder import config,weights,pcc

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--shared',action='store_true');ap.add_argument('--split-ccl',action='store_true');ap.add_argument('--sync-rs',action='store_true');ap.add_argument('--persistent-rs',action='store_true');ap.add_argument('--persistent-ag',action='store_true');ap.add_argument('--disable-cache',action='store_true');args=ap.parse_args()
    torch.set_num_threads(8);torch.manual_seed(927)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
    observed=[[],[]]
    original_ccl={}
    try:
        sd=weights(0);ccl=MultichipCollectives(mesh) if args.shared else None
        layers=[MultichipDecoder.from_state_dict(sd,hf_config=config(),layer_idx=0,mesh_device=mesh,collectives=ccl) for _ in range(2)]
        x=ttnn.from_torch((torch.randn(1,1,32,4096)*.1).bfloat16(),device=mesh,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        persistent_rs = [None, None]
        persistent_ag = [None, None]
        def empty(shape):
            return ttnn.empty(shape,device=mesh,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,memory_config=ttnn.L1_MEMORY_CONFIG)
        if args.persistent_rs:
            persistent_rs=[[empty((2,1,32,4096)),empty((1,1,32,1024))] for _ in range(2)]
        if args.persistent_ag:
            persistent_ag=[empty((1,1,32,4096)) for _ in range(2)]
        if args.disable_cache:
            mesh.disable_and_clear_program_cache()
        if args.split_ccl or args.sync_rs or args.persistent_rs or args.persistent_ag:
            for name in ("reduce_scatter_minimal_async", "all_gather_async"):
                original_ccl[name]=getattr(ttnn.experimental,name)
                def wrap_ccl(name):
                    original=original_ccl[name]
                    def call(*a,**kw):
                        if name == 'reduce_scatter_minimal_async' and args.persistent_rs:
                            kw['persistent_output_buffers']=persistent_rs[idx]
                        if name == 'all_gather_async' and args.persistent_ag:
                            kw['persistent_output_tensor']=persistent_ag[idx]
                        result=original(*a,**kw)
                        if name == 'reduce_scatter_minimal_async' and args.sync_rs:
                            ttnn.synchronize_device(mesh)
                        if args.split_ccl:
                            observed[idx].append((name,[ttnn.to_torch(t) for t in ttnn.get_device_tensors(result)]))
                        return result
                    return call
                setattr(ttnn.experimental,name,wrap_ccl(name))
        for idx,layer in enumerate(layers):
            def wrap(name):
                original=getattr(layer,name)
                def call(*a,**kw):
                    result=original(*a,**kw)
                    label=name+(':'+a[1] if len(a)>1 and isinstance(a[1],str) else '')
                    observed[idx].append((label,[ttnn.to_torch(t) for t in ttnn.get_device_tensors(result)]))
                    return result
                setattr(layer,name,call)
            for name in ['norm','linear','reduce','linear_attention','finish']:wrap(name)
            state=layer.allocate_state(context=64)
            out=layer.prefill_forward(x,state=state)
            observed[idx].append(('final',[ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)]))
        rows=[]
        for i,((label,a),(other,b)) in enumerate(zip(*observed)):
            assert label==other
            finite=[bool(torch.isfinite(x).all()) for x in b]
            rows.append(dict(index=i,operation=label,shape=list(a[0].shape),finite=finite,
                 pcc=[pcc(x,y) if ok else None for x,y,ok in zip(a,b,finite)],
                 max_error=[float((x-y).abs().max()) if ok else None for x,y,ok in zip(a,b,finite)]))
        Path(args.output).write_text(json.dumps(dict(shared=args.shared,options=vars(args),records=rows),indent=2)+'\n')
        print('WORKSPACE_PROBE',rows,flush=True)
    finally:
        for name,original in original_ccl.items():setattr(ttnn.experimental,name,original)
        ttnn.close_mesh_device(mesh)
if __name__=='__main__':main()
