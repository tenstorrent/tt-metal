# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-shape TP4 prefill trace: warmed latency and fresh input/page-table replay."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
import torch
import ttnn
from ttnn.tools import trace_allocation_tracker
from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import MultichipDecoder
from models.autoports.ornith_ai_ornith_1_5_9b.tests.run_decoder import config, weights, pcc
from models.autoports.ornith_ai_ornith_1_5_9b.tests.validate_optimized import buffers, device_only

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--layer',type=int,required=True)
    ap.add_argument('--lengths',default='32,33,127,128,129,1023,1024,1025,4095,4096,4097,8193')
    ap.add_argument('--output',required=True);args=ap.parse_args()
    torch.set_num_threads(8);torch.manual_seed(904)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=64*1024*1024)
    rows=[]
    def host(x, integer=False):
        return ttnn.from_torch(x,dtype=ttnn.int32 if integer else ttnn.bfloat16,
              layout=ttnn.ROW_MAJOR_LAYOUT if integer else ttnn.TILE_LAYOUT,
              mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    def download(x):return [ttnn.to_torch(v) for v in ttnn.get_device_tensors(x)]
    try:
        layer=MultichipDecoder.from_state_dict(weights(args.layer),hf_config=config(),layer_idx=args.layer,mesh_device=mesh)
        for length in map(int,args.lengths.split(',')):
            capacity=((length+31)//32)*32
            state=layer.allocate_state(context=capacity)
            zero=[(x,ttnn.clone(x)) for x in buffers(state)]
            cases=[]
            for _ in range(3):
                x=(torch.randn(1,1,length,4096)*.1).bfloat16()
                table=torch.randperm(capacity//32,dtype=torch.int32)[None]
                cases.append((x,host(x),host(table,True)))
            tx=ttnn.to_device(cases[0][1],mesh);table=ttnn.to_device(cases[0][2],mesh)
            def clear():
                for dst,src in zero:ttnn.copy(src,dst)
            def refresh(case):
                ttnn.copy_host_to_device_tensor(case[1],tx);ttnn.copy_host_to_device_tensor(case[2],table)
            def forward():
                with device_only():return layer.prefill_forward(tx,state=state,page_table=table)
            expected=[]
            for case in cases:
                clear();refresh(case);out=forward()
                expected.append((download(out),[download(x) for x in buffers(state)]));del out
            eager=[]
            refresh(cases[0])
            for repeat in range(28):
                clear();ttnn.synchronize_device(mesh);begin=time.perf_counter();out=forward();ttnn.synchronize_device(mesh)
                if repeat>=3:eager.append((time.perf_counter()-begin)*1000)
                del out
            clear();ttnn.synchronize_device(mesh)
            trace=ttnn.begin_trace_capture(mesh,cq_id=0);out=forward();ttnn.end_trace_capture(mesh,trace,cq_id=0)
            trace_allocation_tracker.acknowledge_corruptible(out)
            scores=[];contributions=[];state_scores=[];agreement=[]
            for case,(wanted,wanted_states) in zip(cases,expected):
                clear();refresh(case);ttnn.execute_trace(mesh,trace,cq_id=0,blocking=True)
                got=download(out)
                scores.extend(pcc(a,b) for a,b in zip(wanted,got))
                contributions.extend(pcc(a-case[0],b-case[0]) for a,b in zip(wanted,got))
                agreement.extend(pcc(got[0],b) for b in got[1:])
                for expected_state,value in zip(wanted_states,buffers(state)):
                    state_scores.extend(pcc(a,b) for a,b in zip(expected_state,download(value)))
            traced=[]
            for repeat in range(28):
                clear();ttnn.synchronize_device(mesh);begin=time.perf_counter()
                ttnn.execute_trace(mesh,trace,cq_id=0,blocking=True)
                if repeat>=3:traced.append((time.perf_counter()-begin)*1000)
            row=dict(layer=args.layer,length=length,eager_prefill_ms=statistics.median(eager),
                     traced_prefill_ms=statistics.median(traced),eager_samples_ms=eager,traced_samples_ms=traced,
                     pcc=min(scores),contribution_pcc=min(contributions),state_pcc=min(state_scores),rank_agreement=min(agreement),
                     fresh_inputs=3,fresh_page_tables=3,captures=1,warmups=3,repeats=25)
            rows.append(row)
            Path(args.output).write_text(json.dumps(dict(records=rows,complete=False),indent=2)+'\n')
            assert min(scores+contributions+state_scores)>=.995 and min(agreement)>=.99999,row
            ttnn.release_trace(mesh,trace)
            del out,state,zero,tx,table,cases,expected
            print('PREFILL_TRACE',length,row['eager_prefill_ms'],row['traced_prefill_ms'],flush=True)
        Path(args.output).write_text(json.dumps(dict(records=rows,complete=True,devices=[0,1,2,3],mesh=[1,4],regime='serial',
             source_sha256=hashlib.sha256((Path(__file__).parents[1]/'tt/multichip_decoder.py').read_bytes()).hexdigest(),
             note='Caller-managed fixed-shape prefill trace candidate; state reset/input upload excluded in both timings.'),indent=2)+'\n')
    finally:ttnn.close_mesh_device(mesh)

if __name__=='__main__':main()
