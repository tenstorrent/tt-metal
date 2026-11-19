# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Discriminate pooled history row updates from prefill numerical failures."""
import argparse
import json
from pathlib import Path
import torch
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_tiled_history_candidate import TiledHistoryCandidate
from models.autoports.ornith_ai_ornith_1_5_9b.tests.run_decoder import config, weights

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--direct',action='store_true');ap.add_argument('--queued',action='store_true');ap.add_argument('--batch',type=int,default=32);args=ap.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
    records=[]
    try:
        cls=TiledHistoryCandidate
        if args.direct:
            from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_history_row_write import DirectHistoryCandidate
            cls=DirectHistoryCandidate
        layer=cls.from_state_dict(weights(0),hf_config=config(),layer_idx=0,mesh_device=mesh)
        states=layer.allocate_batch_state(batch=args.batch,context=64)
        blank=ttnn.clone(states[0].conv)
        expected=torch.zeros(args.batch,1,3,2048,dtype=torch.bfloat16)
        def check(label):
            errors=[]
            for rank in ttnn.get_device_tensors(states[0].conv):
                errors.append(float((ttnn.to_torch(rank).permute(2,1,0,3)-expected).abs().max()))
            for i,st in enumerate(states):
                actual=layer.read_linear_state(st,'conv')
                ranks=[ttnn.to_torch(x) for x in ttnn.get_device_tensors(actual)]
                errors.extend(float((x-expected[i:i+1]).abs().max()) for x in ranks)
            record=dict(label=label,max_error=max(errors),errors=errors)
            records.append(record);Path(args.output).write_text(json.dumps(records,indent=2)+'\n')
            assert max(errors)==0, record
        check('initial_zero')
        for row in ([0,31,7,1,30,12,0,31] if args.batch==32 else [0,5,1,4,2,0,5]):
            value=(torch.arange(3).reshape(1,1,3,1)*17+torch.arange(2048).reshape(1,1,1,2048)%13+row+1).to(torch.bfloat16)
            x=ttnn.from_torch(value,device=mesh,dtype=ttnn.bfloat16,layout=ttnn.ROW_MAJOR_LAYOUT,mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            layer.write_linear_state(states[row],'conv',x)
            expected[row:row+1]=value
            if not args.queued:check(f'write_{row}')
        check('final_queued' if args.queued else 'final')
        ttnn.copy(blank,states[0].conv);expected.zero_();check('reset_zero')
        print('HISTORY_STATE_PASS',flush=True)
    finally:
        ttnn.close_mesh_device(mesh)

if __name__=='__main__':main()
