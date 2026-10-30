# SPDX-License-Identifier: Apache-2.0
"""Device-only BFP4 reset proof; reduced diagnostic, never model performance."""
import json
from pathlib import Path
import time
import torch
import ttnn

torch.set_num_threads(4)
root=Path('models/autoports/qwen_qwen3_8_27b/doc/datatype_sweep')
record=dict(scope='BFP4 paged KV reset primitive on all TP4 ranks; not full-model performance',cases=[])
mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
try:
    for pages in (1,8192):
        shape=(pages,1,32,256)
        # One initial host construction; measured resets below must stay on-device.
        value=ttnn.from_torch(torch.ones(shape,dtype=torch.bfloat16),device=mesh,
            dtype=ttnn.bfloat4_b,layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        addresses=[part.buffer_address() for part in ttnn.get_device_tensors(value)]
        assert len(addresses)==4
        samples=[]
        for fill_value in (1.0,-2.0,3.0):
            ttnn.fill(value,fill_value,output_tensor=value)
            ttnn.synchronize_device(mesh)
            for part in ttnn.get_device_tensors(value):
                actual=ttnn.to_torch(part)
                assert torch.all(actual==fill_value),('nonzero fill',pages,fill_value)
                del actual
            started=time.monotonic()
            result=ttnn.fill(value,0.0,output_tensor=value)
            ttnn.synchronize_device(mesh)
            samples.append((time.monotonic()-started)*1000)
            assert [p.buffer_address() for p in ttnn.get_device_tensors(result)]==addresses
            for part in ttnn.get_device_tensors(value):
                actual=ttnn.to_torch(part)
                assert torch.count_nonzero(actual)==0,('zero fill',pages)
                del actual
        record['cases'].append(dict(shape=shape,dtype=str(value.dtype),all_four_ranks_exact_zero=True,
            buffer_addresses_unchanged=True,ranks=len(addresses),repetitions=3,host_wall_fill_ms=samples,
            timing_scope='Primitive diagnostic; CPU HF control may run concurrently; not ranking evidence'))
        ttnn.deallocate(value)
    record['status']='pass'
finally:
    (root/'bfp4_reset_probe.json').write_text(json.dumps(record,indent=2)+'\n')
    ttnn.close_mesh_device(mesh)
