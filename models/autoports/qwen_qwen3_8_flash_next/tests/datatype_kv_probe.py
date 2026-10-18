"""Discriminating native format checks; neither failure is a model accuracy result."""
import json
from pathlib import Path
import torch
import ttnn

def main():
    torch.set_num_threads(8)
    ttnn.CONFIG.throw_exception_on_fallback=True
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
    results=[]
    try:
        try:
            ttnn.empty((1,1,32,512),dtype=ttnn.bfloat8_b,layout=ttnn.ROW_MAJOR_LAYOUT,device=mesh)
        except RuntimeError as e:
            results.append(dict(attempt='native BFP8 row-major cache',status='unsupported',error=str(e)))
        else:
            raise AssertionError('BFP8 row-major unexpectedly supported; investigate')
        # Adapt to the required native BFP8 tensor layout, then test the consumer.
        kv=ttnn.from_torch(torch.zeros(1,1,32,512),dtype=ttnn.bfloat8_b,layout=ttnn.TILE_LAYOUT,device=mesh,mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        q=ttnn.from_torch(torch.zeros(1,32,1,512),dtype=ttnn.bfloat16,layout=ttnn.ROW_MAJOR_LAYOUT,device=mesh,mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        ids=ttnn.from_torch(torch.zeros(1,1,1,128,dtype=torch.int32),dtype=ttnn.uint32,layout=ttnn.ROW_MAJOR_LAYOUT,device=mesh,mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        try:
            ttnn.transformer.sparse_sdpa(q,kv,ids,256,kv_format=ttnn.transformer.SparseKVFormat.FP8_E4M3,scale=.0625,k_chunk_size=128)
        except RuntimeError as e:
            results.append(dict(attempt='adapted native TILE BFP8 passed to packed consumer',status='unsupported',error=str(e)))
        else:
            raise AssertionError('Packed sparse_sdpa unexpectedly accepts tiled BFP8; investigate')
        ttnn.synchronize_device(mesh)
        Path('models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/kv_bfp8_native_blocker.json').write_text(json.dumps(results,indent=2)+'\n')
        print('KV_FORMAT_BLOCKER_PROVEN')
    finally:
        ttnn.close_mesh_device(mesh)
if __name__=='__main__':main()
