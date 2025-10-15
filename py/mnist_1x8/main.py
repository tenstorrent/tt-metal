from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import ttnn
import my_get_device
import utils
from models.common.utility_functions import (
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
)

import ttnn_supplemental

# Inject all ttnn_supplemental CCL operations into ttnn namespace
ttnn.MeshShardDirection = ttnn_supplemental.MeshShardDirection
ttnn.MeshShardType = ttnn_supplemental.MeshShardType
ttnn.mesh_shard = ttnn_supplemental.mesh_shard
ttnn.all_gather = ttnn_supplemental.all_gather
ttnn.reduce_scatter = ttnn_supplemental.reduce_scatter
ttnn.collective_permute = ttnn_supplemental.collective_permute
ttnn.point_to_point = ttnn_supplemental.point_to_point


def _main(v1): 
  v2 = v1[0]
  v3 = v1[1]
  v4 = v1[2]
  v5 = v1[3]
  v6 = v1[4]
  v7 = v1[5]
  v8 = v1[6]
  v9 = v1[7]
  v10 = v1[8]
  v11 = my_get_device.DeviceGetter.get_device()
  v12 = ttnn.full(shape=ttnn.Shape([]), fill_value=float('-inf'), dtype=ttnn.DataType.FLOAT32, layout=ttnn.Layout.TILE, device=v11, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v13 = ttnn.full(shape=ttnn.Shape([]), fill_value=0, dtype=ttnn.DataType.FLOAT32, layout=ttnn.Layout.TILE, device=v11, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v14 = ttnn.mesh_shard(v2, v11, ttnn.MeshShardDirection.FullToShard, ttnn.MeshShardType.Devices, [8], [-1, 0])
  ttnn.deallocate(v2, False)
  v15 = ttnn.mesh_shard(v3, v11, ttnn.MeshShardDirection.FullToShard, ttnn.MeshShardType.Devices, [1, 8], [-1, 1])
  ttnn.deallocate(v3, False)
  v16 = ttnn.mesh_shard(v4, v11, ttnn.MeshShardDirection.FullToShard, ttnn.MeshShardType.Devices, [8], [-1, 0])
  ttnn.deallocate(v4, False)
  v17 = ttnn.mesh_shard(v5, v11, ttnn.MeshShardDirection.FullToShard, ttnn.MeshShardType.Devices, [1, 8], [-1, 1])
  ttnn.deallocate(v5, False)
  v18 = ttnn.mesh_shard(v6, v11, ttnn.MeshShardDirection.FullToShard, ttnn.MeshShardType.Devices, [8], [-1, 0])
  ttnn.deallocate(v6, False)
  v19 = ttnn.mesh_shard(v7, v11, ttnn.MeshShardDirection.FullToShard, ttnn.MeshShardType.Devices, [1, 8], [-1, 1])
  ttnn.deallocate(v7, False)
  v20 = ttnn.mesh_shard(v8, v11, ttnn.MeshShardDirection.FullToShard, ttnn.MeshShardType.Replicate, [1], [-1])
  ttnn.deallocate(v8, False)
  v21 = ttnn.mesh_shard(v9, v11, ttnn.MeshShardDirection.FullToShard, ttnn.MeshShardType.Replicate, [1], [-1])
  ttnn.deallocate(v9, False)
  v22 = ttnn.mesh_shard(v10, v11, ttnn.MeshShardDirection.FullToShard, ttnn.MeshShardType.Replicate, [1], [-1])
  ttnn.deallocate(v10, False)
  v23 = ttnn.to_layout(v22, ttnn.Layout.TILE, None, memory_config=None)
  ttnn.deallocate(v22, False)
  v24 = ttnn.to_device(v23, device=v11, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v23, False)
  v25 = ttnn.reshape(v24, [32, 784], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v24, False)
  v26 = ttnn.to_device(v15, device=v11, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v15, False)
  v27 = ttnn.to_layout(v26, ttnn.Layout.TILE, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v26, False)
  v28 = ttnn.typecast(v27, ttnn.DataType.FLOAT32, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v27, False)
  v29 = ttnn.to_device(v14, device=v11, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v14, False)
  v30 = ttnn.to_layout(v29, ttnn.Layout.TILE, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v29, False)
  v31 = ttnn.reshape(v30, [1, 128], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v30, False)
  v32 = ttnn.typecast(v31, ttnn.DataType.FLOAT32, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v31, False)
  v33 = ttnn.matmul(v25, v28, transpose_a=False, transpose_b=False, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v28, False)
  ttnn.deallocate(v25, False)
  v34 = ttnn.add(v33, v32, dtype=ttnn.DataType.FLOAT32, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v33, False)
  ttnn.deallocate(v32, False)
  v35 = ttnn.reshape(v13, [1, 1], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v36 = ttnn.maximum(v34, v35, dtype=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v35, False)
  ttnn.deallocate(v34, False)
  v37 = ttnn.reshape(v36, [1, 1, 32, 128], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v36, False)
  v38 = ttnn.all_gather(input=v37, mesh_device=v11, dim=3, cluster_axis=1, num_links=1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v37, False)
  v39 = ttnn.reshape(v38, [32, 1024], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v38, False)
  v40 = ttnn.to_device(v17, device=v11, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v17, False)
  v41 = ttnn.to_layout(v40, ttnn.Layout.TILE, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v40, False)
  v42 = ttnn.typecast(v41, ttnn.DataType.FLOAT32, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v41, False)
  v43 = ttnn.to_device(v16, device=v11, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v16, False)
  v44 = ttnn.to_layout(v43, ttnn.Layout.TILE, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v43, False)
  v45 = ttnn.reshape(v44, [1, 64], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v44, False)
  v46 = ttnn.typecast(v45, ttnn.DataType.FLOAT32, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v45, False)
  v47 = ttnn.matmul(v39, v42, transpose_a=False, transpose_b=False, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v42, False)
  ttnn.deallocate(v39, False)
  v48 = ttnn.add(v47, v46, dtype=ttnn.DataType.FLOAT32, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v47, False)
  ttnn.deallocate(v46, False)
  v49 = ttnn.reshape(v13, [1, 1], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v50 = ttnn.maximum(v48, v49, dtype=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v49, False)
  ttnn.deallocate(v48, False)
  v51 = ttnn.reshape(v50, [1, 1, 32, 64], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v50, False)
  v52 = ttnn.all_gather(input=v51, mesh_device=v11, dim=3, cluster_axis=1, num_links=1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v51, False)
  v53 = ttnn.reshape(v52, [32, 512], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v52, False)
  v54 = ttnn.to_device(v19, device=v11, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v19, False)
  v55 = ttnn.to_layout(v54, ttnn.Layout.TILE, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v54, False)
  v56 = ttnn.typecast(v55, ttnn.DataType.FLOAT32, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v55, False)
  v57 = ttnn.to_device(v18, device=v11, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v18, False)
  v58 = ttnn.to_layout(v57, ttnn.Layout.TILE, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v57, False)
  v59 = ttnn.reshape(v58, [1, 32], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v58, False)
  v60 = ttnn.typecast(v59, ttnn.DataType.FLOAT32, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v59, False)
  v61 = ttnn.matmul(v53, v56, transpose_a=False, transpose_b=False, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v56, False)
  ttnn.deallocate(v53, False)
  v62 = ttnn.add(v61, v60, dtype=ttnn.DataType.FLOAT32, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v61, False)
  ttnn.deallocate(v60, False)
  v63 = ttnn.reshape(v13, [1, 1], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v13, False)
  v64 = ttnn.maximum(v62, v63, dtype=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v63, False)
  ttnn.deallocate(v62, False)
  v65 = ttnn.reshape(v64, [1, 1, 32, 32], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v64, False)
  v66 = ttnn.all_gather(input=v65, mesh_device=v11, dim=3, cluster_axis=1, num_links=1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v65, False)
  v67 = ttnn.reshape(v66, [32, 256], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v66, False)
  v68 = ttnn.to_layout(v21, ttnn.Layout.TILE, None, memory_config=None)
  ttnn.deallocate(v21, False)
  v69 = ttnn.to_device(v68, device=v11, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v68, False)
  v70 = ttnn.matmul(v67, v69, transpose_a=False, transpose_b=False, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v69, False)
  ttnn.deallocate(v67, False)
  v71 = ttnn.to_layout(v20, ttnn.Layout.TILE, None, memory_config=None)
  ttnn.deallocate(v20, False)
  v72 = ttnn.to_device(v71, device=v11, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v71, False)
  v73 = ttnn.reshape(v72, [1, 10], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v72, False)
  v74 = ttnn.add(v70, v73, dtype=ttnn.DataType.FLOAT32, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v73, False)
  ttnn.deallocate(v70, False)
  v75 = ttnn.max(v74, [1], True, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v76 = ttnn.reshape(v12, [1, 1], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v12, False)
  v77 = ttnn.maximum(v76, v75, dtype=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v76, False)
  ttnn.deallocate(v75, False)
  v78 = ttnn.neg(v77, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v77, False)
  v79 = ttnn.add(v74, v78, dtype=ttnn.DataType.FLOAT32, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v78, False)
  ttnn.deallocate(v74, False)
  v80 = ttnn.softmax(v79, 1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v79, False)
  v81 = ttnn.from_device(v80)
  ttnn.deallocate(v80, False)
  v82 = ttnn.to_layout(v81, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
  ttnn.deallocate(v81, False)
  v83 = ttnn.mesh_shard(v82, v11, ttnn.MeshShardDirection.ShardToFull, ttnn.MeshShardType.Replicate, [1], [-1])
  ttnn.deallocate(v82, False)
  v84 = [v83]
  return v84

def create_inputs_for__main(): 
  v1 = ttnn.ones(shape=ttnn.Shape([1024]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v2 = ttnn.ones(shape=ttnn.Shape([784, 1024]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v3 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v4 = ttnn.ones(shape=ttnn.Shape([1024, 512]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v5 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v6 = ttnn.ones(shape=ttnn.Shape([512, 256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v7 = ttnn.ones(shape=ttnn.Shape([10]), dtype=ttnn.DataType.FLOAT32, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v8 = ttnn.ones(shape=ttnn.Shape([256, 10]), dtype=ttnn.DataType.FLOAT32, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v9 = ttnn.ones(shape=ttnn.Shape([32, 28, 28, 1]), dtype=ttnn.DataType.FLOAT32, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v10 = [v1, v2, v3, v4, v5, v6, v7, v8, v9]
  return v10

def test_main(): 
  enable_persistent_kernel_cache()
  v1 = create_inputs_for__main()
  v2 = _main(v1)
  print(v2)
  v3 = 0
  return v3

if __name__ == '__main__':
  test_main()


