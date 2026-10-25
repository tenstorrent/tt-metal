# SPDX-License-Identifier: Apache-2.0
"""Validate allocation accounting on the reduced real stack before full audit."""
import json
from pathlib import Path
import torch
import ttnn
from .full_model_memory import allocation_peak,release_trace_bytes
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator

torch.set_num_threads(8)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
try:
    gen=build_generator('models/autoports/qwen_qwen3_8_27b',mesh,checkpoint='.cache/qwen38-probe',
        layer_indices=[0,3],batch_size=8,context_length=8192,prepare=False)
    gen.prepare_decode(prefill_lengths=(4096,))
    out=Path('models/autoports/qwen_qwen3_8_27b/doc/full_model/memory_probe.json')
    result=allocation_peak(gen,[[100]*4096 for _ in range(8)],out.with_suffix('.allocations.json.gz'))
    result['scope']='Reduced layers0/3 allocator parser validation only; not full-model memory capacity'
    result['trace_allocation']=release_trace_bytes(gen)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
finally:
    ttnn.close_mesh_device(mesh)
