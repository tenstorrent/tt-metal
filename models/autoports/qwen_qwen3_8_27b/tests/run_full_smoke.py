# SPDX-License-Identifier: Apache-2.0
"""All-layer AIME chat smoke; explicitly prepared prompt, not final capability gate."""
import json
import time
from pathlib import Path
import torch
import ttnn
from transformers import AutoTokenizer
from readiness_check.generate import _load_aime24_prompt,DEFAULT_AIME24_PROMPTS_FILE
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator

torch.set_num_threads(8)
tok=AutoTokenizer.from_pretrained('.cache/qwen38-full')
text=tok.apply_chat_template([{'role':'user','content':_load_aime24_prompt(DEFAULT_AIME24_PROMPTS_FILE,0)}],
                            tokenize=False,add_generation_prompt=True)
prompt=tok.encode(text,add_special_tokens=False)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
try:
    start=time.monotonic()
    gen=build_generator('models/autoports/qwen_qwen3_8_27b',mesh,context_length=262144,prepare=False)
    print('FULL_BUILD_SECONDS',time.monotonic()-start,flush=True)
    mem=ttnn.get_memory_view(mesh,ttnn.BufferType.DRAM)
    allocated=mem.total_bytes_allocated_per_bank*mem.num_banks
    print('FULL_ALLOCATED_BYTES_PER_DEVICE',allocated,flush=True)
    gen.prepare_decode(prefill_lengths=(len(prompt),))
    print('FULL_PREPARED',flush=True)
    start=time.monotonic()
    tokens=gen.generate(prompt,128)
    wall=time.monotonic()-start
    completion=tok.decode(tokens,skip_special_tokens=False)
    print(completion,flush=True)
    result=dict(scope='all64layers AIME24 chat smoke; explicit prompt signature preparation',
                input_tokens=len(prompt),output_tokens=len(tokens),tokens=tokens,completion=completion,
                wall_seconds=wall,allocated_bytes_per_device=allocated,counters=gen.counters,
                tracking=True,skip_program_cache=False)
    Path('models/autoports/qwen_qwen3_8_27b/doc/full_model/full_smoke.json').write_text(json.dumps(result,indent=2)+'\n')
finally:
    ttnn.close_mesh_device(mesh)
