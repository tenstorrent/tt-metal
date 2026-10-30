# SPDX-License-Identifier: Apache-2.0
"""Tracy profile of one real layer per kind plus real full terminal and sampler.

Never an all-layer profile or a full-model performance/roofline measurement.
"""
import argparse,json,hashlib
from pathlib import Path
import torch
import ttnn
from tracy import signpost
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator
from models.autoports.qwen_qwen3_8_27b.tests.full_model_evidence import runtime_build
p=argparse.ArgumentParser();p.add_argument('--buffered',action='store_true');p.add_argument('--batch',type=int,default=1);p.add_argument('--output',required=True);a=p.parse_args()
torch.set_num_threads(8)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
context=262144 if a.batch==1 else (33024 if a.batch==32 else 32768)
try:
    gen=build_generator('models/autoports/qwen_qwen3_8_27b',mesh,checkpoint='.cache/qwen38-probe',
        layer_indices=[0,3],batch_size=a.batch,context_length=context,prepare=False)
    gen.prepare_decode(prefill_lengths=(128,4096))
    prompt=torch.full((a.batch,4096),100,dtype=torch.int64)
    # Only two warm decode replays. Profiling every unmeasured token formerly
    # produced tens of GB of raw traces before the one measured decode window.
    gen.generate(prompt.tolist(),3,stop_on_eos=False)
    gen.reset()
    ttnn.synchronize_device(mesh);ttnn.ReadDeviceProfiler(mesh)
    signpost('PERF_PREFILL')
    logits=gen.prefill_forward(prompt,page_table=gen.page_table,kv_cache=gen.kv_cache,prompt_lens=[4096]*a.batch,return_device=True)
    ttnn.copy(ttnn.reshape(logits,[1,1,a.batch,248320//4]),gen.logits)
    ttnn.execute_trace(mesh,gen.sample_trace,cq_id=0,blocking=False)
    ttnn.synchronize_device(mesh);signpost('PERF_PREFILL_END');ttnn.ReadDeviceProfiler(mesh)
    # Construct real KV/recurrent state through4096+127 logical tokens using
    # the public continuation contract. This synthetic token100 history is NOT
    # a127-token autoregressive continuation, and must never be labeled as one.
    continuation=torch.full((a.batch,127),100,dtype=torch.int64)
    continuation_logits=gen.prefill_forward(continuation,page_table=gen.page_table,kv_cache=gen.kv_cache,
        prompt_lens=[127]*a.batch,start_position=4096,return_device=True)
    del continuation_logits
    gen.refresh(torch.full((1,1,1,a.batch),100,dtype=torch.int32),gen.tokens)
    gen.refresh(torch.full((a.batch,),4223,dtype=torch.int32),gen.positions)
    gen.refresh(torch.full((a.batch,1),4223,dtype=torch.int32),gen.rope_indices)
    ttnn.synchronize_device(mesh);ttnn.ReadDeviceProfiler(mesh)
    if a.buffered:
        gen.refresh(torch.zeros(1,dtype=torch.int32),gen.output_index)
        ttnn.synchronize_device(mesh)
    signpost('PERF_DECODE')
    ttnn.execute_trace(mesh,gen.model_trace,cq_id=0,blocking=False)
    ttnn.execute_trace(mesh,gen.buffered_sample_trace if a.buffered else gen.sample_trace,cq_id=0,blocking=False)
    ttnn.synchronize_device(mesh);signpost('PERF_DECODE_END');ttnn.ReadDeviceProfiler(mesh)
    positions_after=ttnn.to_torch(ttnn.get_device_tensors(gen.positions)[0]).reshape(-1).tolist()
    assert positions_after==[4224]*a.batch,positions_after
    Path(a.output).write_text(json.dumps(dict(scope='reduced layers0/3 with real embedding/norm/head/sampling; not fullmodel',
        batch=a.batch,buffered_output=a.buffered,context=context,prompt=4096,decode_position=4223,
        history='Synthetic4096-token100 prompt plus127-token100 continuation prefill; measured decode consumes token100 at4223',
        continuation_logical_tokens=127,continuation_physical_bucket=128,continuation_start_position=4096,
        unmeasured_generation_outputs=3,unmeasured_warm_decode_replays=2,measured_decode_replays=1,
        prepared_prefill_lengths=[128,4096],positions_after_measured_decode=positions_after,
        not_autoregressive_history=True,trace_ids=[str(gen.model_trace),str(gen.buffered_sample_trace if a.buffered else gen.sample_trace)],
        runtime_build=runtime_build(),source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
            [Path(__file__),*Path('models/autoports/qwen_qwen3_8_27b/tt').glob('*.py')]}),indent=2)+'\n')
finally:ttnn.close_mesh_device(mesh)
