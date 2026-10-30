# SPDX-License-Identifier: Apache-2.0
"""Reduced real four-layer generator runtime check; no quality/performance claim."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--context',type=int,default=8192);p.add_argument('--native-end',action='store_true');a=p.parse_args()
torch.set_num_threads(8)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
out=Path(a.output);record=dict(scope='Reduced real layers0/1/2/3, full head and request owner; runtime/trace check only',status='running',config=a.config,context=a.context)
try:
    gen=build_generator('models/autoports/qwen_qwen3_8_27b',mesh,layer_indices=[0,1,2,3],context_length=a.context,precision_config=a.config)
    record['source_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path('models/autoports/qwen_qwen3_8_27b/tt').glob('*.py')}
    record['policies']=[layer.policy for layer in gen.model.layers]
    traces=[str(gen.model_trace),str(gen.sample_trace),str(gen.buffered_sample_trace)];captures=gen.counters['captures']
    record['cases']=[]
    for length in [1,33,4095,4096,4097]:
        prompt=[100+(i%7) for i in range(length)]
        tokens=gen.generate(prompt,8,stop_on_eos=False)
        caches=[value for state in gen.kv_cache for name,value in vars(state).items() if name in ('key','value')]
        addresses=[[part.buffer_address() for part in ttnn.get_device_tensors(value)] for value in caches]
        gen.reset()
        ttnn.synchronize_device(mesh)
        for value,expected in zip(caches,addresses):
            assert [part.buffer_address() for part in ttnn.get_device_tensors(value)]==expected
            for part in ttnn.get_device_tensors(value):
                assert torch.count_nonzero(ttnn.to_torch(part))==0
        repeat=gen.generate(prompt,8,stop_on_eos=False)
        assert tokens==repeat and len(tokens)==8
        assert traces==[str(gen.model_trace),str(gen.sample_trace),str(gen.buffered_sample_trace)]
        assert captures==gen.counters['captures']
        record['cases'].append(dict(length=length,exact_repeat=True,all_rank_cache_reset_exact_zero=True,cache_addresses_unchanged=True,trace_ids=traces,captures=captures))
    if a.native_end:
        assert a.context==262144
        prompt=[100+(i%7) for i in range(262141)]
        tokens=gen.generate(prompt,3,stop_on_eos=False)
        repeat=gen.generate(prompt,3,stop_on_eos=False)
        assert tokens==repeat and len(tokens)==3
        assert traces==[str(gen.model_trace),str(gen.sample_trace),str(gen.buffered_sample_trace)]
        assert captures==gen.counters['captures']
        logits=gen.decode_forward(torch.tensor([[tokens[-1]]]),torch.tensor([262143]),
            page_table=gen.page_table,kv_cache=gen.kv_cache,return_logits=True,return_device=True)
        assert torch.isfinite(gen.host_logits(logits)).all()
        final_positions=ttnn.to_torch(ttnn.get_device_tensors(gen.positions)[0]).long().tolist()
        assert final_positions==[262144]
        assert traces==[str(gen.model_trace),str(gen.sample_trace),str(gen.buffered_sample_trace)]
        assert captures==gen.counters['captures']
        record['native_end']=dict(input_tokens=262141,output_tokens=3,final_decode_input_position=262143,
            final_next_positions=final_positions,finite=True,
            exact_repeat=True,trace_ids_unchanged=True,captures_unchanged=True)
    record['status']='pass'
except BaseException as error:
    record.update(status='runtime_failure',error_type=type(error).__name__,error=str(error)[:1000]);raise
finally:
    out.write_text(json.dumps(record,indent=2)+'\n')
    ttnn.close_mesh_device(mesh)
