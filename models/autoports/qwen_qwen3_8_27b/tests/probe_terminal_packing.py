# SPDX-License-Identifier: Apache-2.0
"""Compare terminal row packing on real reduced-stack final hidden states."""
import argparse,json,time,os
from pathlib import Path
import torch
import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator
p=argparse.ArgumentParser();p.add_argument('--batch',type=int,default=32);a=p.parse_args()
torch.set_num_threads(8)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
try:
    gen=build_generator('models/autoports/qwen_qwen3_8_27b',mesh,checkpoint='.cache/qwen38-probe',
        layer_indices=[0,3],batch_size=a.batch,context_length=512,prepare=False)
    terminal=gen.model.terminal;observed=[]
    def capture_hidden(hidden,**kw):
        observed.append(hidden)
        return terminal(hidden,**kw)
    gen.model.terminal=capture_hidden
    gen.prefill_forward(torch.full((a.batch,33),100,dtype=torch.int64),page_table=gen.page_table,
        kv_cache=gen.kv_cache,prompt_lens=[33]*a.batch,return_device=True)
    gen.model.terminal=terminal
    hidden=observed[-1]
    results={};outputs={}
    for packed in (False,True):
        outputs[packed]=terminal(hidden,pack_decode_rows=packed)
    ttnn.synchronize_device(mesh)
    host={k:gen.host_logits(v).reshape(a.batch,-1) for k,v in outputs.items()}
    corr=float(torch.corrcoef(torch.stack([v.flatten() for v in host.values()]))[0,1])
    assert corr>=.999,corr
    del outputs
    outputs={packed:gen.model.put(torch.zeros(1,1,a.batch,248320//4)) for packed in (False,True)}
    def run(packed):
        logits=terminal(hidden,pack_decode_rows=packed)
        ttnn.copy(ttnn.reshape(logits,[1,1,a.batch,248320//4]),outputs[packed])
    for packed in (False,True):run(packed)
    ttnn.synchronize_device(mesh)
    traces={}
    for packed in (False,True):
        trace=ttnn.begin_trace_capture(mesh,cq_id=0)
        run(packed)
        ttnn.end_trace_capture(mesh,trace,cq_id=0);traces[packed]=trace
    for packed,trace in traces.items():
        ttnn.execute_trace(mesh,trace,cq_id=0,blocking=True)
        begin=time.monotonic()
        for _ in range(100):ttnn.execute_trace(mesh,trace,cq_id=0,blocking=False)
        ttnn.synchronize_device(mesh)
        elapsed_us=(time.monotonic()-begin)*10000
        replayed=gen.host_logits(outputs[packed]).reshape(a.batch,-1)
        assert torch.equal(replayed,host[packed]),('terminal replay differs from eager',packed)
        results[str(packed)]={'host_wall_us_per_replay':elapsed_us,'iterations':100,
            'output_shape':list(outputs[packed].shape),'trace_output_exact':True}
    result=dict(scope='Reduced real activation terminal-only comparison; not full-model performance',batch=a.batch,
        logits_pcc=corr,exact=torch.equal(host[False],host[True]),results=results,
        source='terminal trace host-wall timings; no device-time claim',tracking=os.environ.get('TT_METAL_TRACE_ALLOC_TRACKING')=='1')
    Path(f'models/autoports/qwen_qwen3_8_27b/doc/full_model/terminal_packing_b{a.batch}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
finally:ttnn.close_mesh_device(mesh)
