"""Two-layer interface test; no embeddings, generator, full-model or serving code."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

import ttnn
from models.autoports.google_gemma_4_31b_it.tests.run_optimized_decoder import pcc, weights
from models.autoports.google_gemma_4_31b_it.tt.multichip_decoder import MultichipDecoder
from models.autoports.google_gemma_4_31b_it.tt.optimized_decoder import HF_MODEL, HF_REVISION, OptimizedDecoder

p = argparse.ArgumentParser()
p.add_argument("--seq", type=int, default=33)
p.add_argument("--ring", action="store_true")
p.add_argument("--result", required=True)
a = p.parse_args()
torch.set_num_threads(16)
torch.manual_seed(123)
c = AutoConfig.from_pretrained(HF_MODEL, revision=HF_REVISION).text_config
states = {i: weights(i) for i in [0, 5]}
x = torch.load("bringup/artifacts/optimized_decoder_layer0_inputs2050.pt", weights_only=True)[:, : a.seq + 1]
rope = Gemma4TextRotaryEmbedding(c)
cs = {i: rope(x, torch.arange(a.seq + 1)[None], layer_type=c.layer_types[i]) for i in [0, 5]}
outputs = {}
layouts = {}
for n in [1, 4]:
    if n == 4:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING if a.ring else ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, n), trace_region_size=0)
    try:

        def tt(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            h = ttnn.from_torch(
                t, dtype=dtype, layout=layout, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh) if n == 4 else None
            )
            return ttnn.to_device(h, mesh)

        cls = OptimizedDecoder if n == 1 else MultichipDecoder
        layers = [cls.from_state_dict(states[i], hf_config=c, layer_idx=i, mesh_device=mesh) for i in [0, 5]]
        pages = tt((torch.randperm((a.seq + 32) // 32) + 3)[None].to(torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        caches = [m.allocate_cache(num_blocks=(a.seq + 32) // 32 + 3) for m in layers]
        pref = tt(x[:, : a.seq, None].transpose(1, 2))
        dec = tt(x[:, a.seq :].reshape(1, 1, 1, -1))
        position = tt(torch.tensor([a.seq], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        interfaces = []
        for i, m, cache in zip([0, 5], layers, caches):
            co, si = cs[i]
            pref = m.prefill_forward(
                pref,
                cos=tt(co[:, : a.seq, None].transpose(1, 2)),
                sin=tt(si[:, : a.seq, None].transpose(1, 2)),
                kv_cache=cache,
                page_table=pages,
            )
            dec = m.decode_forward(
                dec,
                cos=tt(co[:, a.seq :].reshape(1, 1, 1, -1)),
                sin=tt(si[:, a.seq :].reshape(1, 1, 1, -1)),
                kv_cache=cache,
                page_table=pages,
                current_pos=position,
            )
            interfaces.append(
                dict(
                    prefill_shape=list(pref.shape),
                    decode_shape=list(dec.shape),
                    prefill_memory=str(pref.memory_config()),
                    decode_memory=str(dec.memory_config()),
                    dtype=str(dec.dtype),
                )
            )
        outputs[n] = [ttnn.to_torch(ttnn.get_device_tensors(t)[0]) for t in [pref, dec]]
        layouts[n] = interfaces
        assert interfaces[0] == interfaces[1]
    finally:
        ttnn.close_mesh_device(mesh)
    del layers, caches, pref, dec, pages, position, m, cache
result = dict(
    seq=a.seq,
    prefill_pcc=pcc(outputs[1][0], outputs[4][0]),
    decode_pcc=pcc(outputs[1][1], outputs[4][1]),
    interfaces=layouts,
    layer_order=["sliding_attention", "full_attention"],
    boundary_conversion=False,
)
Path(a.result).write_text(json.dumps(result, indent=2) + "\n")
assert min(result["prefill_pcc"], result["decode_pcc"]) >= 0.995
print(json.dumps(result))
