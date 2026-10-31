"""Measure the prefill tracing advice on the final real TP4 layer path."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

import ttnn
from models.autoports.google_gemma_4_31b_it.tests.optimized_trace import capture
from models.autoports.google_gemma_4_31b_it.tests.run_optimized_decoder import pcc, weights
from models.autoports.google_gemma_4_31b_it.tt.multichip_decoder import MultichipDecoder
from models.autoports.google_gemma_4_31b_it.tt.optimized_decoder import HF_MODEL, HF_REVISION

p = argparse.ArgumentParser()
p.add_argument("--layer", required=True, type=int)
p.add_argument("--result", required=True)
a = p.parse_args()
torch.set_num_threads(16)
c = AutoConfig.from_pretrained(HF_MODEL, revision=HF_REVISION).text_config
x = torch.load(f"bringup/artifacts/optimized_decoder_layer{a.layer}_inputs262144.pt", weights_only=True, mmap=True)[
    :, :4096
]
cos, sin = Gemma4TextRotaryEmbedding(c)(x, torch.arange(4096)[None], layer_type=c.layer_types[a.layer])
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
tid = None
try:

    def upload(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        host = ttnn.from_torch(
            value.contiguous(), dtype=dtype, layout=layout, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
        )
        return ttnn.to_device(host, mesh)

    model = MultichipDecoder.from_state_dict(
        weights(a.layer),
        hf_config=c,
        layer_idx=a.layer,
        mesh_device=mesh,
        sliding_cache_tokens=2048 if a.layer == 0 else None,
    )
    cache = model.allocate_cache(num_blocks=64 if a.layer == 0 else 128)
    page = upload(
        (torch.arange(128) % (64 if a.layer == 0 else 128)).reshape(1, -1).to(torch.int32),
        ttnn.int32,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    inp = upload(x.unsqueeze(1))
    kw = dict(cos=upload(cos.unsqueeze(1)), sin=upload(sin.unsqueeze(1)), kv_cache=cache, page_table=page)
    out = model.prefill_forward(inp, **kw)
    reference = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)]
    del out
    eager = []
    for _ in range(4):
        start = time.perf_counter()
        out = model.prefill_forward(inp, **kw)
        ttnn.synchronize_device(mesh)
        eager.append((time.perf_counter() - start) * 1e6)
        del out
    start = time.perf_counter()
    tid, out = capture(mesh, lambda: model.prefill_forward(inp, **kw))
    ttnn.synchronize_device(mesh)
    capture_us = (time.perf_counter() - start) * 1e6
    traced, correctness = [], []
    for _ in range(8):
        start = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        traced.append((time.perf_counter() - start) * 1e6)
        correctness.append(min(pcc(ttnn.to_torch(t), r) for t, r in zip(ttnn.get_device_tensors(out), reference)))
    assert min(correctness) >= 0.99999, correctness
    source = Path("models/autoports/google_gemma_4_31b_it/tt/multichip_decoder.py")
    Path(a.result).write_text(
        json.dumps(
            dict(
                layer_type=c.layer_types[a.layer],
                batch=1,
                seq=4096,
                eager_host_us=eager,
                traced_host_us=traced,
                capture_host_us=capture_us,
                pcc=correctness,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                timing_basis="Host elapsed including sync; not device latency or headline roofline.",
            ),
            indent=2,
        )
        + "\n"
    )
finally:
    if tid is not None:
        ttnn.release_trace(mesh, tid)
    ttnn.close_mesh_device(mesh)
