# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import json
import time

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.decoder_factory import Decoder as FunctionalDecoder
from models.autoports.qwen_qwen3_6_35b_a3b.tests.reference import config, layer, pcc, weights

torch.set_num_threads(8)
torch.manual_seed(42)
state = weights(0)
print("WEIGHTS", len(state), flush=True)
hf = layer(state, 0)
x = torch.randn(1, 32, 2048, dtype=torch.bfloat16) * 0.05
with torch.no_grad():
    expected = hf(x, position_embeddings=None)
print("HF_DONE", expected.float().std().item(), flush=True)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
try:
    tt = FunctionalDecoder.from_state_dict(state, hf_config=config(), layer_idx=0, mesh_device=mesh)
    print("UPLOAD_DONE", flush=True)
    cache = tt.new_recurrent_state()
    xx = ttnn.from_torch(x, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    start = time.time()
    got = ttnn.to_torch(tt.prefill_forward(xx, state=cache, plan=tt.prepare_prefill(seq_len=32))).reshape(x.shape)
    score = pcc(expected, got)
    print(
        json.dumps({"pcc": score, "max_abs": (expected - got).abs().max().item(), "seconds": time.time() - start}),
        flush=True,
    )
    assert score >= 0.995
finally:
    ttnn.close_mesh_device(mesh)
