# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Which depth-decoder op does a host write during trace capture? One capture per op.

    source ~/mm3-bringup/common.sh && cd $MM3_WT && with_hw_lock timeout 600 $MM3_PY \
        models/autoports/minimaxai_minimax_music3/scripts/trace_probe_depth.py
"""
import os

import torch

import ttnn
from models.autoports.minimaxai_minimax_music3.reference import hf_llm as R
from models.autoports.minimaxai_minimax_music3.tt.depth_decoder import DEPTH_HIDDEN, LLM_BATCH, TILE, DepthDecoder

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=90_000_000)
mesh.enable_program_cache()
d = DepthDecoder.from_pretrained(mesh, R.weights_dir())
layer = d.layers[0]
seq = d.new_sequence()
rows = d.rows_to_device(torch.randn(2, DEPTH_HIDDEN))
ids = d.code_ids_to_device(torch.tensor([3, 5]), 1)
x64 = ttnn.add(seq, d.pos_embedding)
qkv = d._linear(x64, layer["wqkv"])
qkv_v = ttnn.experimental.view(qkv, (LLM_BATCH, 1, TILE, 3 * DEPTH_HIDDEN))
q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv_v, num_heads=16, num_kv_heads=16, transpose_k_heads=False)
attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True)
gate = d._linear(x64, layer["w_gate"], activation="silu")

probes = {
    "embedding": lambda: ttnn.embedding(
        ids, d.audio_embeddings, layout=ttnn.TILE_LAYOUT, dtype=d.dtype, memory_config=d.mem
    ),
    "embedding_rm": lambda: ttnn.embedding(
        ids, d.audio_embeddings, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=d.dtype, memory_config=d.mem
    ),
    "view_emb": lambda: ttnn.experimental.view(
        ttnn.embedding(ids, d.audio_embeddings, layout=ttnn.TILE_LAYOUT, dtype=d.dtype, memory_config=d.mem),
        (1, 1, TILE, DEPTH_HIDDEN),
    ),
    "linear": lambda: d._linear(rows, d.projection),
    "linear_silu": lambda: d._linear(x64, layer["w_gate"], activation="silu"),
    "scatter_matmul": lambda: ttnn.matmul(
        d.scatter[2], rows, compute_kernel_config=d.compute_config, memory_config=d.mem, dtype=d.dtype
    ),
    "add": lambda: ttnn.add(seq, x64, memory_config=d.mem),
    "copy": lambda: ttnn.copy(x64, seq),
    "rms_norm": lambda: d._rms_norm(x64, layer["ln_in"]),
    "view_qkv": lambda: ttnn.experimental.view(qkv, (LLM_BATCH, 1, TILE, 3 * DEPTH_HIDDEN)),
    "create_heads": lambda: ttnn.experimental.nlp_create_qkv_heads(
        qkv_v, num_heads=16, num_kv_heads=16, transpose_k_heads=False, memory_config=d.mem
    ),
    "sdpa": lambda: ttnn.transformer.scaled_dot_product_attention(
        q, k, v, is_causal=True, compute_kernel_config=d.sdpa_compute_config, memory_config=d.mem
    ),
    "concat_heads": lambda: ttnn.experimental.nlp_concat_heads(attn, memory_config=d.mem),
    "multiply": lambda: ttnn.multiply(gate, gate, memory_config=d.mem),
    "gather_matmul": lambda: ttnn.matmul(
        d.gather[2], x64, compute_kernel_config=d.compute_config, memory_config=d.mem, dtype=d.dtype
    ),
    "heads_all": lambda: d.heads_all(rows),
}
failed = []
for name, fn in probes.items():
    fn()  # compile
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        fn()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.synchronize_device(mesh)
        ttnn.release_trace(mesh, tid)
        print(f"PROBE {name}: ok", flush=True)
    except Exception as e:
        failed.append(name)
        print(f"PROBE {name}: FAIL {str(e).splitlines()[0][:200]}", flush=True)
        try:
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.release_trace(mesh, tid)
        except Exception as e2:
            print(f"  (cleanup failed: {str(e2).splitlines()[0][:120]})", flush=True)
print("FAILED:", failed, flush=True)
if failed:
    # A failed capture leaves the mesh close waiting forever (observed); exit hard, the next open recovers.
    os._exit(0)
ttnn.close_mesh_device(mesh)
