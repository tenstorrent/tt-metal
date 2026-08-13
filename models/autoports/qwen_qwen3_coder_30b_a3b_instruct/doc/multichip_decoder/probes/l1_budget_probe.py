# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Re-derive _DECODE_EXPERT_L1_BUDGET_BYTES for the multichip expert path.

Stage 02 set the single-chip threshold to 40 MB and its own comment calls it
"asserted, not measured": it sits between batch 1's 29.4 MB and batch 2's
58.8 MB, which was enough to separate the only two cases that existed there.
Under EP the intermediates are a quarter of the size (batch x 7.34 MB), so
inheriting 40 MB would silently start putting batch 5 in L1 -- a behaviour
change nobody chose, on a mesh where L1 also has to hold the fabric's and the
CCL's persistent buffers.

This measures the two things the constant is supposed to encode:

  * where L1 stops being *possible* (the allocator refuses), and
  * where it stops being *worth it* (L1 and DRAM measure the same).

Run: python l1_budget_probe.py     (no watcher, no profiler)
Prints ``P|`` lines only.
"""
import statistics
import sys
import time

import torch

import ttnn

sys.path.insert(0, "/home/raahem/tt-metal")
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tests.reference import layer_state_dict, load_config
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt import functional_decoder as F
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt import multichip_decoder as MC
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.weight_mapping import convert_layer_weights

hf = load_config()
tw = convert_layer_weights(layer_state_dict(0), hf)

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=90_000_000, l1_small_size=32768)
try:
    cfg = MC.MeshDecoderConfig.from_hf(hf)
    ctx = MC.mesh_context(mesh)
    weights = MC.upload_multichip_weights(tw, mesh, cfg)
    cos, sin = F.build_rope_cache(hf, 1024, mesh)
    sparsity = MC.build_local_sparsity(mesh, cfg.local_moe)

    def rep(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    for batch in (1, 2, 4, 8, 16, 32):
        nbytes = batch * cfg.local_moe.num_experts * 32 * (2 * 768 + 2048) * 2
        kv = MC.create_mesh_kv_cache(mesh, cfg, batch, 128, block_size=32)
        torch.manual_seed(0)
        x = rep(torch.randn(1, 1, batch, hf.hidden_size) * 0.02)
        pos = ttnn.from_torch(
            torch.full((batch,), 32, dtype=torch.int32),
            dtype=ttnn.int32,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        results = {}
        for tag, budget in (("L1", 1 << 40), ("DRAM", 0)):
            MC._DECODE_EXPERT_L1_BUDGET_BYTES = budget
            try:
                for _ in range(3):
                    out = MC.decoder_layer_decode_multichip(x, weights, cfg, ctx, cos, sin, kv, pos, 32)
                    ttnn.synchronize_device(mesh)
                    ttnn.deallocate(out)
                samples = []
                for _ in range(10):
                    t0 = time.perf_counter()
                    out = MC.decoder_layer_decode_multichip(x, weights, cfg, ctx, cos, sin, kv, pos, 32)
                    ttnn.synchronize_device(mesh)
                    samples.append((time.perf_counter() - t0) * 1e3)
                    ttnn.deallocate(out)
                results[tag] = f"{statistics.median(samples):.4f} ms"
            except Exception as exc:  # allocator refusal is a result, not a crash
                results[tag] = f"FAILED {str(exc)[:110]}"
        print(
            f"P|batch={batch:>2} intermediates={nbytes/1e6:7.2f} MB  L1={results['L1']}  DRAM={results['DRAM']}",
            flush=True,
        )
        ttnn.deallocate(x)
finally:
    MC._DECODE_EXPERT_L1_BUDGET_BYTES = 8 * 1024 * 1024
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
print("P|done")
