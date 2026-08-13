# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""One clean multichip layer pass, for the op-level profiler.

    python -m tracy -v -r -p --sync-host-device -o /tmp/prof_mc_pf profile_layer.py prefill
    python -m tracy -v -r -p --sync-host-device -o /tmp/prof_mc_dec profile_layer.py decode
    tt-perf-report /tmp/prof_mc_pf/reports/*/ops_perf_results_*.csv

A third mode, ``decode-agsum``, profiles the decode layer with the all-reduce
spelled the way ``mesh_plan.md`` §5 chose it -- all-gather of partials plus a
local ``ttnn.sum``. It exists so the ``FillPad`` finding in ``work_log.md`` §7
stays reproducible after the shipped path stopped producing those rows.

A standalone script rather than a pytest selection on purpose. The correctness
tests upload the replicated single-chip baseline alongside the sharded path and
run both, so a capture of one of them interleaves two implementations in one CSV
and every row has to be attributed by hand. This runs the multichip layer and
nothing else: the capture is the layer.

The warm-up pass before the measured one is what keeps kernel compilation out of
the profile.

Which rows the published numbers come from
------------------------------------------
**Every mode runs the layer twice, and the published window is always the LAST
iteration** -- the one with a fully warm program cache. Read the CSV with the
rows of one ``DEVICE ID`` sorted by ``HOST START TS``; ``GLOBAL CALL COUNT`` is
not a per-device ordering.

``decode`` emits 198 rows per die:

    rows   0- 67   the prefill priming pass (68 ops)
    rows  68- 69   the decode position tensor / tilize setup (2 ops)
    rows  70-133   decode iteration 1  (64 ops, 414.521 us on device 0)
    rows 134-197   decode iteration 2  (64 ops, 414.661 us on device 0)  <-- PUBLISHED

``prefill`` emits 536 rows per die on the same pattern:

    rows   0- 13   host upload / tilize of the replicated input
    rows  14-274   prefill pass 1     (261 ops, 9211.67 us on device 0)
    rows 275-535   prefill pass 2     (261 ops, 9221.63 us on device 0)  <-- PUBLISHED

Two invariants make the boundary checkable rather than eyeballed, and a review
of this stage caught a window that violated both:

* **A decode layer contains exactly two ``ReduceScatterMinimalAsync`` and two
  ``AllGatherAsync``.** The layer has two RS+AG all-reduces, so any op-family
  tally showing 3 AG against 2 RS has straddled a layer boundary.
* **A layer starts at the ``LayerNorm`` that follows the previous layer's
  residual ``BinaryNg``, and ends at its own residual ``BinaryNg``.** Rows 131,
  132, 133 (RS, AG, residual add) close iteration 1; row 134 (``LayerNorm``)
  opens iteration 2.

``probes/summarize_perf.py`` does not read this CSV -- the op-level tables in
``README.md`` are transcribed by hand from these row ranges, so the ranges are
named in the README next to every figure derived from them.
"""
import sys

import torch

import ttnn

sys.path.insert(0, "/home/raahem/tt-metal")
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tests.reference import layer_state_dict, load_config
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt import functional_decoder as F
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt import multichip_decoder as MC
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.weight_mapping import convert_layer_weights

MODE = sys.argv[1] if len(sys.argv) > 1 else "prefill"
SEQ = 512
PROMPT = 32

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

    if MODE == "decode-agsum":
        # The plan's spelling: one all-gather of the partials, then a local sum.
        def ag_sum(x, c):
            g = ttnn.experimental.all_gather_async(
                x,
                dim=0,
                multi_device_global_semaphore=c.ccl.get_and_cycle_ag_semaphore_handles(),
                barrier_semaphore=c.ccl.get_and_cycle_barrier_semaphore_handle(),
                num_links=c.num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=c.topology,
            )
            r = ttnn.sum(g, dim=0)
            ttnn.deallocate(g)
            return ttnn.unsqueeze_to_4D(r)

        MC.all_reduce_decode = ag_sum

    torch.manual_seed(0)
    if MODE == "prefill":
        x = rep(torch.randn(1, 1, SEQ, hf.hidden_size) * 0.02)
        for _ in range(2):  # warm up the program cache, then the profiled pass
            out = MC.decoder_layer_prefill_multichip(x, weights, cfg, ctx, cos, sin, sparsity)
            ttnn.synchronize_device(mesh)
            ttnn.deallocate(out)
    else:
        kv = MC.create_mesh_kv_cache(mesh, cfg, 1, 1024, block_size=32)
        full = torch.randn(1, 1, PROMPT + 1, hf.hidden_size) * 0.02
        MC.decoder_layer_prefill_multichip(rep(full[:, :, :PROMPT, :]), weights, cfg, ctx, cos, sin, sparsity, kv)
        ttnn.synchronize_device(mesh)
        tok = rep(full[:, :, PROMPT : PROMPT + 1, :])
        pos = ttnn.from_torch(
            torch.tensor([PROMPT], dtype=torch.int32),
            dtype=ttnn.int32,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        for _ in range(2):
            out = MC.decoder_layer_decode_multichip(tok, weights, cfg, ctx, cos, sin, kv, pos, PROMPT)
            ttnn.synchronize_device(mesh)
            ttnn.deallocate(out)
finally:
    MC.all_reduce_decode = MC.all_reduce
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
print(f"P|profiled {MODE} done")
