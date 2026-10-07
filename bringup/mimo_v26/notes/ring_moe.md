# Ring decode MoE (moe agent): how to use it, what it costs, what was verified

Select with `MIMO_MOE_DECODE=ring` (default `unified`); code `tt/moe_ring.py`, hook in `tt/layer.py moe_decode` (live-mask path). `MIMO_ROUTER=fast` is the default for the ring path (the unified path keeps its own default).
* Pipeline: router_select -> dead-slot routing (weight 0, balanced) -> all_to_all_dispatch_metadata -> moe_compute (ring matmul, fused combine; bfp8 weights, HiFi3 + fp32 accumulate) -> tilize + fast_reduce
  (score-weighted sum over k) -> reduce_scatter over the mesh columns -> all_gather (full hidden, replicated). Input `x [1,1,T,4096]` bf16 tile per device (T = tokens of a mesh row, 1..32; T = 8U for the verify path), output same shape.
* Library: `artifacts/moe_C_<commit>/lib` (moe_compute env knobs patched: MOE_COMPUTE_BFP8_WEIGHTS=1, MOE_COMPUTE_FP32_ACC=1, MOE_COMPUTE_FIDELITY=HiFi3; moe_ring.py sets these defaults at import). LoFi/HiFi2 are NOT accurate (alpha/rel L2).
* Weights: a SECOND copy, ring layout, bfp8, ~214 MB per chip per layer (~53 MB/bank): +9.4 GiB at 47 layers next to the unified prefill copy. Built from the same bf16-dequantised experts, cached on NFS
  `/mnt/tt-data/ssinghal/mimo-wcache/ring_L<L>/` (build once: `pytest -s tests/build_ring_cache.py`, ~60 s/layer cold, 1-2 s warm). Per-(mesh, T) scratch buffers and semaphores are shared by all layers (L1 ~13 KB/bank per T).
* Measured (4x8, real layers 1/5/16, alternating traced, ms/layer unified -> ring): T=1 0.842 -> 0.416; 2 1.083 -> 0.615; 4 1.241 -> 0.789; 8 1.450 -> 0.964; 16 1.608 -> 1.153; 32 1.693 -> 1.188.
  Numerics vs the gated unified path: alpha 0.994-1.0003, rel L2 0.0046-0.0064, PCC >= 0.99999. Gates: 48-layer deep trace 1000 replays no hang at T=1/8/32; bitwise glitch stress 90000 layer-calls, 0 glitches.
* Not done / open: full-model GSM8K and paired step times (needs the full cache build), DRAM headroom cells (B,ISL), B>=64.
