# GLM-4.7-Flash GlobalCB DRAM-Prefetch Port Plan

Goal: overlap decode weight-loads with compute (attack the ~20% M=1 matmul BW efficiency,
the 39.5% Matmul device-time bucket). Port REAP's working mechanism from
`models/experimental/glm4_moe/tt/prefetcher_setup.py`.

## Why this needs live device iteration (not a background session)
The ring matmul configs (`MatmulMultiCoreReuseMultiCast1DProgramConfig`, `gather_in0=True`)
require `num_cores` to divide the weight's N_tiles/K_tiles, and the ring `hop_cores` +
receiver cores must all lie inside `global_cb.all_cores()`. A wrong config **hangs on device**
(same failure class as the blocked-agg attempt). Each of Flash's 6 MLA weights needs its own
gcd analysis + on-device validation. Est. 4-7 days of device debugging alone.

## Grid (DE-RISKED): Flash IS 8x9 (72 cores) = same as REAP
`get_glm_core_ranges` core layout transfers verbatim: senders cols 6,7 (12 active + 6 dummy),
workers cols 0-5 (54 cores, includes 24 receivers + origin), hop core (3,6).

## The mechanism (3 cooperating pieces)
1. **SubDevice split**: prefetcher sub-device (sender cores) + worker sub-device (cols 0-5).
   `create_sub_device_manager([prefetcher, worker], 0)` + `load_sub_device_manager`. DEFER
   creation until after prefill (prefill CCL needs full grid). Stall-group discipline:
   `[prefetcher, worker]` for setup/dram_prefetcher, `[worker]` for decode.
2. **GlobalCB**: `ttnn.create_global_circular_buffer(mesh, sender_receiver_mapping, 800*1088)`
   (single-buffer; WH L1 can't double-buffer). `ttnn.dram_prefetcher(tt_tensors, num_layers, global_cb)`
   streams DRAM-WIDTH_SHARDED weights in, kicked off INSIDE trace capture.
3. **Ring matmuls**: `ttnn.matmul(x_ring, W, program_config=ring_pc, global_cb=cb, sub_device_id=worker)`
   consume weights from the GlobalCB instead of re-reading DRAM. NOTE: matmul not linear
   (gather_in0 has no bias — apply bias separately).

## Flash-specific work (the port cost)
### New file: glm4_moe_lite/tt/prefetcher_setup.py
- Port `get_glm_core_ranges` (verbatim — 8x9).
- Replace hardcoded qkv/oproj ring configs with a GENERIC `make_ring_config(K, N, ...)` that
  computes num_cores = gcd-based divisor of N_tiles. Instantiate per MLA weight WORTH
  prefetching (large ones): **w_kv_b1, w_kv_b2, w_o**, and MoE **w_down** (extend later).
  Skip the small LoRA down-projs (w_q_a: K2048/N768 — ring may not pay off).
- Keep `create_global_cb`, `insert_tensor`, `get_input_tensors`, `ensure_ready`,
  `compile_prefetch`, `start_prefetch`, `stop_prefetch` largely verbatim.

### runtime_config.py
- Add `prefetch: bool` parsing `GLM4_MOE_LITE_PREFETCH` (mirror `dram_sharded_attn`).
- Carry `global_cb` / `worker_sub_device_id` / ring configs on a SEPARATE mutable context
  object (Glm4RuntimeConfig is frozen=True).

### linear_helpers.py
- Add optional `global_cb=None, sub_device_id=None, prefetch_pc=None, prefetch_in_mc=None,
  prefetch_out_mc=None` to `attn_linear` (:277) and `mlp_linear` (:58).
- Add a `gather_in0` branch: reshard act -> WIDTH_SHARDED on ring cores ->
  `ttnn.matmul(..., global_cb=, sub_device_id=)` -> apply bias outside. Mirror
  glm4_moe/tt/attention_tt.py:848-865.

### attention_decode.py
- Thread the prefetch context into the LARGE attn_linear calls (w_kv_b1/b2/w_o at :240-247,
  :455-458, :472). Re-grid SDPA / KV-update / eltwise to worker cores (sub_core_grids /
  subdevice_id) — REAP does this at attention_tt.py:957-1058.

### decoder_layer_tt.py + model_tt.py
- Instantiate the prefetcher at init behind the flag; register weights (analog of glm4_moe
  model_tt.py:687-690). `ensure_ready()` + `compile_prefetch()` BEFORE begin_trace_capture;
  `start_prefetch()` right after capture begins; `stop_prefetch()` before end_trace_capture.
  Set stall groups around these.

## Risks (all need on-device debugging)
1. **Worker sub-device shrinks every decode op from 8-wide to 6-wide** — KV-update, FlashMLA,
   MoE, norms must all be re-gridded to <=6 cols or they assert "cores outside sub-device".
2. **MLA ring-config tuning** — num_cores must divide N_tiles per weight; wrong => hang.
3. **Trace compat** — dram_prefetcher must pre-compile outside capture; preserve the
   garbage-tensor/no-sync dance; warmup shapes must match trace.
4. **L1 pressure** — single-buffer GlobalCB (800 tiles) already maxes L1; Flash keeps decode
   intermediates in L1 too — risk of OOM.

## Recommended scope for first prototype
Attention-only (w_kv_b1/b2/w_o), flag-gated, matching REAP's own Phase-2 attention-only scope.
Validate coherence + measure decode vs 63.1 ms baseline. Extend to MoE w_down as increment.

## COMPLETE per-op re-gridding spec (from REAP glm4_moe, the porting template)
Env gate `GLM4_MOE_PREFETCH=1`. Once the SubDevice mgr is active EVERY decode op must be
confined to worker cols 0-5, via op-specific args:

**Lifecycle (model_tt._capture_decode_trace):** ensure_ready() + compile_prefetch() BEFORE
begin_trace_capture (and BEFORE the compile-warmup forward, so warmup buffers match trace);
warmup runs a full decode loop WITH subdevice active; reset_sem_counters() before capture;
start_prefetch()->garbage right after begin_trace_capture; decode body; stop_prefetch(garbage)
before end_trace_capture. Both dram_prefetcher + stop dealloc are captured in-trace.

**Weights:** DRAM-WIDTH_SHARDED at load when prefetch on (create_dram_sharded_mem_config over
12 dram banks); interleaved copy dropped (prefill uses non-prefetch path). Register per layer
in order [w_qkv, w_o] via insert_tensor (Flash: pick the big MLA weights, e.g. w_o then q_b).

**Per-op restriction arg:**
- Ring matmuls (consume global_cb): ttnn.matmul (NOT linear) with global_cb=, sub_device_id=,
  program_config=ring_pc (grid=bounding box of receiver cores, gather_in0=True, hop=(3,6)),
  memory_config=WIDTH_SHARDED on ring cores; bias applied separately after.
- SDPA: program_config=SDPAProgramConfig(sub_core_grids=<54-core worker CRS>).
- RMSNorm (_sharded_rms_norm): pass worker_core_range=<worker CRS> (num_cores derived from it).
  *** Flash already has _maybe_sharded_norm (committed 29c428b) — add a worker_core_range param. ***
- MoE eltwise (scale/add/combine): sub_core_grids=worker_scg (CoreRange((0,0),(5,8))).
- all_reduce/all_gather (TP & EP reduce): subdevice_id=sub_device_id on every CCL call.
- sparse_matmul (experts): NO explicit arg — relies on worker block including origin (0,0) +
  sender cols unassigned. GOTCHA: verify Flash experts don't spill to cols 6-7.
- KV cache update: no explicit arg; inputs already worker-sharded via RoPE worker shard cfgs.
- head create/concat/batch-slice/user-select: sub_core_grids or core_grid=CoreGrid(y,x<=6) +
  sub_device_id.
- Residual adds, in-trace embedding slice, argmax multiply-tighten: sub_core_grids=worker_scg.
- LM-head all_gather: subdevice_id. lm_head linear/argmax: no explicit (rely on subdevice).
- Full-vocab embedding + to_layout/permute: FORBIDDEN under subdevice — do CPU lookup+upload.

**Plumbing:** read 8 values off the setup object (global_cb, worker_sub_device_id, qkv/oproj
program+mem configs) + rebuild worker_scg=CoreRange((0,0),(5,8)) inline at each level; thread
as kwargs through decoder_layer.forward -> attention.forward + _moe_forward.

**Gotchas:** (1) stall-group both-subdevices for setup/dram_prefetcher, worker-only for decode;
(2) NEVER synchronize_device after dram_prefetcher (hangs, no consumer); (3) warmup with
subdevice active; (4) defer subdevice creation to ensure_ready (prefill needs full grid);
(5) single-buffer GlobalCB; (6) worker block must include origin (0,0); (7) ring num_cores |
N_tiles, ring cores subset of global_cb.all_cores(); (8) subdevice_id into every CCL;
(9) reset_sem_counters before capture; (10) no full-grid ops under subdevice.

**Flash structural delta from REAP:** Flash attention is attention_decode.py (6 MLA weights:
w_q_a/q_b/kv_a/kv_b1/kv_b2/w_o) NOT a fused QKV in attention_tt.py — so ring configs + weight
registration differ (prefetch the big ones: w_o K5120/N2048, q_b K768/N5120; skip tiny kv_b1/b2).

## Expected payoff
The 39.5% Matmul bucket runs at ~20% M=1 BW efficiency; prefetch overlaps the weight load
behind compute. Realistic decode win: several ms (the single largest remaining lever), but
only realized after the full re-gridding + ring-tuning is debugged on device.
