# RESUME CONTEXT — Kimi 61-layer per-layer op2op profiling (saved 2026-06-15 ~00:30)

## THE CURRENT TASK (what to finish)
Profile **all 61 layers** per-layer on the SLOWEST device for a WARM large chunk (kv=51200), reporting
per layer: kernel_sum (compute), op2op_sum (dispatch gap), aggregated like
`kimi_perf_overnight/ops_slowest_device_3L_first.log`, **grouped in 3s**. Do it **WITH and WITHOUT** the
H2D service. Goal: verify whether DEEP layers (3..60) suffer op2op gaps, or whether the gap is only a
pipeline-fill transient in layers 0-1 (my earlier claim, which the user rightly challenged because it came
from a 3-LAYER run where layer 2 was the last layer).

## WHY THIS MATTERS (state of the investigation)
- E1 (kernel-config ring 69->133KB) is FALSIFIED/settled: perf-neutral 1864ms/chunk vs ~1878 baseline,
  KV PCC 0.965 PASS. Reverted. The per-op worker-completion sync theory is a dead end (<2% of chunk).
- My attribution from the 3-LAYER no-service profile: op2op gaps = layer0 dense 9.64ms, layer1 MoE 3.45ms,
  layer2 MoE ~0 (perfectly pipelined). I concluded gaps are a 1-time pipeline-fill transient (<1% over 61L)
  and the standalone chunk is COMPUTE-BOUND. **This 61-layer profile is to PROVE/DISPROVE that for deep layers.**
- The WITH-service profile (earlier 3Lsvc) showed op2op grows +5..25ms/LAYER — that is the SEPARATE +1.4s
  H2D-service per-op dispatch tax (already root-caused). The 61L with-service run quantifies it per-layer too.

## BLOCKERS DISCOVERED (why a single 61-layer run does NOT work)
1. **Profiler DRAM buffer overflow.** Device profiler holds ~6 layer-executions of markers
   (`PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC=48000` bytes = 12000 uint32; runtime cfg
   `config.profiler_dram_bank_size_per_risc_bytes`, set in tt_metal/jit_build/build.cpp:185). 3 layers x 2
   passes (ITERS=2) is SAFE (the 3L runs worked). 61 layers x 2 passes OVERFLOWED -> warning
   "Profiler DRAM buffers were full, markers were dropped! bufferEndIndex=12000" -> it KEEPS early-layer
   markers and DROPS deep-layer markers (exactly the layers we need). Confirmed in profile_61L_noservice.runner.log.
2. **Disk.** Box is 99% full. Was 12G free; a full 61-layer profile dump is ~9GB and the disk-guard in
   profile_61L.sh KILLED the run at 3G free (exit=137). **Currently ~3.0G free — RECLAIM DISK FIRST.**
   Safe to delete: `generated/profiler/reports/* generated/profiler/.logs/*` and old
   `kimi_perf_overnight/ops_61L_*.csv`. (A reclaim cmd was about to run when the user paused me.)

## THE APPROACH TO USE ON RERUN (windowed, user-endorsed: "3 layers at a time, prefix must run")
Profile a 3-layer window [D, D+1, D+2] at depth D by:
- Running `PREFILL_NUM_LAYERS=D+3` so the window is the LAST 3 layers AND the prefix 0..D-1 runs (pipeline
  is genuinely deep — this is what makes deep-layer data valid; a bare NUM_LAYERS=3 always profiles the
  pipeline START = layers 0-2, useless for depth).
- **Clearing the profiler buffer right before layer D** so only the window's ~3 layers accumulate (bounded
  buffer = no overflow; small dump = no disk blowup). Discard the op2op of the window's FIRST op (perturbed
  by the clear); op2op within the window is clean.
Cover depth by SAMPLING first (cheap, answers the question): windows ending near layers 2 (HAVE IT), ~17-19,
~38-40, ~58-60. If all deep windows show op2op~0 like layer 2 -> conclusion holds. Then fill remaining 3s if
the user still wants all 61. Repeat each window WITH service (add PREFILL_FORCE_BUILD_SERVICE=1).

### What still needs building for the windowed approach
- A **layer-index-gated profiler clear**. Need: (a) a ttnn/python API to read+clear the device profiler
  (grep: `read_device_profiler|ReadDeviceProfiler|DumpDeviceProfile` in ttnn/ttnn/*.py + pybind). (b) a hook
  in the forward_chunk layer loop (tt_prefill_transformer.py:385 `for i, layer in enumerate(self.layers)`)
  or the existing `on_layer_complete` callback (plumbed through layer(...)) to call the clear when i==D,
  env-gated by e.g. PREFILL_PROFILE_CLEAR_BEFORE_LAYER=D. Keep it env-gated/default-off (pure runner change).
- ALTERNATIVE (one clean run, no code hook): bump `profiler_dram_bank_size_per_risc_bytes` ~10x so 61
  layers fit, AND ensure ~10G disk free. Risk: requires rebuild (~20min) + the firmware-mismatch trap
  (only touch that one constant; verify git tree after). Disk for ~9GB dump is the hard part on this box.
- ALTERNATIVE: realtime profiler (streams via D2H, no buffer cap) but it had "end<start zone" clock-sync
  warnings (realtime_profiler_tracy_handler.cpp:177, "Skipped 2148 zones"); would need investigation.
RECOMMENDED: windowed + clear hook, depth-sampling first.

## TOOLS READY
- `kimi_perf_overnight/parse_NL.py` — generalized N-layer parser. Usage:
  `python3 parse_NL.py <ops_perf_results.csv> "<label>" <out.log>`. Emits per-layer kernel/fw/op2op +
  op2op%wall + a grouped-in-3 section, for the slowest device, WARM (last) pass. Delimits layers by
  `forward_chunk_layer_{i}_start/_end` signposts (emitted for ALL layers, tt_prefill_transformer.py:385-396).
- `kimi_perf_overnight/profile_61L.sh {noservice|service}` — single-run, disk-guarded (<3G kill). HITS the
  buffer overflow as-is; adapt into a windowed variant (NUM_LAYERS=D+3 + clear-before-D) before reuse.
- Proven profiling recipe (no-service), drops Tracy GUI push to bound disk:
  `TT_METAL_DEVICE_PROFILER=1 PREFILL_MODEL_VARIANT=kimi_k2_6 PREFILL_SP=8 PREFILL_TP=4 PREFILL_NUM_LAYERS=<N>
   PREFILL_MAX_SEQ_LEN=61440 PREFILL_CHUNK_SIZE=5120 PREFILL_NUM_USERS=1 PREFILL_IS_BALANCED=0
   PREFILL_H2D_SERVICE_ID=ds_prefill PREFILL_STANDALONE_CHUNKED=1 PREFILL_STANDALONE_CHUNKED_NCHUNKS=1
   PREFILL_STANDALONE_CHUNKED_ITERS=2 PREFILL_PROFILE_KV=51200 PREFILL_STANDALONE_CHUNKED_SLOT=0
   PREFILL_STANDALONE_CHUNKED_RECORD_ONLY=1 DEEPSEEK_PREFILL_TRACE_DIR=<golden>
   python -m tracy -r -p --disable-device-data-push-to-tracy -m models.demos.deepseek_v3_d_p.tt.runners.prefill_runner`
  (`-r` = generate ops report [REQUIRED]; with-service: add PREFILL_FORCE_BUILD_SERVICE=1).
  golden TRACE dir = /mnt/models/deepseek-prefill-cache/golden/kimi-26/kimi_longbook_56320/vllm-kimi-k26-b783c42e-56321tok

## EXISTING DATA (3-layer, layers 0-2, on disk)
- `kimi_perf_overnight/ops_slowest_device_3L_{first,last}.log` — no-service. last(kv=51200, dev30):
  layer0 9.64ms op2op, layer1 3.45ms, layer2 -0.06ms.
- `kimi_perf_overnight/ops_slowest_device_3Lsvc_{first,last}.log` — with-service.
- `kimi_perf_overnight/ops_3L_{first,last}.csv`, `ops_3Lsvc_{first,last}.csv` — raw ops CSVs (~35MB each).
- Top op2op gaps (no-svc last): RingJointSDPA 2020us, AllGather 1901us, ReduceScatter 707us, RoPE ~500us
  (all in layers 0-1).

## ENVIRONMENT / STATE
- Branch ppopovic/investigation. git tree CLEAN except untracked kimi_perf_overnight/. bh_hal_tensix.cpp
  reverted to 69KB. dev_msgs.h launch=8 (matches lib). Lib build_Release/lib/libtt_metal.so = E1 133KB ring
  (built 14:56) — HARMLESS (perf-neutral; it's a .cpp not in JIT, lib is authoritative). Rebuild to baseline
  optional, NOT required for profiling.
- Device: Blackhole 8x4 (32 chips), bh-glx host. Box currently FREE of co-users (migration/N are kernel threads).
- Committed earlier (kept): worker_config_buffer.cpp kernel_config_entry_count=32 + TT_DISPATCH_SYNC_DEBUG probe.

## CRITICAL GOTCHAS (cost hours today)
- **firmware/host mailbox mismatch** (see memory [[firmware-host-mailbox-mismatch]]): if profiler logs show
  `sync ... timed out` / `complete=0 timeout=416` + a 30-min hang during embedding (0 layer files opened),
  it's a host-lib vs JIT-firmware MAILBOX mismatch (e.g. unreverted dev_msgs.h). tt-smi -glx_reset_auto does
  NOT fix it. Fix: ensure git tree clean (`grep launch_msg_buffer_num_entries dev_msgs.h` == 8), then a
  1-layer smoke must show `[Real-time profiler] sync complete: ... 32` and reach a chunk in ~16s.
- `git checkout fileA fileB` aborts BOTH if fileB pathspec is bogus -> reverts nothing. Checkout one file; re-grep to confirm.
- `pkill -f "runners.prefill"` / `pgrep -f "...prefill..."` MATCHES YOUR OWN SHELL -> self-kill. Kill by PID
  or exclude `$$`. Always wait for FULL reap (zombie holds sysmem); rm /dev/shm/tt_prefill_layer_acks_ds_prefill + tt_d2h_*.
- Cold 61-layer build ~3-4min to first chunk on a HEALTHY device (the 30-min "hang" earlier was the mailbox bug).
- Profiling perturbs little (observer test: prof ON 24.5ms ~= OFF 24.8ms for 1L warm).

## DOCS TO READ ON RERUN
- This file. Then `kimi_perf_overnight/cpp_campaign.md` (full campaign log incl. E1 falsification +
  DEFINITIVE ATTRIBUTION sections). Memory: [[kimi-perf-overnight]], [[firmware-host-mailbox-mismatch]],
  [[kimi-prefill-env-vars]], [[kimi-chunked-prefill-work-state]].
