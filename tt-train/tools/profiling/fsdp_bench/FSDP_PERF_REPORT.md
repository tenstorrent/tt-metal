# FSDP (ttml.fsdp) performance study — Blackhole galaxy, 2026-09-16

Host: bh-glx-120 (32x Blackhole, tt-metal @ 0aeee91ba2e, Release build, Tracy on).
Kit and raw logs: `tt-train/tools/profiling/fsdp_bench/` (see README.md there).

> **Provenance.** This is the lab notebook of the exploration branch (`imichalak/fsdp-perf-fixes`,
> 2026-09-16/17) that led to the FSDP changes shipped on this branch. Knob and file names below are
> the exploration branch's (`fsdp_ccl_subdevice_columns`, `TTML_FSDP_SLOTS`, `fsdp_min_shard_bytes`,
> ...); the shipped design -- `fsdp_overlap_collectives`, `fsdp_ccl_subdevice`, `fsdp_keep_gathered_gib`,
> `ttml.fsdp.enable_overlap`, `SlotSchedule` -- is documented in `docs/FSDP.md`. Ideas that were
> measured and did not pay (tiny-parameter skip, single-queue trickling, all-gathering accumulated
> gradients) were left out of the shipped code; the numbers here are the evidence for both decisions.

## 0. Executive summary

1. **FSDP is not slower than DDP on 4-8 chips; it is 1-7 % faster** (TinyLlama 1.1B on
   4/8 chips, Llama-8B TP4 on 8x4). The extra all-gather/reduce-scatter traffic (~100 ms
   on 8 chips, 6 % of the step) is paid back by the removed all-reduce and the 6x cheaper
   sharded AdamW step. The device is 96 % busy either way; SDPA backward and matmuls own
   the step.
2. **The logs say otherwise because of a bug**: `ThroughputCallback` counts tokens over
   the `"dp"` axis only, so every FSDP run prints TPS/TFLOPS/MFU divided by the FSDP axis
   size (8 chips: 3.5 % printed vs 27.7 % real; 32 chips: 0.7 % vs 22 %). Fixing the
   token accounting (count `fsdp` like `dp`) is a one-line change and probably explains
   the "FSDP is slow" impression.
3. **FSDP does degrade with mesh size, and the root cause is a shard-dim choice, not
   bandwidth**: at 32 chips the step is 2072 ms vs DDP's 1794 ms. The host-only trace shows
   1.42 s of the 2.07 s step spent inside the FSDP Python hooks, 90 % of it on the three
   parameter groups whose per-rank shard is not a multiple of 32 rows (w1/w3 → 176 rows,
   kv → 16, embedding/LM head → 1000). `fully_shard` only checks `rows % N == 0`; ttnn then
   routes those all-gathers / reduce-scatters through its composite split/pad/concat
   fallback, which is ~10x slower on the device and ~10-20x more expensive on the host
   (12.5 ms host per reduce-scatter). Choosing the tile-aligned dim (dim 3 here) avoids it:
   **measured 2072 → 1728 ms per step (-17 %), which makes FSDP32 3.7 % faster than DDP32.**
   Second-order terms: per-op latency grows with the ring (AG 2 MB: 60 µs at N=8 → 181 µs
   at N=32), host dispatch grows with device count (RS 135 → 490 µs), FSDP issues 716
   collective launches per step (180 of them for 4 KB RMSNorm gammas), and nothing overlaps
   with compute.
4. Biggest levers, in order: fix the metric; make the auto shard-dim tile-aware (or pad); bucket parameters per block into one
   all-gather / one reduce-scatter (cuts launches ~8x, removes the gamma tax); skip
   sharding tiny parameters; prefetch block *i+1*'s all-gather during block *i*
   (needs a CCL subdevice or second CQ); `reshard_after_forward=False` where memory
   allows (-2 % step time measured); fold the 1/N scale into the loss.

## 1. What one FSDP training step does (from the code)

`fully_shard(block)` per block + `fully_shard(model)` root. For every managed
parameter `p` (shard dim = rows for `[1,1,O,I]` weights; RMSNorm gammas
`[1,1,1,E]` fall back to dim 3 and ARE sharded — they are 4 KB tensors):

| Phase | Host call (Python, per parameter) | Device work |
|---|---|---|
| pre_forward | `all_gather(shard)` | AG of the full weight (`all_gather_async`, ring, 2 links, fresh output alloc) |
| post_forward | `set_value(shard)`, `deallocate(full)` | free |
| backward_pre | `all_gather(shard)` again (default `reshard_after_forward=True`) | AG #2 |
| backward_post | `reduce_scatter(full_grad)`, `multiply(shard_grad, 1/N)`, `set_value`, `set_grad`, 2x `deallocate` | RS + a full extra eltwise pass over the shard grad |
| grad-accum only | `all_gather(shard_grad)` in backward_pre | AG #3 of the gradient |

TinyLlama (22 blocks): 8 managed params per block (q, kv, out, w1, w2, w3, 2 gammas) +
3 root (tok_emb, fc, ln_fc.gamma) = **179 parameters → per step: 358 all-gathers, 179
reduce-scatters, 179 scalar multiplies, ~700 deallocates, all issued one tensor at a
time from Python**, strictly serialized with compute on the single command queue.
No prefetch / overlap: block *i*'s all-gather starts only when block *i*'s forward
is reached; the ring collective then runs while all Tensix cores sit idle.

Other observations from reading the path:

- `all_gather_async` / `reduce_scatter_minimal_async` are "async" in name only here:
  same CQ, no subdevice, no persistent output buffers, a barrier semaphore per call.
- Each CCL call recomputes `get_num_links()` (queries routing planes for every row of
  the mesh) and `get_topology()` on the host.
- The `1/N` mean is applied as a separate `ttnn.multiply` on every shard grad; DDP does
  the same after its all-reduce. It could be folded into the loss scale (one op).
- `ThroughputCallback` (sources/examples/train/callbacks.py) multiplies per-rank tokens
  by the `"dp"` axis size only. Under FSDP the batch is sharded on the `"fsdp"` axis, so
  the printed TPS / TFLOPS / MFU are **N× too low** (3.5 % printed vs 27.6 % real on 8 chips).
  The `params` header line likewise prints the per-device shard count.
- Memory-efficient runner: forward AG + reshard, recompute-forward AG + reshard, then
  backward_pre AG again → 3 all-gathers per parameter per step (documented TODO).
- Eager (non-lazy) sharding host-roundtrips every full tensor through numpy at
  `fully_shard` time; `train.py` uses lazy init under FSDP so this only matters for
  the GRPO/Qwen path if lazy init is off there.

## 2. End-to-end A/B: TinyLlama 1.1B, 5 samples x 2048 tokens per device

Same per-device compute in every run (batch = 5 x N). Phase times from the naive
profiler (device sync at every phase boundary), averaged over steps 3-10.

| run | chips | step ms | fwd ms | bwd ms | grad-sync ms | optimizer ms | true MFU |
|---|---|---|---|---|---|---|---|
| DDP  | 4  | 1667 | 452 | 1087 |  71 | 46 | 27.8 % |
| FSDP | 4  | 1654 | 478 | 1148 |   1 | 14 | 27.8 % (printed 7.0 %) |
| DDP  | 8  | 1681 | 453 | 1087 |  80 | 46 | 27.5 % |
| FSDP | 8  | 1668 | 484 | 1154 |   1 |  8 | 27.7 % (printed 3.5 %) |
| DDP  | 32 | 1794 | 454 | 1094 | 152 | 47 | 25.8 % |
| FSDP | 32 | 2072 | 582 | 1436 |   1 |  7 | 22.3 % (printed 0.7 %) |
| FSDP tile-aware shard dim | 32 | **1728** | 497 | 1172 | 1 | 7 | 26.8 % (printed 0.84 %) |

Collective cost = FSDP (fwd+bwd) minus DDP (fwd+bwd); volume = 3 x 2.2 GB x (N-1)/N per
device (2 all-gathers + 1 reduce-scatter of the bf16 parameters).

| chips | FSDP collectives ms | volume GB | effective GB/s | DDP all-reduce ms | DDP GB/s | FSDP net vs DDP |
|---|---|---|---|---|---|---|
| 4  |  87 | 4.95 | 57 |  71 | 46 | -17 ms (faster) |
| 8  |  99 | 5.78 | 59 |  80 | 48 | -20 ms (faster) |
| 32 | 471 | 6.39 | 14 | 152 | 28 | +280 ms (17 % slower) |

Takeaways:

1. On 4-8 chips FSDP is **not** slower than DDP: it moves 1.5x the bytes of DDP's
   all-reduce but the sharded AdamW step saves ~38 ms, so it nets out slightly ahead.
2. The printed TPS/TFLOPS/MFU for FSDP is wrong by a factor N (see section 1). This
   alone makes an FSDP run look 8-32x slower in the logs than it is.
3. At 32 chips the collectives collapse from ~58 GB/s to 14 GB/s effective while DDP's
   all-reduce only halves. The bytes per device barely change with N, so this is a
   per-call latency term that grows with ring size, paid 716 times per step
   (2 AG + 1 RS + 1 multiply per parameter, 179 parameters). Section 4 quantifies it.

## 3. Where the FSDP time goes (hook trace), TinyLlama FSDP8

`fsdp_hook_trace.py` wraps the FSDP hooks and every CCL call. Host-only mode adds no
measurable step time (1665 vs 1668 ms); sync mode brackets each CCL call with a device
sync, so its per-call device numbers include ~100 µs of sync overhead and are an upper
bound.

Per step, 8 chips, averaged over steps 3-6:

| hook | calls | host ms (no sync) | host ms (sync mode) |
|---|---|---|---|
| pre_forward (all-gather x8 params) | 23 | 15.1 | 463 |
| post_forward (swap + free) | 23 | 2.2 | 4 |
| backward_pre (all-gather again) | 23 | 14.1 | 123 |
| backward_post (reduce-scatter, 1/N multiply, swap, 2 frees) | 23 | 60.5 | 1157 |
| **total FSDP host time** | | **92 ms** | |

| op | calls/step | host µs/call | device µs/call (sync mode, upper bound) | device ms/step |
|---|---|---|---|---|
| all_gather fwd + bwd | 358 | 65 (gamma 58, 22 MB weight 74) | 219-525 (fc 1.8 ms) | 130 |
| reduce_scatter | 179 | 183 (gamma 150, 22 MB weight 203) | 411-795 (fc 2.2 ms) | 107 |
| multiply 1/N | 179 | 107 | (launch-bound, 65-130 µs from microbench) | ~15 |
| deallocate | 537 | 9 | 0 | 0 |

By tensor class (sync-mode device time per step): the 66 MLP weights 22 MB take 118 ms,
the 66 attention q/out weights 8 MB take 50 ms, the 22 kv weights 2 MB take 20 ms, the
2 embedding/LM-head weights 125 MB take 11 ms, and **the 45 RMSNorm gammas (4 KB each)
take 39 ms — 16 % of all collective time for 0.01 % of the bytes**.

Reading: at 8 chips the host needs ~92 ms per step to issue the ~1400 FSDP-related ops
and the device needs ~100-120 ms to execute them (microbench-based estimate; the sync
trace's 237 ms is inflated by the syncs). The measured end-to-end cost is 99 ms, i.e. the
two pipelines roughly overlap and both are near their limit. At 32 chips both sides
grow (host: AG 153 µs, RS 490 µs per call → ~165 ms; device: ~210 ms) and the measured
cost is 471 ms — the per-op latency chain is now the bottleneck, not bandwidth.


### Device-side profile (Tracy, `prof_tinyllama_fsdp8`, per device per step)

| | ms | % of kernel time |
|---|---|---|
| kernel busy | 1596 | 100 (step is 1668 ms → **device ~96 % busy**) |
| op-to-op gaps (device idle) | 26 | (18 of it is the batch upload at step start) |
| attention (SDPA fwd 8.0 ms + bwd KV 14.7 ms + bwd Q 7.8 ms per layer) | 672 | 42 |
| matmul (399 calls) | 557 | 35 |
| eltwise / data movement (BinaryNg 334 calls = 68 ms, swiglu bwd, reduce, RoPE, concat heads, transpose) | 124 | 8 |
| **all_gather_async (358 calls)** | **62.5** | **3.9** |
| **reduce_scatter_minimal_async (179 calls)** | **38.4** | **2.4** |
| norm, embedding, loss, other | 134 | 8 |
| AdamW on shards (179 calls, 40 µs each) | 7.2 | 0.5 |

Per-call device kernel time in situ (µs): AG 704x2048→5632 rows 300 (fwd) / 341 (bwd,
max 1069), AG 256→2048 rows 123, AG kv 51, AG gamma 22, AG embedding 1560; RS 5632x2048
469 (max 1489), RS 2048x2048 121, RS kv 46, RS gamma 20, RS lm-head 1807. The 45 gammas
cost only ~3 ms of device time per step; their cost is host dispatch (~16 ms), which at 8
chips is hidden because the device is the bottleneck.

Conclusion for 8 chips: FSDP's collectives are **101 ms of device time (6 % of the step)**
and the device is 96 % busy — FSDP is not host-bound and not slow here. The step is
dominated by SDPA backward (33 %) and matmuls; that is where a 27 % MFU comes from, for
DDP and FSDP alike.

Same profile for DDP8 (`prof_tinyllama_ddp8`), per device per step: kernel busy 1607 ms,
idle gaps 31 ms, 1859 ops. Compute is identical to the byte (attention 672, matmul 557,
eltwise 135 incl. the same 334 BinaryNg calls — DDP's `sync_gradients` also does one
`multiply(1/N)` per parameter). Differences:

| | DDP8 | FSDP8 |
|---|---|---|
| collectives | all-reduce = RS 32.3 + AG 30.1 + 13 ms eltwise = **75.5 ms**, 179 x 2 launches, 168-181 µs each | AG 62.5 + RS 38.4 = **101 ms**, 537 launches |
| AdamW | 46.3 ms (258 µs/call on full params) | 7.2 ms (40 µs/call on 1/8 shards) |
| kernel busy | 1607 ms | 1596 ms |

FSDP moves 1.5x the bytes of DDP but removes 39 ms of optimizer work, so on the device
it is a net -11 ms per step at 8 chips. Note the DDP all-reduce here is itself a composite
of the same reduce-scatter + all-gather kernels, so both paths share the CCL roofline.

## 4. CCL primitive roofline (`ccl_microbench.py`, exact calls FSDP makes)

Steady-state device time per call (10 back-to-back launches, one sync), single-call
latency (sync before/after) and host dispatch cost, TinyLlama shapes, bf16, ring MGD.
Full tables: `python report_ccl_tables.py`.

| tensor (full) | MB | N | AG dev µs | AG GB/s | RS dev µs | RS GB/s | AG host µs | RS host µs | 1/N multiply µs |
|---|---|---|---|---|---|---|---|---|---|
| RMSNorm gamma 1x2048 | 0.004 | 8 | (see hook trace) | | | | | | |
| kv 512x2048 | 2 | 4 | 44-51 | 31-35 | 105 | 15 | 26-42 | 96-98 | ~100 |
| kv 512x2048 | 2 | 8 | 60-80 | 23-31 | 171-180 | 10-11 | 43-71 | 161-167 | ~115 |
| kv 512x2048 | 2 | 32 | 181 | 11 | 392 | 5 | 156 | 371 | 116 |
| q/out 2048x2048 | 8 | 4 | 108 | 58 | 135-144 | 44-47 | 29-37 | 76-126 | 63-98 |
| q/out 2048x2048 | 8 | 8 | 132 | 55 | 150 (245*) | 49 (30*) | 43 | 135 (229*) | 76-112 |
| q/out 2048x2048 | 8 | 32 | 213 | 38 | 508 | 16 | 154 | 486 | 128 |
| w1/w2/w3 5632x2048 | 22 | 4 | 253 | 68 | 313-327 | 53-55 | 27-64 | 76-194 | 66-158 |
| w1/w2/w3 5632x2048 | 22 | 8 | 305 | 66 | 333 | 61 | 42 | 132-216 | 71-109 |
| w1/w2/w3 5632x2048 | 22 | 32 | 405 | 55 | 530 | 42 | 153 | 490 | 130 |
| fc / tok_emb 32000x2048 | 125 | 4 | 1345 | 73 | 1670 | 59 | 26-44 | 131-167 | 209 |
| fc / tok_emb 32000x2048 | 125 | 8 | 1568 | 73 | 1772 | 65 | 47-70 | 217-241 | 170-185 |
| fc / tok_emb 32000x2048 | 125 | 32 | 1864 | 68 | 2005 | 63 | 153 | 491 | 128 |

(*) first measurement of the run; likely warm-up noise.
Line (non-torus) MGD at N=8: all-gather drops to 35-41 GB/s (1.6-1.8x slower than ring),
reduce-scatter is equal on large tensors and 1.7x slower on 2 MB ones. The repo's
`bh_galaxy_*_ring_ring` MGDs are the right choice; the 70B config comment recommending
`line_line` is a perf trap.

What the numbers say:

- **Bandwidth is fine for big tensors**: 55-73 GB/s per device on ≥ 8 MB tensors at 4-8
  chips, ~63-68 GB/s even at 32 chips for the 125 MB embedding. With 2 links × 2 directions
  at ~52 GB/s/link (fabric golden) the ceiling is well above this, but the 22 MB MLP
  weights already run at 80-90 % of what the 125 MB tensor gets.
- **Small tensors are latency-bound**: a 2 MB all-gather takes 60-80 µs at N=8 and 181 µs
  at N=32 (11 GB/s); a 2 MB reduce-scatter 171 µs at N=8 and 392 µs at N=32 (5 GB/s). The
  fixed cost grows roughly linearly with ring size (one hop per rank), ~5 µs per hop for
  AG, ~12 µs per hop for RS. The 4 KB RMSNorm gammas pay the full fixed cost for nothing.
- **Reduce-scatter is 1.2-3x slower than all-gather** on the same tensor and 3-4x more
  expensive to dispatch on the host (130-230 µs at N=8, ~490 µs at N=32 vs 43/153 µs for
  AG). At N=32 the host needs 179 × 490 µs ≈ 88 ms per step just to *issue* the
  reduce-scatters, and 358 × 153 µs ≈ 55 ms for the all-gathers.
- **The `1/N` scalar multiply costs 65-130 µs of device time per call independent of
  size** (launch-bound, 179 calls/step → 13-23 ms/step) plus its host dispatch. Folding
  the mean into the loss (or into the optimizer's lr) removes it entirely.
- Host dispatch scales with the number of devices in the mesh (per-device program /
  runtime-arg setup): AG 30 → 43 → 153 µs and RS ~90 → 140 → 490 µs going 4 → 8 → 32.
  On a 32-chip mesh a Python-driven per-parameter FSDP loop is host-bound.

Summing the microbench numbers over TinyLlama's 179 parameters at N=32 gives ~210 ms of
device time per step for AG+AG+RS+mul, versus 471 ms measured end to end: the other
~260 ms is the device waiting on the host (dispatch of ~1400 small ops) — confirmed by
the hook trace in section 3.

### 4b. The 32-chip cliff: tile-misaligned shards hit the composite CCL fallback

`fully_shard` picks `shard_dim = rows` and only checks `rows % N == 0`
(`_pick_shard_dim_from_shape` / the divisibility check in `fully_shard`). It does not check
that `rows / N` is a multiple of the 32-row tile. When it is not, ttnn's
`composite_common::use_composite_all_gather` / `use_composite_reduce_scatter` fire and the
collective becomes a chain of split / pad / gather / slice / concat ops instead of one
kernel. At N = 4 and 8 every TinyLlama shard is tile-aligned; at N = 32 these are not:

| parameter | full rows | rows per shard at N=32 | tiles | path |
|---|---|---|---|---|
| w1, w3 (44 per model) | 5632 | 176 | 5.5 | composite |
| kv_linear (22) | 512 | 16 | 0.5 | composite |
| tok_emb, fc (2) | 32000 | 1000 | 31.25 | composite |
| q, out, w2 (dim 2), gammas (dim 3) | 2048 | 64 | 2 | fast |

Device-synchronized hook trace at N=32 (`trace_sync_fsdp32`, upper bounds incl. sync):

| shape (shard) | AG µs/call | RS µs/call | host µs/call (AG / RS) |
|---|---|---|---|
| 176x2048 (misaligned) | 4470-5320 | 9356 | 4100-4900 / 9000 |
| 16x2048 (misaligned) | 4260-5050 | 8370 | 4000-4800 / 8000 |
| 1000x2048 (misaligned) | 8740-9700 | 22470 | 5100-7800 / 10400 |
| 64x2048 (aligned, same bytes as 16x2048 x4) | 550-600 | 876 | 285 / 670 |
| 64x5632 (aligned) | 825-835 | 1184 | 340-370 / 775 |
| 1x64 gamma | 460-500 | 733 | 250 / 505 |

The misaligned shapes are **~10x slower per call and ~10x more expensive on the host**
(the composite path issues several ops per collective). Of the 1541 ms of (sync-inflated)
collective time per step at N=32, ~1330 ms is on the three misaligned parameter groups.
This, not ring latency, is the dominant term of the +471 ms measured at 32 chips; the
per-hop latency growth (section 4) is the second-order term.

Isolated in the microbench at N=32 (`ccl_32_ring_all`, same bytes, only the shard dim differs):

| tensor | shard dim | shard rows/cols | tile-aligned | AG dev µs | AG GB/s | AG host µs | RS dev µs | RS GB/s | RS host µs |
|---|---|---|---|---|---|---|---|---|---|
| kv 512x2048 (2 MB) | 2 | 16 rows | no | **4082** | 0.5 | **4013** | **7327** | 0.3 | **7252** |
| kv 512x2048 | 3 | 64 cols | yes | 191 | 10.7 | 158 | 382 | 5.3 | 344 |
| w1 5632x2048 (22 MB) | 2 | 176 rows | no | **3680** | 6.1 | **3633** | **8079** | 2.8 | **8006** |
| w1 5632x2048 | 3 | 64 cols | yes | 430 | 52 | 161 | 584 | 38 | 506 |
| w2 2048x5632 (22 MB) | 3 | 176 cols | no | **10171** | 2.2 | **10082** | **7967** | 2.8 | **7910** |
| w2 2048x5632 | 2 | 64 rows | yes | 426 | 52 | 158 | 566 | 40 | 491 |
| fc 32000x2048 (125 MB) | 2 | 1000 rows | no | **7574** | 17 | **4043** | **22187** | 5.7 | **9441** |
| fc 32000x2048 | 3 | 64 cols | yes | 1888 | 67 | 169 | 2058 | 62 | 503 |

Misaligned shards are 4-21x slower on the device and 20-50x more expensive on the host
(the composite path issues many small ops). At N=4 and N=8 none of TinyLlama's shards is
misaligned, which is why those meshes show no cliff.

Host-only trace at N=32 (`trace_host_fsdp32`, no syncs, step time unchanged at 2077 ms):

| | host ms per step |
|---|---|
| pre_forward hooks (179 AG) | 283 |
| backward_pre hooks (179 AG) | 290 |
| backward_post hooks (179 RS + 179 mul + frees) | 846 |
| **FSDP hooks total** | **1423 of the 2077 ms step** |
| of which misaligned shapes (w1/w3 RS 12.5 ms/call, kv RS 7.7 ms/call, w1/w3/kv AG 3.7-3.8 ms/call, emb/fc) | ~1260 |
| of which aligned shapes (AG 246 µs, RS 605-694 µs, gamma RS 448 µs per call) | ~130 |

So at 32 chips the FSDP step is **host-bound**: Python spends 1.4 s per step inside the
FSDP hooks, 90 % of it building the composite split/pad/gather/concat op chains for the
three misaligned parameter groups. The device-side ring latency growth is real but
secondary. The same trace at N=8 (all shards aligned) shows 92 ms of hook time.

**Measured fix** (`exp_tileaware_fsdp32`: `FSDP_TILE_AWARE=1` in `fsdp_hook_trace.py` swaps in a
shard-dim picker that prefers the dim whose per-rank shard is a multiple of 32, otherwise
identical precedence; no parameter left unsharded):

| TinyLlama, 32 chips | step ms | fwd ms | bwd ms | FSDP hook host ms | vs DDP32 (1794 ms) |
|---|---|---|---|---|---|
| FSDP32 as-is | 2072 | 582 | 1436 | 1423 | +15.5 % |
| FSDP32 tile-aware shard dim | **1728** | 497 | 1172 | 193 | **-3.7 %** |
| (control) FSDP8 tile-aware | 1669 | 484 | 1154 | 88 | unchanged vs 1668 |

Remaining FSDP collective cost at N=32 after the fix: 121 ms (vs 99 ms at N=8), i.e. the
ring-latency / launch-count growth is a ~20 ms effect; the other 350 ms was the composite
fallback.

Who else hits it: Qwen3-32B FSDP=32 (`grpo_boolq_qwen3_32b_fsdp.yaml`): tok_emb / lm_head
`[151936, 5120]` → 4748 rows per shard (148.4 tiles) → composite path on two 1.5 GB
tensors (est. several hundred ms per step). Llama-70B TP8/FSDP4 and Llama-8B TP4/FSDP8 are
aligned on every parameter (FSDP falls to dim 3 = 8192/4 or 4096/8 there).

Fix (small): in `_pick_shard_dim_from_shape`, prefer the candidate dim where
`(shape[d] // axis_size) % 32 == 0`; if neither dim qualifies, pad the parameter's shard
dim up to `32 * N` (the padded rows are zero and unused) or leave the parameter replicated.
For TinyLlama at N=32: w1/w3 and kv are aligned on dim 3 (2048/32 = 64), the embedding /
LM head are not on either dim (32000/32 = 1000 rows, 2048/32 = 64 cols → dim 3 works!). So
dim 3 fixes all of them.

## 5. Llama-8B TP4 x {DDP8, FSDP8}, mesh [8,4], batch 16 (2 samples x 2048 per TP group)

Repo configs `training_shakespeare_llama_8b_tp4_{ddp8,fsdp8}.yaml`, steps 3-8 averaged.
Loss trajectories are identical to 3 decimals (10.281 / 10.297 at steps 7/8 in both).

| run | step ms | fwd ms | bwd ms | grad-sync ms | optimizer ms | printed MFU | true MFU |
|---|---|---|---|---|---|---|---|
| DDP8  | 1395 | 318 | 747 | 225 | 93 | 21.9 % | 21.9 % |
| FSDP8 | 1291 | 382 | 879 |   2 | 15 |  2.95 % | 23.6 % |

FSDP's collectives cost 64 ms in forward and 133 ms in backward (197 ms total for
2 AG + 1 RS of the 2 GB-per-TP-rank weights, ~44 GB/s effective), but it removes DDP's
225 ms all-reduce and cuts the AdamW step from 93 to 15 ms. **Net: FSDP is 7.4 % faster
than DDP here**, while the log shows it 7.4x slower because of the token-accounting bug.

### 5b. Gradient accumulation (TinyLlama, 8 chips, 2 micro-batches x 6 samples per device)

| run | micro-batch 1 fwd / bwd ms | micro-batch 2 fwd / bwd ms | step ms |
|---|---|---|---|
| FSDP8 ga=2 | 578 / 1376 | 578 / 1421 | 3997 |
| DDP8 ga=2 | **out of DRAM** in the first forward (bank 3.99 GB allocated of 4.27 GB) | | |

Both micro-batches scale linearly from the ga=1 numbers (6/5 x 484 = 581 fwd; 6/5 x 1154 =
1385 bwd). The second micro-batch pays +45 ms for all-gathering the accumulated shard
grads back to full shape (2.2 GB x 7/8 at ~42 GB/s) — the only FSDP-specific cost of
accumulation. DDP could not run this configuration at all: FSDP's memory saving (weights
+ grads + AdamW state /8) is what makes 6 samples per device fit.

### 5c. Gradient checkpointing (`runner_type: memory_efficient`), Llama-8B TP4, 8x4

| run | step ms | fwd ms | bwd ms | grad-sync ms | optimizer ms |
|---|---|---|---|---|---|
| DDP8 mem-eff  | 1686 | 318 | 1044 | 219 | 93 |
| FSDP8 mem-eff | 1634 | 381 | 1223 |   1 | 15 |

Checkpointing adds 297 ms of recompute to DDP's backward and 344 ms to FSDP's: the extra
47 ms is the third all-gather per parameter (gather for the recompute-forward, reshard,
gather again for backward — the FSDP.md TODO). FSDP stays 3 % faster than DDP overall.

### 5d. Inference under FSDP (what a GRPO rollout pays), TinyLlama, `fsdp_inference_overhead.py`

Every no-grad forward through a `fully_shard`-ed model all-gathers all weights and frees
them again; the GRPO completer runs one such forward per generated token.

| chips | tokens/device | sharded forward | weights kept gathered | gather cost per forward |
|---|---|---|---|---|
| 8 | 32 | 50.2 ms | 20.3 ms | 30.0 ms (1.79 GiB at 64 GB/s) — 60 % of the forward |
| 8 | 256 | 52.9 ms | 22.6 ms | 30.3 ms |
| 32 | 32 | 66.5 ms | 34.8 ms | 31.7 ms (1.98 GiB at 67 GB/s) — 48 % |
| 32 | 256 | 67.9 ms | 35.7 ms | 32.2 ms |

The gather cost is pure bandwidth (bytes of the whole model per device, independent of
sequence length or mesh size), so it scales with model size while the decode compute does
not: for Qwen3-32B in bf16 that is ~62 GiB per decode step per device, i.e. about one
second of all-gather per generated token — the training step (one gather per step) is not
the problem in that config, the rollout is. Fix for models whose unsharded weights fit in
DRAM (≤ ~12B on 32 GB chips): gather once for the whole rollout and reshard afterwards
(2.5x faster forward here). For 32B-class models the rollout needs a different weight
layout (e.g. TP-sharded inference weights, resharded once per training step).

## 6. Opportunities, ranked by (measured or estimated) payoff / effort

| # | Change | Evidence | Expected gain | Effort |
|---|---|---|---|---|
| 1 | **Fix `ThroughputCallback` token accounting** (multiply per-rank tokens by every data-parallel axis: `dp` *and* `fsdp`; also print full param count) | printed MFU 3.5 % vs real 27.7 % | none on runtime; fixes the perception and every dashboard number | trivial |
| 1b | **Make the auto shard-dim tile-aware** (prefer the dim whose shard is a multiple of 32; pad or replicate otherwise) | section 4b: misaligned shards run the composite CCL path, ~10x slower per call and ~10x host cost; dominates the +471 ms at N=32; hits Qwen3-32B FSDP32's embedding/LM head | **measured: FSDP32 2072 → 1728 ms/step (-17 %), now 3.7 % faster than DDP32**; Qwen3-32B: est. hundreds of ms per step | small (a few lines in `fsdp.py`; the experiment picker is in `fsdp_hook_trace.py`) |
| 2 | **Bucket / flatten per FSDP unit**: concatenate a block's shards into one flat buffer and issue one AG + one RS per block (FlatParam-style), with views back into the individual weights | 716 launches/step; per-op fixed cost dominates at N=32 (2 MB RS = 392 µs, 5 GB/s); host RS dispatch 490 µs/call at N=32 | at N=32: ~8x fewer launches, collective cost 471 → ~150-200 ms (est.), i.e. FSDP32 from +17 % to roughly parity with DDP32; at N=8 ~30-40 ms | medium (needs a per-block flat buffer + slicing views; or at least group the 6 matrices with equal inner dim) |
| 3 | **Don't shard tiny parameters** (size threshold, e.g. < 1 MB: RMSNorm gammas, biases) — keep them replicated and all-reduce them in `sync_gradients` | 45 gammas = 180 launches/step, ~16 ms host, 4 KB payloads | removes 25 % of launches for free; -10-15 ms at N=8, more at N=32 | small |
| 4 | **Prefetch / overlap**: kick off block *i+1*'s all-gather while block *i* computes (forward) and block *i-1*'s during block *i*'s backward; run CCL on a subdevice / second CQ so it does not serialize with matmuls | 100 % of the 62 ms AG time at N=8 is exposed today (device 96 % busy but serially) | hides most AG time: up to -60 ms at N=8, several hundred ms at N=32 | large (subdevice + persistent buffers + semaphore management; the FSDP.md TODO) |
| 5 | **`reshard_after_forward=False`** for models that fit, or for the last K blocks | measured: 1668 → 1633 ms (-2 %) at N=8, removes 179 AGs | -2 % step time; costs full-weight residency between fwd and bwd | trivial (already a flag) |
| 6 | **Fold the 1/N mean into the loss** (or lr) instead of a `ttnn.multiply` per shard grad | 179 launches, 107 µs host + 65-130 µs device each; hidden at N=8 (skip experiment: 1667 vs 1668 ms), not necessarily at N=32 | -15-25 ms device / -19 ms host at N=32 | trivial |
| 7 | **Cache `num_links` / topology per (mesh, axis)** in `ttnn_fixed::distributed` instead of querying routing planes per call | host dispatch per CCL op scales with device count | a few tens of µs per call → ~20 ms/step at N=32 | small |
| 8 | **Use ring MGDs** (torus fabric) everywhere; fix the `line_line` recommendation in the 70B config comment | AG 55-73 GB/s ring vs 35-41 GB/s line at N=8 | 1.6-1.8x on every all-gather if anyone runs line | trivial |
| 9 | **Reduce-scatter kernel**: RS is 1.2-3x slower than AG for the same bytes and worst on small tensors; persistent intermediate/output buffers would also cut host cost | microbench RS 8 MB: 150-245 µs vs AG 132 µs (N=8); 508 vs 213 µs (N=32) | up to -20-40 ms/step at N=8, more at N=32 | CCL team |
| 10 | **Memory-efficient runner**: reuse the recompute-forward gather for backward (3 → 2 AGs per param) | documented TODO; see section 5b | -1/3 of AG time under grad checkpointing | small |
| 11 | Grad accumulation: the extra grad all-gather costs +45 ms per micro-batch at N=8 (2.2 GB); keeping accumulated grads sharded and reduce-scattering each micro-batch's grad into them would remove it | `tinyllama_fsdp8_ga2`: bwd 1376 (1st) vs 1421 ms (2nd micro-batch) | -45 ms per extra micro-batch | medium |

What is **not** the problem: bandwidth on ≥ 8 MB tensors (55-73 GB/s per device, ~85 % of
the 125 MB tensor's rate already at 22 MB), the AdamW step (7 ms), the deallocation calls,
or the autograd callback nodes.

### Measurement caveat: the naive phase profiler costs ~57 ms per step

Every end-to-end number above was taken with `TTML_NAIVE_PROFILER=1`, which synchronizes the
device at five points per step and therefore removes the host/device overlap a production run
has (the host issues the next step's dataloader/collate/upload work while the device is still
finishing backward). Re-running TinyLlama FSDP32 with all fixes and **no** profiler gives
1668 ms per step vs 1725 ms with it. Relative comparisons in this report are unaffected (both
arms carried the same syncs), and the inter-step "other" phase is hidden in production; the
final table in section 7 uses profiler-free numbers.

## 7. Fixes landed on branch `imichalak/fsdp-perf-fixes` (2026-09-17)

Each commit is self-contained and was measured on the galaxy before committing.

| commit | change | measured effect |
|---|---|---|
| `[tt-train] Count every data-parallel axis in the training throughput metric` | `ThroughputCallback` multiplies per-rank tokens by every data-parallel axis (`dp` and `fsdp`); header prints the unsharded parameter count | FSDP8 now prints 27.8 % MFU / 1.1B params (was 3.5 % / 137M); no runtime change |
| `[tt-train] FSDP: prefer a tile-aligned shard dim in fully_shard's auto mode` | auto shard dim prefers the dim whose per-rank shard is a multiple of 32; warns otherwise; device-free unit tests | TinyLlama FSDP32 2072 → 1734 ms (-16 %), FSDP hook host time 1423 → 193 ms; 8-chip unchanged |
| `[tt-train] FSDP: don't re-gather weights for a gradient-checkpoint recompute` | `AutoContext.is_backward_in_progress()` (C++ depth counter around `Tensor::backward`); the wrapped forward keeps weights gathered when it is a recompute | Llama-8B TP4 FSDP8 mem-efficient 1634 → 1592 ms (-2.6 %), backward 1223 → 1177 ms; losses identical |
| `[tt-train] Reuse reduce_scatter staging buffers across calls` | per-(spec, dim, axis) cache of the op's two staging intermediates in `CCLResources`, passed as persistent buffers with a fresh output | RS host cost 490 → 425 µs/call at N=32 (neutral at 8); step 1732 → 1725 ms at 32 (noise-level) |
| `fsdp_reshard_after_forward` device-config knob + last block always kept gathered | skips the backward all-gather when the unsharded model fits | TinyLlama FSDP32 1725 → 1696 ms (-1.8 %) with the knob off; loss identical |
| `fsdp_keep_gathered_gib` device-config knob | keeps the last K blocks gathered under a per-device memory budget (middle ground when the whole model does not fit) | 1 GiB keeps 12/22 TinyLlama blocks: FSDP32 1725 → 1703 ms; loss identical |
| `ttml.fsdp.unshard_for_inference()` (+ used around GRPO rollouts) | gather once for a gradient-free block instead of once per forward | TinyLlama no-grad forward 50.2 → 20.3 ms (8 chips); GRPO end-to-end not measured (deprioritized) |

Experiments that did **not** pay and were dropped:

- Skipping the 1/N scalar multiply: 1730 vs 1729 ms at 32 chips (also no gain at 8), so
  folding the mean into a root-level gradient scale is not worth its coupling.
- CCL knob sweep (`ccl_knob_sweep.py`, `results/fix/knobs_{8,32}.json`): `chunks_per_sync`,
  `num_buffers_per_channel` and broadcast mode never beat the defaults. `num_workers_per_link=1`
  for reduce-scatter cuts the per-call cost of small tensors dramatically in isolation
  (32 chips: 2 MiB 381 → 172 µs, 8 MiB 436 → 179 µs; 8 chips: 2 MiB 228 → 87 µs) but loses on
  large ones (125 MiB 1988 → 2301 µs). A size-based heuristic in the wrapper (1 worker ≤ 2 MiB
  per link, ≤ 8 MiB on ≥ 16-device meshes) measured **no end-to-end gain**: FSDP32 1739 vs
  1725-1732 ms, FSDP8 1670 vs 1668 ms. The reduce-scatters are not on the critical path once
  the host runs ahead, so the heuristic was reverted.
- More fabric channels: an MGD with `channels { count: 4 }` fails at fabric init ("Expected 4
  eth links from physical chip 0 to physical chip 1"); the galaxy ring has 2 Ethernet links per
  neighbour pair, so 2 channels (the stock MGDs) is the hardware limit.
- The inter-step "other" phase (dataloader sampling + collate + batch upload + callbacks) is
  ~46 ms at 32 chips for DDP and FSDP alike (15 ms at 8 chips; `InMemoryDataloader.next()`
  alone is 18 ms for 160 samples). It scales with the global batch, not with FSDP.

Cumulative, TinyLlama FSDP32 (5 samples/device, profiler on): 2072 → 1696 ms per step (-18 %)
with resharding off, 1703 ms with a 1 GiB budget, 1725 ms with defaults; from 15.5 % slower
than DDP32 to 1.3-5.5 % faster.

**Final table, profiler off (production conditions), default knobs, steps 9-12:**

| TinyLlama 1.1B, 5 samples x 2048 tokens per device | DDP | FSDP (this branch) | FSDP vs DDP |
|---|---|---|---|
| 8 chips | 1659 ms, 27.8 % MFU | 1645 ms, 28.1 % MFU | -0.8 % |
| 32 chips | 1685 ms, 27.4 % MFU | 1668 ms, 27.7 % MFU | -1.0 % |

FSDP on 32 chips now costs the same per step as on 8 chips with 4x the global batch; the
remaining collective cost (~120 ms/step device time on 32 chips) is bandwidth/latency of
tile-aligned ring all-gathers and reduce-scatters and can only be removed by overlapping it with
compute (section 6, #4).

Research conclusions that bound what is left:

- **Bucketing** (one all-gather per block) needs zero-copy sub-buffer views; `ttnn.view` only
  reshapes, and slicing copies, so it is a ttnn feature request rather than a tt-train change.
- **Prefetch / overlap** needs a CCL subdevice. Reserving even one 10-core column costs ~8 % of
  matmul throughput, while FSDP's collectives are 6-7 % of the step at 5-10k tokens per device,
  so it only pays at ≤ 2-4k tokens per device (70B-class configs), where they are 15-33 %.
- The remaining FSDP32 collective cost (~120 ms/step) is device-side all-gather/reduce-scatter
  ring latency on 8-22 MB tensors; a knob sweep of the CCL ops (workers per link, buffers per
  channel, chunks per sync, broadcast mode) is in `ccl_knob_sweep.py`.

## 8. Overlapping collectives with compute: CCL sub-device + prefetch (2026-09-17)

Implemented on the branch (see `docs/FSDP.md`, "Overlapping collectives with compute"):

- **tt-metal**: `MeshDevice::set_compute_with_storage_grid_size_override()` (the mesh command
  queue requires every program to lie inside one sub-device, and nearly every op sizes its grid
  from `compute_with_storage_grid_size()`); the circular-buffer core-range check in
  `program.cpp` accepts ranges inside any active sub-device.
- **tt-train**: `AutoContext.enable_ccl_sub_device(columns)` builds a 2-sub-device manager
  (compute = grid minus the rightmost column(s), CCL = that column), loads it and applies the
  grid override; the CCL wrappers pass `subdevice_id` and (all-gather) a persistent output;
  `fully_shard` switches to prefetch mode: 3-slot persistent gather pool, next-unit gather
  issued at each unit's pre_forward / previous-unit gather at backward_pre, reduce-scatters
  deferred one unit and issued after a compute drain, `record_event(sub_device_ids=…)` barriers,
  deferred frees and end-of-backward 1/N scaling. `device_config.fsdp_ccl_subdevice_columns: 1`.
- Two mechanism bugs found and fixed on the way: ttnn ops spanning both sub-devices are fatal
  (hence the device-level grid override rather than per-op changes), and the gather-buffer pool
  must be keyed per parameter position, not per shape (q/out and w1/w3 share shapes).

### 8.1 Why one queue was not enough

`subdevice_overlap_probe.py` (8 chips, CCL sub-device on): one matmul + one all-gather
interleaved overlap (C ≈ max(A,B)), but 8 gathers followed by 8 matmuls cost the sum (E),
while (gather, matmul) x8 interleaved overlaps (F). The dispatcher launches programs in order
and a launch waits for the previous program *on the same sub-device*, so a burst of CCL
launches on one queue stalls every compute launch queued behind it. Trickling collectives
between submodules (1-queue mode, autograd callbacks) recovered only part of it; issuing them
on a second hardware command queue recovers all of it. `ttnn` device operations now launch on
the thread's current command queue (`ttnn.decorators.push_current_command_queue_id_for_thread`).

### 8.2 Hazards found with two queues (all fixed, all bit-exact now)

| symptom | cause | fix |
|---|---|---|
| wrong loss from step 1 | gather pool keyed by shape (q/out, w1/w3 share shapes) | key per (slot, parameter position) |
| garbage with memory-efficient runner | trickle callbacks recorded into the recompute graph | callbacks only when grad enabled and not inside backward; recompute passes prefetch=False |
| slow drift at 1 sample/device | optimizer rewrites shards in place on queue 0 while the next step's gathers read them on queue 1 | compute→CCL fence |
| small, run-to-run different drift at 5 samples/device | a compute→CCL dependency not covered by per-slot release events (not identified) | compute drain before every collective group (measured: free when compute-bound) |
| gradient accumulation explodes (1e37) | gathering the carried shard grads into a persistent per-slot buffer -- corrupt even fully serialized (not identified) | fresh gather output + synchronous barrier + free (the path is rare) |
| gradient accumulation: rare bf16-level drift (differs run to run, same convergence) | residual cross-queue race in the accumulation path; any extra compute-side wait removes it (`serialize`, `serialize_rs`, `serialize_ag` are all bit-exact, 4 of 4 runs), compute-side drains do not | **open**; train.py warns; use `TTML_FSDP_OVERLAP_DEBUG=serialize_rs` or run accumulation without overlap (it gains only 1.5 % there) |

### 8.3 Measurements (profiler off, mean of steps 3-12, 2048 tokens/sample, ring MGD)

| config | samples/device | no overlap | overlap (CCL sub-device, 2 CQs) | Δ | losses |
|---|---|---|---|---|---|
| TinyLlama FSDP8 b8 | 1 | 460.5 ms | 434.2 ms (418.6 with per-slot events) | −5.7 % | identical (12 steps) |
| TinyLlama FSDP8 b40 | 5 | 1645 ms | 1611 ms | −2.1 % | identical (12 steps) |
| TinyLlama FSDP8 b8 mem-eff | 1 | 553 ms | 512 ms | −7.4 % | identical @10 |
| TinyLlama FSDP32 b32 | 1 | 482.9 ms | 460.8 ms (438.9 with per-slot events) | −4.6 % | identical (12 steps) |
| TinyLlama DDP32 b32 | 1 | 497.1 ms | -- | | |
| TinyLlama FSDP8 ga2 (b48 x 2) | 6 x 2 | 3996 ms | 3940 ms (4124 with `serialize_rs`) | −1.5 % (+3.2 %) | bit-exact 1 of 5 runs; others drift at the 3rd-4th digit from step 2-5 and converge identically |

Reading: the win is the exposed collective time minus the 8.3 % compute column. At 1
sample/device (collective-bound) it is 5-7 %; at 5 samples/device (compute-bound, ~93 %
device busy) the collectives were already mostly hidden behind the dispatcher queue and the
net is ~2 %. The per-slot event variant was ~4 % faster than the drain variant on the
collective-bound configs (it let the CCL queue run two blocks ahead instead of one) but was not
bit-exact; recovering it needs the missing dependency to be named -- the one open item.

### 8.4 Practical limit reached

With every collective on the CCL sub-device and its own queue, the remaining exposed time at
8 chips is the per-block gather latency that a one-block lookahead cannot hide when a block
computes for less than its gather takes (b8: ~10 ms compute vs ~8 ms of gathers per block plus
the launch tail). Deeper lookahead needs the precise dependency (8.2, row 4). Beyond that the
levers left are the CCL kernels themselves (all-gather at 55-73 GB/s vs 104 GB/s of link
bandwidth on 2 links) and, for larger models, keeping more blocks gathered
(`fsdp_keep_gathered_gib`). Gradient accumulation under overlap is the one configuration left
that is not bit-exact (8.2, last row): the drift is bf16-level and the runs converge the same, and
every variant that adds a compute-side wait after a collective is exact, so a consumer on the
compute queue reads a collective result slightly early in the accumulation flow; it was not
found by inspection (the accumulated-grad gather already drains, gathers into a fresh buffer,
barriers and frees) and the small-model probe (`ga_overlap_probe.py`) does not reproduce it.
Given the 1.5 % it gains on an accumulation step, the recommendation is to run accumulation
without the CCL sub-device until it is understood.

### 8.5 The CCL kernels run slower on a 10-core sub-device (2026-09-17, afternoon)

`ccl_microbench.py --ccl-subdevice N` (8 chips, ring, bf16, device µs per call, shard dim 2):

| tensor | MB | AG full grid | AG 1 column (10 cores) | AG 2 columns (20 cores) | RS full | RS 1 col | RS 2 col |
|---|---|---|---|---|---|---|---|
| q/out 2048x2048 | 8 | 133-143 | 267 | 133-150 | 141 | 189 | 210 |
| kv 512x2048 | 2 | 80 | 105 | 64 | 89 | 74 | 125 |
| w1/w2 5632x2048 | 22 | 304 | 692 | 305 | 328 | 489 | 340 |
| fc 32000x2048 | 125 | 1568 | 3895 | 1582 | 1756 | 2692 | 1817 |

Cause (`all_gather_async_default_program_factory.cpp::default_workers`): the op needs
`(workers + 1 mux) x 2 directions` cores per link and picks the largest of {4, 2, 1} workers that
fits: 20 cores for 4 workers, 12 for 2, 8 for 1. A 10-core column therefore runs every all-gather
with **one worker per direction per link**, 2.3-2.5x slower than the full grid (66 -> 29 GB/s on
22 MB), and reduce-scatter 1.4-1.5x slower. Two columns (20 cores) restore full speed at 16.7 % of
the compute grid; a 12-core row would allow 2 workers at 10 %. This is the largest single reason
the overlap realised ~40 % of its theoretical gain: the hidden collectives were twice as long as
the ones they replaced. `enable_ccl_sub_device(columns, rows)` and
`device_config.fsdp_ccl_subdevice_rows` expose the row option; end-to-end numbers below.

End to end (profiler off, mean of steps 3-12, 1 sample x 2048 tokens per device, losses identical
to the no-overlap run in every row):

| config | no overlap | 1 column (10 cores, 1 worker, 8.3 %) | 1 row (12 cores, 2 workers, 10 %) | 2 columns (20 cores, 4 workers, 16.7 %) |
|---|---|---|---|---|
| TinyLlama FSDP8 b8 | 460.5 ms | 434.2 | 425.3 | **421.7** (-8.4 %) |
| TinyLlama FSDP32 b32 | 482.9 ms | 460.8 | **444.6** (-7.9 %) | 450.0 |
| Llama-8B TP4xFSDP8 mem-eff (2 samples/TP group) | 1591 ms | **1569.5** (-1.4 %) | 1609.5 (+1.2 %) | 1658.6 (+4.2 %) |

Reading: faster collectives beat the extra reserved cores whenever the step is collective-bound;
the row is the better default (10 % of the grid for 2 workers), two columns win only when the
collectives dominate outright. On the compute-bound Llama-8B step the 16.7 % column cost is
larger than the whole collective share, so it loses; even the 1-column gain is small there because
at 2 samples per TP group the collectives are ~6 % of the step. DDP32 at the same batch is 497 ms,
so FSDP32 with overlap is now 10.5 % faster than DDP32.

### 8.6 Two more that did not pay

- **Do not shard tiny parameters** (`fsdp_min_shard_bytes: 65536`, leaves the 45 RMSNorm gammas
  replicated): b8 434 -> 445 ms, b32 461 -> 487 ms. The gammas' 180 launches were overlapped on
  the CCL queue; replicated they cost 45 exposed all-reduces in `sync_gradients` at the end of the
  step (~1 ms each on 32 chips). Only worth it together with a bucketed all-reduce of the tiny
  grads (one launch), which needs concat/split ops per step. Knob kept (default 0), not recommended.
- **Merge the two adjacent drains in backward_pre** (prefetch gather + deferred reduce-scatter
  flush): 434.2 -> 433.6 ms, noise. Kept, it is free.

### 8.7 Where the remaining drift was looked for (items 6 and 7 of the push list)

Facts established with the probes in this round (all in `tools/profiling/fsdp_bench/`):

- **Cross-queue events work** (`cross_queue_event_probe.py`): a queue-1 gather after
  `record_event(cq0)` / `wait_for_event(cq1)` always reads the completed queue-0 result. The probe
  cannot make the gather overtake compute on purpose either: the host enqueues 2048^2 matmuls slower
  than the device runs them, so a chain of 300 is finished 0.4 ms after the gather is issued.
- **The collectives are bitwise deterministic under concurrent compute** (`rs_determinism_probe.py`,
  11 repetitions each, idle and next to a 40 x 4096^2 matmul chain, sub-device and full grid). The
  ring reduce-scatter kernel's accumulation order is fixed by direction and iteration, not by
  arrival, so "timing-dependent reduction order" is ruled out.
- **No collective falls back to fresh buffers or the composite path** in any drifting run (the wrapper
  diagnostics never fire for TinyLlama or Llama-8B TP4).
- **Global semaphores were allocated on the 11x10 compute rectangle only** (CCLResources is built
  after the grid override) while the kernels run on the reserved column; fixed to the full grid. It
  did not change the drift, but it is a real bug.
- **The grid-only control is invalid**: with the sub-device on and the overlap machinery off
  (`TTML_FSDP_OVERLAP=0`) the loss is inf at step 2, because programs on different sub-devices do not
  wait for each other even on one queue. Collectives on the CCL sub-device always need the
  dependency machinery.
- **Drain scope bisection** (`TTML_FSDP_DRAIN_SCOPE=fwd|bwd`, 5 samples/device): without the drain in
  backward the loss is 1e20 from step 3; without it in forward the loss drifts from step 1. Both
  passes depend on it; it does not localise the events design's miss.

What remains open, with the evidence:

| configuration | reference run-to-run | overlap run-to-run | serialize | serialize_ag | serialize_rs |
|---|---|---|---|---|---|
| TinyLlama FSDP8, 1 and 5 samples/device; FSDP32 | exact | exact (every shape, 12 steps) | | | |
| TinyLlama FSDP8 grad accumulation x2 | exact | drifts from step 2-5 (bf16 level) | exact | exact | exact |
| Llama-8B TP4 x FSDP8 mem-eff | exact (1590.3 vs 1591.0 ms) | drifts from step 2; step 1 shifted deterministically 11.898 -> 11.906 | 18.148 vs 18.156 @3 | step 1 exact, @4/@6 shifted | step 1 shifted |

For Llama-8B the step-1 shift is deterministic and disappears when the gathers are serialized, so
a forward consumer reads a gather result early, or the gather and the TP all-reduce (on queue 0, over
the same fabric links) interfere; the two probes above did not reproduce either in isolation. The
row-shaped sub-device (1609.5 ms) loses on this compute-bound step (2 samples per TP group,
collectives ~6 %); one column gains 1.4 %. Recommendation unchanged: overlap is bit-exact and worth
5-8 % on plain FSDP at 1-5 samples/device; for gradient accumulation and TP x FSDP it converges but
is not bit-exact, and its gain there is 1-2 %, so run those without the sub-device until the
early consumer is found. `TTML_FSDP_OVERLAP_DEBUG=serialize_rs|serialize_ag|serialize`,
`TTML_FSDP_DRAIN_SCOPE` and `TTML_FSDP_OVERLAP=0` are the bisect toggles left in place.

**Where the practical limit sits after this round.** TinyLlama FSDP8 at 1 sample/device:
460.5 -> 421.7 ms (two columns) or 425.3 (one row); FSDP32 482.9 -> 444.6 (one row), 10.5 % under
DDP32. The sub-device worker count (8.5) was the largest recoverable piece of the overlap gap; the
two-block lookahead (~4 %) still needs its missing dependency named.

### 8.8 Root cause of the drift: collectives are cross-device, events are per-device (2026-09-17, evening)

Found with `overlap_race_harness.py`, a 300-line reproduction of the overlap schedule (K blocks of
matmuls on queue 0 reading a rotating pool of persistent gather slots filled on queue 1, an in-place
shard update per step, pluggable dependency policies, bitwise checking against a synchronous
reference, per-device mismatch reporting). It reproduces in about one second per variant:

| variant | result |
|---|---|
| no dependencies | corrupt (as expected) |
| per-slot release events (the design that drifted) | **corrupt: 4 of 8 blocks per step, partial (old/new mix), on a different subset of devices each step** |
| drain before every gather (shipped design) | clean |
| dedicated slot per block (no reuse), any policy | clean |
| release events with one block of extra slack (4 slots) | clean |
| single queue, any policy | clean (the in-stream waits serialize everything) |

Everything local was verified correct along the way: event ids are monotonic, the queue-1 wait
never releases before the awaited queue-0 event has completed, compute programs launch in order
(dedicated slots are clean even with no dependencies, so the worker counters are right), reduce-
scatter and all-gather are bitwise deterministic next to heavy compute, and no collective falls back
to the composite path.

**Mechanism.** An all-gather (or reduce-scatter) into a persistent buffer is a *cross-device*
write: when device E's gather starts it pushes its shard into every other device's copy of the slot.
A release event recorded on E's compute queue orders E's gather after E's own compute, not after
device D's, and D may still be a few hundred microseconds behind, reading its copy of the slot. With
per-slot events the time slack is zero (the wait is on exactly the block that is the hazard on the
other device), so lagging devices get their slot overwritten mid-read; the mismatching devices vary
from step to step. The drain design "worked" because it waits for the *next* block as well, one
block of slack. The same mechanism explains the residual drift in the shipped design: the
reduce-scatter staging buffers (section 7) were one set per shape, reused by back-to-back
reduce-scatters on queue 1 with zero slack, and the global semaphores rotated through only two sets.
Gradient accumulation, with its extra collectives, hit it 4 of 5 runs.

**Fixes (this branch).**
1. `CCLResources` rotates 4 staging sets per shape and 8 semaphore sets per queue (slack for the
   skew). Gradient accumulation: bit-exact 3 of 3 runs afterwards (was 1 of 5).
2. `fully_shard` overlap uses 4 gather slots and waits for the release of the unit *after* the slot's
   last reader (`TTML_FSDP_SLOTS`, `TTML_FSDP_SLACK`; `TTML_FSDP_SLACK=drain` is the old behaviour).
   The CCL queue can run two units ahead again.
3. Collectives issued on queue 0 (tensor-parallel all-gathers inside a block, the vocab-parallel
   loss, the backward of `scatter`) run on the compute sub-device, not the CCL sub-device: with two
   queues they were launched on the same cores as the FSDP collectives from queue 1. The staging
   cache is keyed by queue as well. This is the TP x FSDP drift (step-1 shift, run-to-run differences).

Measured after the three fixes (profiler off, mean of steps 3-12; losses vs the no-overlap run):

| config | no overlap | drain design | slots + slack | losses |
|---|---|---|---|---|
| TinyLlama FSDP8, 1 sample/device, 1 row | 460.5 ms | 425.1 | **420.6** (-8.7 %) | identical |
| TinyLlama FSDP32, 1 sample/device, 1 row | 482.9 ms | 444.6 | **439.0** (-9.1 %, DDP32 497.1) | identical |
| TinyLlama FSDP8, 5 samples/device, 1 column | 1645 ms | 1611 | 1614 | identical |
| TinyLlama FSDP8 mem-efficient, 1 sample, 1 column | 553 ms | 512 | 522 | identical |
| TinyLlama FSDP8 grad accumulation x2, 1 column | 3996 ms | 3940 (drifted 4 of 5) | 3941, **bit-exact 4 of 4** | identical |
| Llama-8B TP4 x FSDP8 mem-eff, 1 column | 1591 ms | 1570 (drifted every run) | 1571, **bit-exact 1 of 2** | one run drifts from step 1 |

The two-block lookahead is worth ~1 % on the collective-bound steps, not the ~4 % the events design
had shown before it was known to be corrupting data (part of that speed was never real). The
memory-efficient runner is 10 ms slower with slots+slack than with the drain (its recompute order
makes the slot wait fall back to a drain more often); to be looked at. Llama-8B TP4 x FSDP8 is
bit-exact in one of two runs: one TP-specific hazard is left. The harness does not reproduce it with
a TP-style all-reduce or a fresh-output all-gather on queue 0 next to the queue-1 gathers, with or
without address-recycling pressure (all clean over 48 blocks each), so it is something the real model
does that the harness does not: candidates are the vocab-parallel loss (all-gather + two all-reduces
on the TP axis), the backward of `scatter` (a reduce-scatter on queue 0 whose staging sets are now
keyed by queue) and the memory-efficient recompute order. `overlap_race_harness.py --tp 4` is the
place to add the next candidate.
