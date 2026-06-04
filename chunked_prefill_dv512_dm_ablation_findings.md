# Ring Joint SDPA — Chunked-Prefill d_v=512 Data-Movement Ablation

Branch: `skrstic/ring_joint_sdpa_chunked_perf_sweeps`
Date: 2026-06-03
Hardware: **P150_X4 Quiet Box (QB)** — 4× Blackhole p150b, **11×10 compute grid → 100 SDPA cores** (col 10 = CCL).
Test: `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_create_chunked_perf_table`

## What this measures

The **25.6K + 2.56K** chunked-prefill step for a kimi-style **latent-V MLA** shape with a **large value head dim
(d_v=512)**, profiling only the last/largest chunk (`CHUNKED_ONLY_LAST_CHUNK=1`): a Q chunk attending over a
~25.6K-token K/V prefix.

Single shape: **sp=4 ring-of-4 (tp=1), latent V, heads/device=14, per-device seq=640, q_chunk=32, k_chunk=640,
d_q=576, d_v=512.** 524.08 GFLOPs/chunk, 100 cores. Perf-only (`CHUNKED_SKIP_PCC=1`; DM-off variants run on stale
L1 → garbage output, timing valid).

We isolate the data-movement components on the SDPA cores:
- **K download** — the injector's K DRAM read (`fetch_block(k_gen, …)` in `ring_joint_reader.cpp`).
- **K mcast** — the intra-device store-and-forward broadcast of K (`ChainLink::forward` in `chain_link.hpp`).
- **Q read** — the per-q-chunk Q DRAM read (`read_block(q_gen, …)` in `ring_joint_reader.cpp`).
- **everything else** — latent-V rematerialization (K^T → V, L1→L1), output writeout, causal-mask reads.

In every ablation the **semaphore / CB / barrier handshakes are kept intact** — only the bulk NoC data transfer is
removed — so the pipeline never deadlocks.

## Results

| Variant | Duration | Math Util | Tracy FPU |
|---|--:|--:|--:|
| **DM on** (full kernel, baseline) | 3.190 ms | 59.4% | 31.1–31.2% |
| − K read (download)               | 2.922 ms | 64.9% | 34.0–34.1% |
| − K mcast                         | 2.929 ms | 64.7% | 33.9–34.0% |
| − K read + mcast                  | 2.911 ms | 65.1% | 34.1–34.2% |
| − K read + mcast + Q read         | 2.905 ms | 65.3% | 34.2–34.3% |
| − **all** DM (compute ceiling)    | 2.881 ms | 65.8% | 34.5–34.6% |

Total DM overhead = 3.190 → 2.881 ms = **0.309 ms (9.7% of wall time)**.

### Decomposition of the 0.309 ms DM overhead
| component | cost | share of DM |
|---|--:|--:|
| **K** (read + mcast) | 0.279 ms | **90%** |
| **Q** read | 0.006 ms | 2% |
| **everything else** (V-remat + output write + mask) | 0.024 ms | 8% |

## Conclusions

1. **K is essentially the entire exposed data-movement penalty (~90%).** Q reads cost ~2%, and the remaining NoC
   traffic — latent-V rematerialization, output writeout, causal mask — together cost only 0.024 ms (<1% of wall
   time).
2. **K download and K mcast are NOT additive — they are the same serial critical path.** Removing *either* alone
   recovers ~94–96% of the K gap; removing *both* adds only a final ~0.02 ms. Per K-chunk the injector must
   `download K → mcast K → compute`, and the FPU idles waiting for K (SrcB) whenever it outruns this single
   serial producer. Break either link and the remaining transfer hides behind the matmul.
3. **The d_v=512 DM footprint is much smaller than d_v=128** (9.7% of wall time here vs 26.8% at d_v=128, same
   q32/k640/sp4 shape). The 4× larger V matmul means more compute per K chunk, so the single-injector K cadence is
   hidden far better — same mechanism, smaller exposure.
4. **Lever:** for this geometry optimize the K-arrival chain (read+mcast K once and reuse — the k-stationary
   reorder, or pipeline download‖mcast so the producer cadence drops below the matmul), not Q or V.

## Reproduce

### Hardware / config note
This branch's chunked perf sweeps target the **8-chip P150_X8 cube** (`sp=4 → tp=2`, `sp=8 → tp=1`; both need 8
devices). To run on a **4-chip QB**, three TEMP overrides are required in `test_ring_joint_sdpa.py`:
- `tp_size = MESH_CONFIG.num_devices // _CHUNKED_SP_SIZE` (QB sp=4 → tp=1, a real ring of 4; the branch default
  `tp=2` needs 8 chips and fatals at `control_plane.cpp:1264`).
- `grid_cols = 11` (QB is an 11-wide grid; the branch default `12` puts the CCL column on a dispatch core).
- `CHUNKED_PREFILL_HEADS_PER_RING = 14` (branch default `16`; 14 balances across the 100-core grid far better —
  16 leaves cores padded/idle).

Note the env-derived parametrization ID: with `sp=4` + `per_device=640` the chunk label is **2560** (= 640×4), so
the test ID is `kimi50k-q32-k640-chunk2560` (a `chunk5120` ID would require `per_device=1280` or `sp=8`).

### Baseline run (DM on)
```bash
source python_env/bin/activate
ENV='CHUNKED_SP_SIZE=4 CHUNKED_PER_DEVICE_CHUNK=640 CHUNKED_Q_CHUNK=32 CHUNKED_D_V=512 \
     CHUNKED_LATENT_V=1 CHUNKED_ONLY_LAST_CHUNK=1 CHUNKED_SKIP_PCC=1'
TEST='tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_create_chunked_perf_table[kimi50k-q32-k640-chunk2560]'
env $ENV scripts/run_safe_pytest.sh "$TEST"
# read off "Duration (ms)" / "Math Util" from the per-chunk table for chunk 10.
```

### Compute-ceiling run (all DM off) — no kernel edits
The program factory gates a no-op DM mode on an env var (`noc_dm_gate.hpp` rewrites every bulk NoC read/write to
`((void)0)` while keeping all CB/semaphore handshakes). Just prepend it:
```bash
env TT_RING_JOINT_DISABLE_NOC_DM=1 $ENV scripts/run_safe_pytest.sh "$TEST"
```

### Per-component variants (edit the dataflow kernels, then re-run; JIT recompiles on source change)
Back up first: `cp ring_joint_reader.cpp /tmp/ && cp chain_link.hpp /tmp/`
(dir: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/`)

- **without K read** — in `ring_joint_reader.cpp`, comment out the injector K read
  `fetch_block(k_gen, k_slice, end_seq_tile, cb_k_start_address, k_tile_bytes, true /*transpose*/);` (add `(void)k_gen;`).
  `fetch_block` does *not* push the CB, so commenting it keeps the handshake — no hang.
- **without K mcast** — in `chain_link.hpp` `ChainLink::forward`, comment out both transfer calls
  `noc_async_write_multicast(...)` and `noc_async_write(...)` (keep the surrounding `noc_semaphore_*` /
  `noc_async_writes_flushed`).
- **without Q read** — in `ring_joint_reader.cpp`, the Q path uses `read_block(q_gen, …)`, which bundles
  `cb_reserve_back + fetch_block + cb_push_back`. Replace each `read_block(...)` call with the handshake only
  (drop the data read), e.g.:
  ```cpp
  const uint32_t _qn = q_slice.get_d2_size() * q_slice.get_d3_size();
  cb_reserve_back(cb_q_in, _qn);
  cb_push_back(cb_q_in, _qn);
  ```
  (do this for both the `use_q_subblock_push` loop and the full-slice `else` branch; add `(void)q_gen;`).
  Commenting the whole `read_block` instead would drop the CB push and **hang** compute on `cb_wait_front(cb_q_in)`.
- **combinations** — apply the edits together for the cumulative variants in the table above.

Always restore pristine kernels afterward (`cp /tmp/ring_joint_reader.cpp …`, `cp /tmp/chain_link.hpp …`) and verify
a clean `git diff` on the two files.

### Other data movement (for completeness)
Beyond K-read / K-mcast / Q-read, the remaining NoC traffic on the SDPA cores is:
- **Latent-V rematerialization** — with `CHUNKED_LATENT_V=1`, V is not read from DRAM or mcast; the reader
  transposes K^T → V from K's own L1 buffer (`materialize_v_prefix_from_k`, `noc_async_read_one_packet_with_state`,
  L1→L1). The DRAM-V-read + V-chain paths exist but are statically dead in this config.
- **Output writeout** — the writer streams `cb_out` → DRAM per q-chunk.
- **Causal mask reads** — mask tiles into `cb_mask_in`.

These together account for the 0.024 ms gap between the "− Q read" and "all DM off" rows — negligible on the
critical path.

## Optimization experiments (2026-06-03, follow-up)

Goal: close the 0.309 ms / 6.4-pt DM gap and approach the 2.881 ms / 65.8 % compute ceiling.

### Loop geometry for this config (measured)
`q_local_padded_N = 640` (NOT 32 — q_chunk=32 is only the tile granularity), so **num_q_chunks = 20**,
`all_heads_num_q_chunks = 1·14·20 = 280`, `max_q_per_core = ceil(280/100) = 3`. Inner geometry: ring_size=4,
`iter_num_kv_chunks ≈ 11`, **q_per_core ≈ 3**. So each core processes ~3 Q chunks, and because NHK=1 the K slice
is **identical across the 3 q_iters** → the reader DRAM-reads + mcasts each K chunk **3× redundantly per ring_iter**.

### Timing-only ablations (kernel JIT edits, perf valid / output garbage)
| Variant | Duration | Math Util | vs baseline |
|---|--:|--:|--:|
| baseline (q32) | 3.191 ms | 59.4 % | — |
| **skip redundant K DRAM read** (fetch only q_iter==0) | **2.999 ms** | **63.2 %** | **−0.192 ms (62 % of gap)** |
| skip redundant K read + mcast (gate both on q_iter==0) | 2.999 ms | 63.2 % | −0.192 ms (no extra) |
| compute ceiling (all DM off) | 2.881 ms | 65.8 % | −0.310 ms |

**Exact ablation edit** (timing-only; output is garbage because the L1 buffer holds stale K on q_iter>0, but the
NoC traffic — hence timing — matches a true k-stationary reorder). In `ring_joint_reader.cpp`, the K block at
~L509-538 inside `for q_iter … for k_chunk …`, gate the whole receive/fetch/forward on `q_iter == 0`:
```cpp
// skip-redundant-K-read+mcast: K slice + mcast are identical across q_iter (NHK=1).
if (q_iter == 0) {
    if (k_chain.should_receive(nb, nq)) { k_chain.receive(); }
    else { /* pick k_gen as before */ fetch_block(k_gen, k_slice, end_seq_tile, cb_k_start_address, k_tile_bytes, true); }
    if (k_chain.should_forward(nb, nq, q_iter_local)) { k_chain.forward(cb_k_start_address, k_chunk_tiles, k_tile_bytes); }
}
```
Gating BOTH `receive` and `forward` on the same uniform condition keeps the chain semaphores balanced (no hang).
For read-only, gate just the `fetch_block` line with `if (q_iter == 0)` (add `(void)k_gen;`). `cb_reserve_back` /
`cb_push_back` stay unconditional so compute still gets fed. Run with the baseline command; read chunk-10 row.

**The redundant K DRAM read is the entire exposed redundant cost.** Removing it alone recovers 62 % of the DM gap;
the redundant mcast then hides behind the (4× larger, d_v=512) V matmul and adds nothing more. → the **k-stationary
loop reorder** (read+mcast each K chunk once per ring_iter, loop the 3 q's inside) is the principled lever, ceiling
**≈ 2.999 ms / 63.2 %**.

### Larger-q_chunk alternative (attacks BOTH redundancy and the 65.8 % ceiling)
A bigger q_chunk shrinks `q_per_core` (less K-read redundancy) AND enlarges the matmul (`Sq_chunk_t` 1→2→4), which
should raise the compute ceiling itself. Same math → same PCC; q_chunk is pure tiling. But it OOMs: **q64 needs
1.83 MB > 1.57 MB L1** (over by 254 KB). Root cause: the latent-V K CB is over-allocated **3-deep**
(`k_tiles = Sk_chunk_t·DHt·3 = 1080` tiles ≈ 1.17 MB of L1, bf8_b). Reclaiming the 3rd slot (≈383 KB) is enough to
fit q64. Made `CHUNKED_K_CB_DEPTH` env-tunable (default 3) in the program factory to A/B this.

**Refuted.** The 3rd K-CB slot is essential — it pipelines the latent-V rematerialization:
| Variant | Duration | Math Util |
|---|--:|--:|
| q32, K-CB depth=3 (baseline) | 3.191 ms | 59.4 % |
| q32, K-CB depth=2 | **3.759 ms** | **50.4 %** |
| q64, K-CB depth=2 (fits) | 3.862 ms | 49.1 % |

Dropping depth 3→2 to make room for q64 costs −9 pt (V-remat can no longer overlap compute), far more than the
larger matmul recovers. And depth=4 cannot fit (K alone = 1440·1088 B = 1.57 MB ≈ all of L1). So this config is
**boxed in at q32 / K-CB depth=3** — the only lever left is the **k-stationary loop reorder** (kernel-only, no build,
proven ceiling 2.999 ms / 63.2 %).

### Mechanism: contention, not serial stall (why pipelining download‖mcast does NOT help)
Prior device-zone / dose-response work (see memory `project_chunked_sdpa_operand_wait_proof`) proved the slowdown is
the FPU's **unpacker stalling on SrcB=K** because concurrent K NoC↔L1 writes contend for the **shared L1 arbiter**
(not bank conflicts; bubble scales with NoC round-trip). The implication is decisive: the penalty is set by *total*
K NoC traffic into L1, not by the read→mcast serial latency. So the doc's other suggested lever — *pipeline
download‖mcast* — reschedules the same bytes and would **not** reduce arbiter contention. Only **reducing K traffic**
helps, which is exactly what the k-stationary reorder does (read+mcast each K chunk once per ring_iter instead of
`q_per_core`× ). This is consistent with the ablation: removing the redundant *read* alone captured 62 % of the gap.

### Why the reorder is L1-blocked here (exact budget)
Measured static CB footprint at the target config (`[CB_L1_PROBE]`):

```
total_static_cb_bytes = 1,451,536 B  of  1,572,864 B   →  ~118 KB free
  cb_k_in (latent-V, 3-deep) = 1080 tiles × 1088 B = 1,175,040 B  (81 % of all CBs!)
  everything else                                   ≈   276,496 B
Sq_chunk_t=1  Sk_chunk_t=20  DHt=18  vDHt=16  num_q_chunks=20  max_q_per_core=3
```

A k-outer/q-inner reorder must keep **all `q_per_core`=3 Q chunks' online-softmax accumulators live across the
K-loop** (each K chunk touches every Q before release). The accumulator is `{out_im (Sq·vDHt=16 tiles), max (1),
sum (1)}` per Q. Today there is **one** Q's worth, double-buffered (cb_out_im_A/B = 32 tiles). Three Qs need either:
- **ping-pong** (6×16 = 96 tiles): Δ ≈ +64 out_im tiles ≈ **+131 KB → exceeds the 118 KB free (OOM)**, or
- **single-buffered** (3×16 = 48 tiles): Δ ≈ +16 tiles ≈ **+33 KB → fits**, but requires rewriting the SALAD online
  rescale to update each Q's accumulator in place (no prev/cur swap).

So the reorder is **L1-marginal**: not landable with the simple ping-pong restructure; it needs single-buffered
per-Q accumulators (delicate Phase-2 rewrite) **or** a small `k_chunk` reduction to free L1 (which the dv512 sweep
shows costs util). Either is a scoped, multi-session, hang-prone 2-sided (reader↔compute) change — not a quick edit.

## Conclusion (follow-up)

1. **Root cause is confirmed and quantified:** the exposed DM penalty (0.31 ms / 6.4 pt) is almost entirely the
   **redundant K DRAM read** — K is re-fetched `q_per_core`≈3× per ring_iter because NHK=1 makes the K slice
   identical across the 3 Q chunks a core owns. Removing it (timing-only) recovers **0.192 ms → 2.999 ms / 63.2 %**
   (62 % of the gap to the 2.881 ms / 65.8 % compute ceiling).
2. **The principled fix is the k-stationary (k-outer/q-inner) reorder**, which reads+mcasts each K chunk once. The
   mechanism analysis rules out the cheaper download‖mcast pipeline (contention-bound, not latency-bound).
3. **It is L1-blocked at this config:** only ~118 KB free, and 3 ping-pong accumulators need ~131 KB. Landing it
   requires single-buffered per-Q accumulators or a k_chunk reduction — a scoped multi-session change.
4. **Refuted cheaper levers:** larger q_chunk (OOM; K-CB-depth-2 reclaim craters perf −9 pt), deeper K CB (OOM),
   download‖mcast pipeline (wrong mechanism). The config is genuinely boxed.

## Scoped plan to land the k-stationary reorder (the actual fix; multi-session, hang-prone)

Target: read+mcast each K chunk once per ring_iter; loop the `q_per_core`=3 Q chunks inside. Expected
**~2.999 ms / 63.2 %** (the proven ablation ceiling). This is a 3-sided reader↔compute↔writer change because the
current q-outer streaming order is baked into the accumulator ping-pong, the DRAM-staging "dual-write from DST" /
"planting" tricks, and the `KV_chunks_processed_in_iter % 3` dummy-KV-pop chain phase-alignment
(`compute_streaming.hpp:2020`).

1. **L1 first.** In the reorder, compute spends 3× longer per k_chunk, so the latent-V V-remat 3rd K-slot is likely
   droppable → set the latent K-CB to **depth 2** (frees ≈392 KB), then add **3 single-buffered per-Q accumulators**
   `{out_im 16t, max 1t, sum 1t}` (≈ +50 KB). Net L1 comfortably positive. (`CHUNKED_K_CB_DEPTH` env knob was added
   then reverted; reinstate if useful.) Verify with the `[CB_L1_PROBE]` instrumentation (also reverted — re-add a
   `dbg_total_cb_bytes` accumulator in `allocate_cb`).
2. **Reader** (`ring_joint_reader.cpp`): swap to `for ring_iter → for k_chunk → for q_iter`; do receive/fetch/forward
   once per (ring_iter,k_chunk); push K once; loop q_iter for Q-read (k_chunk==0) only. Keep the chain handshake
   once-per-k_chunk; re-derive the padded-iter mcast-sync.
3. **Compute** (`sdpa_ring_v2`): invert the `for q … for k_chunk …` (L1707/L1766) to `for k_chunk … for q …`; index
   the 3 per-Q accumulators by `q`; rewrite SALAD Phase-2 rescale to update each Q in place (no prev/cur swap);
   re-derive the per-(k_chunk,q) causal limit / `q_start_tile` / `per_q_valid_kv`; rework the dummy-KV-pop alignment.
4. **De-risk:** gate the whole new path behind `chunked_enabled && !is_balanced && NHK==1` (this config) so the
   non-chunked / balanced / joint paths are untouched. Validate against the accuracy test (unset `CHUNKED_SKIP_PCC`)
   until PCC ≥ 0.99, expect several device-reset cycles (CB push/pop imbalance = hang). Then measure perf.

State at end of this session: **tree clean** — all kernel/factory experiments reverted; only this `.md` and the
pre-existing QB test override (`test_ring_joint_sdpa.py`) remain.

## Tracy device-zone campaign — WHERE the penalty lands (measured, not inferred)

Added one per-step zone at a time (marker budget = one per RISC), ran baseline (DM on) vs compute-ceiling
(`TT_RING_JOINT_DISABLE_NOC_DM=1`), paired ZONE_START/END per (core, step) on the steady run-host, CHIP_FREQ=1350 MHz
→ µs = cycles/1350. Calibrated against the whole-kernel `TRISC-KERNEL` zone (3157 µs ≈ the 3.19 ms wall). The
inner step runs ~117–125×/core.

### 1. Whole inner step (`INNER_STEP`, all 3 TRISC are lock-stepped, ~identical)
| | median | mean | p90 | p99 | max | footprint/core |
|---|--:|--:|--:|--:|--:|--:|
| baseline (DM on) | 23.0 µs | 23.8 | 26.2 | 33.8 | ~48 | ~2985 µs |
| ceiling (DM off) | 21.65 µs | 21.7 | 21.7 | 22.7 | ~23 | ~2710 µs |
| **Δ** | **+1.35** | +2.1 | +4.5 | **+11** | **+25** | **+270** |

The penalty is **NOT a uniform slowdown** — the **median barely moves (+1.35 µs, ~6 %)** while the **tail explodes**
(p99 +11, max +25 µs). Mean Δ (+2.1 µs) × 117 steps ≈ +250 µs ≈ the wall penalty (310 µs). So it's a *small uniform
component* + a *fat tail of slow steps*.

### 2. K gate (`cb_wait_front(cb_kt_in)`, on TRISC_0 = unpacker)
| | median | p99 | max | footprint/core |
|---|--:|--:|--:|--:|
| baseline | 1.14 µs | 6.75 | 17.35 | **177 µs** |
| ceiling | 0.03 µs | 0.03 | 0.03 | **4 µs** |

**Δ = +173 µs of genuine unpacker stall on K** — i.e. **56 % of the wall penalty / 64 % of the step penalty is the
unpacker hard-blocking at the K gate.** The tail shape matches INNER_STEP's tail exactly: the slow steps are the ones
where compute waits for K. (Note: this *differs* from d_v=128, where the K-gate wait was ≈0 — at d_v=512 the stall is
real and directly measured.)

### 3. V gate (`cb_wait_front(cb_v_in)`, TRISC_0): **flat — 4 µs baseline = 4 µs ceiling.**
Compute **never** stalls on V; the latent-V remat always keeps up. V is not on the critical path.

### 4. K producer (`K_PRODUCE` = download+mcast per chunk, reader NCRISC)
| | median | p99 | max | footprint/core |
|---|--:|--:|--:|--:|
| baseline | 18.0 µs | 21.2 | 24.3 | **2274 µs** |
| ceiling (no-op) | 1.15 µs | 2.96 | 3.14 | 146 µs |

The producer spends **18 µs/chunk** moving K = **78 % of the 23 µs compute step**, and **2128 µs of K NoC traffic per
core** over the kernel. Because `q_per_core`=3 it repeats this **~3×** (the K slice is identical across the Q chunks a
core owns). So the single-injector producer is **near-saturated** — its K work alone nearly equals the compute step,
and the redundant ×3 pushes it over on a tail of steps.

### Conclusion — why the kernel (59.4 %) is below the ceiling (65.8 %)
Two effects, **both caused by the redundant ×3 K traffic**, neither involving V:
1. **Hard K-gate stalls (+173 µs, the fat tail):** the lone injector's download+mcast (18 µs/chunk × 3 redundant)
   can't stay ahead of compute on a subset of steps → the **unpacker blocks at `cb_wait_front(cb_kt_in)`** (up to
   +17 µs on the worst steps) → the FPU idles.
2. **Uniform execution slowdown (+1.35 µs/step median, ~6 %):** even on non-stalled steps, the producer's K
   NoC→L1 writes contend with the unpacker's K→SrcB L1 reads on the shared L1 arbiter, stretching every step slightly.

Removing the redundant K traffic kills both: the skip-redundant-K-read ablation recovers **0.192 ms** (≈ the +173 µs
gate stall + part of the uniform term), landing at **2.999 ms / 63.2 %** — exactly what these zones predict. This is
the **k-stationary reorder** payoff; V, Q, and download‖mcast pipelining are all confirmed off the critical path.

## Profiling with Tracy device zones

The per-variant numbers above (`Math Util`, `Tracy FPU`) come from Tenstorrent's Tracy device profiler. Beyond the
aggregate util, you can instrument the kernels with **device zones** to see exactly where compute spends each inner
step (e.g. to confirm the FPU is idle waiting on K vs. genuinely busy). This is how the data-movement mechanism was
pinned down. Device zones are kernel-source edits only — they JIT-recompile, **no `./build_metal.sh`**.

### Enabling the profiler

The harness already runs under Tracy when the device profiler env var is set; the perf-table test enables it. Kernel
zones then land in the per-core device CSV with no rebuild:

```bash
export TT_METAL_DEVICE_PROFILER=1
```

Raw per-core markers land in
`generated/profiler/<subdir>/.logs/profile_log_device.csv`, with a persistent copy under
`generated/profiler/reports/<runDate>/profile_log_device.csv`. **Read from `reports/`** — the `.logs/` copy is
sometimes truncated to just a header by post-processing.

### Adding a device zone

Wrap the region of interest in a zone. The compute kernel (`compute_streaming.hpp`) already gates its zones behind a
`profiling_enabled` flag via the `MaybeDeviceZoneScopedN(profiling_enabled, "NAME")` macro; the dataflow kernels
(`ring_joint_reader.cpp`, `chain_link.hpp`) use plain `DeviceZoneScopedN("NAME")`. A zone records a `ZONE_START`/`ZONE_END`
marker pair each time control passes through it. Useful placements (lines on this branch):

| zone | file:line | wraps | what it tells you |
|---|---|---|---|
| whole inner step | `compute_streaming.hpp` `sdpa_inner_loop_step` body (~L773) | Q@Kᵀ + softmax + S@V + reduce | total compute time per step (incl. waits *inside* the step) |
| K gate | `compute_streaming.hpp` `cb_wait_front(cb_kt_in, …)` (~top of Phase 1) | just the K wait | how long compute blocks waiting for K — **≈0 means compute is never K-starved**, the penalty is execution slowdown, not a stall |
| V gate | `compute_streaming.hpp` `cb_wait_front(cb_v_in, …)` (Phase-2 drain) | just the V wait | same, for V |
| Q@Kᵀ matmul | `compute_streaming.hpp` `kt_subblock` loop (~L836) | only the Q@Kᵀ `blocked_matmul_and_pack` loop | localizes the per-step stretch to the matmul |
| K download | `ring_joint_reader.cpp` `fetch_block(k_gen, …)` | the injector's K DRAM read | time the reader pulls the K prefix from DRAM (injector cores only) |
| K mcast/forward | `chain_link.hpp` `ChainLink::forward` | the store-and-forward broadcast | time the reader is blocked forwarding K via the chain |

### The two binding constraints

1. **Marker budget — one per-step zone per RISC thread.** Each RISC thread has only
   `PROFILER_L1_OPTIONAL_MARKER_COUNT = 250` markers; one zone = 2 markers/execution. The inner step runs
   **~ring_size·q_per_core times per invocation** (here ~112×), so a single per-step zone (≈224 markers) just fits.
   **Two per-step zones on the same RISC silently truncate** at ~62 steps each — you only see the first ~half of the
   kernel, and every core reports a suspiciously identical round step count. Add **one** per-step zone at a time, run,
   then swap it for the next. Zones on *different* RISC threads (e.g. a compute TRISC zone + an NCRISC reader zone) are
   fine in the same run.

2. **One `DeviceZoneScopedN` per C++ scope.** `DeviceZoneScopedN`/`MaybeDeviceZoneScopedN` each declare a `hash`/`zone`
   local, so two in the *same* `{}` scope is a "conflicting declaration" **compile error** — and the harness then
   silently runs the stale cached binary (still reports util/dur, but no new zone data). Put each added zone in its
   **own `{}` block**. Always check the run log for `compile failure` and that the zone count in the CSV is non-zero
   before trusting any number.

### Reading the CSV

Run the baseline command above with the zone in place, then in
`generated/profiler/reports/<date>/profile_log_device.csv`:

1. **Filter to one `run host ID`** — the harness profiles multiple invocations into one CSV; pick the last/steady one.
2. **Pair `ZONE_START`/`ZONE_END`** per `(core_x, core_y, RISC processor type, run host ID, zone name)`; each pair's
   timestamp difference is one step's duration.
3. **Filter compute zones to `TRISC_1` (math)** — `DeviceZoneScopedN` emits on all three TRISC threads (unpack/math/pack);
   TRISC_1 is the math pacer. Reader zones are on `NCRISC`.
4. **Convert cycles→ns** with `CHIP_FREQ[MHz]` from the CSV header.
5. Useful aggregates: **per-step median/mean** over all (core, step) samples; **footprint / total per-core** = sum of a
   zone's durations on one core (≈ wall-time that core spent inside it); **% of kernel** = footprint ÷ kernel span.

A median near zero can hide a fat tail — for gate zones (`cb_wait_front`) also check p90/p99/p100 pooled over all
(core, step) samples before concluding "never starved."

## Data-movement speed-up campaign (2026-06-04) — the injector reader is the binding constraint

Goal this session: make K arrive faster (data-movement only, no compute reorder). New per-component zones on the
**injector** reader (NCRISC/NoC0) decompose the producer step:

| reader component (injector) | median/chunk | notes |
|---|--:|---|
| **K DRAM download** (`fetch_block`) | **12.3 µs** | 360 single-tile `noc_async_read_page`, transposed → strided dst (no coalescing). |
| **K mcast** (`ChainLink::forward`) | **5.8 µs** | of which **MC_SEMWAIT = 0.03 µs** (receivers always ready ⇒ producer-bound) + **MC_WRITE = 5.8 µs** (the 391 KB broadcast). |
| **V rematerialization** (`materialize_v_prefix_from_k`) | **4.86 µs** | 320 strided L1→L1 transpose reads (K^T → V). |
| **per-chunk total** | **≈ 23 µs** | **equal to the 23 µs compute step** ⇒ the injector reader is saturated and falls behind on the tail (the +173 µs K-gate stall). |

### The decisive correction: V-rematerialization is the #1 exposed lever, NOT the redundant-K read
A timing-only ablation removing **only** the V-remat L1 reads (CB handshake kept) lands at **2.978 ms / 63.6 %
(−0.212 ms / +4.2 pt)** — *larger* than the skip-redundant-K-read result (2.999 ms / 63.2 %). **This refutes §"everything
else (V-remat+output+mask) = 0.024 ms"** in the original decomposition: V-remat is squarely on the injector's critical
path because all three producer components serialize on the single NoC0 NIU, which is **bandwidth-saturated** (782 KB
read+write per chunk in 18.1 µs ≈ 43 GB/s aggregate; reads and writes share the NIU budget).

### Single-NoC levers — all measured, all dead
| lever | result |
|---|---|
| K-download `barrier_threshold` (throttle outstanding reads) | **worse**: thresh 0→3.190, 128→3.264, 64→3.396, 32→3.709 ms. 360 unthrottled reads is already optimal (latency hidden by depth). |
| mcast flush-defer (overlap mcast drain with V-remat) | **neutral** (3.187 ms) — proves read-VC ‖ write-VC do **not** parallelize on one NIU (BW-shared). |
| mcast valid-signal defer | **worse** (3.785 ms) — starves the 90 receivers; the valid signal is on their critical path. |

### The second NoC is the only spare bandwidth — and it is owned by the writer
NoC1 (BRISC/writer) is the only idle NIU (writer does ~0.024 ms of output). Two relocations attempted:
- **V-remat on NoC1 from the reader** (`noc_local_state_init(1)` + issue reads `noc=1`): **HANGS** — under `DM_DEDICATED_NOC`
  the writer owns `(noc1, RD_CMD_BUF=1)` (it reads prev-out/stats at `ring_joint_writer.cpp:48/54/77`); the reader
  sharing that cmd buffer corrupts it. Reader-side dual-NoC is impossible.
- **V-remat on the writer**: the writer loop is **Q-granular** (no K-chunk inner loop) and a deadlock minefield
  (cb_prev_out/cb_out/cb_signal cycles, TRID prefetch dance); a K-granular V-remat needs a reader→writer per-chunk
  handshake (new CB/semaphore) — the fused-AllGather arg/CB/semaphore-layout landmine that already hung the
  K-mcast-to-writer attempt (`d33553875fa`). Multi-session.

### Conclusion / the one path to the −0.212 ms
The injector reader is exactly balanced with compute, and its only relocatable component (V-remat, 4.86 µs) cannot move
off NoC0 without the writer's NoC1 (cmd-buffer-owned) or a compute change. The V-remat **transpose is intrinsic** and must
happen somewhere; the clean fix is to let compute's **S@V transpose V^T directly from the K^T buffer** (the first
`vDHt` rows of cb_kt_in ARE V^T, contiguous), exactly as Q@K^T already transposes K — eliminating the rematerialization
entirely. That is a contained compute data-sourcing change (S@V operand source + transpose flag + K^T pop lifetime),
measured ceiling **≈ 2.978 ms / 63.6 %**. Pure-data-movement is genuinely boxed here.

## Fewer K downloaders — group rows under one injector (2026-06-04)

The K-mcast pass in `ring_joint_sdpa_program_factory.cpp` builds **one mcast chain per logical row** → on the
10×10 SDPA grid that is **10 injectors**, each reading the full K prefix from DRAM and broadcasting along its own
10-core row. Aggregate DRAM K-read volume is therefore **10× the prefix** (every row re-reads the same K). Idea:
read K from DRAM fewer times by grouping rows under a shared injector (1 injector per 2 rows → 5 downloaders),
each mcasting to a **2-row, 20-core rectangle**. Implemented as a contained factory change (grouping the K-mcast
pass into `rows_per_group`-row bands, injector = max-work core across the band picked under the existing
FIFO-windowed column-exclusion, mcast rect spanning the band's physical rows, `mcast_num_dests` = band_cores−1).
No reader/compute kernel change — the reader is fully driven by the `ChainConfig` args (rect bounds + num_dests).
Gated behind an env knob `CHUNKED_K_MCAST_ROWS_PER_GROUP` (default 2) for A/B without recompiling.

**Correct & no hang.** PCC = **0.9994** at `rows_per_group=2`, and even the 10-row extreme (single injector
mcasting to all 100 cores) ran clean — so adjacent logical rows map to a contiguous physical rectangle (the
multi-row mcast is valid; this had been an open hang risk).

### Sweep (same build, perf-only, chunk-10 row)
| `rows_per_group` | K downloaders | mcast fan-out | Duration | Math Util |
|---|--:|--:|--:|--:|
| 1 (original per-row) | 10 | 1 row (10 cores) | 3.194 ms | 59.4% |
| **2** | **5** | **2 rows (20 cores)** | **3.163 ms** | **59.9%** |
| 5 | 2 | 5 rows (50 cores) | 3.218 ms | 58.9% |
| 10 | 1 | 10 rows (100 cores) | 3.338 ms | 56.8% |

### Findings
1. **5 downloaders is the optimum and a small real win: −0.031 ms / +0.5 pt (3.194 → 3.163 ms, 59.4 → 59.9 %)**
   — roughly **10 % of the 0.31 ms DM gap** to the 2.881 ms / 65.8 % compute ceiling.
2. **The curve is U-shaped — fewer than 5 is *worse* than the per-row baseline** (2 downloaders 3.218 ms, 1
   downloader 3.338 ms). With too few injectors the per-injector mcast fan-out grows and producer parallelism
   collapses: a lone injector serializing a broadcast to all 100 cores costs more than the saved DRAM reads.
3. **Why the win is only ~1 %, consistent with the contention mechanism.** Grouping halves the *aggregate DRAM
   K-read* volume (5 prefix-reads instead of 10), but it does **not** reduce the per-receiver K traffic into L1 —
   every core still receives the same K bytes via mcast. Since the measured bottleneck is per-receiver
   **L1-arbiter contention on K ingest** (set by total K→L1 traffic per core, not by injector count — see the
   operand-wait proof), only the small uniform DRAM-pressure component moves. This independently corroborates
   "contention-bound, not reader-count-bound": you cannot reach the ceiling by changing *who* downloads K, only by
   reducing *how much* K each core ingests (k-stationary reorder) or eliminating the V-remat transpose.

**Status: reverted.** The factory change was backed out after measuring; tree clean again. The knob is worth
re-adding if 5-downloaders' ~1 % is wanted as a free, correctness-preserving default, but it is not a path to the
compute ceiling.

## LANDED 2026-06-04 — V^T-from-K^T compute sourcing (the "one path" realized)

The §"one path to the −0.212 ms" (let compute's S@V read V^T directly from the K^T buffer, eliminating
V-remat) was **implemented and landed**. It is a pure addressing remap, not a transpose gamble: in latent
mode `cb_v_in == cb_kt_in` and `materialize_v` is a verbatim whole-tile copy, so `V[sk][vd]` is
byte-identical to `cb_kt_in[vd*KT_stride + sk]`. S@V now reads K^T at `vd*KT_stride` (one output col per
matmul, ct_dim=1, since vd is KT_stride-strided), the Phase-1 K^T pop defers to after Phase 2, and the
reader stops materializing/pushing the V entry. Gated `chunked_enabled && v_shares_k_buffer`.

| | Duration | Math Util | PCC |
|---|--:|--:|--:|
| baseline | 3.191 ms | 59.4 % | 0.99947 |
| **landed** | **2.991 ms** | **63.4 %** | **0.99947 (exact)** |

−0.200 ms / +4.0 pt, **beating the k-stationary reorder's 2.999 ms ceiling** with a contained kernel-only
change (no build, no fused-op layout churn). New compute ceiling (all DM off) = **2.950 ms / 64.3 %**, so
only **0.041 ms** of exposed DM remains → the reader is now hidden and the k-stationary reorder is retired.
ct_dim=1 raised the compute ceiling ~0.069 ms vs the old blocked V matmul (2.881→2.950); recoverable only
via a batched-vd DST matmul (needs dst_size ≥ vDHt) — marginal, not pursued. Patch: `/tmp/phaseA_vfromkt.patch`.
