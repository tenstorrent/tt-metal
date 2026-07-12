---
name: tt-metal-model-perf
description: >-
  Optimize device performance of a ttnn/tt-metal model (Blackhole/Wormhole) by attacking layout and
  data-movement overhead, not just compute. Use when a model's tt-perf-report is dominated by
  InterleavedToSharded / ShardedToInterleaved, Untilize/Tilize, Slice/Concat, matmul row-chunking, or
  multi-chip collectives (AllGather/ReduceScatter/all_reduce), or when matmuls/convs run at low FLOPs
  efficiency, or by a single op (e.g. a large-output `gather`) tagged "Other". Distilled from optimizing
  the SeamlessM4T v2 CodeHiFi-GAN vocoder (device 286 ms -> 45 ms, ops 6303 -> 782, PCC 0.998), its
  NLLB-style tensor-parallel text encoder on a 4-chip mesh (seq=4096 device 106 ms -> 31 ms,
  ops 12653 -> 508, PCC 0.999), and its conformer speech encoder where one relative-position
  `ttnn.gather` was 82% of device time (mel=4096, 1x4: device 1.78 s -> 0.47 s, ~3.8x via band-windowing).
---

# Optimizing ttnn model device performance (layout & sharding)

## Core mental model

On Tenstorrent, a "compute" op (conv/matmul) is usually a **small** part of wall time. Most device time
goes to **glue**: sharding activations into L1 (`InterleavedToSharded`), unsharding back to DRAM
(`ShardedToInterleaved`), tile<->row-major conversion (`Tilize`/`Untilize*`), and `Slice`/`Concat`.
Activations "live" in DRAM between ops; each op reads DRAM -> shards into L1 -> computes -> writes DRAM.
**The optimization target is that round-trip, not the matmul.**

Read the `tt-perf-report` stacked report and bucket ops by category (Compute / DM / TM). If Compute is a
minority, the win is in DM+TM. The `(in0:dram_interleaved)` tag on an op means its input sits in DRAM.

On a **multi-chip mesh (tensor-parallel)** add two more first-class buckets: **collectives**
(`AllGather`/`ReduceScatter`/`all_reduce` after every row-parallel layer) and **matmul-chunking glue**
(per-chunk `Slice`/`S2I`/`I2S`/`Concat` when a big matmul is sliced to fit L1). On the text encoder these
were ~50% and ~20% of device time — larger than the matmul itself. Bucket them before touching compute.

## Workflow (repeat per bottleneck)

1. **Profile** (warm cache — see gotcha): `python -m tracy -p --op-support-count 100000 -r -m pytest <test>`
   then `tt-perf-report <generated ops_perf_results.csv>`. Read the "Stacked report".
2. **Pick the biggest non-compute bucket.** Find where those ops come from by grepping the op-ID stream
   for the sequence around a hot op (e.g. `conv -> S2I -> Untilize -> I2S -> conv`).
3. **Find the ttnn mechanism.** Check the op's docstring/config for a better path (slice configs, output
   layout, sharded variants). `ttnn.<op>.__doc__` and the op's Config fields are the source of truth —
   don't assume from memory.
4. **Validate in isolation first.** Write a tiny standalone script: build the exact shape, run torch
   reference + the ttnn variant, `check_with_pcc`, and time it. Confirm PCC + speed before touching the model.
5. **Integrate, then verify end-to-end.** Run the model's PCC test, re-profile (device time + op count),
   and warm-bench wall-clock. Check multiple input shapes, not just one.
6. **Commit** one optimization at a time with the measured before/after.

## The layout/sharding playbook (what actually moved the needle)

- **Prefer device-native slicing over manual Python chunking.** If a tensor is too big for one op, don't
  slice it in Python (each chunk pays slice-in / slice-out / concat). `ttnn.conv1d` accepts
  `slice_config=Conv2dSliceConfig(Conv2dDRAMSliceWidth, num_slices=0)` and slices the timeline in DRAM
  itself, managing halo internally — one op instead of a loop. (286->141 ms here.)

- **Keep conv inputs in TILE, not ROW_MAJOR.** `TILE -> sharded` reshard is ~10x cheaper than
  `RM -> sharded` (RM->sharded tilizes on the fly). Also `ttnn.conv1d` **accepts interleaved TILE input
  directly** — a defensive `to_layout(ROW_MAJOR)` "untilize" before a conv is often unnecessary for
  conv-derived inputs (only embedding-derived inputs need it, due to a reshape-volume constraint). Skipping
  that untilize removed a whole bucket here (95->57 ms). Convert a stage input to TILE once and reuse it.

- **Chain convs in L1 with `Conv2dL1FullSliceConfig` + small `act_block_h`.** With both set together, a
  conv runs fully in L1 and **returns a sharded L1 output** that the next conv consumes directly — no
  DRAM `S2I`+`I2S` between them. `act_block_h_override=32` shrinks the circular buffer to fit L1 (the
  default/auto block is too big and OOMs). reshape and eltwise preserve the sharding. (57->45 ms.)

- **`[B,T,C] <-> [B,T,1,C]` reshape is free on ROW_MAJOR but a full relayout on TILE.** Match the layout
  to the reshape (e.g. untilize once, ~0.17 ms, so a per-stage reshape becomes a free view instead of ~2.7 ms).

- **Build tensors directly in the layout the next op wants.** Assembling a tensor channel-major then
  permuting back to NLC costs 2 permutes + layout churn; concatenate on the channel dim in NLC instead.

- **Size DRAM slices by an element budget, not a fixed count.** A fixed slice count (e.g. 128) makes many
  tiny slices, each paying its own PaddedSlice/SliceWrite/Halo. Size each slice to a constant
  `rows*channels` budget so a few large slices fit L1 — far fewer per-slice ops. (Cut op count ~3x here.)

- **Hoist redundant elementwise.** If N blocks consume the same stage input and each starts with the same
  `leaky_relu(h)`, compute it once and share it.

- **Fidelity is rarely the lever for DM-bound models.** HiFi2/LoFi save little when the bottleneck is
  layout, and cost PCC margin. Leave conv/matmul/SDPA fidelity high unless the profile is genuinely
  math-bound. (Text-encoder SDPA HiFi3->HiFi2 bought ~1 ms and ate PCC margin; LoFi failed the gate.)

### Tensor-parallel / multi-chip (mesh) additions

- **Run block-sharded TP matmuls single-shot, not row-chunked.** Slicing the M (token) dim into small
  fixed chunks to fit an L1 "microbatch" tanks block-sharded FLOPs efficiency (79% -> 26% here) and adds
  a per-chunk `Slice` + `S2I` + `I2S` and a final `Concat`. Set the chunk >= the design-max sequence so
  the linear is one matmul (`num_chunks==1` skips the concat); chunk only above what L1 actually holds.
  This was the single biggest win (106 -> 58 ms) and it's often a one-line default someone set too small.

- **Tune the TP collective: `num_links` to the fabric max + `Ring` topology.** `all_reduce`
  (= `ReduceScatter` + `AllGather` after each row-parallel layer) is frequently the top bucket on a mesh,
  and the defaults are conservative (1 link, `Linear`). Raise `num_links` to the chip's ethernet-channel
  count and try `Ring`. (Collective bucket 30.8 -> 10.4 ms; 2 links was the BH-QB max.)

- **Chain consecutive block-sharded matmuls in L1 when their shard layouts match.** If matmul A's output
  block-shard layout equals matmul B's `in0` layout (same grid + shape, e.g. `fc1 -> fc2`), have A keep
  its sharded output and B consume it directly — skips the `S2I -> DRAM -> I2S` round-trip between them.
  (TP analog of keeping `conv1 -> conv2` sharded; -2.6 ms here.)

- **Re-examine DRAM-residency policies after cutting the transient op set.** A "spill residuals to DRAM
  above N rows" rule adopted under a *chunked* matmul regime can be obsolete once the matmul is
  single-shot: L1 may now hold the residual, making every reshard L1-local (`I2S` reads `l1_` not `dram_`)
  and sweeping the residual adds / all-reduce / LN into L1 too. (35 -> 31 ms here, pure data-movement.)

### Gather / relative-position bias (a single op can be the whole model)

- **`ttnn.gather` on a large output is pathologically slow — attack its output *width*, not fidelity.**
  The conformer relative-position bias `rel[b,h,q,k] = q_scores[b,h,q, idx[q,k]]` was one `ttnn.gather`
  producing `[B,H,Q,S]` — **82% of device time** at S=4096 (7.5 ms/call, ~90x off DRAM bandwidth,
  tagged "Other" not even "DM"). `ttnn.gather`'s output shape equals the *index* shape and does not
  broadcast, so the only lever is a smaller index. Bucket it like any other op and shrink the volume.

- **Clamped relative distance ⇒ band-window the gather.** When the distance index is
  `clip(k − q, −L, +R)` (Transformer-XL / conformer style, here L=64 R=8 → 73-entry vocab), it is
  **constant** outside the diagonal band `k ∈ [q−L, q+R]`: keys far left all map to `idx=0`, far right
  to `idx=vocab−1`. For a query block `[q0,q1)` gather only the tile-aligned key window
  `[q0−L, (q1−1)+R]` (~`Qc+L+R` cols, ~600 vs 4096 → **13x smaller gather**, 7.6 -> 0.58 ms) and fill
  the two constant regions from the two clamp columns of `q_scores`. Build the windowed index the same
  way as the full index, just restricted to the window keys — bit-exact (validate `max_abs_err=0`
  against the full gather in isolation before integrating).

- **Rebuild the constant regions with a width-1 broadcast add, never `ttnn.repeat`+`concat`.** First
  attempt materialized the full-width bias via `repeat` of each clamp column then `concat` — the
  `repeat` (+ a `tilize` it induced) became the *new* #1 op (185k µs > the shrunk gather). Instead add
  the bias **directly onto `scores` region-by-region**: a real add on the window slice, and
  `ttnn.add(scores_slice[...,N], col[...,1])` (a width-1 broadcast, ~0.16 ms) on each constant slice.
  No full-width tensor is ever built. (speech encoder total 1.78 s -> 0.64 s -> **0.47 s**.)

- **Softmax is invariant to a per-row constant added across the *full* row.** If the bias' left/right
  regions are per-row constants, you can drop the *larger* region entirely (it contributes a uniform
  shift that cancels in softmax), leaving only the window gather + the smaller region. Cheaper still,
  but **not bit-exact** — gate on PCC, especially on precision-sensitive paths.

## Verification discipline

- Gate every change on the model's **PCC test** (numerical parity vs the torch reference). Keep a margin.
- Measure device time from `tt-perf-report` **device-time-sum** and **op count** — not the op-to-op gap.
- Measure wall-clock with a **warm-cache, device-synced** micro-bench (load real weights, one warm forward,
  then median of N synced forwards). Cold runs include JIT compile and mislead.
- Re-check **multiple input shapes** — layout budgets and shard specs can behave differently per shape.
- **A provably bit-exact op change can still move end-to-end PCC — and the cause is often an op
  aliasing its own input.** The band-windowed rel bias was `max_abs_err=0` vs the full gather in
  isolation, yet full-model PCC dropped 0.9924 -> 0.9872. Root cause: `ttnn.gather` **clobbers/aliases
  its input buffer** in some DRAM-allocation states, so slicing two columns out of `q_scores` *after*
  `ttnn.gather(q_scores, …)` read corrupted values — but only for the later query blocks (once DRAM
  reached the triggering state), which is exactly why a single-gather isolation test was bit-exact.
  **Rule: read anything you still need from an op's input tensor BEFORE calling that op** (gather,
  scatter, and other in-place-capable ops can overwrite their input). Bisect a "bit-exact but PCC
  moved" regression by dumping intermediates (post-stack -> per-layer -> per-sub-block -> per-tile/row
  block) until the first divergence; a difference that is *uniform across one axis* points at a wrong
  broadcasted constant, not at compute.

## Gotchas / dead-ends (don't relitigate)

- **op-to-op gap is mostly cold-compile, not steady state.** The first profiled run after any code change
  JIT-compiles new kernel configs (can be tens of seconds); re-running warm collapses it (~2 s here). To
  compare two versions by gap you must warm both. The true steady-state gap is `warm wall - device time`
  (host dispatch), reduced by cutting op count and eliminated by Metal tracing.
- **Circular-buffer L1 limit is hardware** (~1.5 MB/core on BH). You can't raise it; you shrink the conv CB
  via `act_block_h`. `l1_small_size` is a *reserved* sub-region — raising it leaves *less* for CBs.
- **L1-interleaved input OOMs the conv CB** (the resident input counts against the CB budget). Use
  `Conv2dL1FullSliceConfig` + small `act_block_h`, or feed DRAM input. Never `from_torch(..., L1)` a
  >~10 MB tensor directly — it hard-crashes the driver (not a catchable exception).
- **`ttnn.conv1d` always returns interleaved DRAM output** unless run L1-full; `keep_sharded_output` alone
  doesn't keep it sharded (a later reshape/S2I un-shards it).
- **`conv_transpose2d` only supports `dram_slice_config`** (no L1-full) and its output is upsampled
  (bigger) — its DRAM slicing is inherent.
- **Sharded `layer_norm` needs width/block sharding, not height** — mixing with height-sharded convs
  forces a reshard that negates the gain.
- **Sharded `add` needs identical shard specs** on both operands; reshard the residual to match, but that
  reshard can cost more than a DRAM add saves — measure before adopting.
- **Metal tracing** removes the host-dispatch gap but requires fixed shapes; handle varying input lengths
  by bucketing (round the driving length up to a step) so a few traces cover all inputs. A trace/prewarm
  path that prepares conv weights per-config must be updated when conv modes change, or it hangs.
- **`num_links` is capped by the fabric's ethernet channels between chips.** Exceeding it is a hard
  `TT_FATAL` "link index out of bounds", not a graceful fallback — sweep upward to find the cap (2 on
  BH-QB 1x4). More links ~= proportional collective speedup up to that cap.
- **A single-shot block-sharded matmul returns 4D `[1, 1, m, n]`.** A downstream op that reshapes by
  `x.shape[0], x.shape[1]` (assuming `[batch, seq, ...]`) then gets the wrong volume and `TT_FATAL`s on
  `new_volume == old_volume` — make the output reshape rank-robust when chaining sharded matmul outputs.
- **Verify which implementation a profile came from before trusting it.** Reconcile op codes/counts
  against the current code (a provided report can be a stale or different variant — different chunk size,
  layer count, or fidelity). Here the two given reports differed only by a chunk-size default.
- **Watch the PCC-parse regex in sweep scripts.** A too-narrow grep silently reported "0.9999" when the
  test's real number was 0.9992 — always parse the test's own `PCC @ ...` log line, not a stray match.
