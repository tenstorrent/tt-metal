# Ring joint SDPA — balanced+causal case (mla_100k q=160 k=320)

A knowledge doc for the **balanced + causal** ring joint SDPA path, walked
through with the `mla_100k q=160, k=320` config on a Blackhole 4-device ring as
the running example. Covers what the op does host-side, how the program factory
shapes the work, and what each of the three on-device kernels (reader / compute
/ writer) does per ring iteration.

All file paths are relative to repo root. Line numbers are point-in-time
(branch `skrstic/ring-joint-profiling`, commit `8d34835d601`); re-grep before
quoting in PRs.

---

## 0. Running config

`mla_100k` is DeepSeek MLA-shaped (q=160, k=320) and is the only
balanced+causal config used as the running example throughout this doc:

| Field                 | Value (mla_100k, ring_size=4 BH)                           |
|---|---|
| `b`                   | 1                                                          |
| `nhq` / `nhk` / `nhv` | 29 / 1 / 29 (BH; Galaxy uses 32 / 1 / 32)                  |
| `d_q` / `d_k` / `d_v` | 576 / 576 / 128                                            |
| `seq_len` (global)    | 3200 tokens                                                |
| `q_chunk` / `k_chunk` | 160 / 320                                                  |
| `q_dtype` / `kv_dtype`| bf16 / bf8_b                                               |
| `is_causal` / `is_balanced` | true / true                                          |

Source: `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py:194-215`.

### Derived per-device geometry

| Quantity              | Value (per device)         | Source                                                  |
|---|---|---|
| `local_seq_len`       | 3200 / 4 = 800             | seq_len / ring_size                                     |
| `local_padded_Nt`     | 800 / 32 = 25 tiles        | tile = 32×32                                            |
| `Sq_chunk_t`          | 160 / 32 = 5 tiles         | q_chunk_size / TILE                                     |
| `Sk_chunk_t`          | 320 / 32 = 10 tiles        | k_chunk_size / TILE                                     |
| `DHt` / `vDHt`        | 18 / 4 tiles               | d_q/32, d_v/32                                          |
| `num_local_k_chunks`  | ceil(25/10) = 3            | local_padded_Nt / Sk_chunk_t (rounded up)               |
| coarse half (tiles)   | 25 / 2 = 12                | balanced split point                                    |
| **straddle?**         | **yes** (`12 % 10 ≠ 0`)    | `KCausalStraddleInfo<25,10>`, `ring_utils.hpp:170-179`  |
| `straddle_chunk_id`   | 1                          | floor(12 / 10)                                          |
| `straddle_num_padded_tiles` | 8                    | 10 − (12 mod 10)                                        |

So `mla_100k k=320` *does* produce a straddle chunk: chunk index 1 spans the
boundary between the early-half and late-half of the device's K shard, and on
UP iterations its last 8 columns must be `-inf`-masked.

---

## 1. Op interface

Header: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.hpp`

The host op `ttnn::ring_joint_scaled_dot_product_attention(...)` takes:

- **Local Q/K/V tensors** — `[B, NH, local_padded_N, DH]` (with `B=1, NH=29,
  local_padded_N=800, DH=576/128`).
- **Joint Q/K/V tensors** — same shape but length `L`. Used for the
  "rear"-strategy joint attention attached after the ring (typical for diffusion
  transformer text/image dual streams). For pure ring-only attention `L=0` and
  the joint path becomes a no-op.
- **`persistent_output_buffer_k/v`** — pre-allocated AllGather destination
  tensors (avoids reallocation on every call; the AllGather writes the
  same-shape gathered K/V here).
- **`logical_n`** — unpadded global sequence length. The op tolerates
  `seq_len` being padded up beyond `logical_n`; padding tiles are masked out
  with the lightweight padding-mask machinery (see §7).
- **`is_causal`, `is_balanced`** — booleans. For mla_100k both are `true`.
- **`multi_device_global_semaphore`, `topology`, `cluster_axis`,
  `num_links`, `ccl_core_grid_offset`** — wire the AllGather CCL sub-op into
  the same program; see §2.

Returns `RingJointSDPAResult` = `{output, joint_output, stats}`:
- `output` is the local-Q attention result (per-device shard,
  `[B, NH, local_padded_N, DV]`).
- `joint_output` is the joint-Q result if `L > 0`.
- `stats` carries per-(batch, head, q_tile) running max + LSE so the streaming
  / multi-iter path can renormalize across ring iters; in the streaming path
  the kernel writes them into DRAM each iter and reads them back next iter
  (see §5).

---

## 2. Program factory

File: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp`

The factory produces **one program per device** that contains both the SDPA
kernels *and* the AllGather CCL workers, fused via shared semaphores.

### 2.1 AllGather direction config

Lines `142-164`. `ccl::get_forward_backward_configuration(...)` returns
per-device `(num_targets_forward, num_targets_backward)`. For Ring topology and
**even** `device_index`, the two are swapped:

```cpp
if (topology == Topology::Ring && device_index % 2 == 0) {
    std::swap(num_targets_forward, num_targets_backward);
}
```

Then `forward_writes_expected` / `backward_writes_expected` are wired into the
fused-op signaler. **This swap is the reason the per-(iter, ring_index)
DIAG/UP/DOWN classification is not symmetric across devices** — the iter →
ring_id sequence depends on which direction kicks in first.

### 2.2 Streaming compute v2 selection

Lines `330-367`. Streaming v2 (the path used by all current ring SDPA configs
that fit) is selected when:

```cpp
use_streaming_compute =
    !fp32_dest_acc_en
    && qk_out_subblock_h <= 2
    && Sk_chunk_t % qk_out_subblock_w == 0
    && qk_in0_num_subblocks > 1;
```

For `mla_100k` (Sq_chunk_t=5, Sk_chunk_t=10, bf16/bf8_b → fp32_dest_acc_en
typically off) this evaluates true. Streaming v2 also enables an
`out`-CB shrink to a 2-slot ping-pong if `ring_size==1 || max_q_per_core==1`
(`streaming_shrink_safe`, lines `353-366`). `mla_100k` ring_size=4 with
multiple Q-chunks per core does *not* shrink.

### 2.3 Compile-time vs runtime args

Each kernel gets a long compile-time arg list (subblock geometry,
`Sq_chunk_t`, `Sk_chunk_t`, `local_padded_Nt`, `num_local_k_chunks`,
`num_q_chunks`, `ring_size`, `is_causal`, `is_balanced`,
`use_streaming_compute`, plus tensor accessor descriptors and CCL semaphore
IDs). The runtime args are minimal: per-core `(global_q_start, global_q_end)`
for Q work assignment, plus the fused-op signaler's per-core RT args
(line `1415`).

### 2.4 The TT_METAL_RING_ITER_ONLY measurement hook

Lines `608-616`:

```cpp
const char* ring_iter_only_env = std::getenv("TT_METAL_RING_ITER_ONLY");
const bool ring_iter_only_enabled = (ring_iter_only_env != nullptr);
if (ring_iter_only_enabled) {
    defines["RING_ITER_ONLY_ENABLED"] = "1";
    defines["RING_ITER_ONLY_TARGET"] = std::to_string(std::atoi(...));
}
```

When set, all three kernels short-circuit non-target iters via `#ifdef
RING_ITER_ONLY_ENABLED`, and the AllGather sub-op is **skipped at host level**
(line `1422`: `if (!ring_iter_only_enabled) { ... ring_attention_all_gather ...
}`). The reader then sets `sdpa_wait_for_op_signal=false`
(`ring_joint_reader.cpp:74-79`) so it never waits on AG semaphores.

This means measured iter durations include only the SDPA kernel itself, no
CCL contention. K/V must be pre-populated (the gathered tensors are the actual
buffers; AllGather is the only thing that fills them with real data when not
in measurement mode).

---

## 3. Ring iteration — ring_id sequencer

File: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_utils.hpp`

`RingIdSequencer` (lines `25-91`) is the single source of truth for "which
device's KV does this iter process." Used identically by reader, writer, and
compute (so all three agree on the iter ↔ ring_id mapping).

State: `received[2]={0,0}`, `expected[2]={backward_expected, forward_expected}`,
`curr_dir`, `transfer_idx`. Per call to `get_next_ring_id(sync_fn)`:

- **iter 0:** `sender_ring_id = ring_index` (DIAG, local KV).
- **iter > 0, curr_dir=0 (backward):** `received[0]++`, `sender_ring_id =
  (ring_index + received[0]) % ring_size` — i.e. step *forward* in the ring.
- **iter > 0, curr_dir=1 (forward):** `received[1]++`, `sender_ring_id =
  (ring_index − received[1] + ring_size) % ring_size` — step *backward*.
- After each iter: flip direction if the other direction still has expected
  receives left.

The `sync_fn` callback is where the reader's CCL semaphore wait is injected
(no-op in compute / pre-scan).

`find_last_active_ring_iter(...)` (lines `104-126`) re-runs the sequencer
without sync to figure out the last iter that actually does work — used by all
three kernels for early-out / "is this the last iter that produces output"
decisions.

For ring_size=4, the per-device sequence is *roughly* `d, d+1, d−1, d+2 (mod
4)` or `d, d−1, d+1, d−2`, depending on the direction-config swap from §2.1.
The empirical mapping for the mla_100k Quiet Box run is in
`per_iter_perf_measurement.md`.

---

## 4. Reader kernel — `ring_joint_reader.cpp`

File: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_reader.cpp`

Top-level `kernel_main` runs the ring loop (line ~232 onward). Per iter:

1. **Get `ring_id`** from `RingSDPAOpReceiver` (which wraps
   `RingIdSequencer` + the AllGather semaphore wait).
2. **`RING_ITER_ONLY` short-circuit** (lines `237-241`): if the env var is set
   and this isn't the target iter, skip everything for this iter (the
   sequencer still advances).
3. **Pick K/V source:**
   - Iter 0 (`ring_id == ring_index`, **DIAG**): read from the local K/V
     tensors. No AG wait needed (the local data is always available).
   - Iter > 0: read from the **gathered** K/V tensors filled by the AllGather
     fused into this program. The receiver's sync_fn waits on the AG semaphore
     before reading.
4. **Per-iter K-chunk count** (DIAG/UP/DOWN logic):
   ```cpp
   uint32_t iter_num_kv_chunks = num_kv_chunks;
   if (is_causal && is_balanced && ring_index > ring_id) {     // UP
       if constexpr (Straddle::has_straddle) {
           iter_num_kv_chunks = Straddle::straddle_chunk_id + 1;  // include straddle
       } else {
           iter_num_kv_chunks /= 2;
       }
   }
   ```
   For mla_100k k=320 on UP iters: `iter_num_kv_chunks = 1+1 = 2` chunks
   (chunks 0 and the straddle chunk 1).
5. **Per-Q-chunk skip** (DOWN):
   ```cpp
   const bool balanced_skip_q = q_chunk < half_sequence
                                && is_balanced
                                && ring_index < ring_id;
   if (balanced_skip_q) continue;
   ```
   On DOWN iters, the device drops the early-half Q chunks (its zigzag pair
   covers them).
6. **K/V chain forwarding** — for `nhk == 1` (mla_100k) K is shared across all
   heads of a batch, so the reader uses a **batch-chain** L1→L1 forward
   (one core reads K from DRAM, then forwards the same tiles to all peer Q
   cores via store-and-forward) instead of every core hitting DRAM.
7. **Joint K/V** are read on the iter where `ring_id == ring_size − 1`
   (`do_joint_kv` flag); otherwise no joint work.

Reader instrumentation: line ~150 has a DPRINT
`[RING] rix=<ring_index> iter=<n> rid=<ring_id>` that's emitted only from
`UCK_CHLKC_UNPACK` to keep the kernel binary small (see Notes in
`per_iter_perf_measurement.md`).

---

## 5. Writer kernel — `ring_joint_writer.cpp`

File: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_writer.cpp`

The writer does three things per iter (line `430` onward):

1. **`RING_ITER_ONLY` short-circuit** (lines `432-436`) — same pattern as the
   other kernels.
2. **Per-iter validity check** — `find_last_active_ring_iter` + a per-iter
   `ring_iter_does_work` predicate (lines `441-447`) — skip iters where the
   KV start tile is past `global_n_tile_id` (i.e. all-padded chunks).
3. **Mask construction & output drain.**
   - The writer is the kernel that *generates* the lightweight mask tiles
     (neginf, causal-diagonal stamp, padding partials, straddle padding) into
     `cb_3` once at startup; compute consumes them.
   - Per Q chunk it drives the output DRAM writeback. Two paths:
     - **Single Q-chunk per core** (`single_q_chunk`, line `497`):
       accumulators stay in L1 across ring iters; output is written only on
       `is_last_ring_iter`.
     - **Multi Q-chunk per core**: per-iter `(prev_out, max, sum)` are
       round-tripped through DRAM. `issue_restore_reads(...)` issues NOC reads
       early so they overlap with the previous iter's compute (`save_trid`
       prefetch barrier).
4. **`balanced_skip_q` for DOWN iters** (line `593`):
   ```cpp
   const bool balanced_skip_q = q_chunk < half_sequence
                                && is_balanced
                                && ring_index < ring_id;
   ```
   On non-last iters with `balanced_skip_q`, the writer skips both
   accumulator restore *and* output write. On the last iter, the
   `defer_prefetch = balanced_skip_q && is_last_ring_iter` branch (line `629`)
   normalizes incrementally rather than restoring (because there's no prior
   accumulator to merge with).

---

## 6. Compute kernel — `ring_joint_sdpa.cpp`

File: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/ring_joint_sdpa.cpp`

Top-level `kernel_main`: precomputes `last_active_ring_iter`, then loops
`for (ring_iter ...)` (line `148`):

1. **`get_next_ring_id_and_sync`** — advances the sequencer; emits the DPRINT
   on UNPACK only.
2. **`RING_ITER_ONLY` short-circuit** (lines `156-159`).
3. **Joint flag** — `do_joint_kv = (ring_id == ring_size − 1)`.
4. **`num_kv_chunks`** — `num_local_k_chunks` plus `num_joint_k_chunks` if this
   iter handles joint.
5. **Per-iter K-chunk count + Q-skip flag** — exact same logic as reader
   (lines `224-232`):
   ```cpp
   uint32_t iter_num_kv_chunks = num_kv_chunks;
   if (is_causal && is_balanced && ring_index > ring_id) {     // UP
       iter_num_kv_chunks = has_straddle ? straddle_chunk_id + 1
                                          : iter_num_kv_chunks / 2;
   }
   const bool skip_first_half_q = (ring_index >= ring_id ? false : is_balanced);
   ```
6. **Lightweight mask context** built per iter:
   - `lw_mask.is_causal = (ring_iter == 0 ? is_causal : false)` — causal
     diagonal applied **only on DIAG**.
   - Straddle mask flags fire only on UP iters with a straddle.
   - Padding masks (global_n / local_n / joint_l) fire if their respective
     boundaries fall in this iter's KV window.
7. **Dispatch** — streaming v2 path (`sdpa_ring_v2<...>`) for mla_100k,
   non-streaming `sdpa_ring<...>` otherwise. Both take `iter_num_kv_chunks`,
   `skip_first_half_q`, `lw_mask`, and `is_last_ring_iter`.

The deferred-norm streaming path (compute_streaming.hpp) keeps running
`(max, sum, out_im)` accumulators in `cb_max_*` / `cb_sum_*` / `cb_out_im_*`
and only normalizes on the last iter (or last K-chunk for the non-causal
single-Q case). Per-iter writes to the stats tensor let the *next* iter
restore them.

---

## 7. Lightweight mask plumbing

File: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp` (around `LightweightMaskContext`, lines ~1449-1528).

A single CB (`cb_mask_in` = `c_3`) holds at most:

- `[0]` — neginf full tile (always present when mask is needed)
- `[1]` — causal-diagonal full tile (causal case; stamps -inf above the
  diagonal of one (Sq_chunk_t × Sk_chunk_t) block, used only on DIAG)
- partial tiles for `global_n`, `local_n`, `joint_l`, and **straddle** padding
  (each tile padded such that its trailing columns are -inf for the
  appropriate boundary).

Per K-chunk, `lw_mask.resolve_for_chunk(...)` returns which tile to bcast-add
into the QK product. Resolution priority is: causal (DIAG only) → padding
masks (whichever boundary this chunk crosses) → straddle (UP-iter only).

For `mla_100k k=320`:
- DIAG iter: causal-diagonal stamp on the on-the-diagonal K chunk; padding
  stamps possible on the last chunk if `local_padded_Nt % Sk_chunk_t ≠ 0`
  (here `25 % 10 = 5` ≠ 0 → yes, last chunk has 5 padded columns).
- UP iter: no causal stamp; straddle stamp on chunk index 1 (8 padded
  columns).
- DOWN iter: no causal, no straddle; only padding masks if applicable.

---

## 8. DIAG / UP / DOWN summary

| | **DIAG** (iter 0) | **UP** (`ring_index > ring_id`) | **DOWN** (`ring_index < ring_id`) |
|---|---|---|---|
| KV source | local | gathered (AG) | gathered (AG) |
| K-chunk count | `num_local_k_chunks` (3) | `straddle_chunk_id + 1` (2) — or `/2` if no straddle | `num_local_k_chunks` (3) |
| Q-chunk skip | none | none | first-half Q chunks skipped on non-last iter |
| Causal mask | yes | no | no |
| Straddle mask | no | yes (mla_100k k=320: 8 padded cols on chunk 1) | no |
| Output writeback | yes | yes | only on last iter (normalize-only branch) |
| FLOPs (rough) | full | ~half (half KV chunks) | ~half (half Q chunks); zero on non-last |

UP and DOWN are deliberately asymmetric: UP halves K-side work, DOWN halves
Q-side work. Their *total* FLOPs match (half of full), which is why the
balanced+causal `compute_math_utilization` divides by 2 for `iter > 0`
(`test_ring_joint_sdpa.py:1109`).

The reason DIAG measures *lower* math util than UP/DOWN despite doing more
work is that the causal-diagonal mask + per-K-chunk mask resolution stalls
the FPU on mask-application overhead; UP/DOWN are pure matmul on smaller
chunks.

---

## 9. Where this comes from in code (quick index)

| concern                                | file                                                  |
|---|---|
| Op-level entry / params                | `ring_joint_sdpa_device_operation.hpp/.cpp`           |
| Program build, CB sizes, AG fusion     | `ring_joint_sdpa_program_factory.cpp`                 |
| iter → ring_id sequencer + straddle    | `kernels/dataflow/ring_utils.hpp`                     |
| Reader (K/V fetch, K-skip, Q-skip)     | `kernels/dataflow/ring_joint_reader.cpp`              |
| Writer (mask gen, output drain)        | `kernels/dataflow/ring_joint_writer.cpp`              |
| Compute top-level (ring loop)          | `kernels/compute/ring_joint_sdpa.cpp`                 |
| Streaming v2 inner loop                | `kernels/compute/compute_streaming.hpp`               |
| Mask context + resolution              | `kernels/compute/compute_common.hpp` (`LightweightMaskContext`) |
| Per-iter perf measurement procedure    | `tests/nightly/blackhole/sdpa/per_iter_perf_measurement.md` |

## 10. Known TODOs

- `ring_joint_sdpa.cpp:91-92` and `ring_joint_reader.cpp:118-119`: CB indices
  are hardcoded in both kernels and duplicated from the program factory.
  Should be passed as compile-time args.
