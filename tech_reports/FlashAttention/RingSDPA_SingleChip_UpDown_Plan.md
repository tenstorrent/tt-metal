# Plan — Single-Chip SDPA Perf Proxy for Ring-Joint Up/Down Iters

Companion to `RingSDPA_CausalBalancing.md` and `RingSDPA_IterCases_SingleChip.md`.

## Goal

Extend single-chip SDPA so it can reproduce the **exact** per-device math
work and DRAM layout of a `ring_joint_sdpa` non-diag iter (**up** or
**down**), and add matching test rows to
`tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py`.

The proxy keeps **full-size tensors** `Q[L], K[L], V[L]` — matching what each
device actually holds at iter 0 — and **skips reads/compute** the same way the
ring kernel does for iter 1+. This gives cycle-level fidelity to the ring
per-iter work, including DRAM bank layout, tile addresses, and core
dispatch.

## Guiding constraint

**Do not change current SDPA behavior.** Every change is strictly additive and
gated on a new opt-in `ring_proxy_case ∈ {none, up, down}` field in
`SDPAProgramConfig`. Default is `none` — every existing caller compiles into
identical kernels and produces identical outputs.

## Mapping ring iters → single-chip proxy

| Ring case | Ring kernel semantics                                              | Single-chip proxy                                                                 |
|-----------|--------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| diag      | causal, `balancing=false`, full K per Q                            | `Q[L], K[L], V[L]`, `is_causal=True`, `flatten_work=True`, `proxy=none` — **already works** |
| up        | non-causal, `iter_num_kv_chunks /= 2`, full Q                      | `Q[L], K[L], V[L]`, `is_causal=False`, `flatten_work=True`, `proxy=up`            |
| down      | non-causal, full K, skips light Q stripe via `is_balanced` in loop | `Q[L], K[L], V[L]`, `is_causal=False`, `flatten_work=True`, `proxy=down`          |

**Math equivalence check (MLA 100k per-device: B=1, NHQ=32, q_num_chunks=20, k_num_chunks=20 on 110 cores):**

- iter 0 (causal): `640` slots × ~10 K-iters avg = ~6400 K-iters.
- up (proxy): `640` slots × `10` K-iters (half K, no causal) = `6400` K-iters.
- down (proxy): `320` slots (heavy Q only) × `20` K-iters (full K) = `6400` K-iters.

All three cases land on the same total K-iters = perfectly matched per-device work.

## Execution steps

Steps 1-2 extend the config surface.
Steps 3-8 teach the single-chip program factory about the new cases.
Steps 9-13 add the kernel-side skip logic.
Steps 14-19 update the sprint test file.
Steps 20-23 verify and cross-check.

---

### A. Config surface

#### Step 1 — Add `RingProxyCase` enum to `sdpa_config.hpp`

File: `ttnn/cpp/ttnn/operations/transformer/sdpa_config.hpp`

Add:

```cpp
enum class RingProxyCase : uint8_t {
    None = 0,
    Up = 1,    // simulate ring iter with ring_index > ring_id (skips upper half of K per Q)
    Down = 2,  // simulate ring iter with ring_index < ring_id (skips light Q stripe)
};

struct SDPAProgramConfig {
    // ... existing fields ...
    bool flatten_work = false;
    RingProxyCase ring_proxy_case = RingProxyCase::None;
};
```

Comment on the new field explaining that it's a single-chip perf-proxy for
multi-chip ring-joint iter 1+ work, requires `flatten_work=True` and
`is_causal=False`, and keeps tensor shapes symmetric with iter-0 (full L×L).

#### Step 2 — Expose enum and field via nanobind

File: `ttnn/cpp/ttnn/operations/transformer/transformer_nanobind.cpp`

- Add `nb::enum_<RingProxyCase>(mod, "RingProxyCase").value("NONE", ...).value("UP", ...).value("DOWN", ...)`.
- Extend `SDPAProgramConfig` constructor overload (line 26-42) with `nb::arg("ring_proxy_case") = RingProxyCase::None`.
- Add to `def_rw` list (after `flatten_work`).
- Add to `__repr__`.

**Scope**: existing callers never set `ring_proxy_case` → default `None` → no
observable change.

---

### B. Program factory

#### Step 3 — Validate mode combinations in `sdpa_program_factory.cpp`

Add near the top of `create()` (around the existing `flatten_work` fatals at
line 365-368):

```cpp
const RingProxyCase proxy_case =
    program_config.has_value() ? program_config->ring_proxy_case : RingProxyCase::None;
const bool is_up = (proxy_case == RingProxyCase::Up);
const bool is_down = (proxy_case == RingProxyCase::Down);
const bool is_proxy = is_up || is_down;

if (is_proxy) {
    TT_FATAL(flatten_work, "ring_proxy_case requires flatten_work=true");
    TT_FATAL(!is_causal, "ring_proxy_case requires is_causal=false "
        "(proxy simulates non-diag ring iters, where causality is off)");
    TT_FATAL(!is_chunked, "ring_proxy_case incompatible with chunked prefill");
    TT_FATAL(!use_attention_sink, "ring_proxy_case incompatible with attention_sink");
    TT_FATAL(!use_provided_mask, "ring_proxy_case incompatible with user-provided attn mask");
    TT_FATAL(q_num_chunks % 2 == 0, "ring_proxy_case requires even q_num_chunks");
    if (is_up) {
        TT_FATAL(k_num_chunks % 2 == 0, "ring_proxy_case=up requires even k_num_chunks "
            "(the kernel caps K loop at k_num_chunks/2)");
    }
}
```

#### Step 4 — Lift the `is_causal` TT_FATAL on `flatten_work`

File: `sdpa_program_factory.cpp:366`

Remove:
```cpp
TT_FATAL(is_causal, "SDPAProgramConfig::flatten_work currently requires is_causal=true");
```

Keep the no-chunked, no-attention-sink fatals. Step 3's `is_proxy` validations
handle the new case tightly.

**Scope**: direct path change is only for the new opt-in proxy case (callers
won't otherwise pass `flatten_work=true` with `is_causal=false`).

#### Step 5 — Skip KV chain when `flatten_work=true`

File: `sdpa_program_factory.cpp:872`

```cpp
if (!is_causal && !is_chunked && !flatten_work) {
    // ... existing chain construction ...
}
```

Host still pushes 14 zero chain args per core so the reader's arg layout
holds; `is_chain_participant=0` disables chain branches.

**Scope**: existing causal `flatten_work` never entered this block; existing
non-causal non-flat still enters. Only the new `flatten_work && !is_causal`
path skips.

#### Step 6 — Force streaming compute off when `flatten_work=true`

File: `sdpa_program_factory.cpp:75` (`can_use_streaming_compute`)

Add `bool flatten_work` parameter, early-return `false` when set. Pass
`flatten_work` at the call site around line 428.

**Why**: streaming path (`sdpa.cpp:119-163`) doesn't honor `SDPA_FLAT_WORK`
and would silently fall into hierarchical loops when `is_causal=false` —
destroying the perf proxy.

**Revisit**: once streaming compute is enabled for causal ring-joint SDPA
(and `SDPA_FLAT_WORK` is threaded through the streaming branch), re-enable
streaming here by dropping or refining this gate.

#### Step 7 — Size flat work for UP vs DOWN

File: `sdpa_program_factory.cpp:396`

`total_q_chunks` is declared once and only used inside flat-work math
(zigzag pair count, per-core assignment at line 1376-1385, and the clamps).
No non-flat uses exist, so the simplest change is to make the declaration
itself conditional — no renaming downstream:

```cpp
const uint32_t total_q_chunks = B * NQH * (is_down ? (q_num_chunks / 2) : q_num_chunks);
```

**Rationale**:
- UP: `total = B*NQH*q_num_chunks` (all Q chunks computed; only the K loop halves).
- DOWN: `total = B*NQH*q_num_chunks/2` (only heavy Q half assigned; K loop full).

`flat_zigzag` stays `false` (requires `is_causal=True`) — fine, both proxy
cases want linear distribution.

`max_flat_chunks_per_core` and `q_buffer_factor_flat` derive from
`total_q_chunks` automatically — CB sizing correct.

#### Step 8 — Emit proxy-case defines + compile-time args

File: `sdpa_program_factory.cpp:680-692` (where `SDPA_FLAT_WORK` is added)

```cpp
if (is_up) {
    defines["SDPA_RING_PROXY_UP"] = "1";
} else if (is_down) {
    defines["SDPA_RING_PROXY_DOWN"] = "1";
}
log_debug(tt::LogOp, "SDPA_RING_PROXY: {}",
    is_up ? "up" : is_down ? "down" : "none");
```

If the kernels need the *effective* q_num_chunks or the heavy-Q offset for
decompose, pass them as compile-time args. Simplest: the macros are sufficient
(kernel derives `q_num_chunks/2` from the existing `q_num_chunks` CT arg).

---

### C. Kernel-side skip logic

#### Step 9 — Reader: UP K-loop cap + DOWN q_chunk offset

File: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp`

Two surgical additions, both inside the `#if defined(SDPA_FLAT_WORK)` block
(lines 296-317):

1. **DOWN decompose effective size and offset** — before the
   `decompose_flat_q_index` call (line 303):

   ```cpp
   #if defined(SDPA_RING_PROXY_DOWN)
       constexpr uint32_t _proxy_q_num_effective = q_num_chunks / 2;
       constexpr uint32_t _proxy_q_chunk_offset = q_num_chunks / 2;
   #else
       constexpr uint32_t _proxy_q_num_effective = q_num_chunks;
       constexpr uint32_t _proxy_q_chunk_offset = 0;
   #endif
   const auto _decoded = decompose_flat_q_index(
       global_q_start + _gq, _proxy_q_num_effective, NQH, flat_use_zigzag);
   const uint32_t nb = _decoded.nb;
   const uint32_t nq = _decoded.nq;
   const uint32_t q_chunk_local = _decoded.q_chunk;   // in [0, q_num_chunks_effective)
   const uint32_t q_chunk = q_chunk_local + _proxy_q_chunk_offset;  // heavy-half for DOWN, unchanged for UP/none
   ```

   Then use `q_chunk` everywhere downstream (the existing code already
   propagates `q_chunk` into `q_low_idx` / tile-index math).

2. **UP K-loop cap** — at the `for (k_chunk ...)` loop at line 431:

   ```cpp
   #if defined(SDPA_RING_PROXY_UP)
       const uint32_t k_chunk_end = k_num_chunks / 2;
   #else
       // existing: derived from q_high_idx
       const uint32_t k_chunk_end = (q_high_idx + Sk_chunk_t - 1) / Sk_chunk_t;
   #endif
       for (uint32_t k_chunk = 0; k_chunk < k_chunk_end; ++k_chunk) {
           ...
       }
   ```

   (Replaces the existing implicit bound `k_chunk * Sk_chunk_t < q_high_idx`
   with an explicit `k_chunk_end` so the UP path can override it cleanly.)

Outside the flat-work branch (non-flat readers), no change: the proxy modes
require `flatten_work=True` (step 3 fatal).

#### Step 10 — Writer: DOWN q_chunk offset

File: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/writer_interleaved.cpp`

In the `#if defined(SDPA_FLAT_WORK)` block (line 121 onwards), mirror Step 9's
decompose change: compute `q_chunk_local` via `decompose_flat_q_index` with
`_proxy_q_num_effective`, then `q_chunk = q_chunk_local + _proxy_q_chunk_offset`.

Writer doesn't iterate K, so no UP cap needed.

#### Step 11 — Compute: UP K-loop cap in `sdpa_inner_loop` STANDARD branch

File: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp`

In `sdpa_inner_loop` STANDARD branch (line 1707-1713):

```cpp
uint32_t k_chunk_end;
if constexpr (sdpa_type == STANDARD) {
#if defined(SDPA_RING_PROXY_UP)
    k_chunk_end = k_num_chunks / 2;
#else
    k_chunk_end = (q_high_idx + Sk_chunk_t - 1) / Sk_chunk_t;
#endif
} else {  // RING or JOINT.
    k_chunk_end = iter_k_chunk_end;
}
```

`k_num_chunks` is already a compile-time arg (arg 9 in `sdpa.cpp`) available
inside `sdpa_inner_loop`; if not, thread it from the caller (`sdpa_standard`).

**Verify**: `sdpa_standard` already receives `k_num_chunks` — check the
signature; if missing, extend it (local to the file).

#### Step 12 — Compute: DOWN q_chunk offset in `sdpa_inner_loop` flat branch

File: `compute_common.hpp` around line 1658 (existing `SDPA_FLAT_WORK` branch):

```cpp
#if defined(SDPA_FLAT_WORK)
    constexpr bool _flat_use_zigzag = get_compile_time_arg_val(33) == 1;
#if defined(SDPA_RING_PROXY_DOWN)
    constexpr uint32_t _proxy_q_num_effective = q_num_chunks / 2;
    constexpr uint32_t _proxy_q_chunk_offset = q_num_chunks / 2;
#else
    constexpr uint32_t _proxy_q_num_effective = q_num_chunks;
    constexpr uint32_t _proxy_q_chunk_offset = 0;
#endif
    q_chunk = remap_q_index(q_iter, _proxy_q_num_effective, _flat_use_zigzag)
              % _proxy_q_num_effective + _proxy_q_chunk_offset;
#elif defined BALANCED_Q_PARALLEL
    // ...
```

This keeps the compute kernel's `q_chunk` aligned with reader/writer.

#### Step 13 — Confirm reader/writer arg layout still aligns

Walk-through (verify, no edit):

- Host: std args (0-16) → `global_q_start, global_q_count` (17-18) →
  if `!is_causal`, chain args (19-32).
- Reader + `SDPA_FLAT_WORK` + `!is_causal`: consumes std, flat, chain in
  matching order. ✓
- Writer: no chain args; consumes std + flat. ✓

The new `SDPA_RING_PROXY_*` defines don't add runtime args — purely
compile-time branches inside the existing flat body.

---

### D. Test file

#### Step 14 — Keep `seq_len` single-valued in `ModelConfig` (no change)

File: `tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py`

Q/K/V all have the same length in the proxy. No schema change needed on
`ModelConfig`; just add a new optional field for the proxy case.

```python
@dataclass
class ModelConfig:
    # ... existing ...
    flatten_work: bool = False
    ring_proxy_case: str = "none"  # "none" | "up" | "down"
```

#### Step 15 — Thread `ring_proxy_case` through `run_sdpa`

In `run_sdpa`, map the string to `ttnn.RingProxyCase.*` and pass via
`SDPAProgramConfig(..., ring_proxy_case=...)`. When `ring_proxy_case != "none"`,
force `is_causal=False` (already enforced by the fatal, but set it up
correctly in the test).

#### Step 16 — Add **UP** test suite

Mirror the existing `mla_100k_ring_iter_0` config with a
`mla_100k_ring_iter_up` variant. Identical shape / chunking / dtype as iter-0
— only `is_causal` flips to `False` and `ring_proxy_case="up"`:

```python
ModelConfig(
    name="mla_100k_ring_iter_up",
    nhq=32, nkv=1,
    seq_len=3200,                 # Q and K both full L
    d_q=576, d_k=576, d_v=128,
    is_causal=False,              # proxy requires non-causal
    q_dtype=ttnn.bfloat16, kv_dtype=ttnn.bfloat8_b,
    q_chunk_sizes=[160], k_chunk_sizes=[160],
    flatten_work=True,
    ring_proxy_case="up",
),
```

Pre-flight sanity (satisfies Step 3 fatals):

| Suite | `q_num_chunks` | `k_num_chunks` | Even Q? | Even K? |
|-------|----------------|----------------|---------|---------|
| 100k  | 3200/160 = 20  | 3200/160 = 20  | ✓       | ✓       |

#### Step 17 — Add **DOWN** test suite

Same pairing rule — identical shape to iter-0, `is_causal=False`,
`ring_proxy_case="down"`:

```python
ModelConfig(
    name="mla_100k_ring_iter_down",
    nhq=32, nkv=1,
    seq_len=3200,
    d_q=576, d_k=576, d_v=128,
    is_causal=False,
    q_dtype=ttnn.bfloat16, kv_dtype=ttnn.bfloat8_b,
    q_chunk_sizes=[160], k_chunk_sizes=[160],
    flatten_work=True,
    ring_proxy_case="down",
),
```

Pre-flight sanity (DOWN only requires even `q_num_chunks`; K-loop stays
full-length):

| Suite | `q_num_chunks` | Even Q? |
|-------|----------------|---------|
| 100k  | 20             | ✓       |

After Steps 16-17, the single-chip ring-iter matrix for MLA 100k is:

| Suite       | iter-0 (diag)               | up (non-diag, skip-K)       | down (non-diag, skip-Q)       |
|-------------|-----------------------------|-----------------------------|-------------------------------|
| MLA 100k    | `mla_100k_ring_iter_0` ✓    | `mla_100k_ring_iter_up`     | `mla_100k_ring_iter_down`     |

(128k follow-up out of scope for now — deferred until 100k lands.)

#### Step 18 — Accuracy test handling

- UP: kernel computes `attn(Q, K[:,:,:L/2], V[:,:,:L/2])` over all Q
  positions. Torch reference:
  `torch.nn.functional.scaled_dot_product_attention(Q, K[:,:,:L/2],
  V[:,:,:L/2], is_causal=False)`. Compare all Q output rows.
- DOWN: kernel computes `attn(Q[:,:,L/2:], K, V)` into the heavy-half Q output
  rows; light-half output rows are left untouched (undefined). Torch
  reference: compute only `attn(Q[:,:,L/2:], K, V, is_causal=False)`,
  compare against `tt_back[:,:,L/2:,:]` only.

Update `run_sdpa`'s reference path with a branch on
`ring_proxy_case`:

```python
if ring_proxy_case == "up":
    gt = torch_sdpa(Q, K[:,:,:L//2], V[:,:,:L//2], is_causal=False)
    tt_out = ttnn.to_torch(tt_back)[:, :, :L, :d_v]
elif ring_proxy_case == "down":
    gt = torch_sdpa(Q[:,:,L//2:], K, V, is_causal=False)
    tt_out = ttnn.to_torch(tt_back)[:, :, L//2:L, :d_v]
else:
    # existing path
```

PCC threshold: keep 0.994 for bfp8 kv_dtype — same as iter-0.

#### Step 19 — Perf-table update

In `test_sdpa_create_perf_table`, math util / FLOPs should account for
proxy case (the non-causal rectangle for UP/DOWN is half the full L² one):

- UP: effective compute = `nhq * q_num_chunks * (k_num_chunks/2) * Sq_chunk * Sk_chunk * d`.
- DOWN: effective compute = `nhq * (q_num_chunks/2) * k_num_chunks * Sq_chunk * Sk_chunk * d`.
- Both equal `nhq * L*L / 2 * d` — same as iter-0 causal.

Pass `is_causal=False` and halved seq to `compute_math_utilization` — or
introduce a helper that knows the proxy case. Easiest: for UP pass
`(s, s/2, ..., is_causal=False)`; for DOWN pass `(s/2, s, ...,
is_causal=False)`; for none pass existing.

---

### E. Verification

#### Step 20 — Accuracy smoke tests

```bash
source python_env/bin/activate
scripts/run_safe_pytest.sh \
  "tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_accuracy[mla_100k_ring_iter_up-q160-k160]"
scripts/run_safe_pytest.sh \
  "tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_accuracy[mla_100k_ring_iter_down-q160-k160]"
```

Expected: PCC ≥ 0.994.

#### Step 21 — Determinism

```bash
scripts/run_safe_pytest.sh \
  "tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_determinism[mla_100k_ring_iter_up-q160-k160]"
scripts/run_safe_pytest.sh \
  "tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_determinism[mla_100k_ring_iter_down-q160-k160]"
```

Expected: bit-exact over 10 runs.

#### Step 22 — Perf parity vs ring iter matrix

Run the three single-chip perf-table rows and the ring iter matrix side by
side:

```bash
# Single chip
for name in mla_100k_ring_iter_0 mla_100k_ring_iter_up mla_100k_ring_iter_down; do
  scripts/run_safe_pytest.sh \
    "tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_create_perf_table[$name]"
done

# Multi-chip reference
TT_METAL_RING_ITER_MATRIX_REPEATS=3 \
  scripts/run_safe_pytest.sh \
  "tests/nightly/blackhole/ccl/test_ring_joint_sdpa.py::test_ring_joint_attention_ring_iter_matrix[mla_100k]"
```

Expected: per-iter duration and math util for single-chip
`_ring_iter_0` / `_up` / `_down` align with the ring matrix's `diag` / `up` /
`down` cells (within a few %).

**Reference numbers from ring-joint op measurement** (MLA 100k, 110 cores):

- DOWN case: ~56%+ math util
- UP case: ~52%+ math util

Single-chip proxies should land in the same ballpark — exact match isn't
required (no multi-chip ring overhead in the proxy), but a large gap from
these numbers indicates the proxy isn't faithfully reproducing the per-iter
work.

#### Step 23 — Reality-check streaming gate

One-off assertion in the test (or debug log from program factory) to
confirm `use_streaming_compute=false` when `flatten_work=true` or a proxy
case is active.

---

## Touched files (this change)

| File                                                                                          | Change                                                                                      |
|-----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| `ttnn/cpp/ttnn/operations/transformer/sdpa_config.hpp`                                        | Add `RingProxyCase` enum + `ring_proxy_case` field                                          |
| `ttnn/cpp/ttnn/operations/transformer/transformer_nanobind.cpp`                               | Bind enum + field                                                                           |
| `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp`                   | Validate proxy; lift causal gate; skip chain/streaming on flat; size flat work for DOWN; emit defines |
| `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp`    | UP K-loop cap; DOWN decompose offset (all inside `#if SDPA_FLAT_WORK`)                      |
| `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/writer_interleaved.cpp`    | DOWN decompose offset                                                                       |
| `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp`         | UP K-loop cap in STANDARD branch; DOWN q_chunk offset in FLAT branch                        |
| `tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py`                    | Add `ring_proxy_case` field; add up/down MLA 100k configs; branched accuracy ref; perf-table math util fix |

No changes to: `sdpa_device_operation*` (program config is already threaded),
`sdpa_nanobind.cpp` (uses `SDPAProgramConfig` opaquely), kernel files outside
the flat-work branches.

## Commit-split recommendation

One PR, broken into reviewable commits:

1. **Config surface**: `sdpa_config.hpp` + `transformer_nanobind.cpp` — expose the enum/field with no semantic effect yet.
2. **Program factory**: validations, gate lifts, chain/streaming skips, flat-work sizing.
3. **Reader kernel**: UP K-loop cap + DOWN offset inside SDPA_FLAT_WORK.
4. **Writer kernel**: DOWN offset.
5. **Compute kernel**: UP K-loop cap + DOWN offset.
6. **Sprint test**: up/down configs + reference branches + perf-table fixes.

This lets each piece be reviewed for "does it change current behavior?"
independently.

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Arg-layout drift between host/reader/writer/compute | Step 13 explicit walk-through; all new logic is inside `#if SDPA_FLAT_WORK` and uses compile-time constants, no new runtime args |
| `sdpa_inner_loop` STANDARD branch previously assumed `k_chunk_end` from `q_high_idx` | Step 11: introduce explicit `k_chunk_end` variable; non-proxy path defines it exactly as before |
| DOWN kernel output has undefined light-half rows | Step 18: accuracy test compares heavy half only; document that light-half output is **not** cleared (matches ring kernel behavior — ring doesn't write to skipped Q rows either) |
| q_num_chunks odd | Step 3 fatal |
| k_num_chunks odd (UP only) | Step 3 fatal |
| MLA's `use_mla` / asymmetric `d_q != d_v` | Unchanged — existing flat work already handles MLA (iter-0 test is MLA); proxy inherits that |
| bfloat8_b kv dtype accumulation noise | PCC threshold 0.994 (same as iter-0) |

## Out of scope (follow-ups)

1. **Latent chain-disabled bug in non-causal SDPA (pre-existing)**. Since
   `d4f5ca19df`, the host unconditionally pushes `global_q_start/count` at
   reader positions 17-18, causing non-flat non-causal readers to read chain
   args from the wrong offsets — chain is dormant for all non-causal callers.
   Fix: make host-side push conditional on `flatten_work`. **Not touched
   here** (user constraint: don't change current SDPA behavior).
2. **Streaming-compute + flat_work**. Extend `SDPA_FLAT_WORK` into the
   streaming branch if flat non-causal turns out meaningfully slower than
   streaming would give. Deferred until perf measurement motivates it.
3. **Intra-device zigzag for up/down**. Not needed — proxy cases have
   uniform per-core load. Document and revisit only if profiling reveals
   imbalance.
4. **Third case: joint-L segment**. Ring joint also handles an `L > 0`
   joint-KV tail; not modeled here (iter-0 sprint doesn't either). Add if a
   joint-L sprint is ever needed.
