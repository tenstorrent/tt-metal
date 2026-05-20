# Ring-joint SDPA chunked-prefill — plan

## Goal

Make `ttnn.transformer.ring_joint_scaled_dot_product_attention` support
chunked prefill: a sequence of N calls each passing a short Q slab at absolute
position `[i*c, (i+1)*c)` against a growing K/V cache, producing the same
output rows as a single full-sequence oracle call.

Op-level only — no projections, no RoPE, no MLA layer. Random `fa_rand`
tensors. The test lives in `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py`
and is `@pytest.mark.skipif(CI)` until green.

### Per-chunk attention shape (the key mental model)

The attention matrix is **not** a square lower-triangle. It's a wide
rectangle made of three regions:

```
K cols:  [0 .... q_start_idx)     [q_start_idx .. +Sq)        [q_start_idx+Sq .. total_seq)
         ┌──────────────────┐     ┌───────────────────┐       ┌────────────────────┐
Q chunk: │ dense rectangle  │  +  │  causal triangle  │   +   │  masked out        │
         │ (attend all)     │     │  (attend ≤ row)   │       │  (past logical_n)  │
         └──────────────────┘     └───────────────────┘       └────────────────────┘
```

The right region is already handled by the existing `logical_n` cutoff.
The left rectangle needs no mask. Only the middle `Sq × Sq` strip needs
causal-style masking — a self-contained sub-mask of fixed size independent
of K-shard layout, ring rotation, or `logical_n`.

### Target shape & hardware

- v1 dev shape: `total_seq = 20*1024`, `chunk_size = 5*1024`, `n_chunks = 4`.
- Post-port target: `total_seq = 55*1024`, `n_chunks = 11` (matches the
  `test_kimi_k26_mla_chunked_prefill` torch reference).
- `sp_size = 4` BH (single ring, quiet-box class). MLA configs only
  (`mla_100k`, `mla_128k`, both `NHK = 1`).
- Reference: `torch_joint_sdpa_reference` (not the SDPA op itself — independent
  from any regression in the classic causal codepath during the port).

## Architectural decisions

**A. Chunked prefill rides on the non-causal codepath, not `is_causal=True`.**

Adding `q_start_idx` to the causal codepath would force every chunked-prefill
regression to debug inside the softmax pipeline that non-chunked MLA prefill
already depends on. Instead we add **one new mask mode to the non-causal mask
builder**, gated by `q_start_idx.has_value()`. The `is_causal=True` branch
stays bit-for-bit untouched. Blast radius of bugs is bounded to chunked calls.

**B. The MLA `NHK == 1` K-share path must keep working.**

MLA broadcasts a single K head across all Q heads via the reader's
`k_uses_batch_chain` path. The Q-vs-K `N_local` split (next-step 1) must not
re-shard K. Test coverage stays on `mla_100k` / `mla_128k` until green;
WAN-style `NHK == NHQ` chunked variant comes later.

**C. `q_start_idx` is `Optional[int]`, defaulting `None`.**

`None` ⇒ legacy behavior, codepath unchanged. Any value (including `0`) ⇒
chunked-prefill mode — chunk 0 still needs the causal triangle and can't be
inferred from `q_start_idx == 0`. Mutually exclusive with `is_causal=True`
(`TT_FATAL`). Excluded from `attributes()` so it stays off the program-cache
key.

## Current state — chunked-prefill v1 GREEN

Steps 1, 3, and 4 all landed; step 2 skipped (no longer relevant after
step #3 made chunk 0 correct); step 5 skipped (no hangs surfaced).
mla_100k and mla_128k chunked tests pass all 4 chunks at PCC ≥ 0.999, with
`num_program_cache_entries == 1` (shared across chunks).

### Done

1. **Test scaffolding** — `run_ring_joint_sdpa_chunked` helper +
   `test_ring_joint_attention_sdpa_chunked_accuracy` over `mla_100k`,
   `mla_128k`. Per-chunk PCC/RMSE logged; `num_program_cache_entries`
   asserted == 1 at end. Driven by `is_causal=False` + `q_start_idx=i*c`.

2. **Validation split** in `ring_joint_sdpa_device_operation.cpp`:
   `N_local_q = q_shape[2]` vs `N_local_kv = input_k.logical_shape()[2]`.
   K-gather invariant restored as `N_global == N_local_kv * ring_size`.
   Conflated `q_shape[2] * ring_size == k_shape[2]` and
   `(N_global - logical_n) < N_local` checks dropped (chunked prefill
   intrinsically puts whole ring devices into all-padding K territory
   on early chunks).

3. **`q_start_idx` API surface** plumbed through 6 host files
   (`types.hpp`, device-op `hpp`/`cpp`, `sdpa.hpp`/`cpp`,
   `sdpa_nanobind.cpp`). Mutual-exclusion `TT_FATAL` with is_causal lands.

4. **Test refactored to 20K** dev shape with `torch_joint_sdpa_reference`
   as oracle (~3× faster iteration; SDPA-op regressions can't silently
   invalidate the per-chunk PCC).

5. **Step #1 — Q/K split** of `local_padded_N` Q-side vs K-side in the
   program factory + reader/writer/compute kernels. Boundary fix in
   `ring_iter_does_work` / `find_last_active_ring_iter`. See
   chunked_prefill_step1_findings.md.

6. **Step #3 — `q_start_idx` kernel plumbing + chunked-prefill mask.**
   `chunked_prefill_enabled` CT arg (writer/compute) + `q_start_idx_t` RT
   arg; absolute Q/K tile coords passed to `apply_causal_mask_lightweight`
   on every ring iter. Also fixed pre-existing "first non-skipped iter
   restores stale staging" bug via `is_first_active_iter` tracking in
   compute and writer.

7. **Step #4 — logical_n CT → RT for chunked-prefill.** Conditional drop
   from `attributes()` and `compute_program_hash`. Kernel keeps CT slot
   10/11 as layout hints (constexpr mask-CB layout); runtime decisions
   use RT `logical_nt` refreshed via `override_runtime_arguments`. RT
   slots gated on `chunked_prefill_enabled` to keep the
   `mla_100k-q160-k256` canary under the kernel-config buffer budget.

### Final PCC table (mla_100k, sp=4 BH, q_chunk=k_chunk=160, 20K dev shape)

| Chunk | logical_n |   PCC  |   RMSE  |
|------:|----------:|-------:|--------:|
|     0 |      5120 | 0.9997 | 0.0047  |
|     1 |     10240 | 0.9996 | 0.0069  |
|     2 |     15360 | 0.9994 | 0.0087  |
|     3 |     20480 | 0.9993 | 0.0099  |

No hangs, all 4 chunks complete, `num_program_cache_entries == 1`. mla_128k
matches within rounding.

### Historical pre-step-#3 state (kept for reference)

Before step #3, with `is_causal=True` driving the old causal path:

| Chunk | logical_n |    PCC | RMSE   |
|------:|----------:|-------:|-------:|
|     0 |      5120 | 0.6117 | 0.1541 |
|     1 |     10240 | 0.3533 | 0.2429 |
|     2 |     15360 | 0.2470 | 0.2778 |
|     3 |     20480 | 0.1873 | 0.2996 |

### Root cause of the (historical) PCC collapse

`ring_joint_sdpa_program_factory.cpp:176` derived

```cpp
const uint32_t ... local_padded_N = q_shape[2], ...;
```

and reused `local_padded_N` for both Q chunking (`num_local_q_chunks`)
and K chunking (`num_local_k_chunks = div_up(local_padded_N,
k_chunk_size)`). For chunked prefill `q_shape[2] = 1280` but the K
shard per device is `14080`; the kernel walked only `1280 / k_chunk_size
= 8` K-chunks per ring iter instead of `88`. Fixed in step #1 (Q/K split)
along with the causal cross-device gap fix in step #3.

## Next steps

Ordered so each step has a binary pass/fail signal that doesn't depend on
the next step working.

### 1. Split `local_padded_N` Q-side vs K-side

**Change.** In the program factory and kernel CT args, introduce
`q_local_padded_N = q_shape[2]` and
`kv_local_padded_N = input_k.logical_shape()[2]`. Audit every existing use:
- Q-side: `num_local_q_chunks`, Q ring offset (`ring_id * q_local_padded_N`).
- K-side: `num_local_k_chunks`, K ring offset (`ring_id * kv_local_padded_N`),
  `logical_nt` skip arithmetic, `local_n_has_padding`, `global_n_has_padding`.
- K-share path (`k_uses_batch_chain` when `NHK == 1`) must still pull the
  single K-head shard correctly.

**Test.** Run the chunked test as-is (`is_causal=True`, no `q_start_idx`).
Expected after this step:
- **Chunk 0 PCC → oracle** (≥ 0.99). Its mask semantics are already correct
  — Q rows [0..Sq), K cols [0..Sq), pure causal triangle — the only thing
  wrong was the kernel not iterating over the full K.
- **Chunks 1+ remain broken**: mask still treats Q row 0 as absolute
  position 0.
- Also re-run the non-chunked `test_ring_joint_attention_sdpa_accuracy` to
  confirm no regression on the classic path.

### 2. Padded-K-tail probe (`total_seq` sweep on chunk 0)

**Change.** Test-only: parametrize the chunked test over
`total_seq ∈ {20K, 40K, 55K}` with the same `chunk_size = 5120`.

**Test.** Run **after step 1**. Chunk 0 sees the same real K rows
regardless of total cache size, so:
- **Chunk 0 PCC must be identical across all three `total_seq` values.**
- If equal → padded-tail was an artifact of step 1's broken K traversal,
  no further work needed; skip what was originally next-step 5.
- If unequal → kernel is reading past `logical_nt`. Fix the skip
  (`local_padded_Nt * ring_id + k_chunk * Sk_chunk_t >= logical_nt` must
  fire **before** any tile lands in the CB) before moving on, so step 3's
  correctness signal isn't muddied.

### 3. Wire `q_start_idx` to the kernel + chunked-prefill mask mode

**Change.** Bundle with step 4 (same RT-arg plumbing pass).
- Program factory: **CT arg** `chunked_prefill_enabled = q_start_idx.has_value()`.
  **RT arg** `q_start_idx_value` (0 when disabled).
- Kernel: extend the non-causal mask builder. When the CT bit is set,
  within K cols `[q_start_idx, q_start_idx + Sq)`, apply the standard
  row-major causal pattern with
  - Q absolute row = `q_start_idx + ring_id * q_local_padded_N + local_q_row`
  - K absolute col = `q_start_idx + k_chunk * Sk_chunk + within-chunk col`
  Outside that strip: identity left of `q_start_idx`, existing `logical_n`
  cutoff to the right.
- `is_causal=True` branch untouched.
- Reference pattern: `chunk_start_idx_tensor` runtime-arg plumbing in
  `chunked_flash_mla_prefill`, minus the device tensor (host knows
  `q_start_idx`; one register write beats a DRAM read).

**Test.** Flip the chunked test to `is_causal=False` + `q_start_idx = i*chunk_size`.
- **All 4 chunks hit oracle PCC** (≥ 0.99).
- Sanity check: chunk 0 with `q_start_idx=0` + `is_causal=False` must match
  step 1's chunk 0 result numerically — same mask, different codepath.

### 4. Move `logical_n` from CT to RT

**Change.** `get_compile_time_arg_val(10)` in `ring_joint_sdpa.cpp:27`
becomes a runtime arg; drop `logical_n` from the program-hash.

**Test.** Pure programmatic check, no PCC needed:
- `mesh_device.num_program_cache_entries()` after the chunked loop must
  collapse from `n_chunks + 1` to **2** (one for oracle shape, one for the
  chunked shape).

### 5. Chain-bounds audit with variable `logical_n` (skipped — no hangs)

**Status (post step #3 + step #4):** **NOT NEEDED.** Across 3 consecutive
chunked test runs on mla_100k with sp=4 and `logical_n=5120` (the
worst-case shape where 3 of 4 ring shards carry all-padding K on chunk 0),
no hangs surfaced. Chain bounds are correct for chunked-prefill as-is.

The pre-existing "first non-skipped ring iter restores stale staging"
issue surfaced as a correctness failure, not a hang — it was fixed in
step #3 via `is_first_active_iter` tracking. If a future config does
introduce hangs (e.g., larger ring with extreme padding imbalance), the
prescription would be:

- Per [[feedback_chain_bounds_uniform_q]]: enforce
  `q_iter < real_q_per_core` on the K side as well as the Q side when K
  is short.
- Probe: smaller `logical_n` values on higher-ring configs; watch the
  test log for a stall.

### 6. Test follow-ups (post-port)

- Scale `total_seq` back up to 55K (matches Iva's torch reference).
- Tighten `pcc_threshold` (0.99 → noise-floor of oracle vs SUT).
- Promote out of `@skipif(CI)`.
- In-place device K/V writes instead of full-cache re-upload per iter.
- Add a WAN-style `NHK == NHQ` chunked variant.

## Out of scope for v1

- End-to-end MLA correctness (projections / RoPE / o_proj) — separate
  test, retargets `test_kimi_k26_mla_chunked_prefill.py`.
- Performance: correctness first, perf later.
- Chunk sizes ≠ 5120, total seqs other than 20K (dev) / 55K (final).
- Multiple `(q_chunk_size, k_chunk_size)` pairs per model.
- `is_balanced=True`.
