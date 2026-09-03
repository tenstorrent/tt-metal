# TensorParameter relaxations — `transformer/sdpa` (sparse SDPA)

**Author:** Audrey and Claude

**Purpose:** State the Metal 2.0 `TensorSpecRelaxations` declaration these factories require, per `TensorParameter`, so that neither the auditor nor the porter has to derive it. The readiness sheet's `TensorParameter relaxation` cell points here.

**Covers two sheet rows:**

| Op | Device operation | Factory |
|---|---|---|
| `transformer/sdpa` | `SparseSDPAOperation` | `SparseSDPAProgramFactory` |
| `transformer/sdpa` | `SparseSDPAMsaOperation` | `SparseSDPAMsaProgramFactory` |

The other `transformer/sdpa` rows (`RingJointSDPA*`, `ExpRingJointSDPA*`) are **not** covered — they carry `(legality - pending analysis)`, a different question.

---

## Contract — read this before using the document

This is **not** the same contract as the [offset-base-pointer](../2026-07-19_offset_base_pointers.md) and [3rd-argument](../2026-07-06_tensor_accessor_3rd_arg_triage.md) triage docs. Those are *priors* layered on a scan the auditor runs anyway, so a disagreement means "the doc is stale, trust your own scan."

Here there is no scan to fall back on: the audit recipe forbids re-deriving a relaxation. So this document is **authoritative but perishable**, and the only staleness signal available to a reader who cannot re-derive it is the stamp below.

> **If the staleness check fails, do not substitute your own judgement. Stop, and report the relaxation verdict as UNCONFIRMED.**

### Validity check — confirm these three still hold

Each is one grep. They are the facts the declaration rests on; if all three hold, the analysis holds regardless of what else changed in the op.

1. **The hash still gates on the same two disjuncts.** `compute_program_hash` includes the KV tensor's `logical_shape` *only* when that tensor is sharded or `has_block_cyclic()`, substituting an empty `Shape{}` otherwise — `sparse_sdpa_device_operation.cpp` (`kv`) and `sparse_sdpa_msa_device_operation.cpp` (`k` and `v`, branching independently). **If the branch condition changed, the conditions in §2 are wrong.**
2. **The hit path still pins the KV width and rank.** `validate_non_hashed` re-`TT_FATAL`s `kv[3] == expected_kv_width` and rank 4 on every dispatch, and is still called from `validate_on_program_cache_hit`. **If that check is gone, `match_page_size` is no longer justified and rank is no longer pinned.**
3. **Q, indices and output are still `TT_FATAL`'d to ROW_MAJOR / DRAM / interleaved / `padded == logical`.** That is what makes strict the exactly-right declaration for them rather than an over-tightening.

**If any check fails, report the relaxation verdict as UNCONFIRMED** rather than substituting your own reading.

*Provenance, not a gate:* analysed at `28994778430`, whose `transformer/sdpa` tree is byte-identical to `origin/main`; the four source files were last touched by `69e7e920fc9` (2026-09-02) and the relaxation framework by `29378ce8b50` (2026-09-03). A commit stamp is deliberately **not** the check here — these files carry many unrelated SDPA factories, and a stamp that fires on every unrelated commit trains its reader to ignore it.

---

## 1. For the auditor — verdict and routing

**Both factories: CLEAN.** The relaxation gate conjunct clears for both.

- The op is correct as written; no cache-key defect was found in either.
- Everything each op legitimately does is expressible in the four-bool vocabulary.
- The geometry pin that rides on `dynamic_tensor_shape` introduces **no regression**: every slot taking the flag is interleaved, so the shard term is `nullopt` on both sides and compares nothing.

This clears **the relaxation conjunct only**. Both rows carry `Is able to port? = no`, which this analysis does not address.

One flagged non-blocker, on `SparseSDPAMsaProgramFactory` only: the declared-strict `q` and `indices` slots are marginally *tighter* than the op's tolerance, on `MemoryConfig` detail fields and `Alignment` — fields the op neither hashes nor validates. Strict is the loud direction and both slots have their shape hashed, so nothing legitimately varying is rejected. Recorded for completeness, not routed.

---

## 2. For the porter — what to write

Both factories evaluate their conditions inside `create_descriptor`, which runs only on a cache miss and holds the actual tensors and attributes. Every predicate below is a direct expression of the branch the op's own `compute_program_hash` already takes — if you find yourself computing something the hash does not, you have the wrong predicate.

### `SparseSDPAProgramFactory` — one `kv` slot

| Slot | Condition | Declaration |
|---|---|---|
| `q`, `indices`, `output` | always | strict — no relaxation |
| `kv` | `!t.kv.memory_config().is_sharded() && !attrs.has_block_cyclic()` | `{.dynamic_tensor_shape = true, .match_page_size = true}` |
| `kv` | `t.kv.memory_config().is_sharded()` | strict — no relaxation |
| `kv` | `!t.kv.memory_config().is_sharded() && attrs.has_block_cyclic()` | strict — no relaxation |

### `SparseSDPAMsaProgramFactory` — separate `k` and `v` slots

| Slot | Condition | Declaration |
|---|---|---|
| `q`, `indices`, `output` | always | strict — no relaxation |
| `k` | `!t.k.memory_config().is_sharded() && !attrs.has_block_cyclic()` | `{.dynamic_tensor_shape = true}` |
| `k` | `t.k.memory_config().is_sharded() \|\| attrs.has_block_cyclic()` | strict — no relaxation |
| `v` | `!t.v.memory_config().is_sharded() && !attrs.has_block_cyclic()` | `{.dynamic_tensor_shape = true}` |
| `v` | `t.v.memory_config().is_sharded() \|\| attrs.has_block_cyclic()` | strict — no relaxation |

> **Use `v`'s own `memory_config` for the `v` rows, not `k`'s.** The hash branches per tensor, so a mixed configuration — `k` sharded, `v` interleaved — is a real, distinguishable case. Testing `k` twice silently mis-declares `v`.

### The one rule that differs between the two factories, and why

`SparseSDPAProgramFactory` sets `match_page_size`; `SparseSDPAMsaProgramFactory` does not. Both are correct. The rule:

> **Set `match_page_size` iff the op independently pins the last-dim width.**

Sparse SDPA does — `validate_non_hashed` re-`TT_FATAL`s `kv[3] == expected_kv_width` on every dispatch — so the width cannot vary and pinning it costs nothing while keeping `aligned_page_size` a static CTA. binary_ng's width genuinely varies, so pinning it would reject legitimate calls. For MSA the flag is **inert regardless**: K and V are `TT_FATAL`'d to TILE, and `match_page_size` has no codegen effect off the interleaved row-major path.

This is the highest-probability transcription error in the pair of documents. If you are copying a row, check which factory you are in.

---

## 3. Why — the derivation

Both hashes use the same device: on the relaxed branch they substitute an **empty `Shape{}` sentinel** for the KV tensor's `logical_shape`, which is precisely what lets the batch and sequence dimensions vary across dispatches sharing one program.

**`SparseSDPAProgramFactory`, `kv` Regime A.** With the sentinel in play, `kvs[0]` (B, the cache's batch slots) and `kvs[2]` (T, the cache length) vary freely. Dim 1 is pinned to 1, dim 3 to `expected_kv_width`, and the rank to 4 — all by the hit-path `TT_FATAL`s, which run on every dispatch. So the tolerance is exactly "dims 0 and 2 of the logical and padded shape", which `dynamic_tensor_shape` grants and `match_page_size` then narrows back to the op's real invariant. Asserted live: `test_sparse_sdpa.py` sweeps T ∈ {256, 512, 1024} into one cache entry.

**Regimes B and C are strict for opposite-looking but identical reasons** — in both, the hash *includes* `kv.logical_shape()`, so nothing varies within an entry and the tightest admitting declaration is strict. Regime C must hash it: the factory folds T into two compile-time stride constants (`BC_SHARD_STRIDE_GAP`, `BC_SLAB_STRIDE_GAP`), so a T change on a cache hit would use stale strides.

**MSA `k` and `v`.** Same branch structure per tensor. `v`'s dims 0 and 2 are `TT_FATAL`-locked in lockstep to `k`'s on every hit, so they float exactly when `k`'s do. All T- and B-dependent values are runtime args. Asserted live: nightly `test_sparse_sdpa_msa.py` runs T ∈ {256, 512, 1024} against interleaved K/V in a single entry, and the ND-sharded variant of the same test correctly produces two entries.

**Why `relax_logical_rank` is absent everywhere.** Both ops pin rank 4 by `TT_FATAL` on every dispatch. The flag would be inert, and declaring an inert flag misleads the next reader about what the op tolerates.

---

## 4. Not covered

- `RingJointSDPADeviceOperation` and `ExpRingJointSDPADeviceOperation` — separate sheet rows, `(legality - pending analysis)`.
- Everything except the relaxation declaration. This document makes no claim about either op's portability on any other axis.
