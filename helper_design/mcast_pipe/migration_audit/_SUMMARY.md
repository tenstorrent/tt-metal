DERIVED FROM: migration_audit/{matmul,conv,normalization,transformer_sdpa,data_movement_reduction,ccl_deepseek_examples}.md and current migration/ledger.json

# Migration Audit — rolled-up summary (`mcast_pipe`)

Consolidated from 6 per-group audits. The inventory is the whole-codebase intra-chip mcast+handshake
block inventory; this is the **pre-migration blocker view**.

> **Historical snapshot.** This summary preserves the pre-migration audit and
> dated apply outcomes; it is not current rollout state. As of 2026-08-23 the
> authoritative ledger contains 108 kernels: 31 migrated at API v14, 2 pending,
> and 75 deferred.

## Counts (block-containing kernels; excludes naming false-positives & no-mcast incidental)

| Group | clean | refactor | defer/raw | ref(prior-art) | oos(CCL) | blocks |
|---|---|---|---|---|---|---|
| matmul | 8 | 3 | 2 | – | – | 13 |
| conv | 4 | 2 | 1(partial) | – | – | 7 |
| normalization | 3 | 11 | 0 | – | – | 14 |
| transformer/sdpa | 1 | 3 | – | 1 (chain_link.hpp) | 2 (fabric) | 7 |
| data_movement/reduction | 1 | 7 | 15 (incidental no-mcast) | – | – | 8 true |
| ccl/deepseek/examples | 6 | 12 | – | 2 (mcast.hpp, dataflow_utils) | several (fabric legs) | ~20 |
| **TOTAL (approx)** | **~23** | **~38** | **~18** | **3** | **~5+** | **~69** |

## Headline blockers (the reason we censused before touching ops)

1. **Forks are ternary, not binary** (F1 fence ×3, F2 staging ×3, F3 loopback ×3) and a **new
   F4 linked/barrier** fork — the bake-off space is bigger than Step B assumed. *All* observed in
   production; nothing speculative. → Step E must bake the real 3-way forks, not 2-way.

2. **Rotating-sender / role-flip hybrids** (matmul block-sharded `..._in0_sender_receiver...`,
   group_attn) — one core is sender for block b and receiver for block b′, sender identity rotates
   per-iteration. A fixed sender-object + receiver-object model **cannot express this**. This is the
   single biggest threat to the two-sided `Pipe` premise → **Step ★ must rule on it** (likely:
   `receive(sender_coord, …)` takes a per-call sender; role-flip kernels tagged `refactor`/`defer`).
   - **UPDATE (Round 9, feedback-4.txt):** the rotating-role STAR **IS migratable** with two Pipes (a
     `SenderPipe` + a `ReceiverPipe` sharing the `data_ready` cell, `receive(sender_x, sender_y)`
     taking the rotating coord) — once the **M12b** Flag-path fix lands (re-assert the source cell VALID
     per send; see hazards_catalog H12 amendment). The earlier "cannot express / confirmed hard, same-core
     sender+receiver hangs" verdict was a **symptom of the Round-6 ctor-once-VALID decision** (the
     receiver turn clobbers the shared cell INVALID, so the once-set source went stale), NOT proof of
     infeasibility. block_sharded was quarantined at this audit point; M12b later lifted it. group_attn additionally
     carries an F1 barrier-after-flag disagreement (matmul.md #5) — a separate refactor cost, unchanged.

3. **Flag-only sends with no data** — the *entire* data_movement/reduction group, plus ln_pre and
   gn "go" flags, never call `noc_async_write_multicast` at all (4-byte sem mcast only). A Pipe that
   bundles data+flag must also support **flag-only** (R2). → ★ API must make the data payload optional.

4. **Multi-rectangle dest sets** (R1), **chunked send > NOC_MAX_BURST_SIZE** (R4), **phase-granular
   interleaving** (R3), **NOC1 coordinate swap** (R5) — generality requirements that the "ANY
   rectangle / ANY addr / ANY size" intent commits to but the naive single-rect `send(src,dst,size)`
   sketch doesn't yet cover. → ★ must show the API absorbs these or scopes them out.

5. **Two strong prior-art Pipe shapes already exist** — `deepseek_v3_b1/unified_kernels/mcast.hpp`
   (`deepseek_b1_ops::Mcast`: init/op/teardown, CT-dispatched sender/receiver by core role, unified
   DMArgs) and sdpa `chain_link.hpp` (`ChainLink`). These are the **design templates** for ★ and the
   bake-off baseline. (Both use raw NOC set-state or raw API — the object-API rebuild is this run's job.)

6. **Legacy-API prerequisite narrowed to move.** Sort is now on `Noc`/`Semaphore<>`; its upstream
   ready/done split removes the old single-counter ambiguity. A focused 2026-08-03 re-audit finds its
   phase broadcast expressible by API-v9 Counter `send_signal`/`receive_signal`, with both return
   counters left op-owned. It later migrated at API v9 in `7337302b564`; Move
   subsequently migrated and is stamped at API v14.

## Step-C re-entry delta (2026-08-03)

- Re-audited the three churned sort protocol halves; no sort-directory recall miss.
- Tags and aggregate counts stay unchanged (`refactor`), but the old `DEFER-DESIGN-GAP` rationale no
  longer holds for coordinator/reader at API v9.
- Writer is a helper-neutral `done`-counter companion; its ledger disposition must be resolved
  as part of the atomic sort plan.
- Downstream outcome: Step D mapped sort to the existing v9 API, Step G added the focused
  control-only Counter unit case, and apply completed without an API bump.

## Apply/reconcile outcome (2026-08-03)

- Step G added the focused Counter signal-only coverage; helper suite 72/72.
- Coordinator and reader migrated at API v9 in `7337302b564`; writer remains helper-neutral after
  coupled runtime-ABI cleanup.
- Host build, exact fresh-cache `--dev` path, Ht=2 deadlock pair, and full 7-case long-tensor
  inventory all passed. The ledger records the completed two-kernel migration.

## Width-sharded Conv apply/reconcile outcome (2026-08-03)

- The production activation reader and its width-sharded factory migrated atomically at API v9 in
  `fe866a1d0c4`; no helper change, style bake-off, API bump, or quarantine was required.
- Exact fresh-cache BF16/BF16 filter-3 coverage passed under `--dev` at PCC 0.999956503 with the
  intended JIT path confirmed. The complete feature inventory passed 48 cases with 16 legitimate
  skips; the DRAM-config route passed at PCC 0.998234911.
- The result confirms that ACK-fenced real-loopback completion was the missing invariant behind the
  prior attempt's 25 numerical failures. At that checkpoint the fleet was 13
  migrated kernels and 12 migrated host bindings, with 78 deferred and nothing
  pending or quarantined.

## Clean set (the easy wins that prove the API)
Canonical two-sided P1/C1 pairs: matmul in0/in1 sender+receiver (4), conv weights sender+receiver
(4), ln_post_allgather sender+receiver (2), topk receiver + sampling + kv_cache + rms_sender +
llama worker_receiver + gn_v2 receiver. These are `(EXCLUDE_SRC or INCLUDE_SRC, flag, flush-or-none,
pre_handshake known)` — the spine the bake-off and ★ build on.

## Defer/out-of-scope (not this round)
- Ring/unicast (matmul in0_ring, sdpa ring legs, sort cross-core unicast) — not rectangle-mcast.
- Fabric / cross-chip CCL legs (all_reduce worker_writer, all_to_all fabric leg) — intent exclusion.
- ~15 data_movement single-core / no-mcast kernels — not the block.
