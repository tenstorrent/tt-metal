# Migration Audit — rolled-up summary (`mcast_pipe`)

Consolidated from 6 per-group audits. The census is the whole-codebase intra-chip mcast+handshake
block inventory; this is the **pre-migration blocker view**.

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

6. **Legacy-API call sites** (move, sort) would need a `Noc`/`Semaphore` port *before* migration —
   tag refactor with that prerequisite noted.

## Clean set (the easy wins that prove the API)
Canonical two-sided P1/C1 pairs: matmul in0/in1 sender+receiver (4), conv weights sender+receiver
(4), ln_post_allgather sender+receiver (2), topk receiver + sampling + kv_cache + rms_sender +
llama worker_receiver + gn_v2 receiver. These are `(EXCLUDE_SRC or INCLUDE_SRC, flag, flush-or-none,
pre_handshake known)` — the spine the bake-off and ★ build on.

## Defer/out-of-scope (not this round)
- Ring/unicast (matmul in0_ring, sdpa ring legs, sort cross-core unicast) — not rectangle-mcast.
- Fabric / cross-chip CCL legs (all_reduce worker_writer, all_to_all fabric leg) — intent exclusion.
- ~15 data_movement single-core / no-mcast kernels — not the block.
