# Speculative decode: the draft length K by the live batch (plan row S1)

User direction (2026-10-29): "you need to have different K value based on batch size for best performance."

## Today

* One K per launch: `tt/vllm_spec.py::resolve_spec_decision` caps the catalog's `num_speculative_tokens` by
  `B (K + 1) <= 32` with B = `max_num_seqs` (the users per row), and `SpecPolicy.draft_len` is the K of the captured
  V / C / D graphs. A launch at 28-32 users per row gets K = 0 (no speculation at all), a launch at 4 users per row
  gets K <= 7 whether 1 or 4 users are live.
* The verify is wired for REPLICATED caches only (`DsaSpecState`: `cache.partition` refused; served speculation runs
  the 128 K profile at 8 users per row = the 1 M positions the planner fits). The user-partitioned cache
  (`GLM53_PARTITION_USERS=1`, the serving default at 32 users per row and 1 M) has no verify path.
* The sweep (`tests/test_mtp_spec.py -k spec_decode_lossless`, 45 layers, H1/H2 defaults): the acceptance plateaus at
  ~2.5 drafts per step whatever K, so the shortest K wins (K=3: 24 tok/s per user at 2 per row; K=7: 23.9); at the
  bound itself, K=15 with 2 users per row (B (K + 1) = 32) gives 15.4 tok/s per user AND breaks losslessness: one row's
  two users diverge from plain greedy at token 4 with a 2.69 gap (2026-10-29 13:22, no near-tie). B4_K7 and B8_K3
  (both at the bound) fail the same way: every user of mesh row 5, token 4. So the full tile is an open verify defect
  (`tt/spec_decode.py` candidate-major layout, row 5 specific) and the policy stays below it: `VERIFY_ROWS_CAP` = 31
  (B (K + 1) <= 31) and the table (1, 3), (2, 3), (4, 3), (8, 2), (16, plain). B8_K3 would have given 19.9 tok/s per
  user (1274 aggregate, 3.2 accepted) -- the prize for fixing the full tile.
* The plugin contract already allows a per-row draft count below the width: `propose_draft_tokens` returns
  `draft_token_ids [rows, K_max]` + `num_valid [rows]` (`model_runner.py`), and a row offering 0 takes the plain step.

## The policy

`k_by_rows`: a table of (users per row live, K) measured on the harness (`tests/test_mtp_spec.py` at 45 layers on the
8x4, the `spec_sweep_8x4_B{B}_K{K}` stages): the K with the best tok/s per user at each B among those with
B (K + 1) <= 32, e.g. (1, 15) / (2, 7-15) / (4, 3-7) / (8, 3) / (16, 1) / (32, 0). The step's K = the table entry for
the number of LIVE users per row (the largest B in the table >= the live count); K_max (the plugin's width) = the
table's largest K. Per step: every live row is offered `num_valid = K(B_live)`; above the cliff (no entry) the rows
offer 0 and the step is the plain decode: no occupancy loses the baseline throughput.

## The mechanism (two prerequisites)

1. **Per-entry graph sets.** The verify (V), commit (C) and draft chain (D) graphs are shape-bound: R = B_sub (K + 1)
   candidate rows per mesh row. One `SpecDecoder` graph set per table entry (B_sub, K), captured at boot after the
   decode trace (the same allocation order rule as today: every persistent input before the first capture). Trace
   memory: ~one decode-trace equivalent per entry (3-4 entries).
2. **Batch compaction.** A (B_sub, K) graph verifies B_sub user SLOTS, but the live users sit in arbitrary physical
   slots of the full B. Today's graphs address users by slot index (the KDA state row u of the batched state, the
   DSA page-table row u replicated K + 1 times, the ring / pool ids). The compacted graph takes the live users'
   physical slot ids as device inputs: the KDA state rows gathered by a one-hot matmul into the B_sub working state
   (as `select_onehot` / the wave slot's `cache_idx` do), the verify's page table built from the live slots' rows
   (`_replicated_table` from a device `[B_sub]` index), the commit scattered back the same way. The plain decode
   step keeps running all B slots; only the speculative graphs compact.
3. **Partition-aware verify** (for the partitioned cache at 28-32 users per row and 1 M): the verify's dense
   flash-MLA over the user's pages must run on the OWNER column with all heads (the heads2users prefill partition's
   analog, `TtDSA.prefill_forward`'s `owner_of(user)` path), the candidates' latent appends and the index commit
   routed to that column's slot. Without it the table stops at the replicated profile (8 users per row).

## Order of work

1. The sweep table (running) -> `k_by_rows` in `tt/vllm_spec.py` (launch-time first: the best K for `max_num_seqs`,
   a one-line lookup replacing the fixed catalog K; host-tested).
2. Per-step K from the live users (the policy + `num_valid`), still one graph set at K_max: lossless, no gain yet.
3. Per-entry graph sets + compaction (the gain at low occupancy on a large-batch launch); block tests at 4 layers
   (the served spec test extended with a half-empty batch), then the 8x4 harness.
4. The partition-aware verify (the 1 M / 32-user profiles).
