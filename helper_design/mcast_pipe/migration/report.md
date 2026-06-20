# mcast_pipe rollout — final report (autonomous backlog run, 2026-06-20 @ v7)

## Run header
- **Helper version (CURRENT):** `MCAST_PIPE_API_VERSION 7`. Unit test `test_mcast_pipe.py` green (45/45). No version bump this run.
- **Entry mode:** re-entry. **Migration mode:** `run-all`. **Machine:** single-chip **Wormhole b0 (n150, 64 worker cores)**.
- **Scope:** (1) lift `block_sharded` from quarantine; (2) tier + migrate all 48 pending kernels; (3) keep the 8 design-deferred kernels deferred.
- **Headline:** block_sharded lifted (the helper's per-send flag-VALID re-assert fix worked). Of the 48 pending, **6 net-new migrated + device-verified**, **42 deferred** — 19 on genuine helper **design gaps** (route to tune-dm-helper), 23 on **coverage** (no test runnable on this single chip). The helper was **never modified** (correct — design fixes are tune-dm-helper's job).

## Rollout state @ v7 (from ledger.json) — 69 entries
| status | count |
|---|---|
| **migrated (current @ v7)** | **19** |
| pending | **0** |
| quarantined | **0** |
| deferred | **50** (8 original design-deferred + 19 new design-gap + 23 coverage-gap) |

0 pending and 0 stale → **the migratable-on-this-chip fleet is fully current at v7.** Every remaining kernel is deferred for a documented reason (helper design gap or no single-chip coverage).

## Immediate priority — block_sharded lifted ✅
- `reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp`: restored the clean v7 two-Pipe rotating-role STAR call site from `fa561f3b584`. The helper fix `20cf0df46ee` (re-assert source-cell VALID per send) removed the Round-6 ctor-once-VALID staleness that hung this rotating-role core.
- Validation: `test_matmul_2d_multiple_output_blocks_per_core --run-all` = **56 passed, 72 skipped, 0 failed, no hang**.
- Ledger: quarantined → migrated@v7 (commit `7953a2a2e52`). **0 quarantined remain.**

## Per-tier results (the 48 pending backlog)
| tier | scope | migrated | deferred (design) | deferred (coverage) |
|---|---|---|---|---|
| 1 | clean, single-chip-verified | **2** | 1 | 1 |
| 2a | normalization sharded refactor | **3** | 2 | 0 |
| 2b | reduction + matmul + conv refactor | **1** | 5 | 0 |
| 2c | sdpa refactor | 0 | 2 | 0 |
| 2d | deepseek/moe single-device | 0 | 4 | 1 |
| 3 | legacy-API sort + move | 0 | 5 | 0 |
| — | (coverage-gap, not tiered — unverifiable on 1 chip) | — | — | 21 |
| **total** | | **6** | **19** | **23** |

### Migrated this run (6 net-new, all device-verified, 0 failures)
| kernel | tier | test | result |
|---|---|---|---|
| `reader_mcast_sender_unary_sharded_ln_post_allgather.cpp` | 1 | test_post_allgather_layernorm | 64/0 |
| `reader_mcast_receiver_unary_sharded_ln_post_allgather.cpp` | 1 | test_post_allgather_layernorm | 64/0 |
| `reader_mcast_receiver_unary_sharded_ln.cpp` | 2a | test_layer_norm_sharded_single_stage | 64/0 |
| `reader_mcast_sender_unary_sharded_ln_pre_allgather.cpp` | 2a | test_pre_allgather_layernorm | 32/0 |
| `reader_mcast_receiver_unary_sharded_ln_pre_allgather.cpp` | 2a | test_pre_allgather_layernorm | 32/0 |
| `writer_local_topk.cpp` | 2b | test_topk (W=8192) | 80 passed / 80 xfail / 0 fail |

(plus block_sharded lifted = 7 kernels brought to green this run. Migrated total 12 → **19**.)

### Note on commit hashes
Per-kernel atomic commits + paired ledger write-backs throughout (revert-clean, bisectable). All local — **nothing pushed/rebased/reset**. Migrated-this-run commits: `3a0315dea95`, `fcec91acc5c`, `f348c649e02`, `47f62dbd700`, `e6e5b15ef91`, `a362b90343a` (+ block_sharded `7953a2a2e52`).

## Coverage gaps (23 — migrated NOTHING here; cannot gate on this chip)
Not migrated because no test reaches them on a single WH n150. **NOT silent breakage** — each kernel left in its known-good raw state (the one stale-v4 leftover, `sampling_kernel.cpp`, was restored from dead `Pipe<>` to raw, commit `f67d2b3ce96`).
- **3 matmul programming-example readers** — binary-only (`metal_example_matmul_multicore_reuse_mcast`), no pytest.
- **4 interleaved group_norm** kernels (`reader_mcast_*_unary_gn`, `welford_reader_mcast_*_unary_gn`) — sweep-only (every unit test shards input → v2 factory).
- **1 deepseek sampling** (`sampling_kernel.cpp`) — only `requires_grid_size(101)` tests drive it; this chip has 64 cores.
- **1 moe_gate_mm** (`dm1.cpp`) — `test_moe_mm` unconditionally `pytest.skip()`s on wormhole_b0 + blackhole (issue #44858).
- **14 multi-device CCL / deepseek-prefill / ring** kernels — require mesh ≥2 (rms_allgather ×2, llama AG-matmul worker_receiver, deepseek_prefill dispatch/combine/unified_ffn, moe_gpt tilize ×2, selective_reduce_combine ×2, all_gather_concat_writer, all_to_all_sender_writer, sdpa exp_ring_joint_reader, persistent_h2d_receiver).

## Known gaps — the 8 original design-deferred (stay deferred, per request)
chain / ring / fabric / deepseek-preprogram: `chain_link.hpp` (ref), matmul `in0_ring_all_gather` + `in1_ring_all_gather`, llama `in1_ring_all_gather`, all_reduce `worker_writer.cpp` (fabric), multicast example `coordinator_kernel.cpp`, deepseek `mcast.hpp` + `dataflow_utils.hpp` (preprogram-state).

---

# DESIGN-BLOCKED — needs a helper change (route to tune-dm-helper)

19 pending kernels are migratable in PRINCIPLE but the **v7 API cannot express them**. The helper was NOT
modified (per the failure policy). Grouped by the specific limitation; each is a candidate scope item for a
future `tune-dm-helper` round. (Several kernels hit more than one; listed under the dominant blocker.)

### D1 — Runtime recipient count (`num_dests` is a `get_arg_val`, but `NUM_ACTIVE_RECEIVER_CORES` is a compile-time template) — THE dominant blocker
The host factory computes per-core recipient counts at runtime (varies by core/worker-type). v7 can only
take the count as a compile-time template arg.
- `reader_mcast_sender_unary_sharded_gn_v2.cpp`, `welford_reader_mcast_sender_unary_sharded_gn_v2.cpp` (per-rect runtime counts)
- `flash_mla.hpp` (`num_mcast_dests` runtime)
- `tilize_reader.cpp` / `tilize_writer.cpp` (moe_compute) (3 rects, runtime counts)
- `experimental/conv3d/.../writer.cpp` (Mcast-mode runtime `mcast_num_dests`)
- `dataflow_common.hpp` (sdpa_decode `read_k`) (runtime `num_dests` + D4)
- `move_interleaved_with_overlap.cpp`, `move_stick_layout_interleaved_with_overlap.cpp` (D2/D3 too)
- **Fix:** a SenderPipe runtime-`num_dests` mode (count as a ctor/`send()` runtime arg).

### D2 — Split mcast-dest count vs consumer-ack count (one count can't serve both)
The data/flag broadcast goes to one set of cores; the consumer-ready ack arrives from a different-sized set.
- `reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp` (mcast = total_num_cores-1 incl. noop; ack = total_active-1) — **A/B hang-site flip confirmed on device**
- `reader_bmm_tile_layout_in0_sender_dram_sharded.cpp` (grid mcast vs dram-bank ack; contradictory across worker types)
- `coordinator_single_row_multi_core.cpp` (sort) (start=`number_of_dest` vs substage=`Wt/2`)
- **Fix:** separate `MCAST_DEST_COUNT` and `CONSUMER_ACK_COUNT`.

### D3 — Runtime sem ids (data-ready / consumer sem ids are compile-time template params)
- `reader_mcast_transformer_group_attn_matmul.cpp` (`Semaphore<>(get_arg_val(i++))`)
- sort `coordinator_single_row_multi_core.cpp`, `reader_single_row_multi_core.cpp`
- move `move_interleaved_with_overlap.cpp`, `move_stick_layout_interleaved_with_overlap.cpp`
- **Fix:** allow sem ids as runtime ctor args.

### D4 — Runtime sender/receiver role under one binary (helper faces are distinct compile-time types)
`do_mcast`/`is_controller` selects the role per-core at runtime; v7 SenderPipe and ReceiverPipe are separate compile-time types.
- `dataflow_common.hpp` (sdpa_decode, `do_mcast`), move kernels (`is_controller`)
- **Fix:** a unified role-dispatched Pipe (CT or runtime predicate selects the face).

### D5 — Arbitrary / value-carrying flag (`set_multicast(value)` + `wait(value)`)
Broadcasting a data value (token counts, monotone k+1) in the sem word. v7 has only Flag (fixed VALID) and Counter (inc+1 / wait_min).
- `reader_argmax_interleaved_multicore.cpp` (monotone value k+1)
- `tilize_writer.cpp` (moe_compute) (running token-count value)
- **Fix:** a `send_value(v)` / `wait_value(v)` verb.

### D6 — Multi-rectangle with per-rect modes (1–3 rects, mixed INCLUDE_SRC/EXCLUDE_SRC, single trailing barrier)
A per-rect `.send()` loop is fine for compile-time counts, but these combine D6 with D1/D5 making them unexpressible.
- `reader_argmax_interleaved_multicore.cpp` (2 rects, different loopback modes in one logical send)
- gn_v2 senders, moe_compute tilize_reader, move kernels
- **Fix:** a rect-list `send()` where each entry carries its own mode + count.

### D7 — CHAIN / relay (store-and-forward) topology (only STAR rectangle-mcast supported)
Each link is receiver+sender, forwarding via cross-id `relay_multicast(valid_sem → receiver_sem)`, src≠dst (STAR SenderPipe asserts src==dst).
- `reader_interleaved.cpp` (sdpa prefill, open-coded twin of the deferred `chain_link.hpp`)
- **Fix:** the long-known CHAIN/relay topology mode (same gap as `chain_link.hpp`).

### D8 — Producer-overlapped streaming chunked send (`send()` handles only fully-ready blocks)
`mcast_block_chunked` is an R4 streaming send: per-burst growing `wait_front`, payload > NOC_MAX_BURST. R4 was explicitly out of scope this round.
- `reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp`
- **Fix:** a streaming/chunked `send()` that interleaves producer `wait_front` per burst.

### D9 — Preprogram-state mcast (set-state / issue-txn split; shared cmd buf) — no v7 set-state path
- `kv_cache_update.hpp` (deepseek) — `proposed_helpers.md` already lists set-state as out-of-scope.
- **Fix:** a set-state/issue-txn split API (large scope; likely its own round).

---

## Recommendation for the next tune-dm-helper round
**D1 (runtime num_dests) + D2 (split dest/ack count) + D3 (runtime sem ids)** unblock the largest share —
they're the same theme (move the STAR parameters from compile-time template to runtime args) and would
recover the gn_v2 senders, conv 1D-HS sender, dram-sharded matmul, group_attn, conv3d, and the moe_compute
pair. D5 (value-carrying flag) and D6 (rect-list) are smaller, independent adds. D7 (chain) and D9
(preprogram-state) are large, separate efforts already known as gaps.
