# Archived: Reconcile report — mcast_pipe — 2026-08-03

Static reconcile of the rollout's paper state (`census.txt` + `migration/ledger.json`) against the
tree after the branch was rebased. No device work, no build, no migration, **no API version bump**
(`MCAST_PIPE_API_VERSION` stays 9 — the tree moved, the helper API did not).

## Base resolution

| | ref | note |
|---|---|---|
| current base | `origin/llk_helper_library` @ `4a1d6a97ca9` | HEAD is a clean descendant, 17 commits ahead |
| prior reconcile base | `llk_helper_library` @ `54d8dfb7bef` | **not an ancestor** of the current base — llk_helper_library is periodically force-rebased, so the two tips are divergent lines |
| pre-rebase branch | `backup/mcast-migration-prerebase-20260803` | holds the verified pre-rebase state; used as the byte-comparison reference throughout |

Because the two bases are divergent, "what the rebase did to the rollout" is the diff
`54d8dfb7bef → 4a1d6a97ca9` (upstream churn), *not* `base…HEAD` (which is just our own 17 migration
commits). Both signals were computed separately.

Census and ledger were perfectly in sync going in: 92 entries each, zero one-sided paths.

## Buckets

| bucket | count |
|---|---:|
| `unchanged` | 89 |
| `added` | 0 |
| `removed` | 1 |
| `renamed` | 0 |
| `clobbered` | 0 |
| `rebase-touched` → `needs_recheck` | 2 kernel entries + 4 host_binding rows |

---

## `removed` (1) — record lost, approved at the gate

`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/kernels/dataflow/reader_dispatch.cpp`

Prior ledger state, logged here because the entry is deleted:

```
op_family : ccl / deepseek / examples      status               : deferred
role      : hybrid                        migrated_api_version : null
tag       : refactor                      commit               : null
coverage_confidence : none                last_verified        : null
flags     : ["coverage-gap"]
census_note : data + COUNTER inc_multicast + wait_min | DEFER-COVERAGE: min mesh 2x2 - multi-device
validation_set : ["min mesh 2x2 - multi-device"]
```

**Rename detection (ruled out).** Deleted upstream by `af00262e51d` (#48694, dispatch row-major
refactor). git records it `D`, and the two renames in that same commit
(`reader_untilize_dispatch.cpp → reader_worker_dispatch.cpp` R078,
`writer_dispatch.cpp → writer_sender_dispatch.cpp` R069) are not it. Content-signature check on the
successors confirms the mcast role did not move:

- `reader_worker_dispatch.cpp` — one raw `noc_semaphore_wait_min(turn_sem_ptr, …)` turn-taking
  semaphore. No mcast token.
- `writer_sender_dispatch.cpp` — publishes the receive-buffer addresses to workers by a **per-worker
  unicast loop** (`noc_semaphore_inc(get_noc_addr(ring_noc_x[s], ring_noc_y[s], …), 1)`), plus a
  **fabric** multicast (`fabric_multicast_noc_unicast_atomic_inc`). The header comment says
  "multicast" but the implementation is a unicast fan-out; the fabric leg is out of scope per
  `intent.md` ("Ethernet / cross-chip mcast (CCL) — out of scope").

So the dispatch op no longer has an intra-chip rectangle-mcast call site at all. Genuinely removed.

**The entry was already stale before this rebase.** At the old base `54d8dfb7bef` the file had **no**
`inc_multicast` and no `wait_min` — only two semaphore-address arg reads. Its census note
(`data + COUNTER inc_multicast + wait_min`) described a block that had disappeared upstream sometime
between the 2026-06-19 census bootstrap and 2026-07-28, and the 2026-07-29/30 reconcile did not catch
it. The deletion retires a note that was already describing nothing.

**No coverage lost.** It was the census's F2-counter (`noc_semaphore_inc_multicast`) exemplar. That
coverage survives in 6 other entries, including its own twin in the same op-family:

- `.../deepseek_prefill/combine/device/kernels/dataflow/reader_combine.cpp` — same shape,
  `noc_semaphore_inc_multicast(mcast_counter_sem_noc_addr, 1, num_untilizer_cores_group)` at line 256
- `models/demos/deepseek_v3_b1/unified_kernels/flash_mla.hpp`
- `models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/persistent_h2d_writer.cpp`
- `models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/persistent_d2h_reader.cpp`
- `ttnn/core/tensor/kernels/persistent_d2d_receiver.cpp`
- `ttnn/core/tensor/kernels/persistent_d2d_sender.cpp`

Not present in `test_map.json`, so no test mapping to unwind.
`kernel_annotations/deepseek_prefill_readers.md` documented both halves; the `reader_dispatch` half is
marked RETIRED (retained as design evidence — its three HOLE flags still hold via `reader_combine`),
the `reader_combine` half stays live.

## `clobbered` (0) and kernel-side integrity

All 10 `migrated` entries still `#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"` and construct
`McastArgs`. Stronger: **all 10 kernel files are byte-identical** to the pre-rebase verified state on
`backup/mcast-migration-prerebase-20260803`. The rebase reverted nothing and rewrote nothing on the
kernel side.

Every one of the 91 surviving census files still bears the recognition family. Receiver/counter halves
are spelled with the object API (`Semaphore<> s; s.set/up/wait/wait_min`) and carry no mcast token —
exactly as the census header records — so a mcast-token-only grep flags them as false positives.
Verified by reading `reader_bmm_tile_layout_in0_receiver.cpp`,
`reader_mcast_receiver_unary_sharded_ln.cpp`, `writer_local_topk.cpp`,
`writer_single_row_multi_core.cpp` and `reader_bmm_tile_layout_in0_ring_all_gather.cpp`.

## `rebase-touched` → `needs_recheck` (2 kernel + 4 host rows)

Advisory flag only. Status stays `migrated`, `migrated_api_version` stays 9. It means:
*re-run the mapped tests, verify-only, no rewrite; clear when green.*

Rows: kernels `reader_bmm_tile_layout_in1_sender_writer_padding.cpp` and
`reader_bmm_tile_layout_in1_receiver_writer_padding.cpp`; host bindings
`matmul-in1-mcast:reuse-2d:legacy`, `:reuse-2d:descriptor`, `:reuse-1d:mcast-in1:legacy`,
`:reuse-1d:mcast-in1:descriptor`.

**Why.** The kernels were untouched, but their factories were not:

| factory | upstream churn `54d8dfb7bef→4a1d6a97ca9` | our post-ledger edits |
|---|---|---|
| `matmul_multicore_reuse_mcast_1d_program_factory.cpp` | +203 / −13 | `c946da17d29`, `eb05b3929a3` |
| `matmul_multicore_reuse_mcast_2d_program_factory.cpp` | +93 / −4, touching `mm_in1_sender_writer_args` | `c946da17d29`, `eb05b3929a3` |
| `conv2d_op_sharded_program_factory.cpp` | none — byte-identical to verified state | — |
| `groupnorm_sharded_program_factory.cpp` | none — byte-identical to verified state | — |

`c946da17d29` and `eb05b3929a3` postdate the ledger's last doc commit `62f82dd4a64`, so the recorded
`last_verified: 2026-07-30` predates both the upstream churn and our own rework. Conv and GroupNorm
need no recheck.

**Static wire check performed (evidence recorded on all 6 rows).** `McastArgs` is a fixed-offset
5-word CT block + 4-word RT block, so factory drift would silently misalign every argument after it.
All four positions verified on **both** churned factories:

| wire | kernel expects | 2D factory | 1D in1 factory |
|---|---|---|---|
| sender CT | `McastArgs<10,2>` → idx 10–14, `next_compile_time_args_offset()` = 15 | 10 args at 0–9, mcast 10–14, idx 15 = `KtNt` (`2d:490–509`) | same (`1d:1408–1427`) |
| sender RT | 2 words read, then `rt_args_idx += 4` (idx 2–5) | `in1_tensor`, `start_tile_id`, mcast 2–5, sparsity 6 (`2d:1303–1311`) | same (`1d:1909–1917`) |
| receiver CT | `McastArgs<4,0>` → idx 4–8 | 4 args at 0–3, mcast 4–8 (`2d:575–585`) | same (`1d:1470–1480`) |
| receiver RT | 4 mcast words at idx 0–3 | mcast 0–3, then out tensor (`2d`) | same (`1d:1959–1962`) |
| `MCAST_ARGS` define | required, else the kernel takes the legacy 4-word ABI (`post_offset 14`) | set (`2d:618`) | set (`1d:1512`) |

Each factory has a **single** in1 emission site shared by its legacy and descriptor create paths, so
one wire covers all four host_binding rows. The `MCAST_ARGS` + `SKIP_MCAST` coexistence was checked
and is coherent: 2D sets `MCAST_ARGS` unconditionally and `SKIP_MCAST` only when
`in1_receiver.num_cores() == 0` (`2d:654–656`); in the kernel `MCAST_ARGS` governs only the CT offset
(line 18) while `SKIP_MCAST` only compiles out the send body (lines 224–230). The 1D `mcast_in0=true`
path sets `SKIP_MCAST` *without* `MCAST_ARGS` (`1d:553`) and so runs the legacy ABI — correct, that
path is not in the migrated set.

⇒ The mcast wire is intact. The residual risk is the surrounding unrelated descriptor churn, which
only a device run covers. This is a verify-only re-run, **not** a re-migration.

## `added` (0) — recall sweep

Swept with the `recognition:` family from `primitive_contracts.md` (`noc_async_write_multicast`
±`_loopback_src`/`_one_packet`, `noc_semaphore_set_multicast` ±`_loopback_src`, `relay_multicast`,
`noc_semaphore_inc_multicast`, `get_noc_multicast_addr`, `MulticastEndpoint`, plus the object-API
spellings) across `ttnn/`, `tt_metal/`, `models/`, `tests/`, excluding `third_party` and build trees.

123 hits. 71 are census paths (the remaining census entries are pure-receiver halves with no mcast
token — see above). **All 52 non-census hits already existed at the old base `54d8dfb7bef`**, so the
rebase introduced **zero** new call sites. They are pre-existing scope boundaries, recorded here so a
future reconcile does not re-litigate them:

| group | count | why not a census entry |
|---|---:|---|
| substrate / impl | 8 | `tt_metal/hw/inc/api/dataflow/{noc,noc_semaphore,endpoints,dataflow_api,circular_buffer,dataflow_buffer}.h`, `dataflow_api_addrgen.h`, `impl/emulation/emulated_program_runner.cpp` — these **define** the primitives the helper is built from |
| `tests/tt_metal/**` test kernels | 40 | not an op-family; `intent.md` scopes the census to op-family production kernels ("sweep every op-family: conv, layernorm, sdpa, group-attn, groupnorm, data-movement, deepseek-moe") |
| host program factories | 2 | `moe_compute_program_factory.cpp`, `exp_ring_joint_sdpa_program_factory.cpp` — host side, tracked under `host_bindings`, not the kernel census |
| declaration-only headers | 2 | `conv/conv2d/.../conv_reader_common.hpp` and its quasar twin contain only `struct McastRect` + `using McastDst = noc_traits_t<MulticastEndpoint>::dst_args_mcast_type` — **no primitive invocation**. Support headers included *by* census kernels; their only `Noc` calls are `async_write_zeros` / `write_zeros_l1_barrier` / `read_with_state` |

No `kernel_annotations/` files were added — there were no candidates to classify.

## Commit-hash drift (out-of-bucket fix)

All 6 hashes recorded in the ledger resolved but were **off-branch**, reachable only from
`backup/mcast-migration-prerebase-20260803`. Remapped 1:1 by subject; 5 confirmed by identical
`git patch-id --stable`, the 6th confirmed hunk-for-hunk:

| was (pre-rebase) | now (post-rebase) | subject | confirmation |
|---|---|---|---|
| `ab73b1f5c73f` | `acd84d7f3fc` | Initial kernel migration and tracking documentation | per-file hunks identical; differs only by the dropped `tt_metal/third_party/tt_ops_code_gen` submodule bump |
| `fad21b929d1c` | `59e75d6fc3a` | mcast_pipe: migrate Conv and GroupNorm senders | identical patch-id |
| `75b977e1a04e` | `991b5b6b638` | apply mcast host helper to conv2d height-sharded weights | identical patch-id |
| `261e322ed228` | `51dfb1f1ed6` | Migrate conv2d block weights to multicast host helper | identical patch-id |
| `2d0280d3dacf` | `aeeb28ff007` | Migrate matmul in1 to multicast host helper | identical patch-id |
| `0a796a025c9d` | `bc24a55bf80` | Apply mcast host helpers to sharded groupnorm v2 | identical patch-id |

20 hash occurrences rewritten across `entries`, `host_bindings` and `source_of_truth_note`. Two
on-branch commit *messages* still cite pre-rebase hashes (`baa86dc7116` "record conv2d host migration
for 75b977e1a04", `5320c2d69bd` "…for 261e322ed22") — history is immutable, the mapping above is the
key.

## Advisory — 10 deferred entries whose upstream churn touched protocol lines

17 census paths were churned upstream. One is the removal above; 6 are cosmetic (0 protocol lines:
`flash_mla.hpp`, `kv_cache_update.hpp`, `moe_compute/tilize_{reader,writer}.cpp`,
`quasar/…_1d_mcast_sender_conv_weights…_metal2.cpp`, `quasar/move_stick_layout…`). The remaining **10
changed semaphore/mcast lines**, so their `tag` / `census_note` may have drifted the same way
`reader_dispatch`'s silently did. Per the reconcile contract a churned `deferred` row takes no bucket
action, so **nothing was re-tagged**; these are flagged for a future audit pass:

| entry | tag | protocol/total changed lines | what changed |
|---|---|---:|---|
| `sort/coordinator_single_row_multi_core.cpp` | refactor | 15/40 | `cores_to_coordinator_semaphore` **split into `ready` + `done`** — arg layout and protocol shape changed |
| `sort/reader_single_row_multi_core.cpp` | refactor | 6/6 | same split (`ready` half) |
| `sort/writer_single_row_multi_core.cpp` | refactor | 6/6 | same split (`done` half) |
| `deepseek_prefill/unified_routed_expert_ffn_reader.cpp` | refactor | 19/458 | a `noc.async_write_multicast(MulticastEndpoint{}, …)` **removed**, new comment "sends no data — only the valid sem, so receivers still advance". Census note still claims "in0/in1/gate/up data write_multicast + flag set_multicast" — likely wrong now |
| `experimental/indexer_score/reader_indexer_score.cpp` | refactor | 7/324 | CT arg layout gained a fused-ring flag + `k_local` accessor ahead of the 8 multicast args |
| `reduction/argmax/reader_argmax_interleaved_multicore.cpp` | refactor | 7/12 | `start_sem.set_multicast(…)` reflowed; note about the `reduce_all` path losing per-iteration `done_sem` back-pressure |
| `reduction/topk/writer_local_topk.cpp` | refactor | 7/31 | semaphore-id CT comments reworked |
| `ccl/moe/selective_reduce_combine/writer.cpp` | refactor | 4/63 | `compute_sync_semaphore_id` moved to `get_named_compile_time_arg_val` |
| `all_reduce_create_qkv_heads/worker_writer.cpp` | oos | 3/8 | a `noc_async_write_barrier()` added; new comment that the payload writes only flush (SENT, not landed) and the release mcast rides a different VC |
| `matmul/reader_bmm_tile_layout_in0_receiver.cpp` | clean | 1/4 | comment: drain the mcast-ready atomics (`sender_sem.up`) before returning |

## Artifacts written

```
census.txt                                     REWRITTEN — 92 → 91 lines (reader_dispatch deleted from
                                                the `ccl / deepseek / examples` group)
migration/ledger.json                          REWRITTEN — entry deleted; needs_recheck + wire evidence
                                                on 6 rows; 20 commit hashes remapped; reconciled =
                                                2026-08-03; baseline_ref + reconcile_history added
migration/ledger.md                            REGENERATED from the json
migration/reconcile_2026-08-03.md              NEW — this file
kernel_annotations/deepseek_prefill_readers.md UPDATED — reader_dispatch half marked RETIRED,
                                                reader_combine half kept live
```

Untouched, as required: `ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp` (`MCAST_PIPE_API_VERSION` = 9),
`primitive_contracts.md`, `hazards_catalog.md`, `migration_audit/*`, `test_map.json`, `tiers.md`,
`log/*`, `report.md`.

## Hand-off to `apply-dm-helper`

Re-invoke `apply-dm-helper helper_design/mcast_pipe/`. Work waiting for it:

1. **Verify-only (6 rows, no rewrite).** The matmul-in1 unit: kernels
   `reader_bmm_tile_layout_in1_{sender,receiver}_writer_padding.cpp` + the 4
   `matmul-in1-mcast:*` host bindings. Requires a `./build_metal.sh` first (host code changed), then
   the mapped `MM-IN1-ALL` inventory from `test_map.json` (302 passed / 188 expected skips baseline)
   across both `transpose_mcast` values and the 1D `mcast_in1` path. On green: clear `needs_recheck`,
   refresh `last_verified`, and stamp the current commit. The wire is already statically confirmed —
   a failure here means the surrounding descriptor churn, not the mcast block.
2. **Migrate (0 pending).** Nothing new. No `clobbered`→`pending` rows and no new candidates: the
   recall sweep found zero rebase-introduced call sites.
3. **Optional follow-up** (not scheduled here): re-audit the 10 protocol-churned `deferred` tags
   above, loudest being the three `sort/` kernels and `unified_routed_expert_ffn_reader.cpp`.
