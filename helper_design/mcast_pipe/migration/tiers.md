# Tier plan — mcast_pipe rollout re-entry @ v7 (2026-06-20, autonomous backlog run)

Mode: `run-all`. Machine: single-chip **Wormhole b0**. Helper at **MCAST_PIPE_API_VERSION 7**.
Block_sharded already lifted from quarantine this run (commit 049d2446bd5) → **13 migrated, 48 pending,
8 deferred, 0 quarantined** at the start of this backlog phase.

Worklist = the **48 pending** kernels. Tiered easiest/safest first by `audit tag × API distance ×
verified-coverage strength`. Co-compilation rule honored: a test JIT-builds all kernels in its program;
pending siblings use raw open-coded primitives (compile fine, don't block).

## Hard reality of a single shared chip — coverage split
27 pending kernels are **single-chip-verifiable** (✅ device-verified in a prior round, or 🔶 single-chip
reachable). 21 are **coverage-gap**: their only tests require a **multi-device mesh (≥2)**, the **sweep
harness**, or a **standalone example binary** — none runnable on this one WH chip. Per the skill's
"never migrate without a verified gate" rule (and the lesson from block_sharded / conv-WS: API-correct
migrations that HUNG, caught only because they had device coverage), the 21 are **deferred-on-coverage**
with a one-line reason each — NOT migrated blind. They are a documented gap, not silent breakage.

---

## Tier 1 — clean-tag, single-chip-verified (the spine, 4 kernels)
1. conv `reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp` (clean ✅) — sender twin of migrated receiver. Test: `test_conv2d.py::test_conv_features` HS 256×256.
2. ln `reader_mcast_sender_unary_sharded_ln_post_allgather.cpp` (clean 🔶) — C1 pure-flag no-pre-handshake fresh-slot.
3. ln `reader_mcast_receiver_unary_sharded_ln_post_allgather.cpp` (clean 🔶) — C1 pair. Test: `test_distributed_layernorm_sharded.py::test_post_allgather_layernorm`.
4. deepseek `sampling_kernel.cpp` (clean ✅) — flag-only loop barrier. Test: `test_sampling.py::test_sampling_argmax_single_device_101_cores`.

## Tier 2a — normalization sharded refactor, single-chip-verified (5 kernels)
5. gn_v2 `reader_mcast_sender_unary_sharded_gn_v2.cpp` (refactor-med ✅) — multi-rect ×3, raw-L1 src.
6. gn_v2 `welford_reader_mcast_sender_unary_sharded_gn_v2.cpp` (refactor-med ✅) — multi-rect, per-group loop.
   - Validate 5+6: `test_group_norm.py::test_group_norm_with_block_sharded_v2_8x4_grid` (legacy + welford).
7. ln `reader_mcast_receiver_unary_sharded_ln.cpp` (refactor-high ✅) — phase-2 `wait_min(block+2)` streaming; SENDER twin already migrated@v7. Test: `test_layer_norm_sharded.py::test_layer_norm_sharded_single_stage`.
8. ln `reader_mcast_sender_unary_sharded_ln_pre_allgather.cpp` (refactor-low 🔶) — flag-only mcast + gather HOLE; atomic-barrier.
9. ln `reader_mcast_receiver_unary_sharded_ln_pre_allgather.cpp` (refactor-low 🔶) — atomic-barrier flush.
   - Validate 8+9: `test_distributed_layernorm_sharded.py::test_pre_allgather_layernorm`.

## Tier 2b — reduction + matmul + conv refactor, single-chip-verified (6 kernels)
10. topk `writer_local_topk.cpp` (refactor-low ✅) — companion of migrated reader_final_topk. Test: `test_topk.py` W=8192.
11. argmax `reader_argmax_interleaved_multicore.cpp` (refactor-high 🔶) — 2-rect INCLUDE+EXCLUDE, flag-only. Test: `test_argmax.py` shape [64,128] use_multicore.
12. matmul `reader_bmm_tile_layout_in0_sender_dram_sharded.cpp` (refactor 🔶) — worker-type-dispatched. Test: nightly `test_matmul_dram_sharded.py`.
13. matmul `reader_mcast_transformer_group_attn_matmul.cpp` (refactor 🔶) — rotating-role per-iter barrier. Test: nightly `test_attn_matmul.py::test_group_attn_matmul`.
14. conv `reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp` (refactor ✅) — 3 F3 sub-cases, chunked send. Test: `test_conv2d.py::test_conv_features` BS 32×32 3×3.
15. conv3d `writer.cpp` (refactor 🔶) — 3 modes; unicast-forwarding chain out of scope. Test: `test_conv3d.py`.

## Tier 2c — sdpa refactor, single-chip-verified (2 kernels)
16. sdpa `reader_interleaved.cpp` (refactor ✅) — open-coded chain-link K+V. Test: `test_sdpa_prefill.py::test_sdpa_noncausal`.
17. sdpa_decode `dataflow_common.hpp` (refactor 🔶) — read_k vertical-column mcast; barrier+linked=false. Test: `test_sdpa_decode.py::test_sdpa_decode_sharded` (use_k_mcast).

## Tier 2d — deepseek/moe single-device refactor (5 kernels)
18. `kv_cache_update.hpp` (clean 🔶) — preprogram-state mcast. Test: `test_kv_cache_branch.py`.
19. `flash_mla.hpp` (refactor 🔶) — k-chunk data+flag. Test: `test_flash_mla.py`.
20. moe_gate_mm `dm1.cpp` (refactor 🔶) — one_packet collector→7 cores. Test: nightly `test_moe_mm.py`.
21. moe_compute `tilize_reader.cpp` (refactor 🔶) — value-carrying flag. Test: `test_moe_compute_single_card.py` (1×1 mesh).
22. moe_compute `tilize_writer.cpp` (refactor 🔶) — value-carrying flag. Same test.
   - NOTE: these tests may need model-demo infra not present on a bare WH chip → subagent defers-with-reason if the test can't run here.

## Tier 3 — legacy-API refactor, single-chip-verified (5 kernels)
23. sort `coordinator_single_row_multi_core.cpp` (refactor-med LEGACY)
24. sort `reader_single_row_multi_core.cpp` (refactor-med LEGACY)
25. sort `writer_single_row_multi_core.cpp` (refactor-low LEGACY)
   - Validate 23–25: `test_sort.py::test_sort_long_tensor` (large Wt → multicore-DRAM branch).
26. move `move_interleaved_with_overlap.cpp` (refactor-med LEGACY)
27. move `move_stick_layout_interleaved_with_overlap.cpp` (refactor-med LEGACY)
   - Validate 26+27: `test_move.py::test_move_op` overlap TILE/RM L1-L1.
   - PREREQUISITE: legacy free-function NOC API (`noc_semaphore_set_multicast`) must be ported to the
     modern `Noc`/`Semaphore<>` types the helper sits on. If that port is too invasive or the helper
     can't express the dual-use L1 counter/flag word → defer-with-reason (a prerequisite gap, not a bug).

## Deferred-on-coverage (21 kernels) — NOT migrated this run (no single-chip gate)
- **3 matmul programming-example readers** — binary-only (`metal_example_matmul_multicore_reuse_mcast`), no pytest.
- **4 interleaved group_norm** kernels (`reader_mcast_*_unary_gn`, `welford_reader_mcast_*_unary_gn`) — sweep-only; every unit test shards input → routes to v2.
- **14 multi-device CCL / deepseek-prefill / ring** kernels — require mesh ≥2 (rms_allgather ×2, llama AG-matmul worker_receiver, deepseek_prefill dispatch/combine/unified_ffn, moe_gpt tilize ×2, selective_reduce_combine ×2, all_gather_concat_writer, all_to_all_sender_writer, sdpa exp_ring_joint_reader), plus deepseek_v3_b1 `persistent_h2d_receiver` (no identified single-chip test).

These are listed in the final report under "known gaps / not-verifiable-here". Several are also helper
DESIGN candidates (COUNTER inc_multicast, value-carrying flag, multi-mcast-per-call) → flagged for
tune-dm-helper if a future round adds those modes.

## Order
Tier 1 → 2a → 2b → 2c → 2d → 3. One subagent per tier, strictly sequential (shared device + JIT build).
