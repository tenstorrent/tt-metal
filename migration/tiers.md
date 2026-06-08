# Migration tiers — `mcast_pipe` rollout (mode = run-all)

**Migratable universe = the 20 device-VERIFIED kernels** (`test_map.json` → `verified`). Kernels
whose only coverage is multi-device, or that have no green test on this single-chip machine, are NOT
migrated blindly (run-all's safety net can only protect a kernel that has a runnable mapped test).
They are reported as coverage gaps. Deferred-by-invocation kernels (R6, R4 streaming, legacy
move/sort, fabric/ring CCL, prior-art helpers) are excluded entirely.

Ranking = audit tag (clean < refactor-low < refactor-high) × API distance (canonical send/receive <
multi-rect/loopback < counter/multi-phase/chain) × verified-coverage (all high here, except the conv
halo reader = med). Within a tier: ascending diff size × descending coverage confidence.

Each tier runs as ONE subagent (sequential — shared device + JIT build). Each kernel = one atomic
commit; failures are `git restore`d (tree stays green) and marked FAILED+quarantined.

---

## Tier 0 — proof (DONE)
Helper unit test `test_mcast_pipe.py` (45 cells PASS) + the F3 degenerate + PRE_HANDSHAKE confirms.
Already green; committed in Phase 1.

## Tier 1 — clean spine (canonical `send()`/`receive()`, Flag + flush, strongly tested)
| kernel | role | validation |
|---|---|---|
| reader_bmm_tile_layout_in0_sender_padding | sender | matmul_1d |
| reader_bmm_tile_layout_in0_receiver | receiver | matmul_1d |
| reader_bmm_tile_layout_in1_sender_writer_padding | hybrid | matmul_1d |
| reader_bmm_tile_layout_in1_receiver_writer_padding | receiver | matmul_2d |
| reader_writer_tiled_out_1d_mcast_sender_conv_weights… | sender | conv_HS |
| reader_writer_tiled_out_1d_mcast_receiver_conv_weights… | receiver | conv_HS |
| writer_tiled_out_2d_mcast_sender_conv_weights… | sender | conv_BS |
| writer_tiled_out_2d_mcast_receiver_conv_weights… | receiver | conv_BS |
| reader_mcast_receiver_unary_sharded_gn_v2 | receiver | gn_v2_legacy |
| reader_final_topk | receiver | topk_8192 |
| sampling_kernel (deepseek) | sender (flag-only → send_signal) | deepseek_sampling |

11 kernels. The matmul + conv pairs are the canonical two-sided proof of the API.

## Tier 2 — refactor-low (multi-rect / loopback / unicast-scatter)
| kernel | role | refactor cost | validation |
|---|---|---|---|
| reader_mcast_sender_unary_sharded_gn_v2 | sender | C2 multi-rect (`McastRect` list) | gn_v2_legacy |
| welford_reader_mcast_sender_unary_sharded_gn_v2 | sender | C2 multi-rect | gn_v2_welford |
| welford_reader_mcast_receiver_unary_sharded_gn_v2 | receiver | C2 | gn_v2_welford |
| activation_reader_width_sharded (conv) | hybrid | INCLUDE_SRC loopback + barrier; mixed counter/flag | conv_WS |
| writer_local_topk | sender | unicast scatter + up(counter) — assess Pipe fit; if not rectangle-mcast, DEFER | topk_8192 |

5 kernels.

## Tier 3 — refactor-high (counter / two-phase / chain-link / chunked)
| kernel | role | refactor cost | validation |
|---|---|---|---|
| reader_mcast_sender_unary_sharded_ln | sender | C3 two-phase flag→`Staging::Counter` streaming | sharded_ln |
| reader_mcast_receiver_unary_sharded_ln | receiver | C3 | sharded_ln |
| reader_interleaved (sdpa) | hybrid | open-coded chain-link K+V (`LINK` dual-path) | sdpa_noncausal |
| reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2 | hybrid | migrate loopback/handshake ONLY; keep chunked send (R4) on raw API | conv_BS |

4 kernels.

---

## NOT migrated (gaps / deferred) — see test_map.json
- **Gaps (no validatable test here):** dm1, moe_compute tilize ×2; ln_pre/post_allgather ×4; all
  multi-device CCL legs (~11); interleaved non-v2 group_norm ×4; argmax multicore; conv3d writer;
  matmul dram_sharded; 3 matmul didactic examples.
- **Deferred by invocation:** R6 role-flip (matmul block-sharded, group_attn); R4 streaming chunked
  send (the chunked path of the conv halo reader stays raw); legacy move/sort; fabric/ring CCL;
  chain_link.hpp + deepseek mcast.hpp.
