# DeepSeek-V4.1-Flash prefill: bring-up ops vs a rebase onto newer tt-metal main

Read-only investigation, 2026-11-04, branch `dnijemcevic/model_bringup` @ `796af7887b8` (nothing in the repo was
changed, nothing fetched, nothing built or run).

## 0. Baselines used

| ref | commit | date | note |
|---|---|---|---|
| "latest main" (local `origin/main`, not fetched) | `6b9cefd1b7d` | 2026-10-30 10:32 UTC | newest commit: "[Feature] BEVFormer: selectable tiny, small and base configurations (#62381)" |
| branch merge-base with main | `48444d1f8a1` | 2026-10-22 | "Quasar Matmul 4 (#60926)" |
| branch base on `origin/llk_helper_library` | `6b80daf43e3` | 2026-10-23 | llk_helper_library (tip `c6b7f846f42`, 2026-10-27) = main `48444d1f8a1` + 194 commits |
| Marko / Milos Staletovic's MoE branch `origin/mstaletovic/combine-lean-sender-llk` | `a33173f7288` | 2026-10-30 | the source of the MoE stack (see section 2) |

- main moved **429 commits** from the merge-base to `origin/main` (8 days). Tomorrow's real target will be newer: scale
  the "upstream drift" columns below up accordingly.
- The branch carries **2635 commits** over the merge-base. It sits on `llk_helper_library`, not on bare main, and the
  rebase skill (`models/demos/common/bringup/skill/rebase-bringup/SKILL.md`) rebases onto `llk_helper_library`.
  **Owner decision needed first:** if `llk_helper_library` is not itself rebased onto the new main before tomorrow,
  rebasing the bring-up branch directly onto main drops the helper library that several forks include
  (`kernel_lib/matmul_block_helpers.hpp`: mhc_pre_ttnn, matmul_reduce_scatter; `kernel_lib/tilize_helpers_dataflow.hpp`:
  rms_norm_ttnn; the extended `reduce_helpers_*` API: rms_norm_ttnn, mhc_pre_ttnn, indexer_score, sdpa). The estimates
  below assume the new base still includes `llk_helper_library` (i.e. it is rebased first, or carried along).
- "Fork base": for forks of a stock op, each CHANGELOG records a `Source SHA` that is a branch commit; the source folder
  on the branch equals main @ `48444d1f8a1` (+ llk_helper_library), except the carried changes noted. So "upstream
  drift" = `git log 48444d1f8a1..origin/main -- <source path>`.

## 1. What the SHIPPED model runs

Source of truth for the shipped op set: the FIX.22 profile, the newest full run with every switch at its default
(`bringup/ledgers/galaxy_8x4-full/results/FIX.22_ops.json.gz` / `FIX.22_profile.json`, 2026-11-04, 8x4 Galaxy, rung
`last`, 40 layers, `settings: {}`), cross-checked against `tt/settings.py` defaults and the code. Defaults that select
the fork paths: `MOE_IMPL=block`, `MOE_HEAD=fused`, `MOE_TAIL=fused`, `TAIL_A2A_PREPARED=on`, `CCL_IMPL=fabric_bf16`,
`NORM_IMPL=on`, `MHC_FUSED=on`, `RESIDUAL_MIX=fused`, `ROPE_IMPL=on`, `ROPE_UNIFORM_PROGRAM=on`, `ROPE_CHAINS=on`,
`CKV_FORMAT=bfp4_rows`, `WINDOW_KV_FORMAT=bfp8_tiles`, `INDEX_K/Q_FORMAT=bfp4`, `INDEX_TOPK_OP=fork`,
`CKV_BFP4_PREPARED=on`, `GENERIC_OP_PREPARED=on`, `PREPARED_ALL=on`, `TINY_KERNELS=on`, `EXPERTS_IMPL=bfp4_mse`,
`MAX_WORKER_CORES=110`, `DECODER_SKIP=off`.

Device ops in one chunk (FIX.22, device ms summed over the 40 layers): flat_routed_expert_with_program 53.0,
"(other)" = Python ProgramDescriptor programs (MoE head / exchange / tail, CCLs, window send) 35.3, stock linear 16.0,
sparse_sdpa 11.6, stock matmul 8.4, mhc_post 8.3, rotary_embedding_indexed 5.3, mhc_pre_xing 5.0, stock
all_to_all_async_generic 4.9, ring_indexer_score_dsa 3.3, fabric_all_gather 3.3, rms_norm 3.0, topk_large_indices 1.5,
ckv_bfp4 1.0, mhc_pre_xing_pack 0.4, nlp_concat_heads 0.4, pair_pool 0.2, update_padded_kv_cache 0.1, bfp_round,
slice_write, embedding (< 0.1 each); stock typecast, untilize_with_unpadding, minimal_matmul, scatter, to_layout,
concat, slice, reshape, multiply.

**The shipped model uses 18 bring-up fork folders** (14 with a C++ host side, 4 Python ProgramDescriptor ops):
flat_routed_expert_ttnn, combine_rf, prepared_program, fabric_all_gather_ttnn, fabric_reduce_scatter_ttnn, sdpa,
indexer_score, topk_large_indices, rms_norm_ttnn, mhc_pre_ttnn, mhc_post_ttnn, rotary_embedding_indexed,
update_padded_kv_cache, nlp_concat_heads, slice_write, embedding, ckv_bfp4_ttnn, pair_pool_ttnn.
Plus three non-fork carried code drops from Marko's branch that are not on main: `models/demos/moe_block/`,
`models/demos/deepseek_v3_d_p/tt/moe/tt_k3_token_split_moe.py`, `ttnn/ttnn/operations/matmul_reduce_scatter/` (+
`ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp`).

**Used only behind non-default switches (brief):**
- `moe_grouped_topk` (tt/router.py:142): only `MOE_IMPL=flat` (+`ROUTER_IMPL=fused`); with the block MoE the router is
  Marko's fused router inside the head.
- `nlp_create_qkv_heads` (`nlp_create_q_heads_split`, tt/attention.py:1252, 1312; tt/indexer.py:371, 387): only with
  `ROPE_CHAINS=off` / `ATTN_GLUE` paths; the shipped RoPE chain reads heads in place.
- `sparse_sdpa` kv_format BF16 (tt/attention.py:1402, `WINDOW_KV_FORMAT=bf16_tiles`) and ROW_BFP8 (1373,
  `bfp8_blocks`); the shipped one is ROW_BFP4 (1387).
- The flat MoE path (`MOE_IMPL=flat`): tt/experts.py, tt/weighted_combine.py (flat_routed_expert_with_program at :523,
  combine_rf sparse_combine_rf / l1_zero / combine_plan, raw `ttnn.generic_op` at :437, :481), tt/shared_expert.py.
- `tail_send.py` (prepare_program / run_prepared, combine_rf route): only `DECODER_SKIP=replay128`.
- `moe_block/tt/tail_engine.py`: latent (Kimi-K3) models only.
- Not called by DS41 at all, but compiled into the same bring-up library, so they must still build after the rebase:
  dispatch, combine, offset_cumsum, unified_routed_expert_ffn, moe_ag, moe_grouped_topk, nlp_create_qkv_heads,
  prepare_chunk_recurrence. `dispatch` and `combine` call `tt::tt_fabric::get_forwarding_link_indices` (changed
  signature on main, see 3.1): build-required fixes.

## 2. The MoE: where it lives, fork or copy

**Answer: (c) both, but the "copy" is NOT a copy of main.** The MoE block is code carried from Marko Staletovic's
(git author Milos Staletovic) branch `mstaletovic/combine-lean-sender-llk` @ `856d97d5864` (2026-10-27), commits
`a746a4bb540` (block + K3 exchange), `9518ba45263` (matmul_reduce_scatter + perf_instrumentation.hpp, "as is"),
`14af221b826` (fork sync), `da4dcad040f` (fused head), later `22528347b03` (P.17 MOE_DROP: a newer drop via
`dnijemcevic/model_bringup_lowbit @ 5e86f048ad4`). That branch itself forked from this bring-up branch at
`6b80daf43e3`, so it shares our merge-base with main. **None of it is on `origin/main`**: `models/demos/moe_block`,
`tt_k3_token_split_moe.py`, `ttnn/ttnn/operations/matmul_reduce_scatter` and every `combine_rf` / `flat_routed_expert`
file are absent there (checked with `git ls-tree origin/main`; main's MoE is the unrelated `deepseek_v3_d_p/tt/moe/tt_moe.py`
TtMoe with combine_fabric2d). If the owner remembers "cherry-picked from recent main", that is not what this branch
holds.

The parts, by kind:

| part | where on this branch | kind | on origin/main? |
|---|---|---|---|
| MoE block (Python: MoEBlock, fused head, router / routing streams, SE engine, tail A2A) | `models/demos/moe_block/` (moe_block.py + tt/*.py, ~11k lines incl. JIT kernels as strings) | copy of Marko's branch | no |
| token-split exchange (dispatch, flat experts, combine) | `models/demos/deepseek_v3_d_p/tt/moe/tt_k3_token_split_moe.py` (1685 lines, new file inside a folder that exists on main) | copy of Marko's branch | no (the folder yes, the file no) |
| matmul_reduce_scatter (fused head's SE reduce-scatter planner) | `ttnn/ttnn/operations/matmul_reduce_scatter/` | copy of Marko's branch | no |
| flat routed experts op | `ttnn/ttnn/bringup/flat_routed_expert_ttnn/` (C++ + 69 kernel files) | bring-up fork (source: Marko's branch, same folder) | no |
| combine / dispatch / router / tilize ProgramDescriptor builders | `ttnn/ttnn/bringup/combine_rf/` (15.7k lines Python + inline kernels) | bring-up fork (source: Marko's branch) | no |
| prepared generic_op dispatch | `ttnn/ttnn/bringup/prepared_program/` (C++) | bring-up fork (source: Marko's branch) | no |
| fabric reduce-scatter / all-gather | `ttnn/ttnn/bringup/fabric_reduce_scatter_ttnn/`, `combine_rf/fabric_all_gather_py.py` | bring-up forks (source: `mstaletovic/mimo-v2-dp`) | no |
| DS41 adapter | `models/demos/deepseek_v41_flash_d_p/tt/moe_block_adapter.py`, `tt/prepared_tail.py`, design note `tt/MOE_BLOCK.md` | model code | no |
| helper imported from main's folder | `deepseek_v3_d_p/tt/moe/init_helpers.py` (`ExpertMapping`; branch adds a 4-line `TT_MOE_COMBINE_OVERLAP` env switch) | main file, small carried edit | yes; main changed it (+140 lines, additions only: fabric2d helpers) |

Divergence (from `git diff --shortstat`):
- vs Marko @ `856d97d5864` (what was carried): `models/demos/moe_block` + `deepseek_v3_d_p/tt/moe`: 17 files,
  +2405 / -500 (se_engine3.py +963, moe_block.py +527, router_stream.py +252, tail_a2a.py +191,
  tt_k3_token_split_moe.py +174). Over the six MoE-related paths incl. the forks: 101 files, +13313 / -1688.
- vs Marko's current tip `a33173f7288` (2026-10-30): moe_block + deepseek_v3_d_p/tt/moe: 13 files, +689 / -128
  (our DS41 options: x_gathered, bfp8 SE x/h, sqrtsoftplus router, prepared tail, own rows, bfp8/bfp4 experts...).
  Marko has 16 commits on `models/demos/moe_block`, 8 on `combine_rf`, 3 on `flat_routed_expert_ttnn`, 1 on
  `prepared_program`, 3 on `tt_k3_token_split_moe.py` since `856d97d5864` that this branch has only partly taken (via
  the lowbit drop); notable: `a33173f7288` flat DRISC NoC high-address clear (a correctness fix),
  `3889640f554` routing exchange forwarding slots in DRAM (a traced / untraced hang fix), `de7000d0af7` per-chip
  combine accumulator roles, `74995ca844b` allocator shard-grid memo (core change), `9620cf2db7b` combine ring
  semaphores once per mesh.

**What this means for the rebase:** the MoE copy produces **no git conflicts** (main has none of those paths), so it
rides through `git rebase` untouched; its risk is entirely runtime: it is Python that builds JIT kernels at run time,
so main's API changes surface only on the device (section 3.1: `ttnn.get_forwarding_link_indices` now requires a mesh
device, used at `moe_block/tt/routing_stream.py:818`, `matmul_reduce_scatter_program_descriptor.py:658`,
`combine_rf/sparse_all_gather_2d.py:603`, `combine_rf/fabric_all_gather_py.py:863`). Syncing with Marko's newer
commits is a separate decision from the main rebase (owner: take his 30 newer commits now, or later).

## 3. Main changes that hit the bring-up stack across forks

### 3.1 Build- or run-breaking (must fix)
- **#61503 `7060deaa01f` Fabric getters take a MeshDevice** (2026-10-26): `get_forwarding_link_indices(mesh_device, src,
  dst)`, `get_fabric_topology(mesh_device)`, `get_tt_fabric_channel_buffer_size_bytes(mesh_device)`; Python
  `ttnn.get_forwarding_link_indices` gains a leading `mesh_device` argument.
  - C++ (build breaks): `bringup/fabric_all_gather_ttnn/device/fabric_all_gather_factory.cpp:392, 438`;
    `bringup/sdpa/device/ring_joint_sdpa_program_factory.cpp:3542`, `exp_ring_joint_sdpa_program_factory.cpp:1472, 1514`;
    `bringup/dispatch/device/dispatch_program_factory.cpp:104`; `bringup/combine/device/combine_program_factory.cpp:104`.
  - Python (TypeError at run time): `deepseek_v41_flash_d_p/tt/ccl.py:205-206` (shipped: rs_links),
    `bringup/fabric_all_gather_ttnn/fabric_all_gather_py.py:734`, `bringup/combine_rf/fabric_all_gather_py.py:863`,
    `bringup/combine_rf/sparse_all_gather_2d.py:603`, `moe_block/tt/routing_stream.py:818`,
    `operations/matmul_reduce_scatter/matmul_reduce_scatter_program_descriptor.py:658`.
- **#61262 `57ba884b981` MeshDevice::get_device(coord / ChipId) removed** (use `internal::get_device`):
  `bringup/fabric_all_gather_ttnn/device/fabric_all_gather_factory.cpp:421, 944, 983`. (No Python use.)
- Carried core changes that main also edited (git conflicts during `git rebase`; see the skill's step 2 policy):
  - `ttnn/cpp/ttnn-nanobind/fabric.cpp` (branch: get_neighbor_eth_directions binding, `898f5f1b70d`; main: #61503).
  - `ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async_generic/device/all_to_all_async_generic_program_factory.cpp`
    (branch: unicast init `464fcdc4ce3` + mesh-device guard `898f5f1b70d`; main: #61503, #61262, #61955 fabric mux in
    program-local L1, #62567). **The shipped model calls this stock op 42 times a chunk** (MoE fused tail / ops tail).
  - `tt_metal/impl/allocator/allocator.{cpp,hpp}`, `tt_metal/impl/program/program.cpp` (branch: Marko's shard-grid memo
    via P.17; main: #61985 program-local resource allocation refactor, #61773, #61952, #60969).
  - `experimental/deepseek_prefill/moe_grouped_topk` (stock op; branch: Milos's fp32_selection, `9e82dd309fd`,
    `66e17107625`, `c5f88a66638`; main: #60550 `e5e853c3cd1` low-bit sorting comparison fix in the same compute
    headers). Not shipped by DS41 (the fork is used by the flat path and GLM-5.3).
  - `experimental/deepseek_prefill/hybrid_routed_expert_ffn` combine_fabric2d (branch: P.17 drop; main: #62312 cmb v2,
    #62473). Not called by DS41.
  - `ttnn/ttnn/__init__.py`, `ttnn/ttnn/types.py`, `ttnn/sources.cmake` (main: #58175 MX dtypes, #60803, #60348).
  - `experimental/deepseek_prefill/rotary_embedding_indexed/device/...device_operation.cpp` (branch: `70b3f4e16b5`
    precompile defer, #55962 not on main; main: #62567). Trivial.

### 3.2 Notable but not breaking (as far as visible cheaply)
- #62319 `5ec42be54a3` fabric API headers split into `linear|mesh/tt-1xx/api_impl.h` ("no functional change for WH or
  BH"): every JIT fabric kernel here (combine_rf, moe_block, window_send, fabric_reduce_scatter, flat_gather,
  fabric_all_gather) includes `tt_fabric_api.h`, `edm_fabric_worker_adapters.hpp`, `packet_header_pool.h`,
  `fabric_edm_packet_header.hpp` — all still present on main.
- #61819 / #62534 SFPU headers moved from tt-llk into `llk_sfpu/` (topk, comp, tanh_derivative; guard test): a symbol
  check of every `calculate_* / init_* / llk_math_*` name the shipped forks / moe_block / model kernels use found none
  missing on main.
- tt-llk: 81 commits (491 files); `tt_metal/hw/inc/api`: 43 commits (SFPU perf, custom_mm 2.0, Quasar plumbing,
  #62814 circular_buffer_constants split). Bit-level output changes possible in SFPU paths (exp, quantize rounding
  #59422) the forks share; caught by the fork tests and the model ladder, not by the build.
- `tt_metal/api/tt-metalium/program_descriptors.hpp`, `program_descriptor_patching`, `ttnn/operations/generic`: 0 commits
  (good for the Python ProgramDescriptor forks and prepared_program).
- `ttnn/cpp/ttnn/kernel_lib`: 3 commits on main (#61818 matmul multicast helpers, Quasar DFB-by-reference).

## 4. Per-op detail (shipped forks)

Effort: agent-hours for steps 3-5 of the rebase skill for that op (sync / port, build, its tests), excluding the
whole-model verification (section 6). low < 1, medium 1-4, high > 4.

### 4.1 flat_routed_expert_ttnn (`ttnn.bringup.flat_routed_expert_with_program`, `FlatRoutedExpert`)
- Folder: `ttnn/ttnn/bringup/flat_routed_expert_ttnn/` (C++ op + 69 kernel files + Python flat_expert.py / flat_layouts.py / flat_gather.py; 21k lines).
- Derived from: `ttnn.experimental.deepseek_prefill.flat_routed_expert` on `mstaletovic/mimo-v2-dp` @ `b45a923d73c`;
  current source: the same folder on `mstaletovic/combine-lean-sender-llk` @ `856d97d5864` (synced 2026-10-27, then
  P.17 drop). Not on main.
- Call sites (shipped): `deepseek_v3_d_p/tt/moe/tt_k3_token_split_moe.py:434` (FlatRoutedExpert), `:1205`
  (flat_routed_expert_with_program); `moe_block/moe_block.py:1751`; `deepseek_v41_flash_d_p/tt/experts.py:466-467`
  (weights / layout cache shared with the block, via moe_block_adapter.py:194-222). Flat path only: tt/weighted_combine.py:523, tt/shared_expert.py:89.
- Why a fork: no such op in stock TTNN; Marko's spatially pipelined single-program expert FFN (weights streamed once,
  counts read on device) is the MoE block's expert engine. DS41 fixes: FIX.1 (I 576 / 1152 unpadded, DS41-76), FIX.2
  (DS41-63), FIX.5 patch_addresses program key, bfp4 down / bfp8 gate-up, 110-core cap.
- Delta: typed layouts (`layout=`, LEGACY / marko), weights_nk, hidden, down_fp32, max_y_writers, bfp4 down weights,
  patch_addresses, gate/up split for any I, 2 / 3 subgrid plans, precision-only x/h/y bf16 tiles, stochastic rounding.
- Upstream drift: main: n/a (no source on main). Marko's branch: 3 commits since `856d97d5864` (+302/-15), incl.
  `a33173f7288` DRISC NoC high-address clear (take it).
- Effort: **medium (~2.5 h)**: rebuild on new main APIs (no main source to merge), take Marko's 3 commits, its unit
  suite + model cases; it is 53 ms of the chunk (the hottest op), so re-check perf too.

### 4.2 combine_rf (`ttnn.bringup.combine_rf.*` Python builders)
- Folder: `ttnn/ttnn/bringup/combine_rf/` (Python ProgramDescriptor builders with inline kernels, 15.7k lines).
- Derived from: the same folder on `mstaletovic/combine-lean-sender-llk` @ `856d97d5864` (three-way sync). Not on main.
- Call sites (shipped): `tt_k3_token_split_moe.py:31-58` imports, prepares at :299-301, :336, :542-570, :690-696,
  :730, :769, :807-823, runs at :1004 (fabric_all_gather), :1036, :1095, :1166, :1393; `moe_block/moe_block.py:1014`
  (fabric_all_gather), :1536 (router_topk, ops head only), :1832; `moe_block/tt/fused_head.py:449-450, 671`;
  `routing_stream.py:25-26`; `x_gather3.py:18`; `tail_a2a.py:33`; `prep_stream.py:10`; DS41 `tt/window_send.py:165-166`
  (route / mesh_plan for the shipped window carry send).
- Why a fork: Marko's token-split MoE pieces (weighted fp32 fabric combine, sparse 2D all-gather dispatch, planners,
  router top-k, tilize / untilize, L1 zero) have no stock equivalent; stock deepseek_prefill combine/dispatch were
  slower and not deterministic (DS41-70 deadlock fix, D7 deterministic order).
- Delta: per-call landing pools, column-axis combine, multi-core deterministic planner (cores=, 1573 -> 69 us),
  bf16 running sums (acc_dtype, P.9), combine_snap, persist_l1, precision-only fp32 output / acc_unpack_fp32.
- Upstream drift: main n/a. Marko: 8 commits since base (+678/-139): per-chip accumulator roles, sender options,
  routing buffers' L1 message, forwarding slots in DRAM (hang fix), sparse_all_gather_2d key / ring fixes.
- Effort: **medium (~2 h)**: two Python `get_forwarding_link_indices` call sites, runtime-only failures (JIT), owner
  call on taking Marko's 8 commits; the MoE's determinism / hang history (DS41-70, -118, -123) argues for a Galaxy run.

### 4.3 prepared_program (`ttnn.bringup.prepare_program`, `run_prepared`, `prepared_program.generic_op`)
- Folder: `ttnn/ttnn/bringup/prepared_program/` (C++ device op, 1.4k lines).
- Derived from: same folder on `mstaletovic/combine-lean-sender-llk` @ `856d97d5864`. Not on main.
- Call sites (shipped): `moe_block/tt/fused_head.py:764, 876-877`; `moe_block/tt/tail_a2a.py:765-776`;
  `deepseek_v41_flash_d_p/tt/prepared_tail.py:64, 72`; `tt/window_send.py:249, 253`; `tt/served.py:73, 107`;
  `tt/host_alloc.py:191`; `tt_k3_token_split_moe.py:59, 1036, 1095, 1166, 1614`; also inside fabric_reduce_scatter
  (`prepared=True`) and ckv_bfp4 / bfp_round (`prepared=True`). Non-default: tt/tail_send.py:254, 259.
- Why a fork: host time on 32 chips (P.1 / P.2 / P.14, OV-23): stock `ttnn.generic_op` re-hashes and re-applies every
  runtime arg each call (~1.05 ms host a call for the fused tail); no stock equivalent.
- Delta: a program prepared once, only io tensors' runtime-arg slots / CBs patched on a hit; opt-in `value_slots`.
- Upstream drift: main: ProgramDescriptor / patching API 0 commits. Marko: 1 commit (cache key holds patched slots).
- Effort: **low (~0.5 h)**.

### 4.4 fabric_all_gather_ttnn (`ttnn.bringup.fabric_all_gather`; Python planner `fabric_all_gather_py`)
- Folder: `ttnn/ttnn/bringup/fabric_all_gather_ttnn/` (C++ op + Python prototype / plan, 5.3k lines).
- Derived from: `ttnn.experimental.fabric_all_gather` on `mstaletovic/mimo-v2-dp` @ `b45a923d73c`; not on main.
- Call sites (shipped): `tt/ccl.py:261` (every `all_gather` under CCL_IMPL=fabric_bf16; 52 calls a chunk), `tt/ccl.py:197`
  (`_num_banks`).
- Why a fork: drop-in for `ttnn.experimental.high_bw_all_gather` at link rate (NoC1 port cores, counter-gated relays),
  P.5 CCL_IMPL; stock all_gather slower on the 8x4 host-bound chunk.
- Delta: the op itself; persistent outputs (P.4), Python `plan()`.
- Upstream drift: main n/a for the op, but **build breaks**: #61503 (2 C++ + 1 Python call sites) and #61262
  (`mesh_device->get_device` x3).
- Effort: **medium (~1 h)**: mechanical API port, then its tests on the box.

### 4.5 fabric_reduce_scatter_ttnn (`ttnn.bringup.fabric_reduce_scatter`)
- Folder: `ttnn/ttnn/bringup/fabric_reduce_scatter_ttnn/` (Python ProgramDescriptor op, 2.5k lines).
- Derived from: `ttnn/ttnn/operations/examples/fabric_reduce_scatter` on `mstaletovic/mimo-v2-dp` @ `b45a923d73c`; not on main.
- Call sites (shipped): `tt/ccl.py:147` (`_rs_bf16`: tp_sum / sum_rows / sum_cols under fabric_bf16), `:221`
  (load-time `prepare`), `:326`, `:352`, `:375` (exact paths, used where `exact=True`, e.g. engram gate partials);
  `moe_block/moe_block.py:1023` (ops head only).
- Why a fork: Marko's reduce-scatter over the fabric with exact fp32 SFPU adds (stock RS sums in bf16; DS41-128
  deterministic order) and host-cheap prepared dispatch (P.1).
- Delta: row / column blocks, ring / snake / dual Hamiltonian groups, descriptor cache, `program_hash`, `prepared`,
  opt-in `exact_fp32_add`.
- Upstream drift: none (main n/a, Marko 0 commits). Its planning code imports `fabric_all_gather_py` (4.4's Python fix covers it).
- Effort: **low (~0.5 h)**.

### 4.6 sdpa (`ttnn.bringup.sparse_sdpa`, `SparseKVFormat`)
- Folder: `ttnn/ttnn/bringup/sdpa/` (the whole SDPA family, 42k lines, 17 host .cpp, 49 kernels).
- Derived from: `ttnn/cpp/ttnn/operations/transformer/sdpa`, Source SHA `99f7e834cea8` (2026-09-28, mimo_v2_6_d_p);
  last "Upstream sync to source @ `6b80daf43e3`" (2026-10-27, full sync).
- Call sites (shipped): `tt/attention.py:1387` (kv_format ROW_BFP4 + kv_window, compressed layers),
  `tt/attention.py:111` (ROW_BFP4 record size). Non-default: :1373 (ROW_BFP8), :1402 (BF16).
- Why a fork: DS41's 4-bit compressed KV (FIX.8 / DS41-150: bfloat4_b row records gathered by NoC reads, unpacked as
  bfp4 tiles, plus a bfloat8_b window) has no stock kv format; FIX.15 110-core `max_grid`; precision options
  (high_precision / fp32_scores / bf16_probs, DS41-29) for the realistic DeepSeek precision.
- Delta: kv_format ROW_BFP4 / ROW_BFP8 / window blocks, kv_window, max_grid, sparse_sdpa precision options; other
  models' V-narrower-than-K, ring_mla fp32 DEST, kv_actual_isl=0.
- Upstream drift: 7 commits on `transformer/sdpa` (19 files, +964/-580): **#60493 `4ab2bc99e26` GQA K/V sharing, Q-head
  crossing, K/V retention (sdpa_program_factory.cpp +/-1147 lines, the file the fork's V<K change edits)**, #61759 fake
  cache hits (ring_joint, exp_ring_joint types), #62434 attention-sink unpack reconfig in streaming normalize, #61503
  (ring factories), #61955 fabric mux L1, #62567, #58929. `sparse_sdpa_device_operation.cpp` itself: 1 line.
- Effort: **selective sync (build-required + bug fixes #61759 / #62434): medium (~1.5 h)**; **full sync: high (~5 h)**
  (3-way merge of the 1147-line factory rewrite against the fork's own sdpa changes, then the ring tests). The shipped
  sparse_sdpa path is barely touched either way.

### 4.7 indexer_score (`ttnn.bringup.ring_indexer_score_dsa`, `IndexerScoreProgramConfig`)
- Folder: `ttnn/ttnn/bringup/indexer_score/` (9k lines).
- Derived from: `experimental/indexer_score`, Source SHA `2761111778f1` (2026-09-29, hy4_preview_d_p); synced to `bdc576fa652`, then full sync to `6b80daf43e3` (2026-10-27).
- Call sites (shipped): `tt/indexer.py:181` (config), `tt/indexer.py:435` (ring score).
- Why a fork: DS41-49 (key_compression_ratio 2 refused by the source; natural query blocks over a block-cyclic key
  cache), DS41-117 (q work unit vs 160 rows), bfloat4_b q / k (DS41-150), block epilogue (block max / bias, P.21), fp32 DEST.
- Delta: ratio 2, natural_query_blocks / rows, block epilogue + block_max_out, bfp4 q/k, opt-in fp32 DEST.
- Upstream drift: 1 commit (#62567 is_allocated cleanup, 3 lines).
- Effort: **low (~0.5 h)**.

### 4.8 topk_large_indices (`ttnn.bringup.topk_large_indices`)
- Folder: `ttnn/ttnn/bringup/topk_large_indices/` (10.6k lines incl. its own LLK copy in `device/kernels/llk/`).
- Derived from: `experimental/topk_large_indices`, Source SHA `df92663a019f` (2026-10-29).
- Call sites (shipped): `tt/indexer.py:95-97` (INDEX_TOPK_OP=fork; index top-k 512 and candidate blocks 2048).
- Why a fork: DS41-151 perf: it pre-carried upstream's split LLK (#62035 / #62488 / #62411, not on the old base) plus a
  K' 512 split segmented body (512 x 55296: 1190 -> 675 us; 8x4 1M 2256 -> 954 us). DS41-170: tie order differs from stock.
- Delta: upstream's split body + LLK copy, K' 512 fused segmented body (templated move_segment_result).
- Upstream drift: 2 commits: **#62411 `416ff2dfd8e` (the split itself, 12 files +1031/-168) — now on main**, #62567;
  plus #62035 / #62488 in tt-llk and #61819 (topk SFPU headers moved to llk_sfpu). The fork's 4 LLK copies now differ
  from main's by 8-16 diff lines each (just the K' 512 change).
- Effort: **medium (~1.5 h)**: rebase the fork onto the new source (= main's split op + the K' 512 change), drop or
  shrink the LLK copy, keep the tie / identity tests (test_stock_identity.py).

### 4.9 rms_norm_ttnn (`ttnn.bringup.rms_norm`)
- Folder: `ttnn/ttnn/bringup/rms_norm_ttnn/` (21k lines, C++ host + Python twin).
- Derived from: AI-generated drop-in for `ttnn.rms_norm` (`dnijemcevic/rms_norm_replacement_run0` @ `31b0b1cbc98`);
  no upstream source to sync; last "Rebase onto llk_helper_library 6b80daf43e3: kernel API port" (2026-10-27).
- Call sites (shipped): `tt/norm.py:153, 154, 159, 171` (attn_norm, ffn_norm, q/kv norms, final norm; split mode with
  stats_out across chips); `tt/host_alloc.py:55` (output sizing).
- Why a fork: P.18 / precision-debt J4 (fp32 DEST accuracy fixes), split-across-chips norm with partial stats,
  residual fusion, caller buffers (P.16, DS41-174), pack_precise / bfp8_twin (P.21), max_grid (FIX.15); DS41-186.
- Delta: residual add + returned sum, any placement, TILE / ROW_MAJOR, split program (stats_out / partial_stats /
  input_window), output dtype, output_tensor, pack_precise, bfp8_twin, max_grid.
- Upstream drift: none to merge (`normalization/rmsnorm` 0 commits); depends on llk_helper_library kernel_lib
  (`tilize_helpers_dataflow.hpp`, extended reduce helpers).
- Effort: **low-medium (~0.75 h)**: build-only port; DS41-173 (its suites import a missing `eval` package) still to fix or waive.

### 4.10 mhc_pre_ttnn (`ttnn.bringup.mhc_pre_xing`, `mhc_pre_xing_pack`)
- Folder: `ttnn/ttnn/bringup/mhc_pre_ttnn/` (14.5k lines).
- Derived from: AI-generated (tt_ops_code_gen run 1045, `2026_09_30_0115_run1_mhc_pre` @ `963102ebe2f`); no upstream source.
- Call sites (shipped): `tt/mhc.py:117, 149, 175, 185`; `tt/collapse.py:50, 56`; `tt/norm.py:115` (exact sum x^2,
  n=1); `tt/engram_add.py:130, 143` (engram pack + `engram_gate_eps`, P.24 TINY_KERNELS).
- Why a fork: no stock mHC op; fused pre step (RMS stat, projection, Sinkhorn, collapse) in one program (P.14
  MHC_FUSED), exact fp32 TP partial sums (`partials`), engram pack / gates (P.24), caller buffers (P.16).
- Delta: xing entries (DeepSeek-V4 math), collapse_pre, partials, engram pack mode + engram_gate_eps, bf16 streams,
  hc_output / y_output, max_grid.
- Upstream drift: none to merge; depends on llk_helper_library `matmul_block_helpers.hpp` and the carried
  `perf_instrumentation.hpp`.
- Effort: **low-medium (~0.75 h)**: build port + parity tests; DS41-173 as for rms_norm.

### 4.11 mhc_post_ttnn (`ttnn.bringup.mhc_post`)
- Folder: `ttnn/ttnn/bringup/mhc_post_ttnn/` (6.3k lines).
- Derived from: AI-generated (run 1046, `2026_09_30_0115_run1_mhc_post` @ `84be38c18b7`); no upstream source.
- Call sites (shipped): `tt/residual.py:93, 113, 141, 146`; `tt/engram_add.py:259, 270`.
- Why a fork: no stock op; the mHC residual mix in one fp32 program (RESIDUAL_MIX=fused) and the next step's partial
  row from the same pass (stats_weight, MHC_FUSED), reading post / comb from packed mixes.
- Delta: stats_weight, post_col / comb_col, bf16 streams, output_tensor / stats_output / stats_scratch, max_grid.
- Upstream drift: none.
- Effort: **low (~0.5 h)**.

### 4.12 rotary_embedding_indexed (`ttnn.bringup.rotary_embedding_indexed`)
- Folder: `ttnn/ttnn/bringup/rotary_embedding_indexed/` (3.2k lines).
- Derived from: `experimental/deepseek_prefill/rotary_embedding_indexed`, Source SHA `d2dc8639a5fa` (2026-10-29).
- Call sites (shipped): `tt/rope.py:117, 133, 148, 174` (34 RoPE sites: q / kv / indexer chains, 136 calls a chunk).
- Why a fork: DS41-147 (the stock op stamps one program per chip: 101 us host a call on 8x4 -> 52 us with `uniform`),
  P.22 chain mode (split / to_layout / concat / cast chains as one op: q chain host 260 -> 61 us), FIX.15 max_grid,
  P.16 output_tensor.
- Delta: uniform, output_tensor, max_grid, input_heads, ROW_MAJOR in / out, output_dtype bfp8 / bfp4, group_heads.
- Upstream drift: 1 commit (#62567, 13 lines). The stock source also carries the branch's precompile change (#55962, not on main).
- Effort: **low (~0.5 h)**.

### 4.13 update_padded_kv_cache (`ttnn.bringup.update_padded_kv_cache`)
- Folder: `ttnn/ttnn/bringup/update_padded_kv_cache/`.
- Derived from: `experimental/deepseek_prefill/update_padded_kv_cache`, Source SHA `e3b3caa5c75e` (2026-10-26), synced to `6b80daf43e3`.
- Call sites (shipped): `tt/attention.py:569, 638, 736`; `tt/compressor.py:350, 363`.
- Why a fork: DS41-116: the stock op wants whole tiles a chip; 8x4's ratio-2 TP-dedup stripe is 80 rows (ROW_MAJOR non-tile stripes).
- Delta: ROW_MAJOR copy calls with per-chip stripes / depths in whole rows.
- Upstream drift: 1 commit (#62567, 3 lines).
- Effort: **low (~0.25 h)**.

### 4.14-4.16 nlp_concat_heads / slice_write / embedding (`max_grid` forks)
- Folders: `ttnn/ttnn/bringup/{nlp_concat_heads,slice_write,embedding}/`; sources `experimental/transformer/nlp_concat_heads`,
  `experimental/slice_write`, `embedding`, all Source SHA `222c8142326c` (2026-10-31).
- Call sites (shipped): nlp_concat_heads `tt/attention.py:1506, 1541`; slice_write `tt/attention.py:981, 987, 1022, 1163`,
  `tt/compressor.py:226, 252`; embedding `tt/attention.py:628`, `tt/compressor.py:377`, `tt/model.py:1224`.
- Why a fork: FIX.15 / agent rule 18 (owner 2026-10-30): a Galaxy PSU trips on di/dt, no op may use more than 110 of
  a chip's 120 Tensix cores; these stock ops take no core grid.
- Delta: one option, `max_grid` (interleaved work split inside a rectangle); default = stock program.
- Upstream drift: 1 commit each (#62567 for nlp_concat_heads / slice_write; #60730 round_up_to_mul32 -> tt::round_up
  for embedding; the old function still exists on main).
- Effort: **low (~0.25 h each)**. (Candidates for an upstream `sub_core_grids` port so they can be deleted.)

### 4.17 ckv_bfp4_ttnn (`ttnn.bringup.ckv_bfp4`, `bfp_round`)
- Folder: `ttnn/ttnn/bringup/ckv_bfp4_ttnn/` (Python ProgramDescriptor op, 1.5k lines).
- Derived from: none (new op, FIX.8, 2026-10-28).
- Call sites (shipped): ckv_bfp4 `tt/attention.py:965, 1002, 1031, 1101, 1142, 1172`, `tt/compressor.py:242, 722`;
  bfp_round `tt/indexer.py:237, 245` (and via the RoPE chain's output_dtype bfloat4_b).
- Why a fork: the reference's 4-bit compressed KV / fp4 index q / k: `ttnn.typecast` to bfloat4_b truncates in effect,
  so the device needs SFPU round-to-nearest-even + the 320 B row record layout sparse_sdpa ROW_BFP4 reads (DS41-150);
  prepared / compact for host time (DS41-135, DS41-145).
- Delta: n/a (new op): row records, blocks=True, prefix, compact, prepared.
- Upstream drift: none.
- Effort: **low (~0.25 h)**: re-run its tests.

### 4.18 pair_pool_ttnn (`ttnn.bringup.pair_pool`)
- Folder: `ttnn/ttnn/bringup/pair_pool_ttnn/` (Python ProgramDescriptor op, 0.6k lines).
- Derived from: none (new op, P.24 TINY_KERNELS, 2026-10-31).
- Call sites (shipped): `tt/compressor.py:539`.
- Why a fork: the ratio-2 compressor's softmax-gated pair pool was 16 launches a layer; one prepared program, bit-identical.
- Delta: n/a (new op).
- Upstream drift: none (uses stock SFPU bodies: covered by the symbol check in 3.2).
- Effort: **low (~0.25 h)**.

### 4.19 Carried non-fork MoE code (moe_block, tt_k3_token_split_moe.py, matmul_reduce_scatter)
- Covered in section 2. Call sites: `tt/moe_block_adapter.py:95-377` (build_moe_block_layer), `tt/model.py:530-540, 701-710`,
  `tt/prepared_tail.py:25` (TailA2A), `tt/host_alloc.py:175-191`; fused head `moe_block/tt/fused_head.py:536-539`
  (matmul_reduce_scatter), `se_rs.py:18` (plan_transport).
- Why not stock: Marko's fused MoE head (x gather + shared expert engine + router + routing exchange in one program a
  chip), token-split exchange and fused tail; owner decisions D1-D10 (`reports/moe_block_decisions_2026-10-27.md`),
  FIX.6 (flat hangs on the torus, DS41-130).
- Upstream drift: main n/a; Marko: 16 + 3 commits since `856d97d5864` (section 2).
- Effort: **medium (~3 h)**: no git conflicts, but 3 Python `get_forwarding_link_indices` fixes, JIT kernels that only
  fail on the device (se_engine3's engine, routing streams), the 1-file allocator memo conflict, and the owner's call on
  Marko's newer commits (taking them is extra, ~2-4 h more, and better done as its own task).

## 5. Stock TTNN ops the shipped model calls that changed on main (cheap check)

- `experimental/ccl/all_to_all_async_generic` (42 calls a chunk, MoE tail): 5 main commits, **and a carried branch change
  in the same factory** (see 3.1): conflict + Galaxy re-check.
- `matmul` / `linear` (144 + 41 calls, MATMUL_CONFIG=galaxy tuned configs, tt/mm_configs.py): 14 commits incl. #61818
  multicast helpers migration, #62457 zero per_core dims rejection: re-check that the tuned program configs still
  pass and keep their speed.
- `experimental/minimal_matmul` (2 calls): 4 commits incl. #59415 NoC flush before cb_pop_front (behaviour / perf).
- `ccl/all_gather` (stock fallback under fabric for untiled cases): #61955 fabric mux in program-local L1, #61503.
- unary (softplus / sqrt etc. in non-fused paths): #63324 hash scalar params by bits, #62645; typecast, slice, concat
  (#62063 concat ported to generated DeviceOperation), scatter, untilize_with_unpadding: cleanups / Quasar fixes only.

## 6. Verification cost (rebase skill steps 5 and 7, DS41 share)

- `fork_suite` over all forks (~1 h on a 2x2 per the skill) and re-recorded `tests/source_baseline.json` for the synced
  forks (sdpa, topk_large_indices, indexer_score, the `max_grid` four, update_padded_kv_cache, rotary).
- DS41 ladder on the 8x4 Galaxy (rungs, contract test, fork-call capture + coverage; P.17 took ~20 min a rung) and a
  perf comparison against FIX.22 (host-bound chunk: prepared / fork host costs are the sensitive part).
- Owner-visible risk: tt-llk SFPU changes may move bits (owner rule: never require identical bits, only the gated
  limits).
- About **4 h** of agent time, mostly waiting on device jobs.

## 7. Summary table

| op (folder) | kind | why a fork | delta vs stock | upstream drift to origin/main | effort |
|---|---|---|---|---|---|
| flat_routed_expert (flat_routed_expert_ttnn) | C++ fork, source = Marko's branch | MoE expert engine; no stock op | layouts, bfp4 down, weights_nk, patch_addresses, any I | main n/a; Marko +3 (DRISC NoC fix) | medium ~2.5 h |
| combine_rf.* (combine_rf) | Python JIT builders, source = Marko's branch | token-split dispatch / combine / planners; no stock op | landing pools, multi-core deterministic planner, bf16 acc | main: #61503 (2 Py sites); Marko +8 | medium ~2 h |
| prepare_program / run_prepared (prepared_program) | C++ fork, Marko's | host time on 32 chips | patched-slot prepared dispatch, value_slots | main 0; Marko +1 | low ~0.5 h |
| fabric_all_gather (fabric_all_gather_ttnn) | C++ fork, mimo-v2-dp | link-rate all-gather (P.5) | the op, persistent outputs | **build breaks: #61503, #61262** | medium ~1 h |
| fabric_reduce_scatter (fabric_reduce_scatter_ttnn) | Python fork, mimo-v2-dp | exact fp32 / deterministic RS, prepared | groups, exact_fp32_add, program_hash, prepared | none | low ~0.5 h |
| sparse_sdpa (sdpa) | C++ fork of transformer/sdpa | 4-bit KV rows, 110 cores, precision | ROW_BFP4/BFP8, kv_window, max_grid, precision opts | 7 commits; #60493 1147-line factory rewrite, #61759, #62434, #61503 build break | medium ~1.5 h (selective) / high ~5 h (full) |
| ring_indexer_score_dsa (indexer_score) | C++ fork | ratio 2, natural blocks, bfp4 q/k (DS41-49/-117/-150) | ratio 2, block epilogue, bfp4, fp32 DEST | 1 cleanup | low ~0.5 h |
| topk_large_indices | C++ fork | split LLK early + K' 512 body (DS41-151) | split body + own LLK copy, K' 512 | #62411/#62035/#62488 now on main (fork pre-carried them), #61819 | medium ~1.5 h |
| rms_norm (rms_norm_ttnn) | AI-generated, no source | P.18 accuracy, split norm, caller buffers | many opt-ins + max_grid | none; needs llk_helper_library kernel_lib | low-med ~0.75 h |
| mhc_pre_xing(_pack) (mhc_pre_ttnn) | AI-generated, no source | no stock mHC; fused pre / engram pack | xing, partials, engram gates, caller buffers | none; needs llk_helper_library kernel_lib | low-med ~0.75 h |
| mhc_post (mhc_post_ttnn) | AI-generated, no source | no stock mHC residual | stats_weight, packed mixes, caller buffers | none | low ~0.5 h |
| rotary_embedding_indexed | C++ fork | host time (DS41-147), chains (P.22) | uniform, chain mode, output_tensor, max_grid | 1 cleanup | low ~0.5 h |
| update_padded_kv_cache | C++ fork | non-tile stripes (DS41-116) | whole-row ROW_MAJOR stripes | 1 cleanup | low ~0.25 h |
| nlp_concat_heads / slice_write / embedding | C++ forks | 110-core cap (FIX.15) | max_grid | 1 cleanup each | low ~0.75 h total |
| ckv_bfp4, bfp_round (ckv_bfp4_ttnn) | new Python op | RNE bfp4 / bfp8 records (DS41-150) | n/a | none | low ~0.25 h |
| pair_pool (pair_pool_ttnn) | new Python op | 16 launches -> 1 (P.24) | n/a | none | low ~0.25 h |
| MoE block + K3 exchange + matmul_reduce_scatter | carried copy of Marko's branch (not on main) | fused MoE head / exchange / tail | DS41 options (+689/-128 vs Marko tip) | main n/a (no conflicts); #61503 (3 Py sites); Marko +19 | medium ~3 h |
| carried core changes (all_to_all_async_generic, fabric.cpp binding, allocator, moe_grouped_topk, hybrid_routed_expert_ffn, ttnn __init__/types) | git conflicts | - | - | main edited the same files | medium ~1.5 h |
| non-DS41 forks that must still build (dispatch, combine: #61503) | C++ forks | - | - | build break | low ~0.5 h |
| DS41 verification (fork_suite share, 8x4 ladder, perf vs FIX.22) | - | - | - | - | ~4 h |

**Total for DS41's ops: about 20 agent-hours with a selective sync of sdpa (range 15-25), about 24 h with a full
sdpa sync** — against `origin/main` of 2026-10-30. A newer main adds drift roughly in proportion; the highest-variance
items are the MoE stack (runtime-only failures in JIT kernels, Marko's newer fixes), the sdpa full sync, and the stock
all_to_all_async_generic conflict. Not included: the other bring-up models' verification on this box, the knowledge
staleness pass (skill step 8), and taking Marko's 30 newer MoE commits.

Riskiest items, in order:
1. The MoE stack (flat_routed_expert + combine_rf + prepared_program + moe_block copy): not on main, ~40 % of the chunk's
   device time, JIT kernels that break only on the device, Marko's branch 30 commits ahead (incl. a hang fix and a DRISC
   NoC fix).
2. #61503 / #61262 fabric API changes: build breaks in fabric_all_gather, sdpa ring factories, dispatch, combine, and
   seven Python call sites (one in the model's own tt/ccl.py).
3. The carried change in stock all_to_all_async_generic (shipped, 42 calls a chunk) vs four main commits in the same factory.
4. sdpa: #60493's factory rewrite if the owner picks a full sync.
5. The base question: llk_helper_library must come along (rms_norm, mhc_pre, sdpa, indexer_score, matmul_reduce_scatter
   use its kernel_lib helpers).
