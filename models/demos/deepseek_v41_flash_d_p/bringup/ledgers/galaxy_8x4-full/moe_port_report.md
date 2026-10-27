> CORRECTION (overseer, 2026-10-27 20:12, owner): the block OP supports torus (ttnn/ttnn/bringup/combine_rf/examples/tt_k3_token_split_moe.py:141 torus=(wrap X, wrap Y); Marko's generality map covers 8x4 as TORUS_XY). Only the model's adapter refuses it (tt/moe_block_adapter.py:159, assert citing D10). Statements below that "block refuses torus" mean the adapter, not the op.

# Porting the owner's MoE work (flat / block) onto the local DeepSeek-V4.1 8x4 bring-up

Written 2026-10-27, read-only, on node bh-glx-120-b06u08, repo /home/dnijemcevic/repos/tt-metal.
Refs: merge base `7c06953a55c` (the owner's rebased tip). Local `HEAD` = `5161e706d64`. Origin tip (fetched) =
`origin/dnijemcevic/model_bringup` @ `f81cb90cf5a`. Nothing in the repo was changed. Conflicts below come from
`git merge-tree 7c06953a55c HEAD origin/dnijemcevic/model_bringup` (git 2.34, the old three-tree form).
The uncommitted working-tree edits (tt/runners/kv_contract.py, tests/bringup/contract/*, ledger files) do not overlap
any file origin changed.

---

## 1. New commits on origin since 7c06953a55c

Every commit's author and committer is Djordje Nijemcevic (the owner), all on 2026-10-27 UTC. A = author time, C =
committer time. Three were re-committed at 18:54 when the owner rebased them.

| sha | A / C | subject |
|---|---|---|
| `14af221b826` | 16:41 / 18:54 | Sync flat_routed_expert_ttnn and combine_rf with Marko @ 856d97d5864 (mstaletovic/combine-lean-sender-llk); 86 files |
| `eae0d5cb27b` | 17:22 / 18:54 | EXPERTS_LAYOUT setting, Marko's flat-expert layout by default |
| `a746a4bb540` | 17:29 / 18:54 | Marko's MoE block as an opt-in replacement for the MoE layer (MOE_IMPL=block, default flat); 58 files |
| `8a83b0ca79e` | 18:55 | [bringup] skill: estimates in agent hours / device minutes |
| `9518ba45263` | 19:26 | Carry matmul_reduce_scatter + kernel_lib/perf_instrumentation.hpp from Marko, as is |
| `da4dcad040f` | 19:26 | Marko's fused MoE head behind MOE_HEAD (default ops), sqrt(softplus) fused router |
| `9e22a3bb115` | 19:29 | MOE_HEAD=fused the default, the combine snap kept |
| `f935ea08f65` | 19:44 | MOE_TAIL setting (ops / fused: Marko's TailA2A), default ops |
| `128e45f85be` | 19:50 | MOE_TAIL=fused the default |
| `f81cb90cf5a` | 19:51 | [bringup] F200: the clean wall is timed once the walls settle, not only once the cache is warm |

Local commits on top of the base (`git log 7c06953a55c..HEAD`): `5308dbb660d` (K.1 / F200 engine wait),
`4cb595545e7` (supervision), `be9736f2900` (FIX.5 patch_addresses), `5161e706d64` (F200 dgen_engine $USER expansion).

### Files by area (163 files changed in base..origin)

**ttnn/ttnn/bringup forks / new ops**
- `flat_routed_expert_ttnn/` (14af221b826): about 60 files. Typed layout options `FlatLayoutOptions`
  (device/flat_routed_expert_options.{hpp,cpp}, new `flat_layouts.py` with presets `LEGACY`, `DRISC_ONLY`,
  `GLM_FLASH`, ...). New defaults: DRISC-only gate/up readers, DRISC inside `_with_program`, even split. New
  kernel se5_xfwd.cpp. `output=`, `y_l1=`, `layout=` arguments. FlatProbes replaces MIMO_FL_* getenv. Three worklogs
  (FLAT_EXPERT, FLAT_COMBINE_OVERLAP, TS_HANG). New tests tests/unit/test_closeout_{plans,flat}.py and tests/microbench/.
  Every existing test is pinned to `LEGACY`.
- `combine_rf/` (14af221b826): sparse_combine_rf landing pool, typed options (`local_keep` now defaults to 8; DS41
  passes 2), piggyback credits. New modules routing_a2a.py, routing_pack.py, kv_push_plan.py, glu_act.py, l1_zero.py.
  `ttnn.freeze_program_hashes` on cached descriptors.
- `prepared_program/` (a746a4bb540): a new C++ op `ttnn.bringup.prepare_program` / `run_prepared` plus
  `prepared_program.generic_op`. It caches a generic_op program and patches only the io tensors' runtime-arg slots and
  CBs on a hit. Registered in `ttnn/ttnn/bringup/bringup_nanobind.cpp` (+5 lines) and `INDEX.md` (+2 lines).
- `ttnn/ttnn/operations/matmul_reduce_scatter/` (9518ba45263): Python and JIT kernels only, no C++ build. It is
  imported by moe_block/tt/se_rs.py.

**ttnn core (C++: needs a rebuild)**
- `ttnn/cpp/ttnn/operations/generic/freeze_program_hashes_nanobind.{hpp,cpp}`, `ttnn/cpp/ttnn-nanobind/__init__.cpp`,
  `ttnn/sources.cmake`, `ttnn/ttnn/__init__.py`, `ttnn/ttnn/types.py` (14af221b826).
- `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp` (9518ba45263; a kernel header, JIT).

**Model tt/ (models/demos/deepseek_v41_flash_d_p/tt)**
- `experts.py` (14af221b826, eae0d5cb27b, a746a4bb540): `flat_op()` split out of `_load_flat` and shared with the
  block. `EXPERTS_LAYOUT` → `layout=`.
- `shared_expert.py` (14af221b826, eae0d5cb27b): `layout=` (LEGACY, or marko with rows_hint=S, x_helpers=1).
- `weighted_combine.py` (14af221b826): `local_keep=2` in the combine and `layout=fe.layout` on
  `flat_routed_expert_with_program`.
- `perf_experts.py` (14af221b826): pins `layout=LEGACY`.
- `model.py` (a746a4bb540, da4dcad040f): the MOE_IMPL=block branch in `TtDS41Block.__init__` / `_fns`. Under the
  fused head, ffn_norm keeps the column split.
- `norm.py` (da4dcad040f): `build_norm(..., gather=None|False)`.
- New `moe_block_adapter.py` and `MOE_BLOCK.md` (a746a4bb540 onward).
- `settings.py`: EXPERTS_LAYOUT, MOE_IMPL, MOE_HEAD, MOE_TAIL.

**Tests**: no DS41 test file changed. New `models/demos/moe_block/tests/*` (about 30 files: test_moe_block.py,
test_moe_block_matrix.py, test_fused_head_parts.py, test_tail_a2a.py, protocol models, a mock-galaxy generality
map). Fork unit tests in flat_routed_expert_ttnn/tests pinned to LEGACY.

**Framework (models/demos/common/bringup)**: `testing/profile.py` + `selftest/test_warm_wall.py` +
`dev/BREADCRUMBS.md` (f81cb90cf5a: F200 wall settle). `skill/bringup/SKILL.md` (8a83b0ca79e).

**Other models**
- New `models/demos/moe_block/` (Marko's MoE block: moe_block.py plus tt/fused_head.py, head_plan.py, router_stream.py,
  routing_stream.py, se_engine3.py, se_rs.py, tail_a2a.py, tail_engine.py, x_gather3.py, prep_stream.py).
- New `models/demos/deepseek_v3_d_p/tt/moe/tt_k3_token_split_moe.py` (the token exchange the block runs).
- `models/demos/glm53_flash_d_p/tt/experts_ag.py` and 7 tests in `glm53_flash_d_p_lb/tests/test_flat_*.py`: pinned to
  LEGACY.

---

## 2. The two MoE modes

Selected by the Setting `MOE_IMPL`: env `DS41_MOE_IMPL`, defined at origin `tt/settings.py:359`. The settings prefix is
`DS41_` (settings.py:5, :466). The spec has no settings block, so the mode comes from the env or from the shipped
default. **Default: `flat`.** MOE_BLOCK.md:5 says "the Galaxy keeps flat meanwhile".

### 2a. flat (default): today's path, plus Marko's layout
- Steps: router (tt/router.py, ROUTER_IMPL), then experts (tt/experts.py: column gather, then
  `ttnn.bringup.flat_routed_expert_with_program` with the weighted combine `combine_rf.sparse_combine_rf` appended in
  the same program), then shared_expert (the flat op, bfp8, LoFi), then moe_add (fp32 partials, exact all_gather + SFPU
  add over TP).
- What changes at origin: Setting `EXPERTS_LAYOUT` (origin settings.py:288), **default `marko`**. It means Marko's
  op defaults (DRISC-only gate/up readers, DRISC inside `_with_program`, even split: gate/up on at most 64 cores, down
  on at most 32, no reader-down tails) plus `rows_hint = round(chunk*K/E)` and `x_helpers=1` (experts.py:384-398;
  shared_expert.py:140-142). `legacy` = `flat_layouts.LEGACY` = the plan the 8x4 run has used so far
  (`dict(drisc_dual=True, drisc_in_overlap=False, drisc_pure=-1, even_split=False)`).
- Effect: marko re-lays out the cached weights, so it writes **new weight-cache files**. On 8x4 the routed cache is
  285 GB in /scratch/dnijemcevic/bringup/deepseek_v41_flash_d_p/tt_cache, which takes about 75 min to build cold
  (about 117 s a layer). /scratch has 2.2 TB free. LEGACY keeps the old file names (CHANGELOG: FIX.4 `nk` key kept).
- Measured: 2x2 only. Device time 65.74 ms a chunk on marko vs 68.10 ms on legacy, experts section 15.88 vs 17.85 ms
  (settings.py:288 help text). **Not measured on 8x4.**

### 2b. block: Marko Staletovic's MoE block, the whole MoE layer as one call
- Wiring (origin model.py:359-363, 497-512): `build_moe_block_layer` (moe_block_adapter.py:243). The router,
  shared_expert and moe_add steps return `Keep()`. The experts step runs `TtMoEBlockLayer(x)` → `[1,1,S/SP,H/TP]` bf16,
  typecast to fp32 when GLUE_FOLD is off (adapter:228-240).
- There is one `MoEBlock` per (mesh, chunk), shared by all layers (adapter:208-226). It is built at the first call,
  after the load-time L1, because the exchange bakes the arena's address. Each layer's weights are a `MoELayer`.
- Inside the block (MOE_BLOCK.md:21-33):
  - **Head.** `MOE_HEAD` (origin settings.py:375), **default `fused`**:
    - `fused`: one program a chip (moe_block/tt/fused_head.py HeadSE3). It gathers x over TP on its ports (ffn_norm no
      longer gathers: model.py:330-332, norm.py:188), then runs the shared-expert engine (bfp8, LoFi, clamped SiLU 10)
      with a row reduce-scatter, the router (logits HiFi3, fp32 sums in a fixed order, sqrt(softplus) on the fp32
      polynomial, + bias, top 6, weights / sum x 1.5), the routing exchange (combine_rf.routing_a2a), the combine
      planner and dispatch prep.
    - `ops`: a program a stage. ttnn.linear router partial over H/TP (HiFi4), `ttnn.reduce_scatter`, then softplus and
      sqrt, then `combine_rf.router_topk`. The shared expert is ttnn matmuls (bfp8, LoFi) plus `fabric_reduce_scatter`.
  - **Exchange** (`deepseek_v3_d_p/tt/moe/tt_k3_token_split_moe.py`, routed-only): `sparse_all_gather_2d` dispatch,
    then `ttnn.bringup.flat_routed_expert_with_program` (our FlatRoutedExpert weights, weights_nk, LoFi, bfp8 x / h,
    `down_fp32=True`, `layout=`) plus the `sparse_combine_rf` combine in one program, then tilize. Supporting launches
    go through `ttnn.bringup.prepared_program`.
  - **Tail.** `MOE_TAIL` (origin settings.py:389), **default `fused`**: `fused` = TailA2A
    (moe_block/tt/tail_a2a.py: tilize + add shared + TP all-to-all in one program); `ops` = tilize, add,
    `all_to_all_async_generic`.
- Hard requirements, asserted in the adapter:
  - `EXPERTS_PRECISION=glm` (D8, adapter:147).
  - `EXPERTS_IMPL=bfp4` (adapter:151). MoEBlock also asserts bfloat4_b (moe_block.py RoutedExperts).
  - Config equals `DSV41_FLASH` (top 6, H 5120, I 2304, route_scale 1.5, limit 10; adapter:53-71). The expert count is
    taken from the checkpoint (384 on 8x4).
  - **No torus**: `assert not (g.sp_wraps or g.tp_wraps)` (adapter:159, D10). Under FABRIC_2D_TORUS_* the block refuses
    to build.
  - `EXPERTS_LAYOUT=marko` must equal `RoutedExperts.layout_of(g, m)` (adapter:178-179).
  - `LINKS = 2` links a direction (adapter:47). `COMBINE_SNAP = 2^-19` (adapter:48).
  - Geometry (moe_block.py:219-224): E % chips == 0 (384/32 = 12, fine), and chunk % (chips*32) == 0. On 8x4 that is
    5120 % 1024, fine. **Any chunk that is not a multiple of 1024 tokens on 8x4 is refused** (the moe_block docstring
    says "Not built yet: ... padded chunks").
- Measured on the 2x2 only (chunk 2048, 8 layers, rung last; MOE_BLOCK.md:95-102):

  | path | device ms / chunk | clean wall ms / chunk |
  |---|---|---|
  | flat | 65.74 | 138.0 |
  | block, ops head + ops tail | 66.9 | 160.6 |
  | block, fused head + ops tail | 61.97 | 102-125 |
  | block, fused head + fused tail | 61.2 | 110.8-126.3 |

  Accuracy on the 2x2: ladder L.s4096 / L.last / L.s55296 pass, within about 1% of the ops head (MOE_BLOCK.md:117-122).

### Ops used, per INDEX.md and CHANGELOGs at the origin tip

| op | flat | block | change at origin |
|---|---|---|---|
| `ttnn.bringup.flat_routed_expert_with_program` (flat_routed_expert_ttnn) | yes (weighted_combine) | yes (exchange) | layout options; y_l1; new defaults (CHANGELOG "Sync with Marko @ 856d97d5864") |
| `FlatRoutedExpert` (flat_expert.py) | routed + shared | routed (via `_RoutedWeights`, adapter:105) | `layout=`, `replanned()` |
| `combine_rf.sparse_combine_rf` | column combine (axis 0, deterministic) | whole-mesh combine (snap or deterministic) | landing pool, local_keep |
| `combine_rf.combine_plan` (planner) | yes | ops head only (det planner) | MC2 sizing |
| `combine_rf.routing_a2a`, `routing_pack`, `glu_act`, `l1_zero`, `dispatch_prep`, `sparse_all_gather_2d`, `router_topk` | no | yes | new or synced |
| `ttnn.bringup.prepared_program` (new C++) | no | yes (exchange, head, tail) | new |
| `ttnn.bringup.fabric_reduce_scatter` | no | ops head (shared rows) | — |
| `ttnn/operations/matmul_reduce_scatter` | no | se_rs.py imports `plan_transport` | carried |
| `ttnn.bringup.moe_grouped_topk` | router (ROUTER_IMPL) | no | — |

### Mesh / box support

| | flat | block |
|---|---|---|
| 2x2 QuietBox (spec says 4x p300c, FABRIC_2D) | yes; the origin A/B ran there | yes; only box it ran on |
| galaxy_8x4 (32 x p150b, 12x10 grid, FABRIC_2D) | yes with legacy (W.1 … L.s56320 PASS); **marko layout untested on 8x4** | untested; Marko's generality map (moe_block/tests/test_generality_map.py:57-61) covers 8x4 only as **TORUS_XY**, which the DS41 adapter refuses |
| galaxy 8x4 FABRIC_2D_TORUS_XY | **hangs** in flat_routed_expert_with_program (supervision.md:272-278) | refused by adapter:159 (D10) |
| galaxy_4x4 (TORUS_XY box in spec) | as above | refused |

Fabric: the flat path's combine needs a 10880 B payload (weighted_combine.py:56-57). The block needs at least
10272 B and `l1_small_size` 24576 (MOE_BLOCK.md:164-165). `LINKS=2` is hard-coded in both weighted_combine.py:56 and
adapter:47.

### Program-cache behaviour

| path | does the key hold buffer addresses? | warm under F196? |
|---|---|---|
| origin flat (no FIX.5) | **Yes.** weighted_combine keys `self.seen` on `a(xg)`, ... (origin weighted_combine.py:342, 478-493). The C++ hash includes `p.addrs` (origin flat_routed_expert.cpp:282-299). | **No on 8x4** (FIX.5 brief: +20..40 programs a run, never warm) unless `EXPERTS_XG_PINNED=on` (an overseer hack, default off) |
| our flat + FIX.5 (`EXPERTS_PROGRAM_KEY=patch`, local settings.py:290) | **No.** The hash holds no address. The flat AddrPatch list, arena CBs, writers' args and the combine's args (via `extra_addrs`) are patched on a hit. | **Yes.** 8x4, 40 layers: warm after 1 run (237 → 237 entries), median wall 999.8 ms (be9736f2900 BREADCRUMBS) |
| block | **Yes**, address-keyed. The exchange avoids misses by pinning its buffers: a persistent dispatch buffer pair (tt_k3_token_split_moe.py:1005-1020), an arena asserted at a fixed address (:1080-1090), a persistent y, and a Python `_seen_keys` tuple of addresses (:1118-1135). The key includes each layer's weight addresses, so 40 layers give 40 programs. prepared_program patches io addresses on hits. | 2x2: warm after 1 run (MOE_BLOCK.md:104). 8x4 unknown. The fused tail's host descriptor cache rebuilds on 17 of 80 calls (MOE_BLOCK.md:109-110). |

Note: the FIX.5 brief from the owner forbids pinning activations to dodge the key ("do not pin the model's activations
at fixed addresses", tasks.yaml FIX.5 brief). The block exchange does exactly that, so this is a question for the
owner (section 6).

### Precision

| | flat (shipped defaults) | block |
|---|---|---|
| routed weights | bfp4 (`EXPERTS_IMPL=bfp4`, owner pick P.1, default since P.10) | bfp4, asserted |
| routed math | `EXPERTS_PRECISION=glm`: LoFi, bfp8 x / h, fp32 DEST (down_fp32) | the same; glm only (D8) |
| routed sums (combine) | `COMBINE_ACC=bf16`: bf16 running sums, one pass (P.9), fixed order; fp32 partial output | fp32 accumulators (COMBINE_ACC ignored). Fused head: each term snapped to 2^-19, so sums are order-independent while below 32 (DS41 peak 5.4, measured on the 2x2 s4096 golden). Ops head: deterministic order. |
| shared expert | flat op, bfp8 weights quantized from [N,K], LoFi | bfp8 from W^T (groups along N; DS41-35: not exact for fp8), LoFi (D5) |
| router | whole-row logits HiFi2, fp32 DEST, moe_grouped_topk | fused head: HiFi3 logits, fp32 fixed-order sums. Ops head: HiFi4 partial over H/TP plus the CCL's fp32 add (about TF32), stock softplus. Near-tie picks can move. |
| MoE output | fp32 partials, exact fp32 TP sum (moe_add) | bf16 TILE (typecast to fp32 if GLUE_FOLD=off) |

These settings have no effect under block (MOE_BLOCK.md:144-149): ROUTER_IMPL, ROUTER_SOFTPLUS, COMBINE_ACC,
COMBINE_PLANNER, EXPERTS_SUM, EXPERTS_XG_PINNED, CCL_IMPL (for the MoE collectives), MATMUL_FIDELITY (for the router).
EXPERTS_PROGRAM_KEY (ours) would be ignored too: the block calls the op without `patch_addresses`.

---

## 3. Conflicts with our local commits

14 files are changed on both sides. Textual conflicts are marked **C**; the rest merged cleanly.

| file | | ours | theirs | reconcile |
|---|---|---|---|---|
| models/demos/common/bringup/dev/BREADCRUMBS.md | C | F200 engine-wait entries (5308dbb660d, 5161e706d64) | F200 wall-settle entry (f81cb90cf5a) | keep both. **ID collision: two different F200s.** Renumber one (owner's call), including the `F200` mentions in origin testing/profile.py comments and prints and in selftest/test_warm_wall.py |
| tt/settings.py | clean | EXPERTS_PROGRAM_KEY, EXPERTS_XG_PINNED text | EXPERTS_LAYOUT, MOE_* | keep all |
| tt/weighted_combine.py | C | `patch_addresses=patch, extra_addrs=extra_addrs` | `layout=fe.layout` (+ `local_keep=2` merged clean) | pass all three kwargs |
| ttnn/ttnn/bringup/INDEX.md | C | "... and patch_addresses (FIX.5)" | "typed layout options (layout=, presets ...)" | one line with both |
| flat_routed_expert_ttnn/CHANGELOG.md | C | FIX.5 entry | Sync with Marko entry | keep both, in date order. Add a note to FIX.5 that it now sits on the synced op (y_l1 / layout in `flat_key`) |
| flat_routed_expert_ttnn/README.md | C (2 hunks) | signature with patch_addresses / extra_addrs; test list adds test_with_program_patch.py | signature with y_l1 / layout; test list | signature `..., relay_words=None, y_l1=None, layout=FlatLayoutOptions(), math_fidelity=0, weights_nk=False, patch_addresses=False, extra_addrs=[]`; merge both test lists |
| device/flat_routed_expert_program_factory.cpp | C (1 hunk, :958) | a clang-format reflow of `xm.lit(...)` | `p.xdual ? NR : idx + NR` | take theirs. Our other change there (`apply_addresses` split out of `override_runtime_arguments`) merges clean |
| device/flat_routed_expert_program_factory.hpp | clean | `apply_addresses` declaration | comment rename | — |
| device/flat_with_program_device_operation.cpp | C | the FIX.5 hit-path patching (extra_cbs, by_src, l1 args) | only `FlatProbes::from_env().flat_wp_keylog / flat_wp_log` instead of `std::getenv`, and the message "(y_l1?)" | take ours, then apply theirs' two FlatProbes swaps and the message |
| device/flat_with_program_device_operation.hpp | clean | patch fields, hash | comment | — |
| flat_routed_expert.cpp | C | `setenv("MIMO_FL_Y_L1")` inside `if (!l1_args.empty())`; `if (!patch_addresses)` around `p.addrs`; non-owning `extra` view | setenv removed (now `cfg.y_l1 = y_l1.value_or(!l1_args.empty())`, origin :262) | drop the setenv. Keep `if (!patch_addresses) {` around the l1_args / tensor / optional addresses. Keep the non-owning view and `.patch_addresses / .extra_addrs`. Signature: `..., relay_words, y_l1, layout, math_fidelity, weights_nk, patch_addresses, extra_addrs` (the merged .hpp and nanobind already have this order) |
| flat_routed_expert.hpp | clean | +patch_addresses, extra_addrs | +y_l1, layout | merged order checked: y_l1, layout, math_fidelity, weights_nk, patch_addresses, extra_addrs |
| flat_routed_expert_nanobind.cpp | clean | the two args | layout / output / y_l1 | merged `nb::arg` order matches the .hpp (checked) |
| tests/weighted_case.py | C | `patch_addresses=..., extra_addrs=...` | `layout=fe.layout` | keep both |

### Is FIX.5 superseded? No. It is complementary, and the conflicts are textual only.

- The synced op on origin still hashes every baked address. `p.addrs` gets x, counts, regions, ids, weights, done,
  arena, words, y, reader_down and token_index (origin flat_routed_expert.cpp:282-299), and origin weighted_combine.py
  still keys `self.seen` on `a(xg)` (:478). Without FIX.5, the origin flat path on 8x4 goes back to never-warm (the
  F196 census).
- The supervision hand-off said "the owner says a NEW flat expert op is in the works (also fixes torus): when it lands,
  drop FIX.5" (supervision.md:323-325). This sync is not that op, or at least it does not do either thing.
- FIX.5's patch mechanism still covers the synced op:
  - All address args in the origin program factory still go through `AddrSrc` (52 uses, `apply` at :1737-1748).
  - The new DRISC job-posting args take the gate/up address via `.a(AddrSrc::GateUp)` and the arena via
    `.a(AddrSrc::Arena)` (origin program factory :802-804), so they are patched too.
  - The new combine's extra baked addresses (`shared_rows`, `barrier`; origin sparse_combine_rf.py:2372, 2384) are only
    used by whole-mesh options that `axis=0` refuses (combine_rf CHANGELOG), so DS41's `extra_addrs` list is unchanged.
  - `layout.key()` and `cfg.y_l1` are in `p.flat_key`, which the patch-mode hash includes.
- Not verified: FIX.5 was tested on LEGACY only. Marko's default (DRISC inside `_with_program`, no reader-down tails)
  has not run in patch mode. Re-run `test_with_program_patch.py` on both layouts.
- `test_with_program_patch.py` builds `FlatRoutedExpert` without `layout=`. After the merge it silently gets Marko's
  defaults, while every other fork test pins LEGACY. Parametrize it on LEGACY and the model's marko layout.
- Block mode does not use FIX.5. Extending `patch_addresses` to the exchange (tt_k3_token_split_moe.py:1112-1165) would
  remove its per-layer programs and the buffer pinning. That is optional and is the owner's call.

### Torus hang: not fixed by origin, as far as the code and docs show

- Nothing in the flat_routed_expert_ttnn or combine_rf CHANGELOG entries at the origin tip claims a torus fix.
  TS_HANG_WORKLOG.md is a GLM LoudBox NOC0 hang on FABRIC_2D. FLAT_EXPERT_WORKLOG.md:708 lists the Galaxy's
  FABRIC_2D_TORUS_XY as "untestable here".
- The DS41 flat path still passes no torus information to its column combine (origin weighted_combine.py has no
  topology argument).
- The block supports torus in Marko's own geometry (moe_block.py:234-235). The DS41 adapter refuses it (adapter:159,
  D10).
- So the 10:25 torus hang (flat_routed_expert_with_program cores waiting in se3/se6, DRISC "ready, done 0";
  supervision.md:272-276; logs in posprof/pos0.log) should be presumed **still open** until retried. If retried, the
  candidates are flat + marko layout on TORUS_XY, or the block after D10 is lifted.

---

## 4. What the port needs on our side

**Settings (tt/settings.py)**: everything below arrives with the merge.
- Keep ours: `EXPERTS_PROGRAM_KEY` (patch default), `EXPERTS_XG_PINNED` text.
- Take theirs: `EXPERTS_LAYOUT` (default marko), `MOE_IMPL` (flat), `MOE_HEAD` (fused), `MOE_TAIL` (fused).
- Add to the MOE_IMPL / MOE_BLOCK.md text that `EXPERTS_PROGRAM_KEY` has no effect under block.
- Decide the 8x4 default for EXPERTS_LAYOUT. Recommendation: start with `DS41_EXPERTS_LAYOUT=legacy` to show the merge
  is bit-identical, then switch to marko as the owner's default.

**Model files**
- tt/experts.py, shared_expert.py, model.py, norm.py, perf_experts.py, moe_block_adapter.py, MOE_BLOCK.md: theirs as is
  (no local changes).
- tt/weighted_combine.py: union of the kwargs (section 3). In patch mode `extra_addrs` stays
  `[arena, words, y, out, plan, table, dord]`.
- tt/perf_flat_program_key.py (ours): unchanged. It builds through `build_experts`, so it follows EXPERTS_LAYOUT.
- `perf_experts.py` pins LEGACY in its harness (origin :334-337), so its numbers measure legacy, not the shipped marko.

**Tests**
- Fork: test_with_program_patch.py (see above); weighted_case.py union.
- The bring-up's component / swap tests (bringup/hooks.py `_device_step_fn`) never exercise the block. Only the ladder
  (test_ladder.py through tt/model.py) does (MOE_BLOCK.md:169-172).
- O.1 (bringup-fork-tests) has no model cases for the block's fork calls yet (MOE_BLOCK.md:173-175). Re-capture fork
  sigs: `layout` and `extra_addrs` are new.

**8x4 literals to watch**
- Hard-coded: `LINKS = 2` (adapter:47, weighted_combine.py:56), `FABRIC_PAYLOAD = 10880` (weighted_combine.py:57),
  `COMBINE_SNAP = 2^-19` (adapter:48; the "peak 5.4" premise is from 2x2 s4096), fused-head `GX=min(11, grid.x-1)`
  (moe_block.py:1138).
- MOE_BLOCK.md quotes 2x2 numbers ("engine 10 x 8 cores, 2 row chunks", "arena 1256 KB of 1279 KB", "H/TP 80 tiles,
  512 owned rows a chip"). On 8x4 with chunk 5120 the figures are: 160 owned tokens a chip, 640 a mesh row, N=32
  sources for the combine, H/TP = 1280 (40 tiles).
- Already derived from the mesh: experts per chip and the expert permutation (`expert_permutation(rows, cols, epc)`,
  adapter:93), rows_hint (`chunk*K/E`, which is 80 on 8x4 at chunk 5120), cap (`max(chunk, 256)`), Geometry.
- The `router_split` option (moe_block.py:154-157) is described as "for meshes whose chain binds (the Galaxy's N 32)".
  The adapter does not set it (default off).
- The test harness `test_with_program_patch.py:76` picks E = 384 if R*C == 32 else 96, a box-specific test literal.

**Build**: a full `./build_metal.sh` is required (CMake re-configure for the new GLOBbed `prepared_program/` and the
new `flat_routed_expert_options.cpp` in sources.cmake). It covers:
- the flat_routed_expert_ttnn C++ (both sides changed);
- the new `prepared_program` op and its bringup_nanobind registration;
- the core `freeze_program_hashes` nanobind (ttnn/sources.cmake).

The tt_k3 exchange imports `ttnn.bringup.prepared_program` at module load, so block mode fails at import without the
build. matmul_reduce_scatter and perf_instrumentation.hpp need no build. DRISC uses DRAM programmable cores
(auto-detected; `TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES` overrides it, and the mock generality map sets it
to 1).

---

## 5. Port plan (ordered)

0. **Prepare.** The orchestrator and the uncommitted agent edits are in flight. Do the port in a separate worktree
   (`git worktree add`) or after the owning agent has committed its kv_contract / contract-test edits. Never stash
   them. Follow the hand-off rule: work from the branch the owner pushed.
1. **Rebase our 4 local commits onto origin** (`git rebase --onto origin/dnijemcevic/model_bringup 7c06953a55c`, or
   cherry-pick 5308dbb660d, 4cb595545e7, be9736f2900, 5161e706d64). Resolve as in section 3: BREADCRUMBS F200
   renumbering, and the C++ union in flat_routed_expert.cpp (drop the setenv) and flat_with_program_device_operation.cpp.
   Run clang-format (repo style) on the changed C++ and black 23.10.1 on Python, as the CHANGELOG did.
2. **Build:** `./build_metal.sh` (full, re-configure). Then run the box test: B.1 `test_box.py` (seconds).
3. **Fork tests on device (8x4):**
   - `BRINGUP_TEST_MESH=8x4 scripts/run_safe_pytest.sh --run-all --no-precompile
     ttnn/ttnn/bringup/flat_routed_expert_ttnn/tests/unit/test_with_program_patch.py`, once with LEGACY and once with
     marko (after parametrizing).
   - `tests/unit/` on one Galaxy chip (`--device auto`, as FIX.2).
   - Marko's `test_closeout_plans.py` (host) / `test_closeout_flat.py`.
4. **Flat + legacy identity:** `DS41_EXPERTS_LAYOUT=legacy`, gate W.1 (`BRINGUP_RUNG=seed test_ladder.py`). Expect
   metrics identical to the FIX.5 result (ledger results/FIX.5.json, e.g. err_layer_L24 0.031249). This proves the
   merge alone changes nothing.
5. **Flat + marko** (the origin default):
   - Pre-build the new weight cache outside a gate: the first 40-layer load is about 75 min, over the 3600 s gate
     timeout (supervision.md hand-off gotcha).
   - Re-run W.1, M.1 (`BRINGUP_RUNG=s10240`), then L.s10240, L.last, L.s56320, and L.smoke (test_smoke.py).
   - Check the `EXPERTS_WEIGHTED ... passes=` log line. COMBINE_ACC=bf16 asserts one pass (experts.py `assert
     self.combine_acc != "bf16" or self.wc.P == 1`), and marko's split changes the free cells / accumulators.
6. **Block** (`DS41_MOE_IMPL=block`, fused head + fused tail by default; FABRIC_2D only):
   - First Marko's standalone test, adapted to the Galaxy: `MOE_MESH=8x4 ... models/demos/moe_block/tests/test_moe_block.py::test_moe_block -k dsv41_flash-...`
     (MOE_BLOCK.md:188-198 gives the 2x2 form with `MOE_L1_SMALL=24576 MOE_FABRIC_PAYLOAD=10880`).
   - Then W.1, M.1, L.s10240, L.last, L.s56320 with `DS41_MOE_IMPL=block`.
   - Then `DS41_MOE_HEAD=ops` / `DS41_MOE_TAIL=ops` as the A/B fall-back if fused does not fit L1 on 8x4.
7. **Measure** each mode at the 5 positions (context 0 / 51200 / 102400 / 204800 / 512000, one 5120 chunk, 40 layers):
   - Device time: `/scratch/dnijemcevic/posprof/run.sh`. It runs test_profile.py with `pos_profile_plugin`, writes
     results/`${TAG}<pos>{,_profile,_ops}.json` and reports plain / profiled `*_new_programs` (F197). Summarize with
     `TAG=... python /scratch/dnijemcevic/posprof/summary.py 0 51200 102400 204800 512000`.
   - Host wall (F196 warm, misses refused): `/scratch/dnijemcevic/posprof/run_wall.sh` (BRINGUP_PROFILE_WALL=1). The
     origin F200 now also waits for the walls to settle (WALL_SETTLE_WINDOW 5, 5%, at most 30 runs) and records
     `chunk_wall_settled`, `chunk_wall_cache_warm_runs`, `chunk_wall_loadavg`. run_wall.sh hard-codes `BRINGUP_TASK=WALL$P`:
     copy it and add `${TAG}` so the modes do not overwrite each other's results.
   - Per mode, prefix the env:
     - flat legacy: `DS41_EXPERTS_LAYOUT=legacy TAG=FLATL`
     - flat marko: `TAG=FLATM` (defaults)
     - block fused: `DS41_MOE_IMPL=block TAG=BLKF`
     - block ops/ops: `DS41_MOE_IMPL=block DS41_MOE_HEAD=ops DS41_MOE_TAIL=ops TAG=BLKO`

     Keep `DS41_EXPERTS_PROGRAM_KEY=patch` (the default) for flat. Never set `EXPERTS_XG_PINNED` for the record numbers.
   - Compare against the 8x4 baselines (FABRIC_2D):
     - device time 329.9 / 350.6 / 360.0 / 372.7 / 403.7 ms, from before the warm fix, so MoE is slightly inflated;
     - warm 500k: 383.5 ms;
     - FIX.5 warm wall at 0k: median 999.8 ms. The earlier 860 ms was XG_PINNED.
   - Compare MoE as the sum router + experts + shared_expert + moe_add, because under block everything lands in
     "experts" and the head's programs show as "(other)" (MOE_BLOCK.md:112-115).
   - Use back-to-back alternating runs (A, B, A, B) for walls (origin F200 note: 102 vs 120 ms between processes).
   - The framework's `models/demos/common/bringup/tests/test_positions.py` (X.3) gives wall only. The posprof plugin is
     the device-time tool. Sections need a per-position profile (supervision.md:255-256 framework gap).
8. **Torus, optional.** With the spec on FABRIC_2D_TORUS_XY, retry flat (marko) at position 0 under a STOP-able series
   (expect the 10:25 hang again). Block needs D10 lifted by the owner. The indexer's `Topology.Linear`
   (tt/indexer.py:121) still has to be fixed for torus.

---

## 6. Risks and open questions for the owner

1. **F200 ID collision.** Local F200 (engine wait + $USER expansion) vs origin F200 (wall settle). Which keeps the
   number?
2. **FIX.5 stays.** The synced op still keys on addresses. Confirm that FIX.5 is kept and that the expected "new flat
   expert op that fixes torus" is still to come. Should `patch_addresses` be extended to the block exchange, which
   pins buffers against the FIX.5 brief's rule?
3. **EXPERTS_LAYOUT=marko on 8x4.** It is decided on a 2x2 A/B only. On the 12x10 Galaxy grid it changes the plan
   (no reader-down tails at H 5120, DRISC in `_with_program`), needs a new 285 GB weight cache, and could change
   COMBINE_ACC=bf16's one-pass assertion. Should the Galaxy default stay `legacy` until W.1 / L.* pass on marko?
4. **Block on 8x4.** It has never run there:
   - L1 fit of the fused head / exchange arena with 32 sources is unknown.
   - `LINKS=2` is assumed.
   - `router_split` is unset (the docstring says the Galaxy's N 32 binds).
   - The COMBINE_SNAP 2^-19 premise (sums < 32) was checked only on the 2x2 s4096 golden, not over 40 layers at 8x4.
5. **Torus.** D10 forbids the block on torus, yet Marko's own 8x4 target is TORUS_XY. The flat path hangs on torus. Is
   torus still wanted, and which path gets it first?
6. **Serving (K.1, the uncommitted kv_contract work).** The block refuses chunks that are not a multiple of
   chips*32 = 1024 tokens on 8x4 and has no padded-chunk support. Does tt-d-gen's runner ever drive such chunks?
7. **Precision deltas under block** (bf16 output vs fp32 partials, a different router numeric path, shared-expert bfp8
   grouping along N) are judged by the ladder only. Swap and component tests cannot see the block without a framework
   change (one `moe` step).
8. **Stale docs.** MOE_BLOCK.md's "Untested" list (:159-176) predates its own measurements, for example "never run
   anywhere". The decisions reports it cites (`reports/moe_block_decisions_2026-10-27.md`,
   `reports/moe_block_for_ds41_2026-10-27.md`) are not in the repo. The A/B logs path
   `/localdev/dnijemcevic/bringup/fused_head_2026-10-27/` does not exist on this node.
9. **2x2 cross-checks cannot run on this Galaxy** (FIX.5: "a 2x2 mesh cannot open on this Galaxy"). The owner's 2x2
   numbers cannot be reproduced here, so every 8x4 comparison must be within this node.
