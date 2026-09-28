# MiniMax-M3 prefill: new torus MoE ops, ND-sharded experts, SP=2/4/8 — work log and results, 2026-09-28

Host bh-glx-120-b09u02 (Blackhole galaxy, 8x4 = 32 chips). bf4 routed experts, untraced, `tt-smi -glx_reset` before
every device process. Everything is on **local branches (nothing pushed)**. Raw data: `m3_budget_study/results_torus/`
(`runs.csv`, `logs/`, `profiles/`, `sp8/`, `nd/`, `skew_check/`) and the per-phase summaries `SUMMARY_phase1.md`,
`SUMMARY_phase2.md`, `SUMMARY_nd_sp4.md`, which this report consolidates.

---

## 0. Bottom line

1. **New torus MoE ops work on M3 and cut its MoE communication.** On the full 8x4 torus (SP=8, EP=32),
   `dispatch_fabric2d` + `combine_fabric2d` make a sparse layer **−20% on device** (14.53 → 11.64 ms) and **−13 to −14%
   wall** for 8 sparse layers (W=8192: 106.2 → 91.0 ms). PCC unchanged (min K/V 0.9954 vs 0.9958).
2. **They need a wrap-wired ring of ≥ 4 chips on the dispatch axis**, so they apply to SP=8 (whole galaxy) and to the
   **single** 4x4 torus inside a galaxy (middle rows 2–5), **not to SP=2** (2x4) — and on SP=2 they would not help: every
   send is one hop, and SP=2's MoE is expert-weight-read bound.
3. **On the real 4x4 sub-torus (SP=4) the new ops are worth −11 to −13%** net of the torus-fabric cost.
4. **ND-sharded expert weights (+ the hybrid low-token path) help SP=2:** expert matmul **−24%** on the worst chip,
   stage wall **−3 to −5%**. No effect on SP=8 (4 experts per chip, not weight-read bound).
5. **Chip efficiency is now a tie between SP=2 (ND + hybrid) and SP=4-on-a-real-torus (new ops), ≈ 30 chip-µs per
   token-layer; SP=8 stays 1.3–2× worse.** Because only one 4x4 torus exists per galaxy, the best 4-galaxy throughput
   layout is still **4 independent galaxies × a 4-stage 2x4 (SP=2) pipeline** (~39–41k tok/s simulated, before the
   ND/hybrid gain). SP=8 layouts reach only 12–14.6k tok/s but are the only ones meeting hot-request p99 ≤ 1.5 s.
6. **Correction:** the "one hot expert per layer, same for every prompt" result was an artifact of isolated layer-set
   runs (raw embeddings fed into the first built layer). With real hidden states the hot experts change with the
   prompt, but single-document skew is still large (one expert in 20–90% of tokens; busiest chip 2.5–7.4× the mean at
   EP=32). MoE costs from isolated layer-set runs are therefore **pessimistic**, most of all at EP=32.

---

## 1. Setup done today

- `main` pulled to **e2dfc57** (GitHub token fine, no re-auth), submodules updated, full rebuild (Release, Tracy on,
  unity, clang-20). Later merged `main` again at **1143f58** (brings PR #57850) and rebuilt; rebuilt a third time for the
  bf8 combine change.
- Study branch `vmelnykov/m3_prefill_sp2_study` rebased onto new main → **`vmelnykov/m3_moe_torus`** (one conflict in
  `models/demos/common/prefill/runners/prefill_producer.py`: main's new `_chunk_slice`/`actual_isl` push path; our
  in-flight gate and deep-prefix wrap re-applied on top of it).
- Branches (all local):

| branch | head | content |
|---|---|---|
| `vmelnykov/m3_moe_torus` | 59a5723 | study tools on main e2dfc57 + M3 MoE knobs (Ring, combine v2) |
| `vmelnykov/combine_fabric2d_bf8` | c85e20a | `combine_fabric2d` accepts a BFLOAT8_B TILE input (worktree `/home/vmelnykov/tt-metal-wt-bf8`) |
| `vmelnykov/m3_moe_fabric2d` | 0be73de | the above + PRs #57859/#57654 (dispatch_fabric2d) + bf8 combine + main 1143f58 (#57850) + all knobs + results |

Main commits on `vmelnykov/m3_moe_fabric2d` (first-parent): 59a5723 knobs → 2468088 merge #57859/#57654 → 256d2b1
dispatch v2 knob + Ring o_proj fix + load stats → 9a6f756 merge bf8 combine → 191eeb2 feed bf8 into combine v2 →
e808dab BUDGET_MESH → 2f831d5 merge main 1143f58 → 817cff5 ND-shard / hybrid knobs → 0be73de results.

---

## 2. Research findings (read-only subagents)

### 2.1 The new torus MoE ops

| op / PR | state | what it does | reported gain |
|---|---|---|---|
| `deepseek_prefill.combine_fabric2d` ("combine v2"): #54163, #56340 (DRAM state leak), #56363 (dense forwarding buffer), #56614 (overlappable reader), #56726 (TILE input), eth-aware placement APIs #55535–#55537 | **merged** (09-01 … 09-17) | op-managed store-and-forward along one ring axis: DRAM → eth → next chip's DRAM, CW and CCW streams per link, farthest destination first | v2 at 51–75% of v1 on 8x4 torus (dsv3 56%, "minimax_m27" 65%, gpt-oss 64%) — synthetic bf16 inputs |
| `offset_cumsum` 4th output `all_global_dispatch_offsets` [H, E]: **#57859** | open | every device's dispatch offsets, replicated along the axis (what the relay needs) | offset_cumsum 2–3× faster (34.7 → 17.0 µs) |
| `deepseek_prefill.dispatch_fabric2d`: **#57654** (stacked on #57859) | open | same relay scheme for dispatch; drop-in output layout | 945 → 546 µs (uniform), 1447 → 936 µs (hot), 5K chunk, 8x4 |
| #57952 byte-exact combine tests | open | | |
| issue #57948 | open | back-to-back launches can overwrite a neighbour's forwarding buffer (wrong bytes) / combine counter-reset hang | |
| issue #57955 | open | combine stream-core fallback ignores NoC cost on harvested chips | |

Constraints (checked in the device ops): ring axis extent **even and ≥ 4**, **wrap-wired** (`BoundaryMode::WRAP`
neighbours, `is_axis_wrap_wired`), 2D torus fabric, `num_routed_experts % 16 == 0`, Ring/Torus topology (Linear
rejected), DRAM-interleaved tensors, no fp8, fabric max payload ≥ token + 64 B (12352 B for M3's 6144 emb). **No model's
forward path used either op before today** (DeepSeek/Kimi/GLM/gpt-oss all still run v1); only unit tests.

### 2.2 Physical torus topology of the galaxy (from the cluster descriptor)

- Full 8x4 with `single_bh_galaxy_torus_xy_graph_descriptor.textproto` + `FABRIC_2D_TORUS_XY`: 8-ring on axis 0 (via
  row 7↔0), 4-ring on axis 1. Opened fine on this galaxy (one topology match, degree 4 on every chip).
- **No row 3↔0 or row 7↔4 links.** A (4,4) carved half has no axis-0 wrap; the new ops then silently route the "wrap"
  stream the long way (timings meaningless). The runner log line "4x4_SplitHost_flat_torus_xy" only means a logical
  grouping matched.
- **Exactly one 4x4 torus per galaxy: the middle rows 2–5** (closed by a "2↔5 chord"), used by DeepSeek #48225 with
  `models/demos/deepseek_v3_d_p/experimental_descriptors/single_bh_galaxy_subtorus_xy4_graph_descriptor.textproto` and
  `TT_VISIBLE_DEVICES=2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31`.

### 2.3 M3's MoE before today

`tt/moe/tt_minimax_moe.py` reuses DeepSeek's `dispatch` (v1) → `unified_routed_expert_moe` → v1 `combine` →
`post_combine_reduce`, `cluster_axis=0`, **hard-coded `Topology.Linear`** (DeepSeek/Kimi use Ring on a torus). 128
experts, **top-4**, sigmoid scores; mesh = 4 dispatch groups (columns) × chips along axis 0; experts contiguous per chip.

### 2.4 Why combine v2 needed a typecast, and the fix

The routed-expert op (`unified_routed_expert_ffn`) emits **BFLOAT8_B TILE** (host wrapper hard-codes it; kernels would
pack any output format). Combine v2 **rejected bfp8** (not implemented). DeepSeek would hit the same on adoption (its
routed expert is bfp8 too; the v2 test feeds synthetic bf16). Options: (a) `output_dtype` on the routed-expert op
(~½ day, more L1, 2× output bytes) or (b) bfp8 input in combine v2 (~1–2 days, same numerics as v1, −47% DRAM reads).
**We did (b)** (§3.2).

### 2.5 ND-sharded expert weights and the hybrid low-token path

- **Op support: PR #56638** (a0a25d4, 09-17): `unified_routed_expert_ffn` / `moe_fused_swiglu` / `hybrid_routed_expert_ffn`
  accept DRAM ND-sharded gate/up/down weights (activations stay interleaved). Each core's N-slice of a K-row is one
  shard in one bank, rows rotate across banks → one NoC read per K-row instead of per tile (~370 GB/s vs ~30 GB/s for a
  core pinned to one bank). Shard width must equal `ceil(N_tiles/11)` (11x8 grid) → M3 pads gate/up N 96→99 tiles,
  down 192→198 (+~3% expert DRAM, ~15 MB/chip/MoE layer).
- **Model default: PR #57850** (d4d670a, merged 2026-09-28 13:06, after our first pull): `TtRoutedExpert(weights_dram_nd_sharded=None)`
  = ND-sharded on Blackhole for everything built through DeepSeek's MoE. Weights load host-side then `to_device`, so
  **weight caches are reused**. Reported: routed-expert op 1.20–1.22× (DSv3 2x4 / 8x1), prefill block 1.16×, Kimi MoE
  −5…−6%; gain concentrated at 128–512 tokens per expert.
- **Hybrid (low-token) path:** with a token threshold, experts with ≤ T tokens run **`moe_fused_swiglu`** (the fused
  low-token kernel), the rest `unified_routed_expert_moe`. M3's config already records the measured crossover
  (`ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD_MEASURED = 128`) but M3 did not pass it. The fused op supports M3's SwiGluOai.
- **No matmul kernel was modified today.** We used `moe_fused_swiglu` as-is through the hybrid threshold.

### 2.6 EP load-balancing support

None for prefill: `expert_dispatch_table` cannot express replicas; offset_cumsum and combine v1 assume contiguous
placement; replicated experts are an open TODO (issue #41293); no capacity factor (M3 sizes the dispatch buffer drop-free
and asserts it). A static relabeling is possible without op changes (permute gate rows + correction bias + weights).

### 2.7 Ring-CCL router crash (root cause)

`M3_CCL_TOPOLOGY=ring` at W=8192 on 8x4 (1024 rows/chip) failed in `moe_grouped_topk` (`scores.dtype() == FLOAT32 ||
BFLOAT16`). **M3 bug, not an op bug:** under Ring, attention's fused o_proj + reduce-scatter path (only tuned for
M_tiles=32 = 1024 rows) returns bfloat8_b; nothing cast it back, so the residual, norm and router gate ran in bf8.

---

## 3. Code changes (all default-off unless noted)

### 3.1 M3 knobs (`models/demos/minimax_m3/utils/fabric_env.py`, plumbed through `tt/mlp.py` → `tt/moe/tt_minimax_moe.py`)

| env | values | effect |
|---|---|---|
| `M3_MOE_TOPOLOGY` | linear (default) \| ring | topology of axis-0 dispatch and v1 combine |
| `M3_MOE_COMBINE` | v1 (default) \| v2 | v2 = `combine_fabric2d` with the replicated offsets (`all_global_dispatch_offsets` from #57859), Ring, 2 links; fabric opened with a larger max payload (`get_max_payload_size()`, 14400 B) |
| `M3_MOE_COMBINE_V2_CAST` | 0 (default) \| 1 | 1 restores the explicit bf16 typecast before combine v2 (A/B only) |
| `M3_MOE_DISPATCH` | v1 (default) \| v2 | v2 = `dispatch_fabric2d`; fails before weight load on a non-torus fabric, too-small payload, or odd / < 4 axis extent |
| `M3_MOE_W_NDSHARD` | 0 (default = interleaved, as all earlier runs) \| 1 | passed explicitly to `TtRoutedExpert` (cancels #57850's new default) |
| `M3_MOE_HYBRID_THRESHOLD` | 0 (default off) \| T | experts with ≤ T tokens → `moe_fused_swiglu` |
| `M3_MOE_LOAD_STATS` (+ `_FILE`) | 0 \| 1 | per-layer per-chip / per-column expert load (host readback; measurement only) |

Harnesses open the fabric through `set_fabric_config_from_env()`. Timing harness (`tests/perf/budget_sweep.py`):
`BUDGET_MESH=RxC` opens a mesh directly (sub-torus runs), `BUDGET_ANY_LAYERS`, `BUDGET_MEM`. `m3_budget_study/run_budget.sh`
now resets with `env -u TT_VISIBLE_DEVICES` (a leaked visible-device list reset only 16 chips).

### 3.2 C++: `combine_fabric2d` accepts BFLOAT8_B TILE input (commit c85e20a)

Only the DRAM tile read and the untilizer input CB use the input format; the unpacker dequantises bfp8 → bf16 (same
setup as generic `ttnn.untilize`); tokens stay bf16 in the untilized rows, ring, forwarding buffer and output.
`token_size_bytes` = emb × 2 always; `tile_size_bytes` from `tile().get_tile_size(format)` (1088 B bfp8); `UNT_CB_IN`
takes the buffer's format; bfp8 ROW_MAJOR rejected with a clear message. Untilizer input CB 32 → 17 KB. Test: new
`tile_bfp8` (exact match) and `tile` bf16 cases in `test_prefill_combine.py::test_ttnn_combine_fabric2d`.

### 3.3 Bug fix: Ring o_proj dtype

`tt/attention/prefill.py`: cast the fused-RS output back to the activation dtype (`_as_dtype`); router gate
`ttnn.linear` pins bf16 (`tt/topk.py`). Verified: Ring CCL runs at W=8192 (config R below).

---

## 4. Validation of the new ops on this galaxy

| test | result |
|---|---|
| `test_dispatch_fabric2d.py -k fabric2d-torus-xy-8x4-2link` | **16/16 passed** (381 s), incl. back_to_back ("4 unsynchronised launches byte-exact"), relaunch emb 2880/7168 — no #57948 symptoms |
| `test_dispatch_fabric2d_perf.py` (DeepSeek geometry: emb 7168, 256 experts, top-8, 640 tok/chip) | passed: 10,323,986 ns / 11 launches = **938.5 µs per launch** (−0.58% vs expected) |
| `test_prefill_combine.py::test_ttnn_combine_fabric2d` torus-xy-8x4 (before bf8 change) | 2/2 passed (row_major pcc, perf_no_pcc) |
| same after the bf8 change | **6/6 passed** (row_major, tile, tile_bfp8 × pcc, perf_no_pcc) |

No in-tree 8x4 perf test exists for v1 `dispatch` (only an 8x1 LoudBox one), so the before/after comparison comes from
M3's own zone profiles (§6).

---

## 5. Phase 1 — full 8x4, torus / Ring / combine v2 with the typecast (branch `m3_moe_torus` @ 59a5723)

Configs: A = 1d fabric, Linear, v1 (today). B = 2d torus, Linear, v1. C = torus, MoE Ring + CCL Ring, v1. D = C +
combine v2 (with bf16 typecast). Cm/Cc/Dm isolate the knobs. Tokens: old DeepSeek-id trace. Layer set S8 = 8..15, cold.

| cfg | W=8192 | W=16384 |
|---|---|---|
| A | 106.25 | 196.66 |
| B | 115.65 | 214.82 |
| C | ERROR (router bug) | 207.93 |
| D | ERROR | 216.09 |
| Cm (MoE Ring only) | 115.78 | 214.85 |
| Dm (Cm + v2) | 120.21 | 227.81 |

Zones (layer 3, W=8192, worst chip, ms): A dispatch 2.58, experts_mm 2.92, combine 5.82, moe_reduce 7.46, mlp 10.28,
layer 14.51. Dm combine 6.44 **of which combine_v2_prep (typecast + offsets gather) 4.89**; the `combine_fabric2d` op
itself ≈ 1.5 ms vs 5.8–6.5 ms for v1. Host gap A 37%, Dm 55–57%. 73 ops/sparse layer (75 with v2). PCC A vs D passes.

Take-aways: torus fabric alone +9% (dispatch disables sparse multicast on non-1D fabric); Ring for MoE alone = no
effect; v2 with the typecast is a net loss; 8x4 wall is not host-launch bound in steady state.

---

## 6. Phase 2 — both new ops, bf8 combine, Ring fixed (branch `m3_moe_fabric2d` @ 191eeb2)

Configs: A as above. R = torus + CCL Ring + MoE Ring, v1/v1. E1 = R + combine v2 (bf8 in). E1c = E1 with the old
typecast. E2 = R + dispatch v2. **E3 = R + dispatch v2 + combine v2.** S8, cold.

| cfg | W=8192 | W=16384 |
|---|---|---|
| A | 106.20 | 196.60 |
| R | 110.56 | 206.09 |
| E1 | 97.17 | 184.04 |
| E1c | 112.77 | – |
| E2 | 97.97 | 181.06 |
| **E3** | **91.04 (−14.3%)** | **171.25 (−12.9%)** |

Zones, sparse layer 3, W=8192, worst chip, device ms:

| zone | A | R | E1 | E2 | E3 |
|---|---|---|---|---|---|
| dispatch | 2.59 | 2.87 | 2.77 | **1.59** | **1.59** |
| experts_mm | 2.93 | 2.94 | 2.94 | 2.93 | 2.93 |
| combine | 5.87 | 6.53 | **4.38** | 5.28 | **4.13** |
| moe_reduce | 7.51 | 8.39 | 4.89 | 4.48 | 3.92 |
| mlp | 10.34 | 10.49 | 8.53 | 8.26 | 7.21 |
| layer | 14.53 | 14.99 | 13.06 | 12.76 | **11.64** |

Host gap (profiled single-layer chunk): A 38%, R 40%, E1 67%, E2 47%, E3 66% (combine v2 roughly doubles op-to-op gaps
in that profile; the multi-layer sweeps are still faster).

KV PCC (6 layers, longbook_5120; K / V / index_k): A L3 .99972/.99925/.99975, L4 .99900/.99747/.99927, L5
.99902/.99579/.99933; E3 L3 identical, L4 .99895/.99736/.99924, L5 .99894/.99542/.99929.

---

## 7. SP=8 cost model and 4-galaxy projection (whole 8x4, config E3, M3-tokenized `longbook_56320`)

27 device processes, all OK. Layer sets D8 = 0..2, S8 = 8..15, S5 = 3..7, ST0 = 0..14, ST1 = 15..29.

Cold width (ms): D8 13.5 / 13.9 / 14.8 / 22.1 and S8 69.7 / 72.7 / 91.0 / 172.0 at W = 2048 / 4096 / 8192 / 16384;
S5 46.5 / 59.7 at 4096 / 8192.

Depth, W=4096 (n = 4096 / 512):

| h | 0 | 16k | 65k | 139k | 311k | 549k |
|---|---|---|---|---|---|---|
| D8 | 14.0/14.3 | 15.4/15.2 | 21.7/21.6 | 35.7/35.7 | 71.9/73.9 | 113.6/118.9 |
| S8 | 73.3/73.8 | 74.4/72.8 | 74.3/73.1 | 74.9/73.6 | 75.0/74.1 | 75.2/76.0 |

Wide depth (h = 0 / ~140k / ~545k): D8 W=8192 15.2 / 50.5 / 164.6; D8 W=16384 22.8 / 104.9 / 320.1; S8 W=16384
171.4 / 164.3 / 178.3; S8 W=8192 "fast" 91.0 / 88.8 / 102.1 vs "slow mode" 91.0 / 129.6 / 138.5.

**W=8192 "slow mode":** if a process's first deep forward (a ~0.35 s compile) is followed straight away by a deeper
chunk, the rest of the run stays ~5 ms per sparse layer slower (2/2 reproductions); adding a warm point at 8192 avoids it
(2/2). Not seen at 4096/16384 or in ST0; not KV capacity. Root cause unknown. A similar one-off happened on the SP=4
sub-torus at W=4096 (§9).

Packing (2048-token segments = 256 rows/chip): sparse 2 segments 115.6 vs plain 4096 73.0 ms; 4 segments 194.5 vs
plain 8192 91.0; dense 21.7 vs 13.9 and 37.3 vs 15.0 → **seg_a = 4.6 ms per extra segment per sparse layer** (SP=4 0.33,
SP=2 ≈ 0). Full stages at W=8192 (h = 0 / 139k / 549k): ST0 137.2 / 167.6 / 290.2; ST1 158.6 / 159.1 / 183.2 (fast);
DRAM in use after the deepest point **5.37 GB of 34.2 GB per chip** for 15 layers.

Fits (`sp8/coeffs_sp8*.json`): as-specified dense a 2.79, c 1.12e-8, **p0 5632** (R² .9991); sparse a 5.84, b 6.2e-4,
d 8.1e-7, e 1.75e-4 (R² .933); fast-mode W 4–8k fit sparse R² .924, residuals ≤ 5%. For reference SP=4 (`coeffs.json`)
dense p0 2944, sparse a 2.02; SP=2 (`coeffs_sp2.json`) dense p0 2304, sparse a 1.96.

Simulator (agentic mix, n=4000, align-recompute, embed on stage 0 only, split search, cost budget from a 1.5 s
hot-latency target), best configuration each:

| layout (4 galaxies) | tok/s | fwd p50/p99 ms | hot p50/p99 ms | split |
|---|---|---|---|---|
| **L-b: 4 × 4-stage 2x4 (SP=2)** | **38.5–39.0k** (fcfs 41.0k) | 279/383 | 1428/1947 | 9,17,17,17 |
| L-c: 16-stage SP=2 | 32.5k | 55/102 | 1097/1910 | 1,1,1,5×5,4×8 |
| L-d: 4 × 2-stage (4,4) SP=4 | 27.1k | 447/497 | 1348/**1499** | 27,33 |
| L-a: 8-stage SP=4 (old plan) | 24.9k | 154/199 | 1447/1848 | 3,9,8×6 |
| L-e: 4-stage SP=8 pipeline, hop 15 ms (30 ms ≈ same) | 12.3–13.0k | 243/290 | 1258/**1494** | 9,17,17,17 |
| L-f: 4 × 1-stage SP=8 | 11.4–11.8k | 483/684 | 966/**1367** | 60 |
| L-e / L-f with seg_a = 0 (free packing) | 26.7k / 23.7k | | ≈ 1.5 s | |

SP=8 loses throughput because packing costs 4.6 ms/segment/layer and a sparse layer has a ~9 ms floor; it wins
latency. (L-a…L-d used the older SP=4/SP=2 coefficients with v1 ops and the DeepSeek-id trace.)

---

## 8. SP=2 and SP=4 feasibility (read-only analysis)

- SP=2: both ops reject extent 2; an axis of 2 is never "wrap-wired" (`is_genuine_torus_dim(n) = n > 2`). At extent 2
  v1 is already a direct one-hop send (and on 1D fabric it merges one token's pages for a chip into one sparse
  multicast). Moving dispatch onto the 4-wide TP axis needs all 128 experts per row (+8 GB/chip, doubles weight reads)
  or extra all-gather/reduce-scatter pairs: 1–2 weeks, likely a net loss.
- SP=2 MoE breakdown (sparse layer 3, W=2048, 1024 tok/chip; old trace, 1d, v1) vs SP=4 and SP=8, worst chip (mean):

| zone | SP=2 | SP=4 (512 tok/chip) | SP=8 A | SP=8 E3 |
|---|---|---|---|---|
| dispatch | 0.55 (0.45) | 0.44 | 2.59 | 1.59 |
| experts_mm | 2.45 (1.92) | 1.41 (0.95) | 2.93 (0.62) | 2.93 (0.61) |
| combine | 0.89 (min 0.17) | 1.03 | 5.87 | 4.13 |
| moe_reduce | 1.77 (min 0.43) | 1.31 | 7.51 | 3.92 |
| mlp | 4.89 | 3.26 | 10.34 | 7.21 |
| µs per token per chip (mlp) | **4.78** | 6.36 | 10.1 | 7.04 |

  SP=2 average-chip MLP time: experts_mm 41%, moe_reduce 22%, shared expert 11%, dispatch 10%, combine 8%. experts_mm is
  weight-read bound (16 experts/chip ≈ 475 MB/layer ≈ 250 GB/s). Skew mild at EP=8 (experts_mm max/mean 1.28).
- SP=4: the only 4x4 torus is the middle rows. A mixed carve (one torus 4x4 + two edge 2x4 stages) has a template
  (`tests/tt_metal/tt_fabric/custom_mesh_descriptors/bh_galaxy_split_4x4_2x4_3_mesh.textproto`, LINE-only today), but
  the prefill runner uses one global SP for all stages, so mixed SP needs resharding at the stage boundary.

---

## 9. ND-sharded weights, hybrid path, SP=4 sub-torus, SP=8 ND A/B (branch @ 817cff5, M3-tokenized trace)

### 9.1 SP=2 (2,4), S8' (8..15), 1d, v1 — wall ms

| cfg | W=2048: h = 0 / 16k / 141k / 549k | W=4096: h = 0 / 139k / 549k |
|---|---|---|
| N0 interleaved | 70.22 / 68.12 / 76.69 / 92.91 | 130.21 / 129.12 / 146.50 |
| N1 ND-sharded | 68.19 / 66.08 / 74.61 / 90.20 | 128.00 / 126.70 / 142.38 |
| **N1H ND + hybrid 128** | **66.76 / 64.90 / 73.08 / 88.21** | **126.61 / 125.48 / 141.42** |

Dense sanity (D2, W=4096 cold): 16.94 vs 17.37 ms (noise).

Zones, sparse layer 3, W=2048, device ms (worst chip / mean of 8):

| zone | N0 | N1 | N1H |
|---|---|---|---|
| experts_mm | 2.390 / 1.578 | 2.078 / 1.348 | **1.807 / 1.176** (2 ops: MoeFusedSwiGlu + UnifiedRoutedExpertMoe) |
| dispatch | 0.619 / 0.474 | 0.606 / 0.474 | 0.606 / 0.477 |
| combine | 1.304 / 0.650 | 1.222 / 0.623 | 1.191 / 0.527 |
| moe_reduce | 1.892 / 1.208 | 1.737 / 1.130 | 1.544 / 1.010 |
| mlp | 5.918 / 5.251 | 5.030 / 4.644 | 4.625 / 4.216 |
| layer | 8.731 / 8.671 | 8.470 / 8.081 | 8.185 / 7.655 |

Expert matmul (worst chip): ND −13%, ND + hybrid **−24%**; layer −3.0% / −6.3%; wall −4.7…−5.1% (W=2048), −2.8…−3.5%
(W=4096), ≈ 0.45 ms per sparse layer.

### 9.2 SP=4 on the middle 4x4 torus (all ND-sharded) — wall ms

M4A = carved (4,4) rows 0–3, 1d, v1. M4R = middle torus, Ring, v1. M4E3 = M4R + dispatch v2 + combine v2.

| cfg | W=4096: h = 4k / 139k / 549k | W=8192: h = 8k / 139k / 549k |
|---|---|---|
| M4A | 72.06 / 83.34 / 106.88 | 131.40 / 144.86 / 170.35 |
| M4R | 74.04 / 82.22 / 95.96 | 136.36 / 146.46 / 162.41 |
| **M4E3** | **64.35 / 72.27 / 83.42** | **120.37 / 130.35 / 144.11** |

(First cold h=0 points: W=4096 80.30 / 82.52 / 72.63, W=8192 146.81 / 149.49 / 133.84. The first M4E3 W=4096 run fell
into a slow mode after h=0 — 93.56 / 107.02 ms at 139k / 549k — the repeat with a 4096 warm point stayed fast.)
Torus fabric cost (M4R vs M4A) +2.7…+3.8% shallow, Ring helps at depth (−10% at 549k W=4096). **New ops net of the
torus: −11…−13% at every point.** Total vs M4A: −11…−22% (W=4096), −8…−15% (W=8192). The sub-torus opened with the
default reliability mode; M3's hard-coded Linear TP collectives did not hang.

### 9.3 SP=8, E3, W=4096 — ND A/B (two runs each)

ND=0: 73.03 / 74.85 (h=0), 75.47 / 76.66 (549k). ND=1: 73.90 / 73.41, 77.81 / 75.32. **No effect** (±1.5% noise).

### 9.4 Chip efficiency, sparse (chip-µs per token-layer = wall × chips × 1000 / (W × layers))

| layout | h ≈ 0 | ~141k | 549k |
|---|---|---|---|
| SP=2 (2,4) N1H, W=4096 | 30.9 | 30.6 | 34.5 |
| SP=2 N0, W=4096 | 31.8 | 31.5 | 35.8 |
| **SP=4 4x4 torus M4E3, W=8192** | **29.4** (h=8k) | 31.8 | 35.2 |
| SP=4 carved M4A, W=8192 | 32.1 | 35.4 | 41.6 |
| SP=8 E3, W=8192 (15-layer runs, ND=0) | ~38 | ~41 | ~48 |
| SP=8 E3, W=4096 | ~72 | – | ~74 |

---

## 10. Expert-load (EP) skew — measurement and correction

Phase 2 measured per-chip load with `M3_MOE_LOAD_STATS` on isolated layer set 8..15 and reported one hot expert per
layer, identical on prose and two code prompts (L12 expert 109 in 98.7% of tokens), ~79% of the worst chip's expert
time being imbalance, and that a perfect static relabeling (LPT packing) cuts the per-layer max only 5–13%.

**Verification (whole 8x4, 1d, v1, W=8192):** ISO = layers 8..15 only (the harness embeds tokens itself, so layer 8's
router sees raw embeddings); CONT = layers 0..15 (layers 8–15 see real hidden states). Hot expert : share of tokens,
per-chip max/mean (mean = 1024 pairs):

| layer | ISO | CONT prose (longbook_56320) | CONT code (M3 .py files) |
|---|---|---|---|
| 8 | 126 : 78%, 6.25× | 108 : 68%, 5.70× | 119 : 70%, 5.98× |
| 9 | 31 : 70%, 6.19× | 40 : 58%, 5.30× | 73 : 33%, 2.69× |
| 10 | 1 : 53%, 4.89× | 113 : 71%, 6.77× | 112 : 35%, 3.13× |
| 11 | 96 : 87%, 7.00× | 71 : 59%, 5.22× | 63 : 26%, 2.53× |
| 12 | **109 : 98.7%, 7.91×** | 25 : 46%, 4.80× | 33 : 19.5%, 2.78× |
| 13 | 9 : 75%, 6.30× | 24 : 20%, 2.92× | 67 : 44%, 3.81× |
| 14 | 42 : 95%, 8.60× | 46 : 68%, 6.14× | 95 : 54%, 4.83× |
| 15 | 112 : 74%, 5.94× | 15 : 28%, 2.50× | 44 : 61%, 5.06× |

CONT prose L3–7: 43%, 91% (L4 expert 78), 68%, 21%, 37%; code L3–7: 13%, 49%, 28%, 80% (L6 expert 71), 23%. Experts
with zero tokens: ISO 26–73 per layer, CONT 4–12, code 0–5.

- **The fixed hot IDs (109/9/42/112) are an artifact of feeding embeddings into layer 8** (ISO reproduces them exactly,
  even with a different trace and fabric). With real inputs the hot expert differs by prompt in every layer.
- Single-document skew is real and large: one expert in 20–90% of tokens; busiest chip 2.5–7.4× the mean at EP=32.
  The correction bias is not the cause (tiny spread; hot experts mid-ranking). Likely domain specialisation within one
  document; packing different requests should spread it. TT router not yet checked against a CPU reference (KV PCC
  through the MoE layers stays ≥ 0.995).
- **Consequence:** every MoE cost measured on an isolated layer set (all stage-level studies) saw artificially extreme,
  fixed routing → **pessimistic MoE time, most at EP=32 (SP=8)**. Layout comparisons were like-for-like; full-model
  runner results (only stage 0 embeds) are unaffected. Static replication of "the" hot expert does not apply (it moves
  with content).

---

## 11. Bugs, anomalies and pitfalls found

- **Ring o_proj bf8 → router crash** (M3 bug, fixed, §2.7 / §3.3).
- **Combine v2 needed bf16** (fixed with bf8 support, §3.2); DeepSeek will need the same when it adopts v2.
- **Default golden trace `longbook_qa_eng_prefill_56320_nopad` holds DeepSeek-R1 token ids** (junk under M3's tokenizer).
  All studies before today used it; today's §7, §9, §10 use the M3-tokenized `longbook_56320`.
- **Isolated layer sets embed tokens into the first built layer** (§10) — measure MoE/routing with contiguous layers
  from 0.
- **W=8192 slow-mode warm-up anomaly** (§7), also once at W=4096 on the sub-torus (§9.2). Workaround: add a warm point.
- **`TT_VISIBLE_DEVICES` leaks into `tt-smi -glx_reset`** (resets only the visible chips; init then fails). Fixed in
  `run_budget.sh` (`env -u TT_VISIBLE_DEVICES`).
- The torus fabric logs 8 non-fatal "4 eth channels, but only 2 routing planes" warnings.
- A carved (4,4) half is not a torus; the new ops route the long way silently there.
- `TtRoutedExpert`'s default changed with #57850 (ND-sharded on BH); `M3_MOE_W_NDSHARD=0` keeps comparability.

---

## 12. Open questions and next steps

1. Realistic skew: contiguous layers from 0, packed forwards mixing several prompts, EP=8 and EP=32; CPU-reference check
   of the router on 1–2 layers.
2. Re-measure stage costs with contiguous layer ranges (e.g. 0–15) so the cost model's MoE terms reflect real routing;
   re-run the 4-galaxy simulator for L-b with the ND + hybrid gain (expect ≈ +5%).
3. Make packing cheap on SP=8 (batched per-segment attention) — the one change that could make SP=8 competitive.
4. Root-cause the warm-up "slow mode".
5. Upstream candidates: bf8 input for `combine_fabric2d` (c85e20a), M3 Ring o_proj dtype fix, M3 adoption of
   dispatch/combine v2 knobs once #57859/#57654 merge.

## 13. Where things are

- Summaries: `results_torus/SUMMARY_phase1.md`, `SUMMARY_phase2.md`, `SUMMARY_nd_sp4.md`; SP=8: `results_torus/sp8/`
  (`coeffs_sp8*.json`, `fit_sp8*.txt`, `sim/`); skew: `results_torus/skew_check/{iso,cont,cont_code}.jsonl`.
- Earlier studies: `m3_budget_study/results/REPORT.md` (SP=4 budget study), `results_sp2/REPORT_SP2.md` (SP=2 vs SP=4).
- Raw per-run rows: `results_torus/runs.csv` (prefixes `t1_`, `t2_`, `s8_`, `nd_`, `skew_`), logs and `.env` files in
  `results_torus/logs/`, zone captures in `results_torus/profiles/` and `results_torus/nd/profiles/`.
