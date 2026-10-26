# GLM-5.3-Flash decode roofline on the 8x4 Blackhole Galaxy (2026-10-26, code and analysis only, no device runs)

Per-op, per-layer and per-step roofline of the traced decode step at 4 users per row (32 users) and 32 users per row
(256 users), TP4 over the 4 columns x DP8 over the 8 rows, hybrid routed experts (72 experts per column, the 2048-wide
intermediate split over the 8 rows = 256 per chip) with the EP32 layout as the alternative. Every op gets its bytes,
FLOPs, link bytes, time floor under the stated peaks, the measured device time (which profile), the gap ratio, a bound
class and an action. Model: `models/demos/glm53_flash/tt/roofline.py` (host-only; `tests/test_roofline_host.py`, 19
checks). Regenerate any table with

    python -m models.demos.glm53_flash.tt.roofline --users-per-row 4 --all-layers --markdown
    python -m models.demos.glm53_flash.tt.roofline --users-per-row 32 --moe ep --json

## 1. Headline

| point | step floor (400 GB/s achievable / 512 spec) | tok/s per user ceiling | e2e measured | op-model sum | ratio | target | after the ranked actions (model) |
|---|---|---|---|---|---|---|---|
| 32 users (4/row), hybrid | **32.3 ms / 26.4 ms** | **31.0 / 37.9** (992 aggregate) | 80.55 ms = 12.4 tok/s/u (2026-10-26) | 82.6 ms | 2.5x | 30.3 ms (33 tok/s/u) | 30.7 ms = 32.6 tok/s/u |
| 32 users, EP32 | 40.2 / 31.8 | 24.9 / 31.4 | 67.68 ms rs-remap (71.2 op combine) | 73.1 ms (op combine) | 1.8x | 30.3 ms | 37.2 ms |
| 256 users (32/row), hybrid (dense experts) | **55.7 ms / 43.7 ms** | **18.0 / 22.9** (4600 aggregate) | 115.0 ms = 8.7 tok/s/u (2026-10-21) | 123.5 ms | 2.1x | 52.6 ms (19 tok/s/u) | 62.3 ms = 16.1 tok/s/u |
| 256 users, EP32 rs combine | 60.9 / 46.7 | 16.4 / 21.4 | 123.4 ms (2026-10-22) | 127.2 ms | 2.0x | 52.6 ms | 61.6 ms |

* The **32-user target (30.3 ms) sits 2 ms below the current op graph's floor at the achievable DRAM rate**: tuning alone
  cannot reach it; the floor drops to 28.3 ms with the fused mHC pre-block (A1), 29.2 with bfp4 gate|up experts, 25.2 ms
  (39.7 tok/s/u) with A1 + B5 + bfp4. The op model after the ranked actions lands at 30.7 ms.
* The **256-user target (52.6 ms) is below the floor at 400 GB/s (55.7) and above it at the spec peaks (43.7)**: it
  needs design changes (bfp4 gate|up: 49.3 ms; A1: 51.7; A1 + B5 + bfp4: 45.3 ms = 22.1 tok/s/u), and the action list
  as it stands (62.3 ms) is not enough: the remaining terms are the per-user state traffic (kda_decode 6.0 ms, DSA pool
  gather + SDPA 6.0 ms) and the row collectives running at 2x the CCL fit (5.8 ms).
* At 4 users/row the hybrid layout has the lower floor (28 active slices = 94 MB per chip vs EP32's slowest chip reading
  6 full experts = 160 MB) but is measured slower today (80.6 vs 67.7 ms) because its T=4 path pays the sparse block op
  (683 us instead of the stream op's 341) and ~280 us of padded gather / reduce-scatter glue per layer. E0 + H1 + H2 (below)
  invert that.
* Floor composition per step (ms, hybrid, current op graph): routed experts 10.3 / 20.8 (32 / 256 users), projection
  matmuls 7.0 / 7.0, mHC 5.2 / 5.2, layout glue + launches + gaps 3.6 / 3.1, row collectives 2.5 / 5.8, TP all-reduces
  2.4 / 2.4, DSA state 1.1 / 6.0, kda_decode state 0.8 / 6.0, LM head 0.6 / 0.6.

## 2. Assumptions (every chip number with its source; `roofline.CHIP`)

| constant | value | source |
|---|---|---|
| `dram_gbs_spec` | 512.0 | tt-metal tech_reports/CCLs/CCL_bandwidth.md: Blackhole 8 GDDR6 channels x ~64 GB/s = ~512 GB/s (device spec; tt-perf-report's 'DRAM %' denominator; a per-bank measurement of 64 GB/s agrees) |
| `dram_gbs` | 400.0 | achievable: the best decode-shaped matmuls of the 2026-10-21 profiles reach 393-412 GB/s (kda qkv 393-395, DSA stem 405, indexer query 412: report_prof_ep32_ep_b4 / report_prof2_l3_b4_stream); 1D mcast configs 200-330 GB/s (tt/matmul_configs.py) |
| `expert_stream_gbs` | 326.0 | doc/mesh/expert_kernel_sweep.md: expert_stream op 69 active = 231 MB in 707 us = 326 GB/s (64 %), 28 active = 93.6 MB in 341 us = 274 GB/s (54 %): ~110 us of fixed per-block cost |
| `kda_state_gbs` | 250.0 | decode ledger section 5: kda_decode moves the fp32 state twice at ~250 GB/s (64 MB in 260 us at 32 users/row) |
| `clock_ghz` | 1.35 | tech_reports/GEMM_FLOPS/GEMM_FLOPS.md: Blackhole 1.35 GHz |
| `worker_cores` | 120 | tt-perf-report header of the profiles: 'Architecture: blackhole, Worker cores: 120' (13x10 grid, 120 usable on these P150 Galaxy chips; GEMM_FLOPS.md quotes 13x10 = 130 for a P150 card) |
| `tflops_per_core` | LoFi: 5.4, HiFi2: 2.7, HiFi3: 1.8, HiFi4: 1.35 | GEMM_FLOPS.md: 2*8*16*16 muladds per cycle at 1.35 GHz = 5.4 TFLOPS per matrix engine at LoFi, / fidelity (HiFi2 2.7, HiFi4 1.35); tt-perf-report uses the same per-core peaks for its 'FLOPs %' |
| `elementwise_tflops_per_core` | 0.1728 | tech_reports/matrix_engine/matrix_engine.md: elementwise 8*16 ops per cycle = 0.128 TFLOPS per core at 1 GHz (x1.35 for Blackhole); fp32 elementwise goes through the SFPU and is slower: lower bound only |
| `eth_link_gbs_rated` | 50.0 | CCL_bandwidth.md: Blackhole 400 Gbps = 50 GB/s per link per direction, ~48 payload |
| `eth_link_gbs` | 25.0 | CCL_bandwidth.md: 'Blackhole Galaxy links currently run at half their rated speed' (25 GB/s per link per direction); doc/mesh/README.md ep32 analysis: 2 x 24.2 GB/s on the row axis |
| `links_per_axis` | 2 | models/common/modules/tt_ccl.py get_num_links: BHGLX (2, 2) links per axis |
| `ccl_fit` | all_gather: (3.7, 1.3, 48.0, 'N-1'), reduce_scatter: (1.3, 3.0, 40.0, '(N-1)/N'), all_reduce: (4.4, 4.4, 44.0, '2(N-1)/N'), all_to_all: (1.8, 1.4, 33.0, 'floor(N/2)ceil(N/2)/N') | CCL_bandwidth.md 'Kernel time' fits for Blackhole LINE topologies (the measured Galaxy at 25 GB/s per link; the fits' link count is the machine's 2 per axis, so this module scales their GB/s by link_gbs/25 x links/2); ring fits exist only for N = 8 (all_gather 95 GB/s, reduce_scatter 82, all_reduce 86, all_to_all 68) |
| `hop_latency_us` | 1d: 0.515, 2d: 0.619 | CCL_bandwidth.md per-hop forwarding latency, Blackhole |
| `op_floor_us` | 1.0 | smallest device ops in the profiles: Slice 0.84 us, Typecast 1.1 us (report_prof2_l3_b4_stream); one program launch inside a trace cannot go below ~1 us |
| `op_gap_us` | 0.46 | decode ledger section 3: median in-trace op-to-op gap (0.46 us; 1.0-1.5 after single-core ops) |
| `noc_gbs_per_core` | 86.0 | ASSUMED (not measured in this tree): 64 B/cycle x 1.35 GHz per NoC port; only used to bound L1-resident elementwise work. Measured small-access rates: ~14 GB/s per core with 1 KB tile reads (expert_kernel_sweep.md) |

Ranges: floors are given at the achievable DRAM rate (400 GB/s: the best matmuls of these profiles; this sets the ratio
and the bound class) and at the 512 GB/s spec; CCL floors at the Galaxy's measured 25 GB/s per link (and at the 50 GB/s
rated rate in the `floor_spec` column). The NoC figure is an architecture assumption, not measured in this tree, and is
only used as a lower bound on L1-resident elementwise work. Byte costs: bf16 2 B, fp32 4 B, bfp8 1.0625 B, bfp4 0.5625 B
per element.

Inputs: HF `config.json` of the cached checkpoint (45 layers = 34 KDA + 11 DSA, layers 0-2 dense MLP, 288 experts top-8,
moe_intermediate 2048, hc_mult 4, kv_lora 512, q_lora 1536, index 32 heads x 128, top-2048 of 4-token pools);
`tt/{decoder_layer,kda,dsa,indexer,mhc,moe,mesh,matmul_configs,memory_plan,model_head}.py` and `custom_ops/*` for the
per-chip shapes and dtypes; the decode ledger (`_agents/decode_ledger/LEDGER.md`, `ops_per_layer.csv`); the tracy
profiles (chip (0,0) of one traced replay): `report_prof_ep32_ep_b4` (8x4, EP32 layout, 4 users/row, layer 4, 1.569
ms), `report_prof_ep32_ep_b32` (8x4, EP32, 32 users/row, 2.221 ms), `report_prof_ep32_hybrid_b32_dense` (8x4, hybrid
dense experts, 32 users/row, 2.444 ms), all 2026-10-21 18:29-18:36 on the wt_mesh tree at 531f31d8d79;
`report_prof2_l4_b4_stream` / `report_prof2_l3_b4_stream` ((2,4) block rows 4-5, hybrid stream op, 4 users/row, layers
4 / 3, 2.016 / 2.398 ms, 2026-10-21 13:39-13:41, tree ecc2ef47d27; the DSA attention rows and the hybrid T=4 MoE glue
come from there: same per-chip TP4 shapes, 4x wider expert slices); `doc/mesh/expert_kernel_sweep.md` (expert op
measurements at the 8x4 slice shape: 341 us at 28 active, 707 us at 69 active, dispatch form 779 us at 58 active); the
e2e steps of `doc/mesh/README.md` (2026-10-21/22/26). No profile exists of the 8x4 hybrid at 4 users/row, of any DSA
layer at 32 users/row, of a dense layer or of the head: those rows are projections marked `proj` in the source column.

Per-chip shapes (T users per row, every activation tile-padded to Tp = 32 rows): mHC streams [32, 4 x 4096] fp32 = 2 MB
replicated on every chip; KDA 16 of 64 heads (qkv [4096, 6144] bfp8 26.7 MB, o_proj [2048, 4096] 8.9 MB, fp32 state
[T, 16, 128, 128] = 1 MB per user read and written); DSA 16 of 64 heads (stem [4096, 2336] bf16 19.1 MB, q_b [1536, 4096]
bfp8, indexer query [1536, 4096] bf16 12.6 MB, o_proj [4096, 4096] bfp8 17.8 MB, 2051 selected latent rows x 512 bf16 =
2.1 MB per user gathered into the compact cache and read by the flash-MLA decode); hybrid experts: 72 slices of 3.34 MB
per chip per layer (240 MB; EP32: 9 full experts of 26.7 MB = the same 240 MB), active slices 28 at 32 gathered tokens
and 58 at 256 (layer-4 real routing; the uniform-routing Monte-Carlo gives 45 / 72 for the worst column); shared expert
[4096, 1024] + [512, 4096] bfp8; LM head [4096, 38720] bfp8 168.5 MB per column. Collectives: TP all-reduce of [32, 4096]
bf16 over the 4 columns (2 links); row all-gather of x and the routing + reduce-scatter of the [256, 4096] bf16 expert
partials over the 8 rows (2 links, line).

## 3. Gap table per op (32 users / 256 users per chip; bound = the floor class, action = knob / fuse / tune)

Columns: `n` = device ops in the row; `MB` = bytes read + written per chip; `measured` = chip-0 device time of the profile
in parentheses is the 32-user source (`proj` = projection); the 256-user values come from the matching `ep_b32` / `hyb_b32` ids or are projections (see `roofline.py` for each row); `floor` at 400 GB/s / the CCL fit / the compute peak / 1 us per launch, whichever is
largest; `ratio` = measured / floor; `after` = the expected device time once the row's actions land (knob -> fuse -> tune).
The mHC rows are listed once (the FFN wrap repeats them). Bound classes: `DRAM` (within 1.6x of the DRAM floor), `DRAM
(under-utilized)` (DRAM-dominated but > 1.6x: access pattern or fixed cost), `compute`, `ethernet/CCL`, `launch/gap`
(fixed per-op cost dominates), `L1/NoC` (far below the DRAM rate on few cores or with small accesses), `data-movement`
(layout ops to erase by fusion).

### KDA+MoE (x31 per step)

| op | module | n | MB 32u / 256u | measured us 32u / 256u (profile) | floor us 32u / 256u | ratio | bound | action (knob / fuse / tune) | after us 32u / 256u |
|---|---|---|---|---|---|---|---|---|---|
| mhc.proj | MHC_PRE (x2 wraps) | 1 | 4.3 / 4.3 | 12.9 / 12.7 (ep_b4 #248) | 10.8 / 10.8 | 1.2 / 1.2 | DRAM | A1: fuse into mhc_pre_block (reads x once for projection + sum of squares + h_mix) | 0 / 0 |
| mhc.xx | MHC_PRE (x2 wraps) | 1 | 4.2 / 4.2 | 23.2 / 23.8 (ep_b4 #249) | 10.5 / 10.5 | 2.2 / 2.3 | DRAM | A1: fused pre-block: the square never leaves DST | 0 / 0 |
| mhc.ssp | MHC_PRE (x2 wraps) | 1 | 2.1 / 2.1 | 10.4 / 10.4 (ep_b4 #250) | 5.2 / 5.2 | 2.0 / 2.0 | L1/NoC | A1: fused pre-block | 0 / 0 |
| mhc.ksum | MHC_PRE (x2 wraps) | 7 | 0.5 / 0.5 | 24.9 / 24.7 (ep_b4 #251-257) | 7.0 / 7.0 | 3.6 / 3.5 | data-movement | A1: fused pre-block (A9 alone: fast_reduce_nc, -10 ops) | 0 / 0 |
| mhc.rsqrt_mix | MHC_PRE (x2 wraps) | 2 | 0.0 / 0.0 | 12.3 / 12.5 (ep_b4 #258-259) | 2.0 / 2.0 | 6.2 / 6.2 | launch/gap | A1: fused pre-block | 0 / 0 |
| mhc.sinkhorn | MHC_PRE (x2 wraps) | 1 | 0.0 / 0.0 | 52.1 / 52.1 (ep_b4 #260) | 1.0 / 1.0 | 52.1 / 52.1 | L1/NoC | A4: per-row Sinkhorn kernel (one token row per core, 32 cores) inside the fused pre-block | 8 / 8 |
| mhc.hmix | MHC_PRE (x2 wraps) | 1 | 2.6 / 2.6 | 17.9 / 18.0 (ep_b4 #261) | 6.6 / 6.6 | 2.7 / 2.7 | DRAM (under-utilized) | A1: fused pre-block (reads x once; writes h bf16 sharded) | 0 / 0 |
| mhc.norm | MHC_PRE (x2 wraps) | 4 | 1.3 / 1.3 | 14.0 / 14.0 (A3 2026-10-26) | 4.0 / 4.0 | 3.5 / 3.5 | launch/gap | A1: fused pre-block writes the normed bf16 h directly (norm folded) | 0 / 0 |
| mhc.post_mix | MHC_POST (x2 wraps) | 1 | 4.5 / 4.5 | 34.3 / 34.0 (ep_b4 #271) | 11.1 / 11.1 | 3.1 / 3.1 | DRAM (under-utilized) | P1: tune the generic op's access pattern (125 GB/s today: resident operands, 8 tiles in flight per core) or keep the streams L1-resident between wraps (D13) | 20 / 20 |
| kda.qkv | ATTN_KDA | 1 | 27.4 / 27.4 | 65.2 / 65.3 (ep_b4 #264, 393 GB/s) | 68.5 / 68.5 | 1.0 / 1.0 | DRAM | keep | 65 / 65 |
| kda.gate1 | ATTN_KDA | 1 | 2.6 / 2.6 | 14.1 / 14.0 (ep_b4 #265, 187 GB/s) | 6.6 / 6.6 | 2.1 / 2.1 | L1/NoC | K1: append the 288 gate columns to the qkv weight (one [4096, 6432] matmul at ~390 GB/s) | 7 / 7 |
| kda.gate2 | ATTN_KDA | 1 | 2.9 / 2.9 | 9.4 / 9.1 (ep_b4 #266) | 7.3 / 7.3 | 1.3 / 1.2 | DRAM | K2: fold the block-diagonal [f_b \| g_b] expansion into kda_decode (per-head [18 x 128] weights in L1) | 0 / 0 |
| kda.decode | ATTN_KDA | 1 | 9.7 / 70.5 | 74.9 / 259.7 (ep_b4 #267) | 24.3 / 176.3 | 3.1 / 1.5 | DRAM (under-utilized) | K3: cut the ~49 us fixed cost + 34 us pre-op gap (runtime-arg volume of 80 cores x 3 RISCs: common args + per-core index; kernel prologue); the per-user slope (6.6 us = 3 | 40 / 215 |
| kda.o_proj | ATTN_KDA | 1 | 9.3 / 9.3 | 27.1 / 26.1 (ep_b4 #268, 348 GB/s) | 23.3 / 23.3 | 1.2 / 1.1 | DRAM | keep | 27 / 26 |
| kda.all_reduce | ATTN_KDA | 2 | 0.0 / 0.0 | 51.0 / 51.7 (ep_b4 #269-270) | 26.5 / 26.5 | 1.9 / 1.9 | ethernet/CCL | C2: the MoE all-reduce on the same shape takes 30 us: the extra ~20 us is column skew / input placement (RS waits for the slowest column after kda_decode); B7 gather form | 30 / 30 |
| moe.router_gate | MOE_ROUTER | 1 | 5.0 / 5.0 | 17.8 / 17.8 (ep_b4 #288) | 12.5 / 12.5 | 1.4 / 1.4 | DRAM | R1: widen the grid (9 -> 36 cores, per_core_N 1 -> the 1D config's 280 GB/s -> ~12 us) | 13 / 13 |
| moe.router_topk | MOE_ROUTER | 1 | 0.0 / 0.0 | 12.1 / 13.8 (ep_b4 #289) | 1.0 / 1.0 | 12.1 / 13.8 | launch/gap | keep | 12 / 14 |
| moe.gather_x | MOE_HYB_GATHER | 5 | 0.0 / 0.0 | 95.0 / 101.7 (proj) | 17.6 / 51.0 | 5.4 / 2.0 | ethernet/CCL | H1: T < 32: one all_gather_async on the compact row-major [T, H] (32 KB per chip), no pads / tilize | 35 / 102 |
| moe.gather_routing | MOE_HYB_GATHER | 5 | 0.0 / 0.0 | 25.0 / 15.3 (proj) | 13.5 / 18.2 | 1.9 / 0.8 | ethernet/CCL | H1: gather the top-k lists (T x 8 x 8 B) instead of the dense [T,288] fp32 routing | 15 / 15 |
| moe.experts | MOE_HYB_EXPERTS | 13 | 97.8 / 198.0 | 683.0 / 1198.0 (expert_kernel_sweep.md) | 244.4 / 495.1 | 2.8 / 2.4 | DRAM (under-utilized) | E0 -> E1 -> C12: bfp4 gate\|up experts (GLM53_EXPERT_W13_DTYPE=bfp4): -1/3 of the expert bytes, eval-gated | 180 / 600 |
| moe.rs_rows | MOE_HYB_RS | 10 | 0.0 / 0.0 | 162.0 / 128.5 (proj) | 28.0 / 68.2 | 5.8 / 1.9 | ethernet/CCL | H2: T < 32: write the expert output compact ([rows x T, H] = 256 KB) and all_reduce_rows it (every row slices its T rows): 8x fewer bytes than the padded reduce-scatter,  | 50 / 128 |
| moe.shared_w13 | MOE_SHARED | 1 | 4.8 / 4.8 | 16.9 / 16.5 (ep_b4 #304) | 12.0 / 12.0 | 1.4 / 1.4 | DRAM | S1: fold the 2 slices + clamped_silu_glu into the matmul epilogue (gate/up column-interleaved blocks) | 14 / 14 |
| moe.shared_glue | MOE_SHARED | 3 | 0.2 / 0.2 | 8.0 / 7.8 (ep_b4 #305-307) | 3.0 / 3.0 | 2.7 / 2.6 | data-movement | S1: fused into the w13 matmul | 0 / 0 |
| moe.shared_w2 | MOE_SHARED | 1 | 2.5 / 2.5 | 9.1 / 9.3 (ep_b4 #308) | 6.3 / 6.3 | 1.4 / 1.5 | DRAM | keep | 9 / 9 |
| moe.add | MOE | 1 | 0.8 / 0.8 | 3.6 / 3.5 (ep_b4 #309) | 2.0 / 2.0 | 1.8 / 1.8 | launch/gap | S1: fold into the shared w2 matmul (fused add) or the all-reduce input | 0 / 0 |
| moe.all_reduce | MOE | 2 | 0.0 / 0.0 | 29.7 / 28.7 (ep_b4 #310-311) | 26.5 / 26.5 | 1.1 / 1.1 | ethernet/CCL | keep | 30 / 29 |
| layer.gaps | TRACE | 88 | 0.0 / 0.0 | 85.0 / 85.0 (ledger section 3) | 40.5 / 29.4 | 2.1 / 2.9 | launch/gap | A1: fewer ops: every erased op saves its gap (0.46 us median, 1-1.5 us after single-core ops) | 85 / 85 |

### DSA+MoE (x11 per step)

| op | module | n | MB 32u / 256u | measured us 32u / 256u (profile) | floor us 32u / 256u | ratio | bound | action (knob / fuse / tune) | after us 32u / 256u |
|---|---|---|---|---|---|---|---|---|---|
| mhc.proj | MHC_PRE (x2 wraps) | 1 | 4.3 / 4.3 | 12.9 / 12.7 (ep_b4 #248) | 10.8 / 10.8 | 1.2 / 1.2 | DRAM | A1: fuse into mhc_pre_block (reads x once for projection + sum of squares + h_mix) | 0 / 0 |
| mhc.xx | MHC_PRE (x2 wraps) | 1 | 4.2 / 4.2 | 23.2 / 23.8 (ep_b4 #249) | 10.5 / 10.5 | 2.2 / 2.3 | DRAM | A1: fused pre-block: the square never leaves DST | 0 / 0 |
| mhc.ssp | MHC_PRE (x2 wraps) | 1 | 2.1 / 2.1 | 10.4 / 10.4 (ep_b4 #250) | 5.2 / 5.2 | 2.0 / 2.0 | L1/NoC | A1: fused pre-block | 0 / 0 |
| mhc.ksum | MHC_PRE (x2 wraps) | 7 | 0.5 / 0.5 | 24.9 / 24.7 (ep_b4 #251-257) | 7.0 / 7.0 | 3.6 / 3.5 | data-movement | A1: fused pre-block (A9 alone: fast_reduce_nc, -10 ops) | 0 / 0 |
| mhc.rsqrt_mix | MHC_PRE (x2 wraps) | 2 | 0.0 / 0.0 | 12.3 / 12.5 (ep_b4 #258-259) | 2.0 / 2.0 | 6.2 / 6.2 | launch/gap | A1: fused pre-block | 0 / 0 |
| mhc.sinkhorn | MHC_PRE (x2 wraps) | 1 | 0.0 / 0.0 | 52.1 / 52.1 (ep_b4 #260) | 1.0 / 1.0 | 52.1 / 52.1 | L1/NoC | A4: per-row Sinkhorn kernel (one token row per core, 32 cores) inside the fused pre-block | 8 / 8 |
| mhc.hmix | MHC_PRE (x2 wraps) | 1 | 2.6 / 2.6 | 17.9 / 18.0 (ep_b4 #261) | 6.6 / 6.6 | 2.7 / 2.7 | DRAM (under-utilized) | A1: fused pre-block (reads x once; writes h bf16 sharded) | 0 / 0 |
| mhc.norm | MHC_PRE (x2 wraps) | 4 | 1.3 / 1.3 | 14.0 / 14.0 (A3 2026-10-26) | 4.0 / 4.0 | 3.5 / 3.5 | launch/gap | A1: fused pre-block writes the normed bf16 h directly (norm folded) | 0 / 0 |
| mhc.post_mix | MHC_POST (x2 wraps) | 1 | 4.5 / 4.5 | 34.3 / 34.0 (ep_b4 #271) | 11.1 / 11.1 | 3.1 / 3.1 | DRAM (under-utilized) | P1: tune the generic op's access pattern (125 GB/s today: resident operands, 8 tiles in flight per core) or keep the streams L1-resident between wraps (D13) | 20 / 20 |
| dsa.stem | ATTN_DSA | 1 | 19.5 / 19.5 | 48.3 / 48.3 (blk_l3_b4 #438, 405 GB/s) | 48.9 / 48.9 | 1.0 / 1.0 | DRAM | D1: q_a part in bfp8 (kv_a stays bf16: it is the cached latent): -5.9 MB | 34 / 34 |
| dsa.state_composite | ATTN_DSA | 29 | 0.5 / 0.8 | 100.3 / 184.3 (blk_l3_b4 #439-468) | 29.0 / 29.0 | 3.5 / 6.4 | data-movement | B5: GLM53_DSA_DECODE_STATE=1: one dsa_decode_state program (29 -> 1 op; measured -109 us/layer e2e) | 40 / 75 |
| dsa.q_b | ATTN_DSA | 1 | 7.0 / 7.0 | 20.3 / 20.3 (blk_l3_b4 #445) | 17.6 / 17.6 | 1.2 / 1.2 | DRAM | keep | 20 / 20 |
| dsa.q_view | ATTN_DSA | 2 | 1.0 / 1.0 | 14.5 / 14.5 (blk_l3_b4 #469-470) | 2.6 / 2.6 | 5.5 / 5.5 | data-movement | B5: free head-major view (T <= 32: the packed tile sequence IS the head-major sequence) | 0 / 0 |
| dsa.absorb_q | ATTN_DSA | 1 | 3.0 / 3.0 | 20.0 / 20.0 (blk_l3_b4 #471, 144 GB/s) | 7.5 / 7.5 | 2.7 / 2.7 | L1/NoC | B5: fold into q_b (W_qb1 = Wqb . Wkb1 per head): one matmul; or keep the bmm on 64 cores | 0 / 0 |
| dsa.idx_query | ATTN_DSA | 1 | 12.9 / 12.9 | 31.4 / 31.4 (blk_l3_b4 #472, 412 GB/s) | 32.4 / 32.4 | 1.0 / 1.0 | DRAM | D2: bfp8 indexer query weight (-6 MB) if the selected-pool sets stay identical (A/B on the index) | 18 / 18 |
| dsa.idx_view | ATTN_DSA | 3 | 1.6 / 1.6 | 27.7 / 27.7 (blk_l3_b4 #473-475) | 3.9 / 3.9 | 7.0 / 7.0 | data-movement | B5: free view (tt/indexer.py free_view, part of B5) | 0 / 0 |
| dsa.index_score | ATTN_DSA | 1 | 0.3 / 0.3 | 10.7 / 12.0 (blk_l3_b4 #476) | 1.0 / 1.0 | 10.7 / 12.0 | launch/gap | keep | 11 / 12 |
| dsa.topk_rows | ATTN_DSA | 1 | 0.0 / 0.0 | 4.7 / 6.0 (blk_l3_b4 #477) | 1.0 / 1.0 | 4.7 / 6.0 | launch/gap | keep | 5 / 6 |
| dsa.pool_gather | ATTN_DSA | 1 | 16.8 / 134.4 | 136.4 / 1270.0 (blk_l3_b4 #478) | 42.0 / 336.0 | 3.2 / 3.8 | L1/NoC | B8: (a) dense regime (position <= 2050): attend the paged cache directly, no gather; (b) pool-major cache pages: one 4 KB run per pool instead of 32 x 128 B pieces | 0 / 0 |
| dsa.sdpa_layout | ATTN_DSA | 2 | 2.1 / 2.1 | 8.7 / 8.7 (blk_l3_b4 #479, #481) | 5.2 / 5.2 | 1.7 / 1.7 | data-movement | B5: free views with the packed [1,1,B,H*L] layout | 0 / 0 |
| dsa.sdpa | ATTN_DSA | 1 | 9.6 / 69.2 | 72.9 / 170.4 (blk_l3_b4 #480) | 23.9 / 173.0 | 3.0 / 1.0 | L1/NoC | D3: B <= 8: more cores per head-batch (k_chunk 32 -> 128 over 2080 rows); dense regime: attend the paged cache (B8a) so no compact copy is read twice | 40 / 170 |
| dsa.absorb_o | ATTN_DSA | 1 | 3.0 / 3.0 | 19.2 / 19.2 (blk_l3_b4 #482, 150 GB/s) | 7.5 / 7.5 | 2.5 / 2.5 | L1/NoC | D4: keep as a bmm on 64 cores (the o fold W_o2 [8192,4096] costs +18 MB = +45 us of DRAM floor for the 31 us of ops it removes: roofline-negative at every batch) -> GLM53 | 8 / 8 |
| dsa.o_view | ATTN_DSA | 2 | 1.0 / 1.0 | 11.5 / 11.5 (blk_l3_b4 #483-484) | 2.6 / 2.6 | 4.4 / 4.4 | data-movement | B5: free view | 0 / 0 |
| dsa.o_proj | ATTN_DSA | 1 | 18.4 / 18.4 | 43.7 / 43.7 (blk_l3_b4 #485, 396 GB/s) | 45.9 / 45.9 | 1.0 / 1.0 | DRAM | keep | 44 / 44 |
| dsa.all_reduce | ATTN_DSA | 2 | 0.0 / 0.0 | 39.5 / 39.5 (blk_l3_b4 #486-487) | 26.5 / 26.5 | 1.5 / 1.5 | ethernet/CCL | C2: as the KDA all-reduce: column skew; keep ccl | 30 / 30 |
| moe.router_gate | MOE_ROUTER | 1 | 5.0 / 5.0 | 17.8 / 17.8 (ep_b4 #288) | 12.5 / 12.5 | 1.4 / 1.4 | DRAM | R1: widen the grid (9 -> 36 cores, per_core_N 1 -> the 1D config's 280 GB/s -> ~12 us) | 13 / 13 |
| moe.router_topk | MOE_ROUTER | 1 | 0.0 / 0.0 | 12.1 / 13.8 (ep_b4 #289) | 1.0 / 1.0 | 12.1 / 13.8 | launch/gap | keep | 12 / 14 |
| moe.gather_x | MOE_HYB_GATHER | 5 | 0.0 / 0.0 | 95.0 / 101.7 (proj) | 17.6 / 51.0 | 5.4 / 2.0 | ethernet/CCL | H1: T < 32: one all_gather_async on the compact row-major [T, H] (32 KB per chip), no pads / tilize | 35 / 102 |
| moe.gather_routing | MOE_HYB_GATHER | 5 | 0.0 / 0.0 | 25.0 / 15.3 (proj) | 13.5 / 18.2 | 1.9 / 0.8 | ethernet/CCL | H1: gather the top-k lists (T x 8 x 8 B) instead of the dense [T,288] fp32 routing | 15 / 15 |
| moe.experts | MOE_HYB_EXPERTS | 13 | 97.8 / 198.0 | 683.0 / 1198.0 (expert_kernel_sweep.md) | 244.4 / 495.1 | 2.8 / 2.4 | DRAM (under-utilized) | E0 -> E1 -> C12: bfp4 gate\|up experts (GLM53_EXPERT_W13_DTYPE=bfp4): -1/3 of the expert bytes, eval-gated | 180 / 600 |
| moe.rs_rows | MOE_HYB_RS | 10 | 0.0 / 0.0 | 162.0 / 128.5 (proj) | 28.0 / 68.2 | 5.8 / 1.9 | ethernet/CCL | H2: T < 32: write the expert output compact ([rows x T, H] = 256 KB) and all_reduce_rows it (every row slices its T rows): 8x fewer bytes than the padded reduce-scatter,  | 50 / 128 |
| moe.shared_w13 | MOE_SHARED | 1 | 4.8 / 4.8 | 16.9 / 16.5 (ep_b4 #304) | 12.0 / 12.0 | 1.4 / 1.4 | DRAM | S1: fold the 2 slices + clamped_silu_glu into the matmul epilogue (gate/up column-interleaved blocks) | 14 / 14 |
| moe.shared_glue | MOE_SHARED | 3 | 0.2 / 0.2 | 8.0 / 7.8 (ep_b4 #305-307) | 3.0 / 3.0 | 2.7 / 2.6 | data-movement | S1: fused into the w13 matmul | 0 / 0 |
| moe.shared_w2 | MOE_SHARED | 1 | 2.5 / 2.5 | 9.1 / 9.3 (ep_b4 #308) | 6.3 / 6.3 | 1.4 / 1.5 | DRAM | keep | 9 / 9 |
| moe.add | MOE | 1 | 0.8 / 0.8 | 3.6 / 3.5 (ep_b4 #309) | 2.0 / 2.0 | 1.8 / 1.8 | launch/gap | S1: fold into the shared w2 matmul (fused add) or the all-reduce input | 0 / 0 |
| moe.all_reduce | MOE | 2 | 0.0 / 0.0 | 29.7 / 28.7 (ep_b4 #310-311) | 26.5 / 26.5 | 1.1 / 1.1 | ethernet/CCL | keep | 30 / 29 |
| layer.gaps | TRACE | 131 | 0.0 / 0.0 | 90.0 / 90.0 (ledger section 3) | 60.3 / 49.2 | 1.5 / 1.8 | launch/gap | A1: fewer ops: every erased op saves its gap (0.46 us median, 1-1.5 us after single-core ops) | 90 / 90 |

### KDA+dense (x3 per step)

| op | module | n | MB 32u / 256u | measured us 32u / 256u (profile) | floor us 32u / 256u | ratio | bound | action (knob / fuse / tune) | after us 32u / 256u |
|---|---|---|---|---|---|---|---|---|---|
| mhc.proj | MHC_PRE (x2 wraps) | 1 | 4.3 / 4.3 | 12.9 / 12.7 (ep_b4 #248) | 10.8 / 10.8 | 1.2 / 1.2 | DRAM | A1: fuse into mhc_pre_block (reads x once for projection + sum of squares + h_mix) | 0 / 0 |
| mhc.xx | MHC_PRE (x2 wraps) | 1 | 4.2 / 4.2 | 23.2 / 23.8 (ep_b4 #249) | 10.5 / 10.5 | 2.2 / 2.3 | DRAM | A1: fused pre-block: the square never leaves DST | 0 / 0 |
| mhc.ssp | MHC_PRE (x2 wraps) | 1 | 2.1 / 2.1 | 10.4 / 10.4 (ep_b4 #250) | 5.2 / 5.2 | 2.0 / 2.0 | L1/NoC | A1: fused pre-block | 0 / 0 |
| mhc.ksum | MHC_PRE (x2 wraps) | 7 | 0.5 / 0.5 | 24.9 / 24.7 (ep_b4 #251-257) | 7.0 / 7.0 | 3.6 / 3.5 | data-movement | A1: fused pre-block (A9 alone: fast_reduce_nc, -10 ops) | 0 / 0 |
| mhc.rsqrt_mix | MHC_PRE (x2 wraps) | 2 | 0.0 / 0.0 | 12.3 / 12.5 (ep_b4 #258-259) | 2.0 / 2.0 | 6.2 / 6.2 | launch/gap | A1: fused pre-block | 0 / 0 |
| mhc.sinkhorn | MHC_PRE (x2 wraps) | 1 | 0.0 / 0.0 | 52.1 / 52.1 (ep_b4 #260) | 1.0 / 1.0 | 52.1 / 52.1 | L1/NoC | A4: per-row Sinkhorn kernel (one token row per core, 32 cores) inside the fused pre-block | 8 / 8 |
| mhc.hmix | MHC_PRE (x2 wraps) | 1 | 2.6 / 2.6 | 17.9 / 18.0 (ep_b4 #261) | 6.6 / 6.6 | 2.7 / 2.7 | DRAM (under-utilized) | A1: fused pre-block (reads x once; writes h bf16 sharded) | 0 / 0 |
| mhc.norm | MHC_PRE (x2 wraps) | 4 | 1.3 / 1.3 | 14.0 / 14.0 (A3 2026-10-26) | 4.0 / 4.0 | 3.5 / 3.5 | launch/gap | A1: fused pre-block writes the normed bf16 h directly (norm folded) | 0 / 0 |
| mhc.post_mix | MHC_POST (x2 wraps) | 1 | 4.5 / 4.5 | 34.3 / 34.0 (ep_b4 #271) | 11.1 / 11.1 | 3.1 / 3.1 | DRAM (under-utilized) | P1: tune the generic op's access pattern (125 GB/s today: resident operands, 8 tiles in flight per core) or keep the streams L1-resident between wraps (D13) | 20 / 20 |
| kda.qkv | ATTN_KDA | 1 | 27.4 / 27.4 | 65.2 / 65.3 (ep_b4 #264, 393 GB/s) | 68.5 / 68.5 | 1.0 / 1.0 | DRAM | keep | 65 / 65 |
| kda.gate1 | ATTN_KDA | 1 | 2.6 / 2.6 | 14.1 / 14.0 (ep_b4 #265, 187 GB/s) | 6.6 / 6.6 | 2.1 / 2.1 | L1/NoC | K1: append the 288 gate columns to the qkv weight (one [4096, 6432] matmul at ~390 GB/s) | 7 / 7 |
| kda.gate2 | ATTN_KDA | 1 | 2.9 / 2.9 | 9.4 / 9.1 (ep_b4 #266) | 7.3 / 7.3 | 1.3 / 1.2 | DRAM | K2: fold the block-diagonal [f_b \| g_b] expansion into kda_decode (per-head [18 x 128] weights in L1) | 0 / 0 |
| kda.decode | ATTN_KDA | 1 | 9.7 / 70.5 | 74.9 / 259.7 (ep_b4 #267) | 24.3 / 176.3 | 3.1 / 1.5 | DRAM (under-utilized) | K3: cut the ~49 us fixed cost + 34 us pre-op gap (runtime-arg volume of 80 cores x 3 RISCs: common args + per-core index; kernel prologue); the per-user slope (6.6 us = 3 | 40 / 215 |
| kda.o_proj | ATTN_KDA | 1 | 9.3 / 9.3 | 27.1 / 26.1 (ep_b4 #268, 348 GB/s) | 23.3 / 23.3 | 1.2 / 1.1 | DRAM | keep | 27 / 26 |
| kda.all_reduce | ATTN_KDA | 2 | 0.0 / 0.0 | 51.0 / 51.7 (ep_b4 #269-270) | 26.5 / 26.5 | 1.9 / 1.9 | ethernet/CCL | C2: the MoE all-reduce on the same shape takes 30 us: the extra ~20 us is column skew / input placement (RS waits for the slowest column after kda_decode); B7 gather form | 30 / 30 |
| mlp.w13 | MLP_DENSE | 1 | 27.4 / 27.4 | 76.1 / 76.1 (matmul sweep 4096x6144) | 68.5 / 68.5 | 1.1 / 1.1 | DRAM | S1: the kda qkv runs the same shape at 65 us: apply its config | 66 / 66 |
| mlp.glue | MLP_DENSE | 3 | 1.2 / 1.2 | 12.0 / 12.0 (proj) | 3.0 / 3.0 | 4.0 / 4.0 | data-movement | S1: fused into the w13 matmul | 0 / 0 |
| mlp.w2 | MLP_DENSE | 1 | 13.8 / 13.8 | 43.9 / 43.9 (matmul sweep 3072x4096) | 34.6 / 34.6 | 1.3 / 1.3 | DRAM | keep | 44 / 44 |
| mlp.all_reduce | MLP_DENSE | 2 | 0.0 / 0.0 | 29.7 / 28.7 (as the MoE all-reduce) | 26.5 / 26.5 | 1.1 / 1.1 | ethernet/CCL | keep | 30 / 29 |
| layer.gaps | TRACE | 52 | 0.0 / 0.0 | 40.0 / 40.0 (proj) | 23.9 / 23.9 | 1.7 / 1.7 | launch/gap | A1: fewer ops: every erased op saves its gap (0.46 us median, 1-1.5 us after single-core ops) | 40 / 40 |

### head (x1 per step)

| op | module | n | MB 32u / 256u | measured us 32u / 256u (profile) | floor us 32u / 256u | ratio | bound | action (knob / fuse / tune) | after us 32u / 256u |
|---|---|---|---|---|---|---|---|---|---|
| embed.gather | EMBED | 1 | 2.4 / 2.4 | 20.0 / 20.0 (proj) | 5.9 / 5.9 | 3.4 / 3.4 | launch/gap | keep | 20 / 20 |
| head.norm_cast | HEAD | 2 | 1.0 / 1.0 | 15.0 / 15.0 (proj) | 2.6 / 2.6 | 5.7 / 5.7 | launch/gap | keep | 15 / 15 |
| head.matmul | HEAD | 1 | 171.2 / 171.2 | 432.0 / 432.0 (proj) | 428.1 / 428.1 | 1.0 / 1.0 | DRAM | keep | 432 / 432 |
| head.all_gather_cols | HEAD | 1 | 0.0 / 0.0 | 170.0 / 170.0 (proj) | 162.5 / 162.5 | 1.0 / 1.0 | ethernet/CCL | HD1: argmax per column on the local block + all_gather of 32 (value, id) pairs (-7.4 MB on the link) | 15 / 15 |
| head.argmax | HEAD | 1 | 9.9 / 9.9 | 15.0 / 15.0 (proj) | 24.8 / 24.8 | 0.6 / 0.6 | launch/gap | keep | 15 / 15 |
| layer.gaps | TRACE | 6 | 0.0 / 0.0 | 5.0 / 5.0 (proj) | 2.8 / 2.8 | 1.8 / 1.8 | launch/gap | A1: fewer ops: every erased op saves its gap (0.46 us median, 1-1.5 us after single-core ops) | 5 / 5 |

### EP32 MoE rows (`GLM53_MOE_LAYOUT=ep`; the other rows are identical)

| op | n | MB 32u / 256u | measured us 32u / 256u | floor us 32u / 256u | ratio | bound | action | after us 32u / 256u |
|---|---|---|---|---|---|---|---|---|
| ep.dispatch_glue | 5 | 1.0 / 1.0 | 10 / 24 | 5 / 5 | 2.0 / 4.8 | data-movement | C10: router_topk writes w bf16 / idx u16 ROW_MAJOR directly (-3 ops) | 4 / 10 |
| ep.a2a_dispatch | 1 | 0.0 / 0.0 | 67 / 93 | 28 / 139 | 2.4 / 0.7 | ethernet/CCL | keep | 67 / 93 |
| ep.experts | 2 | 161.0 / 218.1 | 218 / 441 | 402 / 545 | 0.5 / 0.8 | DRAM | E2: placement / replication of hot experts across the column's chips (the skew the combine waits for IS the cost); bfp4 gate\|up | 218 / 441 |
| ep.combine_rs | 1 | 0.0 / 0.0 | 459 / 990 | 68 / 68 | 6.7 / 14.5 | ethernet/CCL | C1 -> E2: hot-expert placement / replication across the column's chips so the slowest sender is near the mean (the RS floor is 68 us) | 160 / 300 |

Reading the table:

* **DRAM-bound at the achievable rate (keep, or change bytes)**: kda.qkv (393 GB/s), dsa.stem (405), dsa.idx_query
  (412), dsa.o_proj (396), kda.o_proj (348), mlp.w13 / w2, moe.shared_w13 / w2 (260-280 GB/s on 16-22 cores: K1 / S1
  fold the glue in). Only fewer bytes help: bfp4 gate|up experts (C12), bfp8 for the indexer query and the q_a part of the
  stem (D1 / D2, both gated on the selected-pool sets staying identical).
* **Experts**: the stream op at 28 active slices runs at 274 GB/s = 1.4x its floor; the ~110 us fixed per-block cost
  (launch x2, x copy, routing scans, DST fill) is the tuning target (E1), bfp4 the byte lever (C12). At 256 users the dense
  layout reads all 72 slices and computes 116 GFLOP (358 us at the HiFi2 peak) for 3.2 GFLOP of routed work: the dispatch
  form (779 us measured) is the fix (E1); its down op is the next lever (~-400 us: expert-parallel split).
* **kda_decode**: two-point fit 48.6 us fixed + 6.6 us per user (303 GB/s marginal: at DRAM rate). The fixed cost plus the
  34 us gap before it (not moved by the rectangular core ranges, B6) are launch-bound (K3: runtime-arg volume of 80 cores
  x 3 RISCs, kernel prologue).
* **mHC**: 18 pre-block ops per wrap of 168 us (post-A3) against a 47 us floor (x read twice, fn 2 MB, h out): the fused
  pre-block (A1, measured 101 us) is the fuse; the single-core Sinkhorn (52 us) is the tune inside it (A4); the post-mix
  generic op moves 4.5 MB in 34 us = 125 GB/s (P1: access pattern). D13 (column-sharded streams) would divide the whole
  wrap by 4 but changes the contract through embedding, head and prefill.
* **Row collectives**: the measured reduce-scatter (128 us) and all-gather (102 us) of the 2 MB / 256 KB tensors run at
  ~2x the CCL report's fit (68 / 51 us): a num_links / chunks_per_sync / topology sweep of `test_ccl_perf.py` is the tune.
  At T = 4 the padded tensors move 8x the needed bytes plus 10 glue ops each: H1 (compact row-major gather) and H2 (compact
  expert output + all_reduce_rows instead of the padded RS) are fuses worth 7.6 ms per step.
* **TP all-reduces**: the attention one (51 us) is 1.9x the fit while the MoE one on the same shape is 1.1x (30 us): the
  difference is column skew after kda_decode / the input placement (C2); B7's gather form measured 46 us (no gain), keep ccl.
* **DSA**: 29 state ops (100 us, launch-bound) -> B5's one program; the absorb bmms run at 144-150 GB/s on 16 cores
  (L1/NoC); the q fold is DRAM-neutral-to-positive but the o fold adds 18 MB of weight (+45 us of floor) for 31 us of ops:
  prefer `GLM53_DSA_FOLD=q`. paged_pool_gather moves 8.4 MB in 136 us (62 GB/s: 128 B pieces, L1/NoC) and is pure waste at
  ISL <= 2050 (B8a); the flash-MLA decode reads the compact copy at 12 cores (D3).

## 4. Per layer type and per step

#### 32 users (4 per row), hybrid, no expert knob (sparse tuned block)

layout: 8x4, 4 users/row = 32 users, moe=hybrid (block), experts bfp8/bfp8, ISL 16, DRAM 400 GB/s (spec 512), links 2 x 25 GB/s (line)

| layer | count | ops | floor_us | floor_spec_us | model_measured_us | ratio | after_us |
|---|---|---|---|---|---|---|---|
| KDA+MoE | 31 | 88 | 680.3 | 557.4 | 1792.9 | 2.6 | 668.2 |
| DSA+MoE | 11 | 131 | 841.3 | 694.2 | 2166.0 | 2.6 | 753.3 |
| KDA+dense | 3 | 52 | 429.5 | 350.3 | 847.4 | 2.0 | 404.9 |
| head | 1 | 6 | 626.7 | 448.3 | 657.0 | 1.1 | 502.0 |

step floor: 32.26 ms at 400 GB/s (26.41 ms at the spec peaks) -> ceiling 31.0 tok/s per user (37.9 at spec), 992 aggregate; target 30.3 ms
step model (sum of the measured / projected ops): 82.61 ms; after the listed actions: 30.72 ms = 32.6 tok/s per user
e2e measured steps: hybrid 2026-10-26 (A3 sharded norm, B6 ranges, distinct ids): 80.55 ms, hybrid 2026-10-21 (distinct ids): 83.1 ms, hybrid 2026-10-21 (same token on every slot: 2 active experts): 69.8 ms, EP32 op combine 2026-10-21 (distinct ids): 71.2 ms, EP32 rs-remap 2026-10-26 (distinct ids): 67.68 ms

per module (us per layer):

| layer | module | ops | floor us | measured us | after us | ratio |
|---|---|---|---|---|---|---|
| KDA+MoE | MHC_PRE_ATTN | 18 | 47 | 168 | 8 | 3.6 |
| KDA+MoE | MHC_POST_ATTN | 1 | 11 | 34 | 20 | 3.1 |
| KDA+MoE | ATTN_KDA | 7 | 156 | 242 | 169 | 1.5 |
| KDA+MoE | MHC_PRE_FFN | 18 | 47 | 168 | 8 | 3.6 |
| KDA+MoE | MHC_POST_FFN | 1 | 11 | 34 | 20 | 3.1 |
| KDA+MoE | MOE_ROUTER | 2 | 14 | 30 | 25 | 2.2 |
| KDA+MoE | MOE_HYB_GATHER | 10 | 31 | 120 | 50 | 3.9 |
| KDA+MoE | MOE_HYB_EXPERTS | 13 | 244 | 683 | 180 | 2.8 |
| KDA+MoE | MOE_HYB_RS | 10 | 28 | 162 | 50 | 5.8 |
| KDA+MoE | MOE_SHARED | 5 | 21 | 34 | 23 | 1.6 |
| KDA+MoE | MOE | 3 | 28 | 33 | 30 | 1.2 |
| KDA+MoE | TRACE | 0 | 40 | 85 | 85 | 2.1 |
| DSA+MoE | MHC_PRE_ATTN | 18 | 47 | 168 | 8 | 3.6 |
| DSA+MoE | MHC_POST_ATTN | 1 | 11 | 34 | 20 | 3.1 |
| DSA+MoE | ATTN_DSA | 50 | 298 | 610 | 249 | 2.0 |
| DSA+MoE | MHC_PRE_FFN | 18 | 47 | 168 | 8 | 3.6 |
| DSA+MoE | MHC_POST_FFN | 1 | 11 | 34 | 20 | 3.1 |
| DSA+MoE | MOE_ROUTER | 2 | 14 | 30 | 25 | 2.2 |
| DSA+MoE | MOE_HYB_GATHER | 10 | 31 | 120 | 50 | 3.9 |
| DSA+MoE | MOE_HYB_EXPERTS | 13 | 244 | 683 | 180 | 2.8 |
| DSA+MoE | MOE_HYB_RS | 10 | 28 | 162 | 50 | 5.8 |
| DSA+MoE | MOE_SHARED | 5 | 21 | 34 | 23 | 1.6 |
| DSA+MoE | MOE | 3 | 28 | 33 | 30 | 1.2 |
| DSA+MoE | TRACE | 0 | 60 | 90 | 90 | 1.5 |
| KDA+dense | MHC_PRE_ATTN | 18 | 47 | 168 | 8 | 3.6 |
| KDA+dense | MHC_POST_ATTN | 1 | 11 | 34 | 20 | 3.1 |
| KDA+dense | ATTN_KDA | 7 | 156 | 242 | 169 | 1.5 |
| KDA+dense | MHC_PRE_FFN | 18 | 47 | 168 | 8 | 3.6 |
| KDA+dense | MHC_POST_FFN | 1 | 11 | 34 | 20 | 3.1 |
| KDA+dense | MLP_DENSE | 7 | 133 | 162 | 140 | 1.2 |
| KDA+dense | TRACE | 0 | 24 | 40 | 40 | 1.7 |
| head | EMBED | 1 | 6 | 20 | 20 | 3.4 |
| head | HEAD | 5 | 618 | 632 | 477 | 1.0 |
| head | TRACE | 0 | 3 | 5 | 5 | 1.8 |

ranked actions (ms saved per step at this point):

| rank | action | ms/step | risk | ops | what |
|---|---|---|---|---|---|
| 1 | E0 | 13.94 | L | moe.experts | GLM53_STREAM_EXPERTS=1: the expert_stream op (341 vs 683 us at 28 active; block test PCC 0.99997); confirm with a layer-4 tracy of the 8x4 hybrid at 4 users/row |
| 2 | A1 | 10.40 | M | layer.gaps, mhc.hmix, mhc.ksum, mhc.norm, mhc.proj, mhc.rsqrt_mix, mhc.ssp, mhc.xx | fuse into mhc_pre_block (reads x once for projection + sum of squares + h_mix) |
| 3 | H2 | 4.70 | M | moe.rs_rows | T < 32: write the expert output compact ([rows x T, H] = 256 KB) and all_reduce_rows it (every row slices its T rows): 8x fewer bytes than the padded reduce-scatter, no pads / slices |
| 4 | E1 | 4.24 | M | moe.experts | expert_stream fixed cost (~110 us: launch x2, x copy, routing scans, DST fill): fold the selector matmul + x copy into the op, pipeline the DST fill |
| 5 | A4 | 3.97 | M | mhc.sinkhorn | per-row Sinkhorn kernel (one token row per core, 32 cores) inside the fused pre-block |
| 6 | H1 | 2.94 | L-M | moe.gather_routing, moe.gather_x | T < 32: one all_gather_async on the compact row-major [T, H] (32 KB per chip), no pads / tilize |
| 7 | C12 | 2.94 | M (accuracy) | moe.experts | bfp4 gate\|up experts (GLM53_EXPERT_W13_DTYPE=bfp4): -1/3 of the expert bytes, eval-gated |
| 8 | B5 | 1.57 | M | dsa.absorb_q, dsa.idx_view, dsa.o_view, dsa.q_view, dsa.sdpa_layout, dsa.state_composite | GLM53_DSA_DECODE_STATE=1: one dsa_decode_state program (29 -> 1 op; measured -109 us/layer e2e) |
| 9 | B8 | 1.50 | M | dsa.pool_gather | (a) dense regime (position <= 2050): attend the paged cache directly, no gather; (b) pool-major cache pages: one 4 KB run per pool instead of 32 x 128 B pieces |
| 10 | P1 | 1.29 | L-M | mhc.post_mix | tune the generic op's access pattern (125 GB/s today: resident operands, 8 tiles in flight per core) or keep the streams L1-resident between wraps (D13) |
| 11 | K3 | 1.19 | M | kda.decode | cut the ~49 us fixed cost + 34 us pre-op gap (runtime-arg volume of 80 cores x 3 RISCs: common args + per-core index; kernel prologue); the per-user slope (6.6 us = 303 GB/s) is already at DRAM rate |
| 12 | C2 | 0.82 | L | dsa.all_reduce, kda.all_reduce | the MoE all-reduce on the same shape takes 30 us: the extra ~20 us is column skew / input placement (RS waits for the slowest column after kda_decode); B7 gather form measured 46 us (no gain) -> keep ccl; D13 column-sharded mHC would turn it into RS + AG of the streams (neutral) |
| 13 | S1 | 0.68 | L | mlp.glue, mlp.w13, moe.add, moe.shared_glue, moe.shared_w13 | fold the 2 slices + clamped_silu_glu into the matmul epilogue (gate/up column-interleaved blocks) |
| 14 | D3 | 0.36 | L-M | dsa.sdpa | B <= 8: more cores per head-batch (k_chunk 32 -> 128 over 2080 rows); dense regime: attend the paged cache (B8a) so no compact copy is read twice |

#### 32 users (4 per row), hybrid, GLM53_STREAM_EXPERTS=1

layout: 8x4, 4 users/row = 32 users, moe=hybrid (stream), experts bfp8/bfp8, ISL 16, DRAM 400 GB/s (spec 512), links 2 x 25 GB/s (line)

| layer | count | ops | floor_us | floor_spec_us | model_measured_us | ratio | after_us |
|---|---|---|---|---|---|---|---|
| KDA+MoE | 31 | 79 | 676.2 | 553.2 | 1460.9 | 2.2 | 668.2 |
| DSA+MoE | 11 | 122 | 837.1 | 690.1 | 1834.0 | 2.2 | 753.3 |
| KDA+dense | 3 | 52 | 429.5 | 350.3 | 847.4 | 2.0 | 404.9 |
| head | 1 | 6 | 626.7 | 448.3 | 657.0 | 1.1 | 502.0 |

step floor: 32.09 ms at 400 GB/s (26.24 ms at the spec peaks) -> ceiling 31.2 tok/s per user (38.1 at spec), 997 aggregate; target 30.3 ms
step model (sum of the measured / projected ops): 68.66 ms; after the listed actions: 30.72 ms = 32.6 tok/s per user
e2e measured steps: hybrid 2026-10-26 (A3 sharded norm, B6 ranges, distinct ids): 80.55 ms, hybrid 2026-10-21 (distinct ids): 83.1 ms, hybrid 2026-10-21 (same token on every slot: 2 active experts): 69.8 ms, EP32 op combine 2026-10-21 (distinct ids): 71.2 ms, EP32 rs-remap 2026-10-26 (distinct ids): 67.68 ms

per module (us per layer):

| layer | module | ops | floor us | measured us | after us | ratio |
|---|---|---|---|---|---|---|
| KDA+MoE | MHC_PRE_ATTN | 18 | 47 | 168 | 8 | 3.6 |
| KDA+MoE | MHC_POST_ATTN | 1 | 11 | 34 | 20 | 3.1 |
| KDA+MoE | ATTN_KDA | 7 | 156 | 242 | 169 | 1.5 |
| KDA+MoE | MHC_PRE_FFN | 18 | 47 | 168 | 8 | 3.6 |
| KDA+MoE | MHC_POST_FFN | 1 | 11 | 34 | 20 | 3.1 |
| KDA+MoE | MOE_ROUTER | 2 | 14 | 30 | 25 | 2.2 |
| KDA+MoE | MOE_HYB_GATHER | 10 | 31 | 120 | 50 | 3.9 |
| KDA+MoE | MOE_HYB_EXPERTS | 4 | 244 | 351 | 180 | 1.4 |
| KDA+MoE | MOE_HYB_RS | 10 | 28 | 162 | 50 | 5.8 |
| KDA+MoE | MOE_SHARED | 5 | 21 | 34 | 23 | 1.6 |
| KDA+MoE | MOE | 3 | 28 | 33 | 30 | 1.2 |
| KDA+MoE | TRACE | 0 | 36 | 85 | 85 | 2.3 |
| DSA+MoE | MHC_PRE_ATTN | 18 | 47 | 168 | 8 | 3.6 |
| DSA+MoE | MHC_POST_ATTN | 1 | 11 | 34 | 20 | 3.1 |
| DSA+MoE | ATTN_DSA | 50 | 298 | 610 | 249 | 2.0 |
| DSA+MoE | MHC_PRE_FFN | 18 | 47 | 168 | 8 | 3.6 |
| DSA+MoE | MHC_POST_FFN | 1 | 11 | 34 | 20 | 3.1 |
| DSA+MoE | MOE_ROUTER | 2 | 14 | 30 | 25 | 2.2 |
| DSA+MoE | MOE_HYB_GATHER | 10 | 31 | 120 | 50 | 3.9 |
| DSA+MoE | MOE_HYB_EXPERTS | 4 | 244 | 351 | 180 | 1.4 |
| DSA+MoE | MOE_HYB_RS | 10 | 28 | 162 | 50 | 5.8 |
| DSA+MoE | MOE_SHARED | 5 | 21 | 34 | 23 | 1.6 |
| DSA+MoE | MOE | 3 | 28 | 33 | 30 | 1.2 |
| DSA+MoE | TRACE | 0 | 56 | 90 | 90 | 1.6 |
| KDA+dense | MHC_PRE_ATTN | 18 | 47 | 168 | 8 | 3.6 |
| KDA+dense | MHC_POST_ATTN | 1 | 11 | 34 | 20 | 3.1 |
| KDA+dense | ATTN_KDA | 7 | 156 | 242 | 169 | 1.5 |
| KDA+dense | MHC_PRE_FFN | 18 | 47 | 168 | 8 | 3.6 |
| KDA+dense | MHC_POST_FFN | 1 | 11 | 34 | 20 | 3.1 |
| KDA+dense | MLP_DENSE | 7 | 133 | 162 | 140 | 1.2 |
| KDA+dense | TRACE | 0 | 24 | 40 | 40 | 1.7 |
| head | EMBED | 1 | 6 | 20 | 20 | 3.4 |
| head | HEAD | 5 | 618 | 632 | 477 | 1.0 |
| head | TRACE | 0 | 3 | 5 | 5 | 1.8 |

ranked actions (ms saved per step at this point):

| rank | action | ms/step | risk | ops | what |
|---|---|---|---|---|---|
| 1 | A1 | 10.40 | M | layer.gaps, mhc.hmix, mhc.ksum, mhc.norm, mhc.proj, mhc.rsqrt_mix, mhc.ssp, mhc.xx | fuse into mhc_pre_block (reads x once for projection + sum of squares + h_mix) |
| 2 | H2 | 4.70 | M | moe.rs_rows | T < 32: write the expert output compact ([rows x T, H] = 256 KB) and all_reduce_rows it (every row slices its T rows): 8x fewer bytes than the padded reduce-scatter, no pads / slices |
| 3 | E1 | 4.24 | M | moe.experts | expert_stream fixed cost (~110 us: launch x2, x copy, routing scans, DST fill): fold the selector matmul + x copy into the op, pipeline the DST fill |
| 4 | A4 | 3.97 | M | mhc.sinkhorn | per-row Sinkhorn kernel (one token row per core, 32 cores) inside the fused pre-block |
| 5 | H1 | 2.94 | L-M | moe.gather_routing, moe.gather_x | T < 32: one all_gather_async on the compact row-major [T, H] (32 KB per chip), no pads / tilize |
| 6 | C12 | 2.94 | M (accuracy) | moe.experts | bfp4 gate\|up experts (GLM53_EXPERT_W13_DTYPE=bfp4): -1/3 of the expert bytes, eval-gated |
| 7 | B5 | 1.57 | M | dsa.absorb_q, dsa.idx_view, dsa.o_view, dsa.q_view, dsa.sdpa_layout, dsa.state_composite | GLM53_DSA_DECODE_STATE=1: one dsa_decode_state program (29 -> 1 op; measured -109 us/layer e2e) |
| 8 | B8 | 1.50 | M | dsa.pool_gather | (a) dense regime (position <= 2050): attend the paged cache directly, no gather; (b) pool-major cache pages: one 4 KB run per pool instead of 32 x 128 B pieces |

#### 256 users (32 per row), hybrid dense layout (the 115 ms e2e)

layout: 8x4, 32 users/row = 256 users, moe=hybrid (block), experts bfp8/bfp8, ISL 16, DRAM 400 GB/s (spec 512), links 2 x 25 GB/s (line)

| layer | count | ops | floor_us | floor_spec_us | model_measured_us | ratio | after_us |
|---|---|---|---|---|---|---|---|
| KDA+MoE | 31 | 64 | 1150.2 | 900.0 | 2456.2 | 2.1 | 1408.4 |
| DSA+MoE | 11 | 107 | 1602.3 | 1264.3 | 3962.8 | 2.5 | 1487.4 |
| KDA+dense | 3 | 52 | 581.5 | 469.0 | 1031.0 | 1.8 | 578.0 |
| head | 1 | 6 | 626.7 | 448.3 | 657.0 | 1.1 | 502.0 |

step floor: 55.65 ms at 400 GB/s (43.66 ms at the spec peaks) -> ceiling 18.0 tok/s per user (22.9 at spec), 4600 aggregate; target 52.6 ms
step model (sum of the measured / projected ops): 123.48 ms; after the listed actions: 62.26 ms = 16.1 tok/s per user
e2e measured steps: hybrid dense 2026-10-21 (distinct ids): 115.0 ms, EP32 rs combine 2026-10-22 (distinct ids): 123.4 ms, EP32 op combine 2026-10-21 (distinct ids): 139.2 ms

per module (us per layer):

| layer | module | ops | floor us | measured us | after us | ratio |
|---|---|---|---|---|---|---|
| KDA+MoE | MHC_PRE_ATTN | 18 | 47 | 168 | 8 | 3.6 |
| KDA+MoE | MHC_POST_ATTN | 1 | 11 | 34 | 20 | 3.1 |
| KDA+MoE | ATTN_KDA | 7 | 308 | 426 | 343 | 1.4 |
| KDA+MoE | MHC_PRE_FFN | 18 | 47 | 168 | 8 | 3.6 |
| KDA+MoE | MHC_POST_FFN | 1 | 11 | 34 | 20 | 3.1 |
| KDA+MoE | MOE_ROUTER | 2 | 14 | 32 | 27 | 2.3 |
| KDA+MoE | MOE_HYB_GATHER | 2 | 69 | 117 | 117 | 1.7 |
| KDA+MoE | MOE_HYB_EXPERTS | 6 | 495 | 1198 | 600 | 2.4 |
| KDA+MoE | MOE_HYB_RS | 1 | 68 | 128 | 128 | 1.9 |
| KDA+MoE | MOE_SHARED | 5 | 21 | 34 | 23 | 1.6 |
| KDA+MoE | MOE | 3 | 28 | 32 | 29 | 1.1 |
| KDA+MoE | TRACE | 0 | 29 | 85 | 85 | 2.9 |
| DSA+MoE | MHC_PRE_ATTN | 18 | 47 | 168 | 8 | 3.6 |
| DSA+MoE | MHC_POST_ATTN | 1 | 11 | 34 | 20 | 3.1 |
| DSA+MoE | ATTN_DSA | 50 | 741 | 1928 | 417 | 2.6 |
| DSA+MoE | MHC_PRE_FFN | 18 | 47 | 168 | 8 | 3.6 |
| DSA+MoE | MHC_POST_FFN | 1 | 11 | 34 | 20 | 3.1 |
| DSA+MoE | MOE_ROUTER | 2 | 14 | 32 | 27 | 2.3 |
| DSA+MoE | MOE_HYB_GATHER | 2 | 69 | 117 | 117 | 1.7 |
| DSA+MoE | MOE_HYB_EXPERTS | 6 | 495 | 1198 | 600 | 2.4 |
| DSA+MoE | MOE_HYB_RS | 1 | 68 | 128 | 128 | 1.9 |
| DSA+MoE | MOE_SHARED | 5 | 21 | 34 | 23 | 1.6 |
| DSA+MoE | MOE | 3 | 28 | 32 | 29 | 1.1 |
| DSA+MoE | TRACE | 0 | 49 | 90 | 90 | 1.8 |
| KDA+dense | MHC_PRE_ATTN | 18 | 47 | 168 | 8 | 3.6 |
| KDA+dense | MHC_POST_ATTN | 1 | 11 | 34 | 20 | 3.1 |
| KDA+dense | ATTN_KDA | 7 | 308 | 426 | 343 | 1.4 |
| KDA+dense | MHC_PRE_FFN | 18 | 47 | 168 | 8 | 3.6 |
| KDA+dense | MHC_POST_FFN | 1 | 11 | 34 | 20 | 3.1 |
| KDA+dense | MLP_DENSE | 7 | 133 | 161 | 139 | 1.2 |
| KDA+dense | TRACE | 0 | 24 | 40 | 40 | 1.7 |
| head | EMBED | 1 | 6 | 20 | 20 | 3.4 |
| head | HEAD | 5 | 618 | 632 | 477 | 1.0 |
| head | TRACE | 0 | 3 | 5 | 5 | 1.8 |

ranked actions (ms saved per step at this point):

| rank | action | ms/step | risk | ops | what |
|---|---|---|---|---|---|
| 1 | E1 | 17.60 | M | moe.experts | stream dispatch form (expert_kernel_sweep.md section 10: 779 us at 58 active, weights read once, no dense compute of the non-routed pairs); the down op's expert-parallel split next (~-400 us) |
| 2 | B8 | 13.97 | M | dsa.pool_gather | (a) dense regime (position <= 2050): attend the paged cache directly, no gather; (b) pool-major cache pages: one 4 KB run per pool instead of 32 x 128 B pieces |
| 3 | A1 | 10.45 | M | layer.gaps, mhc.hmix, mhc.ksum, mhc.norm, mhc.proj, mhc.rsqrt_mix, mhc.ssp, mhc.xx | fuse into mhc_pre_block (reads x once for projection + sum of squares + h_mix) |
| 4 | C12 | 7.52 | M (accuracy) | moe.experts | bfp4 gate\|up experts: -1/3 of the expert bytes, eval-gated |
| 5 | A4 | 3.97 | M | mhc.sinkhorn | per-row Sinkhorn kernel (one token row per core, 32 cores) inside the fused pre-block |
| 6 | B5 | 2.11 | M | dsa.absorb_q, dsa.idx_view, dsa.o_view, dsa.q_view, dsa.sdpa_layout, dsa.state_composite | GLM53_DSA_DECODE_STATE=1: one dsa_decode_state program (29 -> 1 op; measured -109 us/layer e2e) |
| 7 | K3 | 1.52 | M | kda.decode | cut the ~49 us fixed cost + 34 us pre-op gap (runtime-arg volume of 80 cores x 3 RISCs: common args + per-core index; kernel prologue); the per-user slope (6.6 us = 303 GB/s) is already at DRAM rate |
| 8 | P1 | 1.26 | L-M | mhc.post_mix | tune the generic op's access pattern (125 GB/s today: resident operands, 8 tiles in flight per core) or keep the streams L1-resident between wraps (D13) |
| 9 | C2 | 0.84 | L | dsa.all_reduce, kda.all_reduce | the MoE all-reduce on the same shape takes 30 us: the extra ~20 us is column skew / input placement (RS waits for the slowest column after kda_decode); B7 gather form measured 46 us (no gain) -> keep ccl; D13 column-sharded mHC would turn it into RS + AG of the streams (neutral) |
| 10 | S1 | 0.65 | L | mlp.glue, mlp.w13, moe.add, moe.shared_glue, moe.shared_w13 | fold the 2 slices + clamped_silu_glu into the matmul epilogue (gate/up column-interleaved blocks) |
| 11 | K2 | 0.31 | M | kda.gate2 | fold the block-diagonal [f_b \| g_b] expansion into kda_decode (per-head [18 x 128] weights in L1) |
| 12 | K1 | 0.24 | L | kda.gate1 | append the 288 gate columns to the qkv weight (one [4096, 6432] matmul at ~390 GB/s) |
| 13 | R1 | 0.20 | L | moe.router_gate | widen the grid (9 -> 36 cores, per_core_N 1 -> the 1D config's 280 GB/s -> ~12 us) |
| 14 | D1 | 0.16 | M (accuracy) | dsa.stem | q_a part in bfp8 (kv_a stays bf16: it is the cached latent): -5.9 MB |

#### 32 users, EP32 (op combine at T=4, e2e 71.2 ms; rs remap 67.68 ms)

layout: 8x4, 4 users/row = 32 users, moe=ep (block), experts bfp8/bfp8, ISL 16, DRAM 400 GB/s (spec 512), links 2 x 25 GB/s (line)

| layer | count | ops | floor_us | floor_spec_us | model_measured_us | ratio | after_us |
|---|---|---|---|---|---|---|---|
| KDA+MoE | 31 | 64 | 868.9 | 686.0 | 1561.2 | 1.8 | 816.4 |
| DSA+MoE | 11 | 107 | 1029.8 | 822.9 | 1955.3 | 1.9 | 922.5 |
| KDA+dense | 3 | 52 | 429.5 | 350.3 | 847.4 | 2.0 | 404.9 |
| head | 1 | 6 | 626.7 | 448.3 | 657.0 | 1.1 | 502.0 |

step floor: 40.18 ms at 400 GB/s (31.82 ms at the spec peaks) -> ceiling 24.9 tok/s per user (31.4 at spec), 796 aggregate; target 30.3 ms
step model (sum of the measured / projected ops): 73.10 ms; after the listed actions: 37.17 ms = 26.9 tok/s per user
e2e measured steps: hybrid 2026-10-26 (A3 sharded norm, B6 ranges, distinct ids): 80.55 ms, hybrid 2026-10-21 (distinct ids): 83.1 ms, hybrid 2026-10-21 (same token on every slot: 2 active experts): 69.8 ms, EP32 op combine 2026-10-21 (distinct ids): 71.2 ms, EP32 rs-remap 2026-10-26 (distinct ids): 67.68 ms

per module (us per layer):

| layer | module | ops | floor us | measured us | after us | ratio |
|---|---|---|---|---|---|---|
| KDA+MoE | MHC_PRE_ATTN | 18 | 47 | 168 | 8 | 3.6 |
| KDA+MoE | MHC_POST_ATTN | 1 | 11 | 34 | 20 | 3.1 |
| KDA+MoE | ATTN_KDA | 7 | 156 | 242 | 169 | 1.5 |
| KDA+MoE | MHC_PRE_FFN | 18 | 47 | 168 | 8 | 3.6 |
| KDA+MoE | MHC_POST_FFN | 1 | 11 | 34 | 20 | 3.1 |
| KDA+MoE | MOE_ROUTER | 2 | 14 | 30 | 25 | 2.2 |
| KDA+MoE | MOE_EP_DISPATCH | 6 | 32 | 77 | 71 | 2.4 |
| KDA+MoE | MOE_EP_EXPERTS | 2 | 402 | 218 | 218 | 0.5 |
| KDA+MoE | MOE_EP_COMBINE | 1 | 68 | 459 | 160 | 6.7 |
| KDA+MoE | MOE_SHARED | 5 | 21 | 34 | 23 | 1.6 |
| KDA+MoE | MOE | 3 | 28 | 33 | 30 | 1.2 |
| KDA+MoE | TRACE | 0 | 29 | 64 | 64 | 2.2 |
| DSA+MoE | MHC_PRE_ATTN | 18 | 47 | 168 | 8 | 3.6 |
| DSA+MoE | MHC_POST_ATTN | 1 | 11 | 34 | 20 | 3.1 |
| DSA+MoE | ATTN_DSA | 50 | 298 | 610 | 249 | 2.0 |
| DSA+MoE | MHC_PRE_FFN | 18 | 47 | 168 | 8 | 3.6 |
| DSA+MoE | MHC_POST_FFN | 1 | 11 | 34 | 20 | 3.1 |
| DSA+MoE | MOE_ROUTER | 2 | 14 | 30 | 25 | 2.2 |
| DSA+MoE | MOE_EP_DISPATCH | 6 | 32 | 77 | 71 | 2.4 |
| DSA+MoE | MOE_EP_EXPERTS | 2 | 402 | 218 | 218 | 0.5 |
| DSA+MoE | MOE_EP_COMBINE | 1 | 68 | 459 | 160 | 6.7 |
| DSA+MoE | MOE_SHARED | 5 | 21 | 34 | 23 | 1.6 |
| DSA+MoE | MOE | 3 | 28 | 33 | 30 | 1.2 |
| DSA+MoE | TRACE | 0 | 49 | 90 | 90 | 1.8 |
| KDA+dense | MHC_PRE_ATTN | 18 | 47 | 168 | 8 | 3.6 |
| KDA+dense | MHC_POST_ATTN | 1 | 11 | 34 | 20 | 3.1 |
| KDA+dense | ATTN_KDA | 7 | 156 | 242 | 169 | 1.5 |
| KDA+dense | MHC_PRE_FFN | 18 | 47 | 168 | 8 | 3.6 |
| KDA+dense | MHC_POST_FFN | 1 | 11 | 34 | 20 | 3.1 |
| KDA+dense | MLP_DENSE | 7 | 133 | 162 | 140 | 1.2 |
| KDA+dense | TRACE | 0 | 24 | 40 | 40 | 1.7 |
| head | EMBED | 1 | 6 | 20 | 20 | 3.4 |
| head | HEAD | 5 | 618 | 632 | 477 | 1.0 |
| head | TRACE | 0 | 3 | 5 | 5 | 1.8 |

ranked actions (ms saved per step at this point):

| rank | action | ms/step | risk | ops | what |
|---|---|---|---|---|---|
| 1 | A1 | 10.40 | M | layer.gaps, mhc.hmix, mhc.ksum, mhc.norm, mhc.proj, mhc.rsqrt_mix, mhc.ssp, mhc.xx | fuse into mhc_pre_block (reads x once for projection + sum of squares + h_mix) |
| 2 | C1 | 6.68 | L | ep.combine_rs | GLM53_EP_RS_REMAP=1 for T < 32 (validated 2026-10-26: 67.68 vs 71.2 ms = -84 us/layer); the RS itself is ~130 us, the rest is the senders' skew |
| 3 | E2 | 5.88 | M-H | ep.combine_rs, ep.experts | placement / replication of hot experts across the column's chips (the skew the combine waits for IS the cost); bfp4 gate\|up |
| 4 | A4 | 3.97 | M | mhc.sinkhorn | per-row Sinkhorn kernel (one token row per core, 32 cores) inside the fused pre-block |
| 5 | B5 | 1.57 | M | dsa.absorb_q, dsa.idx_view, dsa.o_view, dsa.q_view, dsa.sdpa_layout, dsa.state_composite | GLM53_DSA_DECODE_STATE=1: one dsa_decode_state program (29 -> 1 op; measured -109 us/layer e2e) |
| 6 | B8 | 1.50 | M | dsa.pool_gather | (a) dense regime (position <= 2050): attend the paged cache directly, no gather; (b) pool-major cache pages: one 4 KB run per pool instead of 32 x 128 B pieces |
| 7 | P1 | 1.29 | L-M | mhc.post_mix | tune the generic op's access pattern (125 GB/s today: resident operands, 8 tiles in flight per core) or keep the streams L1-resident between wraps (D13) |
| 8 | K3 | 1.19 | M | kda.decode | cut the ~49 us fixed cost + 34 us pre-op gap (runtime-arg volume of 80 cores x 3 RISCs: common args + per-core index; kernel prologue); the per-user slope (6.6 us = 303 GB/s) is already at DRAM rate |

#### 256 users, EP32 rs combine (e2e 123.4 ms)

layout: 8x4, 32 users/row = 256 users, moe=ep (block), experts bfp8/bfp8, ISL 16, DRAM 400 GB/s (spec 512), links 2 x 25 GB/s (line)

| layer | count | ops | floor_us | floor_spec_us | model_measured_us | ratio | after_us |
|---|---|---|---|---|---|---|---|
| KDA+MoE | 31 | 64 | 1275.1 | 972.1 | 2540.2 | 2.0 | 1386.8 |
| DSA+MoE | 11 | 107 | 1727.2 | 1336.4 | 4067.8 | 2.4 | 1486.8 |
| KDA+dense | 3 | 52 | 581.5 | 469.0 | 1031.0 | 1.8 | 578.0 |
| head | 1 | 6 | 626.7 | 448.3 | 657.0 | 1.1 | 502.0 |

step floor: 60.90 ms at 400 GB/s (46.69 ms at the spec peaks) -> ceiling 16.4 tok/s per user (21.4 at spec), 4204 aggregate; target 52.6 ms
step model (sum of the measured / projected ops): 127.24 ms; after the listed actions: 61.58 ms = 16.2 tok/s per user
e2e measured steps: hybrid dense 2026-10-21 (distinct ids): 115.0 ms, EP32 rs combine 2026-10-22 (distinct ids): 123.4 ms, EP32 op combine 2026-10-21 (distinct ids): 139.2 ms

per module (us per layer):

| layer | module | ops | floor us | measured us | after us | ratio |
|---|---|---|---|---|---|---|
| KDA+MoE | MHC_PRE_ATTN | 18 | 47 | 168 | 8 | 3.6 |
| KDA+MoE | MHC_POST_ATTN | 1 | 11 | 34 | 20 | 3.1 |
| KDA+MoE | ATTN_KDA | 7 | 308 | 426 | 343 | 1.4 |
| KDA+MoE | MHC_PRE_FFN | 18 | 47 | 168 | 8 | 3.6 |
| KDA+MoE | MHC_POST_FFN | 1 | 11 | 34 | 20 | 3.1 |
| KDA+MoE | MOE_ROUTER | 2 | 14 | 32 | 27 | 2.3 |
| KDA+MoE | MOE_EP_DISPATCH | 6 | 144 | 117 | 103 | 0.8 |
| KDA+MoE | MOE_EP_EXPERTS | 2 | 545 | 441 | 441 | 0.8 |
| KDA+MoE | MOE_EP_COMBINE | 1 | 68 | 990 | 300 | 14.5 |
| KDA+MoE | MOE_SHARED | 5 | 21 | 34 | 23 | 1.6 |
| KDA+MoE | MOE | 3 | 28 | 32 | 29 | 1.1 |
| KDA+MoE | TRACE | 0 | 29 | 64 | 64 | 2.2 |
| DSA+MoE | MHC_PRE_ATTN | 18 | 47 | 168 | 8 | 3.6 |
| DSA+MoE | MHC_POST_ATTN | 1 | 11 | 34 | 20 | 3.1 |
| DSA+MoE | ATTN_DSA | 50 | 741 | 1928 | 417 | 2.6 |
| DSA+MoE | MHC_PRE_FFN | 18 | 47 | 168 | 8 | 3.6 |
| DSA+MoE | MHC_POST_FFN | 1 | 11 | 34 | 20 | 3.1 |
| DSA+MoE | MOE_ROUTER | 2 | 14 | 32 | 27 | 2.3 |
| DSA+MoE | MOE_EP_DISPATCH | 6 | 144 | 117 | 103 | 0.8 |
| DSA+MoE | MOE_EP_EXPERTS | 2 | 545 | 441 | 441 | 0.8 |
| DSA+MoE | MOE_EP_COMBINE | 1 | 68 | 990 | 300 | 14.5 |
| DSA+MoE | MOE_SHARED | 5 | 21 | 34 | 23 | 1.6 |
| DSA+MoE | MOE | 3 | 28 | 32 | 29 | 1.1 |
| DSA+MoE | TRACE | 0 | 49 | 90 | 90 | 1.8 |
| KDA+dense | MHC_PRE_ATTN | 18 | 47 | 168 | 8 | 3.6 |
| KDA+dense | MHC_POST_ATTN | 1 | 11 | 34 | 20 | 3.1 |
| KDA+dense | ATTN_KDA | 7 | 308 | 426 | 343 | 1.4 |
| KDA+dense | MHC_PRE_FFN | 18 | 47 | 168 | 8 | 3.6 |
| KDA+dense | MHC_POST_FFN | 1 | 11 | 34 | 20 | 3.1 |
| KDA+dense | MLP_DENSE | 7 | 133 | 161 | 139 | 1.2 |
| KDA+dense | TRACE | 0 | 24 | 40 | 40 | 1.7 |
| head | EMBED | 1 | 6 | 20 | 20 | 3.4 |
| head | HEAD | 5 | 618 | 632 | 477 | 1.0 |
| head | TRACE | 0 | 3 | 5 | 5 | 1.8 |

ranked actions (ms saved per step at this point):

| rank | action | ms/step | risk | ops | what |
|---|---|---|---|---|---|
| 1 | E2 | 28.98 | M-H | ep.combine_rs, ep.experts | placement / replication of hot experts across the column's chips (the skew the combine waits for IS the cost); bfp4 gate\|up |
| 2 | B8 | 13.97 | M | dsa.pool_gather | (a) dense regime (position <= 2050): attend the paged cache directly, no gather; (b) pool-major cache pages: one 4 KB run per pool instead of 32 x 128 B pieces |
| 3 | A1 | 10.45 | M | layer.gaps, mhc.hmix, mhc.ksum, mhc.norm, mhc.proj, mhc.rsqrt_mix, mhc.ssp, mhc.xx | fuse into mhc_pre_block (reads x once for projection + sum of squares + h_mix) |
| 4 | A4 | 3.97 | M | mhc.sinkhorn | per-row Sinkhorn kernel (one token row per core, 32 cores) inside the fused pre-block |
| 5 | B5 | 2.11 | M | dsa.absorb_q, dsa.idx_view, dsa.o_view, dsa.q_view, dsa.sdpa_layout, dsa.state_composite | GLM53_DSA_DECODE_STATE=1: one dsa_decode_state program (29 -> 1 op; measured -109 us/layer e2e) |
| 6 | K3 | 1.52 | M | kda.decode | cut the ~49 us fixed cost + 34 us pre-op gap (runtime-arg volume of 80 cores x 3 RISCs: common args + per-core index; kernel prologue); the per-user slope (6.6 us = 303 GB/s) is already at DRAM rate |
| 7 | P1 | 1.26 | L-M | mhc.post_mix | tune the generic op's access pattern (125 GB/s today: resident operands, 8 tiles in flight per core) or keep the streams L1-resident between wraps (D13) |
| 8 | C2 | 0.84 | L | dsa.all_reduce, kda.all_reduce | the MoE all-reduce on the same shape takes 30 us: the extra ~20 us is column skew / input placement (RS waits for the slowest column after kda_decode); B7 gather form measured 46 us (no gain) -> keep ccl; D13 column-sharded mHC would turn it into RS + AG of the streams (neutral) |

#### floor sweep over the users per row (hybrid, measured-routing interpolation)

| users/row | users | step floor ms (400 GB/s) | floor ms (spec peaks) | ceiling tok/s per user (spec) | aggregate ceiling tok/s | active experts per column |
|---|---|---|---|---|---|---|
| 1 | 8 | 24.8 | 20.7 | 40.3 (48.3) | 323 | 11 |
| 2 | 16 | 27.7 | 23.0 | 36.0 (43.6) | 577 | 18 |
| 4 | 32 | 32.3 | 26.4 | 31.0 (37.9) | 992 | 28 |
| 8 | 64 | 37.7 | 30.5 | 26.5 (32.8) | 1699 | 38 |
| 16 | 128 | 45.0 | 36.0 | 22.2 (27.8) | 2845 | 48 |
| 32 | 256 | 55.7 | 43.7 | 18.0 (22.9) | 4600 | 58 |

#### design variants of the floor (same op graph unless named)

| variant | 32u floor ms | ceiling tok/s/u | 256u floor ms | ceiling tok/s/u |
|---|---|---|---|---|
| current op graph, bfp8 experts | 32.3 | 31.0 | 55.7 | 18.0 |
| bfp4 gate\|up experts | 29.2 | 34.3 | 49.3 | 20.3 |
| bfp4 gate\|up + down | 27.6 | 36.2 | 46.1 | 21.7 |
| A1 fused pre-block | 28.3 | 35.3 | 51.7 | 19.3 |
| B5 DSA state fused (+ q/o folds) | 32.2 | 31.0 | 55.6 | 18.0 |
| A1 + B5 + bfp4 gate\|up | 25.2 | 39.7 | 45.3 | 22.1 |
| uniform routing (worst column) | 38.8 | 25.8 | 60.6 | 16.5 |
| rated links 50 GB/s | 31.5 | 31.7 | 53.3 | 18.8 |
| ring row axis | 31.3 | 32.0 | 53.0 | 18.9 |
| DRAM at the 512 GB/s spec | 27.1 | 36.9 | 46.0 | 21.7 |
| EP32 layout | 40.2 | 24.9 | 60.9 | 16.4 |

#### floor composition per step (ms, hybrid)

| term | 32 users | 256 users |
|---|---|---|
| routed experts (active slices) | 10.3 | 20.8 |
| kda_decode state | 0.8 | 6.0 |
| mHC pre + post (2 wraps) | 5.2 | 5.2 |
| projection matmuls (qkv, o_proj, gates, stem, q_b, idx, shared, mlp) | 7.0 | 7.0 |
| row collectives (gather x2 + RS) | 2.5 | 5.8 |
| TP all-reduces | 2.4 | 2.4 |
| DSA state (pool gather + SDPA + index) | 1.1 | 6.0 |
| LM head + embedding | 0.6 | 0.6 |
| layout glue + launches + gaps | 3.6 | 3.1 |

## 5. The next 10 actions, ranked by ms saved per step (32 users unless noted; 256-user value in parentheses)

| rank | id | action | ms/step 32u (256u) | risk | mechanism | validation |
|---|---|---|---|---|---|---|
| 1 | E0 | `GLM53_STREAM_EXPERTS=1` for decode at T <= 32: the expert_stream op (341 us at 28 active) instead of the sparse tuned block (683 us, 13 ops) that the no-knob e2e path runs (`tt/moe.py`: `stream` only with the knob; the 83.1 ms e2e = 68.7 ms of modelled ops + 42 x 0.34) | 13.9 (0) | L | knob; the op is validated (PCC 0.99997, 3 device ops) | layer-4 tracy of the 8x4 hybrid at 4 users/row (none exists), then the 32-user e2e |
| 2 | A1 | fused mHC pre-block (`GLM53_MHC_PRE_BLOCK=1`): 18 ops / 168 us -> 1 op / 101 us per wrap, measured; 22-layer chain 37.7 -> 33.3 ms | 10.4 (10.5) | M | fuse (custom op exists; mesh PCC 0.70 of 2026-10-21 audited as not a NOC hazard, idle-core gather race fixed) | `test_mhc_pre_block_mesh.py` bit-level vs the composite on all 8 block chips, the 45-layer chain, the 8x4 logits gate |
| 3 | E1 | expert op: fold the selector matmul + x copy into the op, pipeline the DST fill / pack per expert (the ~110 us fixed cost); at 256 users the dispatch form instead of the dense layout (1198 -> 779 us) and the down op's expert-parallel split | 4.2 (17.6) | M | tune the custom op (kernel work) | `test_moe_block.py` / `test_expert_kernel_perf.py` chip test at 28 / 58 active; 8x4 e2e |
| 4 | H2 | hybrid combine at T < 32: compact [rows x T, H] expert output (256 KB) + `all_reduce_rows`, each row slicing its T rows, instead of the padded [256, 4096] reduce-scatter + 9 pad / slice ops (162 -> ~50 us) | 4.7 (0) | M | fuse / replace a CCL on 8x fewer bytes | `test_mesh_moe.py` ep/hybrid vs HF on the block, `test_ccl_perf.py` all_reduce_rows |
| 5 | A4 | per-row Sinkhorn: one token row per core (32 cores) instead of ONE core for the whole tile (52 -> ~8 us per wrap); needs a kernel change (the batched reference graph is +250 us) | 4.0 (4.0) | M | tune inside the fused pre-block | bit-level vs `mhc_split_sinkhorn` per row |
| 6 | H1 | hybrid gathers at T < 32: one `all_gather_async` of the compact row-major [T, H] (32 KB) and of the top-k lists instead of all_broadcast + 8 untilize / concat / tilize ops on the padded tiles (120 -> ~50 us) | 2.9 (0) | L-M | fuse (erase 8 layout ops; CCL-latency-bound floor 31 us) | block gather test, 8x4 e2e |
| 7 | C12 | bfp4 gate\|up experts (`GLM53_EXPERT_W13_DTYPE=bfp4`): -1/3 of the expert bytes (floor 32.3 -> 29.2 ms / 55.7 -> 49.3) | 2.9 (7.5) | M (accuracy) | dtype | `tests/test_expert_dtype.py` + the eval gate |
| 8 | B5 | `GLM53_DSA_DECODE_STATE=1` + `GLM53_DSA_FOLD=q`: 29 state ops -> 1 program, free head-major views (measured -1.2 ms per step at 64 K; the o fold is roofline-negative) | 1.6 (2.1) | M | fuse (custom op exists; pool-block diagnostic open) | `test_dsa_decode_state_mesh.py` chip 0 + block, 64 K needle |
| 9 | B8 | paged_pool_gather: (a) at positions <= 2050 attend the paged cache directly (no gather, no compact copy); (b) pool-major pages (one 4 KB run per pool instead of 32 x 128 B pieces) for long contexts | 1.5 (14.0 proj) | M | (a) second captured trace per regime; (b) layout change in the gather + state writers | `test_dsa_sparse` dense positions bit-equal; `test_paged_pool_gather.py` golden |
| 10 | P1 | mhc_post_mix access pattern: 4.5 MB in 34 us = 125 GB/s on 120 cores (resident operands, 8 tiles in flight per core) -> ~20 us | 1.3 (1.3) | L-M | tune the generic op | tracy row 34 -> 20, bit-level vs today |
| 11 | K3 | kda_decode fixed cost (48.6 us) + the 34 us pre-op gap: common runtime args + per-core index instead of 80 x 3 per-core arg vectors, kernel prologue; the per-user slope is at DRAM rate | 1.2 (1.5) | M | tune / dispatch | tracy gap column before the op; `test_kda_decode.py` golden |
| 12 | C2 | attention TP all-reduce 51 -> 30 us (column skew / input placement; the MoE one on the same shape runs at 30) | 0.8 (0.8) | L | CCL input memory config, column skew diagnosis | `test_ccl_allreduce_gather.py`, tracy per chip |

Sum of 1-12 at 32 users: ~50 ms of the 82.6 ms model -> ~31 ms (32.6 tok/s/u; the floor is 32.3 ms before A1 / C12
lower it). At 256 users the ranked items (E1, B8, A1, C12, A4, B5, K3, P1 = 58 ms) take the model from 123.5 to ~62 ms;
the 52.6 ms target then needs the DSA state terms at their floors (SDPA / index on the paged cache, bfp8 pooled keys) and
the row collectives at the CCL fit (2x today): see the design-variant table (A1 + B5 + bfp4 floor 45.3 ms).

EP32 (`GLM53_MOE_LAYOUT=ep`): C1's T < 32 remap (validated 67.68 ms) is worth 6.7 ms over the op combine; the next EP item
is the column's expert skew (E2, 5.9 ms: the combine waits for the slowest of 8 senders; placement / replication), after
which EP32's floor (40.2 ms at 32 users) stays above the hybrid's (32.3) because its slowest chip reads 6 full experts.

## 6. Items already landed or measured, as the tables reflect them

| item | status | in the tables |
|---|---|---|
| A3 sharded mHC input norm | default `GLM53_MHC_NORM=sharded` since 2026-10-26 (chip-0 A/B PCC 0.999998, traced wrap 246 -> 210 us; 8x4 e2e 80.55 ms) | `mhc.norm` = 4 ops / 14 us (was typecast 3.3 + 1-core rms_norm 46.4); the pre-block is 18 ops / 168 us |
| A1 fused pre-block | measured 203 -> 101 us per wrap on one chip; 22-layer EP chain 37.7 -> 33.3 ms; default off (mesh A/B pending) | `--mhc-pre-block` swaps the 18 rows for `mhc.pre_block_fused` (101 us; floor 10.7); ranked #2 |
| B5 DSA decode state program | measured -1.2 ms per step at 64 K (11 x ~109 us); default off (pool-block diagnostic) | `--dsa-state-fused`: 29 rows -> `dsa.state_fused`, free views, q/o folds; ranked #8; the o fold flagged roofline-negative |
| C1 reduce-scatter EP combine | default at T % 32 == 0 (139.2 -> 123.4 ms at 256 users); the T = 4 remap validated 2026-10-26 (71.2 -> 67.68 ms) | EP rows `ep.combine_rs`: 459 us op path at 32 users (after C1 300), 990 us e2e-derived stage at 256 users incl. the senders' skew |
| B7 gather-form TP all-reduce | 46 vs 51 us: no gain at 4 users/row; default stays `ccl` | `kda.all_reduce` / `dsa.all_reduce` keep the ccl composite; C2 notes the 20 us of skew |
| B6 rectangular core ranges | kept; e2e 83.08 vs 83.07 ms (no change); the 34 us gap before kda_decode remains | `kda.decode` gap 34.2 us; K3 targets it |
| fused router (`GLM53_ROUTER_FUSED=1`) | default since 2026-10-21 (17 ops / 168 us -> 2 ops / 30 us) | `moe.router_gate` 17.8 + `moe.router_topk` 12.1 us |

## 7. Caveats and the measurements that would tighten the model

* Every measured per-op value is chip (0,0) of one traced replay; CCL rows absorb the row / column skew (the EP32 combine
  stage at 256 users is derived from the e2e deltas, not profiled); the step waits for the slowest chip.
* No 8x4 hybrid profile at 4 users/row exists: the hybrid T = 4 MoE rows (gathers 120 us, experts 683 / 351 us, RS 162 us)
  are the block's glue + the sweep's chip-level expert measurements; the e2e arithmetic (83.1 ms vs the 68.7 ms model with
  the stream op) says the no-knob path runs the sparse block. One layer-4 tracy run settles E0, H1 and H2.
* No DSA profile at 32 users/row: paged_pool_gather is projected linear in users (39 us per user; 1270 us at 32), the
  flash-MLA decode at max(73 us, its DRAM floor); these two rows are 14 ms of the 123.5 ms model, and the e2e (115.0) says
  the projection is ~8 ms pessimistic (the gather must scale sub-linearly past 8 users as more cores engage).
* The CCL fits are the report's Blackhole line fits at 25 GB/s per link with 2 links; the model's reduce-scatter / all-gather
  run at 2x those fits, so the "CCL floor" may be optimistic for `*_minimal_async` as configured (chunks_per_sync 10, 2
  workers per link).
* Dense layers 0-2 and the LM head are modelled from the matmul sweep and the byte counts (no profile); together 3.2 ms of
  the step at either point.
* The compute floors use the full 120-core chip; decode-shaped matmuls are DRAM-bound by 5-30x, so only the dense expert
  layout at 256 users (116 GFLOP per layer per chip) is compute-relevant.
