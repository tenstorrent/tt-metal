# Prefill routed experts through ttnn.experimental.deepseek_prefill (DSV41_PREFILL_MOE=unified, default OFF)

Files: `tt/prefill_unified_moe.py` (pipeline, weights, mapping), `tt/prefill_layer.py` (`_moe_unified`, used by the column-split layer only), `tt/dsv41_model.py`
(`UNI_MOE`, `UNI_LAYERS`), `tests/test_unified_moe.py` (standalone single-layer test), `tools/build_unified_cache.py`, `tools/devrun_pf.sh`.

## Mapping (mesh 4 rows x 8 columns, x replicated over the columns, every row its own users)
* Dispatch group axis = mesh ROWS (cluster_axis 0, `dispatch_group_size` 4, `seq_len_per_chip` = tokens of the row in the chunk = U*C).
* The 8 COLUMNS are 8 independent dispatch groups of 48 experts each (12 per chip): expert e -> group (column) e // 48, chip (row) (e % 48) // 12, local slot e % 12
  (`ExpertMapping`, column-major, models/demos/deepseek_v3_d_p). This is the layout the research report suggested; the standalone test checks it against the moe_compute path
  and a torch reference.
* Every column routes the same tokens; dispatch / combine only move tokens between the 4 rows of a column, `post_combine_reduce` keeps the experts of the column's own group,
  the 8 partial sums are added by a reduce-scatter over the columns (see "Findings" for why it is over the hidden dim + all_to_all, not over tokens).

## Layer flow (column-split layer, per chunk and layer; N = U*C tokens per row, n = N/8 own tokens per column)
1. own normed hidden rows hh_own [1,1,n,D] (own 32-token chunks of all groups, concatenated)
2. router on the own rows only (per 32-row slice, as before), routing (expert ids + weights) packed in fp32 row-major pages of 384 values and all-gathered over the columns
3. all_gather of hh_own over the columns -> [1,1,N,D] in row order [column][own chunk][32] (dispatch does not care about the order)
4. masked_bincount + offset_cumsum -> dispatch (RM bf16) -> ONE `unified_routed_expert_moe` (ClampedSiluGlu, limit 10, 12 local experts, bf8 weights) -> combine -> post_combine_reduce
5. ttnn.reduce_scatter over the hidden dim (columns) + all_to_all_async_generic hidden-shard -> own tokens -> [1,1,n,D], sliced into the own 32-token chunks for the mHC expand.
Shared expert, mHC, attention, Engram are untouched.

## Numerics
The op is fixed: LoFi, bf8 activations in, bf8 intermediates and bf8 output (compute_kernel_config fidelity / fp32 acc are ignored by the op, measured: identical results and time).
It applies the swiglu clamp (|up| <= 10, gate <= 10) that the moe_compute path does not (moe_compute has no clamp).

## Findings / pitfalls (all reproduced in tests/test_unified_moe.py with DSV41_UM_DEBUG=1)
* reduce_scatter_minimal_async over the TOKEN dim of the big partial-sum tensor (also in 256- and 512-row pieces) corrupts some 32-row blocks (nondeterministic positions).
  The generic `ttnn.reduce_scatter(dim=hidden)` + `all_to_all_async_generic` is exact at N = 256 / 512 / 2048 / 4096 rows (N = 8192 only ran in the 6-layer traced test, not compared).
* all_gather of tensors that are <= 1 tile wide (routing [n, 12], tile or row-major pages of 12-48 B) drops blocks of rows; gathering 384-value row-major pages is exact.
* DRAM: the unified weights are a second copy of the bf8 expert weights (497 MB / chip / layer, 19.9 GB / chip for 40 layers); the moe_compute decode weights are the same size, so
  decode + unified prefill for all 40 layers does NOT fit a 32 GB chip next to KV and the rest. DSV41_UNI_NODECODE=1 skips the moe_compute weights (prefill-only runs);
  DSV41_UNI_LAYERS=2-9 limits the unified path to a layer subset (decode intact).

## Measurements (BH Galaxy 4x8, bf8 experts, links=2, own-row routing)
Single layer, standalone test (layer 3, random normal hidden, tokens per mesh row N, MoE section of one layer = router + gathers + experts + reduce):
| N per row | moe_compute path (T=256 per call) | unified | PCC vs torch clamp (first 16 tokens) |
| 256 | 2.47 ms | 2.42 ms | 0.99976 (baseline 0.99998) |
| 512 | 4.88 ms | 2.71 ms | 0.99976 |
| 2048 | 23.1 ms | 5.8 ms (links=1) | 0.99976 |
| 4096 | 46.3 ms | 7.7 ms (links=1: 10.0) | 0.99976, 0 bad 32-row blocks vs the baseline output |
| 8192 | -- | 19.2 ms (links=1 only measured) | |
Stage times at N=4096, links=2: bincount+cumsum 0.35, dispatch 1.5, experts 1.96, combine 1.07, post_combine_reduce 0.86, router (own rows) 0.7, gathers + reduce-scatter + all_to_all ~2 ms.

Whole model, 40 layers, prefill only (DSV41_PREFILL_MOE=unified DSV41_UNI_NODECODE=1 DSV41_PREFILL_ONLY=1), TTFT of the demo (second call) / total tok/s:
| scenario | baseline (grid) | unified |
| isl4k_b16 (ISL 3720) | 20.17 s, 2951 tok/s | 11.99 s, 4965 tok/s |
| isl8k_b4 (ISL 7443) | 10.99 s, 2708 tok/s | 6.55 s, 4547 tok/s |
| isl32k_b8 (ISL 30059) | 77.2 s, 3114 tok/s | 44.8 s, 5364 tok/s |
The replay loop alone: 18.77 -> 10.61 s, 10.88 -> 5.49 s, 74.2 -> 43.4 s.
Chunk size (6 layers, U=4): replay per row token is flat from 512 to 2048 tokens per user (0.113-0.115 ms for unified, 0.205-0.210 ms for the moe_compute path): a bigger chunk does not help once the experts see >= 256 rows each.

## Unified prefill MoE next to the decode weights (pf_unifit)
Both expert copies are fully sharded over all 32 chips (384 experts, 12 per chip, no replication): decode moe_compute copy = 497 MB/chip/layer
(ring layout [8 cores, L, E, groups, K, 4 tiles], N 9 -> 10 tiles padding), unified copy = 451 MB/chip/layer (12 x 3 tensors of 5120x2304 bf8; 56.9 MiB/bank
measured: 0-3 layers 295.8 -> 523.2 MiB/bank). 40 layers: 2371 MiB/bank (decode) + 2274 MiB/bank (unified) vs ~3880 MiB/bank of DRAM. Re-sharding cannot help (already 1/32 per chip);
only ONE shared copy (the unified op reading the decode ring layout, or moe_compute reading the unified layout: a kernel change) fits all 40 layers.
DSV41_UNI_LAYERS=auto (DSV41_UNI_RESERVE_MIB, DSV41_UNI_MAX): hybrid, unified weights added after the model is built for as many layers as fit. tools/memtab_grid.py: grid DRAM headroom table.

## ONE weight copy: the unified op reads the decode ring weights in place (DSV41_UNI_RING=1, pf_onecopy)
`unified_routed_expert_moe` has a RING_WEIGHTS mode (selected by the weight tensors: rank-6 DRAM HEIGHT_SHARDED = the moe_compute `w0_w1` / `w2`, passed as the same tensor
EPC times in the gate/up/down lists). Host: `unified_routed_expert_ffn_device_operation.cpp` (validation), `..._program_factory.cpp` (GRID_X = 8 = one N column per ring core /
DRAM bank, per_core_N 9 (gate/up) and 20 (down), N inferred from the stored shapes, `RING_WEIGHTS` define). Kernels (JIT): `kernels/weight_runs.hpp` `RingWeights`
(gate/up: per-tile scatter reads of page ((((c*E+e)*G+g)*R+k)*4+t), t = [gate n0, up n0, gate n1, up n1]; down: one 4-tile read per group, K rows of a core stored per 9-row chunk
rotated by the core id, chunk ch at stored chunk (c - ch) mod 8). Needs 8 live DRAM banks (validated). Decode reads nothing different: the buffers are only read.
Build: `tools/oc_build.sh` (private relink of only this op's unity TU against the main build's flags/objects into pf_onecopy_build/lib; run with LD_LIBRARY_PATH=<that>/lib, `tools/oc_run.sh`
does it). Test: `DSV41_UM_RING=2 pytest tests/test_unified_moe.py` (1: ring only, 2: ring + the copy to compare), `DSV41_UM_DECODE=1` checks the decode block before/after.
Layer 3 (real FFN inputs of the S=128 dump, N=512): PCC ring vs unified-copy 0.99985-0.99994, vs moe_compute baseline 0.99985-0.99987, vs torch 0.99981 (copy: 0.99976).
Time per layer (router + gathers + experts + RS), per row N tokens: 512: ring 3.45 ms / copy 2.71 / moe_compute 6.29 (2 calls); 2048: 5.45 / 4.73; 4096: 8.52 / 7.72 / 38.5 (16 calls).
Experts stage at 4096: 2.77 ms (copy 1.95; 64 instead of 88 cores). Decode block T=32 before/after the ring ops: 0.912 / 0.913 ms, output bit-identical.
40 layers, no second copy, decode weights present (DSV41_UNI_RING=1, no NODECODE), scenario 4096:1024 U=4: first-token logits PCC vs the CPU dump 0.9614, argmax 16/16 (moe_compute baseline 0.9655, 16/16).
Not done / ideas: DSV41_RING_GRID_Y=10 (80 cores) hangs; gate/up are single-tile reads (issue-bound at few tokens per expert, the 512 gap); 7-bank chips need uneven per-column N.

## Default ON (unified prefill MoE + ring weights) and the C++ rebuild procedure
Policy (tt/uni_policy.py): `DSV41_PREFILL_MOE` unset / `auto` / `unified` = unified prefill MoE reading the decode ring weights in place (`DSV41_UNI_RING`, default 1); `DSV41_PREFILL_MOE=off`
(or 0 / none / baseline / moe_compute) = the old moe_compute prefill path. The model logs ONE line and uses the old path when the loaded _ttnncpp has no ring mode (it scans the mapped library),
the chip has no 8 live DRAM banks, or (U=1) `uni_policy.U1_DEFAULT_ON` is False and DSV41_UNI_U1 != 1; per chunk `colsplit_active(U, C)` (U*C % 256 and, for the unified path, any U) decides
whether the layers use the unified op or their moe_compute path. With the unified path the column split also applies to U=8 (no DSV41_MOE_G=8 needed): the unified op does not use the grouped
moe_compute program that hangs at 8 users/row. Chunk budget without DSV41_PREFILL_ROW_TOKENS: `chunk_rule.auto2_budget(U, unified=True)` (UNIFIED_TABLE {1:1024, 2:2048, 4:8192, 8:1024, 16:2048}, U>16: 128*U);
U=2/8 entries are conservative guesses. `DSV41_UNI_NODECODE=1` (prefill-only measurement mode) implies the copy path (no ring).

### Rebuilding the op after a C++ change (no full build; ~25 s, safe for running jobs)
The op (`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn`) is ONE unity translation unit of `_ttnncpp.so`; its kernels are JIT (a kernel .cpp edit needs no build at all).
1. Never run `ninja` in build_Release (it re-runs CMake and rebuilds everything stale).
2. `tools/oc_main_rebuild.sh <tree> <tag>` replays the op's compile and the _ttnncpp link command ninja recorded (`ninja -t commands`, saved in /mnt/tt-data/ssinghal/wt/pf_onecopy_build/oc_{compile,link}_cmd.txt),
   writes the new object / library under a temp name in the same directory, checks the new library contains the ring-mode marker, hard-links the old files as `*.pre_<tag>` (backup), and renames the new ones
   over `build_Release/lib/_ttnncpp.so`, `build_Release/ttnn/_ttnncpp.so` and the unity .o (rename = new inode: running processes keep the old library mapped, a new process sees the old or the new complete file).
3. The python bindings (`ttnn/ttnn/_ttnn.so`) only need relinking when `unified_routed_expert_ffn_types.hpp` / the nanobind TU change (they did not here); `import ttnn` works unchanged.
4. Private variant for development (does not touch the main build): `tools/oc_build.sh` builds from a worktree into pf_onecopy_build/lib; run with `LD_LIBRARY_PATH=/mnt/tt-data/ssinghal/wt/pf_onecopy_build/lib`.
5. Verify with the one-layer test from the tree only: `DSV41_UM_RING=2 DSV41_UM_REAL=1 DSV41_UM_N=512 pytest models/demos/blackhole/deepseek_v41_flash/tests/test_unified_moe.py` (ring vs copy PCC 0.9998+).
Done for the main build on 2026-10-06 (old library kept as build_Release/lib/_ttnncpp.so.pre_ring1); verified on host .34 with only the main build (PCC ring vs copy 0.99985-0.99994, 3.72 ms/layer at N=512).
## MoE collectives: fused / overlapped alternatives (pf_moeoverlap; flags default OFF)
Survey result (BH, this build): fused AG+matmul (`all_gather_minimal_matmul_async`, `strided_all_gather_minimal_matmul_async`) only run on the 1D fabric (they hard-code
`LowLatencyPacketHeader`; llama_mlp.py comment still true), `matmul_reduce_scatter_async` is tested on a 1x4 box only, `minimal_matmul_strided_reduce_scatter_async` races on BH
(gpt_oss_d_p gates it off), `deepseek_moe_reduce_scatter` / `moe_gpt` / `selective_reduce_combine` are decode-shaped (TG tests skipped on BH), `post_combine_reduce` fuses only the
top-k weighted sum (DSV3 `TtReduceModule` = pcr + a separate hidden-dim `reduce_scatter`, i.e. exactly our path), fp8 dispatch/combine exist (BH only) but `post_combine_reduce`
requires a bf16 combine output. DSV3 hides the shared expert behind dispatch on a one-row sub-device (`overlap_shared_expert_with_dispatch`, `SubDeviceTraceController`).
Standalone single layer (layer 3, ring weights, links=2), ms:
| stage | N=4096 random | N=4096 real FFN input (layer 3, one 4096-token user, rolled per row) | N=512 |
| all_gather hidden / to RM | 0.48 / 0.22 | same | 0.08 / 0.03 |
| router (16 slices) + packed routing AG | 0.85 | 0.85 | 0.18 |
| bincount+cumsum / dispatch / experts / combine / pcr | 0.13 / 1.51 / 2.79 / 1.04 / 0.85 | 0.13 / 3.93 / 6.06 / 3.14 / 0.85 | 0.12 / 0.27 / 2.44 / 0.33 / 0.18 |
| reduce_scatter(hidden) + all_to_all | 0.48 + 0.17 | same | 0.09 + 0.03 |
Real tokens double the dispatch / experts / combine times (hot experts: in the layer-3 dump one expert receives 63x the mean token count; chip load max/mean ~5, a greedy re-mapping of
experts to chips only gets it to ~4.7): the "collectives 4-5x above floor" are mostly hot-expert skew (ingress of the hot chip), not fabric inefficiency.
Not working on this build / no gain: one `all_to_all(in_dim=tokens, out_dim=hidden)` + local sum instead of reduce_scatter + all_to_all gives WRONG data (PCC 0.001) and is 8x slower
(1.39 ms); untilize before the all-gather (all-gather of row-major pages) 1.32 ms vs 0.71 ms; `num_workers_per_sender` 1-4: no change; dispatch/combine `Topology.Ring` over the 4 rows:
output identical, 6.21 vs 6.32 ms (DSV41_UNI_TOPO=ring, -0.11 ms, not enabled in the A/B).
Implemented:
* `DSV41_UNI_ROUTER=batched`: one router for all own rows (`router_select` extended to any T % 32 == 0: rows strided over the cores, JIT kernel only): bit-identical expert ids
  and weights, router + packed AG 0.85 -> 0.31 ms per layer at N=4096 (0.18 -> 0.17 at N=512).
* `DSV41_MO_OVERLAP=1` (`DSV41_MO_OVERLAP=prep` builds the split weights only; tt/moe_overlap.py): shared expert on a Tensix sub-device (rows 1..9) concurrent with `dispatch` on
  sub-device 0 (row 0), DSV3 style. Needs separate gate / up bf8 weights (+24 MB per chip per layer, split from w01 at build), explicit 2D matmul configs on the 12x9 sub-device grid,
  `sub_core_grids` for the GLU multiply, and a SEGMENTED chunk trace (a sub-device manager cannot be loaded / cleared inside a capture): `SegTrace` captures the chunk forward as
  81 traces split at the load / clear points. Single layer: output bit-identical (MoE partial `torch.equal`), shared-expert PCC 1.0000; section 7.74 -> 6.39 ms (N=4096 random),
  15.51 -> 14.16 (real tokens), 5.01 -> 4.34 (N=2048), 3.54 -> 3.38 (N=512).
In-process A/B session modes: `DSV41_SESSION=id@a,id@b,id@c,id@a` with `DSV41_MODE_A/B/C="K=V,K=V"` (tools/mo_run40.sh).
Real-token N=4096 (hot experts): dispatch/combine `Topology.Ring` over the 4 rows (DSV41_UNI_TOPO=ring, identical output): MoE section 14.10 -> 12.14 ms; with shared||dispatch 12.19 ms
(vs 15.51 sequential), plus the batched router -0.54 ms.
40 layers, prefill only, in-process A/B (a = unflagged, b = batched router, c = b + shared||dispatch, d = c + ring), replay device time per call / TTFT, first tokens identical in all modes:
| scenario | a | b | c | d |
| isl4k_b16 (3720) | 9.61 s / 10.70 s | 9.53 / 10.53 | 9.31 / 10.37 | 9.20 / 10.29 (second session: a 9.61-9.50 / 10.64) |
| isl8k_b4 (7443) | 4.75 s / 6.18 s | 4.71 / 6.70* | 4.60 / 6.02 | not run |
(* host-side noise; replay time is the comparable number.) 6-layer traced chunk logits: bit-identical to the unflagged run for c and d.

### MoE overlap re-measured on main df4ffc9f838 (ported as 62a893e7838 / 6fca7020e68 / 01c49e743da / 5c09e3dbab2 on ssinghal/dsv4p1-pf-overlap2; flags default OFF)
Grid env (no MEMLOG, SPEC=0, 40 layers), in-process A/B via tools/mo_grid.sh, modes a=base, b=batched router, c=b+shared||dispatch, d=c+ring topology, e=ring topology only. TTFT ms, 4k (3720 tokens):
* B=16 .43 (logs/mogrid_b16_h43.log): a 8755 / 8753 (first a 10626 = cold), b 8715, c 8604, d 8501 / 8474, e 8634  -> d -3.0%, c -1.7%, e -1.4%, b -0.4%. Outputs identical in all modes (16 users).
* B=32 .44 (logs/mogrid_b32_h44.log): a 17495 / 16940, c 16589, d 16361 -> c -3.6%, d -4.9% vs mean(a). BUT in c and d the 32 identical prompts no longer give identical outputs (row groups of 8 users diverge at the 2nd generated token; a: all 32 equal): not bit-identical at U=8 per row. Cause not investigated.
Long ISL not run (scope: 4k only until a confirmed gain).
