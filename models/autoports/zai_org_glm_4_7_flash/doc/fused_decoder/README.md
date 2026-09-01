# GLM-4.7-Flash fused decoder — stage report

Target: `zai-org/GLM-4.7-Flash` (`Glm4MoeLiteForCausalLM`), single Blackhole
p150-class chip (device 0, 1x1 mesh), branch `ttmodelmanager/glm47-flash-probe`.

Implementation: `models/autoports/zai_org_glm_4_7_flash/tt/fused_decoder.py`
(`FusedDecoder`, a graph-fused subclass of `FunctionalDecoder` with the same
public prefill/decode/paged-cache contract).
Tests: `tests/test_fused_decoder.py` (all against `FusedDecoder`),
`tests/test_fused_perf.py` (before/after perf), `tests/test_long_context.py`
with `GLM47_DECODER=fused`.
Work log: `work_log.md`.

## Headline results (same session, functional -> fused, synthetic weights, bf16 + bf8 experts)

| window | functional | fused | delta |
|---|---|---|---|
| moe traced decode, batch 1, ctx 1024 (wall) | 1.532 ms/tok | **1.035 ms/tok** | -32.4% |
| dense traced decode, batch 1, ctx 1024 (wall) | 1.008 ms/tok | **0.969 ms/tok** | -3.9% |
| moe warmed prefill S=2048 (wall) | 268.3 ms (7635 t/s) | **210.0 ms (9752 t/s)** | -21.7% |
| dense warmed prefill S=2048 (wall) | 19.5 ms (105250 t/s) | **15.3 ms (133477 t/s)** | -21.3% |
| moe decode device time (tt-perf-report) | 1166.9 us/step | 787.3 us/step | -32.5% |
| dense decode device time | 923.5 us/step | 890.4 us/step | -3.6% |
| moe prefill device time | 268.0 ms | 209.8 ms | -21.7% |
| dense prefill device time | 18.9 ms | 14.9 ms | -21.2% |
| moe decode ops per step | 66.4 | 61.2 | |
| moe 202751-token prefill | 95.7 s (2119 t/s) | 90.3 s (2246 t/s) | -5.6% |

Wall-clock JSON: `perf_wallclock_{func,fused}_{prefill,decode}_{moe,dense}.json`
(functional baseline re-measured in the same session as the fused runs).
tt-perf-report tables/CSV: `tracy/{moe,dense}/{prefill,decode}_{func,fused}_perf_report.{txt,csv}`
plus stacked CSV/PNG, all filtered by the signpost windows
`PERF_{PREFILL,DECODE}_{MOE,DENSE}_{FUNC,FUSED}`. The raw full-session ops CSV
(`tracy/ops_perf_results.csv`, 15 MB) is disk-only (repo 500 KB commit limit);
the committed per-window CSVs carry every measured row.

## Applied rewrites (all PCC-verified on device, kept only when measured faster)

Dedicated fused ops:
1. **Indexed/gather `ttnn.sparse_matmul` for batch-1 decode MoE**: the
   `ttnn.topk` expert ids become a device-resident uint16 index list; the
   expert matmuls compute ONLY the token's top-4 experts and emit compact
   `[1, 4, B, N]` tensors instead of dense `[1, 64, B, N]`. Routing weights
   come from `ttnn.gather` at the same ids (scatter-mask construction
   deleted). Trace-capturable: static shapes, ids read on device (op's
   program-cache test proves re-dispatch with new ids).
2. **`ttnn.experimental.nlp_concat_heads_decode`** replaces the decode output
   head concat (transpose -> untilize -> reshape -> tilize). Needs a
   single-rectangle input core range on Blackhole's 13-wide grid (a row-wise
   32-core set splits into 3 ranges and trips the op's subcoregrid mode).
3. **`ttnn.transformer.concatenate_heads`** replaces the prefill output head
   permute -> reshape.
4. **`ttnn.experimental.slice_write`** replaces the rolling `ttnn.concat`
   chunk accumulator in `prefill_forward` and in the MoE down-split loop
   (the old accumulator re-copied the whole prefix once per chunk; at 99
   chunks / 202k tokens that was O(n^2) traffic).

Graph rewrites:
5. **Fused `wqkv_a` matmul** (shared-LHS): `wq_a` and `wkv_a` concatenated
   host-side; one matmul + width slices (deepseek_v3 mla1d idiom). Measured
   65 -> 44 us traced for the pair at decode.
6. **Packed routed `gate_up` weight** `[1, E, 2048, 3072]`: one sparse matmul
   instead of two; the in0 multicast per (block, expert) pair runs once.
   Measured: decode indexed gate/up 180 -> 144 us; prefill grouped gate/up
   84.9 -> 68.1 ms per 1024-token chunk (both including the added slices).
7. **Per-head `wq_b` for prefill** `[1, nh, 768, 256]`: the broadcast-batched
   matmul emits `[1, nh, S, qk_head]` directly, deleting the
   reshape + permute head split (~3.5 ms per 1024-token chunk).
8. **Real per-block union sparsity in prefill MoE**: gate/up receive the max
   of the 32-token block's routing weights instead of all-ones; the down
   matmul receives the per-chunk expert union. Exact because non-selected
   experts have exactly zero routing weight (the same bf16 tensor drives the
   mask and the combine). Skips ~13% of expert compute at S=2048 and most of
   it at short prefills.
9. **Batch-1 identity-transpose elision**: at B=1 the (1,2) transposes of
   `[1,1,1,d]` tensors (rope cos/sin, kv halves) are logical identities and
   are skipped (3 fewer dispatches/step).

Op merging:
10. **SiLU folded into matmuls** (`activation="silu"`) for the dense-layer and
    shared-expert gate projections, and **into the expert multiply**
    (`input_tensor_a_activations=[ttnn.UnaryOpType.SILU]`) for the routed
    path — measured in-kernel (mul 1.37, mul+lhs-silu 1.57, separate
    silu+mul 2.80 ms at [1,64,1024,1536]). The sparse matmul kernel hardcodes
    `FUSE_ACTIVATION=0`, so the eltwise fold is the only available site.
11. **Routing weights applied to `h` (width 1536) instead of the down output
    (width 2048)**: row scaling commutes with the right matmul; 25% less
    eltwise traffic.
12. **`routed_scaling_factor` (1.8) folded into the routed down weights**
    host-side (constant output scale commutes with the matmul; block-fp
    quantization preserves relative error). The routing-weight chain loses
    its trailing scalar multiply; the shared expert keeps separate unscaled
    weights.
13. **Router activation typecast deleted**: the fp32 router matmul takes the
    bf16 activations directly (mixed-dtype matmul, fp32 acc/out, probe PCC
    0.999996); bf16 -> fp32 conversion is exact so selection semantics are
    unchanged (verified by the tie machinery on real weights: identical tie
    at pos 514).

## Rejected / blocked candidates (with evidence)

| candidate | verdict | evidence |
|---|---|---|
| `ttnn.experimental.moe_compute` (single-kernel tilize+matmul+act+matmul+combine, the dedicated DeepSeek MoE op) | blocked on target | matmul output fails PCC on Blackhole, skipped in CI pending fix: tenstorrent/tt-metal#50038 (`tests/ttnn/nightly/unit_tests/operations/experimental/test_moe_compute_single_card.py`) |
| fold `W_UK` into `wq_b` host-side (`fold_uk`) | rejected, slower | traced decode moe 1.652 vs 1.335 ms/tok, dense 1.550 vs 1.232; prefill a wash. 1.5x weight bytes on a bandwidth-bound projection |
| per-head batched `wq_b` in DECODE | rejected, slower | isolated traced q path: 372 us (batched) vs 123 us (flat + untilize/reshape/tilize). At M=1 tile the broadcast-batched matmul only runs through the default non-mcast path: explicit program configs and `MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig` TT_FATAL on mismatched batch dims (`bmm ... batch dimension 1 mismatch ... only allowed when ... MatmulMultiCoreReuseMultiCast1DProgramConfig with ... mcast_in0=false`). Prefill keeps the batched layout (M=2048 amortizes it; +7.9 MB/layer for the second copy) |
| pack shared-expert / dense-layer gate+up | rejected analytically | regular matmuls have no per-group mcast to save; packing forces the SiLU out of the matmul and adds 2 slices: 4 dispatches vs 3, identical weight bytes |
| indexed sparse matmul for prefill (G*E compact groups) | blocked by op contract | indexed mode requires `num_active <= E` (host validation `must be <= the number of sparse groups`); G*E = 2048 > 64 |
| indexed sparse matmul for batch>1 decode | rejected analytically | union size is data-dependent; a static 4B-slot index list duplicates expert compute against all B rows (>= union cost) and needs a scatter to re-mask per-token weights. Union path kept for B>1 |
| per-token (G=S) sparse groups in prefill | rejected analytically | sparse groups are M=32-row tiles; 1-token groups waste 31/32 of each tile |
| `deepseek_moe_fast_reduce_nc_fused` (score-weighted combine) | blocked by layout contract | consumes the `all_to_all_dispatch` compacted `[experts_k, 1, tokens, hidden]` layout, not the sparse-matmul `[1, E, T, hidden]` layout; only reachable via the moe_compute pipeline (blocked above) |
| `ttnn.experimental.nlp_create_qkv_heads_decode` for the q-only head split | blocked by op contract | the op splits fused q+k+v; with `num_kv_heads=0` (no kv in our absorbed-MLA q tensor) it segfaults host-side (probed on device 0, core dump; device verified healthy afterwards) |
| single rope call for q+kv (concat 21 heads) | rejected analytically | saves one 2-us rope dispatch but adds a concat and two intra-tile row slices of the same bytes |
| `ttnn.rms_norm(residual_input_tensor=...)` | rejected analytically | returns only the normed value; the raw sum is still needed for the outer residual, so no dispatch is saved |
| sparse-matmul output zero-fill removal | out of scope (op-internal) | both functional and fused pay a full-output fill before every sparse matmul (~2.0 + 1.4 ms per moe prefill chunk, measured in both baselines); removing it needs an op change, not a graph change |

## Skill pattern checklist (every graph-fusing pattern assessed)

| skill pattern | disposition |
|---|---|
| elementwise activation recognition | applied (SiLU into matmuls + eltwise lhs fold) |
| softmax | n/a — no spelled-out softmax; attention runs in flash MLA ops |
| RMSNorm | already `ttnn.rms_norm` (functional stage) |
| distributed RMSNorm | n/a — single chip |
| SDPA | already `chunked_flash_mla_prefill` / `paged_flash_multi_latent_attention_decode` |
| split-QKV + split-heads | MLA has no fused qkv; the analogue (fused wqkv_a + head-layout matmul) applied; `nlp_create_qkv_heads_decode` blocked (see rejected table) |
| create QKV heads (decode) | blocked (segfault at num_kv_heads=0, rejected table) |
| concat heads (decode) | applied (`nlp_concat_heads_decode`) |
| concatenate heads (prefill) | applied (`ttnn.transformer.concatenate_heads`) |
| RoPE | already `rotary_embedding_llama` (both modes); decode identity transposes elided |
| TopK | already `ttnn.topk`; its indices now feed the indexed sparse matmul directly |
| RepVGG conv-sum / conv folds / BN folds / pad+pool | n/a — no convolutions |
| shared-LHS matmul | applied twice (wqkv_a; packed expert gate_up); assessed and rejected for shared-expert and dense-layer gate+up (rejected table) |
| spatial mean | n/a |
| permute-reshape-permute | applied in spirit: prefill q reshape+permute and v permute+reshape replaced by layout-producing matmul / dedicated ops |
| matmul/linear + activation | applied (`activation="silu"`) |
| input-arg activation on binary | applied (`input_tensor_a_activations=[SILU]`) |
| matmul + bias -> linear | n/a — the only bias (router e_score bias) applies after the sigmoid, not after the matmul |
| permute/transpose + matmul (transpose_b) | n/a — remaining transposes are batch-dim (1,2), not [-1,-2] |
| slice after matmul -> narrower operand | inverse applied (wider fused matmuls + slices where mcast/dispatch savings dominate); no narrowing candidates remain |
| numeric-stable softmax | n/a |
| reduction + reshape (keepdim) | already keepdim everywhere |
| scaled-sum -> mean | applied in the functional stage (router centering uses `ttnn.mean`); the new scalar fold moves x1.8 into weights |
| RoPE decode fold | already decode-mode rope; the surrounding transposes elided at B=1 |

## Deferred to the optimized-decoder stage (program configs / sharding, not graph shape)

- The b={20} absorbed matmuls (`w_uk` 47 us, `w_uv` 99 us per moe decode step)
  run at ~90-150 GB/s on default configs; deepseek_v3 runs the identical
  matmuls DRAM-sharded-batched (`wkv_b1/wkv_b2` in `mla1d.py`), which needs
  matching batch dims (per-head in0) plus DRAM-sharded weight layouts.
- `LayerNormDeviceOperation` runs on 1 core (33-52 us/step across 4 norms).
- The big dense-layer matmuls (2048x10240 etc., ~660 us of the 890 us dense
  decode step) need dram-sharded / multicast program configs.
- Decode dispatch gap (~245 us/step at 61 ops) shrinks with op-count or
  multi-cq work, beyond graph shape.

## Correctness evidence (PCC vs HF fp32 reference layer, bar 0.995; bf4 arm 0.99)

Logs: `logs/pytest_fused_synth.log` (20 passed), `logs/pytest_fused_real.log`
(3 passed), `logs/pytest_fused_long.log` (5 passed) — all on the final code.

| test | result |
|---|---|
| prefill moe S=17/64/65/512/1024/1057/3000 | 0.999956 / 0.999674 / 0.999678 / 0.999381 / 0.999404 / 0.999421 / 0.999409 |
| prefill dense S=17/512/3000 | 0.999987 / 0.999994 / 0.999994 |
| decode moe (prefill 509, 8 steps) | per-step 0.99997x, agg 0.99998 |
| decode dense | 0.99999 |
| fused-vs-functional dense (no routing discreteness) | prefill 1.000000, decode 1.000000 |
| paged latent cache vs exact linear reference | 0.999990 (permuted page table) |
| batch 8 mixed non-aligned positions (union path) | all users >= 0.99996 |
| batch 32 (union path, 2 update groups) | 29/32 >= 0.995, 3 proven sub-ulp ties |
| traced decode replays | 0.99997x + bit-identical repeat replay |
| traced stress: 96 replays, 3 sweeps x 32 positions | all at bar or proven tie; bitwise-repeatable |
| REAL weights moe prefill S=512 / decode agg | 0.999244 / 0.99702 (pos-514 tie, same as functional) |
| REAL weights dense prefill / decode | 0.999993 / 0.99999 |
| REAL weights + bf4 experts prefill / decode agg | 0.997066 / 0.99489 (functional: 0.997074 / 0.99498) |

Long context (identical ladder to the functional stage, `GLM47_DECODER=fused`,
JSON in this directory):

| 202751-token evidence | dense (control) | moe |
|---|---|---|
| prefill wall | 70.6 s (2871 t/s) | 90.3 s (2246 t/s) |
| latent cache PCC vs exact linear ref | 0.999988 | 0.999989 |
| window start / middle / end (agg) | 0.999989 / 0.999939 / 0.999704, 32/32 rows each | 0.999464 / 0.998915 / 0.994917, 31/31/28 rows, every below-bar row proven routing flip, 0 unexplained |
| decode at position 202751 (full cache) | 0.999995 | 0.999979 |

Plus S=8191 vs the full HF fp32 reference (prefill 0.999459, decode 0.999975)
and the aligned S=202752 prefill (cache 0.999989, final window 28/32 + 4
explained, 0 unexplained): `long_context_aligned_202752.json`.

## Contract preservation

- Public API, paged-cache geometry/dtype/ops, decode tensor positions,
  trace-capturability, determinism (prefill + traced decode bitwise) — all
  preserved and tested. `doc/context_contract.json` gains a `fused_decoder`
  section: `capability_reduction: none`, supported context 202752.
- Non-aligned logical lengths remain first-class (S = 17/65/509/1057/3000/
  8191/202751 tested); padding/chunking stays internal.
- Weight footprint: +7.9 MB/layer (prefill `wq_b_heads` copy) = ~0.36 GiB at
  47 layers; full-model projection stays within the 32 GiB budget (details in
  the contract file).
- Batch: prefill per-user, decode up to 32 users (tile width). Batch>1 decode
  uses the union-sparsity path (indexed compact output cannot express a
  data-dependent union size); batch-1 — the deployment decode shape — takes
  the indexed path. Both are tested.

## Runtime fallback audit

`test_runtime_no_host_fallback`: `ttnn.from_torch` / `to_torch` / `as_tensor`
are monkeypatch-tripwired during fused prefill and decode and never fire; the
module imports torch only inside `from_state_dict` (statically asserted).
Remaining tilize/untilize in the measured paths, all deliberate and measured:
the decode q head split (the layout-free alternative, broadcast-batched wq_b,
is 3x slower — see the rejected table) plus the small pair inside the
post-`nlp_concat_heads_decode` batch slice (~25 us/step; the concat-heads op
pads users to a tile and the pair still beats the replaced
untilize/reshape/tilize); in prefill, the `slice_write` chunk accumulator
lowers to an untilize + slice_write + tilize composite at each chunk boundary
(~330 us of the 15.2 ms dense window) — cheaper than the rolling-concat
baseline it replaced, whose cost grew with the accumulated prefix.

## Watcher

`TT_METAL_WATCHER=2` on the final code over both layer kinds (decode PCC,
moe-512 prefill incl. fp32-acc chunked flash, cache content, traced decode +
stress): 6 passed, 26 dumps, no watcher exceptions, asserts, NOC/L1 sanitize
errors or faults (`logs/watcher/generated/watcher/watcher.log.gz`, bit-exact
gzip of the 1 MB disk-only raw log; pytest log `logs/pytest_fused_watcher.log`).

## Artifacts

- `perf_wallclock_*.json` — before/after wall clock (8 windows)
- `tracy/{moe,dense}/{prefill,decode}_{func,fused}_perf_report.{txt,csv,console.log}`
  + stacked CSV/PNG — tt-perf-report per window on the final code
- `long_context_{moe,dense}.json`, `long_context_aligned_202752.json`
- `logs/pytest_fused_{synth,real,long,watcher}.log`, `logs/pytest_fused_perf_wall.log`
- probe: `../../probe/fusing_contract_probe.py` (10 on-device op-contract
  probes backing the design decisions)
