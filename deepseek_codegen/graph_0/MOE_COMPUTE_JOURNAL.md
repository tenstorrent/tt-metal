# MoE Compute Integration Journal

## RESULT (2026-06-08) — combine hang FIXED on device ✅; now debugging PCC=0.778

Device recovered (user "reset" cleared the fabric wedge; boot still Jun5 12:00 = NOT a host reboot, so
a tt-smi/glx reset evidently DID clear it this time — contrary to earlier experience). Ran
`MOE_USE_COMPUTE=1 MOE_DEBUG_SYNC=1 timeout 600 python3 moe_test.py` with DEFAULT topology (auto, no env
overrides): all_gather_x OK → all_to_all_dispatch_metadata OK → **moe_compute OK** (previously HUNG
right here!) → weighted_k_sum OK → all_reduce_async OK → mesh_partition OK; moe_block=2.91s, EXIT=1.
=> **PR #45764's fix RESOLVES the combine hang on BH 4x8 cax=0 with default topology** — confirms the
static analysis (num_links=2≠ohsd=4 termination-barrier bug).

NEW ISSUE: e2e PCC = **0.777957** (FAIL, floor 0.99), max|Δ|=0.95, BOTH outputs — WORSE than the
0.955668 "routed-zeroed" baseline, so the routed contribution is STRUCTURALLY WRONG (not mere bf4
imprecision, which would IMPROVE on 0.955). RULED OUT: double-applied scores — `compute_combine_golden`
sets `output_ref=matmul_goldens` with NO score mult and `validate_combine` checks against that, so
combine_out is per-k RAW expert outputs and the tail's score-multiply is correct. SUSPECTS (debugging):
(a) combine_out token/k LAYOUT for cax=0 (golden token dim = tokens*cluster_factor via batch_rep_idxr;
my buffer assumes [k,32,7168]); (b) all_reduce(axis1) reduction axis / possible double-count across the
4 cax=0 rows; (c) bf4 weight-prep LAYOUT (never numerically validated — combine always hung before).
Log: logs/moe_compute_fix_test.log.

**RESOLVED (2026-06-08) — e2e PCC = 0.995591 PASS (both outputs)! moe_compute works end-to-end.** The
bug was NONE of (a/b/c): a DTYPE mismatch. The sparse path's `ttnn_typecast_101` is BFLOAT16
(moe_test.py:760) but `run_routed_experts` returned FLOAT32; the downstream
`add(matmul_33[bf16], typecast_101)` (moe_test.py:913) misread the fp32 tile layout as bf16 → scrambled
garbage (delta_add ⟂ routed, 7.5× too large) → e2e 0.778 INVARIANT to routed values (zeros stayed
0.955). Diagnosis chain: per-device routed matched sparse (PCC 0.986, uniform across all 32 devices) +
MOE_ZERO_ROUTED→0.955 (combine/tail side-effects harmless), yet a magnitude check said a 1.276× routed
should give e2e ~0.998 not 0.778 → the added tensor ≠ the dumped tensor → dtype. FIX:
`routed = ttnn.typecast(routed, BFLOAT16)` before return in run_routed_experts. moe_block device time
~2.91s (vs sparse 2.10s — optimize next).
RESIDUAL: routed is still ~1.276× the sparse routed (combine_out scale; PCC-invariant so it also passes
the PR's PCC-threshold combine check, and is only ~31% of the e2e signal so e2e still 0.9956). Likely
bf4 weight-prep or a moe_compute matmul scale; worth a torch-ref check before the main.py 2-layer port
(per-layer errors compound). Diagnostic env flags (opt-in, off by default): MOE_DUMP_ROUTED,
MOE_NO_TAIL_SCORE, MOE_ZERO_ROUTED, MOE_WEIGHT_DTYPE + compare_routed.py / reconcile.py.

## LATEST (2026-06-06) — PR #45764 combine-fix STAGED; device wedged pending reboot

**The combine hang has an upstream fix.** Open PR **#45764** (gajanan-choudhary, "Get moe_compute
functional and tested across multiple models on BH LB") includes commit `d6a8898bf6b "#43444: Fix
selective_reduce_combine local-termination deadlock on BH"`. Root cause: the combine writer kernel's
termination barrier did `noc_semaphore_wait(termination_sync, num_data_parallel_cores - 1)` but must
wait on `num_workers_per_link - 1` (= num_worker_cores / num_links). When those differ the semaphore
target is never reached → **exactly our HANG**. The PR plumbs `num_workers_per_link` as a writer CT
arg and fixes the wait (writer.cpp), refactors worker-layout into `detail::compute_worker_layout`
(program_factory.cpp/.hpp + device_operation.cpp validate), and makes `auto_output_width_shard_dim`
+ `num_data_parallel_cores` ring-aware for `bh_ring_size ∈ {8,12,16}`. Also includes
`9fcd6fd6df3 Support 1D linear (non-ring) fabric topology in MoE compute barrier`.

**STAGED locally (NOT yet device-verified):** overlaid the PR's 9 moe source files onto this branch
via `git checkout pr-45764 -- <files>` (writer.cpp; the 3 selective_reduce_combine device files +
.hpp; moe_compute_device_operation.cpp + _types.hpp; moe_compute_nanobind.cpp; moe_compute_utils.cpp;
ttnn/_experimental/moe_compute_utils.py), then RE-APPLIED the local `num_buffers 14->13` workaround
on top (the PR fixes the HANG, not the separate L1 overlap #46208 — our 4x8/epd=8/hidden=7168 still
overflows mux L1 at 14; num_data_parallel_cores stays 4 so buffer sizing is unchanged). Rebuilding
ttnncpp (+ `cp build_Release/ttnn/_ttnncpp.so build_Release/lib/_ttnncpp.so`). `pr-45764` is a local
fetch ref (HTTPS — origin SSH key is denied). moe_compute_block.py only imports get_tilize_drain_core
and calls the unchanged op signature, so the overlay doesn't break it.

**Why the fix applies to our 4x8 config (static analysis, 2026-06-06 — HIGH confidence):** read
writer.cpp + moe_compute_device_operation.cpp. The combine writer's per-link "sync core" waits for
`wait_count` peer increments on the termination semaphore, THEN closes the fabric mux connections
(`close_direction_connections`, writer.cpp:353) and fires the inter-device global barrier. Bug:
`wait_count` was `num_data_parallel_cores - 1`, must be `num_workers_per_link - 1`, where
`num_workers_per_link = (num_token_parallel_cores * num_data_parallel_cores) / num_links`. Our values:
`num_token_parallel_cores = output_height_shard_dim = 4`; `num_data_parallel_cores = 4` (hidden=7168 ->
224 tiles, ring-aware largest d<=4 with 224%d==0 and ring_n(12)%d==0 => 4); `num_links =
get_num_links(cax=0) = 2` (BH has 2 eth ch/link). => num_worker_cores=16, num_workers_per_link=8. OLD
wait=3 but 7 peers increment -> the sync core completes after only 3, then CLOSES the mux connections
while the other 4 workers are still sending -> those workers hang -> moe_compute hangs. EXACTLY our
symptom. WH never hit it: WH num_links=4 == ohsd=4 so num_workers_per_link == num_data_parallel_cores
(OLD==NEW); BH num_links=2 != ohsd=4 triggers it. The fix (wait = num_workers_per_link-1 = 7) makes the
sync core wait for ALL peers before closing — correct for us. Our LOCAL barrier == the PR's validated
BH-LB deepseek_v3 case (BH, num_links=2, ohsd=4, hidden=7168); only the inter-device half differs
(ours Ring/4-dev cax=0 vs their Linear/8-dev cax=1) and Ring is the WH-validated path. Real
confirmation still pending the device test.

**Manual tail reviewed (2026-06-06, looks correct):** run_routed_experts' combine consumption →
weighted-k-sum → all_reduce(axis1) → mesh_partition was the only never-exercised path (combine always
hung before it). Verified statically: `scr` stays allocated after dispatch (only scr_l1 is freed);
score broadcast is right (combine_out[k,t,:] × score[t,k] via permute→[k,t,1] over hidden, fp32 accum,
sum over k); `outs[5]` is the combine_out slot; mesh_partition(dim=3,cax=1) rebuilds the TP [1,32,896].
No bug found → decent odds the first post-fix run passes PCC, not just avoids the hang.

**DEVICE STILL WEDGED — needs HOST REBOOT (only the user can):** boot is still `Jun 5 12:00` (the
same boot the "make it work" sweep wedged); the 8h wait started Jun5 17:36 = AFTER that boot, so no
reboot occurred. Sanity open (2026-06-06 01:40) fails with the known fabric wedge:
`topology_mapper: "1 target node(s) are not mapped to any global node: 0"`. Per memory
[[tt-device-timeout-recovery]] this needs a host reboot; `tt-smi -r` is HARMFUL here — I will NOT run
it, and I'm running ON this host so I can't reboot without killing this session.
UPDATE 2026-06-07 16:15: STILL wedged after ~46h (host up 2 days, no reboot). `tt-smi -ls` shows ALL
Blackhole galaxy chips ALIVE over PCIe (no 0xffffffff brick) → it's purely the fabric control plane, so
a host reboot will cleanly recover. Three watcher cycles (01:54 Jun6 → 16:14 Jun7) all reported the
identical `node 0 not mapped` wedge. Everything else is DONE (fix staged+built+verified, tail reviewed);
sole blocker is the reboot. Standing by (watcher relaunched).

**AUTO-RESUME PLAN (executes the moment the host is rebooted):**
1. Sanity: `python3 /tmp/sanity_dev.py` must print `DEVICE_OPEN_OK`.
2. Test the fix (simplest first — default topology, which previously hung):
   `cd deepseek_codegen/graph_0 && MOE_USE_COMPUTE=1 MOE_DEBUG_SYNC=1 timeout 600 python3 moe_test.py`
   SUCCESS = `[moe_dbg] moe_compute OK` prints (it used to hang right here) + an e2e PCC.
3. If still hanging: `bash sweep_combine.sh` (Ring/1, Linear/1, bh_ring 8/12/16).
4. On success: record PCC (expect ~0.95-0.99 from bf4), flip moe_test.py default to moe_compute, then
   enter the device-time optimization loop (see "NEXT PHASE" at the bottom).

---

> ## OUTCOME (2026-06-05) — read this first
> **Goal:** make moe_test.py use `ttnn.experimental.moe_compute`. **Result:** integrated and
> proven working on BH 4x8 for dispatch (`all_to_all_dispatch_metadata`), the expert MATMUL
> (`moe_compute(compute_only=True)` runs clean), weight-prep (bf4), and the shared-FFN+residual
> tail. **BLOCKER:** moe_compute's combine (`selective_reduce_combine`) HANGS on our 4-device
> dispatch ring (cluster_axis=0 within the 4x8 BH galaxy). Ruled out: auto/Ring topology,
> Linear+num_links=2, 13-buffer mux. It works on WH 16x8 (test_optimized_moe_decode_block) and
> BH 1x8 LoudBox (test_moe_compute_6U), so this looks like a moe_compute limitation for a small
> BH dispatch ring — needs escalation to the moe_compute owners. `compute_only` can't substitute
> (it double-buffers, retaining only 2 of 8 experts; the combine is mandatory to consume them).
>
> **Final test state (SAFE):** `moe_test.py` defaults to the original sparse_matmul path
> (PCC=1.0, no hang). The moe_compute path is OPT-IN via `MOE_USE_COMPUTE=1` (+ MOE_DEBUG_SYNC=1
> recommended); it currently hangs at the combine — a clean repro for the op owners.
> Code lives in `moe_compute_block.py`.
>
> **tt-metal is now PRISTINE** (the local BH num_buffers 14->13 tweak was reverted; `.so` rebuilt at
> 14). The L1-overlap bug it worked around is instead captured for upstream by:
>   - `repro_moe_compute_combine_l1_overlap.py` — minimal standalone repro (VERIFIED to FATAL on
>     unmodified tt-metal @ 918fad97336: "Mux L1 memory [...] overlaps with L1 tensor [...]").
>   - `ISSUE_moe_compute_combine_l1_overlap.md` — bug-report draft (per .github bug_report.yml).
>
> **Combine-hang repro:** `MOE_USE_COMPUTE=1 MOE_DEBUG_SYNC=1 timeout 600 python3 moe_test.py`
> -> prints `[moe_dbg] all_to_all_dispatch_metadata OK` then hangs at moe_compute (no
> `moe_compute OK`); EXIT=124. (Always run device tests under a timeout; do NOT `tt-smi -r` on this
> galaxy — see memory tt-device-timeout-recovery.)


Goal: make `moe_test.py` use the fused `ttnn.experimental.moe_compute` op for the
expert computation, replacing the current `sparse_matmul`-based path, to speed up
device execution. Keep the block's PCC check against the captured e2e golden.

Run:
```bash
cd /home/ubuntu/tt-metal/deepseek_codegen/graph_0
source /home/ubuntu/tt-metal/python_env/bin/activate
export TT_METAL_HOME=/home/ubuntu/tt-metal PYTHONPATH=/home/ubuntu/tt-metal ARCH_NAME=blackhole TT_METAL_CCACHE_KERNEL_SUPPORT=1
python3 moe_test.py
```

## Baseline (current sparse_matmul path)
Run 2026-06-04, warm JIT kernel cache + warm ccache:
```
ttnn_add_27:      PCC=1.000000 max|Δ|=0.000e+00 shape=(32,32,896) PASS
ttnn_rms_in_4d_4: PCC=1.000000 max|Δ|=0.000e+00 shape=(32,1,32,896) PASS
device_open : 6.577   ce_cache_load: 0.657   input_load: 0.011   moe_block: 2.102   TOTAL: 9.347
```
`moe_block` = 2.10s is the figure to beat (it's the on-device MoE region; includes
dispatch + 2×sparse_matmul + combine + tilize + all_reduce + the gate/router).

## Architecture understanding

### Current graph MoE path (sparse_matmul based), in run_moe_block():
1. RMS norm + gate matmul (router logits)  → reduce_scatter/all_gather
2. `deepseek_grouped_gate` (router) → weights[32,1,8] bf16 + indices[32,8] int32
3. `all_to_all_dispatch` (cluster_axis=0) → sparse tokens [16,1,32,7168] + metadata
4. `moe_expert_token_remap`
5. `sparse_matmul` gate_up: in [16,1,32,7168] × W_gate_up(main_const_eval_gate_up, ~8GB)
   → [16,8,32,4096]; slice into gate[..2048]/up[2048..]; SiLU multiply → [16,8,32,2048]
6. `sparse_matmul` down: × W_down(main_const_eval_39, ~4GB) → permute → [8,1,512,7168]
7. `all_to_all_combine` (cluster_axis=0)
8. `deepseek_moe_post_combine_tilize`
9. `all_reduce_async` (cluster_axis=1) + mesh_partition → residual add → outputs

Dims: hidden=7168, intermediate N=2048, 256 routed experts, selected_k=8, 4x8 mesh.
Dispatch axis = 0 (4 devices). matches reference "deepseek_v3" model config
(N=2048, hidden=7168, selected_experts_k=8).

### moe_compute pipeline (target), from models/common/modules/moe/tt_moe_decode.py:
`all_to_all_dispatch_metadata` → `moe_compute` → `deepseek_moe_fast_reduce_nc_fused`
→ reduce_scatter. moe_compute fuses dispatch-consume + w0/w1 matmul + activation +
w2 matmul + combine. Weights packed via prepare_w0_w1/prepare_w2 → bfloat4_b,
DRAM height-sharded. Reference unit test: tests/nightly/tg/ccl/moe/test_moe_compute_6U.py.

### Key API (ttnn.experimental.moe_compute), from moe_compute_nanobind.cpp:
Positional: tilize_input_tensor (sparse buffer, ROW_MAJOR L1), tilize_expert_indices_tensor,
tilize_expert_scores_tensor, tilize_expert_mapping_tensor, matmul_w0_w1_tensor, matmul_w2_tensor.
kw: layer_id, output_height_shard_dim, intermediate_size (required), has_bias=False,
cluster_axis, topology, num_links, mux_core_range_set, output_memory_config,
optional_output_tensor, optional_cross_device_semaphore, activation_type, compute_only, bh_ring_size.
Returns 6 tensors: (per_expert_total_tokens, expert_activation, e_t, l1_matmul_out, _, combine_out).

## Precision concern
moe_compute uses bfloat4_b weights → reference PCC thresholds ~0.986 (SILU). Current
test golden PCC floor is 0.99 and currently hits 1.0 (identical op sequence). Switching
to moe_compute (bf4) will NOT be bit-exact vs the captured golden. Plan: validate the
moe_compute matmul against a torch golden of the SAME logical weights (proves op
correctness) and/or relax the e2e PCC floor for the moe path. TBD after first run.

## Findings (empirical, from introspect.py + main.py)

Mesh (4,8), cluster_axis=0 (4 dispatch devices), num_replicated=8 (axis 1), 8 experts/device,
256 routed experts, hidden=7168, N=2048, k=8, tokens_per_device=32 (batch=128 over 4 dispatch dev).
This EXACTLY matches `test_optimized_moe_decode_block.py` (cluster_axis=0, hidden=7168, N=2048,
k=8) — the canonical 2D-mesh moe_compute pipeline. Their mesh is (16,8) w/ 2 experts/dev; ours
(4,8) w/ 8 experts/dev. Same per-device token shapes (32 tokens, 896 TP-hidden slice).

Logical weights already on device in ce_cache (per-device shards, BFLOAT8_B):
- `main_const_eval_gate_up` per-dev (1,8,7168,4096) = concat(W0[:,:,:,:2048], W1[:,:,:,2048:]) (L,E,K,2N)
- `main_const_eval_39`      per-dev (1,8,2048,7168) = W2 (L,E,N,K)
Both are exactly the (L,E,K,N)/(L,E,N,K) formats prepare_w0_w1/prepare_w2 want.

`main_const_eval_37` (var_76) = expert_mapping, per-dev (1,1,256,32), replicated, ONE-HOT
[expert, device]: mapping[e,d]=1 iff expert e on device d. moe_compute wants the canonical
[num_devices, experts] linear-coord form → derive: canon[*, e] = argmax_d onehot[e,d].

Activations (per device, TP-hidden 896 = 7168/8, 32 tokens):
- in_ttnn_add_22 (32,896) residual; in_ttnn_reshape_117 (32,1,896); in_ttnn_rms_in_4d_3 (1,1,32,896)
- out_ttnn_add_27 (1,32,896); out_ttnn_rms_in_4d_4 (1,1,32,896)

## Graph MoE structure (run_moe_block) — replacement boundary
- KEEP router (lines ~105-316): produces `ttnn_multiply_58` scores [32,1,8] bf16 +
  `ttnn_typecast_86` indices [32,8] int32; and `ttnn_typecast_78` = normed FFN input [32,896].
- REPLACE lines ~318-755: eq-mask + dispatch + 2×sparse_matmul + combine + post_combine_tilize
  + all_reduce + mesh_partition + matmul_29/30 score path + multiply_60 + sum_18 → `ttnn_typecast_101`
  (the routed MoE output, per-dev [.,32,896]).
- KEEP shared-expert FFN (matmul_31/32/33, lines ~756-901) → adds to typecast_101 → +residual → add_27.

## Integration recipe (low-risk: reuse graph's known-good BH all_reduce tail)
1. x = all_gather(typecast_78 →[32,1,1,896], dim=3, cluster_axis=1) → [32,1,1,7168] per dev.
2. idx = typecast_86 → [32,1,1,8] uint16 (L1 height-shard); scr = multiply_58 → [32,1,1,8] bf16.
3. all_to_all_dispatch_metadata(x, idx, scr, canon_expert_mapping, cluster_axis=0, output_tensors=preallocated).
4. moe_compute(sparse, out_idx, out_scr, canon_em, tt_w0w1(bf4), tt_w2(bf4), output_height_shard_dim=4,
   intermediate_size=2048, cluster_axis=0, mux=(1,1)-(3,3), optional_output_tensor=combine_buf(zeros)).
   → combine_out [k=8, 32, 7168] per dev; slot[k,token]=expert output if token's k-th expert on this
   device's column (m1), else 0 (preallocated zeros).
5. Manual tail (reuses graph's all_reduce_async + mesh_partition, known-good on BH):
   weighted = combine_out * scores[k,token,1]; sum over k → [32,7168] partial;
   all_reduce_async(cluster_axis=1) → full hidden summed across axis1;
   mesh_partition(dim=3, cluster_axis=1) → [1,1,32,896] = routed MoE output = typecast_101.
6. Weights: slice gate_up→w0,w1; w2=const_eval_39; typecast bf16; prepare_* (C++ device) + quantize bf4;
   cache to disk. Done in main() outside the moe_block timing.

Precision: bf4 weights (graph used bf8 sparse_matmul) → expect PCC < 1.0 vs e2e golden;
reference SILU threshold ~0.986. Will measure; adjust PCC_FLOOR w/ justification if needed.

## Attempts log
(below, newest last)

### A1: standalone prototype (moe_compute_proto.py) on (4,8) BH, synthetic weights + manual tail
Validates op + my manual tail vs torch golden before touching moe_test.py.
- Run 1: AttributeError `ttnn.MoEActivationFunction` → fix: `from ttnn.operations.ccl import MoEActivationFunction`.
  (dispatch_metadata compiled+ran OK on (4,8) BH — good sign.)
- Run 2: got into moe_compute combine, then TT_FATAL: "Mux L1 memory [base=0x1b200,end=0x95330]
  overlaps with L1 tensor 0x933c0" (selective_reduce_combine_program_factory.cpp:94). L1 overflow
  by ~8KB. Root cause: moe_compute combine input CB scales with experts_per_device
  (num_buffers=epd=8 when not double-buffered, factory line 332/348) → large L1 tensor low in
  memory; mux (sized for epd=2 deepseek calibration, ~21KB headroom) collides. hidden=7168 is the
  binding BH shape. Need only ~4.4% mux reduction.
  Levers (cleanest first): (1) num_links↑ (num_full_size_channels=div_up(num_workers,num_links),
  num_links*neighbors(2)<=9 cores → max 4); (2) output_height_shard_dim↑ → per-core combine buffer
  = token_seg*total_tokens/num_token_parallel_cores shrinks (factory line 348); (3) patch BH
  num_buffers 14→13 (sanctioned by factory comment for "new shape variant pushes per-core L1").
- Run 3: num_links=4 → TT_FATAL "Requested link index 2 out of bounds, 2 ethernet channels
  available" — our BH 4x8 mesh has only 2 eth channels per ring link, so num_links<=2 (auto was
  already <=2). num_links lever exhausted.
- FIX applied: patched selective_reduce_combine_program_factory.cpp BH num_buffers 14->13 (frees
  ~33KB > 8KB overflow; sanctioned by the factory's own comment). Rebuilt with
  `ninja -C build_Release ttnncpp`. CAVEAT: the target only rebuilt build_Release/ttnn/_ttnncpp.so;
  runtime loads build_Release/lib/_ttnncpp.so (separate file, via RUNPATH) — had to
  `cp build_Release/ttnn/_ttnncpp.so build_Release/lib/_ttnncpp.so` for the change to take effect.
- Run 4 (before the cp): mux end UNCHANGED (0x95330) → confirmed stale lib/.so. After cp, proceeding
  straight to the real integration (faster: real on-device weights, no synthetic host-prep; e2e
  golden is the correctness signal; prototype remains as a tail-isolation fallback).

## Integration into moe_test.py (A2)
Added `moe_compute_block.py`: derive_expert_mapping (one-hot var_76 -> canonical [32,256]),
prepare_moe_weights (slice gate_up->w0/w1, const_eval_39->w2, typecast bf16, C++ prepare_*,
quantize_weights_via_host->bf4 DRAM-sharded; disk-cached at moe_io/wcache), MoEComputeState
(semaphores + preallocated dispatch/combine buffers + all_reduce pools), run_routed_experts
(all_gather x hidden axis1 -> dispatch_metadata -> moe_compute -> weighted-k-sum -> all_reduce(axis1)
-> mesh_partition -> [1,32,896]). moe_test.py: import + MoEComputeState built in main() (timed as
`moe_setup`, outside moe_block) + run_moe_block region (old lines 320-757) replaced by one
run_routed_experts call producing ttnn_typecast_101. Shared-expert FFN + residual kept verbatim.
- Run A2.1: TT_THROW "Tensor is not allocated" in quantize_weights_via_host(w2_prepped). W0/W1
  quantize succeeded; W2 failed.
- Run A2.2 (deallocate-after-quantize): same failure → not a deallocate-ordering bug.
- Isolated via debug_w2.py: prepare_w2_tensor_for_moe_compute returns w2_prepped with
  storage=DEVICE but **on_device=False (deallocated)**. ROOT CAUSE: ce_cache weights are bf8 TILE;
  feeding TILE to the C++ prepare_* helpers (a) makes their torch-style reshapes operate on
  tile-ordered elements (WRONG layout), and (b) hits a latent prepare_w2 bug — its final
  `result = to_layout(padded, TILE); padded.deallocate(force=True)` is a no-op when padded is
  already TILE, so it force-deallocates the returned tensor. The reference uploads ROW_MAJOR
  (from_torch default), which makes that to_layout a real copy and keeps reshapes correct.
- FIX: feed ROW_MAJOR bf16 to prepare_w0_w1 / prepare_w2 (to_layout(typecast(bf8->bf16),ROW_MAJOR)).
  Verified in debug_w2: w2_prepped on_device=True, from_device OK, QUANTIZE OK -> bf4. Cleared the
  stale TILE-input W0/W1 cache (was likely numerically wrong) and rebuilt.
- Run A2.3 (ROW_MAJOR weights): TT_FATAL reshape on uint16 (reshape supports bf16/fp32/int32/uint32
  only). FIX: reshape int32 indices then typecast uint16.
- Run A2.4: TT_FATAL "Physical shard shape (1,8) must be tile 32x32 sized" — dispatch indices/scores
  L1 height-shard [1,8] needs ROW_MAJOR (router emits TILE). FIX: to_layout ROW_MAJOR on x/idx/scr
  before dispatch (matches canonical, whose from_torch inputs are ROW_MAJOR).
- Run A2.5 (integ5): compiled the full pipeline, then **DEVICE HANG** in moe_block execution — log
  frozen 12 min, no compiler procs, python at 130% CPU in synchronize_device (host spinning on a
  hung device op). NOTE: the prototype never actually ran moe_compute to COMPLETION (it errored at
  enum, then mux FATAL during program creation), so integ5 is the FIRST real moe_compute execution
  on the 4x8 BH galaxy — and it hangs. Killed; recovered with `tt-smi -r` x2 (EXIT=0).
  (Galaxy warning: "CPLD FW v1.16+ required for tt-smi -r; else use -glx_reset" — but -r worked.)
- Run A2.6 (instrumented, MOE_DEBUG_SYNC=1, timeout 420, ./logs/integ_dbg.log): synchronize+print
  after each op to pinpoint the hang. all_gather/all_reduce_async/mesh_partition are PROVEN (the
  baseline sparse path uses them); suspects are all_to_all_dispatch_metadata and moe_compute's
  combine (BH moe_compute only validated on 1x8 LoudBox LINE; ours is 4x8 RING, 4-device dispatch
  ring on cluster_axis=0). Result: (running)
  Fallback if moe_compute combine hangs: compute_only=True (skips mux/combine) + reuse the graph's
  proven all_to_all_combine, OR force topology=Linear / smaller num_links on the combine.
- Run A2.6 (instrumented) CONFIRMED: `[moe_dbg] all_gather_x OK` + `all_to_all_dispatch_metadata OK`
  printed, then HANG (173% CPU, frozen) at **moe_compute** (no `moe_compute OK`). So dispatch_metadata
  works; moe_compute's combine hangs on the 4x8 ring.
- Applied fix (UNTESTED): moe_compute(topology=Linear, num_links=2) — the BH LoudBox-validated combo.

## DEVICE BRICKED — needs reboot (blocker, 2026-06-04 ~14:35+)
After killing the A2.6 hang and tt-smi -r x2, the device became UNOPENABLE:
`open_mesh_device(FABRIC_1D_RING,(4,8))` -> topology_mapper.cpp "1 target node(s) are not mapped to
any global node: 0". Recovery attempts ALL failed: tt-smi -r x5 (CPLD<v1.16, doesn't restore fabric
control plane); tt-smi -glx_reset / -glx_reset_auto (USER_RESET finds all 32 chips, then IPMI
POST_RESET fails). Installed ipmitool (per user), but host has NO IPMI BMC (`/dev/ipmi0` absent,
ipmi_si finds no BMC) so the IPMI reset can't run. test_system_health PASSES (all 32 chips present,
TestMeshFullConnectivity OK; the 264 "link DOWN" are expected-unused channels). => the fabric
control-plane/ring state is stuck and only a HOST REBOOT (or out-of-band BMC reset) will clear it.
See memory [[tt-device-timeout-recovery]].

## RESUME AFTER REBOOT
1. Re-run: `cd deepseek_codegen/graph_0 && source ../../python_env/bin/activate &&
   TT_METAL_HOME=/home/ubuntu/tt-metal PYTHONPATH=/home/ubuntu/tt-metal ARCH_NAME=blackhole
   TT_METAL_CCACHE_KERNEL_SUPPORT=1 MOE_DEBUG_SYNC=1 timeout 600 python3 moe_test.py 2>&1 | tee logs/run.log`
   (weights are cached in moe_io/wcache; the moe_compute topology=Linear,num_links=2 fix is in place.)
2. If `[moe_dbg] moe_compute OK` now prints -> Linear topology fixed the hang; check e2e PCC.
3. If moe_compute STILL hangs (no `moe_compute OK`): kill, tt-smi -r, and switch to the fallback —
   moe_compute(compute_only=True) for the matmuls only + reuse the graph's proven all_to_all_combine
   (cluster_axis=0) + the existing manual all_reduce(axis1)+mesh_partition tail. This avoids the
   selective_reduce_combine mux entirely.
4. ALWAYS keep MOE_DEBUG_SYNC=1 + a tight timeout so a re-hang is caught fast and localized.

## 2026-06-04 23:20 scheduled-resume attempt — device NOT usable (needs another reboot)
Host had been rebooted (uptime from 15:03) but at 23:20 the device had LEAKED TLB windows:
`set_fabric_config(FABRIC_1D_RING)` -> "Failed to allocate TLB window ... tt_tlb_alloc -12 for
2097152" with NO live holder (verified: no /dev/tenstorrent fd, no mmap, no hung procs; the
"metal-moe-compute" tmux is THIS claude session, not another user). MISTAKE: ran one `tt-smi -r`
to clear it -> "Reset failed" (CPLD<1.16) + "Error when re-initializing chips!"; now chip 16 reads
0xffffffff ("board should be reset") and `tt-smi -ls` lists 0 boards. So `-r` bricked it further.
=> Needs another HOST REBOOT. Did NOT get to run moe_test / test the topology=Linear fix.
NEXT TIME: on the scheduled resume, ONLY do the sanity open; if it fails for ANY reason
(TLB / node-0 / 0xffffffff), report + reschedule — do NOT run tt-smi -r on this galaxy.
- 2026-06-05 03:25 backup retry: uptime still boot=Jun4 15:03 (NOT rebooted); sanity open still
  fails "0xffffffff PCIe ID 16, board should be reset". Did NOT reset. Rescheduled retry 09:25 UTC.

## 2026-06-05 ~15:13 — device rebooted (boot Jun5 12:00), resumed. topology=Linear did NOT fix combine.
- Sanity open OK (healthy). Ran moe_test (topology=Linear,num_links=2, MOE_DEBUG_SYNC=1, timeout 600):
  `[moe_dbg] all_gather_x OK` + `all_to_all_dispatch_metadata OK`, then HANG at moe_compute again
  (EXIT=124 timeout, no `moe_compute OK`). So topology=Linear + num_links=2 does NOT fix the combine
  hang on BH 4x8. IMPORTANT: my original config matched the WORKING WH 16x8 test_optimized (auto
  topology) and Linear matched the BH 1x8 LoudBox test — neither works on BH 4x8 => moe_compute's
  selective_reduce_combine appears unsupported/broken on a 4-device dispatch ring (cluster_axis=0)
  within a 4x8 BH mesh. Did NOT tt-smi -r; the timeout-kill released the device cleanly (sanity
  open OK afterwards — no brick this time).
- Next: compute_only=True diagnostic (matmul only, no combine/mux) to confirm the matmul half works
  and isolate the hang to the combine. (MOE_COMPUTE_ONLY=1 path in moe_compute_block.run_routed_experts.)
- RESULT (compute_only diagnostic): `[moe_dbg] moe_compute(compute_only) OK` — the MATMUL half of
  moe_compute RUNS cleanly on BH 4x8 (no hang). matmul output (slot 4) shape=(120, 2, 32, 7168)
  [internal per-core layout: 120 ~= matmul cores, 2 = double-buffer expert slots, 32 tokens, 7168
  hidden]. => the hang is DEFINITIVELY in selective_reduce_combine only.
  With routed MoE zeroed (diagnostic), e2e PCC vs golden = 0.955668 (FAIL<0.99) for BOTH outputs ->
  confirms the kept shared-expert FFN + residual path is correct; the routed contribution is the
  remaining ~4.4%. So: dispatch + matmul + weight-prep + shared-FFN + residual all CORRECT; only the
  routed combine is missing because moe_compute's combine won't run on BH 4x8.

## STATUS SUMMARY (decision point)
moe_compute is integrated; on BH 4x8: dispatch_metadata OK, matmul (compute_only) OK, combine HANGS
(ruled out: auto/Ring topology, Linear+num_links=2, 13-buffer mux). The combine is the sole blocker.
Options to finish the routed path:
 (A) compute_only + custom combine: reshape moe_compute's (120,2,32,7168) per-core matmul output and
     route per-expert outputs back to token home (an all_to_all_combine) + weighted-k-sum. High
     effort (reverse-engineer the layout; see test_moe_compute_6U.prepare_output_tensor_from_combine_writer)
     + device-hang risk.
 (B) Accept matmul-validated; treat moe_compute's BH-4x8 combine as a known op limitation to escalate.
 (C) User knows a moe_compute BH-4x8 combine fix (flag / mesh-graph-descriptor / build).

## 2026-06-05 — "make it work, try all options" attempt + CURRENT BLOCKER (device wedged)
Set up an env-driven combine-config sweep to try the remaining options without recompiling:
- `moe_compute_block.run_routed_experts` now reads the combine config from env (opt-in via
  `MOE_USE_COMPUTE=1`; default test path stays sparse): `MOE_TOPOLOGY`={Ring|Linear|unset=auto},
  `MOE_NUM_LINKS`={1|2|unset=auto}, `MOE_BH_RING`={8|12|16}, `MOE_OHSD`=output_height_shard_dim(=4).
- Re-applied the BH `num_buffers=13` workaround (rebuilt build_Release/lib/_ttnncpp.so) so epd=8 gets
  PAST the L1 overlap to actually reach the combine. NOTE: issue #46208 + the repro are on PRISTINE
  14; this 13 delta is LOCAL-ONLY for experiments — revert with
  `git checkout HEAD -- ttnn/.../selective_reduce_combine/device/selective_reduce_combine_program_factory.cpp`
  then `ninja -C build_Release ttnncpp && cp build_Release/ttnn/_ttnncpp.so build_Release/lib/_ttnncpp.so`.
- `sweep_combine.sh`: runs each config as its own timeout-guarded subprocess (a hang only kills that
  attempt), sanity-checks the device between configs, stops on success or a wedge.

RESULT: the sweep could NOT run — the **device WEDGED before the first config executed**. Symptom
(repeatable, also on a bare `sanity_dev.py` device-open): SEGFAULT / core dump at open with
`"NOC address of a hugepage does not match the expected address"` +
`"Cannot access soc descriptor for {14,15,17,18,23,31} before device driver is initialized"` — i.e.
driver/hugepage state corruption on the same flaky chips. No stale process holds the device. This is
NOT the combine hang; the device needs a **HOST REBOOT** (do NOT `tt-smi -r` — see memory
[[tt-device-timeout-recovery]]; `-r` is broken on this CPLD and worsened it before). Likely trigger:
the L1-overlap repro caught a TT_FATAL mid-program-creation then `close_mesh_device` — cleanup after a
caught device-op FATAL appears to leave hugepage/NOC state inconsistent.

## WHAT NEEDS TO BE DONE (resume here)
0. **REBOOT the host** (only the user can; `tt-smi -r`/`-glx_reset` do not recover this galaxy — no
   IPMI BMC). Confirm `python3 /tmp/sanity_dev.py` → `DEVICE_OPEN_OK` before any device work.
1. **Run the sweep** (opt-in, all under `timeout`; success == `[moe_dbg] moe_compute OK` then PASS):
   `cd deepseek_codegen/graph_0 && bash sweep_combine.sh`
   — tries Ring/nl=1, Linear/nl=1, Ring/nl=2+bh_ring=8, Linear/nl=1+bh_ring=8, Ring/nl=1+bh_ring=16.
   Already tried & HUNG (skip): auto/Ring (nl auto≤2), Linear+nl=2.
2. **If still hanging, remaining levers** (add to the sweep):
   - combine semaphore on `get_moe_combine_cores` vs the full worker grid (wire `MOE_SEM_COMBINE_CORES`
     into `MoEComputeState`; the 6U test uses combine cores, test_optimized uses full grid).
   - `output_height_shard_dim` sweep (`MOE_OHSD` ∈ {2,8}) — changes combine token-parallel core count.
   - reduce `tokens_per_device` to 16 (process the 32 tokens in 2 chunks, concat outputs) —
     `test_compute_tg` cut 32→16 on 4×8 "due to L1"; may also relieve the combine.
   - cherry-pick + rebuild the relevant OPEN PRs: **#45764** (gajanan-choudhary: moe_compute
     functional+tested on BH LB — edits selective_reduce_combine), **#45904** (amorrisonTT:
     dispatch_metadata Linear topology), **#46113** (amorrisonTT: >1 shared expert). One may fix the
     BH multi-device combine.
3. **If NO config works:** moe_compute's combine is genuinely unsupported on a BH 4-device dispatch
   ring (cluster_axis=0 in a 4×8 galaxy). It's escalated via issue **#46208** (assigned @amorrisonTT);
   ask the op owners — @dchenTT wrote the BH path (PR #45294), @gajanan-choudhary owns BH-LB
   moe_compute — whether BH multi-axis combine is on the roadmap, and file the separate combine-hang.
4. **When a working combine config is found:** the manual tail in `run_routed_experts`
   (weighted-k-sum → `all_reduce_async`(axis1) → `mesh_partition`) already converts moe_compute's
   `combine_out` into `ttnn_typecast_101`; flip moe_test.py's default to the moe_compute path, confirm
   e2e PCC vs the captured golden (expect ~0.95–0.99 from bf4 weights; validate vs a torch golden if
   <0.99), and record `moe_block` timing vs the 2.10s sparse baseline.

## RESTATED INITIAL TASK (for handoff)
Run `deepseek_codegen/graph_0/moe_test.py` and optimize it for faster **device** execution, with the
PRIMARY requirement that the test **use the `ttnn.experimental.moe_compute` op** (replacing the current
`sparse_matmul`-based routed-expert path). Any other ops in the graph may be changed; iterate on the
test and make moe_compute work for it. Keep `MOE_COMPUTE_JOURNAL.md` updated with everything tried,
what worked / didn't, and results.

STATUS vs task: moe_compute is integrated (opt-in via `MOE_USE_COMPUTE=1`) — `all_to_all_dispatch_metadata`
+ expert matmul + bf4 weight-prep all verified on BH 4×8; the routed **combine hangs** (the blocker;
L1-overlap precursor filed as #46208). The default test path remains the working `sparse_matmul`
(PCC=1.0, `moe_block` 2.10s). The op is **not yet doing the routed compute end-to-end here** — pending
the combine fix / the remaining sweep above (which is blocked until the host is rebooted).

## NEXT PHASE — continuous device-time optimization program (for the next agent)
Once `moe_compute` is actually USED end-to-end here (the combine works — see "WHAT NEEDS TO BE DONE"),
switch into a continuous, autonomous device-time optimization loop modeled on Andrej Karpathy's
`karpathy/autoresearch` (human-authored instructions + ONE file the agent iterates + ONE metric +
fixed/comparable measurement + a git commit per useful win + run the loop and DON'T STOP).

### How autoresearch maps onto this project
- `program.md` (human instructions / direction)  → THIS journal (this section + "RESTATED INITIAL TASK").
- `train.py` (the single file the agent edits)    → `moe_compute_block.py` + `moe_test.py`, and then
  `main.py`'s MoE block once it's integrated into the full 2-layer test.
- `prepare.py` (fixed, never modified)            → `moe_io/` (captured block inputs + goldens),
  `moe_io/ce_cache/` (consteval weights), `moe_io/wcache/` (bf4 moe_compute weights). These define
  correctness — do NOT change them.
- `val_bpb` (one metric, lower=better)            → **ESTIMATED FULL-MODEL e2e DECODE DEVICE-TIME (ms)**,
  lower=better, GATED by correctness (PCC vs captured golden ≥ floor). An optimization only counts if
  PCC still passes AND device time drops.
- fixed 5-min budget (comparable)                 → measure the SAME way every run: warm JIT + warm
  ccache; measure **device** time (Tracy) and/or **trace** steady-state, NOT cold Python dispatch
  wall-clock; median of ≥5 iters; record both first-run (compile) and steady-state.

### Steps (do in order; loop forever on the last one)
1. **INTEGRATE**: flip `moe_test.py`'s default to the moe_compute path; confirm PCC. Then port the
   same change into `main.py`'s layer-1 MoE block and run the **FULL 2-layer decode test**
   (`main.py` / `main_capture.py`) WITH the op; PCC-check the full-graph live-outs.
   - **main.py integration surface (surveyed 2026-06-06):** main.py has NO `run_moe_block` function —
     the MoE is INLINED in the generated forward across 2 layers (2x `deepseek_grouped_gate`; 3x
     `all_to_all_combine`/`ttnn_typecast_101`; 4x `main_const_eval_gate_up` refs ⇒ weights are
     PER-LAYER). Porting = locate each layer's routed region [`all_to_all_dispatch` → 2x `sparse_matmul`
     → `all_to_all_combine` → post_combine_tilize → all_reduce → `ttnn_typecast_101`] inline and swap in
     `run_routed_experts`, PER LAYER. Needed `moe_compute_block` changes: (a) `layer_id` is hardcoded 0
     → thread a per-layer `layer_id` into `moe_compute` + the preallocated dispatch/combine buffers;
     (b) `MoEComputeState` builds weights from a single ce_cache `gate_up`/`_39` → build per-layer
     weight sets. Do this AFTER moe_test.py is confirmed on-device (the standalone block is the cheaper
     validation loop).
2. **BASELINE the metric** (do this BEFORE optimizing): measure the full 2-layer graph's per-decode-step
   device time, EXTRAPOLATE to the full model, and write the first leaderboard row below. Full model =
   DeepSeek-V3.2-Exp — verify the real layer count from the config (DS-V3 ≈ 61 layers; this test runs 2).
   Estimated full e2e ≈ per-layer device-time × num_layers + fixed overhead (embed + final norm +
   lm-head + sampling). Record the exact formula in the row.
3. **PROFILE with Tracy**: run under the device profiler (`TT_METAL_DEVICE_PROFILER=1` + tracy capture;
   see tt-metal `tools/profiler` / docs) to get a per-op device-time breakdown; rank the slowest ops
   (likely: CCL collectives, moe_compute, the big matmuls, layout/typecast churn).
4. **OPTIMIZE — the loop, DON'T STOP**: take the slowest op from the profile, form ONE hypothesis,
   edit `moe_compute_block.py`/`main.py`, re-run, GATE on PCC, measure device time. Levers (any way
   possible): enable **trace** (`begin/end_trace_capture` + `execute_trace`) to kill per-op dispatch in
   steady state; fuse ops; remove redundant typecast/to_layout/reshape; better mem-configs (L1 vs DRAM,
   sharding); tune matmul program-configs / math-fidelity (bf8/bf4 where PCC allows); cut CCL hops /
   tune num_links / topology; overlap the shared-expert FFN with the routed path. After EACH useful
   win: append a leaderboard row (new estimated full-model e2e) AND `git commit` it (convention below).
   Then re-profile and repeat. Keep going.

### Leaderboard (one row per useful optimization; metric = estimated full-model e2e decode ms, LOWER is better)
| date | change | commit | block device-ms (2L, traced) | per-layer ms | EST full-model e2e ms | PCC | notes |
|------|--------|--------|------------------------------|--------------|-----------------------|-----|-------|
| 2026-06-04 | baseline: sparse_matmul path (moe_compute NOT yet used) | (uncommitted) | TBD (use Tracy/trace; 2.10s wall is dispatch-bound, not device) | TBD | TBD (fill in step 2) | 1.0000 | starting point |
| … | … | … | … | … | … | … | … |

### Commit convention (keep the history)
Work on a branch off `main`. One commit per useful optimization (only when PCC still passes AND device
time improved — never commit a regression). Message: `moe-opt: <change> → est full e2e <X> ms (was
<Y>), PCC <p>`; end commit messages with the `Co-Authored-By: ...` trailer. Keep the leaderboard row and
the commit in lockstep so the table == git history.

Structure adapted from Andrej Karpathy's `karpathy/autoresearch` (one-file iteration, one metric,
fixed comparable measurement, commit-per-win, autonomous loop).
</content>
</invoke>
