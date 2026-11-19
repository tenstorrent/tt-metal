# Known issues detail: deepseek_v41_flash_d_p

## Scaled router vs golden sits at the device noise floor; judge it by the noise-floor limit
FIX.2 (2026-10-16): the fixed PCC 0.99 is gone (F76, every limit 2 x the noise floor of the reference's dtypes). At the reference kernel's attention dtypes (bf16 q / P / output, fp32 scores / max / sum) S.swa.11 records router vs golden 0.9862 (limit 0.254), router vs CPU 0.99987, pcc_swap_out 0.99982 (limit 0.9923). The history below is kept for the numbers.

Gate S.swa.11 (scaled 2x2 checkpoint, 96 experts, top 6) fails only `swap_router_vs_golden` (the router component
test's default PCC 0.99 against the golden dense routing weights). Every other check passes, pcc_swap_out 0.999848.

| run | attn_out vs golden | ffn_norm vs golden | router vs golden | rows reselected vs golden |
|---|---|---|---|---|
| CPU reference, golden input | 0.999999 | 0.999999 | 0.999999 | 0 |
| CPU, attention = its bf16 precision model (CC.precision_model) | 0.999631 | 0.999831 | 0.984229 | 0.131 |
| CPU, q_a + kv + attention bf16 models | 0.999206 | 0.999639 | 0.978566 | 0.178 |
| CPU, attn_hc .. ffn_norm bf16 models | 0.999187 | 0.999611 | 0.976934 | 0.187 |
| device, defaults (WINDOW_SDPA=high_precision) | 0.999811 | 0.999911 | 0.988729 | - |
| device, S.swa.10 (router on the CPU) | 0.999811 | 0.999911 | 0.988758 | - |
| device, DS41_WINDOW_SDPA=exact (comparison only) | 0.999930 | 0.999964 | 0.992686 | - |

The router loss is about linear in the attention's rel error (0.58 x rel). The router itself matches the CPU step on
the same inputs (pcc 0.99985, rel 9e-5, reselected 0.0015). The real-size run passed this check at 0.9959 (384
experts). The CPU numbers come from a host-only script (run_block with overrides = CC.precision_model of the listed
steps on golden layer 0, chunk of the component rung). Levers: owner decision on WINDOW_SDPA=exact (OV-3), or a
framework vs-golden check for selection outputs that splits rows by routing (K-39), as step_errors does for vs CPU.

S.swa.11 attempt 2 (host-only, `/tmp/s11b/attn_variants.py`): the reference block with only the window attention's
operands rounded as a device variant would (scale folded into q, fp32 output, no in-tile partial-dot rounding, every
other step the CPU reference). TF32 = fp32 read by the FPU (K-30).

| q | probabilities | attention rel vs golden | ffn_norm pcc | router vs golden | rows reselected |
|---|---|---|---|---|---|
| fp32 | fp32 | 0.0017 | 0.999999 | 0.999956 | 0.000 |
| bf16 | bf16 (today's sparse_sdpa high_precision) | 0.0138 | 0.999956 | 0.992121 | 0.066 |
| tf32 | bf16 | 0.0114 | 0.999970 | 0.994163 | 0.049 |
| bf16 | tf32 | 0.0127 | 0.999963 | 0.993652 | 0.053 |
| tf32 | tf32 | 0.0069 | 0.999988 | 0.995644 | 0.037 |

The device adds the in-tile rounding (DS41-16) and the other steps' noise: 0.9887 on device vs 0.9921 here. An
fp32 q + fp32 probability sparse_sdpa (fork extension; ~+0.0035 in emulation, estimated ~0.991 on device, between
today's 0.9887 and exact's 0.9927) would be a slower path than the default, chosen only to clear a fixed PCC: left to
the owner / overseer.

S.swa_engram.14.test.1 (layer 1, scaled): the fp32 CPU reference from the golden block input already fails the
0.99 PCC (0.984586; reselected 0.1245 vs the golden). Golden boundaries are bf16, and layer 1's input is a rounded
fp32 stream, so the chain's fp8 q-dq steps flip (DS41-1): q_resid 0.999985, kv 0.99989, attn_out 0.99944, ffn_norm
0.99985 on the CPU. The swap test now splits the router vs golden by routing (DS41-32): device 0.9845 / 0.1284
reselected / rel 0.0053 on the same-expert rows, the CPU 0.9846 / 0.1245 / 0.0051.

## The hc steps' own-floor column checks need at least 12 mantissa bits of hc_fn: no single-pass FPU weight passes them

DS41-43. FIX.4 attempt 101 (2026-10-16). CPU, no device: weight_floor_cpu.py (out of tree: runs/run1/evidence/FIX.4/; code below)
builds the gate's own `component_checks.Expect` (the same floor: input nudges, weights fixed at fp32) and evaluates a
"perfect device": the fp32 CPU step on the golden inputs with only hc_fn rounded, everything else exact. Worst
vs-CPU column-bias ratio (limit 2; `*` = vs-golden check fails too):

| step | TF32 rne | TF32 truncate | 9 bits rne (srcA, today) | 11 bits | 12 bits |
|---|---|---|---|---|---|
| L0 attn_hc | 5.30 | rel 6.3e-5 > 3.5e-5 * | 26.5 (col 22) * (col 10 2.8) | 4.43 | 1.25 |
| L0 ffn_hc | 2.87 | rel 1.7e-4 > 1.5e-4 * | 11.3 | 2.03 | 1.01 |
| L1 attn_hc | 1.65 | 40.0 * | 5.90 | 1.22 | 0.76 |
| L1 ffn_hc | 4.07 | 22.8 | 5.18 | 1.46 | 0.96 |
| L2 attn_hc | 7.41 | 40.1 * | 3.61 | 1.31 | 1.65 |

FPU sources hold at most TF32 (10 bits; srcA 9 at HiFi4, DS41-42), so no single-pass weight format passes. Only a split
or exact matmul would, and the owner ruled those out. A weight in in0 / srcB at full TF32 would also fail (4 of 5).
It is better than srcA's 9 bits on 4 of 5 steps, but x would then sit in srcA at 9 bits truncated, which is worse for
the assembled model's fp32 streams (K-30). Not taken; an option for the owner.

Device attribution, L0 attn_hc chunk 1, today's tree (FP32_WEIGHT_ROUND=srca, HC_SINKHORN=sfpu_fp32; probe_010.py):
- step rel: device vs CPU 1.07e-5; host emulation of the 9-bit weight alone (fp64 matmul, fp32 Sinkhorn) vs CPU
  7.85e-6; device vs that emulation 1.04e-5. Mixes vs fp64: device 7.46e-5, emulation 4.99e-5, device vs emulation
  5.63e-5 (column means -1.2e-5..-4.0e-5 of rms: the FPU's accumulation, DS41-16).
- column 22 bias of rms: device +1.38e-4, emulation +1.57e-4, device vs emulation -1.8e-5. Column 10 (vs golden):
  +1.66e-4 / +1.90e-4. So the weight's rounding causes them.
- row 900 (worst, 2.5 x): |dev - cpu| 1.34e-4 vs median row 1.6e-5. The emulation alone gives 3.96e-5, device vs
  emulation 9.47e-5, the Sinkhorn 1.25e-7 (exact). That row's mixes error 1.47e-2 is below the median row's 2.88e-2,
  so the FPU's accumulation error is ordinary there; row 900's Sinkhorn just amplifies it.
- Every SFPU stage is exact (REPORT.md: 2.5e-8..5e-8; Sinkhorn alone 6.5e-8..7.6e-8, DS41-40), and the sum of x^2 is
  exact (4.7e-7). Nothing on the device path is a bug or a lower-precision choice at HiFi4 / fp32 acc / approx off.

Evidence script (`PYTHONPATH=$PWD python /localdev/dnijemcevic/bringup/deepseek_v41_flash_d_p/runs/run1/evidence/FIX.4/weight_floor_cpu.py 0 5`, CPU, ~25 min):

```python
# FIX.4 attempt 101 (CPU only): can ANY weight the FPU holds pass the step's own-floor checks?
# The gate's own Expect (floor = input nudges, weights fixed at fp32) evaluates a "perfect device": the fp32 CPU step
# with only hc_fn rounded (TF32 rne / truncate, 9-bit rne = srcA at HiFi4), everything else fp32 exact.
import sys, torch
from models.demos.common.bringup.testing.harness import spec, component_golden
from models.demos.common.bringup.testing.component import _step
from models.demos.common.bringup.testing import component_checks as CC
from models.demos.common.bringup.testing import model_precision

def rnd(w, bits, trunc=False):
    w = w.float().contiguous(); sh = 23 - bits; i = w.view(torch.int32)
    if trunc:
        r = (i >> sh) << sh
    else:
        lsb = (i >> sh) & 1; r = ((i + ((1 << (sh - 1)) - 1) + lsb) >> sh) << sh
    return r.view(torch.float32)

s = spec(); model_precision.apply(s); g, c = component_golden(s)
cases = [(0, "attn_hc"), (0, "ffn_hc"), (1, "attn_hc"), (1, "ffn_hc"), (2, "attn_hc")]
if len(sys.argv) > 1:
    cases = cases[int(sys.argv[1]) : int(sys.argv[2])]
for layer, step in cases:
    ref = s.hooks().reference(s, layers=[*s.state_sources(layer), layer], dtype=torch.float32)
    st = _step(ref, layer, step)
    want = g.layer(c, layer)[st.output].float()
    ex = CC.Expect(s, ref, layer, st, g, c, want, None, None)
    key = {"attn_hc": "hc_attn_fn", "ffn_hc": "hc_ffn_fn"}[step]
    w = ref.w[layer]; fn0 = w[key].clone()
    print(f"\n==== L{layer} {step}", flush=True)
    for name, fw in (("fp32", fn0), ("tf32_rne", rnd(fn0, 10)), ("tf32_trunc", rnd(fn0, 10, True)),
                     ("9bit_rne", rnd(fn0, 9)), ("11bit_rne", rnd(fn0, 11)), ("12bit_rne", rnd(fn0, 12))):
        w[key] = fw
        outs = {x.name: CC.run_cpu(ex.cpu_step, x) for x in ex.cases}
        w[key] = fn0
        print(f"-- weight {name}", flush=True)
        bad = ex.evaluate(outs)
        print(f"   -> {len(bad)} failing checks", flush=True)
```

## The tail check after an index step uses a fixed 0.01; its noise floor is 0.026

`testing/component.py:_check_tail` re-runs the steps after the last swapped one from the CPU step's output and
compares the block out. Its limit is 2 x the same tail run from the step's nudged noise-floor outputs, but only
when those are floating point; the indexer's are int32 top-k, so it falls back to `swap_out_rel` = 0.01 ("fixed").
Probe `tests/ttnn/unit_tests/operations/ds41_indexer/probes/probe_003.py` (the gate's run_swap_test with
`_check_tail` wrapped; layer 2, golden chunk 1, 2048 rows, start 2048, 2048 keys visible to the last row), tail rel
vs the block with the CPU indexer output:

| top-k fed to the tail | tail rel | worst row | overlap vs CPU (mean / worst) | rows with a flip |
|---|---|---|---|---|
| device (bf16 scores, fp32 DEST head sum) | 0.02299 | 0.145 | 0.99447 / 0.97852 | 1947 |
| CPU fp32, inputs nudged in their bf16 cells, seed 0 | 0.02634 | 0.187 | 0.98637 / 0.94922 | 2046 |
| same, seed 1 | 0.02554 | 0.146 | 0.98633 / 0.95117 | 2048 |
| same, seed 2 | 0.02615 | 0.133 | 0.98604 / 0.94727 | 2048 |
| CPU, final score rounded to bf16, lowest-index ties | 0.02339 | 0.186 | 0.99474 / 0.97852 | 1882 |

Rows without a flip: device 3.9e-4 (top-k order: device descending score, CPU ascending index), flipped rows ~0.020.
A flipped key swaps one sparse-attention input whose attention weight the index score does not bound, so each flip
moves its row by ~2 %. The official code computes `index_score` in bf16 (`hf/inference/model.py:557-558`: bf16 q,
bf16 k_cache, einsum out bf16, bf16 weights_proj), so bf16 scores are the reference's precision; the device's fp32
DEST head sum is already finer. The F72 rule applied to the int case would give 2 x 0.0263 = 0.0527 >= 0.0230.

Confirmed by the debugger (S.enc_source.07 debug, probe_004.py): 2048 rows, no duplicate index on either side, the
valid (>= 0) count equal on every row, 101 rows with the same index set (0 rows in the same order). Putting those
101 rows in the CPU's order takes their tail row rel from 3.9e-4 to exactly 0 (total 0.022992 -> 0.022991). So the
whole 0.023 comes from near-tie flips; no index is lost, duplicated or out of range on the device.

Which rounding makes the 0.023 (S.enc_source.07 implement attempt 102, probe
`tests/ttnn/unit_tests/operations/ds41_indexer/probes/probe_005.py`: the CPU indexer with the device's two bf16
points switched on one at a time, fed to the same tail):

| top-k fed to the tail | tail rel | worst row | rows with a different index set |
|---|---|---|---|
| device (bf16 gates, bf16 scores) | 0.02299 | 0.145 | 1947 |
| CPU, bf16 gates + bf16 scores | 0.02245 | 0.146 | 1883 |
| CPU, fp32 gates + bf16 scores | 0.02339 | 0.186 | 1882 |
| CPU, bf16 gates + fp32 scores (= the device with an fp32 top-k) | 0.01017 | 0.132 | 395 |

The bf16 score rounding before the top-k accounts for all of the 0.023. Rounding the gates to bf16 adds nothing.
An fp32 top-k does not exist (`topk_large_indices` and `ttnn.topk` reject anything but BFLOAT16). Even if it did,
with the score op's bf16 gates it still gives 0.0102 > 0.01. Only fp32 gates plus fp32 scores plus an fp32 top-k,
which is the CPU's arithmetic and finer than model.py's bf16, could pass the fixed limit. That trades speed for
exactness, so it is the owner's decision (brief rule). The remaining paths are still a waiver or a framework floor
for int outputs.

The framework fix, measured (S.enc_source.07 implement attempt 103, probe
`tests/ttnn/unit_tests/operations/ds41_indexer/probes/probe_006.py`): the frozen gate's `run_swap_test` with only the
`is_floating_point()` filter lifted in `_check_tail` (component.py:600; int nudged outputs fed to the tail as they
are). Every check passes: `swap_indexer_out` 0.022992 <= 0.052679 (2 x the nudged floor 0.026339), err_swap_out
0.030351 <= 0.063240, every step check ok, `run_swap_test` True. That one-line change in
`models/demos/common/bringup/testing/component.py` (read-only for model steps), or the owner waiver, closes the step.

Done as framework F92 (7ca27ba1741): `_check_tail` feeds int nudged outputs to the tail as they are
(component.py:605-612).

## The swap tests' indexer checks failed on order and on bf16 score ties, not on logic

DS41-102 (FIX.10 attempt 3, 2026-10-22).

The swap tests of indexer blocks (enc_source_14, enc_source_engram_17; dec_candidate_14 and dec_reindex_13 by the same
path) were last run 2026-10-16/17, when the indexer's match limit was 0.0. Under the current framework (F101 flips,
F138 limits from the noise floor) they fail:

| test | swap_indexer_vs_cpu match (limit) | flips | unexplained | recall | recall_tie |
|---|---|---|---|---|---|
| enc_source_14 (L2) | 0.0015 (0.922) | 1967 | 1471 | 0.9942 | 0.9975 |
| enc_source_engram_17 (L14) | 0.0016 (0.860) | 1579 | 218 | 0.9979 | 0.9983 |

CPU only, layer 2, golden inputs (script /tmp/fix10diag/idx_ties.py; framework's NF.flips, 3 nudge seeds):

| selection vs native fp32 | flips | unexplained | recall | recall_tie |
|---|---|---|---|---|
| nudged inputs (the floor) | 339 | 0 | 0.99967 | 1.0 |
| native scores rounded to bf16 (topk_large_indices is bf16-only) | 1879 | 1277 | 0.99458 | 0.99777 |
| official arithmetic (numerics official, dtype bf16: DeepSeek's GPU code) | 2048 | 2048 | 0.96375 | 0.96809 |

- Order: the reference returns ids ascending (lowest_index_topk); the device, descending by score. The native set
  sorted by score matches the reference elementwise at 0.0014, the device's value. Order carries no meaning (sparse
  attention reads a set; model.py topk(sorted=False)). K-42: index outputs need topk_overlap, not match.
- Ties: the device's flips are what bf16 scores give; DeepSeek's own code is 6x further from native. The flip rule uses
  the nudge and tie_rtol (0.002 x |cutoff|, less than one bf16 step) and not the official arithmetic's margin (F128).
- Not from P.21 or FIX.10: P.21 did not touch tt/indexer.py (last changed P.19 066340ecb1c); the device's flips match
  what bf16 scores alone give, so no indexer change since 2026-10-17 is needed to explain them.
- enc_source_engram_17 also hits pytest-timeout (300 s) in the same run.

## Chips 7 / 10 lost the real-time profiler: two cards came back in an identity IOMMU domain after reset

(8x4 Galaxy, FIX.3, DS41-120.)

- Host: kernel cmdline `iommu=pt` (Weka's NIC needs passthrough); `/etc/systemd/system/tt-iommu-domains.service`
  (`/usr/local/sbin/tt-iommu-domains.sh apply`, oneshot at boot, 2026-10-22 10:16) unbinds each TT card, writes
  `DMA-FQ` to its `/sys/kernel/iommu_groups/<g>/type`, rebinds. Domain type is runtime only.
- 30 cards are `DMA-FQ`; `/dev/tenstorrent/7` = 0000:08:00.0 (group 79) and `/dev/tenstorrent/10` = 0000:43:00.0
  (group 103) are `identity`. Their PCI device / group directories were re-created after the service ran (08:00.0 at
  2026-10-25 16:56:07, group 79 at 16:56:48; group 103 at 2026-10-22 10:34:48): the functions were re-enumerated (board
  reset / PCI rescan) and came back in the boot default domain, which the boot-only service never fixes.
- Metal: `Cluster::iommu_enabled_` is read from the FIRST MMIO chip only (tt_metal/llrt/tt_cluster.cpp, "Cache IOMMU
  status"), so `d2h_uses_hugepage_fallback` is false (system_memory_manager.cpp) and the eligibility check passes
  (Blackhole needs IOMMU for 64-bit PCIe addressing) for every chip. `D2HSocket::init_host_buffer` (d2h_socket.cpp)
  mmaps an anonymous 2 MB + 4 KB buffer (fifo + bytes_sent + connector state, page aligned) and pins it.
- UMD per chip: `PCIDevice::detect_iommu` reads `iommu_group/type` (identity -> disabled); `map_for_dma` then pins with
  `TT_DMA_FLAG_CONTIGUOUS` (0x4), which the KMD refuses (EINVAL) because 4 KB anonymous pages are not physically
  contiguous. UMD also logs "IOMMU state mismatch for chip 7: expected enabled, got disabled" at open.
- The brief's hypothesis, tested: "contiguous needed" holds only for these two chips (their domain); "hugepages run out"
  does not: the D2H buffer is never hugetlb (HugePages_Free 0 while a mesh is open is the dispatch sysmem).
- Probe (FIX.3, 8x4 FABRIC_2D, one ttnn.add on every chip with a real-time callback): records from 30 chips, none from
  7 and 10; `IsProgramRealtimeProfilerActive()` True (it means "at least one chip"), so the framework's require_rt
  passes and a profile silently has no device time for 2 of 32 chips.
- Fix at the cause needs root: `sudo systemctl restart tt-iommu-domains` (idempotent; converts only the two cards).
  Durable: the service should also run on a TT device add (udev), the host owner's change. Metal side (outside this
  model's paths): take the IOMMU state per chip (or refuse a mixed cluster loudly), and expose which chips run the
  real-time profiler. Check: framework F194 (orchestrator.iommu_mismatch / iommu_problem, sysfs): a PREFLIGHT warning at
  launch; no model-side check (FIX.3's tt/rt_profiler_check.py was reverted by the overseer: a host fact is the framework's).
- Other effect of the mismatch, not measured: chips 7 / 10 run dispatch over hugepage sysmem (identity), the other 30
  over IOMMU-mapped sysmem.

## The contract's per-layer KV limit (ring drift) fails deep layers over every position on 8x4
(DS41-125, 8x4, K.1 attempt 5)
test_adapter_acks on galaxy_8x4 (40 layers, chunk 5120, golden s10240) failed only on window_kv "nope" PCC of the
dumped deep layers (all other configs and layers passed, all acks on time, pad rows zero, replicas equal):

| check | L20 | L21 | L24 | limit L20 / L21 / L24 |
|---|---|---|---|---|
| ack slot 0 [0, 2536) | 0.9810 | 0.9853 | 0.9805 | 0.9896 / 0.9886 / 0.9837 |
| ack slot 1 [0, 5120) | 0.9873 | pass | pass | |
| final slot 0 / 1 [0, 6072) | 0.9888 / 0.9886 | pass | pass | |

The limit is `server_rules.state_threshold`: max(0.93, `e2e_limits.pcc_limit(drift-s10240, state_window_kv_Lxx)`),
and that drift entry is measured on the ring state (positions 10112-10239). The ladder L.s10240 measures the device
there at err 0.083 (L20), 0.096 (L21), 0.112 (L24). The window KV of a layer is kv_norm(wkv x) of the layer's input:
for L20 that is L19's output, which the same ladder measures over every position at err 0.159 (limit 0.267). PCC grows
with the range's length ([0, 2536) 0.981, [0, 5120) 0.987, [0, 6072) 0.989, ring about 0.997), so the earliest
positions carry the largest relative error. That is the accumulated hidden-state error the ladder accepts. No index or
layout error is visible: layers 0-3 and 14 and every compressed / index cache pass. No CPU diagnosis at device
formats was possible 20 layers deep. The limit was not changed (OV-15: no rule, threshold or check). The owner's options:
a waiver, or a per-position drift for the contract's window-KV limit (golden generation).
Owner decision B (2026-10-27): the last 128 positions of each range (caches.GoldenKV(window="last")). Since FIX.19
(DS41-193, owner 2026-11-01) the acks / runner-contract tests use GoldenKV(window="drift"): every position against
the official reference's drift on the same rows (section "The contract's window-KV limit must come from the official
drift on the checked rows, not the ring").

## The galaxy goldens have no official_floor.json, so component limits are one bf16 rounding of the output

The frozen component tests limit each step to calib (2) x the larger of the official arithmetic (official_floor.json
next to the component goldens, noise_floor.arith), the input nudge, and one rounding of the output to its official
dtype. `/data/$USER/bringup/deepseek_v41_flash_d_p/golden_full/galaxy_8x4/s10240_c5120/` has no official_floor.json
(only `golden_scaled/quietbox_2x2/s4096_c2048/` has one). So on the 8x4 the L00 attention limit is 2 x 0.00166, the
output's bf16 rounding, and the input nudge is only 0.00029. The test's describe line has no "+ official arithmetic".

The reference at its own precision, measured on the CPU on the galaxy golden input (chunk 1 of s10240,
`ref.with_numerics("official", dtype)` vs the native golden):

| step | official fp32 | official bf16 (GPU) | nudge | device (8x4, shipped) |
|---|---|---|---|---|
| L00 attention | 0.0323 | 0.0327 | 0.00029 | 0.0171 |
| L02 attention | 0.0367 | 0.0371 (its fp4 KV is not counted: the compressed input is given) | 0.0249 (fp4/16) | 0.0604 |

2x2 official_floor.json (official taps over the official chain), vs the golden: L00 0.045, L02 0.074, L14-L24 0.22-0.25.

L00, the native step with the device's formats added one at a time (CPU emulation, rel vs the golden, bias = median row
ratio - 1):

| item | alone | all but this one |
|---|---|---|
| bfp8 GEMM inputs (wq_b x with the scale folded in, wo_a x, wo_b x; ties away) | 0.0117 | 0.0064 |
| the same, packer double rounding (DS41-137) | 0.0139, bias +0.65 % | (all: 0.0149, +0.67 %) |
| sparse_sdpa HiFi2: in0 (q, P) 6 explicit mantissa bits (K-30) | 0.0074, bias -0.56 % | 0.0129 (no change) |
| bf16 activations (q, kv, sdpa out, inverse rope, wq_b / wo_b out) | 0.0037 | 0.0137 |
| dense weights bfp8 (DENSE_WEIGHTS) | 0.0030 | 0.0126 |
| RoPE cos / sin bf16 | 0.0012 | 0.0129 |
| P bf16 before P @ V | 0.0006 | 0.0131 |
| attention sink bf16 | 0.00005 | 0.0129 |
| all (single rounding) | 0.0129 | |

On the device, ATTN_FIDELITY=HiFi4 (diagnostic run only): L00 0.0171 -> 0.0162, bias -0.03 % -> +0.48 %. That is the
packer's positive bias showing once HiFi2's negative bias is gone. The CPU model explains 0.013-0.015 of the device's 0.017.
Only bfp8 GEMM inputs are large, and they are the device's 8-bit form of the reference's fp8 e4m3 quantization (F208;
the reference's own rounding costs 0.032 on the same input). Raising them would go above the reference. L02's excess
over its nudge limit comes from the 4-bit compressed KV (CKV_FORMAT=bfp4_rows, owner F208): bf16 KV 0.018, bfp4 0.060.
Even then the device stays below the reference's own 0.074.

## P.9 tips test_adapter_acks' L20 window-KV check by rounding, not a bug
(DS41-141; first written by FIX.12 as "Quick A/Bs skip the contract tests: P.9 tipped test_adapter_acks unseen")
FIX.12's gate (code e627e29e63e = P.10 + the lowbit phase 1 merge 7f55821f579) failed only test_adapter_acks:
`chunk 0: ack slot 0 [0, 2536) (last-window 128 rows): 0_window_kv layer 20 nope pcc 0.98855 < 0.98957`.
The last pass of that unit was FIX.9 (2026-10-29 06:22); P.7-P.10 were judged on quick A/Bs ("full gate pending"),
which run no contract test. Bisection on the same code, one switch at a time (scripts/run_safe_pytest.sh, logs in
/scratch/dnijemcevic/fix12/):

| run | L20 nope PCC | result |
|---|---|---|
| defaults (P.10 + merge) | 0.98855 | FAIL |
| DS41_ATTN_SP32=off (P.9 path) | 0.98828 | FAIL |
| DS41_ATTN_SP32=off DS41_ATTN_SP32_PROJ=off (P.8 path) | - | PASS (0 failures) |

The lowbit merge is bit-identical at 8x4: all 207 accuracy metrics of FIX.12's ladder units equal P.10's to the bit.
Ladder (quick measure) err_state_window_kv_L20: P.7 / P.8 0.0900, P.9 0.1056, P.10 0.1094 (limit 0.1515);
err_layer_L19 0.1666 / 0.1654 / 0.1670: L20's input is no worse, only its last-128-row sample moved. P.9 is not
narrower anywhere: wo_b sums all of K in fp32 DEST and rounds once to bf16 (the reference's output dtype) where the
P.8 path summed 4 bf16 partials in a bf16 reduce-scatter. So this is DS41-125's gap (the limit is the ring drift at
positions 10112-10239 of s10240, applied to positions 2408-2535 of a 2536-token prefix) tipped by a rounding
reshuffle, not an index or layout bug. Nothing raised above the reference; no test or limit changed. Options (owner):
a waiver for this check, or a per-position drift for the contract's window-KV limit.
test_runner_contract (the same window-KV check through the tt-d-gen engine) fails with the identical value
(`slot 0 turn 1 (last-window 128 rows): 0_window_kv layer 20 nope pcc 0.98855 < 0.98957`).

### FIX.13 (overseer, device allowed): rounding or bug? Rounding.
Device probe (scripts/tt-probe.sh --mesh, layers 0-20 of the full model, chunk 0 of the golden's tokens, the
contract's own call: note_tokens / begin_chunk / embed / layer(i, h, 0, state, end); script and analysis in
/scratch/dnijemcevic/fix13/: probe_dump.py, ana1-5.py, dumps on/ and off/). It reproduces the failing value: L20 window
KV nope PCC on the last 128 rows of [0, 2536) 0.98855 (defaults, P.9 + P.10), 0.99131 with ATTN_SP32=off
ATTN_SP32_PROJ=off.

1. The padded partial chunk is not involved. Same code (defaults), end=2536 with the PAD_ID tail vs end=5120 with real
   tokens: every layer's output rows [0, 2536) and its window KV are bit-identical (rel 0, 0 differing rows, L0-L20).
   So no pad row reaches a valid row: not through the 160-row projections, the last chip's share (rows 2400-2559 of
   SP row 3, TP chip 3, holding the 24 pad rows), the all_to_all, any norm or reduction, or the MoE.
2. There is no localized excess, as an index, layout or row-share bug would leave. Error vs the golden per 160-row
   chip group (16 groups, rows < 2536), on / off:
   - L0 out 0.0409 / 0.0411, every group equal to the third digit (chip 15: 0.024 / 0.024)
   - L0 window KV 0.0048 / 0.0051 (P.10's single fp32-DEST q_a / wkv sum is closer)
   - L0 layer update (out - in) per stream x TP column quarter (16 cells): equal to the third digit (0.0594 / 0.0597)
   - L1-L3 the same picture; L14 out 0.0947 / 0.0952; L20 out 0.0282 / 0.0284
   - L20 window KV over every position 0.1816 / 0.1819. Per group, on is better in 8 of 16 and worse in 8; the last
     group is 0.130 / 0.113.
3. The on vs off difference grows smoothly with depth and evenly over the chip groups: rel 0.011 (L0), 0.024 (L5),
   0.045 (L14), 0.040 (L19). Per group at L20 it is 0.10-0.34 (chip 15: 0.160, not an outlier). That is how a
   deep-layer system amplifies two equally valid roundings: each path is about as far from the golden as from the other.
4. The check is a sample of that spread. L20 nope PCC on 286 sliding 128-row windows (ends 256..2536, step 8):
   on - off has mean -0.00005 and std 0.0032 (min -0.0097, max +0.0095), and on is worse in 150 of 286. Below
   0.98957: on 203, off 191 of 286. The failing window's -0.0028 is the 15th percentile. Off passes only because this
   one window happens to fall on its good side. With the full chunk, on's windows ending at each chip boundary
   2560-5120 give 0.9887-0.9960.
5. CPU at device formats (rule 14): the first point where on and off differ is L0's attention, by 0.011. There the
   device-format CPU model of FIX.11 (DS41-140, L00 attention) explains the device's error from its bfp8 GEMM inputs
   and bf16 activations, and both paths sit at the same distance from the golden. Deeper layers have no per-step
   golden on this prompt (DS41-125: no CPU diagnosis 20 layers deep). Nothing in P.9 is narrower than the reference.

Verdict: honest rounding (DS41-125's limit-vs-position gap), no bug. No code change; the waiver stays for the owner
(options as before: keep the waiver, or a per-position drift for the contract's window-KV limit).
Later (FIX.19, DS41-193): the contract's window-KV limit became per position (caches.GoldenKV(window="drift")); this
check passes and the L20 waiver was removed (1eaa7164991).

## P.23 OWN_ROWS tips the acks / runner-contract window-KV checks by rounding, not a bug
(DS41-192, FIX.18 overseer fix, galaxy 8x4, code 3a1a390acd8 = P.23..FIX.17.) K.1R's test_adapter_acks and
test_runner_contract failed three checks with OWN_ROWS=on and passed with OWN_ROWS=off (overseer, /scratch/dnijemcevic/diag_acks).
Device probes (scripts/tt-probe.sh --mesh, DS41_OWN_ROWS=on / off, the contract's 2536-token prompt with the PAD tail;
scripts, dumps and logs in /scratch/dnijemcevic/fix18/: probe.py, probe_moe.py, probe_fu.py, ana.py, ana_fu.py) reproduce
every failing value to the digit: L20 0.98947 (off 0.99195); L35 0.98910 (off 0.99857); L38 0.98994 (off 0.99651).

1. Per step, golden input (layers 0-3, 14, 20; the golden's "in" through the device layer): on and off are equally far
   from the golden at every step (L00 experts 0.06047 / 0.06029, L03 out 0.03071 / 0.03027, L20 wkv 0.00804 / 0.00804,
   L20 out 0.02468 / 0.02467). attn_collapse, attn_norm, attention and the window KV are bit-identical on vs off.
2. First difference: attn_hc's mixes ([S, 24] fp32: hc_fn^T over the whole 4H row on one chip vs TP partials added in
   mesh order): 96-99 % of entries differ, at most 1.6e-5 abs / 2.3e-4 rel. attn_residual's bf16 streams then differ at
   0.13-0.20 % of the elements (one bf16 rounding flipped), ffn_norm's bf16 output at 3.3 %. Neither order is more precise
   (attn_hc vs golden 0.00052 / 0.00052 at L00).
3. MoE block alone (golden ffn_norm, bf16) through the OWN_ROWS path (owners a2a into the fused head's x, the local
   fused tail) vs the column split: bit-identical, layers 0-3, deterministic over 2 runs. The engram columns and the
   norms only move bytes or sum in fp32.
4. Chain (layers 0-20, start 0): L20 window KV over every position: on 0.1836, off 0.1842 vs the golden. Per TP
   column, the same to the third digit (L20 on 0.205 / 0.177 / 0.175 / 0.175, off 0.212 / 0.175 / 0.180 / 0.165): no
   position or chip pattern. 286 sliding 128-row windows (ends 256..2536, step 8), L20 nope PCC: on - off mean +0.0005,
   std 0.0040, on worse in 156; below 0.98957: on 192, off 186; the failing window is the 16th percentile of on - off.
5. Follow-up (the acks sequence: [0, 2536), then [2496, 6072) on the same state, keep_history=False): rows
   [2496, 2536) of the first chunk vs the follow-up, window-KV rel per layer: L01-L16 on = off (0.003-0.07), L17-L38 on
   larger (L35 0.142 vs 0.048). Against the golden the recomputed rows are as accurate as the first ones in both modes
   (L20 on c0 / c2 0.0868 / 0.0848, off 0.0929 / 0.0946; L24 on 0.1342 / 0.1354, off 0.1282 / 0.1291). The "last block"
   check compares two device runs, each about 0.13 off the golden. Their mutual distance is one sample of that spread.

Verdict: a genuine one-rounding difference (the fp32 summation order of the mHC mixes and the norm statistics), with no
lower-precision choice behind it. It is amplified chaotically over depth, as P.9 was (DS41-141). No code changed. The
owner's options are as before: a waiver for these checks, or a per-position drift for the contract's window-KV limit
(DS41-125). The 0.99 last-block rule is tt-d-gen's own, and its limit is not tied to the noise floor.
Later: the L20 check passes FIX.19's per-position limit (DS41-193); the last-block 0.99 checks stay under the owner's
waiver (spec.yaml unit_waivers, widened to any layer 2026-11-05).

## The contract's window-KV limit must come from the official drift on the checked rows, not the ring
(DS41-193, FIX.19 overseer fix, owner 2026-11-01; closes the DS41-125 / DS41-141 / DS41-192 gap for the dumped layers.)
The acks / runner-contract tests compared a dumped layer's window KV on the last 128 rows of each range against
calib^2 x the drift of the final ring (positions 10112-10239 of s10240). FIX.19 produced the official reference's
window KV at every position (CPU, numerics official, bf16, layers 0-24 resident, both 5120-token chunks: 530.5 +
557.8 s = 1096 s; `tests/bringup/contract/official_kv.py`, file `<golden dir>-official/window_kv.safetensors`). Its last
128 rows reproduce drift-s10240's state_window_kv_L20 / L21 exactly (pcc 0.99739 / 0.99716, rel 0.0722 / 0.0754).

Official vs native golden, nope PCC (pe in the test output):

| layer | [0, 2536) | [0, 6072) | [0, 10144) | ring |
|---|---|---|---|---|
| L00 | 0.99952 | 0.99951 | 0.99951 | 0.99952 |
| L03 | 0.99799 | 0.99821 | 0.99826 | 0.99850 |
| L14 | 0.98458 | 0.98930 | 0.99078 | 0.99380 |
| L20 | 0.98311 | 0.98998 | 0.99234 | 0.99660 |
| L21 | 0.98778 | 0.99223 | 0.99363 | 0.99648 |
| L24 | 0.98386 | 0.98969 | 0.99145 | 0.99525 |

The reference's own error grows towards the start of the prompt, as the device's does (DS41-125). The FIX.18 device dumps
(L20, OWN_ROWS on / off, prompt [0, 2536)), each against the new limit on the same rows:
- [0, 2536): nope 0.97809 / 0.97794 vs limit 0.93244 (official 0.98311); pe 0.99310 / 0.99309 vs 0.97721
- last 128 rows: nope 0.98947 / 0.99195 vs 0.96765 (official 0.99191); pe 0.99690 / 0.99783 vs 0.98952
- 286 sliding windows, nope median 0.98761 (std 0.0151); the official median is 0.98926 (std 0.0128). The device's window
  PCC tracks the reference's own (correlation 0.95).
The device sits about as far from the golden as the reference at its own precision. Before, the limit was a different
window's margin. Check: `caches.GoldenKV(window="drift")`, `GoldenKV.limits` (drift._float_entry per channel,
e2e_limits.pcc_limit, floor R.PREFILL_GOLDEN_PCC); a noise input fails (pcc ~0). Analysis:
/scratch/dnijemcevic/fix19/predict.py, predict.log.

## P.26 WINDOW_KV_BF16

P.26 perf (galaxy 8x4, attempt 1, 2026-11-01): a 16-bit window KV switch + contract relative-error diagnostics (the BREADCRUMBS section; BREADCRUMBS.md is at the 500 KB hook limit, DS41-194).

- **Setting** WINDOW_KV_FORMAT gains `bf16_tiles` (made the default for the A/B as the brief says; `bfp8_tiles` = P.18's
  path, unchanged; `bf16` = the pre-P.18 row table, unchanged). It is above the reference's e4m3 window KV (rule 7):
  owner's comparison switch (2026-11-01), not a realistic default. The owner decides from the A/B.
- **What bf16_tiles is** (tt/attention.py `region_bf16`, `region_geometry`, `_window_region`, `_put_rows`, `_sdpa`):
  the same per-chip region [carry 128 | own q_rows] per layer, the same load-time region index lists, the same single
  neighbour send (last chip's kept tail -> chip 0), non-chunk-aligned starts as before. Kept as bfloat16 with no
  rounding step: region [1, 1, 128 + q_rows, 512] bf16 ROW_MAJOR (one row a page), kept tail [1, 1, 128, 512] bf16.
  Step 2: own rows mask_chip -> to_rm -> `ttnn.bringup.slice_write` at row 128 (in place of ckv_bfp4). Step 3: the
  carry rows (served slab slice or `cache.carry`) slice_written into the tail as they are (no to_layout / ckv_bfp4).
  Step 4: `WindowSend(mesh, 128 + q_rows, 1024, carry_pages=128)`.
- **Send** (tt/window_send.py): page count of the carry is a parameter (`carry_pages`, default 4 = P.18). New
  compile-time arg B: pages read per NoC barrier into L1 before their packets go out (16 for 1 KiB row pages, 16 KiB of
  L1; 1 when a page takes more than one packet, so the bfp8 path's loop is the P.18 loop). Still one core a chip, prepared
  (patch count 2 a chip asserted, OV-23).
- **sparse_sdpa**: no fork change. Deviation from the brief's wording ("dense band read in the sparse_sdpa fork ... if
  it needs a bf16 key option, add it"): a bf16 ROW_MAJOR region is read by the existing bf16 row reader (one 1 KiB row a
  page, interleaved over the banks, so one NoC read per row is the layout's densest read; DS41-182 for why not TILE
  layout): compressed layers via the fork's ROW_BFP4 `kv_window` (bf16 rows, the pre-P.18 bf16 kv_window path);
  window-only layers via `ttnn.bringup.sparse_sdpa(kv_format=BF16, max_grid=...)` (the source program with every fork
  option off) instead of stock `ttnn.transformer.sparse_sdpa`, because the stock op takes no core cap (rule 18); the
  old `bf16` path keeps its stock call, unchanged.
- **Memory a chip** (bytes of the allocated buffers, tt/perf_window_tiles.py on the 8x4): region per layer bfp8_tiles
  166464 B (9 x 18496) vs bf16_tiles 294912 B (288 x 1024): 40 layers 6.66 MB vs 11.80 MB (+5.14 MB); kept tail per
  layer per slot 73984 B vs 131072 B. (2x4 geometry, q_rows 640: 443904 vs 786432 B a layer.)
- **Byte test** `tt/perf_window_tiles.py` now parametrized over fmt bfp8 / bf16 x q_rows 160 / 640: 4 passed on the 8x4
  (19.9 s). Warm send, enqueue + sync: bfp8 374-437 us, bf16 384-702 us (dominated by the sync; host enqueue 209-343 us).
- **Rule 18 / cores**: send 1 core; slice_write and to_rm (untilize_with_unpadding) under max_grid / sub_core_grids;
  sparse_sdpa fork max_grid (110). No ckv_bfp4 on the bf16 path.
- **Contract diagnostics** (tests/bringup/contract/caches.py `GoldenKV.note_err` / `record_errors` / `official_rows`,
  `channels`, `rel_err`; test_adapter_acks, test_runner_contract): next to every window-KV PCC check (drift mode of the
  dumped layers, every position; the ring snapshots of the other layers; the last-block device-vs-device checks), the
  relative L2 error ||device - ref|| / ||ref|| per channel (nope / pe), and where the check compares with the golden and
  official_kv has the rows (the dumped layers, FIX.19's window_kv.safetensors), the official reference's relative error
  on the same rows. Recorded per (test, kind, layer, channel), the worst device value and the official on that same
  range: `err_acks_wkv_<kind>_L<ll>_<ch>` / `..._official`, `err_runner_wkv_...` (kind drift / ring / last / full /
  lastblock). Never gated: no limit, no threshold (names miss the gate's err_layer_L* / err_state_*_L* patterns); PCC
  checks and waivers unchanged. A missing official file only drops the *_official values (drift mode still needs it).
- OV-3: nothing padded. OV-23: WindowSend stays prepared. OV-25: no fork suite run (no fork changed).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tt/perf_window_tiles.py -s`; `python -m models.demos.common.bringup.orchestrator
  measure --task P.26 --spec <spec> [--part last|test_adapter_acks|test_runner_contract]`; old path:
  DS41_WINDOW_KV_FORMAT=bfp8_tiles.
- **Accuracy (change on, measure parts, this code)**: rung `last` every layer / state within its limit: L00 0.02471
  (bfp8_tiles P.18 0.025307), L03 0.07680 (0.07673), L20 0.16476 (0.16442), L39 0.6691 / 1.533, final hidden 0.19066 /
  0.24219, top5 1.0. test_adapter_acks: 0 failures; test_runner_contract (tt-d-gen engine feeder): 0 failures.
  Relative error, worst range (both tests alike; the served 0_window_kv cache is bf16 in both formats, DS41-143, so
  these move only through the downstream layers): drift L20 nope 0.2088 (official 0.1838), pe 0.1164 (0.1067);
  L14 nope 0.2064 (0.1755); L00 nope 0.0054 (0.0314); last block L35 nope 0.1349, L38 nope 0.1380 (device vs device).
  The A/B against bfp8_tiles (and its speed at 51200 / 512000) is the orchestrator's.
- **Final measure** (`orchestrator measure --task P.26`, change on, rc 0 every run; contract tests and rung reused on
  the same code): profile rung chunk_wall_ms 198.7 / device_ms_total 161.92; p51200 wall 202.8 (spread 31.1), device
  172.31, attention section 45.0 ms; p512000 wall 233.7 (spread 10.8), device 211.43, attention 45.19 ms. No change-off
  side measured here (the orchestrator's A/B).

## The device's 1.55x of the official hidden error is 2.1-2.2x its logit KL, but ~0.1-0.2 pp of top-1

BREADCRUMBS section of FIX.20 (overseer fix, 8x4, 2026-11-01); BREADCRUMBS.md is at the 500 KB hook limit (DS41-194).

- **Model change (the only one):** `tt/settings.py` WINDOW_KV_FORMAT default `bf16_tiles` -> `bfp8_tiles` (owner
  2026-11-01: bf16_tiles is a switch only, the shipped window KV stays 8-bit, as the reference's e4m3; the setting's
  text says so). The P.26 ladder unit (rung last) passes on it: final hidden rel 0.1902 (limit 0.2422; bf16_tiles gave
  0.1907), top-1 0.9971, top-5 1.0.
- **New:** `tests/bringup/test_logit_kl.py` (the FIX.20 unit) and `tests/bringup/official_final_norm.py` (the official
  final norm, CPU, made once: `<golden dir>-official/final_norm-<rung>.safetensors` + `.json` with the golden's
  content_hash and the file's own sha256; a mismatch is refused). The rung is in the file name because rungs last and
  s56320 share one golden but not one feed. The official run is reference/drift.py:replay (all 40 layers, bf16,
  numerics official), fed as the ladder feeds the device; its final-norm rel equals drift-<rung>.json final_hidden
  exactly (s10240 0.12807, last 0.12109). Made in **1838 s** (s10240, 2 chunks from 0) and **931 s** (last, 1 chunk
  after the golden prefix).
- **Method:** logits = final_norm @ head.weight (fp32, 256-row blocks), log-softmax in float64; vs the native golden
  per position: KL(p_golden || p_x), top-1 agreement, top-5 overlap (|top5 & top5| / 5), next-token NLL (the rung's
  last position has no next token). Gate: kl_mean_<rung> <= calib^2 (4) x official kl_mean. Sanity: golden logits'
  argmax = the golden's stored top32_ids[:, 0] on 1024 / 1024 rows; golden vs its bf16 rounding KL 3.5e-8.
- **Results** (5120 positions each; device = shipped bfp8_tiles; wkv16 = DS41_WINDOW_KV_FORMAT=bf16_tiles):

| rung | x | final_norm rel | kl_mean | kl_p99 | kl_max | top-1 | top-5 ovl | NLL (golden) |
|---|---|---|---|---|---|---|---|---|
| s10240 | device | 0.1971 | 0.00608 | 0.1267 | 2.62 | 0.9951 | 0.8886 | 0.0238 (0.0175) |
| s10240 | wkv16 | 0.1966 | 0.00622 | 0.0947 | 3.34 | 0.9955 | 0.8888 | 0.0249 |
| s10240 | official | 0.1281 | 0.00277 | 0.0360 | 2.08 | 0.9971 | 0.9241 | 0.0197 |
| last | device | 0.1902 | 0.00508 | 0.0790 | 2.39 | 0.9965 | 0.8941 | 0.0306 (0.0258) |
| last | wkv16 | 0.1907 | 0.00447 | 0.0627 | 2.01 | 0.9971 | 0.8945 | 0.0291 |
| last | official | 0.1211 | 0.00242 | 0.0308 | 1.56 | 0.9975 | 0.9271 | 0.0272 |

  Limits: s10240 0.01110, last 0.00969: both pass (device 0.55x / 0.52x of the limit).
- **Answer to the owner:** the device's 1.54-1.57x of DeepSeek's hidden-state error becomes 2.1-2.2x its KL (KL ~
  rel^2: 1.55^2 = 2.4), but at the tokens it is small: top-1 agreement with the golden 0.9951 / 0.9965 vs the
  official 0.9971 / 0.9975 (10 / 5 more flips in 5120), next-token NLL +0.0041 / +0.0034 nats a token over the
  official (+0.0063 / +0.0048 over the golden). Top-5 overlap 0.89 vs 0.92 counts near-zero-probability tail tokens.
  Window KV 8 vs 16 bits is not visible at the tokens: kl_mean 0.00608 vs 0.00622 (s10240), 0.00508 vs 0.00447
  (last); the rest of the gap to the official is elsewhere.
- **20 worst positions** (device, shipped) are in the FIX.20 unit log and generated/fix20_logit_kl/<rung>-<variant>.json
  (pos, token, next, KL, official KL, golden / device / official top-1). Top of the list: s10240 pos 9834 KL 2.62
  (official 0.85; golden 5887, device and official both 458), 6295 KL 2.09 (official 0.03; golden 334, device 11146);
  last pos 56220 KL 2.39 (official 0.19; golden 411, device 411, official 260), 53760 KL 1.71 (official 0.61). 7 (s10240)
  and 9 (last) of the 20 worst are positions the official also moves (official KL > 0.15).
- **Gate:** `gate FIX.20 --check`: P.26 PASS (621 s), FIX.20 PASS (1859 s: 4 device cases, ~230-246 s model load each).
- OV-3: nothing padded. OV-23: no Python-built op added or touched. OV-25: no fork suite run.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.deepseek_v41_flash_d_p.tests.bringup.official_final_norm s10240 last`
  (only for a new golden); `gate FIX.20 --check --unit FIX.20`.

## The device's extra error over DeepSeek's reference is its bfloat4_b routed-expert weights

(FIX.21 FORMAT_ATTRIBUTION: which device format makes the device's error larger than DeepSeek's reference.)

(BREADCRUMBS.md is 9 bytes under the 500 KB hook, DS41-194: this section is FIX.21's breadcrumb too.)

What: `reference/ds41_ref.py` got a diagnosis-only device-format emulation on top of numerics official at bf16
(`FORMAT_SITES`, `DS41Reference.with_formats(site='device')`, `bfp_round`, `fpu_operand`, `attend_fpu`); the goldens
and every gate stay native. `reference/test_format_attribution.py` runs it on layers 0-9 chained as the ladder feeds
the device (`reference/drift.py:replay`), rung `last` (s56320's last chunk, 51200-56319, golden prefix loaded): one
5120-token chunk a run (s10240 needs two), deep enough that the compressed history / index formats matter, and the
device's own numbers on it are FIX.20's ladder. CPU, 16 threads, nice 19: official ~270 s a run, all-device ~600 s
(the per-use bfp4 rounding of every routed expert's weights is half of it).

Sites ('device' = tt/settings.py's shipped choice; 'official' = model.py's own quantizer):

| site | device | official | loo L09 | add L09 | (medians L00-L09) |
|---|---|---|---|---|---|
| experts_w | routed weights bfloat4_b, 16 along K of [N, K], RNE (EXPERTS_IMPL=bfp4) | fp4 e2m1 x ue8m0 / 32, exact | 0.960 | 0.946 | 0.976 / 0.949 |
| ckv | compressed KV bfloat4_b / 16, RNE (CKV_FORMAT=bfp4_rows) | fp4 e2m1 / 16, e4m3 scale | 0.178 | 0.278 | 0.193 / 0.295 |
| index_k | keys bf16 in the state, scored bfloat4_b (INDEX_K_FORMAT) | fp4 e2m1 / 32 ue8m0 | 0.018 | 0.046 | 0.021 / 0.045 |
| moe_sums | routed terms snapped 2^-19, fp32 sum, bf16 out; shared expert 4 TP partials bf16 + bf16 adds | fp32 sums, one bf16 rounding | 0.001 | 0.017 | 0.001 / 0.018 |
| dense_w | fp8 dense weights (and bf16 wo_a) as bfloat8_b of W^T / 16 along N (DENSE_WEIGHTS) | exact | 0.006 | 0.016 | 0.006 / 0.009 |
| fidelity | HiFi2: bf16 in0 truncated to 6 explicit bits, in1 9 (hc_fn + its srca weight, compressor wkv / wgate / wk, weights_proj, sdpa q and P) | fp32 / bf16 GEMMs | -0.000 | 0.002 | 0.005 / 0.003 |
| index_q | query bfloat4_b RNE (INDEX_Q_FORMAT) | fp4 e2m1 / 32 | -0.003 | 0.004 | 0.000 / 0.004 |
| index_score | fp32 DEST score, bf16 out | bf16 einsum / relu*w / head sum | 0.004 | 0.000 | 0.002 / 0.000 |
| engram_rows | rows bfloat8_b RNE (ENGRAM_ROWS) | bf16 exact | -0.005 | 0.000 | 0.001 / 0.004 |
| experts_act | x / h / y bfloat8_b (half away), fp32 act, bf16 weight after w2 (EXPERTS_PRECISION=glm) | fp8 x, r*h; bf16 gate / up / out | -0.001 | -0.060 | 0.006 / -0.061 |
| shared | x / h bfloat8_b, bfp8 weights (SE_PACK_PRECISE) | fp8 x / h | -0.029 | -0.024 | -0.029 / -0.035 |
| window_kv | bf16 state, read as bfloat8_b RNE (WINDOW_KV_FORMAT=bfp8_tiles) | e4m3 / 32 ue8m0 | -0.042 | -0.117 | -0.055 / -0.144 |
| gemm_in | fp8 GEMM inputs bfloat8_b (half away; GEMM_INPUTS) | e4m3 / 32 ue8m0 | -0.071 | -0.174 | -0.079 / -0.174 |

gap_closed (loo) = (err_device - err_loo) / (err_device - err_official), addone = (err_add - err_official) / (same),
on err_layer_L09: device 0.12522, official 0.07862 (gap 0.0466; at L00 the device is BETTER, -0.0031, from gemm_in
and window_kv). Sums: loo 1.02, add-one 0.93: the sites are close to additive; experts_w and ckv overlap a little
(their add-ones sum to 1.22, their loos to 1.14), the negative sites offset part of that.

Validation (before any attribution): the all-device run vs FIX.20's device errors on the same rung, err_layer_L00..L09
and every err_state of those layers (26 metrics, compress_partial / engram_tail exact on both): ratio 0.893-1.007,
median 0.986 (layers 0.965-0.993; window_kv states 0.89-0.99, the bf16 served cache, DS41-143). The official run
reproduces drift-last.json to 5 digits. How fidelity is emulated: FPU_BITS (flat_routed_expert_ttnn/tests/reference.py,
K-30): at HiFi2 / LoFi in0 (srcB) keeps 6 explicit mantissa bits, in1 (srcA) 9 at HiFi2 / 4 at LoFi, truncated; so a
bfp8 / e4m3 / bfp4 operand is never cut, only bf16 activations are (the sites listed); the routed / shared experts at
LoFi have bfp4 / e4m3-valued weights (<= 4 bits) and bfp8 x / h: no-ops there. bfp rules: test_bfp_packer_rounding.py
rne_host (host packer, ckv_bfp4 / bfp_round / the rope chain SFPU packs) and `device` (bfp8_pack_precise / typecast,
half away), checked value for value by test_bfp_rules_match_packers; denormal groups pack as zeros.

Not sites (equal to official at bf16 or no-ops by the rule above): BF16_ACTIVATIONS / FFN_NORM_DTYPE / norm outputs
(official rounds the same points to bf16), FP32_ACC (needed == all), the router (HiFi3 on bf16 x: no truncation; bf16
weights folded into experts_act), CCL bf16 sums other than the MoE's (OWN_ROWS / ATTN_SP32 leave none on this path).
Not modelled: the double rounding of bf16 -> bfp8 where a norm packs bfp8 straight from fp32 (NORM_GEMM_DTYPE), the
flat op's h packer (one half-away rounding assumed), ring orders of the bf16 reduce-scatter.

Side finding (index keys): with the keys scored as bfloat4_b the top-k misses 0.168 / 0.152 of the native ids (L02 /
L08) vs 0.108 / 0.086 with the reference's own fp4 (add_index_k vs official; the device 0.164 / 0.155): bfloat4_b's 3
magnitude bits under a 16-value exponent select worse than e2m1 with a ue8m0 per 32, unlike the per-step numbers in
INDEX_K_FORMAT's note (7.6-15.7 % vs 8.0-16.6 %). It costs the layer outputs little (loo 0.018).

At the tokens (part 4; 40 layers from the golden prefix, rung last's chunk, test_logit_kl's metric and its cached
official final norm; CPU 16 threads: all-device 2360 s, minus experts_w 1356 s):

| run | final_norm rel | kl_mean | kl_p99 | top1 | top5 | nll |
|---|---|---|---|---|---|---|
| device (FIX.20) | 0.190 | 0.00508 | | 0.9965 | | |
| emulation, all device | 0.188 | 0.00464 | 0.0799 | 0.9969 | 0.894 | 0.0278 |
| emulation, all device but experts_w official | 0.100 | 0.00143 | 0.0177 | 0.9979 | 0.943 | 0.0244 |
| official (DeepSeek, bf16) | 0.121 | 0.00242 | 0.0308 | 0.9975 | 0.927 | 0.0272 |

So the emulation also matches the device end to end (KL 0.91 x, final norm 0.99 x), and with exact expert weights
the device's other formats land BELOW DeepSeek's own reference (KL 0.59 x official, final norm 0.83 x).

Recommendation: A/B EXPERTS_IMPL=bfp8 on the device first (exact for the checkpoint's e2m1 x ue8m0 values, DS41-10):
leave-one-out says it removes 0.96 of the hidden-state gap at L09. Cost from tt/settings.py: P.10 experts call 3.47
-> 4.47 ms a layer (2x2, chunk 2048), weights 8.9 -> 16.8 GiB a chip at 40 layers on 8x4 (MOE_IMPL=block asserts
bfloat4_b today: tt/moe_block_adapter.py, so the A/B needs the block to take bfp8 weights). Second: CKV_FORMAT=bf16
(0.18-0.28; +3.3x table memory, sparse_sdpa -0.4 %).

Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.21 --spec
models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check` (runs cached in
generated/fix21_format_attribution/<golden hash>/v1/; delete them, or bump EMU_VERSION, after changing a site).

## P.27 EXPERTS_BFP8: routed-expert formats on the block MoE, and DRAM per chip at 1M

Perf pick P.27 (galaxy 8x4, 2026-11-02; DS41-197, DS41-198). EXPERTS_IMPL takes four values, all of them on
MOE_IMPL=block (the default stays bfp4, F241; the orchestrator A/Bs bfp4 vs bfp8 here, P.28 / P.29 the other two):

| value | w1 / w3 | w2 | rounding on the host | flat op wdtype |
|---|---|---|---|---|
| bfp4 | bfloat4_b | bfloat4_b | none (ttnn's RNE packing, `_bfp4_round`) | bf4 |
| bfp8 | bfloat8_b | bfloat8_b | none (exact, `_bfp8_exact`, DS41-10 / DS41-35) | bf8 |
| bfp4_mse | bfloat4_b | bfloat4_b | `_bfp4_mse_round`: per 16 values along K the larger or the one-lower exponent, least squared error, ties the larger | bf4 |
| bfp8_down4 | bfloat8_b | bfloat4_b | w2 `_bfp4_mse_round` | bf8d4 (fork option down_weights_bf4) |

What changed:
- `tt/experts.py`: `IMPLS` (value -> wdtype and per-matrix format), `_bfp4_mse_round` (chunked; the group's
  largest result has the chosen exponent, so ttnn's packing is lossless: asserted at load, `_bf4_mse_lossless`),
  `expert_weight_round(impl)` (the CPU emulation of each value: bfp8 identity, bfp4 `_bfp4_round`, bfp4_mse /
  bfp8_down4 w2 `_bfp4_mse_round`), `impl_dtypes`. `bfp4_stats` (changed fraction, worst ||dW|| / ||W||) is logged
  for every bfloat4_b matrix (EXPERTS_BFP4 ... impl=, matrices=). The weight cache key already carries the value
  (`L<layer>.<impl>.<hash>`), so each value has its own cache files.
- `tt/moe_block_adapter.py`: the bfp4-only assert removed; `block_dtypes()` takes the op's dtypes
  (`MoEDtypes.routed_weights` / `routed_down_weights`).
- `models/demos/moe_block/moe_block.py` (shared): `MoEDtypes.routed_down_weights` (None = routed_weights);
  RoutedExperts keeps the bfp4-only rule for the block's own packing and checks a caller's op against MoEDtypes;
  TokenExchange.build sets K3_WDTYPE to the op's wdtype while K3 builds its placeholder flat op (DS41-197) and
  checks the placement keys (down, rdn, relays, coords, arena_tiles, ...) equal; other models (bfp4, no caller op)
  unchanged (a difference stays a warning there). static_l1_bytes differs on every format (K3's placeholder plans
  without max_y_writers: 18304 B vs the op's 12160 B on bf8 / bf8d4): logged only.
- The flat op's L1 plan already sizes everything from w_tile (1088 B for bfp8): gate/up split under GU_L1_BUDGET,
  rings, regions; the block's arena (1208 KB, 35 cells, 16 accumulators) is the same for every value.
- Fork flat_routed_expert_ttnn: down_weights_bf4 (CHANGELOG 2026-11-02; tests/unit/test_down_bf4.py 2 / 2 on 8x4:
  bf8d4 == bf8 bit for bit on bfp4-exact down weights).

Layer 0 through the block (tests/test_experts_formats.py, s10240 chunk 0, rel / PCC of the layer output vs the
golden; cold cache miss: ~140 s a layer for bfp8 / bfp8_down4 / bfp4_mse):

| value | rel | PCC | load-time stats (layer 0) |
|---|---|---|---|
| bfp4 | 0.0350 | 0.99940 | 16.65 % changed, worst matrix 0.1196 |
| bfp4_mse | 0.0247 | 0.99970 | 7.75 % changed, worst matrix 0.0672 |
| bfp8_down4 | 0.0206 | 0.99979 | w2 7.62 % changed, worst 0.0645 |
| bfp8 | 0.0152 | 0.99989 | exact |

Host (tests/test_experts_bfp4_mse.py, 3 / 3): ttnn's bfloat4_b packing of `_bfp4_mse_round` is bit-exact (random,
ties, saturation, e2m1 groups, 6 real expert matrices); per matrix ||dW|| / ||W|| 0.063-0.065 (RNE 0.112-0.118);
the lower exponent wins on 31 % of the tested groups.

DRAM per chip (tests/bringup/test_dram.py; one slot, 1,048,576 tokens = max_seq 1,049,600, 40 layers, chunk at
1,039,360; the mesh allocator is lockstep, so every chip reads the same; allocator capacity 31.831 GiB a chip;
P.27 attempt 2, 2026-11-03, both sides after the state-walk fix below):

| GiB a chip | bfp4 (gate unit) | bfp8 (measure, change on) |
|---|---|---|
| routed experts | 8.899 | 16.809 (x 1088 / 576) |
| other weights (incl. indexer _kbuf / _bmax work buffers sized to max_seq, ~0.07 a layer) | 6.440 | 6.440 |
| served caches (engine slabs) | 1.408 | 1.408 |
| slot state (other) | 0.786 | 0.786 |
| tables | 0.884 | 0.884 |
| buffers + unattributed | 0.036 | 0.036 |
| after load | 18.454 | 26.364 |
| peak (layer 36, outside the MoE / attention calls; 795 samples) | 19.634 | 27.544 |
| chunk at 1,039,360 (s, incl. sampling) | 185 | 146 |

bfp8 at 1M: fits the physical capacity (27.54 of 31.83 GiB, 4.29 GiB free at peak) but not the 27.2 GiB budget
(peak +0.34 GiB over; after load 26.36 fits). bfp8_down4 would be ~ 16.81 - 2.67 = 14.1 GiB of experts (w2 at bfp4),
peak ~24.9 GiB (estimate; P.29 measures it).
test_dram fix (attempt 2): DS41DeviceState holds `self.model` (tt/model.py:208), so the state's walk took every weight
as "state" (attempt 1's bfp8 run: state 7.227, other_weights 0); the walk now skips the model and its mesh.

Not done here (paths outside P.27's): bringup/hooks.py:devfmt_component and reference/ds41_ref.py's experts_w site
still emulate bfp4 vs exact only; both should call `tt/experts.py:expert_weight_round(impl)` per matrix.

## The MoE block's exchange plans its combine from K3's own placeholder flat op, built in K3_WDTYPE (env, default bf4)

DS41-197 (P.27). models/demos/deepseek_v3_d_p/tt/moe/tt_k3_token_split_moe.py TtK3TokenSplitMoE.__init__ reads
`self.wdtype = os.environ.get("K3_WDTYPE", "bf4")` and builds a "fake" FlatRoutedExpert with it (routed_only: the real
weights come per call); `self.op.plan` then gives writers, cells, relays, the arena assertion and the deterministic
planner's schedule. The plan depends on the weight tile (576 / 1088 B: gate/up split under GU_L1_BUDGET, rings,
offsets), so a bfp8 routed op under a bf4 placeholder would run with a combine planned for another flat layout.
moe_block TokenExchange.build now builds K3 inside `_k3_weight_format(op)` (K3_WDTYPE = the op's wdtype, restored
after) and `_check_k3_plan` asserts the placement keys equal for the new formats. That K3 file is outside the bring-up
paths; a `wdtype=` argument there would replace the environment hand-off. More in "P.27 EXPERTS_BFP8" above.

## A least-squares bfp4 exponent halves the expert weights' error

DS41-198 (P.27): numbers and the per-value table in "P.27 EXPERTS_BFP8: routed-expert formats on the block MoE, and
DRAM per chip at 1M" above (tt/experts.py:_bfp4_mse_round; tests/test_experts_bfp4_mse.py, tests/test_experts_formats.py).

## P.28 EXPERTS_BFP4_MSE: wiring check, cache warm, change-on ladder

Perf pick P.28 (galaxy 8x4, 2026-11-03; DS41-198, DS41-199, DS41-200). No re-implementation: P.27's value checked.
(BREADCRUMBS.md is at the 500 KB hook limit, DS41-194: this is P.28's section; the file has only a merged heading.)

Checked, works as built by P.27:
- Cache names: `L<layer>.bfp4_mse.<hash>.flat.bf4.lay<layout>.{w_gu,w_d}.tensorbin` + `.done.json` with the stats,
  next to the bfp4 files with the same layout hash and sizes (2.55 + 5.10 GB a layer a box); no collision with bfp4.
- Load log: `EXPERTS_BFP4 layer=N impl=bfp4_mse matrices=w1,w3,w2 ...` and `EXPERTS_LOAD ... impl=bfp4_mse`,
  `MOE_BLOCK ... experts_impl=bfp4_mse`; `_bf4_mse_lossless` asserted on every matrix of all 40 layers (no failure).
- CPU emulation: `tt/experts.py:expert_weight_round` follows the value; bringup/hooks.py:devfmt_component and
  reference/ds41_ref.py's experts_w site do not (DS41-200; both outside P.28's paths, never gated).

Fixed: EXPERTS_IMPL's default was committed as bfp8 by P.27 (the perf template's "make the change the default"),
against the owner's "default stays bfp4" (F241). tt/settings.py: back to bfp4. The A/B sets the switch on both sides.

Added: tests/perf/test_experts_cache_warm.py (DS41-199): the first measure (cold) hit pytest's 3600 s at layer 19.

Load-time stats, all 40 layers (from the cache markers; bfp4 = the same layers' existing bfp4 markers):

| layer | bfp4_mse changed % | worst rel (expert, matrix) | bfp4 changed % | bfp4 worst rel |
|---|---|---|---|---|
| 0 | 7.75 | 0.0672 (79, w1) | 16.65 | 0.1196 |
| 1 | 7.67 | 0.0705 (9, w3) | 16.41 | 0.1179 |
| 2 | 7.30 | 0.0659 (163, w1) | 16.37 | 0.1181 |
| 3 | 7.16 | 0.0662 (154, w1) | 16.39 | 0.1198 |
| 4 | 7.21 | 0.0654 (244, w1) | 16.39 | 0.1190 |
| 5 | 7.26 | 0.0650 (23, w1) | 16.38 | 0.1183 |
| 6 | 7.28 | 0.0656 (111, w1) | 16.38 | 0.1193 |
| 7 | 7.23 | 0.0658 (171, w1) | 16.38 | 0.1189 |
| 8 | 7.28 | 0.0658 (0, w1) | 16.39 | 0.1200 |
| 9 | 7.32 | 0.0651 (75, w1) | 16.37 | 0.1183 |
| 10 | 7.43 | 0.0665 (219, w1) | 16.41 | 0.1191 |
| 11 | 7.46 | 0.0685 (295, w1) | 16.42 | 0.1199 |
| 12 | 7.53 | 0.0680 (70, w1) | 16.49 | 0.1215 |
| 13 | 7.59 | 0.0698 (193, w1) | 16.61 | 0.1235 |
| 14 | 7.56 | 0.0693 (146, w2) | 16.60 | 0.1226 |
| 15 | 7.12 | 0.0793 (48, w2) | 16.65 | 0.1313 |
| 16 | 7.61 | 0.0717 (189, w3) | 16.62 | 0.1261 |
| 17 | 7.61 | 0.0751 (27, w2) | 16.68 | 0.1277 |
| 18 | 7.53 | 0.0748 (161, w2) | 16.71 | 0.1290 |
| 19 | 7.39 | 0.0744 (289, w1) | 16.66 | 0.1288 |
| 20 | 7.49 | 0.0726 (367, w3) | 16.52 | 0.1253 |
| 21 | 7.45 | 0.0720 (215, w1) | 16.49 | 0.1230 |
| 22 | 7.38 | 0.0727 (184, w1) | 16.56 | 0.1247 |
| 23 | 7.19 | 0.0736 (196, w1) | 16.53 | 0.1259 |
| 24 | 7.38 | 0.0726 (80, w3) | 16.51 | 0.1246 |
| 25 | 7.24 | 0.0699 (135, w2) | 16.41 | 0.1249 |
| 26 | 7.17 | 0.0720 (31, w2) | 16.40 | 0.1258 |
| 27 | 7.16 | 0.0698 (305, w2) | 16.38 | 0.1252 |
| 28 | 7.10 | 0.0685 (7, w2) | 16.38 | 0.1233 |
| 29 | 7.08 | 0.0694 (69, w2) | 16.37 | 0.1244 |
| 30 | 7.08 | 0.0679 (127, w2) | 16.36 | 0.1241 |
| 31 | 7.24 | 0.0693 (18, w1) | 16.44 | 0.1243 |
| 32 | 7.08 | 0.0757 (111, w2) | 16.41 | 0.1255 |
| 33 | 7.08 | 0.0724 (19, w2) | 16.42 | 0.1295 |
| 34 | 7.13 | 0.0675 (31, w3) | 16.46 | 0.1213 |
| 35 | 7.43 | 0.0750 (55, w3) | 16.76 | 0.1287 |
| 36 | 7.21 | 0.0759 (7, w2) | 16.62 | 0.1324 |
| 37 | 7.08 | 0.0984 (123, w2) | 16.76 | 0.1676 |
| 38 | 6.81 | 0.0999 (250, w2) | 17.17 | 0.1702 |
| 39 | 6.81 | 0.1068 (7, w2) | 17.73 | 0.1781 |

Range: bfp4_mse 6.81-7.75 % changed, worst 0.065-0.107; bfp4 16.36-17.73 %, worst 0.118-0.178 (~0.58 x).

Change-on ladder, rung last (orchestrator measure --part last; err = rel to the golden, limit = the noise floor;
bfp4 / bfp8 = P.27's off / on sides of the same rung):

| metric | bfp4_mse | bfp4 | bfp8 | limit |
|---|---|---|---|---|
| err_final_hidden | 0.140 | 0.190 | 0.101 | 0.242 |
| err_layer_L09 | 0.101 | 0.126 | 0.081 | 0.157 |
| err_layer_L29 | 0.111 | 0.149 | 0.084 | 0.189 |
| err_layer_L38 | 0.126 | 0.165 | 0.096 | 0.220 |
| err_state_window_kv_L28 | 0.115 | 0.159 | 0.088 | 0.175 |
| top5_overlap | 1.0 | 1.0 | 1.0 | >= 0.994 |
| worst share of its limit | 0.89 (topk_idx_L08) | 0.91 (window_kv_L28) | | |

Host memory during the cold load: the pytest's anon RSS ~235 GiB between layers, ~297 GiB at a layer's peak, ~1 GiB
growth a layer (MemAvailable >= 167 GiB).

Re-run: `DS41_EXPERTS_IMPL=bfp4_mse PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile
models/demos/deepseek_v41_flash_d_p/tests/perf/test_experts_cache_warm.py -s` (if the cache is cold), then
`python -m models.demos.common.bringup.orchestrator measure --task P.28 --part last`.

## P.29 EXPERTS_BFP8_DOWN4: wiring check, cache warm, DRAM at 1M, change-on measure

Perf pick P.29 (galaxy 8x4, 2026-11-03; DS41-198, DS41-199, DS41-200). No code change: P.27's value checked.
(BREADCRUMBS.md is at the 500 KB hook limit, DS41-194: this is P.29's section; its heading there reads FIX.20/P.27-29.)

Checked, works as built by P.27:
- Cache names: `L<layer>.bfp8_down4.<hash>.flat.bf8d4.lay9c3da689dd35.{w_gu,w_d}.tensorbin` + `.done.json`
  (9.63 + 2.55 GB a layer a box, 454 GB for 40 layers); own names, no collision with bfp4 / bfp8 / bfp4_mse.
  P.27's layer 0 file hit (the warmer's arguments give the model's layout hash).
- Load log: `EXPERTS_BFP4 layer=N impl=bfp8_down4 matrices=w2 ...`, `EXPERTS_LOAD ... impl=bfp8_down4 cache=hit`
  (4-5 s a layer), `_bf4_mse_lossless` held on every w2 of the 40 layers (warm rc 0).
- CPU emulation: `tt/experts.py:expert_weight_round("bfp8_down4")` = w1 / w3 identity, w2 `_bfp4_mse_round`.
  bringup/hooks.py:1074 devfmt_component and reference/ds41_ref.py's experts_w still do not follow it (DS41-200,
  outside P.29's paths, never gated).
- Default stays bfp4 (F241): tt/settings.py unchanged.

Warm: `DS41_EXPERTS_IMPL=bfp8_down4 ... tests/perf/test_experts_cache_warm.py -s`, 39 misses at 133-149 s a layer
(host 41-42 s), 1 h 35 min.

Load-time stats (w2 only: the bfloat4_b matrix; bfp4 = all three matrices, its own markers):

| layer | changed % | worst rel (expert) | bfp4 changed % | bfp4 worst |
|---|---|---|---|---|
| 0 | 7.62 | 0.0645 (79) | 16.65 | 0.1196 |
| 1 | 7.66 | 0.0669 (9) | 16.41 | 0.1179 |
| 2 | 7.28 | 0.0634 (159) | 16.37 | 0.1181 |
| 3 | 7.21 | 0.0637 (319) | 16.39 | 0.1198 |
| 4 | 7.27 | 0.0638 (15) | 16.39 | 0.1190 |
| 5 | 7.35 | 0.0640 (70) | 16.38 | 0.1183 |
| 6 | 7.34 | 0.0653 (359) | 16.38 | 0.1193 |
| 7 | 7.27 | 0.0649 (183) | 16.38 | 0.1189 |
| 8 | 7.34 | 0.0640 (322) | 16.39 | 0.1200 |
| 9 | 7.40 | 0.0648 (80) | 16.37 | 0.1183 |
| 10 | 7.47 | 0.0651 (272) | 16.41 | 0.1191 |
| 11 | 7.49 | 0.0678 (35) | 16.42 | 0.1199 |
| 12 | 7.50 | 0.0658 (212) | 16.49 | 0.1215 |
| 13 | 7.53 | 0.0677 (217) | 16.61 | 0.1235 |
| 14 | 7.41 | 0.0693 (146) | 16.60 | 0.1226 |
| 15 | 6.99 | 0.0793 (48) | 16.65 | 0.1313 |
| 16 | 7.37 | 0.0712 (189) | 16.62 | 0.1261 |
| 17 | 7.38 | 0.0751 (27) | 16.68 | 0.1277 |
| 18 | 7.25 | 0.0748 (161) | 16.71 | 0.1290 |
| 19 | 7.18 | 0.0738 (289) | 16.66 | 0.1288 |
| 20 | 7.29 | 0.0718 (367) | 16.52 | 0.1253 |
| 21 | 7.31 | 0.0676 (343) | 16.49 | 0.1230 |
| 22 | 7.20 | 0.0702 (116) | 16.56 | 0.1247 |
| 23 | 7.15 | 0.0724 (196) | 16.53 | 0.1259 |
| 24 | 7.21 | 0.0695 (116) | 16.51 | 0.1246 |
| 25 | 7.08 | 0.0699 (135) | 16.41 | 0.1249 |
| 26 | 7.15 | 0.0720 (31) | 16.40 | 0.1258 |
| 27 | 7.16 | 0.0698 (305) | 16.38 | 0.1252 |
| 28 | 6.98 | 0.0685 (7) | 16.38 | 0.1233 |
| 29 | 7.01 | 0.0694 (69) | 16.37 | 0.1244 |
| 30 | 7.04 | 0.0679 (127) | 16.36 | 0.1241 |
| 31 | 7.21 | 0.0687 (256) | 16.44 | 0.1243 |
| 32 | 6.98 | 0.0757 (111) | 16.41 | 0.1255 |
| 33 | 6.99 | 0.0724 (19) | 16.42 | 0.1295 |
| 34 | 6.99 | 0.0653 (31) | 16.46 | 0.1213 |
| 35 | 7.11 | 0.0744 (55) | 16.76 | 0.1287 |
| 36 | 6.93 | 0.0759 (7) | 16.62 | 0.1324 |
| 37 | 6.75 | 0.0984 (123) | 16.76 | 0.1676 |
| 38 | 6.51 | 0.0999 (250) | 17.17 | 0.1702 |
| 39 | 6.70 | 0.1068 (7) | 17.73 | 0.1781 |

Range: 6.51-7.66 % of w2 changed, worst 0.063-0.107 (the same w2 matrices as bfp4_mse's worst on the late layers).

DRAM per chip at 1M (test_dram, change on; bfp4 / bfp8 = P.27's table, capacity 31.831 GiB, budget 27.2):

| GiB a chip | bfp4 | bfp8_down4 | bfp8 |
|---|---|---|---|
| routed experts | 8.899 | 14.172 | 16.809 |
| after load | 18.454 | 23.727 | 26.364 |
| peak (layer 36) | 19.634 | 24.907 | 27.544 |
| chunk at 1,039,360 (s) | 185 | 145 | 146 |

bfp8_down4 fits the 27.2 GiB budget at 1M (2.29 GiB under at peak; P.27's estimate 14.1 / 24.9 held).

Change-on ladder, rung last (measure; limit = noise floor):

| metric | bfp8_down4 | bfp4_mse | bfp4 | bfp8 | limit |
|---|---|---|---|---|---|
| err_final_hidden | 0.117 | 0.140 | 0.190 | 0.101 | 0.242 |
| err_layer_L09 | 0.089 | 0.101 | 0.126 | 0.081 | 0.157 |
| err_layer_L29 | 0.094 | 0.111 | 0.149 | 0.084 | 0.189 |
| err_layer_L38 | 0.106 | 0.126 | 0.165 | 0.096 | 0.220 |
| err_state_window_kv_L28 | 0.092 | 0.115 | 0.159 | 0.088 | 0.175 |
| top5_overlap | 1.0 | 1.0 | 1.0 | 1.0 | >= 0.994 |

Worst share of a limit: 0.88 (err_state_topk_idx_L08 0.151 / 0.172). Profile (change on, the measure's own):
device 182.8 ms, chunk wall 202.0 ms; p51200 chunk wall 210.0 ms (spread 23.4), all four parts rc 0. The off side
and the comparison are the orchestrator's A/B.

Re-run: warm (above) if the cache is cold, then `python -m models.demos.common.bringup.orchestrator measure --task P.29`.

## P.30 DECODER_SKIP=exact: the decoder skipped on every chunk but the request's last

Owner pick 2026-11-03 (arXiv 2609.19969, exact version). Switch `DECODER_SKIP` (tt/settings.py, off | exact,
default off: the owner decides; the A/B sets it on both sides).

What changed (tt/ only, no fork, no new Python-built op: OV-23 has nothing to prepare):
- `TtDS41Model.begin_chunk(tokens, start, last=True)`: the per-chunk argument (the caller knows; the model never
  infers it and never pulls back). Default True = every layer, today's path; ignored under off. The framework harness
  (ladder, profile) calls `begin_chunk(tokens, start)`, so the gate's rung `last` and profile take last=True and run
  the same work on both sides; only tests/bringup/test_decoder_skip.py and tests/perf/test_decoder_skip_time.py
  pass False. `tt/runners/adapter.py` `prefill_chunk(.., last=True)` passes it through (tt-d-gen's header has no flag yet).
- Decoder = the loaded layers from the candidate source on (20-39). On a skipped chunk `layer(20)` runs
  `TtDS41Block.kv_only` (attn_hc, attn_collapse, attn_norm, compressor: the compressed KV and index keys, written
  into the served caches and the attention table as in the full block, the same modules); layers 21-39 return the
  `Skipped` marker; every decoder layer's `TtWindowCache.skip(start, end)` notes the positions as missing (host only)
  and drops the kept tail. `final_norm(Skipped)` returns it.
- Mask (DS41-201): `TtWindowCache.carry_masked(start)` (the carry [start - 128, start) touches `missing`) ->
  `carry_class(start)` = 0: TtWindowAttention / TtSparseAttention use the window index lists of position 0 (positions
  before the chunk dropped, sentinel tail), and no carry is read (`carry`, `carry_blocks`, `carry_owner`, the region's
  chip-0 carry return zero / none). The index memo key carries the class. A zero carry alone would not be a mask (a zero
  key still takes softmax weight). `computed` (after a full run) and `load_host` / `load_ring` / `free` trim or clear
  `missing`, so a golden prefix (rung last) is not masked: the gate's ladder is the off path's.

Exactness: decoder layer 20 + k is exact from chunk start + (k + 1) x 127 (window 128); the last token and every
layer's last 128 window rows need a last chunk >= 20 x 127 + 128 = 2668 rows (the caller's >= 2688).

Accuracy (test_decoder_skip.py, s10240: chunk 0 last=False, chunk 1 last=True; limits calib x the rung's drift):
skip and off agree to every printed digit on the checked rows (the class-0 lists equal class 128's from row 127 on).
| metric | skip | off | limit |
|---|---|---|---|
| skip_err_final_hidden (rows [7660, 10240)) | 0.19732 | 0.19732 | 0.25615 |
| skip_top5_overlap (191 sampled rows) | 1.0 | 1.0 | >= 0.9943 |
| worst skip_err_window_kv (L27) | 0.17080 | 0.17080 | 0.21494 |
| skip_err_window_kv_L39 | 0.15318 | 0.15318 | 0.22105 |
| skip_err_compress_kv_L20 / index_k_L20 | 0.18861 / 0.11569 | same | drift limits |
Top-1 1.0 over the rows, last token top-1 match. Structural checks pass (decoder window rows of chunk 0 zero on the
skip path, layer 20's compressed KV of chunk 0 written).

Timing (tests/perf/test_decoder_skip_time.py, 8x4, one model load (216 s), a fresh state per start, random ids,
steady_state(start), 5 warm synced runs (median) + one rt-profiler run (device: every program, slowest chip a call)):
| chunk at | nonlast (last=False): wall / device ms | last (after skipped chunks, carry masked) | full (= off) |
|---|---|---|---|
| 51200 | 108.3 / 83.61 | 194.9 / 167.06 | 194.5 / 167.89 |
| 512000 | 112.7 / 91.83 | 225.4 / 205.89 | 229.8 / 206.00 |
A non-last chunk costs 50 % (51200) / 55 % (512000) of the full chunk's device time and 56 % / 49 % of its wall;
the masked last chunk equals the full one (its decoder layers read no carry: -0.8 / -0.1 ms device). A prompt of n
chunks takes (n - 1) non-last + 1 last chunk (plus the caller's pull-back of a last chunk shorter than 2688).
The gate's ladder / profile pass no `last` (framework harness): both A/B sides run the full chunk there, so the A/B
table shows no difference by design; these numbers are the change.

Gate units: test_decoder_skip PASS (gate --check); measure --part test (test_decoder_skip) rc 0, --part last (ladder,
change on) rc 0 (final hidden 0.190 / 0.242, top5 1.0). Re-run: the two test commands in their docstrings; `measure
--task P.30`.
Attempt 2 (2026-11-03): no code change. DS41-201's bullet got its Found: and its detail section (the knowledge check refused attempt 1). `measure --task P.30` rc 0, every part reused on the same code: test_decoder_skip, last (final hidden 0.190 / 0.242, top5 1.0), profile (159.79 ms device, 199.4 ms wall), p51200 (200.0 ms wall), p512000 (228.9 ms wall); the ladder and profile run last=True on both sides (framework harness), so they show no A/B difference by design.

## Masking a window carry whose rows were never computed: start class 0, not a zero carry

DS41-201 (P.30 DECODER_SKIP=exact). On a request's last chunk after skipped chunks, the decoder layers (20-39) have no
window rows for the positions before the chunk. Zeroing the carry is not a mask: a zero key still enters the softmax
(score 0, weight exp(-max)) and dilutes the real rows. The mask is the window index lists of position 0 (class 0): they
drop every position before the chunk start (sentinel tail), and no carry is read (`carry`, `carry_blocks`,
`carry_owner`, the region's chip-0 carry return zero / none). `TtWindowCache.missing` (host-only interval set by
`skip(start, end)`) drives `carry_masked(start)` -> `carry_class(start)` = 0; `computed`, `load_host` / `load_ring`,
`free` trim or clear it, so a golden prefix is never masked. The index memo key carries the class.
Result (s10240, chunk 0 last=False, chunk 1 last=True): the class-0 lists equal class 128's from row 127 on, so decoder
layer 20 + k is exact from start + (k + 1) x 127; the checked rows agree with the off path to every printed digit
(final hidden 0.19732 both, limit 0.25615; worst window KV L27 0.17080 both, limit 0.21494; top-5 1.0).
Test gotcha: `testing/e2e_limits.check` takes the limit kind from the metric's `err_` prefix; a `skip_err_*` name gets
the overlap limit (KeyError 'overlap'), so the test calls `err_limit` / `overlap_limit` directly.

## P.31 DECODER_SKIP=replay128: the decoder tail and the short turn

Owner pick 2026-11-03 (DeepSeek-V4.1-Flash's SWA Bounded Replay, arXiv 2609.19969, n_win 128; approximate by design;
default off, ab.keep_default). Third value of P.30's switch `DECODER_SKIP` (off | exact | replay128).

What it does (tt/model.py; one code path with exact but the tail length):
- `begin_chunk(tokens, start, last=None, replay=False)`. `last` None (new default: not told, the framework harness and
  the engine) runs every layer in every value (= P.30's True under exact). False skips the decoder (P.30). True under
  replay128 runs the decoder only on the chunk's last <= 128 rows. `replay=True` (a request's first chunk on a cached
  prefix, started `model.replay_rows` before its new tokens): under replay128 every layer's window is truncated at the
  chunk start (`TtWindowCache.truncate`: the carry masked as P.30's `missing`, start class 0); under exact a no-op.
  The caller's numbers: `replay_rows` exact 0 (the prefix's last chunk ran every layer: its window rows exist),
  replay128 128; `last_rows` exact 2688 (P.30's pull-back of a short last chunk), replay128 0 (no pull-back).
- The decoder tail (`build_tail`, called at load before the first chunk; lazily by the first tail chunk else): the
  decoder layers 20-39 as a TtDS41Model of their own (`tail_of`), chunk `tail_chunk` = 32 SP TP rows rounded up to the
  indexer's top-k keys (1024 on 8x4 = one tile row a chip: the 128 rows sit on mesh row 0, the rest of the chunk is the
  pad past actual_end; 512 on 2x2 / 4x4 / 2x4). Its own window tables, rope rows, attention / indexer / norm / mHC
  weights (duplicated: ~2.4 GiB a chip for 20 layers' attention, not measured), its MoE the main layer's at the tail
  chunk (`TtMoEBlockLayer.at_chunk` -> `MoELayer.at`: the routed experts' device weights shared, the flat op replanned),
  no compressor (layer 20's compressed KV / index keys come from the main state: `DS41DeviceState(compress=)` shares
  them). Its indexers score keys the 5120-row chunks laid out: `TtIndexer.key_rows` + the fork option
  `ring_indexer_score_dsa(natural_query_rows=)` (indexer_score CHANGELOG, P.31). The window path below 128 rows a chip
  is bfp8_blocks (DS41-204). The sparse attention's window lists are the layer's own tables' (`_index_static`: base-0
  table).
- The request's last chunk under replay128: layers 0-19 over the chunk; layer 20 `kv_only` over the chunk (P.30's KV
  projection: every compressed row and index key); then dec = ceil32(end - 128), rows [dec, end) moved to the tail's
  load-time input by `tt/tail_send.py:TailSend` (2D fabric, prepared, one program a (offset, pieces) pattern; DS41-203);
  the tail's layers 20-39 at start dec with every window truncated there; `final_norm` returns a `TailHidden` (the
  tail's rows, positions from `.start`), `to_host` / `logits` / `free` take it. The tail's own state (`state.tail_state`)
  holds its window caches (tail geometry: NOT the served 0_window_kv layout; decode would need them in the server's;
  window KV is not gated for replay128, owner) and its top-k / candidate stores.
- L1: the main and the tail MoE blocks coexist (DS41-202): `reserve_l1` 48 KB at the top before the main block is
  built; the tail block's 41216 B of persistent words go there (checked: no new allocated L1 block below the
  reservation; logged). MoEOptions.one_arena=False for the tail block (at 1024 tokens its exchange arena would start at
  187584, below a head program's static CBs ending at 193920 on core (10, 0)), and the head's arena re-settled after the
  exchange build (moe_block.py; a no-op when it lands where baked).
- Runner (tt/runners/adapter.py): `last` defaults to None (the engine never passes it): every layer, as before.

The gate's ladder / profile pass no `last` (framework harness): both A/B sides run the full chunk there (no difference
by design, as P.30). The change is measured by tests/bringup/test_short_turn.py (gate unit P.31#test_short_turn):

| short turn: 512 tokens on a 51200-token cached prefix (rung last's golden state), warm | exact | replay128 |
|---|---|---|
| chunk | [51200, 51712) | [51072, 51712) (128 replayed) |
| wall (median of 5) ms | 224.1 | 216.4 |
| device (rt profiler) ms | 178.27 | 142.62 |
| final hidden rel at the last position (limit 0.2422) | 0.1339 | 0.1985 |
| top-5 / top-1 at the last position | 1 / 1 | 1 / 1 |
| final hidden rel over the last 128 rows (diagnostic) | 0.1776 | 0.5388 |
The 128-row error is the truncated windows (the tail's first rows see few window keys; the last row sees 128).
Change-on ladder rung last via measure: rc 0, final hidden 0.1902 / 0.2422 (= P.30), top-5 1.0.
Fork test test_natural_query_rows (8x4): 128 (start, device) pairs, min PCC 0.9999984, worst rel 0.00193.
Load: model 225-260 s, tail 50 s (in test_short_turn; not in the ladder / profile, which never build it).

Gotchas: `ttnn.point_to_point` is 1D-only (DS41-203). The indexer's validation tied the queries' rows to the key stripe
twice (the q_isl check and the causal geometry): the fork option covers both. MoE blocks: see DS41-202.
Re-run: `DS41_DECODER_SKIP=replay128 DS41_ENGRAM_TABLE=resident scripts/run_safe_pytest.sh --run-all --no-precompile
models/demos/deepseek_v41_flash_d_p/tests/bringup/test_short_turn.py -s` (exact: DS41_DECODER_SKIP=exact);
`BRINGUP_TEST_MESH=8x4 scripts/run_safe_pytest.sh --run-all
ttnn/ttnn/bringup/indexer_score/tests/unit/test_natural_query_rows.py`; `measure --task P.31`.
OV-3: nothing padded to get around a limit (the tail chunk is the smallest whole-tile SPMD chunk; its rows past
actual_end are the model's usual pad). OV-23: one Python-built op added (TailSend), prepared (prepare_program /
run_prepared, one program a pattern; host time not separated). OV-25: no fork suite run; only the new fork test.

## Two MoE blocks (two chunk sizes) on one mesh: the second's persistent L1 lands inside the first's arenas

DS41-202 (P.31). An MoEBlock (models/demos/moe_block) keeps persistent words at the top of L1 (semaphores, mailboxes,
the exchange's words) and bakes two per-call arenas into its kernels: the fused head's and the exchange's (the
exchange's = the largest contiguous L1 left less a reserve, tt_k3_token_split_moe.py). A second block (the decoder
tail's, 1024 tokens) built after the main one puts its words below the main block's, i.e. inside the main block's
arenas; and its own head arena moved by the 24 KB hole its exchange left unused ("se_engine3: the arena landed at
674368, not 649792"). Sequence of failures seen on 8x4: (1) one arena a forward: the tail head's static CBs on core
(10, 0) end at 193920, the exchange arena starts at 187584 (TT_THROW program.cpp:2641); (2) two arenas: the head
arena 24 KB off; (3) with a re-settle: the exchange's arena 24 KB off ("landed at 163008, not 187584").
Fix: reserve 48 KB of L1 at the top before the first (main) block is built (`moe_block_adapter.reserve_l1`, from
`TtDS41Model.build_tail`), free it just before the tail block is built: the tail's words fill it (41216 B measured),
both blocks bake their arenas in the same L1 left and take them in turn. The block checks every new allocated L1 block
lies inside the reservation (allocator block table, DS41-114) and logs its size. Cost: the main block's exchange arena
is 48 KB smaller (only when the tail is built: the ladder / profile never reserve).

## replay128's short turn: device -20 %, wall -3 %: the chunk is host-bound

DS41-205 (P.31). The table is in "P.31 DECODER_SKIP=replay128" above (exact 224.1 / 178.27 ms, replay128 216.4 /
142.62 ms wall / device, 8x4, 512 tokens on a 51200-token prefix). Device: the encoder half (~89 ms) + layer 20's KV
projection + the tail's 20 layers at 1024 rows (~50 ms, ~2.5-2.9 ms a layer vs ~4.2 at 5120: small-shape ops and the
collectives are latency-bound). Wall: the host dispatches the same ~40 layers (~5 ms a layer, DS41-99), plus kv_only's
steps, the two prepared tail sends and the tail MoE's second L1 arena a layer (one_arena off). The lever left is the
host (trace), not the tail's device time.

## FIX.22 (overseer, 8x4, 2026-11-04): EXPERTS_IMPL default bfp4 -> bfp4_mse
- What: tt/settings.py EXPERTS_IMPL default `bfp4` -> `bfp4_mse` (owner 2026-11-04) and its description's owner-decision
  line; nothing else. Why: P.28's A/B, err_final_hidden 0.140 vs 0.190 for bfp4, same bytes, time within noise (DS41-198).
- Cache: tt_cache/8x4/v1/experts-5254448103b6/8x4 holds 40 `L*.bfp4_mse.*.done.json` markers; the ladder's load logged
  `cache=hit` for all 40 layers (0 misses), 1.2-2.8 s a layer (DS41-199: a miss would be 145-195 s).
- Gate `FIX.22 --check`: PASS (1543 s). Rung last: err_final_hidden 0.1399 (limit 0.2422), top5_overlap 1.0 (>= 0.9943),
  err_layer_L39 0.622 (limit 1.533), every err_layer / err_state ok. Profile wall: median 194.6 ms over 10 warm runs
  (program cache 174 -> 174, 0 gap uploads).
- Known gap (DS41-200): bringup/hooks.py devfmt_component still rounds expert weights only for `bfp4`, so the F137
  CPU diagnosis models exact weights under the new default. It never gates; hooks.py is outside this step's paths.
- OV-3: nothing padded. OV-23: no Python-built op added or touched. OV-25: no fork suite run.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate FIX.22 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check`.

## P.32 WINDOW_KV_KEEP: full | ring | none, the window KV a slot keeps

Owner pick 2026-11-04 / 11-05 (the memory half of DeepSeek-V4.1-Flash's bounded window, arXiv 2609.19969). Switch
`WINDOW_KV_KEEP` (tt/settings.py, full | ring | none; default full until the owner decides; the A/B sets it).

What changed (tt/ only; no fork, no Python-built op):
- `tt/attention.WindowRing`: [1, 1, 192, 512] bf16 ROW_MAJOR a layer on every chip, position p at row p mod 192
  (32-row blocks), host bookkeeping per block (first position, valid end, owner chip or every chip). Written with
  `ttnn.slice` + `ttnn.bringup.slice_write` (one or two pieces: the wrap).
- `TtWindowCache(keep=, ring=)`: ring = a ring of its own, no served slab; none = the model's working ring of the layer
  (`WindowTables.shared_ring`, shared by every slot of the model), readable only by the chunk right after this cache's
  own last write. Writes: `write` (replicated rows: every chip), `write_local` (P.11: the SP row's block, owner = the
  row's chip (r, 0)), `write_own` (new, the bfp8_tiles / bf16_tiles region path: each chip its own rows, owner = the
  chip; the served write's TP all_gather and update_padded_kv_cache are gone on that path). A chunk writes
  [max(start, b - 192), b), b = ceil32(end): a follow-up at align_down(end, 64) finds [start - 128, start) there.
  The chunk's write comes before its carry read in every path, so the write first takes the carry from the ring when
  the chunk will read it (`_before_ring_write` -> `_pending`, handed out by `carry`); blocks held by one chip are
  gathered from it (the same local rows of every chip, all_gather over the mesh, the owner's kept). Kept tails (rtail,
  P.11 blocks, GLUE_FOLD rows) still carry whole consecutive chunks (no ring read). `carry_owner` is None (no slab).
  `load_host` / `load_ring` / `to_host` / `ring_host` / `served_bytes` read or write the ring alone.
- `TtDS41Model.begin_chunk` -> `_window_policy` (ring / none, host only): every layer whose window the chunk reads
  (not the decoder on a skipped chunk or replay128's tail chunk) must have its carry kept (`carry_ready`: position 0,
  a P.30 `missing` mask, a kept tail, the ring); else DECODER_SKIP=exact / replay128 truncate that window at the start
  (P.31's `truncate`: start class 0) and off raises ValueError, except on a slot that never held a window row
  (DS41-206). `replay=True` under replay128 or none truncates every read window (a request on a cached prefix keeps
  nothing under none). The decoder tail state (P.31) takes `none` (its windows are truncated at every tail chunk).
- `DS41DeviceState(keep=)` / `new_state(keep=)`, `window_bytes()`; `TtDS41Model.skip_mode`, `use_decoder_skip(mode)`
  (tests: a model loaded with replay128 runs exact / off too). `tt/runners/kv_contract.DS41ContractKV` holds no
  0_window_kv under ring / none (`served_configs(window=False)`); `tt/runners/adapter.py` refuses ring / none
  (tt-d-gen reads every position; its change is pending).

Accuracy (tests/bringup/test_window_ring.py, s10240's tokens: prompt [0, 5120) last=False + [5120, 8021) last=True,
follow-up to 10240 from align_down(8021, 64) = 8000 (ring) or pulled back (none: exact 5312, replay128 7872, replay);
full beside it as P.30 / P.31 run it; limits calib x the s10240 drift):

| | final hidden (limit 0.25615) | top-5 (>= 0.9943) | worst gated window KV / its limit |
|---|---|---|---|
| exact, rows [8000, 10240): full | 0.14427 | 1.0 | L39 0.596 |
| exact: ring | 0.14427 | 1.0 | L39 0.596 (every layer = full to 5 digits) |
| exact: none (truncated at 5312) | 0.17330 | 1.0 | L19 0.694 |
| replay128, last position: full (replay from 7872) | 0.25290 | 1 | L9 0.550 |
| replay128: ring (from 8000, the encoder's carry kept) | 0.23748 | 1 | L17 0.557 |
| replay128: none (= full's path) | 0.25290 | 1 | L9 0.550 |

Window KV a chip a slot: full 1.251 GiB at 1M (29.5 MB at the test's 15360 with replay128's tail caches), ring 7.50 MiB
(40 x 192 KiB), none 0 (the model's working rings: 11.25 MiB, 60 layers incl. the tail model's, once per model).
Refusal under off: ring at 2048 (older than the ring) and none at 4992 (a follow-up) raise in begin_chunk; ring at 5056
(held) is accepted. Ladder rung last via measure (ring): rc 0, final hidden 0.1399 / 0.2422 (= full, FIX.22), top-5 1.0.

Timing (8x4, test_profile through measure / gate --check --unit, ring on unless said; wall = clean median of 10 warm
runs, device = rt profiler; the timed runs take the kept tail, so the ring's cost there is its write):
| chunk | ring, 110-core copies | ring, WINDOW_RING_CORES=8 | full, same code (gate unit, DS41_WINDOW_KV_KEEP=full) |
|---|---|---|---|
| plain [51200, 56320) | 210.6 / 163 ms | 206.4 / 161 ms | wall median 282.2 (disturbed: runs 191.7-204.9 plus four of 282-371) / 161 ms |
| p51200 last (replay128) | 218.3 / 136 | (final measure) | |
| p51200 nonlast (exact) | 120 / 87 | (final measure) | |
| p512000 last / nonlast | 230 / 157, 123 / 97 | (final measure) | |
Device time is the full path's (161 vs 161 ms): the region path drops the served write's TP all_gather and
update_padded_kv_cache and adds the ring's untilize + 2-4 small copies a layer. The host pays ~80 us a stock copy
on 32 chips at 110 cores (cProfile, p51200 last: WindowRing.put 100 calls a chunk); one row of 8 cores cut 4 ms of
wall. Each layer's steady write is the last chip's 160 own rows (one or two pieces: the ring's wrap) and the 32 rows
before them (the previous chip's): a single-op write needs one chip holding the 192 rows (e.g. the bfp8 region of
the last chip, a bfp8 ring: 40 x 6 x 18496 B = 4.4 MB, the owner's ~5 MB) - not done here (a format change of the kept
rows; the ring keeps the served cache's bf16).

DRAM per chip, test_dram (1M, one slot), ring via measure vs P.27's full (bfp4; P.28-29 left the window caches as they
were): after load 17.210 vs 18.454 GiB (served caches 0.156 vs 1.408: 0_window_kv gone; slot state 0.794 vs 0.786:
+7.5 MiB of rings), peak 18.389 vs 19.634 GiB, routed experts 8.899 (bfp4_mse, same bytes).

Gotchas: a chunk at start s writes a whole chunk's table rows, so a test state needs max_seq >= s + chunk (K-69).
Re-run: `DS41_ENGRAM_TABLE=resident scripts/run_safe_pytest.sh --run-all --no-precompile
models/demos/deepseek_v41_flash_d_p/tests/bringup/test_window_ring.py`; host check of the ring bookkeeping (fake ttnn,
every write path, 4 meshes): /tmp/p32/sim_ring.py; `measure --task P.32`.
OV-3: nothing padded. OV-23: no Python-built op added or touched. OV-25: no fork suite run.

## A window keep mode that refuses missing carries must let a never-written slot start anywhere

DS41-206 (P.32). The frozen test_dram runs one chunk at 1,039,360 on a fresh slot and the F230 position profiles one
at 51200 / 512000 on a fresh state; under full they read the never-written served cache (zeros). Under ring / none
with DECODER_SKIP=off the owner's rule refuses a start whose carry is not kept, which would fail them.
`TtWindowCache.history` (set by a write or a load) tells a slot that never held a window row: begin_chunk truncates
its window at the chunk (no row read, start class 0) under every DECODER_SKIP value, and refuses only a slot that held
rows but not this carry. The profiles' timed runs (steady_state) take the kept tail as under full.

## replay128's decoder-tail window KV exceeds the e2e limits from layer 26-28 on, equally under full

DS41-207 (P.32, test_window_ring, replay128, the window KV of rows [10112, 10240) at 10240 vs the s10240 golden):
full / ring / none L26 0.145 / 0.144 / 0.145 (limit 0.158), L28 0.181 / 0.181 / 0.181 (0.180), L31 0.194 / 0.193 /
0.194 (0.148), L35 0.336 / 0.345 / 0.336 (0.304), L39 0.322 / 0.323 / 0.322 (0.221). The decoder tail's windows are
truncated to its own 128 rows (P.31, approximate by design; window KV not gated for replay128, owner): recorded as
ring_diag_window_kv_L<i>_<keep>_replay128 beside full's; layers 0-19 are gated (worst 0.56 of the limit).

## P.32 attempt 2: the final measure, and the position profiles' re-run (DS41-208)

Attempt 1 was cut short (rc -9) during its final `measure --task P.32` (code stamp 3a6db08f5e44f5e3, ring on); the
measure ran to the end. On that code: ladder rung last rc 0 (final hidden 0.1399 / 0.2422, top-5 1.0);
test_window_ring rc 0 (final hidden ring_exact 0.1443, ring_replay128 0.2375, none_exact 0.1733, none_replay128 0.2529,
limit 0.2561; top-5 1.0 in all four; 120 gated window-KV values, worst 0.69 of its limit (L19 none_exact); served
window bytes a chip: full 1.343 GB at 1M, ring 7.86 MB, none 0 (+ 11.8 MB working rings a model)); test_dram rc 0
(load 17.210 GiB, peak 18.389 GiB, served caches 0.156 GiB); profile rc 0 (plain [51200, 56320): wall 215.9 ms,
device 158.8 ms; p51200 last 223.5 / 134.4, nonlast 120.6 / 87.2; p512000 last 209.0 / 155.5, nonlast 117.6 / 95.8).
The A/B report's F230 rungs p51200 / p512000 (DECODER_SKIP=off) failed: their first command (the wall) passed, the
WALL_CLEAN command raised in begin_chunk ("the chunk at 51200 needs the window rows [51072, 51200) ... not kept
(layer 0 keeps positions [56128, 56320])") at testing/profile.py:592, the counting run after
`accurate_reload()` = `steady_state(None)`. A position profile's state is fresh (load_prefix does nothing) and every
run repeats the same chunk on it: under full the re-run reads the same never-written zeros; under ring the slot now
holds rows (history), so DS41-206's exemption no longer applied.
Fix: `TtWindowCache.origin`, the start of a fresh slot's first chunk (set in `TtDS41Model._window_policy` while
`history` is False; cleared by `load_host` (a loaded prefix existed before) and `free`). Under DECODER_SKIP=off a chunk
whose carry is not kept is refused unless start <= origin: a restart of the slot's window where it began (truncated
there, as its first run). A slot from position 0 has origin 0, so another request's prefix, a resume older than the
ring and a none follow-up are refused as before (test_window_ring's refusal cases start at 0). Host check:
/tmp/p32/sim_policy.py (fresh 51200 ok, re-run 51200 ok, kept 56320 ok, 60000 refused; from 0 then 2048 refused under
off, truncated under exact).
On the fixed code (measure, ring on): p51200 rc 0 (wall 209.3 ms), p512000 rc 0 (wall 231.6 ms), test_window_ring rc 0
(refusal ring: 2048 refused, 5056 accepted; none: 4992 refused). last, test_dram and profile were measured on the code
before this host-only change (rc 0, above); the orchestrator's A/B re-runs them.

## Window KV rows away from the rung's end: the golden keeps only rings; recompute them from the dumped inputs

DS41-209: see "Golden rows before the cut (DS41-209)" in the next section (P.33 DECODER_CUT_TEST).

## P.33 DECODER_CUT_TEST: prefill stopped at an even cut 128 tokens short of the end (BREADCRUMBS section)
- What: test only, no model change. `tests/bringup/test_decoder_cut.py`: rung s10240 tokens, all 40 layers,
  DECODER_SKIP=exact, prefill [0, 5120) and the partial chunk [5120, 10112) (cut = (10240 - 128) // 2 * 2), both
  `begin_chunk(.., last=False)`, end passed to every `model.layer`, ids past the cut PAD_ID. Checks at the cut: the
  encoder windows (layers 0-19, rows [9984, 10112) via `ring_host(cut)`) and layer 20's compressed KV / index keys over
  [0, 10112), limits `e2e_limits.err_limit` on state_window_kv_L<i> / state_compress_kv_L20 / state_index_k_L20.
- Answer to the brief's question: the model CAN end a prefill at a cut (an end not on a chunk boundary, last=False):
  nothing had to change. Gate (8x4, 394 s incl. a ~200 s load): PASS, 22/22. Window KV L00 0.0049 (limit 0.058) ...
  L18 0.150 (0.271); worst L14 at 0.563 of its limit; compress_kv_L20 0.180 (0.323), index_k_L20 0.112 (0.311).
- Golden rows before the cut (DS41-209): the golden's window_kv is the ring at 10240 (rows [10112, 10240)), snapshots
  only at 0 / 5120 / 10208; chunk_01 dumps the `kv` step only for the step-dump layers (0, 1, 2, 3, 14, 20, 21, 24).
  The other encoder layers' rows are recomputed on the CPU: DS41Reference(numerics native, layers=[]) with only
  attn_norm.weight / attn.wkv.weight / attn.kv_norm.weight, attn_collapse -> attn_norm -> kv on the previous layer's
  dumped `out` rows (attn_collapse takes `pre` from the packed input; attn_hc is not needed). Bit-identical to the dump
  on layers 2 / 3 (rel 0, gated in the test at <= one rounding) and to the 10208 snapshot's overlap (6e-7). 1.7 s.
- Facts for decode: the decoder layers (20-39) hold no window row of [0, cut) (checked); layer 20's compressed KV has
  0 non-zero rows past the cut (pad rows not written; `cut_diag_ckv_rows_past_cut`). The limits for the window rows
  are the drift of the 128-row ring at the rung's end (adjacent rows), not of [cut - 128, cut) itself.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup gate P.33 --spec
  models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml --check` (via devjob).

## P.34 POST_COLLAPSE (perf, 8x4, 2026-11-06)
- **What**: setting POST_COLLAPSE (default on). A residual whose mhc_post already emits the next hc step's stats row now
  also writes that step's inputs: ffn_hc's from attn_residual, the next layer's attn_hc's from ffn_residual, an engram
  layer's attn_hc's from the engram add. Fork option `ttnn.bringup.mhc_post(collapse_pre=, hc_scale=, hc_base=,
  hc_norm_width=, hc_norm_eps=, hc_eps=, hc_sinkhorn_iters=, hc_output=, y_output=)` returns (X', stats, hc, y).
  - y = sum_k pre_k X'_k. Per column, one DEST window after the mix: the pre-block tile plus the bf16 X' tiles already
    in cb_xs for the matmul, mhc_pre_xing's ymix, then packed to bf16.
  - hc = DeepSeek's coefficients + Sinkhorn of the summed stats row, on the row's root core.
  - The SFPU bodies are copied line for line from mhc_pre_xing_compute.cpp into kernels/mhc_post_hc_sfpu.hpp; keep the
    two in step.
  - The pre is the residual's own mixes (DS41-3), columns 0..3. Pinned site buffers are reused
    (fork.ffn_hc.y; fork.attn_hc.hc / .y; DS41-174), so no new allocation.
  - DevH.stats carries (hc, y) instead of a row. attn_hc / ffn_hc take a tuple as finished; layer 0 and any attn_hc
    without an emitted row keep mhc_pre_xing.
  - Conditions: MHC_FUSED, OWN_ROWS, BF16_ACTIVATIONS, GLUE_FOLD on and HC_SINKHORN=sfpu_fp32
    (tt/model.py:post_collapse). Otherwise the old path runs.
- **Removed a chunk** (40 layers, layer 0's attn_hc and both engram layers' attn_hc counted): 79 mhc_pre_xing
  launches. 40 are ffn_hc. 39 are attn_hc of layers 1..39: 37 from ffn_residual, 2 from engram_add (layers 1 and 14).
  Each launch re-read [160, 4 x 5120] bf16 a chip (6.5 MB) and wrote hc + y.
  - Added per mhc_post call: one DEST window a column (5 tile copies from L1, ymix), the y write (1.6 MB), one Sinkhorn
    tile a row on the roots, 3 + 33 common runtime args.
  - X.3 numbers: attn_hc 2.66 + ffn_hc 2.46 ms device and 3.13 + 4.54 ms host a chunk, almost all of it these calls.
    The orchestrator's A/B (51200 / 512000) gives the measured before / after.
- **Bit identity**: tests/unit/test_mhc_post_collapse.py (8x4, 8 passed). X', stats, hc and y are byte-equal to
  mhc_post(stats_weight=) + mhc_pre_xing(stats, X', deepseek=True, collapse_pre=). Covered:
  - shapes [160, 5120], [640, 1280], [64, 640];
  - the 11 x 10 grid;
  - the pre as the packed mixes and as its own tensor;
  - a cache hit with other scalars into given buffers.
- **Gotcha (DS41-210)**: on a cache hit, override_runtime_arguments sees every core's per-core args padded to the
  kernel's longest list. The first version appended the 3 addresses after the writer's variable-length lists, so only
  the 5 root cores (13 args, the max) were patched. 760 of 800 y tiles went stale and later calls corrupted outputs.
  Fixed with common runtime args.
- **Quick, switch on** (`measure --task P.34 --part quick`, code 57b8b781c453d8a7, DS41_POST_COLLAPSE=on,
  ENGRAM_TABLE=resident). rc 0, all 102 layer limits held. Worst layer L39: 0.62766 vs limit 1.5332. L10: 0.11294 vs
  0.17737. host_transfers_per_layer 0. chunk_wall 183.2 ms, but spread 278.7 ms (a noisy host); the first run of the
  same kernels gave 172.9 ms, spread 8.2. The layer errors are identical in both runs.
- **Not run** (brief): the off side, the profiles, the gate. The orchestrator measures off / on at 51200 and 512000.
- **Re-run**: `scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/mhc_post_ttnn/tests/unit/test_mhc_post_collapse.py`;
  `python -m models.demos.common.bringup.orchestrator measure --spec <spec> --task P.34`.

## The combine's snapped fp32 sum is order-free only below 2^24 q = 32; routed sums reach 21376

DS41-235 (FIX.23, 2026-11-12; ported onto the pre-replication base by P.50, where COMBINE_SPLIT ships off until
the owner decides from P.50's A/B).

- Repro (one process, warm vs warm is enough): the code prompt (tt-metal C++ sources, testing/b2b.py ids), 2-3 chunks
  of 5120 from position 0, 3 sequences on one loaded model, each on a fresh state:
  `DET_WL=code DET_CHUNKS=3 DET_RUNS=3 scripts/run_safe_pytest.sh --run-all --no-precompile
  models/demos/deepseek_v41_flash_d_p/tests/bringup/probe_determinism.py` (on exp/fix23_snapshot 4e90f611e86 only, not ported: a bisect probe; DET_READ=layers: per layer; DET_MOE=combine
  / all: inside each MoE layer). Every chunk's final hidden differed between runs (max 0.089 / 0.154 / 0.258).
- First divergence (a readback after every layer and op): layer 37 of chunk 0 (layer 38 of chunk 1);
  routed_experts_combine's output differs while its inputs that carry values (layer 36's output, the fused head's
  route / regions / plan / shared expert output) are bit-identical. (head.xbuf / table / token_index and the dispatch's
  output differ in EVERY layer, also where the combine is identical: arrival-order staging and unused slots, not values.)
  Its 45 differing elements (run 1) / 29 (run 2) all have |y| >= 37.25 (median 197 / 103); no element under 32 ever
  differs. Combine output peaks (code, chunk 0): L15 5312, L20 16640, L28 21376, L39 17152; chunk 1: L36-39 768-17152.
- Cause: accumulator_compute.cpp multiplies each landed row by its gate weight, snaps it to a multiple of q = 2^-19
  ((v + C) - C) and packs it onto the token's fp32 sum with packer L1 accumulation, in landing order. q-multiples add
  exactly only while the sum stays below 2^24 q = 32 (and terms below 2^22 q = 8: (v + 24) has a coarser or finer ulp
  past that); beyond, each add rounds and the result depends on the order the packets landed (the fabric's timing).
  The setting's note "routed sums peak at 5.4" (2026-10-27) was measured on another input.
- Fix: MoEBlockConfig.combine_snap_split (setting COMBINE_SPLIT; FIX.23's default on, P.50's base off): hi = v snapped to 2^18 q = 0.5, lo = v -
  hi (exact) snapped to q; two fp32 sums (CB_ACC doubled), out = lo + hi once. Exact to |hi sums| < 2^23 and terms
  below 2^21. Values: the exact sum of the snapped terms, bit-identical to the single sum wherever that was exact.
- Before / after (tests/test_determinism.py: novel and code interleaved, 4 chunks, 3 sequences each, the first after
  load cold): off - code max |diff| per chunk run 1 [0.0156, 0.0156, 0.051, 0.079], run 2 [0.090, 0.154, 0.097, 0.219]
  (1443-1956 rows of 5120), novel 0 this time (DS41-233's cold chunk 0 difference is the same race, it does not show
  every run); on - 0 everywhere.

## A light chip finishes its own experts at the DRAM floor, 0.467 ms, before any whole-expert copy lands
- Measured (8x4, chunk 10 = tokens [51200, 56320), MOE_REPLICAS=off, 3 profiled runs a workload, real-time profiler):
  routed_experts_combine (experts + combine, one program a chip) summed over 40 layers, slowest chip / shortest chip:
  novel 56.3 / ~50 ms, code 47.0 / ~39 ms, random ids (the p51200 position profile) 63.2 / ~58 ms. The shortest
  chip is the one that started last at the start barrier (DS41-248), so it is the work without host-dispatch skew.
- Per-chip time model, fitted on those 360 layer calls (tests/moe_exact_model.py `fit`):
  `reach_k = max(A x active experts, B x row tiles)`, layer = C0 + max_k reach_k; A = 19.9 MB / 512 GB/s = 38.9 us
  (rmse 0.068 ms at A, 0.073 at 1.5 A, 0.100 at 2 A: the data prefer the DRAM rate), B = 7.55 us a 32-row tile,
  C0 = 0.235 ms, R^2 0.966. With 12 experts a chip the floor is 0.467 ms: the median chip sits on it (busiest / mean
  ~4.4 in tokens, but only ~2.3 in time).
- Exact split (greedy planner; one replica a receiving chip, the 13th slot; a hot expert may get several), max mode
  (copies free) vs deadline mode (each copy = 19.9 MB at P.47's 0.71 ms a link pair + 18 us an extra hop, store and
  forward along the axis-0-then-axis-1 path, one copy at a time a link, starting one weight read after the home
  starts the expert): novel 21.9 -> 10.1 ms a chunk, code 10.6 -> 2.9, random ids 30.9 -> 16.5. 0 of 335 (novel)
  copies land before the receiver reaches the replica: they land at >= 0.75 ms, the receivers reach it at ~0.47.
  Halving the copy time (two paths a copy) keeps 20.9 / 10.1 / 27.7 ms. Three gate runs agreed within ~0.3 ms.
- Not modelled: the dispatch's own imbalance (it also moves rows to replicas), the combine rows on the same links
  (~10 % of a link), and the home's L1 window: in ~94 % of the planned copies the copy outlasts the home's compute of
  that expert (+ one weight read), so the source would need to hold the blocks or re-read them from DRAM
  (`src_window_short`, results/P.55a_exact_bound.json).

## A copy streamed in the layer's own combine moves ~7 GB/s a link while the combine runs, ~20 after
tests/test_moe_exact_copy.py, layer 20, 8x4, a random 5120-token chunk (the combine alone 0.84-0.89 ms, every chip's
own work 0.62-0.65 ms: its ports' main loops), MOE_EXACT_BLOCK 32 pieces (340 KiB), cap 8, AICLK 1350 MHz. Latency =
the receiver's waiter port: its start barrier to the last flag of the copy; added = the combine program's device time
(real-time profiler, slowest chip, median of 6, stock calls alternating in the window) minus the stock one.

| case | copies | latency ms | added ms | first block ms |
|---|---|---|---|---|
| no job | 0 | - | +0.009 | - |
| pair, 1 link | 1 | 1.397 | +0.630 | 0.065 |
| pair, both links | 1 | 0.890 | +0.106 | 0.062 |
| pair, 4 links (half via chip 4's spare slot, relayed) | 1 (+1 relay) | 0.761 | +0.008 | 0.064 |
| neighbour, 32 copies at once | 32 | 0.954 | +0.254 | 0.038 |
| real split, novel L20 (P.55a counts; 4 hot experts, relay trees depth 5) | 23 | 1.254 | +0.372 | 0.054 |
| real split, code L20 (4 hot experts, depth 3) | 10 | 1.235 | +0.371 | 0.057 |

Block arrivals (pair, both links, path 0, every 4th block, us): 65, 255, 402, 533, 623, 693, 762, 831: ~47 us a block
while the combine's main loop runs (rows first, the experts reading their weights from the same DRAM), ~17 us after
(~20 GB/s a link). One link alone 1.40 ms = 15.5 GB/s average, the rate expert_slot_copy reaches a link (P.47's
0.71 ms used both links). A relay level adds ~0.05-0.1 ms (tree depth 5: the last receiver 1.25 ms vs the first 0.79).
Two copies on one port go one after the other (the code split at first: 1.80 ms with 2 jobs a port; the tree now
prefers a parent whose port toward the child is free: 1.24 ms).
P.55a's model (moe_exact_model.py, chunk 10 at 51,200, every layer) with a copy edge taking the measured pair time:
both links 0.89 ms: novel 5.87 ms a chunk (0 of 174 copies on time), code 1.25 (0 of 41); 4 links 0.76 ms: novel 8.74
(0 of 269), code 2.44 (0 of 74). P.55a's own 0.71 / 2-path assumption gave 20.9 / 10.1.
The final gate run (its recorded metrics, same code): pair both links 0.876 ms / +0.137, 4 links 0.772 / -0.004,
neighbour 32 0.961 / +0.195, novel split 1.229 / +0.419, code split 1.229 / +0.297; the model: both links novel 6.16
(0 of 183), code 1.36 (0 of 46); 4 links novel 8.55 (0 of 259), code 2.30 (0 of 63).

## The exact split pays only with copies already in place; streamed in the layer's own combine it loses

Chunk 10 at 51,200 (the device's own prefill of chunks 0-9, re-runs on the steady path), 8x4, 40 MoE layers, plans from
the same chunk's counts (tests/moe_exact_model.Plan; A 38.9 us, B 7.55 us a row tile from P.55a's fit), median of 3
profiled runs (real-time profiler; device = each program's slowest chip, summed):

| prompt | side | device ms | experts+combine | dispatch | head | replicas |
|---|---|---|---|---|---|---|
| novel | off | 185.9 | 56.5 | 20.0 | 20.0 | 0 |
| novel | fill (copies before the chunk, max-mode plan) | 161.5 | 39.2 | 11.2 | 17.8 | 512 |
| novel | exact (streamed, deadline plan at 0.89 ms a copy) | 195.0 | 68.7 | 17.8 | 19.0 | 290 |
| code | off | 170.5 | 46.5 | 14.4 | 17.1 | 0 |
| code | fill | 158.3 | 38.9 | 11.2 | 16.6 | 295 |
| code | exact | 175.6 | 51.7 | 13.9 | 17.4 | 67 |

Every side bit-identical to off on every MoE layer's output (the exact side with every receiver's spare slot spoiled
first: the receivers' wait works), the repeats bit-identical. The split also halves dispatch (its destinations even
out). Streamed: the 23 (novel) / 8 (code) layers with copies take 2.10 / 1.98 ms each vs off 1.62 / 1.51 and fill
0.93 / 0.96 (+1.0-1.2 ms a layer over the same split with the copies in place); the layers without copies +0.07 /
+0.02 ms (the exact program with empty jobs, after a copy layer's skew). P.55a's model with the whole-copy wait
predicts 1.21 ms for those layers at 0.89 ms a copy edge; recalibrated it plans: 1.25 ms a copy -> novel 0.77 ms a
chunk (44 copies), code 0.06; 1.6 ms -> 0.04 / 0. A copy tree lands in 1.25 ms (P.55b, one tree) to ~1.9 ms (19
receivers of one expert, depth ~5), when the off layer's busiest chip is done at 1.0-1.9 ms: a copy within the call
cannot arrive before the receiver would have to start the replica.

## P.56's own host cost makes the chunk host-bound; a one-layer trace replays bit-identical
DS41-256: where the host time of a chunk goes (P.57, 8x4, 2026-11-14).
Measured by tests/test_host_bound.py (one process, shipped settings + MOE_FILL=piggyback, ENGRAM_TABLE=resident; side
off = MOE_REPLICAS=prev switched off in the loaded model). host = first host call to the last op enqueued; wall = to
the end of the closing sync; device = program real-time profiler, slowest chip a program, summed.

| chunk | host | wall | device | regime |
|---|---|---|---|---|
| novel c1-10 mean, off | 160.5 | 180.9 | ~175-177 | device-bound (wall = device + ~4-8) |
| novel c1-10 mean, prev | 185.0 | 186.6 | ~156-160 | host-bound (wall = host + 1.6 drain) |
| p51200, off | 163.5 | 202.5 | 194.6 | device-bound (wall = device + 7.9) |
| p51200, prev | 181.7 | 184.1 | 162.8 | host-bound (wall = host + 2.4) |

Host parts, novel chunk 10, prev (instrumented 195.7 ms vs clean 190.3; the 5.4 ms wrapper overhead sits mostly in b):
(a) Python in tt/ 34.4 (attention 11.0, ffn_residual 7.0, experts 4.1; ~6 ms of it is ttnn.deallocate via ccl.dealloc,
1211 calls); (b) ttnn op launches 139.9 for 1732 calls (C++ 135.3, Python-registered forks 4.6); (c) run_prepared 8.9
(planner 3.9, window_send 3.0); (d) transfers / waits 10.0 (the layer-1 wait for the async engram job 7.6, the engram
rows' copy 2.4); (e) per chunk 2.5 (begin_chunk 0.5, embed ids 1.4, P.56 guard 0.6); (f) other 0.
Top launches (us a call x calls): routed_experts_combine 731 x 40 (402 off), fused_head 301 x 40, slice 54 x 211,
high_bw_all_gather 106 x 92, mhc_post 100 x 82, linear 55 x 144, rotary_embedding_indexed 56 x 136, slice_write 43 x
164, update_padded_kv_cache 87 x 48, topk_large_indices 437 x 9, allocate_tensor_on_device 61 x 58, sparse_dispatch_2d
89 x 40. Host by section (Python + its launches): experts 64.7 (50.5 off), attention 43.0, ffn_residual 27.3,
compressor 13.4, engram_rows 10.0 (the wait), indexer 9.0, kv 8.5, q_a 7.3, attn_residual 5.9.
cProfile (testing/profile.py host_profile, p51200 prev): decorators.py FastOperation.__call__ self 130 ms of ~192, the
engram worker lock 11.7 ms.

Trace prototype (side off, layer 21 at novel chunk 10, input cloned each run, 47 programs): capture after 3 warm runs
(no "Cannot load new binaries": DS41-100's address-keyed caches are gone), 20 replays bit-identical to the untraced
output (which repeats bit for bit), no hang. Host a layer 6.45 -> 0.020 ms; device a layer 3.93 (rt profiler) -> 3.54
(10 replays back to back / 10).

Trace blockers for a chunk-to-chunk trace (code reading at b8a57740c25):
1. per-start host arguments baked in the commands: RopeTable.rows (ttnn.slice at start), update_padded_kv_cache
   kv_actual_global / valid_global, the window server row, DeviceScalars.at(start), compressor slice_write row offsets,
   ckv_bfp4's row0 value slot, ring_indexer_score_dsa chunk_start_idx and kv_len (the program grows with the prefix),
   the window start class. Lift: one trace per start (memory not measured) or start as a device tensor (K-48 style).
2. state written per chunk into tensors allocated in the chunk and kept after it (top-k / candidate stores keyed by
   start, DS41-2). Lift: preallocated per-position stores.
3. host inputs: token ids (embed: a new from_torch tensor a chunk) -> a fixed input buffer; engram rows already go into
   reused buffers (tt/engram.py copy_host_to_device_tensor) but layer 1 waits for the worker inside the chunk -> finish
   (or prefetch one chunk ahead) before the replay.
4. P.56: the host guard (steady chunks are 40/40 ready; chunk 0 runs off -> keep chunk 0 eager) and the combine's ring
   slot `ring[o % 44]` chosen on the host each call (40 calls a chunk: the pattern repeats every 11 chunks). Lift: the
   slot from the device ordinal G[0], or W a multiple of the calls a chunk.
5. not blockers: host reads in the chunk (0; MOE_COUNTS off), CCL semaphores (host-cycled handles and the combine's
   device call counters replay fine in a fixed call sequence: the prototype), allocations inside the capture.

## TP experts lose to EP on the flat op: a 32-row tile barely gets cheaper as the expert narrows
(DS41-257, P.58, Galaxy 8x4, tests/test_moe_tp4_check.py; real-time profiler, slowest chip, median.)

- Calibration of `ttnn.bringup.moe_block_main.flat_routed_expert` alone, bfp4 weights_nk, 11 x 10 cap:

  | width I | slots a chip | A (one more active slot, 32 rows) | B (one more 32-row tile) | B per full-width equivalent |
  |---|---|---|---|---|
  | 2304 (EP) | 12 | 51.1 us | 7.66 us | 7.66 us |
  | 512 (TP4 stand-in for 576) | 48 | 14.6 us | 6.25 us | 28.1 us (3.7x) |
  | 256 (TP8 stand-in for 288) | 48 | 13.7 us | 4.85 us | 43.6 us (5.7x) |

  A per byte holds roughly (1.28x worse at 512); B does not scale with the width: per row the op moves an H = 5120
  wide x row in and an H wide y row out whatever I is, and under TP every chip of a group processes every row of the
  group's experts (4x / 8x EP's rows a chip). The "balanced compute" estimate (B / g) is therefore wrong by ~4x.
- At P.55a's chunk-10 counts (40 layers), flat op a chunk, novel / code: EP 52.5 / 42.5 ms, TP4 86.0 / 77.5 (x 1.125
  from 512 to 576), TP8 125.9 / 119.0 (two 48-slice calls a layer: an upper bound).
- Communication (head-driven, steered to layer 20's counts; extrapolated to 6 experts a token): TP4 dispatch +0.07 /
  +0.10 ms a layer, combine +0.10 / +0.15 (novel / code); TP8 dispatch ~+0.5, combine ~+1.7 (uniform K 8 -> 16 the
  combine cost went 0.09 -> 0.37 ms: superlinear, the linear extrapolation to K 48 is rough).
- Per chunk at 51,200, device ms (EP off and EP + P.56 measured by P.56; TP = EP off + experts difference + 40 x
  dispatch added): novel 175.2 / 156.1 / TP4 215.3 / TP8 334.8; code 164.0 / 153.1 / 209.1 / 329.8.
- Precision: summing 4 / 8 partial rows in the combine's fp32 accumulators: rel error vs fp32 0.0561 / 0.0564 vs EP's
  0.0560 (uniform routing, 512 tokens): no cost.
- Op limits met on the way (each is work for a real TP build): flat op even gate/up split (flat_routed_expert_plan.cpp
  "no even gate/up split": 2 I / 32 tiles into a multiple of 16 cores; 576 and 288 have none), fused head top-K even
  and <= 16 (routing rows in 64 B), routing planner N x slots <= 1024 (head_plan.cpp), flat op <= 64 slots a chip.

## P.56's combine host cost was its per-call fill arg rewrites, not the ring slot choice
(DS41-259, P.60, Galaxy 8x4, 2026-11-14.)

- Probe (tests/test_moe_prev_host_probe.py, 4 layers, novel chunks 0..3, median host us a combine call; plus
  temporary C++ timers per pass of 16 calls): off ~440-520; prev legacy ~810-930, of it set_fill ~350-470 (fill
  ports' args ~150-210, the flat expert readers' wait args ~190-260); legacy with ring[0] on every call 978 (no help);
  legacy with set_combine_fill_exact once a pass 794 (no help: the C++ hit still rewrites every arg, the token is new
  every call). With ring + wait False (MOE_PREV_HOST=device): set_fill ~5 us (the signature compare), launch ~470-525.
- Why so slow: set_fill runs for each of the 32 per-chip programs and does a GetRuntimeArgs (a kernel lookup in the
  program's maps + a shared_ptr copy) per fill port and per reader core; ~5-6 us a chip.
- Full model (gate unit, chunks 1..10 mean): combine us a call off 340 / legacy 687 / device 362; host ms a chunk
  novel 164.7 / 188.1 / 172.4, code 162.3 / 182.7 / 170.1.
- The planner's run_prepared: 55.7 us a call in a tight loop (same io), 58.1 alternating layers' io, ~95 us inside
  the chunk: the patches are not the cost; the rest is the 32-program dispatch with cold host caches (DS41-131).

## The fused head's and the combine's per-chip programs differ in structure, not only in arg values

(DS41-269; P.63b.) What one program for the whole mesh needs.

Diff of the 32 per-chip programs (DS41_MESH_DIFF=<prefix>, diagnostic; common/mesh_uniform.cpp
mesh_uniformity_report / mesh_views_dump; 8x4, novel chunk 1, MOE_REPLICAS=prev):

| op | kernels | chip-dependent args a chip | structural differences |
|---|---|---|---|
| fused head | 11 (3 compute) | 4,280 per-core + 5 common (DM 4,046, compute 3,290 incl. different lengths) | 4 defines carry the chip index (RA_ME_ xg_sender / xg_watcher / ra_y_send / routers, RAP_ME routers, RT_SMAP_ME, RW_ME router_compute); arg-list lengths differ by chip (xg_sender 296 / 235, se_ncrisc 39 / 63, se_compute 44..60); patched address args sit at chip-dependent indices (xg_sender: +2 / +4 shifts); router_compute arg 10 is an address with a per-column offset. Core ranges, CBs, semaphores, compile args: the same. |
| combine (flat expert + combine) | 17 (4 compute) | 3,047 per-core (down_weight_reader 31 a core: DRAM bank coordinates; port writer up to 317; accumulator reader up to 96; accumulator_compute 1) | port cores and accumulator cells differ by chip (port_placement.cpp place_nearest: next to each chip's Ethernet cores) - so CB and semaphore core ranges differ too. Union: 14 port cores, 21 accumulator cells. |

Mechanism built (MESH_PROGRAMS2=on, fork option mesh_program; head only, the combine not started):
* host (assemble_mesh_program): chip 0's descriptor, every list padded to the kernel's longest (free: the
  dispatcher already sends each core of a kernel at the kernel's max length); defines that differ become an arg
  (`(get_arg_val<uint32_t>(n))`); per chip and slot an entry: delta (added: constants, or the same buffer at another
  offset), absolute (a constant where chip 0 has an address), anchor + offset (an address read at another index
  than chip 0: one extra arg per core and buffer the host patches per call). Table: DRAM, replicated, a 960 B page
  per (core unit, fabric chip id); landing: an L1_SMALL page a core. Head: 220 units, 4,599 entries (chip 1), 138
  anchors, 6.76 MB table, 26,113 args a launch (per-chip legacy ~20.7K). A host check rebuilds every chip's args
  from the mesh args and its rows (passed).
* device (common/device/kernels/mesh_overlay.hpp): DM kernels read their row by my_device_id and apply it before
  anything else; a compute kernel's entries are applied by the first DM kernel on its core into the three TRISC arg
  copies (launch message rta_offset[2..4]) and released by a flag arg the TRISCs spin on.

## The first mesh fused-head launch (MESH_PROGRAMS2=on) hung the 8x4

(DS41-271; P.63b.) 1-layer probe, sides on (P.63) then mesh; on ran clean and bit-identical; the mesh head's
first launch timed out (metal_context.cpp:806), then PCIe MMIO timeouts; triage could not attach; the safe runner
reset and recovered the box ("All Detected Links are healthy"). No further device run (owner safety rule).
Candidates, in the order to check (on a 2x2 first, or in slow dispatch with the watcher):
1. the compute release: trisc_args() address (watcher off: no count word), the "fence" ordering on BH, TRISCs
   spinning with invalidate_l1_cache(); a TRISC that never sees its flag hangs the program;
2. a staged compute kernel whose stager DM kernel reads its args before... (no: the prologue is first) - check
   every head kernel with MESH_OVL_FLAG has a DM kernel with a unit on every one of its cores;
3. the NOC read of the row: InterleavedAddrGen page id (unit x chip ids + my_device_id) and an L1_SMALL landing
   at the same address on every core (MeshBuffer sharded over the unit cores);
4. a wrongly applied delta on a fabric connection arg (eth channel, EDM addresses) - the host check covers values,
   not the device's decoding (desc bits: slot 14 | common 14 | type 15-16 | anchor 17-30).

## The intermittent DECODER_SKIP=decode hang: one chip drops off PCIe (a different one each time), its CCL partners wait

(DS41-283; FIX.36, 2026-11-18, bh-glx-110-c10u02, KMD 2.11.0, firmware 19.16.0, FABRIC_2D_TORUS_XY.)

History on the shipped tree (STATIC_TABLES=on, DECODER_SKIP=decode): test_skip_positions hung 2 of 4 runs (11-17 14:26
at p256000, 23:18 at p1024000 about 3 s into the steady warm-up after p768000's profiled run; 11-15 FIX.34 and 11-18
00:48 passed all 6 points). Each hang was followed by the node going down within 5-10 min.

Reproduction (one model load, 20 min budget): `DS41_DECODER_SKIP=decode HANGLOOP_MINUTES=20 scripts/run_safe_pytest.sh
--run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/test_skip_hang_loop.py -s`: the test's per-point
work (fresh state p + chunk; steady and resumed: the real previous chunk at p - chunk, then the chunk at p, 23 runs a
kind, the last one under the rt profiler), points 256000 and 1024000 alternating, ~32 s a round (184 chunks).
Run fix36_loop3: rounds 0-12 clean (2,392 chunks, layer 19 output of p256000 steady bit-identical every round), hung
at round 13, p256000, `resumed run0 prev`: the first chunk after the steady kind's rt-profiled run (finish_rt: sync,
1 s idle, unregister). Log kept: /scratch/dnijemcevic/fix36/loop3.log (+ inspector_loop3/, the triage log).

What the hang looks like:
- 04:31:22 "Timeout detected" (5 s dispatch timeout), then TT_THROW "TIMEOUT: device timeout, potential hang detected,
  the device is unrecoverable" (system_memory_manager.cpp:806).
- tt-triage did not run: UMD TopologyDiscovery -> wait_eth_core_training -> "MMIO per-op timeout: 4B load took 55527
  us (budget=2 ms)". So the op in flight per chip is unknown (no inspector / dispatcher read possible).
- Teardown: every chip (0-31) "Timeout waiting for physical cores to finish: 15-3, 15-2": these are the prefetch /
  dispatch cores (DispatchKernelInitializer::wait_for_dispatch_cores), stuck because the work they dispatched never
  completed; not a program of the model. Tensix reads still worked on every chip.
- Chip 15 only: "Skipping RISC reset on device 15 during teardown: MMIO per-op timeout ... 55569 us" and "Skipping L1
  ECC report on device 15": a core of chip 15 (the RISC reset path; eth cores are the candidates, as the triage's
  first failure was an eth read) no longer answers MMIO. Chip 15 = phys 15 / M0D14, one of the 4 chips the control
  plane warns about at every open ("in direction 2 has 4 eth channels, but only 2 routing planes are available":
  phys 7 / 15 / 31 / 23). One sample: whether the same chip fails every time is not known (the 23:18 log was cut
  at device 17 by the overseer's stop; 14:26 left no log).
- The safe runner then started recover-hosts (sys-triage + tt-smi -glx_reset) by itself.

What it is not (evidence):
- Not a semaphore equality race in the model's fabric sends: window_send / snapshot_send / tail_send wait with
  noc_semaphore_wait_min and take arrivals back with an atomic -1/-count.
- Not ENGRAM_ASYNC's worker writing CQ 0 unlocked: FDMeshCommandQueue takes lock_api_function_() in
  enqueue_mesh_workload and the write paths; pinned writes are blocking.
- Not the rt profiler as such: the RealtimeProfilerManager runs its push kernel and host receiver from mesh open in
  every run (eligibility: MMIO chip + IOMMU), registered callback or not; only the consumer thread is per callback.

Open leads, cheapest first (each costs a box reset if it hangs; the owner decides):
1. Same chip? Re-run the loop (it hangs in ~8 min) and read which chip fails MMIO at teardown. Always chip 15 points
   at that chip / its links (a hardware or firmware question for the cluster admins), not at the model.
2. Idle-then-burst: both located hangs came right after the device sat idle (1 s settle; a point change). Rule 18's
   di/dt trip shows as dead MMIO too. tests/perf/test_core_audit.py (FIX.15) covers a ladder chunk only: audit the
   skip-positions flow (DECODER_SKIP=decode, new_state at 1M, resumed write_start) for programs over 110 cores.
3. Without the profiler: HANGLOOP_PROFILE=0 (and a sleep 1 s instead) separates the profiler's unregister from idle.
4. With the watcher (`--dev`) if 1-3 give nothing: slower, but triage then has the last waypoints.

### FIX.36 attempt 2 (2026-11-18): the hang is a chip dropping off PCIe; the 120-core programs capped

- HANGLOOP_PROFILE=0 (no rt profiler, OV-43's first lead) hung too: round 7, `p256000 resumed run5 cur` (mid-stream,
  no idle before it); chip 3's MMIO dead ("Setting power state failed on device 3: Input/output error",
  PcieHangError). So the profiled run / its 1 s idle is not the trigger, and the dead chip is not always 15.
- HANGLOOP_WATCH (an rt-profiler callback over the whole loop, per kernel set the most cores, per chip the last
  programs done) found two kernel sets on 120 cores, every chunk, on every chip: eltwise_binary_no_bcast (binary_ng
  FPU, 20 a chunk a chip: the MoE output add in moe_block_main_adapter.py, one a full layer under DECODER_SKIP=decode)
  and eltwise_binary_sfpu_scalar (5 a chunk: one of the scalar masks lt / eq / ge / maximum; it went away with them).
  Everything else is at or under 110. MOE_BLOCK_SRC=main (owner 2026-11-06) came after FIX.15's audit (2026-10-31).
- The same loop (unfixed) hung in round 16, chip 7 MMIO dead. Last program per chip (watch): chips 4-7 and 28-31 at rid
  37291 (rope_chain), i.e. never finished 37292 (ring_attention all_gather + index compute, 104 cores); chips 12-15 /
  20-23 at 37294, the other 16 at 37304. Chip 7's group stopped, its partners waited in the CCL: a chip lost at the
  device level, not a model program waiting on a semaphore.
- Fix: every stock binary call left uncapped in tt/ gets sub_core_grids (sub_cores(x)) or cores.maximum (8 sites in
  attention, compressor, indexer, engram_add, moe_block_main_adapter, sparse_attention's STATIC_TABLES=off remap).
  The same ops on fewer cores: same values (elementwise).
- After: fix36_fix1 (HANGLOOP_PROFILE=0, watch on, 30 min): 41 rounds, 7,544 chunks clean, no program over 110, the
  p256000 steady layer-19 output bit-identical to round 0 in all 40 comparisons. Before: hangs at rounds 13, 7 and 16
  (mean ~12): 41 clean rounds at that rate has a chance of about 3%. The causal link (120-core di/dt trip -> chip
  drops) is the likely one (rule 18's own failure mode, a random chip each time), not proven by a single experiment.

## STATIC_TABLES=off ran 120-core programs at odd chunk counts (rule 18; FIX.36 audited STATIC_TABLES=on only)

(DS41-289; P.73 attempt 2, 2026-11-18, 8x4, DECODER_SKIP=decode, tests/test_skip_hang_loop.py HANGLOOP_GATHER_ROWS=alt.)

FIX.36 capped the 120-core programs it saw with STATIC_TABLES=on. STATIC_TABLES=off had never been audited. At odd
chunk counts (every A/B and skip-positions point: nb = 11, 51, 101, 151, 201) the history slab holds nb x 80 compressed
rows a chip, which is not a multiple of 32. Found with the hang-loop watch, then charged to calls with
tests/perf/test_core_audit.py (BRINGUP_POS=1024000):
- both switch values: `ServedRows.slab.slice_rows`: ttnn.slice of the ND-sharded ROW_MAJOR slab with a non-tile H
  takes the composite path (slice.cpp needs_rm_composite_input: to_memory_config to L1 interleaved, slice, back to
  DRAM), whose copies take no sub_core_grids: 120 cores, 6 calls a chunk (0.64 ms). `_topk_rows`' two ttnn.bitcast
  calls had no sub_core_grids: 120 cores, 8 calls each.
- HISTORY_GATHER_OVERLAP=off only: `to_layout(sl, TILE, sub_core_grids=...)` pads through tilize_with_zero_padding
  without the grid (119 cores); the TP gather of the padded rows takes ttnn.all_gather's composite (all_broadcast +
  concat; all_gather.cpp says it ignores sub_core_grid): the concat's ROW_MAJOR program on 120 cores, ~11 a chunk.

Fix (bit-identical; rows_on vs rows_off at layer 19 identical in every round):
- tt/compressor.TtServedRows: an own slab (the ladder / tests; never the engine's) gets rows rounded up to
  lcm(32, the chip's rows a chunk) (update_padded_kv_cache needs rows % the per-chip chunk rows == 0; plain 32
  rounding TT_FATALs there), so ttnn.slice takes its native, capped ND-sharded path. `_natural` reads the first crows.
  An engine slab of non-tile rows would still take the composite: none at the deployment size (204 chunks, even).
- tt/sparse_attention._topk_rows: both bitcasts get sub_core_grids.
- tt/cores.to_tile: a padded tilize as tilize_with_val_padding(0.0, sub_core_grids) (the same zero padding).
- tt/ccl.all_gather: where stock would take the composite under the cap, `_composite_capped` runs the same steps
  (all_broadcast at its defaults, untilize each piece, ROW_MAJOR concat, zero-pad tilize) on the capped grid.

After: loop3 (one load, the switch flipped every round): 16 rounds, 8 a side, 2,944 chunks with no program over 110
cores on any chip and determinism 0 on both sides; then a hang in round 17 (rows_on), p256000 resumed run 10 cur:
the DS41-283 signature again. Chips 0-3 and 24-27 never finished rid 13887 (ring_attention all_gather + indexer
score, 104 cores; their last program the rope_chain before it), the other 24 waited further on; chip 25's MMIO dead at
teardown ("4B load took 55505 us"); tt-triage failed; the safe runner reset the box (32 boards back). So the 120-core
programs were not the only trigger: a chip drops with every program at or under 110. FIX.36's 41 clean rounds
(STATIC_TABLES=on) were one sample; the rate at ~2,900 chunks is the rate it had before the caps (~2,400). Log:
/scratch/dnijemcevic/phase_1117/p73a2/loop3.log (audit.log, loop2.log, coreaudit_on.log next to it).

## The chunk's exposed host time was two engram stalls, not launch cost

Method (tests/perf/test_host_idle_map.py, no device profiler for the walls): host marks at begin_chunk, embed, every
layer, the engram waits; one extra run under the program real-time profiler with no op patching: each idle gap of a
chip's timeline (a program starting after every earlier one ended) charged to the host segment that launched it. The
rt run itself is slower (its receiver thread takes the GIL; worse at a 0.5 ms switch interval), so its segment sizes
are indicative; the clean marks are the measure.

| | off (P.77 tree) | HOST_HIDE=on |
|---|---|---|
| clean wall, median of 7 | 106.5 ms | 93.8 ms |
| host done launching | 82.0 ms | 69.3 ms |
| layer 1 rows ready (worker: hash / gather / upload) | ~19.7 (4.6 / 12.2 / 2.5) | ~6 (0.8 / 3.5 / 1.1) |
| main thread blocked for the rows | 14.3 ms | ~0 |
| layer 2 / 4 / 8 host | 10.1 / 6.9 / 6.5 | 10.2 / 2.0 / 7.2 |
| device busy (slowest chip) | 80.7 | 80.7 |

What is left (~13 ms): the host needs ~22 ms to launch embed, layers 0-2 and the P.75 / P.76 prefetches (~4 ms in
take_engram_rows) for ~14 ms of device work. Layer 2 (the first KV / index source) alone takes 107 launches, 10.2 ms
(ttnn.bringup.topk_large_indices 1.9 ms host in one call there, < 0.11 ms in layers 8 / 14). After layer 3 the host
(~2 ms a layer) leads the device (~4 ms a layer) and the device never idles again. Next levers: host cost of the
source layers' indexer / compressor launches, or a trace of the layers after the rows (blocked by the per-start host
values P.21 lists).

GIL: the worker thread runs a little Python between its GIL-free steps (numba with nogil, copy_host_to_device_tensor);
while the main thread dispatches, each re-take waited up to the 5 ms switch interval (the torch hash read 3-4.6 ms
in the worker for ~1 ms of work). numba's spinning OpenMP pool also slows torch ops next to it (token_map[ids] 0.2 ->
1.7 ms): keep the worker's work inside the compiled kernels.
