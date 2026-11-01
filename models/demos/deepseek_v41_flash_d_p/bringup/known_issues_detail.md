# Known issues detail: deepseek_v41_flash_d_p

## (RESOLVED by the overseer) The canonical prompt was shorter than the scaled rung

Resolved 2026-10-15 by the supervisor: `scaled.ladder` is now 55296 tokens in 2048-token chunks, within the existing
prompt (56320 tokens); no prompt rebuild.

The alternatives that were open before the resolution (both outside SG.1's paths): set spec `text.length: 65536` and
rebuild the prompt (`prompt.build(force=True)`; ids stay prefix-identical, new sha), or make `prompt.default_length`
count `scaled.ladder` as well as the real ladder rungs.

(Moved here from the bullet in C.swa.kv test attempt 2: the RESOLVED note pushed the bullet past 600 characters.)

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

## The 'small' second input of a compressed-layer attention misses the fixed 0.03 floor at the reference's precision

C.enc_source.attention attempt 1 (tt/sparse_attention.py, layer 2, 2x2, chunk 2048). Every check passes except the
auto second input `small` (every float input scaled by a power of 2 to RMS ~1e-3): rel 0.0352 vs the CPU step,
limit 0.03 = the fixed second-input floor (component_second_rel); its calibrated limit is 0.0057 (2 x the input nudge,
0.0029). Golden 0.0350 <= 0.1616, vs CPU 0.0316 <= 0.0668, chunk0 0.0292 <= 0.0545, layer14 0.0287 <= 0.0529, mixed
0.0298 <= 0.0505, big 0.0230 <= 0.0491.

Probe `tests/ttnn/unit_tests/operations/ds41_sparse_attn/probes/probe_004.py` (device o captured before the inverse
RoPE, float64 statistics):

| case | device o vs CPU o | kernel dtypes (bf16 q, P, o) vs CPU o | CPU tail on bf16(CPU o) | CPU tail on device o | device step |
|---|---|---|---|---|---|
| small | rel 0.00227, norm ratio 0.99911, bias -8.9e-4 | rel 0.0018, bias -2.2e-4 | 0.0329 | 0.0352 | 0.0352 |
| golden | rel 0.00203, norm ratio 0.99992, bias -8.2e-5 | rel 0.0018, bias -7e-6 | 0.0311 | 0.0316 | 0.0316 |

- The official kernel (inference/kernel.py sparse_attn) returns o in q's dtype, bf16. Rounding the CPU's own o to bf16
  flips 3.1 % of the fp8 values after the inverse RoPE and gives rel 0.0329 on `small`, above the 0.03 floor. The
  input nudge (q_resid bf16; kv / compressed / top-k fixed) moves nothing there: the scores are ~0 (row max 0.000-0.002),
  the softmax is flat, so q's rounding does not reach the output.
- What the device adds beyond the kernel's dtypes is a coherent shrink of o (norm ratio 0.99911, per head 0.99813 to
  1.00006) only in the flat-softmax regime (golden -8e-5). The sink is not the cause (|sink| < 0.6, its term is 0.2 % of
  the denominator, the fit per head is inconsistent). It fits DS41-16: every in-tile partial dot of P@V is truncated to
  about TF32, and averaging many keys of mixed sign (uniform P) amplifies it. No setting removes it (no split matmul).
- Raising the output to fp32 (WINDOW_SDPA_OUT=fp32) would be above the reference (FIX.2), so not done. The check needs
  an owner waiver (spec tests.waivers: `enc_source.attention`) or a framework floor for second inputs that includes
  the official arithmetic's excess (as the golden's official_floor does: 0.0808).

## NUMERICS=native fails the official per-step swap checks upstream of the experts
P.8 (attempts 1 and 2), change on (EXPERTS_COMBINE=weighted NUMERICS=native FFN_NORM_DTYPE=bf16), frozen tests in gate
order: the 8 test_c_*_experts pass (vs CPU rel 0.0298 L0 .. 0.0410 L24, limits 0.085 .. 0.120; deterministic);
test_swap_swa_12_experts fails swap_q_a (rel 0.011596 <= 0.01040) and swap_ffn_hc (0.000197 <= 0.00015, covered by the
owner's hc_fn waiver, DS41-43). Attempt 1 also ran swa_engram_15 (q_a 0.013907 <= 0.01191) and enc_source_14 (q_a
0.012939 <= 0.01114, compressor 0.101759 <= 0.07459, indexer unexplained flips 170).

Probe (attempt 2, golden q_a input, chunk of the component rung, 2x2): CPU native q_a = rms_norm(mm(bf16(x), wq_a),
q_norm) vs the official CPU (fp8 q-dq of x first): rel 0.011637 (L0), 0.013901 (L1). Device NUMERICS=native vs CPU
native 0.000143 / 0.000160; device official vs CPU official 0.000101 / 0.000105. Bias device native - CPU native
-6.7e-7 / +5.9e-7 of the rms. So the swap failure is exactly the official fp8 rounding the device skips (rule 17), no
device error: the per-step limits are 2 x the official arithmetic's noise floor and the reference hook stays official
(owner, P.8 brief). The compressor's fp4 q-dq (rel ~0.1) is the same kind.

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

## Real-time profiler off on chips 7 and 10 (8x4 Galaxy, FIX.3, DS41-120)

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

## FIX.21 FORMAT_ATTRIBUTION: which device format makes the device's error larger than DeepSeek's reference

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
