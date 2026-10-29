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

## DS41-125: contract KV limit vs deep layers over every position (8x4, K.1 attempt 5)
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

## Quick A/Bs skip the contract tests: P.9 tipped test_adapter_acks unseen
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
