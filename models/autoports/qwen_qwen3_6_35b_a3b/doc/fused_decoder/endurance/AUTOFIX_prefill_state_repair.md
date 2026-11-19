# Prefill state-only repair: shared-input CPU evidence

CPU/source investigation only. No TTNN imports in the analysis script, no hardware execution, no decoder edits. The parent supplied `prefill_state_device.json.tensors.pt` and the newer `prefill_state_cumsum.json.tensors.pt`.

**The smallest supported next repair is a sequential FP32 SFPU state-only scan, keeping the native prefill core output.** It eliminates almost all observed cache error in the CPU comparison. It is not bit-identical to HF's chunk64 final state: 168 of 524,288 BF16 cache elements differ. Therefore the complete endurance gate must decide whether this repair is sufficient; do not describe the repaired cache as exact HF without further evidence.

## Shared-input experiment

`prefill_state_cpu_comparison.py` uses the captured normalized Q/K, V, g and beta, so no projection, convolution, norm or gate recalculation confounds the comparison. It compares HF's chunk recurrence at sizes16/32/64/128, HF's sequential recurrence, and an explicit sequential implementation with the validated eight chunks of16 left-to-right sum schedule. State remains FP32 for all128 tokens and is rounded to BF16 only at the public-call boundary.

On the newer cumsum capture, using TT inputs versus HF inputs gives **identical CPU results for every tested schedule**, including bit-exact reconstruction of HF chunk64 raw state. This isolates the large native-state discrepancy from the remaining numerically immaterial frontend differences.

| State implementation, exact frontend | Raw relative L2 vs HF chunk64 | BF16 cache differing codes | Raw max absolute error |
|---|---:|---:|---:|
| HF chunk16 | 2.366121e-7 | 127 | 5.066395e-7 |
| HF chunk32 | 2.360034e-7 | 151 | 5.960464e-7 |
| HF chunk64 | 0 | 0 | 0 |
| HF chunk128 | 3.186484e-7 | 263 | 9.834766e-7 |
| HF sequential | 3.335790e-7 | 168 | 6.556511e-7 |
| Explicit CPU-order sequential sums | 3.335790e-7 | 168 | 6.556511e-7 |

The explicit sum implementation and HF sequential state are bit-identical in FP32. The native cumsum-prefill result has raw relative L2 approximately0.0034014 and public BF16 relative L2 approximately0.0042224, orders of magnitude larger than the chunk-size/reordering differences above.

The original prefill capture had remaining frontend discrepancies. CPU chunk64 using those TT inputs had raw relative L2 0.00031045 and6,309 BF16 cache differences; native raw relative L2 was0.00345051. The cumsum projection repair removes that separate frontend confound.

## Minimal model-local state scan

Retain native `chunk_gated_delta_rule` for `o`, and compute only its replacement final state from the same normalized K/V/g/beta and the original incoming public state:

```text
S = original incoming FP32 state
for each logical prefill token:
    decayed = S * canonical_exp(g_token)
    memory = canonical_sum_128(decayed * k_token[:, :, None])
    delta = (v_token - memory) * beta_token[:, None]
    S = decayed + k_token[:, :, None] * delta[:, None, :]
return native_core_output, S
```

No Q scaling, Q-state output contraction, or output norm is needed in this state-only scan. Reuse the validated canonical decay lookup rather than substituting an unvalidated exponential. Reuse the exact FP32 reduction schedule already proven for decode. Use separate SFPU multiply/add for the state update; native MAD cancellation was independently shown not to preserve IEEE-FMA residual precision. Preserve the original state before native output copying, and do not round state between internal chunks or tokens. For a continuing public prefill call, consume the once-rounded public incoming state as the existing contract requires.

For128 tokens, this is approximately128 state contractions/updates rather than128 complete decode calls. It has a significant prefill cost that must be measured. It leaves traced decode unchanged and keeps native prefill core output. The 168 cache-code differences are distributed across heads; six heads match exactly. The largest public-cache relative L2 among the sequential heads is0.00019390 at head26, still far below its native0.00401944. Full per-head evidence is in `prefill_state_repair_per_head.json`.

Do not truncate the scan to a fixed recent tail without evidence: some heads retain long-term state, and continuation/non-aligned/public-call behavior must remain intact.

## Why native chunk32 versus HF chunk64 is insufficient as an explanation

HF's chunk implementation computes FP32 triangular-attention algebra and carries state between chunks (`modeling_qwen3_5_moe.py:245–323`). Changing only chunk size on CPU changes raw state by roughly2–3e-7 relative L2, not0.0034.

The native scan explicitly uses matrix products for state reads and updates: `chunk_gdn_scan.cpp:8–12` documents `v_new = T_inv @ (v_beta + nkd @ S)` and `S = (dl*I) @ S + k_dec_t @ v_new`; `chunk_gdn_math.hpp:79–105` implements those through `matmul_tiles`. FP32 storage/accumulation does not establish full-FP32 operand precision through the matrix unit. The older single-kernel implementation similarly performs `w @ S`, `k_dec_t @ v_new` and scalar state multiplication (`chunk_gated_delta_rule.cpp:380–407`). These source facts support investigating FPU intermediate narrowing. They do not attribute a precise fraction of the measured error to one native instruction.

## If sequential state repair fails endurance

The CPU experiment proves exact HF chunk64 replay is possible from the now-correct frontend. A model-local reproduction would need HF's chunk64 cumulative-gate arithmetic, triangular inverse update order, and full-FP32 matrix products/sum order. Merely selecting native chunk64 does not remove matrix-unit narrowing. Implementing that full schedule is substantially more complex than the state-only sequential correction and should be justified by a failed full-stream discriminator, not presumed necessary now.

## Artifacts and command

```bash
python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/prefill_state_cpu_comparison.py \
  --capture models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/prefill_state_cumsum.json.tensors.pt \
  --label cumsum
```

- `prefill_state_cpu_comparison.json`: original capture comparison.
- `prefill_state_cpu_comparison_cumsum.json`: new exact-frontend comparison.
- `prefill_state_cpu_cumsum_hf_states.pt`, `prefill_state_cpu_cumsum_tt_states.pt`: local CPU state tensors.
- `prefill_state_repair_per_head.json`: per-head BF16 state comparison.
- `prefill_state_cumsum.json` and its tensor file: parent-owned native device evidence.

No hardware repair or endurance success is claimed by this CPU-only memo.
