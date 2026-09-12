# AutoDebug: batch-3 changed-token accuracy

Source-only investigation, 2026-09-11. No implementation changes, target
execution, or device access were performed by this investigator. Existing
logs include the stage owner's exact functional control.

## Finding

The observed failure is shared with the unchanged functional decoder. There
is no source evidence establishing a new uneven-group or trace-state bug.
Two concrete shared precision differences from the installed HF implementation
are plausible causes; neither is proven to cause this input's failed PCC.
The existing 0.995 gate must remain in force during diagnosis.

Inspected implementation SHA256:

- Fused: `a362a3e1be350b1fdd2edfa50aaf724727bcd2d37822883c673632bad758ff04`.
- Functional: `c790a57d53a3d2bbf9461400088dbd6281aba72ce98285513badc6bb8ad9623c`.

The exact failing case is real layer 0, batch 3, length 257, continuation
enabled, seed `123 + 257`. HF processes the 257-token prefix in one call;
TT's continuation processes 33, 128 and 96 tokens. The original TT prefill
uses 128, 128 and 1 tokens, but the test substitutes the continuation state
before all decode checks (`tests/run_fused_decoder.py:177`).

| Existing evidence | Fused | Functional control |
| --- | ---: | ---: |
| Prefill HF PCC | 0.99894449696 | 0.99892975785 |
| Continuation HF PCC | 0.99893819912 | 0.99894828779 |
| Original-token eager and traced HF PCC | 0.99995905161 | 0.99995726347 |
| Changed-token traced HF PCC | **0.99362123013** | **0.99300992489** |

Sources: `uneven_batch3_linear.log`,
`uneven_batch3_functional_control.log`, and exact commands in `commands.log`.
The fused sweep's lengths 1, 2, 3, 31, 33 and 129 pass all applicable checks.
Its JSON stops before the failed length-257 row; the log is the failure
evidence. The passing original-token output and deterministic replay do not
establish changed-token eager/trace parity, which is currently unmeasured.

## Ranked hypotheses

### 1. Shared recurrent-prefix precision mismatch

**Verified source discrepancy; leading hypothesis, not a proven root cause.**

In the installed `python_env/lib/python3.12/site-packages/transformers/`:

- `models/qwen3_5/modeling_qwen3_5.py:262-264` promotes the chunk recurrence
  inputs to FP32; the returned recurrent state is FP32.
- `cache_utils.py:925-930` derives `LinearAttentionLayer.self.dtype` from the
  convolution state, which is BF16 for this BF16 checkpoint/input path.
- `cache_utils.py:937` allocates the recurrent cache using that `self.dtype`,
  and `:989` copies the FP32 recurrence result into it. This rounds the HF
  prefix state to BF16 before the next cached token.
- Both TT implementations allocate FP32 recurrence buffers and retain FP32
  between calls (`tt/fused_decoder.py:136-138,426`;
  `tt/functional_decoder.py:111-113,275`).

Thus the expected changed token consumes an HF BF16-rounded prefix state,
while TT consumes an FP32 prefix state computed through differently sized
chunks. A different query can expose prefix-state errors that the original
query does not, and the gated normalization/MLP can amplify the difference.
This explains why eager/traced original-token PCC can be excellent while a
changed token fails in both implementations. It does not yet prove the
observed magnitude or identify the batch member responsible.

HF's BF16 cache is actual installed-reference behavior, not automatically a
test bug. A diagnostic FP32 HF cache or state-aligned oracle must not silently
replace the required unmodified-HF oracle. Also, rounding TT state after
every internal chunk is not equivalent to HF's one prefix-boundary rounding;
do not make that broad runtime change as the first experiment.

### 2. Shared RMSNorm preparation and compute precision

**Verified source discrepancies; secondary hypotheses.**

HF `Qwen3_5RMSNorm.forward` computes normalization in FP32 and multiplies by
`1.0 + self.weight.float()` before converting the final output to BF16
(`modeling_qwen3_5.py:745-750`). Both TT loaders calculate that folded weight
in FP32 but upload it with the helper's default BF16 dtype
(`fused_decoder.py:56,69-76`; `functional_decoder.py:57,70-75`). This loses
the extra precision of the addition before normalization uses the weight.

Also, both generic `_norm` methods omit `compute_kernel_config`; the explicitly
configured HiFi4/FP32 matmul config does not apply to those calls. The native
RMSNorm default is `approx_mode=true, fp32_acc=false`
(`ttnn/cpp/ttnn/operations/normalization/rmsnorm/rmsnorm.cpp:16-19,62`).
These paths affect every token, including the prefix from which recurrence
is built. Input-dependent amplification is plausible but unmeasured.

If state-aligned diagnostics do not explain the gap, test folded FP32
weights and explicit RMSNorm compute config separately. Do not combine them
into one trial. The layernorm validator does not require a tiled gamma to
be BF16; confirm the complete selected program supports the mixed precision
before retaining a change. Keep `linear_attn.norm.weight` separate: that norm
has no `1 + weight` transform and its fused operation requires BF16 weights.

### 3. Other inherited rounding differences

HF's `l2norm` runs multiply/reduction/rsqrt/multiply in the query's BF16 dtype
before the recurrence promotion (`modeling_qwen3_5.py:240-243,259-264,331-336`).
The functional implementation explicitly normalizes in FP32 and casts the
result to BF16 (`functional_decoder.py:227-235`); the fused flat scan uses
its in-kernel normalization (`chunk_gated_delta_rule.cpp:200-204` and
`device/kernels/compute/chunk_gdn_prep.cpp:402-436`).

HF's gated norm also has deliberate intermediate BF16 conversions
(`modeling_qwen3_5.py:193-202`) that differ from the fused FP32 scan-output
normalization. These are relevant if the first two controls fail; current
evidence does not justify blaming either native fused operation.

## Trace, grouping and test-oracle audit

- The test deep-copies HF prefix state before original-token decode
  (`run_fused_decoder.py:92-95`), and the changed-token HF call consumes that
  copy (`:300-303`). No intervening mutation of `prefix_cache` is apparent.
- TT snapshots are taken after continuation. Every original/repeated/changed
  replay restores the same state buffers. Refresh writes the input buffers
  captured by the closure; `execute_trace(..., blocking=True)` completes
  before the host comparison (`:210-231,305-315`). No source-level stale
  pointer or missing restore is apparent.
- Linear attention ignores position/RoPE in both implementations and in the
  HF layer. The changed per-user positions do not imply a mismatched prefix
  for this layer kind; the per-user truncated-cache oracle is used only for
  full attention.
- With 48 value heads and 110 cores, fused scan groups are 2 and 1. Native
  head-major output is `[B*HV,T,V]`, and native final state is `[B,HV,K,V]`
  (`chunk_gated_delta_rule.cpp:340-363`). Concatenating each on dimension zero
  preserves users 0, 1, 2 in the expected order. The subsequent gated-norm
  operation derives batch from leading dimension divided by `HV`. No shape
  contradiction was found.
- Padded g=0 and beta=0 make extra recurrence steps identity in the delta-rule
  equations, even with nonzero padded convolution Q/K/V. History updates use
  the logical `t` before the padded recurrence and preserve the last three
  real tokens (`fused_decoder.py:384-402`).
- A source audit cannot exclude a native B=1/B=2 program-cache or dataflow
  problem. However, an inherited arithmetic decode path fails the same input,
  and the fused smaller B3 cases pass, so uneven grouping is currently a
  lower-priority hypothesis.

## Discriminating experiment sequence

1. **Add diagnostics without weakening assertions.** For this same seed,
   save expected and actual changed-token outputs before the existing gate,
   including per-user PCC, maximum absolute error and RMS error. Add changed
   output to the existing functional/fused comparison artifacts. Restore
   prefix state and run the changed token eagerly as well as through the
   existing trace, then compare outputs and resulting conv/recurrent states.
   Eager/trace equality with the same HF failure rejects a trace-specific
   explanation. High fused/functional changed-output PCC rejects a newly
   introduced fused grouping error as the primary explanation.

2. **Measure the cache discrepancy directly.** Record actual HF conv and
   recurrent dtypes. Preserve the FP32 recurrent result returned during HF
   prefill before `update_recurrent_state` rounds it, without modifying that
   HF call's original result/cache. Compare both that value and the stored
   BF16 cache to TT's prefix state; compare conv history against the last
   three entries of HF's four-entry history.

3. **Controlled changed-token state swaps.** Retain the original HF expected
   output as the primary gate. In separate diagnostic runs use (a) HF conv
   history and original BF16 recurrent state promoted to FP32 as TT's prefix,
   and (b) HF conv history and the captured pre-rounding FP32 recurrent state.
   Each trial restores the exact selected prefix before eager/replay. Compare
   with corresponding HF cached-token runs using deep-copied state. This
   distinguishes prefix-state disagreement from changed-token arithmetic.
   Diagnostic state upload belongs exclusively in test setup, outside the
   measured/device-only path.

4. **Isolate normalization only if needed.** Keep the current cache policy
   and first test FP32 folded norm-weight upload alone on the exact failed
   case. Separately test passing the established compute config to generic
   RMSNorm. Record HF PCC for prefill, continuation, original token and changed
   token, plus functional equivalence. Keep a change only if its effect is
   reproducible and supported; then rerun the affected existing gate set.

5. If these controls leave a B3-specific discrepancy, feed exactly the same
   saved inputs and states through three independent B1 scans and compare
   users with the B3 2+1 grouping. Do not regenerate different random B1
   inputs. A user-2-only divergence would justify lowering that case through
   the slice/concat/native program-cache path.

Do not lower PCC, choose a different random seed to hide the failure, or
declare an oracle defect merely because a higher-precision reference is
closer. This report identifies experiments and concrete numerical contract
differences; it does not establish a completed fix or a stage pass.
