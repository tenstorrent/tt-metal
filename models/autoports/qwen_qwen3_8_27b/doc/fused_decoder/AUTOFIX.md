# AutoFix: real batch-three changed-token accuracy

## Starting evidence

`AUTODEBUG.md` investigated real layer 0, batch 3, length 257, seed 380,
with TT continuation chunks 33/128/96 and stock HF prefix length 257.
`uneven_batch3_linear.log` records changed-token PCC **0.9936212301**;
the frozen functional control also fails at **0.9930099249**. The acceptance
threshold remains **0.995**, and the stock installed HF reference is retained.

## Controlled experiments

`tests/probe_fused_batch3.py` reproduces the exact input and the original
`randperm(30)` RNG consumption. It intercepts HF's recurrent-cache update to
copy the incoming state before the unchanged HF update rounds it, and runs
five test-only prefix-state substitutions. All host uploads/readbacks occur
outside the guarded device-only decode. JSON and `.pt` artifacts preserve the
stock HF output, diagnostic aligned outputs, and TT outputs/states.

| Hypothesis/control | Measured result | Verdict |
|---|---|---|
| HF rounds FP32 recurrence to BF16; TT retains FP32 | Confirmed dtypes. HF stored versus pre-copy state PCC 0.99999865; changed output with raw FP32 HF cache versus stock HF PCC **0.99999750**, RMSE 0.001626 | Real discrepancy, refuted as the primary cause |
| Changed input uses stale trace state or uneven grouping mishandles user 2 | Original failure reproduces exactly in eager and replay; outputs, final recurrence and conv history are bitwise equal in all five state controls. User PCCs are **0.98158002, 0.99895525, 0.99913859** | Trace-specific cause refuted; user 0 dominates, not the leftover group |
| Prefix arithmetic contributes | TT recurrence versus raw HF PCC 0.99963661, RMSE 0.001829. Substituting only HF recurrence yields stock-HF output PCC **0.99661028**; substituting HF conv alone yields 0.99227643 | Verified contribution; state replacement is diagnostic only |
| BF16 folded RMSNorm weights cause the failure | FP32 input/post-attention folded weights alone worsen changed PCC to **0.99253255** | Refuted as a remedy; no weight dtype change retained |
| Generic RMSNorm defaults differ materially from HF FP32 normalization | Passing the already-established `decoder.ckc` to generic RMSNorm alone improves changed PCC to **0.99988180**, prefill to **0.99996527**, continuation to **0.99996512**. Changed per-user PCCs: **0.99950445, 0.99999183, 0.99997640** | Verified remedy |

Artifacts: `probe_batch3_cache.*`, `probe_batch3_norm_weight.*`,
`probe_batch3_norm_compute.*`. These three controls used runtime SHA256
`a362a3e1be350b1fdd2edfa50aaf724727bcd2d37822883c673632bad758ff04`;
the latter two changes were test-only monkeypatches. The initial log precision
label mistakenly named conv width 12288; tensor artifacts show the actual
shape `[3,3,10240]`, corrected in JSON and derived directly in the probe.

The state-aligned HF oracle gives 0.99558604 against the original TT output;
this is diagnostic evidence, not a replacement acceptance result. Passing
with substituted state does not authorize host state injection in runtime.

The final-source probe `probe_batch3_corrected.*` independently confirms the
same **0.99988180** stock-HF result and bitwise changed eager/replay state
parity. `probe_batch3_functional_retry.*` reproduces unchanged-functional
**0.99300992** and bitwise eager/replay parity. Its first attempt
(`probe_batch3_functional.log`) stopped at host page-config validation because
the diagnostic restore assumed row-major conv for the functional tiled cache;
the probe now obtains the target cache layout directly. No reset was needed.

**Frozen-control limitation:** saved changed-token output PCC is
**0.99996871** between pre-fix fused and frozen functional (both fail HF),
but **0.99181628** between corrected fused and frozen functional. The corrected
result passes stock HF while the frozen functional result retains its norm
error. Thus this particular changed-token case does not satisfy a 0.995
corrected/frozen-functional equivalence gate. Existing representative B1
prefill/decode equivalence gates still pass; this distinction must remain
visible in stage review, not be reported as universal functional equivalence.
See `probe_batch3_equivalence.json` and its saved `.pt` inputs.

## Minimal retained change

Generic RMSNorm in `tt/fused_decoder.py` explicitly receives the existing
HiFi4/nonapproximate/FP32-destination compute config for **linear attention**.
BF16 weights/activations/convolution history and FP32 recurrent state remain
unchanged. Frozen `functional_decoder.py` is untouched. Context remains 262144.

A blanket change to full attention was separately measured in
`fixed_full_benchmark.*`: decode median 2.296987 ms versus the previously
correct default 2.258905 ms. Since no full-attention failure required this
change, the retained fix is conditional on `linear_attention`. Full attention
continues passing `None`, preserving its native default behavior.

Final runtime SHA256:
`3f324562925d42fb0c2997672f86fe68926a21869d3f2d351d1f5c3ba9792035`.

## Performance controls

The final source's B1/S128 benchmarks pass HF and frozen-functional PCC gates.
Linear median warmed prefill/decode: **3.074463 / 2.494085 ms**; full:
**2.754356 / 2.260432 ms**. The frozen stage baselines were 4.84831/3.35083 ms
and 3.55431/2.44381 ms respectively. Full decode returns to its prior default
latency; no full-attention arithmetic change is retained.

Three paired final-source B1 linear benchmarks alternate default/split order.
The split control is the closest valid alternative; `untilize_out=False`
followed by the explicit row-major conversion preserves correctness.

| Trial | Default decode median, ms | Split-control decode median, ms |
|---|---:|---:|
| 1: default then split | 2.493290 | 2.489839 |
| 2: split then default | 2.493872 | 2.492960 |
| 3: default then split | 2.491243 | 2.493221 |
| Pooled 90 samples | 2.492811 | 2.492334 |

The winner reverses, and pooled medians differ by only 0.000476 ms (0.019%).
This does not establish a repeatable split-projection improvement. Keep the
existing simpler graph. All samples and precise commands are retained in
`corrected_pair{1,2,3}_{default,split}.*`, `corrected_paired_summary.json` and
`commands.log`. Pooled prefill medians are 3.077893/3.080678 ms; prefill uses
identical runtime operations in these two controls.

## Verification and reproduction

The final-source original B3 sweep in `corrected_batch3_linear.*` passes
lengths **1,2,3,31,33,129,257**, including continuation, original eager/traced
decode, repeated replay, changed input, and device-only guards. Minimum
changed-token PCC is **0.99988180** at 257; no threshold or input seed changed.

Exact stage-runner commands (checkout root):

```bash
bash models/autoports/qwen_qwen3_8_27b/tests/run_fusion_experiment.sh corrected_batch3_linear --layer 0 --batch 3 --lengths 1,2,3,31,33,129,257 --continuation
bash models/autoports/qwen_qwen3_8_27b/tests/run_fusion_experiment.sh corrected_linear_benchmark --layer 0 --length 128 --benchmark --compare-dir models/autoports/qwen_qwen3_8_27b/doc/fused_decoder/control
bash models/autoports/qwen_qwen3_8_27b/tests/run_fusion_experiment.sh corrected_full_benchmark --layer 3 --length 128 --benchmark --compare-dir models/autoports/qwen_qwen3_8_27b/doc/fused_decoder/control
```

The focused probes used the same environment setup as
`tests/run_fusion_experiment.sh`, then this exact command, with no `--variant`
for `probe_batch3_cache`, `--variant fp32_folded_norm` for
`probe_batch3_norm_weight`, and `--variant norm_compute` for
`probe_batch3_norm_compute`:

```bash
export TT_MODEL_BRINGUP_ROOT=/home/mvasiljevic/qwen38-full-rerun/codex-home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.4
export TT_AUTODEBUG_ROOT=/home/mvasiljevic/qwen38-full-rerun/codex-home/plugins/cache/tenstorrent-skills/tt-autodebug/0.1.5
BRINGUP_EXPORTS=$(python_env/bin/python "$TT_MODEL_BRINGUP_ROOT/scripts/environment.py")
eval "$BRINGUP_EXPORTS"
export PYTHONPATH=.:$PYTHONPATH
timeout -k 10 900 python_env/bin/python models/autoports/qwen_qwen3_8_27b/tests/probe_fused_batch3.py --snapshot /home/mvasiljevic/hf-cache/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --output models/autoports/qwen_qwen3_8_27b/doc/fused_decoder/probe_batch3_cache.json
```

On the corrected source, `--variant native_default` explicitly restores the
pre-fix RMSNorm policy for diagnosis. `--baseline` selects the unchanged
functional implementation. These are diagnostic controls, not stage passes.

Formatting and syntax checks passed:

```bash
python_env/bin/python -m black --target-version py312 models/autoports/qwen_qwen3_8_27b/tests/probe_fused_batch3.py models/autoports/qwen_qwen3_8_27b/tt/fused_decoder.py
python_env/bin/python -m py_compile models/autoports/qwen_qwen3_8_27b/tests/probe_fused_batch3.py models/autoports/qwen_qwen3_8_27b/tt/fused_decoder.py
```

No C++ or CMake changed, so no build was needed. Hardware jobs ran serially;
no reset, process kill, recovery, watcher/profiler overlap, or host fallback
occurred. The parent stage will refresh affected long-context, watcher and
profiler evidence and independent review before declaring the stage passed.

The final and functional diagnostic commands use the same probe invocation
above with output stems `probe_batch3_corrected` (no variant) and
`probe_batch3_functional_retry` (`--baseline`) respectively. The initial
functional attempt used output stem `probe_batch3_functional`.

## Final status

The original stock-HF changed-token failure is fixed with a minimal verified
linear-attention RMSNorm change. The frozen-functional adversarial changed
output still differs below 0.995 as documented above; stage approval remains
with the parent review and full validation. Hardware was closed and handed
back after the final/functional probes. No runtime edits are pending here.
