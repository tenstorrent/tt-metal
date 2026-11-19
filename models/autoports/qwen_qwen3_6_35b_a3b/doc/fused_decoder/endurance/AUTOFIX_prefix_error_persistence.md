# AutoFix: prefix-state error persists through BF16 cache rounding

**The captured prefix recurrent-state error survives the full8192-token stream, even when every subsequent decode input and operation is canonical CPU HF.** It is not safe to defer prefill repair on the assumption that decay will erase it.

CPU-only replay completed in253 seconds. No TTNN import or device work. Artifacts: `prefix_error_replay_cpu.py`, `prefix_error_replay_cpu.json` (compact checkpoint metrics, initial capture/script hashes). Reproduce from repo root:

```bash
python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/prefix_error_replay_cpu.py
```

## Controlled experiment

Initialize two recurrent states from the saved HF and TT incoming states at position128 in `endurance_precise_probe.json.pos128.pt`. Their initial relative L2 difference is .01050923. Give **both** canonical HF convolution history, norm, QKV/A/B projection, Q/K normalization, g, beta and V for the complete seed913 decode stream. Use HF's separate FP32 recurrence operations followed by its BF16 public-cache rounding at every decode call. Only the initial recurrent state differs; this excludes the TT prefix convolution-history discrepancy and all subsequent TT arithmetic discrepancies.

Validation is unusually strong: regenerated input norm, QKV projection, Q, K, V, g, beta, HF-seeded incoming/outgoing recurrent state and core output are **bit-identical** to all nine available canonical captures at128/129/130/1024/3979/5011/7551/7703/8157. Hence the paired comparison is not based on an approximate surrogate frontend.

In parallel, transport the initial difference through the unrounded linear recurrence in FP64, using the same canonical FP32 decay coefficients and BF16-normalized K/beta. This separates ordinary mathematical contraction from the actual BF16-cache evolution.

| Absolute position, incoming state | Unrounded initial-error transport, relative to HF state | Paired BF16-cache initial-only difference | Observed TT capture difference |
|---|---:|---:|---:|
|128|.01050923|.01050923|.01050923|
|1024|.00069832|.01169728|.01211738|
|2048|.00023311|.01287384|—|
|3979|.00004647|.01238155|.01267365|
|5011|.00002889|.01368676|.01499966|
|7551|.00000956|.01244385|.01184153|
|7703|.00000913|.01206220|.01177386|
|8157|.00000761|.01154606|.01134563|
|8319|.00000695|.01129977|—|

After the final update, paired cached states still differ in271685/524288 elements, relative L2 .01132001, max absolute .03125. Initial-only core-output relative differences are .007769 at3979, .035343 at5011, .009769 at7703, .009870 at8157. These are core metrics, not full-decoder PCC claims.

Similar norms to the observed TT error do not establish a linear attribution percentage: later TT discrepancies interact with BF16 rounding, and error vectors may differ. The experiment nevertheless proves that initial state alone is sufficient to sustain a comparable state-error magnitude after8K canonical updates.

## Why a simple decay argument fails

Before cache rounding, the exact difference equation per head is

`E_next = exp(g) * (I - beta * k k^T) * E`.

The source recurrence decays state, subtracts its K projection from V, and adds the beta-weighted rank-one update. V cancels in the difference. The rank-one factor has eigenvalue`1 - beta*||k||²` along K and1 orthogonally. Across the replay, maximum`beta*||k||²` is .98123395, so that factor is nonexpansive; repeated canonical linear updates do contract the transported initial difference.

Decay alone is also highly head-dependent. Checkpoint A_log spans−3.984375 to4.28125 and dt_bias spans−5.28125 to15.5625. Integrating actual g over the stream, the slowest scalar-only survival factor before the last update is .160403 for head23, versus .012598 for head15 and .003068 for head7. One sampled g or an average across heads cannot justify a uniform forgetting claim. The actual unrounded error contracts more because of the rank-one updates.

HF's implemented public update is instead `Q_BF16(update(state))`. The paired difference includes the **difference of two rounding errors** at every call. Those errors depend on the differing state values; they do not cancel. Therefore the linear contraction bound does not bound the repeatedly quantized pair by itself. The measured unrounded transport shrinks roughly1500-fold relative to state norm, while the canonical quantized pair retains~1.1% error.

## Consequence for the next repair

If the latest exact-norm/exact-recurrence candidate passes with a canonical HF prefix seed, the evidence supports repairing prefill recurrent-state construction next. The present experiment is sufficient to reject “the prefix error should wash out by8K.” It does not prove that prefix repair alone fixes a candidate that still fails under canonical seeding; that would indicate remaining decode differences as well.

Preserve BF16 cache rounding at every **public decode** call, and only once per **public prefill** invocation. Removing or weakening it to make errors decay would change the canonical model semantics. Compare a repaired prefill state at the same BF16 cache boundary as this experiment.
