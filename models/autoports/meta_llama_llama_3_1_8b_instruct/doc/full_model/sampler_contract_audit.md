# Common sampler contract audit

Both common implementations were read before final selection:
`models/common/modules/sampling/sampling_1d.py` and
`models/common/sampling/` (`SamplingGenerator`/`TTSampling`). The comparison is
against the exact P300 1x4 TP4 contract, not a generic feature checklist.

| Contract | Sampling1D | SamplingGenerator/TTSampling |
| --- | --- | --- |
| TP4 local logits | Direct `[1,1,32,32768]` padded-vocab input | Compatible when vocab 128256/padded vocab 131072 are supplied |
| Split gather | Caller sets the accepted two links directly | `max_top_k=32` derives one link (`tt_sampling.py:157-164,552-587`) |
| Fixed/inactive slots | Static batch-32 caller-owned buffers | Supported by its formatter and slot state |
| `tt_out_tok` | Stable output tensor supported | Stable output tensor supported |
| Parameters | Persistent caller tensors, trace compatible | Persistent module tensors, request-boundary updates |
| Explicit seeds | Real request seed then `UINT32_MAX` permits device RNG advance in trace | Explicit seeds copy H2D every token and disable internal trace (`generator.py:384-418,856-932`) |
| Penalties | Separate `Penalties1D`; not enabled in this stage | Built in, but accumulator update is outside internal sampler trace |
| Logprobs on P300 | Shared calculator unavailable on four devices | Same four-device limitation (`tt_log_probs.py:392-399`) |
| Integration | Exact config maps directly | Requires a legacy `args` facade plus changes for two links and trace-stable explicit seeds |

`Sampling1D` is selected. The alternative is rejected by source contract and
is not benchmarked as if it were equivalent: its stock configuration would be
less optimized (one link) and its explicit-seed path would be untraced with
per-token host work. Modifying those contracts would be custom sampler work,
which is unnecessary because Sampling1D fits.

The decoder's tensor-parallel reductions use a two-link Ring. Sampling1D's
selected exact greedy path computes local argmax, packs each rank's FP32
`(value, global-token)` candidates, and uses one two-link gather on the 1x4
mesh. The Ring setting applies to force-argmax configuration but is clamped to
Linear on fewer than eight devices.

Separately, both semantically greedy strategies within Sampling1D were run on
exact full-model logits. Exact rank-candidate greedy and full-vocabulary
force-argmax returned the same token; an adversarial batch-32 oracle also
matched host argmax on every row. The selected path measured 0.974 ms eager
and 0.795 ms traced; force-argmax measured 1.242 ms eager and was rejected.
Sampling is 8.84% of separate model-plus-sampler trace time and does not
dominate token-out. Exact fields are in `sampler_comparison.json`.
