# Canonical split sampling

Both common implementations were inspected before token-out implementation:
`models/common/sampling/generator.py` (SamplingGenerator/TTSampling) and
`models/common/modules/sampling/sampling_1d.py` (Sampling1D). Both support vocabulary
shards, top-k/top-p and `tt_out_tok`. SamplingGenerator owns parameter, penalty,
seed, logprob and trace-bucket state and supports additional topology families.
Sampling1D accepts explicit parameter/seed buffers and caller-owned traces; it fits
this TP4 1x4 generator directly. Its logprob support is topology restricted and it
has no penalty owner. Select Sampling1D; reject SamplingGenerator for this stage's
contract because its additional state/trace ownership requires an unnecessary
adapter. Penalties and logprobs are not advertised by this readiness generator.
No custom sampler was written.

The model trace produces vocabulary-sharded logits in a persistent buffer. The
separate sampler trace writes `tt_out_tok=self.tokens`, the exact persistent model
input. Position and RoPE indices increment on device in the model trace; seeds
increment in the sampler trace. Parameter updates happen at request boundaries.
`set_sampling_params` accepts per-slot or scalar k/p/temperature/seed. Seeds are
preserved exactly in uint32 backing storage; the valid range reserves enough
increments for the configured context and avoids the runtime random-seed sentinel.
Reset restores the chosen seeds. The reduced trace probe covers seed42 and a high
uint32 seed, sampled k8/p0.9/temperature2, reset determinism, and greedy alternation.

|Semantically greedy common strategy|B1 replay µs|B32 replay µs|Decision|
|---|---:|---:|---|
|Sampling1D normal top-k, k1/p0/temp1|451.889|487.976|Selected|
|Sampling1D force-argmax|1625.024|2454.754|Rejected: slower|

These are 1000-replay host-wall sampler-only measurements with trace allocation
tracking disabled, not device times or full-model performance. Both select the global
maximum and agree with first-index host argmax on real reduced-stack logits. Both
also pass synthetic boundary maxima at token IDs7,62079,62080,124160,186240,248319,
covering all vocabulary shards and the final valid token. Artifacts:
`sampling_b1.json`, `sampling_b32.json`, `probe_sampling_b1_03.log`,
`probe_sampling_b32_03.log`. The force-argmax comparison is genuinely greedy, not a
sampled alternative. Local candidate width32 and power-of-two padding are common
sampler requirements; they do not alter semantic k=1.

The LM head packs decode users into tile rows. B32 terminal trace time fell from
53.104ms to1.975ms with exactly equal logits (`terminal_packing_b32.json`), preserving
the inherited inter-layer residual layout. Reduced profiler reports separately
attribute terminal and sampler work; their scope is not all64layers.

Optimized generation reads only selected token IDs for caller output/EOS stopping.
It never writes those IDs back for feedback. `host_sampling=True` explicitly selects
greedy diagnostic compatibility and rejects non-greedy parameters; it is excluded
from all optimized performance. Teacher forcing explicitly overwrites the next token
via the readiness callback and is labeled separately. EOS comes from checkpoint
`generation_config.json`; fixed128-token benchmark requests explicitly disable EOS
stopping. Finished mixed-batch rows remain physical slots until peers finish, while
returned output ends at EOS; external schedulers can instead use inactive slots in
the low-level API.

Final outK4 reduced B1 device profiling (`tracy/b1/trace_phase_timing.json`),
using synthetic continuation at4223, measures the
complete sampler-trace envelope at558.194us maximum across devices, versus
2641.116us for the two-layer model trace; each includes its own operations and
gaps. Their per-device maxima must not be summed as a single-device timeline.
This is reduced-stack attribution, not full-model device timing. At the actual
long-generation divergence, both common samplers select the same unique maximum279
from identical saved BF16 logits (`long_quality_sampler_only.json`); host-sampling
and device-sampling512 outputs are identical. The quality failure is not repaired
by changing sampler choice.

Reduced B32 bounded capture (`tracy/b32/trace_phase_timing.json`) measures sampler
maximum575.425us and two-layer model-trace maximum5066.132us, using synthetic
continuation state at position4223. The complete combined window is5544.812us
(`reduced_stack_timing.json`); differing per-device maxima are not additive.
Capture1 is the prepared inactive-row model variant and is intentionally not
replayed in this all-active timing window; report warnings for that trace do not
indicate a missing measured replay. These are not all64 device-time measurements.

The pre-repair B1 capture remains under `tracy/b1_outk48_baseline`. Its longer
unmeasured replay history differs from the bounded final harness, so prefill
envelopes are not a controlled geometry speed comparison. Final B1/B32 synthetic
continuation scope is explicit in each `scope.json`. `kernel_policy_audit.json`
confirms final native B1 K blocks4/17 with LoFi/BF8 output in the recorded operations.
