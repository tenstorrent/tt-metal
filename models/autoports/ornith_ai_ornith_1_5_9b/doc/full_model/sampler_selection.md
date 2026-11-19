# Common sampler selection

Selected: `models/common/sampling/` in this tt-metal checkout. The generator
owns trace capture and persistent slot state around its existing device math.
No custom argmax, probability distribution, top-k/top-p, or penalty arithmetic
was introduced.

The alternative inspected was
`/home/ttuser/ornith-9b/dependencies/tt-transformers/src/tt_transformers/modules/sampling/sampling_1d.py`,
commit `25ebcdb3311bb86d1cb4ceff77bb8bf481f4cc73`. This is an interface/source
comparison, not a claimed cross-library timing benchmark.

| Requirement | Common SamplingGenerator | Sampling1D | Decision for this generator |
|---|---|---|---|
| Persistent state | Owns parameters, TTPenalties histories/counts, seed manager and trace buckets | Declarative config owns index/seed/user-ID buffers; k/p/temp are per-call inputs | Common supplies the model-owned penalty/history machinery needed for slot reset/resume |
| TP4/CCL | Existing sampling CCL interface accepts the model mesh/topology adapter | Config exposes mesh, CCL, topology, gather links and device buffers | Both can represent this topology; no single-chip fallback needed |
| Device token feedback | `tt_out_tok` supported | `tt_out_tok` supported | Both meet the persistent token-output requirement |
| Trace boundary | Common high-level capture uses a broad corruptible scope; its seeded high-level route can do eager work | Caller supplies persistent parameters and wraps module execution; lazy device buffers must be materialized before capture | Own a strict split trace wrapper; do not use the common high-level trace or seeded eager route |
| Greedy / sampled | Native common top-k/top-p, force-argmax option, temperature | Native top-k/top-p and optional force-argmax; k/p/temp must be all supplied or all absent for argmax | Common supports both required modes under the same delivered wrapper |
| Layout/padding | Sampling operates on physical 32-row buffers and vocabulary shards; local candidates are gathered | Explicit padded versus valid vocab sizes, maximum top-k and optional power-of-two shard padding | Keep the model's existing vocabulary shards and format physical 32 rows once |
| Seeds | Existing seed/history data structures; wrapper restores admission state and advances device seeds inside the sampling trace | Mutable seed buffer or per-call override; caller owns request progression | Wrapper owns deterministic device progression and inactive-row preservation in either design |
| Penalties | TTPenalties applies presence, frequency and repetition with prompt/output state | `decode_forward` accepts already-processed logits; its docstring assigns penalty application to the caller | Choosing Sampling1D would require separately attaching penalty/history orchestration; common already supplies it |
| Log probabilities | Common log-probability machinery exists | Optional sampled-token log probabilities from the top-k path; argmax returns none | This stage exposes full unmodified host logits for plugin-owned log-probabilities/processors, with the declaration recorded in serving_contract.json |

Sampling1D is rejected for the additional penalty/history integration it would
require here, not because it lacks TP sampling or token feedback. Its configurable
padding/layout strategies remain possible future optimization candidates.

The delivered greedy path is common top-k 1 with tile-shaped local candidates.
`sampler_probe3.json` compares it with common forced argmax on the same frozen,
real reduced-model logits: both match the host-greedy winner (token 470), while
the selected path takes approximately 0.482 ms versus 2.537 ms. This is a focused
sampling-only comparison, not broad tie coverage. Full-model AIME accuracy,
repeated greedy requests, explicit host compatibility, and penalty-adjusted
host winners provide additional correctness evidence. The reduced terminal
profile measures sampling/feedback at about 0.605 ms including trace gaps,
roughly 6.8% of full-model token-out latency; sampling does not dominate.

Seeded continuation, slot permutation/reset, mixed prefill, penalty transitions
and inactive preservation are covered by `sampling_transition2.json`,
`prefill_generation.json`, `sampling_state_probe.json`,
`serving_decode_probe_final2.json` and the runtime fallback tripwire. Neither
implementation was modified outside the model directory.
