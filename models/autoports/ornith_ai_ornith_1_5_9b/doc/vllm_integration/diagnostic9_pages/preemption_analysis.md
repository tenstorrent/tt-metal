# Forced-preemption comparison after Entry3

Plugin `2a1504554aa7babb1d426dfbca8c7db8c0a795ef` repairs retained block-row replacement. **All three strict uninterrupted/resumed token comparisons remain failed.** This audit does not waive them or establish exact equivalence. Both runs use reduced layers0/3, batch3 and context262144.

| Mode | Diagnostic8 first difference | Diagnostic9 first difference |
| --- | --- | --- |
| Device greedy | index62: 37 → 387 | index49: 53 → 498 |
| Device seeded | index22: 264 → 51 | index22: 264 → 51 |
| Host greedy penalties | index62: 37 → 387 | index49: 53 → 498 |

Indices are zero-based; arrows mean baseline → forced. All requests produce100 tokens with `length` termination; all three resets succeed. All baseline token sequences are unchanged between runs. Source: each run's `preemption.json`; machine-readable extraction is [preemption_analysis.json](preemption_analysis.json).

The new [server.log](server.log) shows actual model-facing page replacement: greedy lines328/357 and host749/779 change `[8,7,6,5,4,0,0,0]` to `[4,5,6,7,8,0,0,0]`; seeded539/568 changes the reverse. Every resumed prefill has start0, end151, original prompt129, slot0; its following decode reloads inputs. The old log lacks page IDs, so its actual physical rows cannot be reconstructed from that log.

Forced host raw selected-token logprobs are identical across old/new versions through index31 and first change at index32: -7.03425407409668 → -7.051009178161621. This is absolute decode position160, the first newly needed logical page5 after replay, consistent with the repaired append-alias mechanism. The old/new forced seeded tokens also first change at index32 (77 → 28). This timing is supporting evidence, not a complete scheduler-to-device page ledger: decode page rows were not logged.

The baseline/forced host score difference starts earlier at index22 and is unchanged by Entry3: -6.95357608795166 versus -7.005373001098633. The independent [canonical control](../replay_numerics_served_context.json) reproduces the seeded first difference (output23:264 versus51), using the same22-token prefix and serving context. All four penalty tables and presampling seed275437 match; postdraw seed275438 matches the expected count. Decode/replay raw logits have PCC0.9999462981210137, max absolute difference0.25 and mean absolute difference0.03308195094049898. Host/device prefill raw logits are bitwise identical. This isolates the first seeded discrepancy to canonical recomputation differences independently of plugin page ownership.

The original canonical control stops at output23, has a larger physical pool than serving, and does not compare recurrent, convolution or live KV state. The later [greedy follow-up](../replay_greedy_follow50.json) independently reproduces both served greedy branches through50 outputs and their first divergence at index49 (53 versus498). The preceding49 tokens and pre-sampling state match, including seed275464; raw logits at the differing decision have PCC0.9999743544 and maximum absolute difference0.0625.

**All three strict trajectory comparisons remain failed.** The [aligned consumed-prefix151 comparison](../replay_model_state151_interpretation.md) now passes structural checks: host/device replay states and sampled inactive regions are exact, and active KV differences begin at decode row129 after129 exact prompt rows. This demonstrates a canonical computation-route difference, without establishing harmless rounding or an acceptable numerical tolerance. Full32-layer/B32 acceptance is separate from this reduced audit and remains in progress.
