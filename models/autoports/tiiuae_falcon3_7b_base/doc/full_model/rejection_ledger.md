# Full-model rejection ledger

The optimized multichip decoder rejection ledger remains authoritative for
projection geometry, dtype/fidelity, activation, CCL, KV-cache, and residual
layout choices. This stage adds the following decisions:

| Alternative | Rejected because |
| --- | --- |
| Single-chip or replicated full model | Violates the required TP4 decoder, rank-local KV, and ring-collective strategy. |
| Host embedding, decoder, norm, or LM head | Breaks the device-resident autoregressive path and inter-layer layout. |
| Full-vocabulary logits gather in token-out | Hardware controls are slower: 1.0086 ms for force-argmax and 1.0089 ms for actual `TTSampling`, versus 0.8064 ms for split greedy. It also moves 131,072 logits per row instead of four packed rank candidates. |
| `SamplingGenerator` / `TTSampling` for this TP4 path | Exact greedy is semantically valid, but uses the slower full-vocabulary gather; mutable request seeds/parameters also disable that abstraction's internal trace ownership. |
| Model-local custom sampler | Unnecessary because the common `Sampling1D` abstraction fits after adding its reusable exact-greedy method. |
| Host argmax or token copy-back feedback | Semantically unnecessary; split sampling writes directly into the next model-trace token buffer. |
| Untraced sampler after traced model | Adds dispatch overhead and breaks canonical split sampling. |
| Per-token page-table/position rebuild | Persistent device state advances positions; page table copies occur only for actual changes. |
| Map only logically live KV pages | Dynamic paged SDPA reads a rounded power-of-two/eight-page window before causal masking. Logical-only coverage left a masked tail entry invalid at position 64. The generator maps and validates the complete rounded window with in-range, non-aliased, cross-slot-disjoint ownership. |
| Keep address-bound traces alive while reset fills caches or persistent inputs | TTNN can allocate temporary program/output state during fills. Reset synchronizes and releases the trace pair first, scrubs the same cache/input buffers in place, and recaptures exactly once on the next optimized request. Persistent buffer addresses remain stable. |
| Rebuild traces or persistent inputs per decode token | The trace pair is stable for the request; tokens feed back device-to-device, positions advance in persistent tensors, and unchanged page tables cause zero host copies. |
| Keep traces alive across host-sampling compatibility or cache identity/shape changes | Those paths can allocate or rebind buffers; release before switching, then recapture for the next optimized request. |
| Reduce 32,768-token capability | No physical limit: complete 28-layer state plus full-context K/V uses 2.731 GB/device; maximum-context execution peaks at 2.734 GB/device with 27.349 GB/device still allocatable. |
| Invent a chat template | Exact tokenizer metadata has no chat template; native completion prompting is the only faithful base-model contract. |
| Add vLLM integration | Explicitly outside the full-model stage. |
