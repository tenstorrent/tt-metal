# Request identity correction

Preserve RESULT and initial payload provenance as historical evidence, but replace their per-request prompt-length assignment: request0/actual dataset7/logged152 has481tokens; request1/actual dataset20/logged189 has258tokens. Pinned842da tokenizer applied to each saved native message list with thinking enabled produces exact full-ID equality with that same request’s SSE prompt IDs. Evidence: bringup/artifacts/gpqa-loop-onset-native-capture/prompt-identity-check.json.

The old preparation assigned unordered physical prefill lengths to request order without IDs or tokenizer confirmation. This was an analysis/provenance error, not evidence of runtime request crossing. Corrected prior-chain attribution:481-token chain has period2 onset4103 (two-token-vocabulary onset4091);258-token chain onset3178. Both19789-sample chains, pair timing/cost and safe diagnostic stop findings remain unchanged. Original hidden outputs still unknown.

Host-only focused confirmation compares exact tokenizer IDs and own-request native SSE IDs for both requests, not just lengths or physical-row order. No device rerun/source change. Before building any causal prefix use captured native request IDs and this mapping, never splice across request identities.
