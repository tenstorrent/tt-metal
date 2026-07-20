# Full-model rejection ledger

| Candidate | Decision | Evidence / reason |
|---|---|---|
| Sampling1D exact split greedy | Selected | Exact semantic match, compact candidate gather, 0.8035 ms/call, direct device feedback |
| TTSampling exact greedy | Rejected for optimized path | Exact but full-vocabulary gather, 1.0091 ms/call, broader mutable state and trace constraints |
| Sampling1D force argmax | Rejected for optimized path | Exact comparison oracle but full-logit gather, 1.0054 ms/call |
| Custom sampler | Rejected as unnecessary | A common sampler satisfies exact greedy, trace, shape, and performance contracts |
| Host argmax / Python feedback | Rejected | Violates traced split-sampling contract; retained only as an explicit compatibility mode |
| Full logits readback per token | Rejected | Unneeded data movement and violates optimized token-out boundary |
| Single-chip or replicated full model | Rejected | Violates inherited TP4 optimized multichip contract |
| Host-side embedding, layer, norm, LM head, or cache | Rejected | Full autoregressive path is device resident |
| DRAM inter-layer residual | Rejected | Native L1 width-sharded BF16 residual is preserved through all 40 layers |
| BFP8 CCL/residual | Rejected | Inherited selected policy is BF16 CCL/residual; no precision regression accepted |
| BF16 KV cache | Rejected | Inherited paged-cache policy is BFP8_B and is required for batch-32 full-context capacity |
| Reduced advertised context | Rejected | 32768 is physically feasible; full-stack last-page execution and batch-32 allocation pass |
| Invented chat template | Rejected | Exact base tokenizer has no `chat_template`; native completion format is the honest reference |
| Per-token page-table/position rebuild | Rejected | Persistent traced inputs update on device; host copy deltas are zero at steady state |
| Combined model+sampling trace | Rejected | Separate traces preserve the split-sampling contract and make sampler cost explicit |
| Watcher and Tracy together | Rejected | Device-use contract requires separate diagnostic runs; both passed separately |

The optimized multichip decoder's existing rejection ledger remains authoritative for decoder-internal topology, core-grid, CCL, and program-config alternatives. This ledger records only full-model decisions and does not reopen those rejected policies.
