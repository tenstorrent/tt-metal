# Optimized multichip decoder stage review

Verdict: `clean-pass`

The final independent `$stage-review` found no required work, other
concerns, or hard-check gaps.

The reviewer independently inspected the full user and skill contracts,
`tt/multichip_decoder.py`, `tests/test_multichip_decoder.py`,
`doc/context_contract.json`, the stage README/work log/manifest, all final
performance and correctness gates, candidate evidence, raw profiler CSVs,
corrected tables, activity histograms, and provenance hashes. It used only
read-only source/artifact checks and no TT hardware.

Verified resolutions:

- EP4 profiler accounting uses exact rank-local activity. Decode tables use
  critical-rank K=3/2 at S=17 and K=2/2 at S=128; prefill sparse modeling is
  explicitly manual because one scalar cannot represent token/rank
  variation. All eight authoritative table hashes and both raw CSV hashes
  match provenance, and no obsolete `*active4*` artifact remains.
- `test_fused_o_rs_deferred_through_post_attention_moe` measures fused
  local O+RS through sharded residual addition, distributed
  post-attention RMSNorm, row-sharded router, delayed gather at the selected
  sparse gate/up boundary, all three gate-selected sparse matmuls, expert
  reduction, and final layer output. Both layers pass exact top-4 and final
  PCC 0.999450/0.999677; the coherent candidate is slower at
  0.659886/0.737228 ms versus 0.448808/0.526294 ms.
- The final `after_review2` default is the reproduced 1x4 path with EP4
  gate-selected active experts, two persistent minimal async all-reduces,
  and both fused alternatives disabled.
- The complete suite, real-weight length matrix, worker/Tensix watcher,
  advertised-context endpoint, real-weight fallback-throw run, and final
  device health all pass.
- The runtime retains exactly three sparse active-expert matmuls, no dense
  all-expert fallback, arbitrary logical sequence-length support, and no
  collective at the replicated BF16 inter-layer boundary.

Residual risk is limited to the documented fixed batch-1 1x4 Blackhole
P300c scope, scalar-tool limitations for prefill EP modeling, and the
physical ACTIVE_ETH watcher instrumentation-capacity limit. Full-model and
vLLM work remain intentionally out of scope.
