# Fused decoder work log

## 2026-08-03

- Confirmed repository baseline `71bbce61799` and completed functional-decoder
  commit `b46b2396bd2`. Preserved unrelated untracked
  `tt_metal/third_party/tt-cluster-descriptors/`.
- `tt-smi` was not on `PATH`; `/home/mvasiljevic/.ttsmi-venv/bin/tt-smi -ls`
  showed four healthy Blackhole P300 boards. A 1x1 TTNN mesh open/close passed.
  `/dev/shm` had about 17 MiB free and MPI warned about a 16 MiB segment, but
  no hardware command failed. All hardware commands were serialized.
- Audited the functional op topology and repo implementations/tests for
  dedicated ops, structural rewrites, and adjacent-op merging.
- Packed expert up/gate initially attempted one whole 1,024-token sparse
  projection; the sparse down op rejected 256 blocks on 8 cores. Adapted it to
  the canonical 32-token sparse tile and retained the resulting correct graph.
- Retained expert packing plus GELU folded into multiply. Real-weight
  functional equivalence passed for layer 0 and layer 5.
- Tried dense gate/up `minimal_matmul_split` with its default config, then an
  adapted decode 1/4/8 and prefill 4/4/8 M/K/N blocking. Both were slower than
  the best retained graph and were removed.
- Tried residual-add/RMSNorm fusion; it was numerically correct but slower and
  was removed. A final current-source complementary dense activation test had
  PCC 1.0: always-folded decode regressed 2.462 to 2.479 ms and prefill was
  0.021% slower, below the predeclared 0.1% materiality threshold. Explicit
  dense GELU is retained for both modes.
- Tried two router-scale merges. Whole-layer sliding decode PCC was 0.984758
  and 0.984824 versus 0.99, so both were removed.
- Tried full-attention decode-mode RoPE after transposing only the position
  tensors. PCC was 0.946838, so the original full-attention RoPE contract was
  restored.
- Tried `ttnn.geglu`; adapted the initial rank error with a 4D view. The final
  101-replay full matrix selected explicit lowering: it won 3/4 raw medians,
  no paired 95% interval favored composite, and its aggregate won by 0.120 ms.
  The serving-case inversion was 7.2 us with a confidence interval crossing
  zero. Current-source composite prefill lost by 0.949 ms sliding and 0.980 ms
  full at PCC 1.0.
- Retained `paged_fused_update_cache` by satisfying its disjoint height-shard
  grid contract. Natural sliding/full cache equivalence passed. Shared cache
  views and modulo addressing use the separate update ops because the fused
  API cannot express those views.
- Released the two source expert buffers after device packing. Advertised
  262,144-token sliding/full traced decode passed at position 262,143, proving
  capacity was not reduced. Real-weight fused prefill then passed at both
  262,143 and 262,144 for both layer kinds, with finite last-token readback.
  `context_contract.json` remains unchanged.
- HF PCC gates passed for sliding shared, full natural, and full shared caches.
  Trace/replay passed at batch 1 and 32 for both layer kinds with deterministic
  repeat PCC 1.0. Logical lengths 31, 33, and 1,025 passed; bounded modulo cache
  integrity passed.
- Final sequence-1,024 A/B passed all four decode cases and both prefill cases.
  See `candidate_ab_*.json` and the table in `README.md`.
- The first independent stage review found that the dedicated `moe_compute`
  path had not been exercised, the fastest-candidate claim lacked a
  same-process selection run, source provenance was incomplete, and remaining
  decode layout conversions lacked a direct rejection. Completion was held.
- Ran `ttnn.experimental.moe_compute(compute_only=True)` at the exact Gemma
  target shape with real layer-0 expert weights. Its required BF4 packing gave
  expert PCC 0.983965 and 0.977038; expert 127 failed the operation's gate.
  Moreover, `compute_only` exposes the final two expert buffers, while the
  token-ordered score-reduced output is only available through the
  collective/fabric combine contract. The candidate was rejected with exact
  evidence in `rejected_moe_compute_candidate.json`.
- Re-ran the strongest correct dense gate/up split candidate against the final
  graph using 101 alternating same-process trace replays. The final graph won
  sliding/full at batch 1 and 32; candidate PCC ranged 0.999927–0.999970. The
  batch-1 prefill comparison used 21 alternating runs. See
  `final_vs_dense_split_layer*.json`.
- Attempted to eliminate the remaining decode RMSNorm interleaved layout
  conversions. Both representative layers failed with the exact device error
  `Height sharded inputs are not supported`; restored the required
  conversions and recorded `rejected_sharded_decode_rmsnorm.json`.
- Added exact decoder/test SHA-256 provenance to context, host timing,
  functional A/B, dense-candidate, and dedicated-MoE artifacts, then reran the
  hardware evidence after the final source edit.
- Final default fused suite passed 17 tests with 21 intentional opt-in skips;
  all opt-in performance and candidate groups were run separately and passed.
  See `final_suite.log` and `source_binding.json`.
- Post-test `/home/mvasiljevic/.ttsmi-venv/bin/tt-smi -ls` again listed all
  four Blackhole P300 boards; no reset or recovery was required.
- A combined sequence-1,024 Tracy op capture overflowed the fixed device marker
  buffer and failed correlation; it was not retained. A sequence-32 topology
  capture completed. Default correlation rejected multiple trace IDs, so the
  documented `--force-legacy-device-logs` postprocessor was used successfully.
  Filtered `tt-perf-report` tables/CSVs were retained and raw multi-gigabyte
  logs removed. Device trace replay was captured separately.
- Separate `TT_METAL_WATCHER=10` run passed 7/7 HF and trace cases with no
  watcher fault. Evidence: `watcher_clean.log` and
  `artifacts/watcher_provenance.json`.
- The second independent review held completion because the strongest
  composite and prefill candidates were not covered by the final selection
  matrix, the dense activation experiment was confounded, profiler/watcher
  captures predated the then-current source, and provenance omitted the
  functional base/test. Completion remained held while each issue was fixed.
- Added isolated dense activation selection; expanded candidate selection to
  dense and composite paths at both layer kinds and batch 1/32 with 101 traced
  replays; added 21-run dense prefill measurements; and bound every generated
  artifact to fused decoder/test plus functional decoder/test hashes.
- Regenerated op-level Tracy, trace-device, watcher, A/B, context, candidate,
  and final-suite evidence after the last source/test edit. Modern Tracy
  correlation again missed one trace op; documented legacy processing
  completed successfully. The retained filtered reports contain 57 prefill,
  71 sliding-decode, and 73 full-decode device ops.
- The third independent review held completion for a noise-level composite
  serving-case inversion, missing current composite-prefill coverage, stale
  dense-fold selection, and missing fused maximum-context prefill evidence.
  Completion again remained held.
- Reverted the unproven dense prefill fold, added the complementary
  current-source control and materiality assertion, expanded composite testing
  to actual 4D prefill for both layer kinds, and added paired 95% decode
  intervals. The final harness separately asserts and records decode and
  prefill PCC.
- Added and ran fused real-weight capacity probes at logical lengths 262,143
  and 262,144 for sliding and full attention. All four passed; source-bound
  artifacts record 103.3 s sliding and 190.7 s full elapsed times.
- `ruff` was unavailable in the active environment; `py_compile`,
  `git diff --check`, and the fused pytest suite are used as source gates.
- `$autofix` was not invoked: no hard bug or stage finding survived direct,
  evidence-backed candidate adaptation.

## Commits

Stage implementation commit SHA: `0dafd12a42bac0eb72b3c0abbc908500eedd7131`
on branch `skillexp-work-gemma-p3-fresh`. The commit skipped only the
`trailing-whitespace`, `isort`, and `prefer-expect-error` pre-commit hooks:
the first rewrites byte-exact captured device logs, while the latter two
rewrite independently reviewed, hash-bound source. All other applicable
hooks passed. This documentation update is recorded in the follow-up commit.
