# Report requirements

Canonical spec: `test_report.md` (in this same references/ folder — read it; it also carries the "Agreed decisions" section that
resolves the ambiguous points). Condensed here so the skill is self-contained.

## Scenarios × modes
- **Scenarios:** `warm` (1 chunk over a filled cache), `cold` (full prefill 0→cache, cache grows per
  chunk), `long` (1 chunk over a 0.5M-class cache). Toggle controls the tables AND the graph.
- **Modes:** `sparse` (v3.2 DSA: indexer + top-k `sparse_sdpa`) and `dense` (v3.1 ring-MLA baseline).
- Durations MUST come from Tracy reports; the graph MUST be verified against actual code (not a proxy).

## The 11 requirements
1. Op list: duration (absolute + relative), call count; sortable by execution order or duration.
2. MLA dataflow graph: nodes = ops, edges = tensors; nodes label internal weights; edges label tensors;
   all tensors label dims, dtype, mem layout (TILE default, ROW_MAJOR flagged), distribution (SP|TP|repl);
   nodes shaded red (longest=red, shortest=white); nodes labelled with duration (abs + rel to full trace).
3. A toggle selecting which scenario is shown in (1) and (2).
4. Dense is the baseline — clearly report how much better/worse sparse is vs dense.
5. Cold: also track top-N (default 10) longest ops across iterations + total duration per iteration.
6. Graph MUST be verified against the actual code, not a proxy/assumption.
7. Reported durations MUST be based on Tracy reports.
8. Appendix: (8.1) per node, link `file:line` + a critical code snippet; (8.2) full Tracy reports.
9. Self-sufficient — present evidence, copy the reports in.
10. Commit number, branch name, and bullet-point key changes / what the experiment is about.
11. Hardware details — box (LoudBox/QuietBox/Galaxy), Blackhole/Wormhole, card name, mesh.

## Decisions settled while building (this session)
- **Graph nodes = semantic blocks authored from code**, each node's time = its constituent ops' real
  per-call Tracy time (not raw op-code rows).
- **Two-level graph:** semantic ↔ ops (global toggle + per-node expand into the intra-block op dataflow).
- **Two graphs by mode** (sparse and dense are structurally different forwards), swapped by a mode toggle.
- **Comparison headline = total critical-path ms per scenario, sparse vs dense.**
- **Mode colours report-wide:** teal = sparse, amber = dense.
- **Measured on the box you have, framed against the Galaxy target** (the test is a per-chip Galaxy proxy).
- Data-integrity caveats are part of the deliverable (clobbered/recovered runs, any unmeasured combo).
