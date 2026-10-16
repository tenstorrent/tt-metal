# routed-expert-prefill: full campaign export

Every attempt of the Dream-RSI campaign on `ttnn.experimental.deepseek_prefill.unified_routed_expert_moe`, exported from the git refs
`dream/routed-expert-prefill/*` and `$DREAM_HOME/routed-expert-prefill` on bh-qbge-12.

Best valid attempt: `r08-b04-a01`, score 1.7812. See `FINAL.md`.

| Path | Contents |
|---|---|
| `FINAL.md` | final report: result, lineage of the best node, dead ends, open leads, policy evolution |
| `index.json`, `index.csv` | one row per committed attempt (73): round, branch, parent, policy, mechanism, validity (after overrides), score, per-shape µs / speedup / PCC / max_abs; `lost` lists attempts that never committed |
| `attempts/<node_id>/` | proposal, context, reflection, node.json, eval/ (score.json, summary.md, ops.csv), `patch.diff` (vs parent), `cumulative.diff` (vs the campaign root) |
| `ledger/` | the ledger branch: policies v0..vN with every dreaming candidate and its replay results, per-round manifest / decisions / summary, aborted rounds, baseline |
| `tree.html`, `history.md` | the visual tree and the worker-facing history, as of the end of the campaign |
| `transcripts.tar.gz` | full stream-json transcripts of every worker and policy-development session |
| `dream_routed-expert-prefill.bundle` | git bundle with every attempt commit, node tag, branch and the ledger |

`valid`/`score` in the index apply ledger overrides (e.g. attempts invalidated by a later campaign rule);
`eval_valid`/`eval_score` are what the eval tool recorded at the time.

## Restore the git structure

```bash
git fetch agent_orch/campaigns/routed-expert-prefill/export/dream_routed-expert-prefill.bundle 'refs/*:refs/*'
git log --oneline dream/routed-expert-prefill/root..dream/routed-expert-prefill/n/r08-b04-a01
git checkout dream/routed-expert-prefill/n/r08-b04-a01   # build + run the campaign test to reproduce
```

Raw profiler output (tracy, device logs; ~240 MB per attempt) is not included; it stays under
`$DREAM_HOME/routed-expert-prefill/reports/` on bh-qbge-12.
