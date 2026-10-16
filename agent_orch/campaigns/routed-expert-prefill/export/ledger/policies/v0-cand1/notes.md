# v0-cand1: collapse close + recover role (the r02 dreaming's cand2, re-scored on r01-r03)

**Change over v0 (two behaviors, both tested in the r02 dreaming):**
- **Collapse close:** close a branch when its valid head is more than `5 x (1 + beta) x noise` below
  the branch's earlier best (8% at beta 0.6).
- **Recover role:** tag a refinement `recover` when its head failed or is more than noise below its
  revert target. The why names the revert target and, when another branch leads by more than noise,
  a node to port from.

The code was rewritten into one flag-based file that cand2-cand5 share. Only `CLOSE_COLLAPSE` and
`RECOVER_ROLE` are on here.

**r03 evidence:**
- The collapse close does not fire in r03; the worst drop was b03-a02 at -4.9%.
- The recover tags matched the workers' hand recoveries in all three cases:
  - b01-a01 runtime_error → a02 repaired it;
  - b03-a02 below the root → a03 rebased onto b02-a02;
  - b03-a03 accuracy_fail → a04 rebased onto b02-a03.

**Replay (r01-r03, beta 0.6):** V 1.5229 vs v0 1.5204 (+0.0025).

| round | V | attempts |
|---|---|---|
| r01 | 1.4636 | 15 |
| r02 | 1.5347 | 14 |
| r03 | 1.5703 | 16 |

The result is the same at every beta.

**Online caveat:** `run_campaign.sh` passes only `node,parent` to workers (line 70). `role` and
`why` land in `decisions.jsonl` but never reach the worker prompt. The recover hint has no online
effect until the driver forwards it.

**Verdict:** a tie with v0, below one attempt's cost. Kept as the base for cand2-cand5.
