# v1: frontier prune + collapse close + recover role

Promoted from `v0-cand3` after the r03 dreaming. The code is the same; only the docstring's first
line changed.

**What it does differently from v0:**
- **Frontier prune:** from a branch's second attempt on, close the branch if its head is valid and
  more than `(0.5 + beta) x noise` below the round's best revealed score (1.1% at beta 0.6). Failed
  heads stay open because failures are repairable.
- **Collapse close:** close a branch whose valid head fell more than `5 x (1 + beta) x noise` below
  the branch's earlier best.
- **Recover role:** tag refinements of failed or regressed heads `recover`, with a revert target and
  a port hint. These tags only reach `decisions.jsonl`, because `run_campaign.sh` forwards only
  `node,parent` to workers.
- The plan is unchanged: W=4, R=4, the campaign defaults.

**Replay (r01-r03, beta 0.6):** V 1.5306 vs v0 1.5204, +0.0102, about two attempts' cost.

| round | V | attempts |
|---|---|---|
| r01 | 1.4736 | 11 |
| r02 | 1.5355 | 11 |
| r03 | 1.5828 | 11 |

- v1 beats v0 on each of the three rounds.
- The beta sweep runs from 1.5315 to 1.5281 (beta 0.2 to 1.0), and every beta beats v0.
- In r02, v1 gives up the round best 1.5697 for 1.5630, a 0.43% gap that is inside noise.

**Why:** in every round, branches that fall behind the leader are turned into rebases onto the
leader by their workers. Keeping them open buys a parallel draw of the leader's code, not a new
lineage, and the top nodes now sit within noise of each other. See `../v0-cand2..5/notes.md` for
the ablations:
- the flat gate costs V;
- closing failed heads loses r01's best;
- a stall close or a plateau stop is a tie that is fragile in beta.

**What to watch online:**
- How often step 4 runs with only one or two workers.
- Whether a branch closed at 1-2% behind the leader would have overtaken it. r02's b02 is the one
  recorded case, and its gain was within noise.
