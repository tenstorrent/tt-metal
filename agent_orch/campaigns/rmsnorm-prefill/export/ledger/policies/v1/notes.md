# v1: parallel refine + gap-to-leader pruning (promoted from v0-cand1)

What changed vs v0: one rule. An open branch is closed when its anchor (best
valid score on it) trails the round's best by more than
`noise_pct * (2 + 8*beta)` (6.8% at beta 0.6). Anchors, not heads, are compared,
so a regression followed by a repair keeps its slot. Branches with no valid
result are never gap-closed. Plan is unchanged (campaign defaults W=4, R=4).

Evidence and replay: see `../v0-cand1/notes.md`. r01 V = 1.1882 vs v0's 1.1732.
The 6 attempts v0 spent refining b02/b03 (9.6% and 11% behind the leader
after step 1) are saved, and the round best (b04-a04, 1.2132) is still reached.

Candidates considered (r01, beta 0.6):
| id | change | V |
|---|---|---|
| v0 | baseline | 1.1732 |
| v0-cand1 | gap-to-leader pruning | 1.1882 |
| v0-cand2 | root-relative no-signal pruning | 1.1882 (tie) |
| v0-cand3 | cand1 + patience (min 2 attempts) | 1.1732 |
| v0-cand4 | cand1 + final-attempt focus | 1.1907 (+0.0025 < 1 attempt: tie, overfit) |
| v0-cand5 | cand1 + plateau / hard-failure closes | 1.1882 (rules never fire) |

Tool note: `replay.py` writes `trace.update(res)` after building the per-step
list, so `traces.jsonl` holds the step count (int) under `"steps"` instead of
the step list. The per-step decisions above were read from a patched in-memory
copy of `simulate`.
