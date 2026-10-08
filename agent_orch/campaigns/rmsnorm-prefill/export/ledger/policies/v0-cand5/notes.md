# v0-cand5: cand1 + plateau and repeated-hard-failure closes

**Change:** on top of cand1, close a branch after `2 + round(beta)` valid
results in a row within noise of their parent (plateau), or after two
consecutive hard failures (hang, runtime_error, accuracy_fail) of the same
class. Build/compile errors never close a branch.

**Trace evidence:** none in r01. There were no hard failures, and the only within-noise
result (b04-a04, +0.47%) is a branch's last attempt anyway. These are
generalization guards for rounds with failures or plateaus.

**Replay (r01):** V = 1.1882, identical to cand1 (the rules never fire).

**Verdict:** a tie, with more rules. Kept cand1. Reconsider once a recorded round
contains plateaus or repeated hard failures, because then replay can score these
rules.
