# v2-cand5: cand1 + late steps keep a valid head that is still improving (overfit check)

**Change (one behavior over cand1):** in late steps, cand1 ends every valid, above-root line.
cand5 also keeps a valid head that beat its parent by more than the frontier band (`0.5 + beta` x
noise, 1.1% at beta 0.6). This is the "premature stop" guard from POLICY_DEV §3.

**Why:** cand1's gain comes from one observation: r01-b01 after a +2.4% step (1.4376 → 1.4716) then
collapsing to 1.0363. With one deep round, it can't be told apart from luck. cand5 measures what
cand1's cut costs when the guard is on.

**Replay (r01-r07, beta 0.6):** V 1.6342, the same as v2 at every beta. r01 goes back to v2's path:
step 3 refines b01-a02 (still improving) and b04-a02 (repair), and b01 is collapse-closed at step 4.

**Verdict:** this confirms that cand1's +0.0004 is entirely the cut of one still-improving line in
one round. Neither direction is supported by more than one branch. No change.

## Phase conclusion (r07 dreaming)

- **No v3: v2 stays.** The best candidates (cand1-cand4) score 1.6346 vs v2's 1.6342. That is
  +0.0004, under one attempt's cost (0.005), so they are ties and the simpler policy is kept.
- **r07 adds no signal that tells candidates apart.** It is another one-step round at the replay
  ceiling (best + 0.02 = 1.7587), like r02-r06. Every candidate replays it identically.
- **Replay ceiling:**
  - Every round after the first is at the ceiling (4 attempts, 1 step). A wider round would be out
    of support.
  - Round 1's non-overfit ceiling is cand1's 10 attempts (+0.0004 on the mean).
  - The 7-attempt path (V 1.4836) needs below-root pruning after one attempt, which r01 refutes.
- **If the default beta goes above 0.7,** promote cand4. v2's high-beta second step loses about
  0.008 V for +0.001 best.
- **r07's open problems are outside the policy:**
  - **Seeded port slot:** re-rooting on b02-a01 drops b01-a01's +1.6%.
  - **Mechanism dedupe:** the planner rebalance is the convergent next #1 for r08.
  - **Per-attempt eval counts:** in-attempt failures are invisible in the ledger.
  - Both slot ideas would need a batch item's `role`/`why` to reach the worker.
    `run_campaign.sh` still forwards only `node,parent` (line 70), so a policy-side hint would have
    no effect online.
