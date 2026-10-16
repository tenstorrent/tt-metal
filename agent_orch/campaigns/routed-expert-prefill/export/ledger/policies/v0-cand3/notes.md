# v0-cand3: cand1 + frontier prune from each branch's second attempt (no flat gate)

**Change (one behavior over cand1):** close a branch when it has at least two attempts and its head
is valid and more than `(0.5 + beta) x noise` below the round's best revealed score (1.1% at beta
0.6). Failed heads are exempt, because failures are repairable. Compared with cand2, this drops
cand2's flat-step gate, so it is the simpler rule.

**Why the gate is not needed:**
- In r02 and r03, cand2 and cand3 close the same branches.
- In r01, the gate blocked a useful close at step 2. Step 2 was not flat (+2.4%), but b02
  (1.4460, -1.7%) and b03 (0.9626) were already behind b01-a02 (1.4716).
- Closing them costs b03-a04 (1.4856, the runner-up), not the best.
- The best came through b04's failed head (b04-a02 accuracy_fail → a03 1.4622 → a04 1.5011), which
  the exemption keeps open.

**Replay (r01-r03, beta 0.6):** V 1.5306 (+0.0077 vs cand1, +0.0102 vs v0, about two attempts'
cost). It beats v0 on every round, and every round's best is kept or replaced by a node within noise.

| round | V | vs v0 | attempts | best |
|---|---|---|---|---|
| r01 | 1.4736 | +0.0125 | 11 | 1.5011 kept |
| r02 | 1.5355 | +0.0058 | 11 | 1.5630, -0.43% vs 1.5697, inside noise |
| r03 | 1.5828 | +0.0125 | 11 | 1.6103 kept |

**Beta sweep:** 1.5315 / 1.5306 / 1.5306 / 1.5281 / 1.5281, with mean attempts 10.7 / 11 / 11 / 12 / 12
for beta 0.2-1.0. Beta trades attempts for patience, and every beta beats cand1 and v0.

**Shape of a round under this rule:** the round still starts 4-wide. After step 2 it keeps only
branches within the band of the leader plus any failed head. By step 4 one or two workers remain.
Wall time is unchanged because the steps are synchronous. Device and build cost drop by about 30%.

**Risks online:**
- With 4 x 4 grids, the rule leaves step 4 nearly serial.
- If a round's late gains come from a branch that trailed by more than about 1% at step 2 or 3,
  the rule closes it. r02's b02 is such a case, and here it cost only a within-noise draw.
- A side check that judges branches by their best node instead of their head, with no flat gate,
  lost r01's best (V 1.5247). Judge the head.

**Verdict:** the best candidate that is not a tie with a simpler one. Promoted as v1.
