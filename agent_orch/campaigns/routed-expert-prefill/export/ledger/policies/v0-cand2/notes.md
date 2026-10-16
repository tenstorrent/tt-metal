# v0-cand2: cand1 + frontier prune after a flat step

**Change (one behavior over cand1):** after a step whose running best gained no more than one noise
band, close every branch whose head is valid, has at least two attempts, and is more than
`(0.5 + beta) x noise` below the round's best revealed score (1.1% at beta 0.6). Failed heads are
exempt.

**Motivation (all three rounds):** workers on branches that fall behind rebase onto the round's
best by hand:
- r01: b02-a03, b03-a03 and b04-a03;
- r02: b02-a04 and b01-a04;
- r03: b03-a03, b04-a03 and b03-a04.

A branch behind the frontier is therefore a spare worker slot, not a lineage, and the next round
re-roots on the best node anyway. The flat gate was meant to keep slots open while the round was
still gaining.

**Replay (r01-r03, beta 0.6):** V 1.5273 (+0.0044 vs cand1, +0.0069 vs v0).

| round | V | attempts | effect |
|---|---|---|---|
| r01 | 1.4636 | 15 | step 2 was not flat, so no frontier close; same as cand1 |
| r02 | 1.5355 | 11 | steps 2 and 3 are flat; b01 closed at step 3, b02 at step 4 |
| r03 | 1.5828 | 11 | b03 and b04 closed after step 2, b01 after step 3; best 1.6103 kept |

- **r02:** closing b02 at step 4 loses its rebase to 1.5697. The best becomes b03-a04 1.5630, -0.43%,
  inside noise.
- **Beta sweep:** 1.5290 / 1.5281 / 1.5273 / 1.5248 / 1.5248 for beta 0.2-1.0. Higher beta keeps
  more slots and costs V, every value is above cand1, and attempts rise with beta.

**Side check (not saved):** using the branch's best valid node instead of its head gives V 1.5270.
It keeps r02's 1.5697 only because b02's anchor sat 1.0% below the best against a 1.1% band, which
is too thin to rely on.

**Verdict:** an improvement over cand1 (+0.0044, just under one attempt's cost), but cand3 shows
that the flat gate is the part that costs V.
