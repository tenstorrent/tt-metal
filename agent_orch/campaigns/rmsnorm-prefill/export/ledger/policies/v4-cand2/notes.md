# v4-cand2: v4 + close a branch whose head broke a campaign rule

**Change (one behavior):** a head with fail_class forbidden_edit closes its branch. A forbidden edit is not
a repairable bug. The branch's idea is the forbidden change, and a child built on it inherits the edit.

**Trace evidence:** r04 after the human override: b01-a02 (HiFi2) -> v4 refines it as "recover" -> b01-a03
is also HiFi2 (forbidden). The same thing happened on b02 at a03.

**Replay:** mean V 1.3447 (+0.0003 vs v4, a tie). Only r04 changes: 11 attempts, V 1.3814 (b01 closed at step 3).
Sweep: 0.2 1.3340 / 0.4 1.3451 / 0.6 1.3447 / 0.8 1.3338 / 1.0 1.3323.

**Verdict: not promoted alone.** The gain is far below one attempt's cost. verify_node now enforces
forbidden_patterns and workers are told the rules, so forbidden heads should be rare. The rule is
principled, though. See cand5 for it on top of cand1.
