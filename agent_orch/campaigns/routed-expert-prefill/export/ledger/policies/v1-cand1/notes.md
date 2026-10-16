# v1-cand1: v1 + leader guard on the frontier prune

**Change (one behavior over v1):** the frontier prune never closes the branch that holds the
round's best node. The branch's next attempt is tagged `recover`, with that node as the revert
target.

**Evidence (r04 summary, policy observations):** r04 closed b02 after step 2. Its head b02-a02
(1.6093) was 1.12% below the round best 1.6275, and that best was b02's own a01. No branch was
left refining the round's best node. b02-a02's reflection had named a one-file repair.

**Replay (r01-r04, beta 0.6):** V 1.5511, identical to v1 on every round and at every beta.

| round | V | attempts | why there is no change |
|---|---|---|---|
| r01 | 1.4736 | 11 | the leader is always a branch head when the prune runs; b01 is closed by the collapse rule |
| r02 | 1.5355 | 11 | the step-3 leader b04 is closed by the collapse rule (a02 fell 11%) |
| r03 | 1.5828 | 11 | the leader is a head at every step |
| r04 | 1.6125 | 9 | b02 stays open, but r04 ran v1, so b02-a03 was never recorded. The extra item reveals nothing ("out of support") |

**Verdict:** a tie. Replay can't test it: the only round where it fires was recorded under the
rule it removes. It's still a sensible online guard if deep rounds return. Not kept, because ties
go to the simpler policy.
