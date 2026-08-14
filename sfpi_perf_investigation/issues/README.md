# Issue drafts, one per ask

[../ISSUE.md](../ISSUE.md) is the umbrella writeup: it makes the whole argument in one
place and is the right thing to read first. These seven files are the same asks split into
individually fileable issues, so each can be triaged, assigned and closed on its own.

Each draft is self-contained — kind, worth, compiler version, current codegen with
assembly, the rewrite that fixes it, measured impact, and a repro — so it can be filed
without the reader needing the umbrella first.

| # | draft | kind | worth |
|---|---|---|---|
| 1 | [Commute integer compares to a CC polarity `SFPIADD` can fuse](01-commute-integer-compares.md) | missed opt | 3 instr per compare against a constant |
| 2 | [Predicated result emits a redundant liveness copy](02-predicated-store-lowering.md) | missed opt | 1 instr per predicated result |
| 3 | [`dst_reg++` emits a separate `TTINCRWC`](03-fold-dst-reg-incr-into-store.md) | missed opt | 1 instr per DEST row |
| 4 | [`&&` abandons CC chaining when a term needs a helper](04-and-chain-falls-off-a-cliff.md) | missed opt | 6 instr per predicate |
| 5 | [No total-order float compare reachable from `v_if`](05-total-order-float-compare.md) | **missing feature** | 1–2 instr, plus 4 more in workarounds it deletes |
| 6 | [`vInt`/`vUInt` compares are wrong over the full range](06-integer-compare-overflow.md) | **correctness** | unblocks `calculate_binary_comp_uint`; removes a 6-instr fold |
| 7 | [`vSMag`/`SM32` compares lower to a two's-complement subtract](07-vsmag-compare-lowering-bug.md) | **likely bug** | correctness; also delivers most of ask 5 |

## Suggested filing order

Asks 1–4 are the bulk of the regression and need no new API or ISA surface — they are all
"sfpi already emits this, just not from this source form." Filing them first gives the
largest win for the least design discussion.

Ask 7 is worth filing early despite being last in the table: if `vSMag` compares are
simply mis-lowered, fixing that bug delivers most of ask 5 without any new API.

Ask 6 is the only one that is a live correctness bug in shipped kernels rather than a
performance problem, so it may warrant a different priority than its position suggests.

## Consistency

Every C++ snippet in these drafts is quoted verbatim from a probe in the parent directory
and tagged with a `// verified: <file> :: <function>` marker. Run

```sh
../verify_quotes.py    # checks the quotes still match
../run.sh              # recomputes every instruction count quoted here
```

to confirm nothing has drifted.
