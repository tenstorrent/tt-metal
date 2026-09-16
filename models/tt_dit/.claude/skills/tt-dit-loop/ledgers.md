# Ledgers

Four append-only files under `ledgers/`. **Grep them; never read one whole.**
They answer different questions, which is why they are not one file.

| Ledger | Row per | Answers |
|---|---|---|
| `attempts.md` | Every round, win or loss | "What has been tried?" |
| `optimizations.md` | Correct **and** measurably better only | "What actually shipped?" — short enough to read whole |
| `source-ideas.md` | Each source consulted, with provenance and verdict | "Has this already been investigated?" |
| `amendments.md` | A measurement contradicting the plan; retractions | "Which of our beliefs were wrong?" |

`source-ideas.md` is the highest-value one and the least obvious. Without it,
successive rounds re-read the same PR, the same kernel, the same upstream
implementation, and re-derive the same rejection. The rejection *is* the result.

## Row formats

Fixed, so `grep` and `awk` suffice.

**`attempts.md`**
```
| r7 | 2026-08-05 | a1b2c3d | vae-decode | fold LayerScale into out proj | 4612ms −1.4% | pcc 0.99942 | kept |
```

**`optimizations.md`** — only rows that passed the gate *and* beat `best`:
```
| r5 | a1b2c3d | bf16 compute dtype to match reference | 6410 → 4680 ms (−27.0%) | pcc 0.9994 | artifacts/round-5/ |
```

**`source-ideas.md`** — including negative results:
```
| r7 | ttnn.experimental.rotary_embedding_llama_fused_qk | rejected | device op asserts seq_len==1, decode-only | ttnn/cpp/.../rotary_embedding_llama_fused_qk_device_operation.cpp:54 |
| r7 | gh pr 41822 "wan rope fusion"  | not applicable | different head layout | — |
| r8 | searched: fused groupnorm across ttnn/ | not found | only rmsnorm/layernorm have distributed variants | — |
```

**`amendments.md`** — the existing numbered protocol, format in
`../shared/journal-protocol.md`. Numbering continues across rollovers and across
compactions of any predecessor journal.

## Rollover

Past ~500 lines, roll to `attempts-02.md`, `attempts-03.md`, … and update the
ledger index line in `CAMPAIGN.md`.

**Rolling over is not compaction.** Nothing is summarised, nothing is deleted.
The previous file is closed and a new one opened. This is the rule the 300KB
journal violated when it was compacted from 4972 lines to 14KB — the content was
recoverable only because it happened to be committed.

## What goes where

| Situation | Ledger |
|---|---|
| Tried a change, it regressed | `attempts.md`, verdict `forensic` |
| Tried a change, it won and the gate passed | `attempts.md` **and** `optimizations.md` |
| Read a PR / kernel / upstream impl and ruled it out | `source-ideas.md` with the reason |
| Searched for something and it does not exist | `source-ideas.md` — negative results are results |
| A measurement contradicted the plan | `amendments.md` |
| An earlier amendment turned out wrong | `amendments.md` as a **retraction**, original untouched |
| A pitfall that is general to tt_dit | Graduate to `../shared/known-issues.md`, not a ledger |

## Discipline

Write the rows **before** the next round starts, not batched at the end. A round
that commits code without a lineage row and an attempt row is unrecoverable —
the next agent sees a commit it cannot explain.

Anything reported to the user in chat also lands in a ledger. Chat is ephemeral.
