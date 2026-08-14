#!/usr/bin/env python3
"""Enumerate all FSM-valid LLK call sequences for the contract fuzzer (docs/llk-formal-contract.md §14).

Alphabet: {CFG, RCFG, INIT, EXE, UNI} x operand {A, B} (EXE inherits the live operand).
Prunes to lifecycle-FSM-valid skeletons (init-before-execute, re-init-after-reconfigure),
then decorates with operand tags (A/B, canonical first=A). Writes scenarios.json.
An independent AI auditor then labels each covered / blind / invalid vs contract §1-9.
Run: python3 tools/analysis/fuzz_sequences.py
"""

import itertools
import json

ALLOWED = {
    "INITIAL": {"CFG"},
    "CONFIGURED": {"INIT", "RCFG"},
    "INIT": {"EXE", "INIT", "UNI", "RCFG"},
    "EXE": {"EXE", "UNI", "INIT", "RCFG"},
    "UNINIT": {"INIT", "RCFG"},
    "RECFG": {"INIT", "RCFG"},
}
NEXT = {
    "CFG": "CONFIGURED",
    "RCFG": "RECFG",
    "INIT": "INIT",
    "EXE": "EXE",
    "UNI": "UNINIT",
}


def skeletons(maxlen):
    out = []

    def walk(seq, state):
        if len(seq) >= 2 and "EXE" in seq and state in ("EXE", "UNINIT"):
            out.append(tuple(seq))
        if len(seq) == maxlen:
            return
        for tok in ALLOWED[state]:
            walk(seq + [tok], NEXT[tok])

    walk([], "INITIAL")
    return out


def decorate(skel):
    pos = [i for i, t in enumerate(skel) if t in ("CFG", "RCFG", "INIT", "UNI")]
    res = []
    for combo in itertools.product([0, 1], repeat=len(pos)):
        if combo and combo[0] != 0:
            continue  # canonical: first operand = A
        res.append(dict(zip(pos, combo)))
    return res


def render(skel, tags):
    toks = []
    last = "?"
    for i, t in enumerate(skel):
        if i in tags:
            op = "A" if tags[i] == 0 else "B"
            last = op
            toks.append(f"{t}_{op}")
        else:
            toks.append(f"{t}({last})")
    return " ".join(toks)


if __name__ == "__main__":
    skels = skeletons(6)
    scen = sorted({render(s, tg) for s in skels for tg in decorate(s)})
    json.dump(scen, open("scenarios.json", "w"))
    print("skeletons", len(skels), "scenarios", len(scen))
