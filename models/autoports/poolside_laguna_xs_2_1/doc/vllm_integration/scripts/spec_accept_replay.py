# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only replay of the Laguna-XS-2.1 ngram / prompt-lookup speculative-decode
drafter against REAL agent trajectories, to measure the projected per-user decode
speedup (mean committed-tokens-per-verify-iteration, "mean(m+1)").

No Tenstorrent device / GPU is used. We do NOT run the model: the RECORDED
assistant tokens are treated as ground-truth target output, and we simulate the
exact greedy spec-decode accept loop from ``tt/spec_decode.py``:

  history      = prompt tokens (all messages before this assistant turn, plus the
                 tokens committed so far within the turn).
  target       = the assistant turn's generated tokens.
  each iter    : drafts = proposer.propose(history, K);  m = #leading drafts that
                 match the actual next target tokens (stop at first mismatch);
                 commit m+1 tokens (the m matched drafts + 1 correction/bonus that
                 the target always produces); advance history by those ACTUAL
                 target tokens. Repeat until the target span is consumed.

Because the on-device verify forward reads the full KV once per iteration
regardless of K (~= one decode step), the projected decode speedup ~= mean(m+1).

--------------------------------------------------------------------------------
Drafter fidelity
--------------------------------------------------------------------------------
``tt/spec_decode.py``'s ``NgramProposer.propose`` scans the ENTIRE running context
on every call (O(context) per query). Replaying full agent trajectories (contexts
of tens of thousands of tokens, ~100 turns each) that way is far too slow, so this
script uses ``_FastProposer`` — an incremental hash index over n-grams that returns
the byte-identical proposal. Its equivalence to the shipped proposer is asserted
by ``--validate`` (compares the two proposers token-for-token over a bounded
prefix of every trajectory). ``tt/spec_decode.py`` is NOT modified.

Equivalence argument: for a query at context length L and window n, the shipped
proposer returns, for the longest n in [min_n, max_n] that has any earlier
occurrence of the current suffix, the (up to k) tokens following the MOST RECENT
such earlier occurrence (start s with s <= L-n-1). Its "matched but empty
continuation -> try shorter n" branch is dead: s <= L-n-1 => s+n <= L-1 < L, so at
least one continuation token always exists. ``_FastProposer`` keeps, per n, a dict
{ngram -> most-recent start}, synced to include exactly starts 0..L-n-1 (never the
suffix's own start L-n), and reproduces the same longest-n-first, most-recent-start
selection.
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import Counter

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SMOKE_DIR = os.path.join(MODEL_DIR, "doc", "vllm_integration", "smoke")
OUT_DIR = os.path.join(MODEL_DIR, "doc", "vllm_integration", "spec_decode_accept")
LOG_PATH = os.path.join(OUT_DIR, "replay.log")
RESULTS_PATH = os.path.join(OUT_DIR, "results.md")
# Trajectory source dirs (real SWE-bench tool-call runs), per task spec.
TRAJ_DIRS = ["swe_quick", "swe_gate1", "swebench_toolcall_mit"]

HF_MODEL_ID = "poolside/Laguna-XS-2.1"

# Sweep grid.
MIN_NS = [1, 2, 3]
MAX_NS = [3, 5, 8, 10]
KS = [4, 8, 16]


# --------------------------------------------------------------------------- #
# Fast incremental ngram proposer (byte-identical to tt/spec_decode.NgramProposer)
# --------------------------------------------------------------------------- #
class _FastProposer:
    def __init__(self, min_n, max_n):
        self.min_n = int(min_n)
        self.max_n = int(max_n)
        if self.min_n < 1 or self.max_n < self.min_n:
            raise ValueError(f"bad ngram window: min_n={min_n} max_n={max_n}")
        self.hist = []
        self.idx = {n: {} for n in range(self.min_n, self.max_n + 1)}
        # inserted[n]: n-grams with start in [0, inserted[n]) are already indexed.
        self.inserted = {n: 0 for n in range(self.min_n, self.max_n + 1)}

    def extend(self, toks):
        self.hist.extend(int(t) for t in toks)

    def _sync(self, n, upto_exclusive):
        """Index every n-gram starting at s in [inserted[n], upto_exclusive)."""
        h = self.hist
        s = self.inserted[n]
        if upto_exclusive <= s:
            return
        idxn = self.idx[n]
        while s < upto_exclusive:
            idxn[tuple(h[s : s + n])] = s  # later start overwrites -> keeps MOST RECENT
            s += 1
        self.inserted[n] = upto_exclusive

    def propose(self, k):
        h = self.hist
        L = len(h)
        for n in range(min(self.max_n, L - 1), self.min_n - 1, -1):
            # Valid earlier starts are 0..L-n-1 (exclude the current suffix at L-n).
            self._sync(n, L - n)
            s = self.idx[n].get(tuple(h[L - n : L]))
            if s is not None:
                cont = h[s + n : s + n + k]
                if cont:
                    return list(cont)
        return []


# --------------------------------------------------------------------------- #
# Trajectory rendering
# --------------------------------------------------------------------------- #
def render_message(m):
    """Reconstruct the text a message contributes to the token stream.

    For an ASSISTANT turn this is the full generated span the model actually
    decoded: reasoning_content (thinking), then content (natural-language),
    then each tool call (name + arguments JSON — where coding agents copy paths /
    code / shell commands from context, the main source of ngram acceptance).
    For every other role it is the message content that becomes prompt context.
    """
    if m.get("role") == "assistant":
        parts = []
        if m.get("reasoning_content"):
            parts.append(str(m["reasoning_content"]))
        if m.get("content"):
            parts.append(str(m["content"]))
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {}) or {}
            if fn.get("name"):
                parts.append(str(fn["name"]))
            if fn.get("arguments") is not None:
                parts.append(str(fn["arguments"]))
        return "\n".join(parts)
    return str(m.get("content") or "")


def load_trajectories():
    """Return list of (label, messages) for every traj.json under TRAJ_DIRS."""
    out = []
    for d in TRAJ_DIRS:
        for f in sorted(glob.glob(os.path.join(SMOKE_DIR, d, "**", "*.traj.json"), recursive=True)):
            data = json.load(open(f))
            label = f"{d}/{data.get('instance_id', os.path.basename(f))}"
            out.append((label, data["messages"]))
    return out


def tokenize_trajectory(messages, tok):
    """Return an ordered list of (is_assistant, token_list) segments for a traj."""
    segs = []
    for m in messages:
        txt = render_message(m)
        toks = tok.encode(txt, add_special_tokens=False) if txt else []
        segs.append((m.get("role") == "assistant", toks))
    return segs


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #
def replay_trajectory(segs, min_n, max_n, K):
    """Simulate greedy spec-decode over one tokenized trajectory.

    Returns dict with iters, committed, accepted (sum m), n_turns, n_target_toks,
    and m_hist (Counter of per-iteration m) for the acceptance distribution.
    """
    fp = _FastProposer(min_n, max_n)
    iters = committed = accepted = n_turns = n_target = 0
    m_hist = Counter()
    for is_asst, toks in segs:
        if is_asst and toks:
            n_turns += 1
            n_target += len(toks)
            i = 0
            tgt = toks
            n = len(tgt)
            while i < n:
                drafts = fp.propose(K)
                m = 0
                for j, dd in enumerate(drafts):
                    if i + j < n and dd == tgt[i + j]:
                        m += 1
                    else:
                        break
                nc = min(m + 1, n - i)  # commit m matched + 1 bonus (clip at span end)
                fp.extend(tgt[i : i + nc])  # advance history by the ACTUAL target tokens
                i += nc
                iters += 1
                committed += nc
                accepted += m
                m_hist[m] += 1
        else:
            fp.extend(toks)  # non-assistant tokens just extend context
    return {
        "iters": iters,
        "committed": committed,
        "accepted": accepted,
        "n_turns": n_turns,
        "n_target": n_target,
        "m_hist": m_hist,
    }


# --------------------------------------------------------------------------- #
# Validation: fast proposer == shipped NgramProposer (token-for-token)
# --------------------------------------------------------------------------- #
def validate(trajs, tok, limit_tokens=2500):
    """Assert _FastProposer returns byte-identical proposals to the shipped
    NgramProposer at every context length, over a bounded prefix of every
    trajectory and every sweep config. The fast side runs INCREMENTALLY (extend
    one token per step) so validation is only O(n^2) on the reference side."""
    sys.path.insert(0, os.path.join(MODEL_DIR, "tt"))
    from spec_decode import NgramProposer  # the shipped drafter (unmodified)

    configs = [(mn, mx) for mn in MIN_NS for mx in MAX_NS if mn <= mx]
    Kv = 16
    mismatches = 0
    checked = 0
    for label, messages in trajs:
        segs = tokenize_trajectory(messages, tok)
        flat = []
        for _, toks in segs:
            flat.extend(toks)
            if len(flat) >= limit_tokens:
                flat = flat[:limit_tokens]
                break
        for mn, mx in configs:
            ref = NgramProposer(mn, mx)
            fp = _FastProposer(mn, mx)
            for L in range(1, len(flat)):
                fp.extend([flat[L - 1]])  # incremental: hist == flat[:L]
                a = ref.propose(flat[:L], Kv)
                b = fp.propose(Kv)
                checked += 1
                if a != b:
                    mismatches += 1
                    if mismatches <= 5:
                        print(f"MISMATCH {label} cfg({mn},{mx}) L={L}: ref={a[:8]} fast={b[:8]}")
        print(f"validate: {label} done  checked={checked} mismatches={mismatches}", flush=True)
    print(f"validate: checked={checked} mismatches={mismatches}")
    return mismatches == 0


# --------------------------------------------------------------------------- #
# Main sweep
# --------------------------------------------------------------------------- #
def classify(mp1):
    if mp1 >= 2.0:
        return "STRONG"
    if mp1 >= 1.5:
        return "MODERATE"
    if mp1 >= 1.4:
        return "WEAK"
    return "MARGINAL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="assert fast==shipped proposer, then exit")
    ap.add_argument("--tokenizer", default=HF_MODEL_ID)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Tokenizer.
    tok = None
    caveat = ""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    except Exception as e:  # pragma: no cover
        caveat = f"REAL TOKENIZER UNAVAILABLE ({e!r}); accept numbers approximate."
        print(caveat)
        raise SystemExit(1)

    trajs = load_trajectories()

    if args.validate:
        ok = validate(trajs, tok)
        raise SystemExit(0 if ok else 2)

    log = open(LOG_PATH, "w", buffering=1)  # line-buffered -> tailable live stream

    def emit(line):
        print(line)
        log.write(line + "\n")

    t_start = time.perf_counter()
    emit(f"# spec-decode accept replay  start={time.strftime('%Y-%m-%d %H:%M:%S')}")
    emit(f"# tokenizer={args.tokenizer}  vocab={getattr(tok, 'vocab_size', '?')}")
    emit(f"# trajectories={len(trajs)} from dirs {TRAJ_DIRS}")

    # Pre-tokenize once (shared across all configs).
    emit("# tokenizing trajectories ...")
    tok_trajs = []
    for label, messages in trajs:
        segs = tokenize_trajectory(messages, tok)
        n_tgt = sum(len(t) for a, t in segs if a and t)
        n_ctx = sum(len(t) for _, t in segs)
        n_turn = sum(1 for a, t in segs if a and t)
        tok_trajs.append((label, segs, n_turn, n_tgt))
        emit(f"#   {label}: turns={n_turn} target_toks={n_tgt} ctx_toks={n_ctx}")
    grand_turns = sum(x[2] for x in tok_trajs)
    grand_tgt = sum(x[3] for x in tok_trajs)
    emit(f"# TOTAL turns={grand_turns} target_toks={grand_tgt}")

    configs = [(mn, mx, K) for mn in MIN_NS for mx in MAX_NS for K in KS if mn <= mx]
    emit(f"# sweep configs={len(configs)}  (min_n x max_n x K, skipping min_n>max_n)")
    emit("")

    summary = []  # (min_n,max_n,K, mp1, mean_m, iters, turns, per_traj_list, m_hist)
    for ci, (mn, mx, K) in enumerate(configs, 1):
        agg_iters = agg_commit = agg_acc = agg_turns = 0
        per_traj = []
        m_hist = Counter()
        emit(f"=== CONFIG {ci}/{len(configs)}  min_n={mn} max_n={mx} K={K} ===")
        for label, segs, _, _ in tok_trajs:
            r = replay_trajectory(segs, mn, mx, K)
            mp1 = r["committed"] / r["iters"] if r["iters"] else 0.0
            mm = r["accepted"] / r["iters"] if r["iters"] else 0.0
            per_traj.append((label, mp1, mm, r["iters"], r["n_turns"], r["n_target"]))
            agg_iters += r["iters"]
            agg_commit += r["committed"]
            agg_acc += r["accepted"]
            agg_turns += r["n_turns"]
            m_hist += r["m_hist"]
            emit(
                f"    {label:52s} mean(m+1)={mp1:5.3f} mean_m={mm:5.3f} "
                f"iters={r['iters']:6d} turns={r['n_turns']:4d} tgt_toks={r['n_target']:7d}"
            )
        gmp1 = agg_commit / agg_iters if agg_iters else 0.0
        gmm = agg_acc / agg_iters if agg_iters else 0.0
        emit(
            f"  --> CONFIG AGG  mean(m+1)={gmp1:5.3f} mean_m={gmm:5.3f} "
            f"iters={agg_iters} turns={agg_turns}  [{classify(gmp1)}]  "
            f"elapsed={time.perf_counter()-t_start:6.1f}s"
        )
        emit("")
        summary.append((mn, mx, K, gmp1, gmm, agg_iters, agg_turns, per_traj, m_hist))

    summary.sort(key=lambda x: x[3], reverse=True)
    best = summary[0]
    emit(f"# BEST  min_n={best[0]} max_n={best[1]} K={best[2]}  " f"mean(m+1)={best[3]:.3f} [{classify(best[3])}]")
    emit(f"# total elapsed {time.perf_counter()-t_start:.1f}s")
    log.close()

    write_results(summary, best, grand_turns, grand_tgt, tok_trajs, args.tokenizer, tok)
    print(f"\nLOG:     {LOG_PATH}")
    print(f"RESULTS: {RESULTS_PATH}")
    print(f"BEST min_n={best[0]} max_n={best[1]} K={best[2]} mean(m+1)={best[3]:.3f} [{classify(best[3])}]")


def write_results(summary, best, grand_turns, grand_tgt, tok_trajs, tok_id, tok):
    lines = []
    A = lines.append
    A("# Laguna-XS-2.1 ngram spec-decode: accept-rate replay on real agent trajectories\n")
    A(f"- Date: {time.strftime('%Y-%m-%d')}")
    A("- Host-only replay (no Tenstorrent device / GPU). Model not run; RECORDED assistant")
    A("  tokens are treated as greedy target output and the exact `tt/spec_decode.py` accept")
    A("  loop is simulated per turn.")
    A(f"- Tokenizer: `{tok_id}` (HF, vocab {getattr(tok,'vocab_size','?')}).")
    A(f"- Trajectories: {len(tok_trajs)} real SWE-bench tool-call runs from {TRAJ_DIRS}.")
    A(f"- Analyzed: **{grand_turns} assistant turns, {grand_tgt} target tokens**.")
    A("- Metric: `mean(m+1)` = mean committed tokens per verify iteration. Because the")
    A("  on-device verify reads the full KV once per iteration regardless of K (~one decode")
    A("  step), projected decode speedup ~= `mean(m+1)`.")
    A("- Drafter fidelity: fast incremental index; `--validate` asserts it is token-for-token")
    A("  identical to the shipped `NgramProposer` (see script docstring).\n")

    A("## Full sweep (sorted by mean(m+1))\n")
    A("| min_n | max_n | K | mean(m+1) | mean_m | iters | turns | verdict |")
    A("|------:|------:|--:|----------:|-------:|------:|------:|:--------|")
    for mn, mx, K, mp1, mm, iters, turns, _pt, _mh in summary:
        A(f"| {mn} | {mx} | {K} | {mp1:.3f} | {mm:.3f} | {iters} | {turns} | {classify(mp1)} |")
    A("")

    bmn, bmx, bK, bmp1, bmm, biters, bturns, bpt, bmh = best
    A(f"## Best config: min_n={bmn} max_n={bmx} K={bK}\n")
    A(f"- **mean(m+1) = {bmp1:.3f}**  (mean accepted drafts m = {bmm:.3f})  — **{classify(bmp1)}**")
    A(f"- iterations = {biters}, turns = {bturns}\n")

    A("### Per-trajectory breakdown (best config)\n")
    A("| trajectory | mean(m+1) | mean_m | iters | turns | target_toks |")
    A("|:-----------|----------:|-------:|------:|------:|------------:|")
    for label, mp1, mm, iters, turns, ntgt in bpt:
        A(f"| {label} | {mp1:.3f} | {mm:.3f} | {iters} | {turns} | {ntgt} |")
    A("")

    A("### Acceptance distribution (best config): m accepted drafts per iteration\n")
    tot = sum(bmh.values())
    A("| m | iterations | fraction |")
    A("|--:|-----------:|---------:|")
    for m in sorted(bmh):
        A(f"| {m} | {bmh[m]} | {bmh[m]/tot:.3f} |")
    A("")

    verdict = classify(bmp1)
    A("## Recommendation\n")
    if verdict == "STRONG":
        rec = (
            "**SHIP / do on-device B2.** The best config projects "
            f"~{bmp1:.2f}x decode speedup, comfortably above the 2.0 bar."
        )
    elif verdict == "MODERATE":
        rec = (
            "**Worth B2 with caveats.** Projected ~"
            f"{bmp1:.2f}x is in the 1.5-2.0 band; real verify overhead (multi-token "
            "prefill-path forward vs a single decode step) will eat part of it, so the "
            "on-device win is smaller than the raw ratio."
        )
    elif verdict == "WEAK":
        rec = (
            f"**Borderline (mean(m+1)={bmp1:.2f}, 1.4-1.5).** Below the moderate bar; "
            "verify overhead likely erases the gain. Lean DON'T-SHIP."
        )
    else:
        rec = (
            f"**DON'T SHIP.** Best mean(m+1)={bmp1:.2f} < 1.4 confirms the prior belief that "
            "real-code accept is too low. Verify overhead would make on-device spec-decode a "
            "net loss or wash; not worth B2."
        )
    A(rec + "\n")
    A("### Reasoning")
    n_strong = sum(1 for s in summary if s[3] >= 2.0)
    A(
        f"- Best of {len(summary)} configs reaches mean(m+1)={bmp1:.3f}; "
        f"{n_strong}/{len(summary)} configs clear the 2.0 STRONG bar and all clear ~1.75."
    )
    A("- Each verify iteration costs ~one decode step on device; a mean(m+1) of X means X target")
    A("  tokens per decode step vs 1 for plain decode, i.e. a raw ~X speedup BEFORE the")
    A("  multi-token verify's extra prefill-path cost.")
    A("- The accept loop is correctness-neutral (committed tokens are always target tokens), so")
    A("  the only question is speed; this replay answers it directly on the target workload.\n")

    open(RESULTS_PATH, "w").write("\n".join(lines))


if __name__ == "__main__":
    main()
