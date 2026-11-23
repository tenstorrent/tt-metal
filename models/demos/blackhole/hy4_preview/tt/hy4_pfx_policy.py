# SPDX-License-Identifier: Apache-2.0
"""Pure-python (numpy-free) policy pieces of the Hy4 row-local prefix cache, shared by Hy4Engine and the CPU tests / the AgentX replay simulator (program hy4pfx). No ttnn, no torch.

* plan_groups   : split a lock-step prefill call into sub-calls by start depth (fixes the 'lockstep' loss). mode 'thr' = HY4_PFX_SPLIT=1 (gap > thr tokens), 'dp' = HY4_PFX_SPLIT=dp (exact minimum of the
                  summed call cost, cost of a call = max(prompt len) - min(start); a call costs the same for 1 or 4 rows).
* pick_victim   : which free row is overwritten by a request with no usable hit: 'lru' (old behaviour) or 'short' (the shortest retained chain first = cheapest to recompute).
* recompute_cost_s / pick_victim(mode='cost') / rank_victims : victim by COST (hy4sched, flag HY4_PFX_EVICT_COST): the chain that is cheapest to lose = recompute seconds x (optional) reuse probability. Shared with
                  the capacity tiers (hy4cap ChainStore can call rank_victims for its demotion order instead of a second implementation).
* stash_pages   : number of leading pool pages to save before a short request overwrites the head of a long retained chain (HY4_STASH=1), 0 = do not stash.
* restored_chain: the chain that is valid again after the stash pages were copied back.
"""
from __future__ import annotations


def plan_groups(starts, lens, mode="thr", thr=8192, chunk=512):
    """starts / lens {row: tokens}. -> list of row lists, cheapest call first. One group when splitting does not pay."""
    rows = sorted(starts, key=lambda r: (starts[r], r))
    if len(rows) <= 1:
        return [rows]
    if mode == "dp":
        n = len(rows)
        best, arg = [0] + [float("inf")] * n, [0] * (n + 1)
        for e in range(1, n + 1):
            for s in range(e):
                g = rows[s:e]
                c = max(lens[r] for r in g) - starts[g[0]]
                if best[s] + c < best[e]:
                    best[e], arg[e] = best[s] + c, s
        groups, e = [], n
        while e > 0:
            groups.append(rows[arg[e] : e])
            e = arg[e]
        groups.reverse()
    else:
        groups, cur = [], [rows[0]]
        for r in rows[1:]:
            if starts[r] - starts[cur[0]] > thr:
                groups.append(cur)
                cur = [r]
            else:
                cur.append(r)
        groups.append(cur)
    groups.sort(key=lambda g: (max(lens[r] - starts[r] for r in g), g[0]))
    return groups


# Cold-prefill cost model of Hy4: t(n) = A_TOK * n + B_TOK2 * n^2 seconds. A_TOK = 3.4 ms/token is the short-context cold rate and 4841 s for a cold 1M prefill is the long-context point (both [M] in
# HY4_PORT_PLAN / SERVING_PARITY); B_TOK2 follows from the two points: (4841 - 3400) / 1e12 = 1.44e-9 [C]. Only the ORDER of chains matters for victim choice (the model is monotone in n).
A_TOK = 3.4e-3
B_TOK2 = (4841.0 - 3.4e-3 * 1048576) / 1048576.0**2


def recompute_cost_s(n_tokens):
    n = max(0, int(n_tokens))
    return A_TOK * n + B_TOK2 * n * n


def keep_value(n_tokens, age_s=0.0, tau_s=0.0):
    """worth of keeping a chain = seconds saved if it is hit again, discounted by the chance that it is: exp(-age/tau) (tau_s <= 0: no discount, pure recompute cost)."""
    import math

    v = recompute_cost_s(n_tokens)
    return v * math.exp(-max(0.0, age_s) / tau_s) if tau_s > 0 else v


def rank_victims(items, now, tau_s=0.0):
    """items = [(key, n_tokens, last_used_time, protected)] -> keys, cheapest to lose first; protected items last (never before an unprotected one). The shared ordering for row-local chains and for tier stores."""
    return [
        k for k, _n, _t, _p in sorted(items, key=lambda it: (bool(it[3]), keep_value(it[1], now - it[2], tau_s), it[2]))
    ]


def pick_victim(cands, ret, ret_t, cur_row, mode="lru", now=None, tau_s=0.0):
    """cands = free rows without a usable hit; ret[r] = retained chain (len() works) or None; ret_t[r] = time the chain was written."""
    n = lambda q: 0 if ret[q] is None else len(ret[q])
    if mode == "cost":
        import time

        now = time.time() if now is None else now
        return min(cands, key=lambda q: (keep_value(n(q), now - ret_t[q], tau_s), ret_t[q], q != cur_row))
    if mode == "short":
        return min(cands, key=lambda q: (n(q), ret_t[q], q != cur_row))
    return min(cands, key=lambda q: (n(q) > 0, ret_t[q], q != cur_row))


def stash_pages(
    chain_len, prompt_len, shadow_pages, margin_tokens=4096, min_chain=32768, page=512, max_prompt=24576, ratio=4
):
    """pages [0, n) to save so that a request of prompt_len tokens (+ margin_tokens of decode) can run on a row holding a chain of chain_len tokens without destroying it; 0 = no stash (chain short, or the
    footprint does not fit the shadow pages)."""
    if (
        chain_len < min_chain or prompt_len > max_prompt or chain_len < ratio * prompt_len
    ):  # a longer prompt is itself worth keeping: its chain would be dropped at the release
        return 0
    n = -(-(prompt_len + margin_tokens) // page)
    if n > shadow_pages or n * page >= chain_len:
        return 0
    return n


def restored_chain(chain, n_pages, final_pos, spec_rows=8, page=512):
    """chain (sequence) after the n_pages leading pages were copied back. The request wrote positions up to final_pos + spec_rows (verify rows beyond the committed length): if that reaches page n_pages
    the pages from there on are junk -> the chain is valid only for its first n_pages * page tokens; else the whole chain is valid again.
    """
    if (final_pos + spec_rows) // page >= n_pages:
        return chain[: n_pages * page]
    return chain


TIER_KEYS = ("row_tokens", "xrow_tokens", "cold_tokens", "row_reqs", "xrow_reqs", "cold_reqs")


def tier_account(tier, hits, start, plan, lens):
    """per-call tier attribution of the prompt tokens of one prefill call (flag HY4_PFX_TIER_LOG). hits {row: aligned hit}, start = the call's common start, plan {row: (src row, h)} of cross-row copies,
    lens {row: prompt tokens}. A request's served tokens are min(hit, start): from the row's own retained chain ('row'), from a chain copied from another row ('xrow') or computed ('cold': prompt - served).
    Updates the cumulative dict ``tier`` in place and returns this call's numbers."""
    call = dict.fromkeys(TIER_KEYS, 0)
    for r, n in lens.items():
        served = min(hits.get(r, 0), start) if hits.get(r, 0) > 0 else 0
        tier_key = "xrow" if (served > 0 and r in plan) else "row"
        if served > 0:
            call[tier_key + "_tokens"] += served
            call[tier_key + "_reqs"] += 1
        call["cold_tokens"] += n - served
        if served == 0:
            call["cold_reqs"] += 1
    for k, v in call.items():
        tier[k] = tier.get(k, 0) + v
    return call


def tier_line(call, tier, xrow=None, stash=None, blocks=None):
    tot = sum(tier.get(k, 0) for k in ("row_tokens", "xrow_tokens", "cold_tokens")) or 1
    return (
        f"pfx_tiers call={call} cum={ {k: tier.get(k, 0) for k in TIER_KEYS} } "
        f"cum_hit_row={100.0 * tier.get('row_tokens', 0) / tot:.1f}% cum_hit_xrow={100.0 * tier.get('xrow_tokens', 0) / tot:.1f}% "
        f"cum_cold={100.0 * tier.get('cold_tokens', 0) / tot:.1f}% xrow={xrow} stash={stash} blocks={blocks}"
    )
