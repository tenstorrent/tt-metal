# SPDX-License-Identifier: Apache-2.0
"""Pure-python (numpy-free) policy pieces of the Hy4 row-local prefix cache, shared by Hy4Engine and the CPU tests / the AgentX replay simulator (program hy4pfx). No ttnn, no torch.

* plan_groups   : split a lock-step prefill call into sub-calls by start depth (fixes the 'lockstep' loss). mode 'thr' = HY4_PFX_SPLIT=1 (gap > thr tokens), 'dp' = HY4_PFX_SPLIT=dp (exact minimum of the
                  summed call cost, cost of a call = max(prompt len) - min(start); a call costs the same for 1 or 4 rows).
* pick_victim   : which free row is overwritten by a request with no usable hit: 'lru' (old behaviour) or 'short' (the shortest retained chain first = cheapest to recompute).
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


def pick_victim(cands, ret, ret_t, cur_row, mode="lru"):
    """cands = free rows without a usable hit; ret[r] = retained chain (len() works) or None; ret_t[r] = time the chain was written."""
    n = lambda q: 0 if ret[q] is None else len(ret[q])
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
