# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Speculative decoding for Laguna-XS-2.1 (ngram / prompt-lookup drafter), batch=1.

Mirrors the generator-level TT speculative-decode contract used by
``models/demos/gemma4/tt/spec_decode.py`` (draft K → batched verify → greedy /
speculative-sampling accept), but replaces gemma4's trained *assistant draft
model* with a **host-side ngram (prompt-lookup) proposer**. Laguna ships no MTP
head and no small draft variant, so a draft *model* is unavailable; ngram needs
none — it proposes the K tokens that followed the most recent earlier occurrence
of the current suffix in the running context. This is ideal for the
coding/agentic workload (output heavily copies paths / code from the prompt),
which drives a high acceptance rate.

Loop (one iteration commits ``m + 1`` tokens, 0 ≤ m ≤ K):

  1. Draft: the ngram proposer returns ``K = draft_len`` candidate tokens from the
     running token history (or fewer / none on a short or no match).
  2. Verify: the target runs ONE forward over ``[anchor, d0, …, d_{K-1}]`` at
     consecutive positions ``P-1 … P-1+K`` via the model's prefill/verify path,
     appending KV and yielding per-position logits ``[K+1, vocab]``.
  3. Accept: greedy (argmax match) or speculative sampling. Committed = the
     matched draft prefix + one bonus/correction token. KV rollback at batch=1 is
     implicit — rejected positions are overwritten by the next iteration's verify.

Correctness: the committed tokens are ALWAYS produced by the target verify, so
greedy speculative decode matches plain greedy decode and sampling matches the
target distribution — independent of the drafter's accuracy (which only affects
the acceptance rate / speed). The verify forward runs the multi-token prefill
path rather than the batch-1 decode path, so its numerics differ from single-step
decode by ~1e-5; that flips only target near-ties (top-2 logit gap ≲ 1), so
greedy spec-decode is token-identical to plain greedy up to the first near-tie
token and produces an equally-valid greedy trajectory thereafter.

The verify contract is the ONLY device dependency, provided by the generator as::

    verify_forward(tokens, start_pos, page_table=..., kv_cache=...,
                   page_tables_per_layer=...) -> host logits [S, vocab]

so this module is generator-agnostic and unit-testable against any object that
exposes it (see ``tests/`` and the standalone driver in ``doc/``).
"""

import time

import torch


def _to_probs(logits_row, temperature, top_p, top_k):
    """torch logits [vocab] -> probability vector [vocab] under temp/top-p/top-k.

    ``temperature <= 0`` returns a one-hot (greedy) distribution.
    """
    logits_row = logits_row.float()
    if not temperature or temperature <= 0:
        probs = torch.zeros_like(logits_row)
        probs[int(torch.argmax(logits_row))] = 1.0
        return probs
    logits_row = logits_row / temperature
    if top_k and 0 < top_k < logits_row.numel():
        kth = torch.topk(logits_row, top_k).values[-1]
        logits_row = torch.where(logits_row < kth, torch.full_like(logits_row, float("-inf")), logits_row)
    probs = torch.softmax(logits_row, dim=-1)
    if top_p and 0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
    s = probs.sum()
    return probs / s if s > 0 else probs


class NgramProposer:
    """Prompt-lookup drafter: propose the K tokens that followed the most recent
    earlier occurrence of the current suffix in the running context.

    Matches vLLM's ngram proposer semantics: try the longest suffix first
    (``max_n`` down to ``min_n``); on the newest earlier match, return up to
    ``k`` following tokens. No match -> empty proposal (that iteration verifies
    just the anchor, committing exactly one target-greedy token — never wrong,
    only slower).
    """

    def __init__(self, min_n=1, max_n=3):
        self.min_n = int(min_n)
        self.max_n = int(max_n)
        if self.min_n < 1 or self.max_n < self.min_n:
            raise ValueError(f"bad ngram window: min_n={min_n} max_n={max_n}")

    def propose(self, token_ids, k):
        """token_ids: list[int] running context. Return up to ``k`` draft ids."""
        n_ctx = len(token_ids)
        for n in range(min(self.max_n, n_ctx - 1), self.min_n - 1, -1):
            suffix = token_ids[-n:]
            # Scan for the newest earlier occurrence of `suffix` (excluding the
            # current trailing one). Search from the latest possible start back.
            for start in range(n_ctx - n - 1, -1, -1):
                if token_ids[start : start + n] == suffix:
                    cont = token_ids[start + n : start + n + k]
                    if cont:
                        return list(cont)
                    break  # matched but nothing follows at this length; try shorter n
        return []


class SpeculativeDecoder:
    """Batch-1 ngram speculative decoder over a target generator's verify path.

    Args:
        generator: object exposing ``verify_forward(tokens, start_pos, ...) ->
            [S, vocab]`` host logits (Laguna's ``generator_vllm`` adapter, or any
            stub with the same contract).
        kv_cache / page_table / page_tables_per_layer: forwarded verbatim to
            ``verify_forward`` (the paged KV the prefill already filled).
        stop_tokens: iterable of stop ids (decode halts after emitting one).
        draft_len: K, number of tokens proposed per iteration.
        ngram_min_n / ngram_max_n: ngram suffix window.
    """

    def __init__(
        self,
        generator,
        kv_cache=None,
        page_table=None,
        page_tables_per_layer=None,
        stop_tokens=None,
        draft_len=4,
        ngram_min_n=1,
        ngram_max_n=3,
        align=64,
        verify_mode="prefill",
        traced=False,
    ):
        self.gen = generator
        self.kv_cache = kv_cache
        self.page_table = page_table
        self.page_tables_per_layer = page_tables_per_layer
        self.stop_tokens = set(stop_tokens or [])
        self.draft_len = int(draft_len)
        if self.draft_len < 1:
            raise ValueError("draft_len must be >= 1")
        # Suffix-prefill flash-attention requires start_pos % align == 0 (align = the KV block size,
        # 64; q_chunk=32 ∧ k_chunk=64 → lcm 64 — see generator_vllm.verify_forward). The verify window
        # therefore starts at the 64-boundary at/below the anchor and re-feeds the known history tokens
        # in between (their KV rewrite is idempotent); only the trailing K+1 rows are read.
        self.align = int(align)
        if verify_mode not in ("prefill", "decode"):
            raise ValueError(f"verify_mode must be 'prefill' or 'decode', got {verify_mode!r}")
        self.verify_mode = verify_mode
        # traced=True replays a captured B=K+1 decode trace for the verify (decode mode only) —
        # removes host op-dispatch overhead so verify ≈ one decode step for all K+1 candidates.
        self.traced = bool(traced)
        self.proposer = NgramProposer(ngram_min_n, ngram_max_n)
        # Populated by generate(): per-iteration accepted counts (accept-rate stats).
        self.last_accepts = []

    def _verify_window(self, history, anchor_pos, drafts):
        """Target verify → the accept array g[0..K] ([len(drafts)+1, vocab]); row i is the
        target-greedy distribution for the same slot as draft d_i (row K = bonus).

        - "decode" mode: the K+1 candidates [anchor, d0..] sit in the BATCH dim at consecutive
          positions P-1..P-1+K (fast paged-decode SDPA; race-safe sequential KV write). No alignment —
          decode handles arbitrary per-row positions. Row j predicts positions[j]+1 directly.
        - "prefill" mode: reuse the (suffix-)prefill path over an align-64 window ending at the anchor
          (re-feed known history for chunk_start alignment), reading only the trailing K+1 rows."""
        anchor = int(history[-1])
        real = [int(d) for d in drafts]
        if self.verify_mode == "decode":
            toks = [anchor] + real + ([anchor] * (self.draft_len - len(real)) if self.traced else [])
            positions = [anchor_pos + j for j in range(len(toks))]  # P-1 .. P-1+K
            return self.gen.verify_greedy_decode(
                toks,
                positions,
                page_table=self.page_table,
                kv_cache=self.kv_cache,
                page_tables_per_layer=self.page_tables_per_layer,
                traced=self.traced,
            )
        A = (anchor_pos // self.align) * self.align
        window = [int(t) for t in history[A : anchor_pos + 1]] + real
        S = len(window)
        n_rows = len(drafts) + 1
        rows = list(range(S - n_rows, S))  # trailing anchor + drafts rows
        return self.gen.verify_forward(
            window,
            A,
            page_table=self.page_table,
            kv_cache=self.kv_cache,
            page_tables_per_layer=self.page_tables_per_layer,
            logit_rows=rows,
        )

    @staticmethod
    def _accept_greedy(drafts, verify):
        """Return (m, committed). ``verify`` is EITHER per-row logits [K+1, vocab] (prefill / eager
        decode) OR precomputed per-row greedy token ids [K+1] (traced decode-verify). g[i] is the
        target-greedy token for the same slot as draft d_i, so the matched prefix + one bonus (g[m])
        are all target-verified tokens."""
        n = len(drafts) + 1
        if hasattr(verify, "dim") and verify.dim() == 1:  # [K+1] greedy ids
            g = [int(verify[j]) for j in range(n)]
        else:  # [K+1, vocab] logits
            g = [int(torch.argmax(verify[j])) for j in range(n)]
        m = len(drafts)
        for i, d in enumerate(drafts):
            if d != g[i]:
                m = i
                break
        committed = list(drafts[:m]) + [g[m]]
        return m, committed

    def _accept_sampling(self, drafts, verify_logits, temperature, top_p, top_k):
        """Speculative-sampling acceptance. Matches the target distribution.

        ngram proposes deterministically (no draft distribution), so the proposal
        density q(d) is a point mass at d: the accept ratio min(1, p(d)/q(d))
        reduces to accepting d with probability p(d) (target prob of the draft),
        and the correction on rejection samples from the residual max(p-δ_d,0)
        renormalized — which is exactly ``p`` with the drafted token removed."""
        K = len(drafts)
        committed = []
        m = K
        for i in range(K):
            p = _to_probs(verify_logits[i], temperature, top_p, top_k)  # target dist for slot i
            d = drafts[i]
            pd = float(p[d])
            if torch.rand(()) < min(1.0, pd):
                committed.append(d)
            else:
                resid = p.clone()
                resid[d] = 0.0
                s = resid.sum()
                corr = int(torch.argmax(p)) if s <= 0 else int(torch.multinomial(resid / s, 1))
                committed.append(corr)
                m = i
                break
        if m == K:
            p = _to_probs(verify_logits[K], temperature, top_p, top_k)
            committed.append(
                int(torch.multinomial(p, 1)) if (temperature and temperature > 0) else int(torch.argmax(p))
            )
        return m, committed

    def generate(
        self, prompt_tokens, max_new_tokens, temperature=0.0, top_p=1.0, top_k=0, on_progress=None, progress_every=64
    ):
        """Run speculative decode from a prefilled prompt.

        The paged KV for ``prompt_tokens`` must already be filled (by the
        generator's prefill). ``prompt_tokens`` is the FULL context (its length
        sets the first anchor position P = len(prompt_tokens), anchor token =
        prompt_tokens[-1] at position P-1).

        ``on_progress(tokens_done, max_new_tokens, mean_accept)`` is called every
        ``progress_every`` committed tokens for a tailable live stream of the run.

        Returns (generated_ids, accepts_per_iter). ``accepts_per_iter`` records
        the number of accepted drafts m each iteration (for accept-rate stats).
        """
        greedy = not temperature or temperature <= 0
        if self.verify_mode == "decode" and not greedy:
            raise NotImplementedError(
                "decode verify is greedy-only (on-device top-k=1 sampler returns ids, not logits); "
                "use verify_mode='prefill' for temperature>0 speculative sampling"
            )
        history = list(int(t) for t in prompt_tokens)
        out = []
        accepts = []
        next_report = progress_every
        K = self.draft_len
        while len(out) < max_new_tokens:
            drafts = self.proposer.propose(history, K)
            anchor_pos = len(history) - 1  # P-1
            verify = self._verify_window(history, anchor_pos, drafts)  # [K+1, vocab] logits OR [K+1] ids

            if greedy:
                m, committed = self._accept_greedy(drafts, verify)
            else:
                m, committed = self._accept_sampling(drafts, verify, temperature, top_p, top_k)
            accepts.append(m)

            for tok in committed:
                out.append(tok)
                history.append(tok)
                if tok in self.stop_tokens:
                    self.last_accepts = accepts
                    return out, accepts
                if len(out) >= max_new_tokens:
                    break
            if on_progress is not None and len(out) >= next_report:
                on_progress(len(out), max_new_tokens, sum(accepts) / max(1, len(accepts)))
                next_report += progress_every
        self.last_accepts = accepts
        return out[:max_new_tokens], accepts


def plain_greedy_via_verify(
    generator,
    prompt_tokens,
    max_new_tokens,
    kv_cache=None,
    page_table=None,
    page_tables_per_layer=None,
    stop_tokens=None,
    align=64,
    verify_mode="prefill",
):
    """Reference greedy decode driving ONE verify (no drafts) per step — the baseline the spec output
    must match token-for-token. Uses the SAME verify path as spec-decode (batched-vs-single numerics
    common-mode). In "decode" mode this is a plain single-token decode step (B=1, no seq-write).
    Returns (generated_ids, seconds)."""
    stop = set(stop_tokens or [])
    history = list(int(t) for t in prompt_tokens)
    out = []
    t0 = time.perf_counter()
    while len(out) < max_new_tokens:
        anchor_pos = len(history) - 1
        if verify_mode == "decode":
            g = generator.verify_greedy_decode(
                [int(history[-1])],
                [anchor_pos],
                page_table=page_table,
                kv_cache=kv_cache,
                page_tables_per_layer=page_tables_per_layer,
                traced=False,
            )
            tok = int(g.reshape(-1)[0])
            out.append(tok)
            history.append(tok)
            if tok in stop:
                break
            continue
        else:
            A = (anchor_pos // align) * align
            window = [int(t) for t in history[A : anchor_pos + 1]]  # aligned prefix ending at the anchor
            logits = generator.verify_forward(
                window,
                A,
                page_table=page_table,
                kv_cache=kv_cache,
                page_tables_per_layer=page_tables_per_layer,
                logit_rows=[len(window) - 1],
            )
        tok = int(torch.argmax(logits[0]))
        out.append(tok)
        history.append(tok)
        if tok in stop:
            break
    return out, time.perf_counter() - t0
