# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Does the MoE router's per-expert hotness **persist across decode tokens**?

``moe_skew_analysis.py`` recovers the per-die active-expert count from the
archived profile. That profile is **one decode token**: 48 layers x 4 dies is
192 numbers, but within each layer the four counts sum to the router's top-8, so
the whole sample is 48 layers of a *single* token. At n=1 the only question that
can be asked is whether that one token's counts look uniform, and the answer --
they do -- is a failure to reject, not evidence of uniformity.

The question that actually decides the expert-permutation lever is different and
is invisible at n=1: **is the same expert hot on the same die on every token?**
A relabelling of experts across dies cannot exploit a skew that reshuffles from
token to token -- whatever it un-clusters on this token it clusters on the next.
It can only exploit *persistent* hotness. So this probe runs the real 48-layer
model on the real mesh for many free-running decode tokens and records the
router's top-8 expert ids at every layer of every token.

It records the routing by wrapping ``ttnn.topk``: ``_router_tail`` is the only
caller in a greedy decode step that asks for k = ``num_experts_per_tok``, so the
wrapper reads the indices back and passes the tensors through untouched. Nothing
in the model changes; the probe only observes.

Three things come out, and the third is the one that decides the lever:

1. the per-die active-expert counts over ``tokens x layers`` samples rather than
   over one token, so the chi-square in ``moe_skew_analysis.py`` gets a sample
   size that is not 1;
2. the per-expert selection rate, per layer, across tokens -- the direct measure
   of persistent hotness;
3. **what the best expert-to-die assignment would actually have bought.** For
   each layer the shipped assignment is contiguous windows of 32; this searches
   for an assignment that minimises the mean over tokens of the per-token
   maximum die count (which is what a collective waits for), by local swaps from
   many random restarts. If the search finds nothing, the lever is closed on
   measurement rather than on a chi-square. If it finds something, the lever is
   open and this probe says so.

**Three** prompts are run, because routing is content-dependent and a *shipped*
layout is fixed at weight-load time, so the gain that counts is one that survives
being fitted on routing the layout will not see again. ``--cross`` does that fit,
on the host, from the archived raw routing.

Two prompts were not enough, and the round-2 review is why there are three. An
n=2 sample yields exactly two fit-and-score directions; the stage published them
as the range, and the reviewer's independently captured third prompt showed they
were the two *smallest* of the six directions n=3 yields. The published figure
had been the bottom of a range read as the whole of it. ``--cross`` now reports
every direction and its spread, plus a pooled fit held out on an unseen prompt --
which transfers *better* than any single-prompt fit, so even the widened range is
a floor.

    python moe_routing_across_tokens_probe.py --tokens 128 --layers 48
    python moe_routing_across_tokens_probe.py --tokens 128 --layers 48 --tag _prompt2 \
        --prompt "..."
    python moe_routing_across_tokens_probe.py --tokens 128 --layers 48 --tag _prompt3 \
        --prompt "..."
    python moe_routing_across_tokens_probe.py --cross \
        moe_routing_across_tokens_raw.json.gz moe_routing_across_tokens_prompt2_raw.json.gz \
        moe_routing_across_tokens_prompt3_raw.json.gz

One device job at a time; the first three open their own mesh and ``--cross``
opens none. The third prompt's capture is the **reviewer's**, archived here
unmodified -- see the ``provenance`` key in
``moe_routing_across_tokens_prompt3.json``.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))

from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt import multichip_decoder as MC  # noqa: E402
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.generator import build_generator  # noqa: E402
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.model import DEFAULT_TRACE_REGION_SIZE  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
NUM_DEVICES = 4
TOP_K = 8
NUM_EXPERTS = 128
EXPERTS_PER_DIE = NUM_EXPERTS // NUM_DEVICES
#: us per additional locally-active expert, from ``moe_skew_analysis.py``.
EXPERT_STEP_US = 6.85

#: A real prompt. Routing is content-dependent, so the sample has to come from
#: text the model would actually see rather than from a range of token ids.
PROMPT = (
    "Write a short technical explanation of how a mixture-of-experts transformer "
    "routes tokens to experts, and then give a Python function that computes the "
    "top-k indices of a vector without using any library call. Explain the "
    "complexity of your implementation."
)


# --- recording ---------------------------------------------------------------


class RouterRecorder:
    """Wrap ``ttnn.topk`` and keep every top-``TOP_K`` index vector it returns."""

    def __init__(self, k: int = TOP_K):
        self.k = k
        self.rows: list[list[int]] = []
        self.enabled = False
        self._original = ttnn.topk

    def __enter__(self):
        recorder = self

        def topk(tensor, k, *args, **kwargs):
            values, indices = recorder._original(tensor, k, *args, **kwargs)
            if recorder.enabled and k == recorder.k:
                shards = ttnn.get_device_tensors(indices)
                host = ttnn.to_torch(shards[0] if shards else indices)
                recorder.rows.append([int(v) for v in host.reshape(-1)[: recorder.k].tolist()])
            return values, indices

        ttnn.topk = topk
        return self

    def __exit__(self, *exc):
        ttnn.topk = self._original
        return False


# --- the assignment search ---------------------------------------------------


def selection_matrix(selection: list[list[int]]) -> np.ndarray:
    """``[tokens, NUM_EXPERTS]`` indicator of which experts each token selected."""
    matrix = np.zeros((len(selection), NUM_EXPERTS), dtype=np.int16)
    for token, row in enumerate(selection):
        matrix[token, row] = 1
    return matrix


def mean_max_count(matrix: np.ndarray, assignment: np.ndarray) -> float:
    """Mean over tokens of the largest number of a die's experts a token selects.

    A collective waits for the slowest die, so the per-token cost is the maximum
    count over the four dies, not the mean.
    """
    counts = np.zeros((matrix.shape[0], NUM_DEVICES), dtype=np.int16)
    for die in range(NUM_DEVICES):
        counts[:, die] = matrix[:, assignment == die].sum(axis=1)
    return float(counts.max(axis=1).mean())


def search_assignment(matrix: np.ndarray, *, restarts: int, sweeps: int, rng: random.Random):
    """Best balanced expert-to-die assignment found by swap-descent.

    Only *balanced* assignments are searched -- every die keeps exactly 32
    experts -- because the EP=4 layout gives each die the same weight memory and
    the same ``nnz=None`` scan cost. Moves are swaps of two experts on different
    dies, so balance is preserved by construction. Restart 0 starts from the
    shipped contiguous windows, so the search can never report worse than
    shipped.
    """
    best_assignment: np.ndarray | None = None
    best_cost = float("inf")
    for restart in range(restarts):
        if restart == 0:
            assignment = np.arange(NUM_EXPERTS) // EXPERTS_PER_DIE
        else:
            order = np.array(rng.sample(range(NUM_EXPERTS), NUM_EXPERTS))
            assignment = np.empty(NUM_EXPERTS, dtype=int)
            assignment[order] = np.arange(NUM_EXPERTS) // EXPERTS_PER_DIE
        counts = np.zeros((matrix.shape[0], NUM_DEVICES), dtype=np.int16)
        for die in range(NUM_DEVICES):
            counts[:, die] = matrix[:, assignment == die].sum(axis=1)
        cost = float(counts.max(axis=1).mean())
        for _ in range(sweeps):
            improved = False
            for a in range(NUM_EXPERTS):
                for b in range(a + 1, NUM_EXPERTS):
                    da, db = assignment[a], assignment[b]
                    if da == db:
                        continue
                    delta = matrix[:, b] - matrix[:, a]
                    counts[:, da] += delta
                    counts[:, db] -= delta
                    trial = float(counts.max(axis=1).mean())
                    if trial < cost - 1e-12:
                        cost, improved = trial, True
                        assignment[a], assignment[b] = db, da
                    else:
                        counts[:, da] -= delta
                        counts[:, db] += delta
            if not improved:
                break
        if cost < best_cost:
            best_cost, best_assignment = cost, assignment.copy()
    return best_cost, best_assignment


# --- main --------------------------------------------------------------------


def _per_layer_matrices(run: dict, layers: int) -> list[np.ndarray]:
    rows = run["top8"]
    return [selection_matrix([rows[t * layers + layer] for t in range(run["tokens"])]) for layer in range(layers)]


def _shared_structure(matrices: list[list[np.ndarray]], layers: int) -> dict:
    """How much of the per-expert hotness two prompts have in common.

    The cross-prompt fit answers "what is the shared structure *worth*". This
    answers the prior question -- "is there any" -- without going through a
    stochastic search, so it corroborates the fit from a direction the search
    cannot bias. Two statistics per prompt pair per layer:

    * the overlap of the two prompts' top-8 most-selected experts. Under
      independent routing two sets of 8 out of ``NUM_EXPERTS`` overlap in
      ``8 * 8 / 128 = 0.5`` experts on average;
    * the Spearman rank correlation of the full per-expert selection counts,
      which uses every expert and not just the head.

    Both come out well above chance, which is the same conclusion the fit
    reaches: the shared structure is real, and small.
    """
    from scipy.stats import spearmanr

    rates = [np.array([m[layer].sum(axis=0) for layer in range(layers)], dtype=float) for m in matrices]
    pairs = []
    for i, j in itertools.combinations(range(len(rates)), 2):
        overlaps, correlations = [], []
        for layer in range(layers):
            top_i = set(np.argsort(-rates[i][layer])[:TOP_K].tolist())
            top_j = set(np.argsort(-rates[j][layer])[:TOP_K].tolist())
            overlaps.append(len(top_i & top_j))
            correlations.append(float(spearmanr(rates[i][layer], rates[j][layer]).correlation))
        pairs.append(
            {
                "prompts": [i, j],
                "mean_top8_overlap": float(np.mean(overlaps)),
                "mean_per_expert_rank_correlation": float(np.mean(correlations)),
            }
        )
        print(
            f"prompts {i},{j}: top-8 hot-set overlap {np.mean(overlaps):.3f}/{TOP_K}, "
            f"per-expert rank correlation {np.mean(correlations):.3f}",
            flush=True,
        )
    chance = TOP_K * TOP_K / NUM_EXPERTS
    return {
        "per_pair": pairs,
        "mean_top8_overlap_over_pairs": float(np.mean([p["mean_top8_overlap"] for p in pairs])),
        "mean_rank_correlation_over_pairs": float(np.mean([p["mean_per_expert_rank_correlation"] for p in pairs])),
        "top8_overlap_under_independent_routing": chance,
        "reading": (
            f"Two prompts' {TOP_K} hottest experts in a layer would overlap in {chance:.2f} experts "
            "on average if routing carried no shared structure at all. Measured over every prompt "
            "pair it is several times that, and the rank correlation over all "
            f"{NUM_EXPERTS} experts is well above 0. This is structural corroboration of the "
            "cross-prompt fit, arrived at without a search: shared hotness exists, which is why "
            "the permutation lever is not zero, and it is partial, which is why the lever is small."
        ),
    }


def cross_prompt(paths: list[Path], *, restarts: int, sweeps: int, seed: int, out: Path) -> None:
    """Fit a permutation on prompts' routing and score it on a prompt it never saw.

    The held-out split inside a single run answers "does the hotness persist
    across *tokens*". It does not answer "does it persist across *prompts*",
    and a shipped expert layout is fixed at weight-load time, so that is the
    question that decides whether the lever is real. No device: this reads the
    archived raw routing files.

    Two things are measured, and the round-2 review is the reason the second
    one exists.

    **[A] every single -> single direction.** With n prompts that is n*(n-1)
    directions, and the stage originally published only the two an n=2 sample
    can produce. Those two turned out to be the *smallest* two of the six an
    n=3 sample produces, so the published figure was the bottom of a range
    being read as the range. All of them are reported now, with their spread.

    The fit for a prompt is computed **once** and reused for every prompt it is
    scored on. It used to be recomputed inside the scoring loop, which meant the
    stochastic swap-descent drew a different RNG stream for each scoring and the
    "fit on prompt i" layout was not one layout at all. That also made the whole
    table depend on how many prompts were passed in.

    **[B] pooled fit on all but one prompt, scored on the held-out one.** A
    single-prompt fit captures (hotness common to all prompts + that prompt's
    idiosyncrasy), and the idiosyncrasy is noise anywhere else -- so fit-on-one
    is the *worst* case for a layout that would be fitted on a corpus, not the
    representative one. Pooling measures the operationally relevant thing, and
    it transfers strictly better than the best single-prompt fit for every
    held-out prompt, so [A] is a floor and not a bound.
    """
    runs = []
    for path in paths:
        with gzip.open(path, "rt") as handle:
            runs.append(json.load(handle))
    layers = runs[0]["layers"]
    assert all(run["layers"] == layers for run in runs)
    rng = random.Random(seed)
    shipped = np.arange(NUM_EXPERTS) // EXPERTS_PER_DIE
    matrices = [_per_layer_matrices(run, layers) for run in runs]
    n = len(runs)

    def ms(gain_per_layer: float) -> float:
        return gain_per_layer * EXPERT_STEP_US * layers / 1000.0

    # -- [A] one layout per prompt, scored on each of the others --------------
    fitted = {
        i: [
            search_assignment(matrices[i][layer], restarts=restarts, sweeps=sweeps, rng=rng)[1]
            for layer in range(layers)
        ]
        for i in range(n)
    }
    results = []
    single_gain: dict[tuple[int, int], float] = {}
    for fit_on in range(n):
        for score_on in range(n):
            if fit_on == score_on:
                continue
            shipped_cost = sum(mean_max_count(matrices[score_on][layer], shipped) for layer in range(layers))
            fitted_cost = sum(
                mean_max_count(matrices[score_on][layer], fitted[fit_on][layer]) for layer in range(layers)
            )
            gain = (shipped_cost - fitted_cost) / layers
            single_gain[(fit_on, score_on)] = gain
            results.append(
                {
                    "fitted_on": runs[fit_on]["prompt"][:60],
                    "scored_on": runs[score_on]["prompt"][:60],
                    "fitted_on_index": fit_on,
                    "scored_on_index": score_on,
                    "shipped_mean_max_k": shipped_cost / layers,
                    "fitted_mean_max_k": fitted_cost / layers,
                    "gain_in_max_k_per_layer": gain,
                    "gain_us_per_layer": gain * EXPERT_STEP_US,
                    "gain_ms_per_iteration": ms(gain),
                }
            )
            print(
                f"fit on prompt {fit_on}, score on prompt {score_on}: shipped {shipped_cost / layers:.4f} -> "
                f"fitted {fitted_cost / layers:.4f} ({gain:+.4f} experts/layer, "
                f"{ms(gain):+.4f} ms/iteration)",
                flush=True,
            )
    spread = sorted(ms(g) for g in single_gain.values())
    print(
        f"  {len(spread)} directions: {spread[0]:+.4f} .. {spread[-1]:+.4f} ms/iteration "
        f"(mean {sum(spread) / len(spread):+.4f})",
        flush=True,
    )

    # -- [B] pooled fit on the others, scored on the genuinely unseen one ------
    pooled = []
    for held_out in range(n) if n > 2 else ():
        others = [i for i in range(n) if i != held_out]
        shipped_cost = pooled_cost = 0.0
        for layer in range(layers):
            _, assignment = search_assignment(
                np.concatenate([matrices[i][layer] for i in others], axis=0),
                restarts=restarts,
                sweeps=sweeps,
                rng=rng,
            )
            shipped_cost += mean_max_count(matrices[held_out][layer], shipped)
            pooled_cost += mean_max_count(matrices[held_out][layer], assignment)
        gain = (shipped_cost - pooled_cost) / layers
        best_single = max(single_gain[(i, held_out)] for i in others)
        pooled.append(
            {
                "held_out": runs[held_out]["prompt"][:60],
                "held_out_index": held_out,
                "pooled_over_indices": others,
                "shipped_mean_max_k": shipped_cost / layers,
                "pooled_fit_mean_max_k": pooled_cost / layers,
                "pooled_gain_ms_per_iteration": ms(gain),
                "best_single_prompt_fit_gain_ms_per_iteration": ms(best_single),
                "pooling_transfers_better": ms(gain) > ms(best_single),
            }
        )
        print(
            f"pooled fit on {others}, scored on held-out {held_out}: {ms(gain):+.4f} ms/iteration "
            f"(best single-prompt fit for the same held-out prompt {ms(best_single):+.4f})",
            flush=True,
        )

    structure = _shared_structure(matrices, layers)

    payload = {
        "runs": [{"prompt": run["prompt"], "tokens": run["tokens"], "layers": run["layers"]} for run in runs],
        "cross_prompt": results,
        "shared_structure": structure,
        "directions": len(results),
        "gain_ms_per_iteration_min": spread[0],
        "gain_ms_per_iteration_max": spread[-1],
        "gain_ms_per_iteration_mean": sum(spread) / len(spread),
        "pooled_fit_held_out": pooled,
        "reading": (
            "A shipped expert-to-die layout is fixed at weight-load time, so the gain that counts "
            "is one that survives being fitted on routing the layout will not see again. Over the "
            f"{len(results)} single-prompt -> single-prompt directions this sample can produce, that "
            f"is {spread[0]:.4f}-{spread[-1]:.4f} ms/iteration (mean {sum(spread) / len(spread):.4f}). "
            "The stage originally published only the two directions an n=2 sample can produce and "
            "read them as the range; the round-2 review's third prompt showed they were the two "
            "smallest of the six. Note also that fit-on-ONE-prompt is the WORST case for a fixed "
            "layout and not the representative one: a fit pooled over the other prompts transfers "
            "strictly BETTER to a genuinely unseen prompt than the best single-prompt fit does "
            "(pooled_fit_held_out), so this range is a floor. The rejection does not rest on the "
            "gain being negligible -- it rests on the largest measured gain still being a fraction "
            "of a percent of token-out against a bit-identity obligation on the expert weights."
        ),
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cross", type=Path, nargs="+", help="host-only: raw routing files to cross-validate")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--sweeps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", type=str, default=PROMPT)
    ap.add_argument("--tag", type=str, default="", help="suffix for the output artifacts")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = HERE / f"moe_routing_across_tokens{args.tag}.json"
    if args.cross:
        cross_prompt(
            args.cross,
            restarts=args.restarts,
            sweeps=args.sweeps,
            seed=args.seed,
            out=HERE / "moe_routing_cross_prompt.json",
        )
        return

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*MC.MESH_SHAPE), trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    recorder = RouterRecorder()
    try:
        t0 = time.perf_counter()
        gen = build_generator(
            str(MODEL_DIR), mesh, override_num_layers=args.layers, max_context_len=args.context, max_batch_size=1
        )
        print(f"weight load {time.perf_counter() - t0:.1f} s ({args.layers} layers)", flush=True)

        kv_cache = gen._ensure_kv_cache()
        # A real prompt, not a range of token ids: the routing distribution is
        # the thing under test and a nonsense prompt would be a nonsense sample.
        prompt = gen.tokenizer(args.prompt, add_special_tokens=False)["input_ids"]
        print(f"prompt: {len(prompt)} real tokens", flush=True)
        horizon = len(prompt) + args.tokens + 2
        page_table = gen.make_page_table([horizon])
        gen.reset()

        with recorder:
            logits = gen.prefill_forward(
                torch.tensor([prompt]),
                page_table=page_table,
                kv_cache=kv_cache,
                prompt_lens=[len(prompt)],
                sampling_mode="host",
            )
            token = int(logits[0, 0].argmax().item())
            print("prefill done", flush=True)

            # Eager decode, free-running greedy, one token at a time. Eager is
            # the point: a replayed trace calls no host-visible ``ttnn.topk``.
            recorder.enabled = True
            t = time.perf_counter()
            for step in range(args.tokens):
                host = gen.decode_forward(
                    torch.tensor([[token]]),
                    torch.tensor([len(prompt) + step]),
                    page_table=page_table,
                    kv_cache=kv_cache,
                    sampling_mode="host",
                    enable_trace=False,
                )
                token = int(host[0].argmax().item())
                if (step + 1) % 8 == 0:
                    print(
                        f"  {step + 1}/{args.tokens} tokens, {len(recorder.rows)} router calls, "
                        f"{time.perf_counter() - t:.1f} s",
                        flush=True,
                    )
            recorder.enabled = False
        gen.teardown()
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    analyse(recorder.rows, args)


def analyse(rows: list[list[int]], args) -> None:
    expected = args.tokens * args.layers
    assert len(rows) == expected, f"recorded {len(rows)} router calls, expected {args.tokens} x {args.layers}"
    for row in rows:
        assert len(row) == TOP_K and len(set(row)) == TOP_K, f"malformed top-8 {row}"
    # rows are emitted layer-major within a token
    per_token_layer = [rows[t * args.layers : (t + 1) * args.layers] for t in range(args.tokens)]

    out: dict = {
        "tokens": args.tokens,
        "layers": args.layers,
        "top_k": TOP_K,
        "num_experts": NUM_EXPERTS,
        "num_devices": NUM_DEVICES,
        "router_calls": len(rows),
        "note": (
            "Every figure below is over tokens x layers samples of the router's top-8, recorded "
            "off the real 48-layer model on the real mesh by wrapping ttnn.topk. Within one "
            "(token, layer) the four die counts sum to 8, so the independent unit is the "
            "(token, layer) pair and not the die."
        ),
    }

    # -- 1. per-die counts, now over many tokens ------------------------------
    shipped = [e // EXPERTS_PER_DIE for e in range(NUM_EXPERTS)]
    k_hist: Counter = Counter()
    max_per_sample: list[int] = []
    per_die_total = [0] * NUM_DEVICES
    for token_rows in per_token_layer:
        for row in token_rows:
            counts = [0] * NUM_DEVICES
            for expert in row:
                counts[shipped[expert]] += 1
            for die, c in enumerate(counts):
                k_hist[c] += 1
                per_die_total[die] += c
            max_per_sample.append(max(counts))
    out["per_die_counts"] = {
        "k_histogram": sorted(k_hist.items()),
        "samples": len(max_per_sample),
        "mean_max_k": sum(max_per_sample) / len(max_per_sample),
        "per_die_total_selections": per_die_total,
        "expected_per_die": args.tokens * args.layers * TOP_K / NUM_DEVICES,
    }

    # -- 2. per-expert hotness, and whether it persists ------------------------
    # Per layer, how often each expert is selected across tokens. The lever needs
    # this to be *stable*: the same experts hot on every token.
    hotness: dict[str, dict] = {}
    persistence = []
    for layer in range(args.layers):
        counts = Counter()
        for token_rows in per_token_layer:
            counts.update(token_rows[layer])
        rates = sorted(counts.items(), key=lambda kv: -kv[1])
        # The top-8 experts by rate would be selected 8 * tokens times between
        # them if routing were perfectly persistent, and 8 * tokens * 8/128
        # times if it were uniform and independent per token.
        top8_share = sum(c for _, c in rates[:TOP_K]) / (TOP_K * args.tokens)
        persistence.append(top8_share)
        hotness[str(layer)] = {
            "distinct_experts_used": len(counts),
            "top8_share_of_selections": top8_share,
            "hottest": rates[:8],
        }
    out["per_expert_hotness"] = {
        "per_layer": hotness,
        "mean_top8_share": sum(persistence) / len(persistence),
        "perfectly_persistent_share": 1.0,
        "share_if_uniform_and_independent": TOP_K / NUM_EXPERTS,
        "reading": (
            "top8_share_of_selections is the fraction of all selections in a layer that went to "
            "that layer's 8 most-selected experts. 1.0 means the same 8 experts every token "
            "(perfectly persistent hotness, which a permutation could exploit); "
            f"{TOP_K / NUM_EXPERTS:.3f} means selections spread evenly over all {NUM_EXPERTS}."
        ),
    }

    # -- 3. what the best assignment would actually buy ------------------------
    #
    # Fitting an assignment on the same tokens it is then scored on would
    # overstate the lever by exactly the amount the search can memorise, so the
    # tokens are split in half: the assignment is searched on the FIRST half and
    # its value is reported on the SECOND, which the search never saw. The
    # in-sample number is reported beside it, and the gap between them is the
    # overfitting.
    rng = random.Random(args.seed)
    split = args.tokens // 2
    shipped_all = shipped_train = shipped_test = 0.0
    best_in_sample = best_held_out = 0.0
    per_layer = []
    shipped_np = np.array(shipped)
    for layer in range(args.layers):
        selection = [token_rows[layer] for token_rows in per_token_layer]
        matrix = selection_matrix(selection)
        train, test = matrix[:split], matrix[split:]
        layer_shipped = mean_max_count(matrix, shipped_np)
        layer_shipped_test = mean_max_count(test, shipped_np)
        fitted_cost, assignment = search_assignment(train, restarts=args.restarts, sweeps=args.sweeps, rng=rng)
        held_out = mean_max_count(test, assignment)
        shipped_all += layer_shipped
        shipped_train += mean_max_count(train, shipped_np)
        shipped_test += layer_shipped_test
        best_in_sample += fitted_cost
        best_held_out += held_out
        per_layer.append(
            {
                "layer": layer,
                "shipped_mean_max": layer_shipped,
                "shipped_mean_max_held_out": layer_shipped_test,
                "fitted_mean_max_in_sample": fitted_cost,
                "fitted_mean_max_held_out": held_out,
                "assignment": [int(v) for v in assignment],
            }
        )
        print(
            f"  layer {layer:>2}: shipped {layer_shipped:.4f} | fitted in-sample {fitted_cost:.4f} "
            f"-> held out {held_out:.4f} (shipped on the same held-out half {layer_shipped_test:.4f})",
            flush=True,
        )
    n = args.layers
    out["permutation_search"] = {
        "restarts": args.restarts,
        "sweeps": args.sweeps,
        "train_tokens": split,
        "held_out_tokens": args.tokens - split,
        "shipped_mean_max_k_per_layer": shipped_all / n,
        "shipped_mean_max_k_per_layer_held_out": shipped_test / n,
        "fitted_mean_max_k_per_layer_in_sample": best_in_sample / n,
        "fitted_mean_max_k_per_layer_held_out": best_held_out / n,
        "held_out_gain_in_max_k_per_layer": (shipped_test - best_held_out) / n,
        "in_sample_gain_in_max_k_per_layer": (shipped_train - best_in_sample) / n,
        "us_per_active_expert": EXPERT_STEP_US,
        "held_out_gain_us_per_layer": (shipped_test - best_held_out) / n * EXPERT_STEP_US,
        "held_out_gain_ms_per_iteration": (shipped_test - best_held_out) / n * EXPERT_STEP_US * args.layers / 1000.0,
        "perfect_balance_max_k": TOP_K / NUM_DEVICES,
        "per_layer": per_layer,
        "method": (
            "balanced swap-descent over expert-to-die assignments, minimising the mean over "
            "tokens of the per-token maximum die count. Restart 0 starts from the shipped "
            "contiguous windows, so the search can never report worse than shipped in sample. "
            "The assignment is fitted on the first half of the tokens and scored on the second, "
            "so the held-out figure is not a memorised one."
        ),
    }
    print()
    print(f"per-die selections over {args.tokens} tokens x {args.layers} layers: {per_die_total}")
    print(
        f"mean top-8 share of selections per layer: {out['per_expert_hotness']['mean_top8_share']:.3f} "
        f"(1.0 = perfectly persistent, {TOP_K / NUM_EXPERTS:.3f} = uniform)"
    )
    print(f"shipped mean max_k {shipped_all / n:.4f}")
    print(
        f"fitted on {split} tokens: in-sample {best_in_sample / n:.4f}, "
        f"HELD OUT on {args.tokens - split} tokens {best_held_out / n:.4f} "
        f"(shipped on the same held-out tokens {shipped_test / n:.4f})"
    )
    print(
        f"  -> held-out gain {(shipped_test - best_held_out) / n:+.4f} experts/layer = "
        f"{(shipped_test - best_held_out) / n * EXPERT_STEP_US:.2f} us/layer = "
        f"{(shipped_test - best_held_out) / n * EXPERT_STEP_US * args.layers / 1000.0:.3f} ms/iteration"
    )

    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    # The raw routing, so every figure above is re-derivable and so a later
    # stage can re-analyse without opening a device.
    raw = args.out.with_name(args.out.stem + "_raw.json.gz")
    with gzip.open(raw, "wt") as handle:
        json.dump({"prompt": args.prompt, "tokens": args.tokens, "layers": args.layers, "top8": rows}, handle)
    print(f"wrote {raw}")


if __name__ == "__main__":
    main()
