# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side sampling for the talker and the code predictor.

The checkpoint ships `do_sample: true` and both of its decoders sample: the talker with
temperature 0.9, top_k 50, top_p 1.0 and `repetition_penalty` 1.05, the code predictor
("subtalker") with the same temperature and top_k but no penalty. Greedy decoding is not
a conservative substitute. Upstream's own package, on CPU, greedy, runs past the end of
the text and fills the rest of its budget with a silence code: measured 699 frames of a
700 frame budget for a 4 sentence prompt, against 414 frames and a clean stop when
sampling. So this is not a device artefact and not something to work around downstream.

Logits arrive on host anyway, one row of 3072 (talker) or 2048 (predictor) per step, so
sampling here costs nothing worth measuring.

The order below is the order `transformers` applies: the penalty is a logits *processor*
and runs first, then the temperature, top_k and top_p *warpers*. Reproducing that order
matters, because dividing by the temperature before the penalty would change which
tokens survive top_k.
"""

import torch


def apply_repetition_penalty(logits, seen, penalty):
    """Divide positive logits of already-seen ids, multiply negative ones.

    That asymmetry is the upstream definition: both moves push a seen id down, whichever
    side of zero its logit is on.
    """
    if penalty == 1.0 or not len(seen):
        return logits
    index = torch.as_tensor(sorted(set(int(token) for token in seen)), dtype=torch.long)
    scores = logits[index]
    logits[index] = torch.where(scores < 0, scores * penalty, scores / penalty)
    return logits


def sample(logits, seen=(), temperature=0.9, top_k=50, top_p=1.0, penalty=1.0, generator=None, suppress=()):
    """One id from a single row of logits, matching `transformers`' processor order.

    `seen` are the ids the penalty applies to. `suppress` are ids this draw may not return,
    as `-inf` before anything else: that is where `SuppressTokensLogitsProcessor` sits, and
    a suppressed id at `-inf` cannot take one of the k places.
    """
    logits = logits.detach().float().reshape(-1).clone()
    if len(suppress):
        logits[torch.as_tensor(suppress, dtype=torch.long)] = -float("inf")
    logits = apply_repetition_penalty(logits, seen, penalty)

    if temperature and temperature != 1.0:
        logits = logits / temperature

    if top_k and 0 < top_k < logits.numel():
        floor = torch.topk(logits, top_k).values[-1]
        logits[logits < floor] = -float("inf")

    if top_p is not None and top_p < 1.0:
        ordered, order = torch.sort(logits, descending=True)
        cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
        # Keep the first token whose cumulative mass crosses top_p, as upstream does.
        drop = cumulative - torch.softmax(ordered, dim=-1) >= top_p
        logits[order[drop]] = -float("inf")

    probabilities = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probabilities, 1, generator=generator))
