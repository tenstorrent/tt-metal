# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Needle-in-a-haystack prompt builder for the GLM-5.3-Flash long-context smokes (host only).

``build_needle_prompt(tokenizer, n_tokens, depth, needle, question)`` returns the token ids of one chat
turn (the model's chat template, generation prompt appended) whose user message is ``n_tokens`` worth of
varied natural filler sentences with ``needle`` inserted at the fractional ``depth`` of the filler and
``question`` at the end. The filler is drawn from sentence templates x subjects x activities x places x
time phrases with a seeded generator, so no sentence repeats and the pooled-key index is not degenerate
(repetitive text makes the indexer's scores tie). The chat template opens the answer with ``<think>``;
``no_think=True`` (default) appends ``</think>`` so the greedy continuation is the answer itself (the
template's own empty-reasoning form for assistant turns), which keeps the retrieval check short.

    python -c "from models.demos.glm53_flash.tests.needle import *; ..."   (see test_needle_smoke.py)
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

DEFAULT_NEEDLE = "The secret number for the vault is 7391."
DEFAULT_QUESTION = "What is the secret number for the vault? Reply with only the number."
DEFAULT_ANSWER = "7391"

_SUBJECTS = [
    "The lighthouse keeper",
    "A retired cartographer",
    "The village baker",
    "My neighbour's daughter",
    "The night-shift nurse",
    "An apprentice glassblower",
    "The ferry captain",
    "A travelling bookseller",
    "The orchestra's oboist",
    "Our chemistry teacher",
    "The museum's archivist",
    "A young beekeeper",
    "The station master",
    "A marathon runner from the coast",
    "The town's only locksmith",
    "A weather observer",
    "The gardener at the old monastery",
    "A violin maker",
    "The harbour pilot",
    "A doctoral student in geology",
    "The tram conductor",
    "An elderly watchmaker",
    "The festival's stage manager",
    "A seed librarian",
    "The mountain rescue volunteer",
    "A radio astronomer",
    "The canal lock operator",
    "A pastry chef on holiday",
    "The bridge engineer",
    "A botanical illustrator",
    "The schoolhouse caretaker",
    "A mapmaker's assistant",
    "The vineyard's foreman",
    "A sign-language interpreter",
    "The observatory's night guard",
    "A cellist from the valley",
    "The mill owner",
    "A junior surveyor",
    "The harbour-side florist",
    "An amateur meteorologist",
]
_ACTIVITIES = [
    "repaired the brass hinges of the old gate",
    "counted the migrating geese over the marsh",
    "sorted a crate of mismatched porcelain",
    "rehearsed a lullaby in the empty chapel",
    "measured the rainfall in a tin cup",
    "traded a basket of plums for a map",
    "tuned the piano that nobody plays",
    "sketched the ferry as it left the pier",
    "tested the new rope on the belfry pulley",
    "stacked firewood against the north wall",
    "catalogued the moths caught by the lamp",
    "mended the sail with waxed thread",
    "walked the dog along the flooded towpath",
    "baked rye bread for the harvest fair",
    "polished the lenses of the lighthouse lamp",
    "read the almanac aloud to the children",
    "replanted the lavender beside the well",
    "copied the tide tables into a notebook",
    "argued cheerfully about the price of salt",
    "carried a lantern across the frozen pond",
    "wound the clock in the council chamber",
    "taught the apprentices to splice cable",
    "photographed the frost on the greenhouse glass",
    "labelled jars of honey with brown paper",
    "swept the leaves from the observatory steps",
    "rowed across the inlet before sunrise",
    "wrote a letter that was never sent",
    "repainted the numbers on the fishing boats",
    "listened to the shortwave for the shipping news",
    "pressed wildflowers between pages of a ledger",
    "fixed the squeaking wheel of the hay cart",
    "sold hot chestnuts outside the theatre",
    "traced the old boundary stones with chalk",
    "whistled to the horses in the lower field",
    "dried apples on the warm chimney ledge",
    "compared two translations of the same poem",
    "balanced the ledger by candlelight",
    "waited for the delayed evening train",
    "restrung the beads of a broken necklace",
    "examined the cracks in the quay wall",
]
_PLACES = [
    "near the slate quarry",
    "behind the customs house",
    "in the courtyard of the inn",
    "on the far side of the estuary",
    "at the edge of the pine plantation",
    "beside the disused railway cutting",
    "under the arches of the viaduct",
    "in the shadow of the grain silo",
    "along the cliff path",
    "outside the shuttered cinema",
    "by the mooring posts",
    "in the upstairs room of the library",
    "on the terrace of the sanatorium",
    "at the crossroads above the village",
    "inside the abandoned tannery",
    "next to the war memorial",
    "on the pebble beach",
    "in the back garden of the rectory",
    "at the bus shelter by the bridge",
    "across from the cooperative dairy",
    "in the meadow below the reservoir",
    "at the foot of the water tower",
    "in the workshop behind the forge",
    "on the landing of the tenement stairs",
    "among the allotments",
    "at the fish market",
    "beneath the elm by the schoolyard",
    "in the waiting room of the clinic",
    "on the roof of the bakery",
    "at the mouth of the river",
]
_TIMES = [
    "before dawn",
    "just after the noon bell",
    "late on a Tuesday",
    "during the first snow of the year",
    "while the fog still held",
    "on the morning of the regatta",
    "as the tide turned",
    "the day the fair opened",
    "after the last ferry had gone",
    "in the week of the eclipse",
    "when the orchard was in blossom",
    "at the height of the heatwave",
    "on the eve of the election",
    "the night the power failed",
    "during the autumn gales",
    "shortly before the harvest",
    "once the floodwater had receded",
    "in the lull between two storms",
    "the afternoon the circus arrived",
    "a few hours after the storm warning",
]
_TEMPLATES = [
    "{s} {a} {p} {t}.",
    "{t_cap}, {s_low} {a} {p}.",
    "{s} {a} {t}, {p}, and nobody thought it strange.",
    "{p_cap}, {s_low} {a} {t}.",
    "It was {t} when {s_low} {a} {p}.",
    "{s} quietly {a} {p}; it was {t}.",
    "People still mention how {s_low} {a} {p} {t}.",
    "{s} {a} {p}, which was unusual {t}.",
    "According to the log, {s_low} {a} {t} {p}.",
    "{t_cap} {s_low} {a}, as usual, {p}.",
]


def _lower_first(s: str) -> str:
    return s[0].lower() + s[1:] if s and s[0].isupper() and not s.startswith("I ") else s


def filler_sentences(rng: random.Random, n: int) -> list[str]:
    """``n`` distinct natural sentences (no sentence is produced twice for one generator)."""
    out, seen = [], set()
    while len(out) < n:
        s, a, p, t = rng.choice(_SUBJECTS), rng.choice(_ACTIVITIES), rng.choice(_PLACES), rng.choice(_TIMES)
        key = (s, a, p, t)
        if key in seen:
            continue
        seen.add(key)
        tpl = rng.choice(_TEMPLATES)
        out.append(
            tpl.format(
                s=s, s_low=_lower_first(s), a=a, p=p, p_cap=p[0].upper() + p[1:], t=t, t_cap=t[0].upper() + t[1:]
            )
        )
    return out


@dataclass
class NeedlePrompt:
    ids: torch.Tensor  # [S] int64 token ids (chat template + generation prompt [+ </think>])
    n_filler: int  # filler sentences used
    needle_sentence_index: int  # the needle's index among the user message's sentences
    needle_token_start: int  # first token of the needle sentence in ``ids`` (approximate: by prefix tokenization)
    needle: str
    question: str
    answer: str | None


def _chat_ids(tokenizer, user_text: str, no_think: bool) -> list[int]:
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}], add_generation_prompt=True, tokenize=True
    )
    ids = list(ids["input_ids"]) if hasattr(ids, "keys") else list(ids)
    if no_think:
        close = tokenizer.convert_tokens_to_ids("</think>")
        assert isinstance(close, int) and close >= 0, "the tokenizer has no </think> token"
        ids.append(close)
    return ids


def build_needle_prompt_info(
    tokenizer,
    n_tokens: int,
    depth: float,
    needle: str = DEFAULT_NEEDLE,
    question: str = DEFAULT_QUESTION,
    *,
    seed: int = 0,
    no_think: bool = True,
    answer: str | None = DEFAULT_ANSWER,
) -> NeedlePrompt:
    """``NeedlePrompt`` of about ``n_tokens`` tokens (within a few tokens: the filler is trimmed by whole
    sentences, then the sentence count is adjusted so the final id count never exceeds ``n_tokens``).
    ``depth`` in [0, 1]: the needle goes after ``depth * n_filler`` filler sentences."""
    assert 0.0 <= depth <= 1.0, depth
    rng = random.Random(seed)
    overhead = len(_chat_ids(tokenizer, f"{needle}\n\n{question}", no_think))
    budget = n_tokens - overhead
    assert budget > 0, (n_tokens, overhead)
    # estimate the sentence count from the mean sentence length, then correct on the real id count
    probe = filler_sentences(rng, 64)
    mean_len = sum(len(tokenizer.encode(s + " ", add_special_tokens=False)) for s in probe) / len(probe)
    n = max(1, int(budget / mean_len))
    rng = random.Random(seed)

    def compose(n_fill: int):
        sents = filler_sentences(rng.__class__(seed), n_fill)
        k = int(round(depth * n_fill))
        body = sents[:k] + [needle] + sents[k:]
        text = " ".join(body) + "\n\n" + question
        return sents, k, text, _chat_ids(tokenizer, text, no_think)

    sents, k, text, ids = compose(n)
    # grow / shrink by whole sentences until the count is the largest one <= n_tokens
    while len(ids) > n_tokens:
        n -= max(1, (len(ids) - n_tokens) // max(1, int(mean_len)))
        sents, k, text, ids = compose(n)
    while True:
        s2, k2, t2, ids2 = compose(n + 1)
        if len(ids2) > n_tokens:
            break
        n, sents, k, text, ids = n + 1, s2, k2, t2, ids2
    # the needle's token offset: tokenize the user text up to (but not including) the needle
    prefix_text = " ".join(sents[:k]) + (" " if k else "")
    head = len(_chat_ids(tokenizer, prefix_text, no_think)) - (
        3 if no_think else 2
    )  # minus <|assistant|><think>[</think>]
    return NeedlePrompt(
        ids=torch.tensor(ids, dtype=torch.int64),
        n_filler=len(sents),
        needle_sentence_index=k,
        needle_token_start=head,
        needle=needle,
        question=question,
        answer=answer,
    )


def build_needle_prompt(tokenizer, n_tokens: int, depth: float, needle: str, question: str, **kw) -> torch.Tensor:
    """Token ids ``[S]`` int64 of the needle prompt (see ``build_needle_prompt_info``)."""
    return build_needle_prompt_info(tokenizer, n_tokens, depth, needle, question, **kw).ids


def retrieved(text: str, answer: str) -> bool:
    """Whether the decoded continuation contains the needle's answer (digits / words compared without spaces)."""
    norm = lambda s: "".join(ch for ch in s if not ch.isspace()).lower()
    return norm(answer) in norm(text)
