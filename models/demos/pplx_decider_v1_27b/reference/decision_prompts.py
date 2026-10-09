# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic 25-row decision prompt set for the stage-6 end-to-end decision golden.

Rows use the decider's input format (autojev ``DecisionInput``: ``{"state", "question"}``),
text only. Short rows are fixed text; longer rows build their ``state`` from small templates
with a fixed seed (order logs, CSV tables, server logs, FAQ entries, review lists) and grow
the number of entries until the chat-templated prompt lands in its length band.

Tokenization reproduces the reference app (snapshot ``source/src/autojev/model.py``):
``decision_messages(row, codes)`` -> ``processor.apply_chat_template(..., tokenize=False,
add_generation_prompt=True, enable_thinking=False)`` -> ``processor(text=[text])``, batch 1,
no padding, max_length 8192. ``decision_messages``, ``options`` and ``answer`` are imported
from the snapshot's own autojev package (read-only).

CLI (prints the per-row length table, needs no weights)::

    python -m models.demos.pplx_decider_v1_27b.reference.decision_prompts
"""

from __future__ import annotations

import itertools
import json
import random
import string
import sys
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

SEED = 20261009
MAX_LENGTH = 8192
BUCKETS = (128, 1024, 2048, 4096, 8192)
# Inclusive token-length bands (after the chat template) and how many rows each must hold.
BANDS = {"short": (128, 1024), "medium": (1025, 2048), "long": (2049, 4096), "xl": (4097, 8192)}
BAND_COUNTS = {"short": 10, "medium": 6, "long": 5, "xl": 4}
DEFAULT_SNAPSHOT = Path(
    "/local/ttuser/gtobar/hf/hub/models--perplexity-ai--pplx-decider-v1-27b/snapshots/"
    "b01a5cbaca5391f73bd55103d4f27e8982cd5e60"
)


def bucket_of(seq_len: int) -> int:
    """Smallest of 128/1024/2048/4096/8192 that holds ``seq_len``."""
    for b in BUCKETS:
        if seq_len <= b:
            return b
    raise ValueError(f"seq_len {seq_len} exceeds {BUCKETS[-1]}")


def band_of(seq_len: int) -> str | None:
    for name, (lo, hi) in BANDS.items():
        if lo <= seq_len <= hi:
            return name
    return None


# ----------------------------------------------------------------------------------------------
# Reference-app tokenization (snapshot autojev package + processor, both read-only)
# ----------------------------------------------------------------------------------------------


def import_autojev(snapshot: Path = DEFAULT_SNAPSHOT):
    """Import the snapshot's ``autojev.model`` without constructing ``DecisionModel``.

    ``autojev/types.py`` uses PEP 695 ``type X = ...`` aliases (Python >= 3.12) and only holds
    type annotations. On older interpreters a stub module with the same names (as ``Any``)
    is registered first, so ``model.py`` itself is imported unchanged.
    """
    src = str(Path(snapshot) / "source" / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    if sys.version_info < (3, 12) and "autojev.types" not in sys.modules:
        import autojev  # noqa: F401  (package __init__ is a docstring only)

        stub = types.ModuleType("autojev.types")
        for name in ("Answer", "Content", "DecisionInput", "ImageInput", "JSONValue", "Question"):
            setattr(stub, name, typing.Any)
        sys.modules["autojev.types"] = stub
    import autojev.model as model

    return model


@dataclass
class AppTokenizer:
    """The reference app's prompt rendering and tokenization, batch 1, no padding."""

    snapshot: Path = DEFAULT_SNAPSHOT
    processor: object = field(init=False)
    codes: list[str] = field(init=False)
    token_ids: list[int] = field(init=False)
    temperature: float = field(init=False)

    def __post_init__(self):
        from transformers import AutoProcessor

        self.autojev = import_autojev(self.snapshot)
        self.processor = AutoProcessor.from_pretrained(str(self.snapshot), local_files_only=True)
        cfg = json.loads((Path(self.snapshot) / "decision_config.json").read_text())
        self.codes, self.token_ids, self.temperature = cfg["codes"], cfg["token_ids"], float(cfg["temperature"])

    def text(self, row: dict) -> str:
        messages = self.autojev.decision_messages(row, self.codes)
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def input_ids(self, row: dict) -> list[int]:
        encoded = self.processor(text=[self.text(row)], padding=True, return_tensors="pt")
        if int(encoded["attention_mask"].sum()) != encoded["input_ids"].shape[1]:
            raise AssertionError("batch-1 encoding must have no padding")
        return encoded["input_ids"][0].tolist()

    def count(self, row: dict) -> int:
        return len(self.autojev.options(row["question"])[0])

    def verify_answer_codes(self) -> dict:
        """Re-run the answer-code checks of ``DecisionModel.__init__`` against this tokenizer."""
        tokenizer = self.processor.tokenizer
        candidates = list(string.ascii_uppercase) + [
            "".join(pair) for pair in itertools.product(string.ascii_uppercase, repeat=2)
        ]
        codes = [c for c in candidates if len(tokenizer.encode(c, add_special_tokens=False)) == 1][:255]
        token_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in codes]
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Choose an option."}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        single_after_prefix = all(
            tokenizer.encode(prefix + c, add_special_tokens=False) == prefix_ids + [t] for c, t in zip(codes, token_ids)
        )
        result = {
            "n_codes": len(codes),
            "distinct_token_ids": len(set(token_ids)),
            "single_token_after_prefix": single_after_prefix,
            "codes_match_config": codes == self.codes,
            "token_ids_match_config": token_ids == self.token_ids,
        }
        if not (
            len(set(token_ids)) == 255 and single_after_prefix and codes == self.codes and token_ids == self.token_ids
        ):
            raise AssertionError(f"answer-code checks failed: {result}")
        return result


@dataclass
class PromptSpec:
    """One row. ``build(n)`` returns ``(DecisionInput, meta)``; ``n`` scales the state
    (entries in a generated log/table) and is ignored by fixed rows (``target is None``).
    ``meta["expected"]`` is the intended answer, used only for spot checks."""

    id: str
    band: str
    build: Callable[[int], tuple[dict, dict]]
    target: int | None = None  # token-length target for scalable rows


def fixed(row: dict, expected) -> Callable[[int], tuple[dict, dict]]:
    return lambda n: (row, {"expected": expected})


# ----------------------------------------------------------------------------------------------
# Short rows (128-1024 tokens): fixed text
# ----------------------------------------------------------------------------------------------


def short_specs() -> list[PromptSpec]:
    rows: list[PromptSpec] = []
    rows.append(
        PromptSpec(
            "s01_ticket_routing",
            "short",
            fixed(
                {
                    "state": "Subject: Charged twice for my March invoice\n\nHi team, I looked at my bank statement "
                    "this morning and the monthly subscription fee of $49.00 was taken two times on March 3. "
                    "My account email is the same as this message. Could you refund the duplicate charge? "
                    "Everything else with the product works fine. Thanks, Dana",
                    "question": {
                        "type": "choice",
                        "instructions": "Which team should handle this request?",
                        "criteria": {
                            "billing": "Charges, invoices and refunds",
                            "technical_support": "Errors, bugs and integration problems",
                            "sales": "Questions about buying or upgrading a plan",
                        },
                    },
                },
                "billing",
            ),
        )
    )
    rows.append(
        PromptSpec(
            "s02_review_positive",
            "short",
            fixed(
                {
                    "state": "Product: 1.7 L electric kettle (stainless steel)\nRating left blank.\nReview: I have used "
                    "this kettle every morning for three months. It boils a full jug in about four minutes, the "
                    "lid opens with one hand and the auto shut-off has never failed. It is quieter than my old "
                    "one and easy to descale with a little vinegar. Would buy again and already recommended it "
                    "to my sister.",
                    "question": {"type": "noul", "instructions": "Is this product review positive overall?"},
                },
                "true",
            ),
        )
    )
    rows.append(
        PromptSpec(
            "s03_restaurant_stars",
            "short",
            fixed(
                {
                    "state": "We went for dinner on Friday. The table was not ready when we arrived and we waited "
                    "35 minutes despite a reservation. The soup arrived lukewarm and the pasta was overcooked. "
                    "The waiter was friendly and apologised, and the dessert was decent, but for the price we "
                    "expected much better. We probably will not return.",
                    "question": {
                        "type": "score",
                        "instructions": "How satisfied is the reviewer with the restaurant?",
                        "criteria": [
                            "1 star: very dissatisfied",
                            "2 stars: dissatisfied",
                            "3 stars: neutral",
                            "4 stars: satisfied",
                            "5 stars: very satisfied",
                        ],
                    },
                },
                "1",
            ),
        )
    )
    rows.append(
        PromptSpec(
            "s04_recipe_cuisine",
            "short",
            fixed(
                {
                    "state": "Recipe card\nIngredients: 2 cups short-grain rice, 3 tbsp rice vinegar, 1 tbsp sugar, "
                    "1 tsp salt, 4 sheets nori, 200 g fresh salmon, 1 cucumber, 1 avocado, soy sauce, wasabi, "
                    "pickled ginger.\nSteps: Rinse and cook the rice, season it with the vinegar mixture and let "
                    "it cool. Place nori on a bamboo mat, spread rice, add strips of fish and vegetables, roll "
                    "tightly and slice into eight pieces. Serve with soy sauce, wasabi and ginger.",
                    "question": {
                        "type": "choice",
                        "instructions": "Which cuisine does this recipe belong to?",
                        "criteria": {"italian": None, "japanese": None, "mexican": None, "indian": None},
                    },
                },
                "japanese",
            ),
        )
    )
    rows.append(
        PromptSpec(
            "s05_unit_conversion",
            "short",
            fixed(
                {
                    "state": "Training log for the week. Monday: easy run of 2.5 km around the park. Wednesday: "
                    "intervals on the track. Friday: rest. Sunday: long run. The coach asked every runner to "
                    "report the Monday distance in meters on the team spreadsheet.",
                    "question": {
                        "type": "choice",
                        "instructions": "What is the Monday distance expressed in meters?",
                        "criteria": {
                            "25 m": None,
                            "250 m": None,
                            "2,500 m": None,
                            "25,000 m": None,
                            "2,050 m": None,
                            "2.5 m": None,
                        },
                    },
                },
                "2,500 m",
            ),
        )
    )
    rows.extend(short_specs_2())
    return rows


def short_specs_2() -> list[PromptSpec]:
    rows: list[PromptSpec] = []
    rows.append(
        PromptSpec(
            "s06_math_check",
            "short",
            fixed(
                {
                    "state": "Worksheet, question 4. A student multiplied two numbers by hand and wrote the "
                    "following line in the answer box:\n\n17 x 23 = 391\n\nThe teacher wants to mark each answer "
                    "box as correct or incorrect before returning the worksheets on Monday.",
                    "question": {
                        "type": "noul",
                        "instructions": "Is the student's multiplication correct?",
                        "criteria": {"true": "The product is correct", "false": "The product is wrong"},
                    },
                },
                "true",
            ),
        )
    )
    rows.append(
        PromptSpec(
            "s07_language_id",
            "short",
            fixed(
                {
                    "state": 'Message received through the hotel contact form:\n\n"Guten Morgen! Wir kommen am '
                    "Freitag gegen 18 Uhr an. Gibt es einen Parkplatz in der Nähe des Hotels? Vielen Dank und "
                    'bis bald, Familie Weber"',
                    "question": {
                        "type": "choice",
                        "instructions": "In which language is the message written?",
                        "criteria": {
                            "en": "English",
                            "de": "German",
                            "fr": "French",
                            "es": "Spanish",
                            "it": "Italian",
                        },
                    },
                },
                "de",
            ),
        )
    )
    airports = {
        "LHR": "London Heathrow",
        "CDG": "Paris Charles de Gaulle",
        "FRA": "Frankfurt",
        "AMS": "Amsterdam Schiphol",
        "MAD": "Madrid Barajas",
        "FCO": "Rome Fiumicino",
        "LIS": "Lisbon",
        "DUB": "Dublin",
        "CPH": "Copenhagen",
        "VIE": "Vienna",
        "ZRH": "Zurich",
        "ARN": "Stockholm Arlanda",
    }
    rows.append(
        PromptSpec(
            "s08_itinerary_airport",
            "short",
            fixed(
                {
                    "state": "Itinerary summary\nDay 1: Train from Porto to the capital, check in near the river.\n"
                    "Day 2: Walking tour of Alfama and Belem, tram 28 in the afternoon.\nDay 3: Day trip to "
                    "Sintra.\nDay 4: Return flight home departing from the capital's main airport at 09:40.",
                    "question": {
                        "type": "choice",
                        "instructions": "Which airport does the return flight depart from?",
                        "criteria": airports,
                    },
                },
                "LIS",
            ),
        )
    )
    rows.append(
        PromptSpec(
            "s09_bug_severity",
            "short",
            fixed(
                {
                    "state": "Bug report #4471\nTitle: Tooltip text on the settings page is misaligned by a few "
                    "pixels\nEnvironment: Desktop app 3.2.1, Windows 11\nSteps: Open Settings, hover over the "
                    "'Theme' label.\nExpected: Tooltip centered under the label.\nActual: Tooltip appears about "
                    "4 px to the right. All functions work normally; purely cosmetic.",
                    "question": {
                        "type": "score",
                        "instructions": "How severe is this bug?",
                        "criteria": ["low: cosmetic or minor", "medium: degraded feature", "high: data loss or outage"],
                    },
                },
                "0",
            ),
        )
    )
    rows.append(
        PromptSpec(
            "s10_match_winner",
            "short",
            fixed(
                {
                    "state": "Final score, regional basketball league, round 12\nRiverside Hawks 88 - 94 Hilltop "
                    "Foxes\nQuarter scores: Hawks 22-19-25-22, Foxes 20-26-24-24\nTop scorer: J. Moreno (Foxes) "
                    "with 27 points. Attendance 3,150.",
                    "question": {
                        "type": "choice",
                        "instructions": "Which team won the match?",
                        "criteria": {"Riverside Hawks": None, "Hilltop Foxes": None},
                    },
                },
                "Hilltop Foxes",
            ),
        )
    )
    return rows


# ----------------------------------------------------------------------------------------------
# Small template vocabularies and generators for the scalable rows (fixed seeds)
# ----------------------------------------------------------------------------------------------

COLORS = ["red", "blue", "green", "black", "white", "grey", "navy", "olive", "yellow", "orange", "teal", "beige"]
ITEMS = [
    "t-shirt",
    "hoodie",
    "rain jacket",
    "running shoe",
    "backpack",
    "water bottle",
    "desk lamp",
    "coffee mug",
    "yoga mat",
    "umbrella",
    "wool scarf",
    "travel pillow",
]
MATERIALS = ["cotton", "recycled polyester", "canvas", "steel", "bamboo", "ceramic"]
CITIES = [
    "Austin",
    "Denver",
    "Portland",
    "Boston",
    "Chicago",
    "Seattle",
    "Phoenix",
    "Atlanta",
    "Dallas",
    "Miami",
    "Detroit",
    "Raleigh",
    "Tampa",
    "Omaha",
    "Tucson",
    "Madison",
]
FIRST = [
    "Ana",
    "Ben",
    "Carla",
    "David",
    "Elena",
    "Farid",
    "Grace",
    "Hugo",
    "Ines",
    "Jonas",
    "Kira",
    "Liam",
    "Maya",
    "Nora",
    "Omar",
    "Priya",
    "Quinn",
    "Rosa",
    "Sven",
    "Tara",
    "Umar",
    "Vera",
    "Wen",
    "Yusuf",
    "Zoe",
]
LAST = [
    "Alvarez",
    "Brooks",
    "Chen",
    "Dubois",
    "Evans",
    "Fischer",
    "Garcia",
    "Hansen",
    "Ito",
    "Jensen",
    "Kowalski",
    "Larsen",
    "Moreau",
    "Nguyen",
    "Okafor",
    "Patel",
]
STATUSES = ["processing", "shipped", "delivered", "in transit"]
SERVICES = ["auth", "catalog", "payments", "search", "notifications"]
ENDPOINTS = ["/api/login", "/api/items", "/api/cart", "/api/checkout", "/api/search", "/api/profile"]


def rng_for(name: str) -> random.Random:
    return random.Random(f"{SEED}:{name}")


def person(rng: random.Random) -> str:
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def order_csv(rng: random.Random, n: int, *, forced: dict[int, list[str]] | None = None) -> str:
    """CSV of ``n`` orders. ``forced`` maps a row index to a full replacement row."""
    lines = ["order_id,date,customer,city,item,qty,total_usd,status"]
    for i in range(n):
        if forced and i in forced:
            lines.append(",".join(forced[i]))
            continue
        oid = 3000 + i * 7 + rng.randint(0, 6)
        date = f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        item = f"{rng.choice(COLORS)} {rng.choice(ITEMS)}"
        qty = rng.randint(1, 4)
        lines.append(
            f"{oid},{date},{person(rng)},{rng.choice(CITIES)},{item},{qty},{qty * rng.randint(9, 89)}.00,"
            f"{rng.choice(STATUSES)}"
        )
    return "\n".join(lines)


def server_log(rng: random.Random, n: int, *, error_weights: dict[str, float], error_rate: float) -> list[str]:
    """``n`` log lines; a line is an ERROR with probability ``error_rate``, its service drawn by weight."""
    services, weights = list(error_weights), list(error_weights.values())
    out = []
    for i in range(n):
        ts = f"2024-06-14T{8 + i // 3600 % 12:02d}:{i // 60 % 60:02d}:{i % 60:02d}Z"
        if rng.random() < error_rate:
            svc = rng.choices(services, weights)[0]
            msg = rng.choice(
                [
                    "upstream timeout after 3000 ms",
                    "connection refused by db-primary",
                    "unhandled exception in request handler",
                    "queue publish failed, retrying",
                ]
            )
            out.append(f"{ts} ERROR [{svc}] {msg}")
        else:
            svc = rng.choice(SERVICES)
            ep = rng.choice(ENDPOINTS)
            out.append(f"{ts} INFO  [{svc}] {rng.choice(['GET', 'POST'])} {ep} 200 {rng.randint(8, 240)} ms")
    return out


REVIEW_BITS = {
    5: ["Absolutely love it.", "Exceeded my expectations.", "Works perfectly every day.", "Best purchase this year."],
    4: [
        "Very good overall.",
        "Does the job well with a small quirk.",
        "Solid quality for the price.",
        "Happy with it, minor packaging dent.",
    ],
    3: ["It is okay.", "Average, nothing special.", "Some good points and some bad.", "Fine but not great."],
    2: [
        "Disappointed with the quality.",
        "Stopped working properly after a week.",
        "Not worth the price.",
        "Customer service was slow to answer.",
    ],
    1: [
        "Broke on the first day.",
        "Completely useless, returned it.",
        "Terrible experience overall.",
        "Arrived damaged and did not work.",
    ],
}


def review_list(rng: random.Random, n: int, weights: dict[int, float], product: str) -> str:
    """``n`` short reviews of ``product`` (no star numbers shown); sentiment drawn by ``weights``."""
    levels, w = list(weights), list(weights.values())
    lines = []
    for i in range(n):
        lvl = rng.choices(levels, w)[0]
        a, b = rng.sample(REVIEW_BITS[lvl], 2)
        lines.append(f"Review {i + 1} by {person(rng)} ({rng.choice(CITIES)}): {a} {b} ({product})")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
# Medium rows (1025-2048 tokens)
# ----------------------------------------------------------------------------------------------

ACCOUNT_NOTES = [
    "updated the newsletter preferences",
    "changed the account password",
    "added a second shipping " "address",
    "asked about gift wrapping",
    "joined the loyalty program",
    "redeemed a $10 voucher",
    "set the default currency to USD",
    "confirmed the phone number",
    "downloaded an invoice copy",
]


def account_notes(rng: random.Random, n: int) -> str:
    return "\n".join(
        f"Note {i + 1} (2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}): customer {rng.choice(ACCOUNT_NOTES)}."
        for i in range(n)
    )


def build_sku_lookup(n: int) -> tuple[dict, dict]:
    rng = rng_for("m01")
    combos = rng.sample([(c, m, it) for c in COLORS for m in MATERIALS for it in ITEMS], 120)
    criteria = {f"SKU-{1000 + k}": f"{c} {m} {it}" for k, (c, m, it) in enumerate(combos)}
    target_key = f"SKU-{1000 + 77}"
    c, m, it = combos[77]
    notes = account_notes(rng_for("m01-notes"), n)
    state = (
        f"Customer message: Hello, last week I bought the {c} {it} made of {m}. It fits well but the stitching "
        f"came loose after two washes. I would like an exchange for the same product.\n\nAccount notes:\n{notes}"
    )
    question = {
        "type": "choice",
        "instructions": "Which catalog product is the customer talking about?",
        "criteria": criteria,
    }
    return {"state": state, "question": question}, {"expected": target_key}


SUPPORT_CATEGORIES = {
    "shipping_delivery": "Late, lost or damaged deliveries and tracking",
    "returns_refunds": "Returning items and getting money back",
    "account_access": "Login, password and account settings",
    "billing": "Charges, invoices and payment methods",
    "product_question": "How a product works or its specifications",
    "technical_issue": "Website or app errors",
    "feedback": "General praise or suggestions",
    "other": "Anything else",
}


def build_support_thread(n: int) -> tuple[dict, dict]:
    rng = rng_for("m02")
    turns = [
        "Customer: My parcel was supposed to arrive on Tuesday and the tracking page has said 'in transit' "
        "for six days now. Order 58120."
    ]
    agent = [
        "Agent: Thanks for reaching out, I am checking the carrier status for order 58120 now.",
        "Agent: The carrier shows the parcel at the regional hub; I have opened a trace request.",
        "Agent: I am sorry for the delay. I will update you as soon as the carrier replies.",
    ]
    customer = [
        "Customer: Thanks. Is there any new information from the carrier today?",
        "Customer: The tracking page still has not changed, the parcel is still not here.",
        "Customer: I need these items for a trip next week, so the delivery date matters.",
    ]
    for i in range(n):
        turns.append(rng.choice(agent if i % 2 == 0 else customer))
    state = "Support thread (newest last)\n" + "\n".join(turns)
    question = {
        "type": "choice",
        "instructions": "Which support category fits this thread?",
        "criteria": SUPPORT_CATEGORIES,
    }
    return {"state": state, "question": question}, {"expected": "shipping_delivery"}


def build_orders_refunded(n: int) -> tuple[dict, dict]:
    table = order_csv(rng_for("m03"), n)
    assert "refunded" not in table
    state = "Export of the order table, Q2 report:\n" + table
    question = {"type": "noul", "instructions": "Does the table contain any order whose status is 'refunded'?"}
    return {"state": state, "question": question}, {"expected": "false"}


STAR_LEVELS = [
    "1 star: very dissatisfied",
    "2 stars: dissatisfied",
    "3 stars: neutral",
    "4 stars: satisfied",
    "5 stars: very satisfied",
]


def build_reviews_positive(n: int) -> tuple[dict, dict]:
    reviews = review_list(rng_for("m04"), n, {5: 0.5, 4: 0.35, 3: 0.1, 2: 0.05}, "desk lamp")
    state = "Recent customer reviews for the adjustable LED desk lamp:\n" + reviews
    question = {
        "type": "score",
        "instructions": "Overall, how satisfied are these customers with the desk lamp?",
        "criteria": STAR_LEVELS,
    }
    return {"state": state, "question": question}, {"expected": "3 or 4"}


STORES = [f"{city} {suffix}" for city in CITIES[:15] for suffix in ("Downtown", "Mall")]  # 30 stores


def build_store_receipt(n: int) -> tuple[dict, dict]:
    rng = rng_for("m05")
    hit = n // 2
    lines = ["receipt,store,date,items,total_usd"]
    for i in range(n):
        if i == hit:
            lines.append(f"R-50917,{STORES[11]},2024-05-18,3,64.50")
            continue
        lines.append(
            f"R-{40000 + i * 13 + rng.randint(0, 12)},{rng.choice(STORES)},2024-05-"
            f"{rng.randint(1, 31):02d},{rng.randint(1, 9)},{rng.randint(5, 300)}.{rng.randint(0, 99):02d}"
        )
    state = "Point-of-sale export, May 2024:\n" + "\n".join(lines)
    question = {
        "type": "choice",
        "instructions": "Which store processed receipt R-50917?",
        "criteria": {s: None for s in STORES},
    }
    return {"state": state, "question": question}, {"expected": STORES[11]}


def build_server_500(n: int) -> tuple[dict, dict]:
    log = server_log(rng_for("m06"), n, error_weights={"auth": 1.0}, error_rate=0.0)
    log.insert(n // 3, "2024-06-14T08:00:41Z WARN  [catalog] GET /api/items 500 1502 ms internal server error")
    state = "Access log excerpt for the web service:\n" + "\n".join(log)
    question = {
        "type": "noul",
        "instructions": "Did the service return any HTTP 500 (internal server error) response in this log?",
    }
    return {"state": state, "question": question}, {"expected": "true"}


def medium_specs() -> list[PromptSpec]:
    return [
        PromptSpec("m01_sku_lookup_120opt", "medium", build_sku_lookup, 1700),
        PromptSpec("m02_support_thread", "medium", build_support_thread, 1200),
        PromptSpec("m03_orders_refunded", "medium", build_orders_refunded, 1500),
        PromptSpec("m04_reviews_positive", "medium", build_reviews_positive, 1650),
        PromptSpec("m05_store_receipt", "medium", build_store_receipt, 1900),
        PromptSpec("m06_server_500", "medium", build_server_500, 2000),
    ]


# ----------------------------------------------------------------------------------------------
# Long rows (2049-4096 tokens)
# ----------------------------------------------------------------------------------------------

FAQ_VERBS = ["change", "cancel", "track", "update", "download", "reset", "delete", "renew", "transfer", "pause"]
FAQ_OBJECTS = [
    "delivery address",
    "order",
    "subscription",
    "password",
    "invoice",
    "payment method",
    "gift card",
    "account email",
    "loyalty points",
    "newsletter preferences",
    "phone number",
    "return label",
    "warranty claim",
    "saved cart",
    "username",
    "billing address",
    "shipping speed",
    "store pickup",
    "wishlist",
    "two-factor login",
    "profile photo",
    "membership",
    "review",
    "coupon code",
]
CHAT_FILLER = [
    "Customer: Hi there.",
    "Agent: Hello! How can I help you today?",
    "Customer: I have a question.",
    "Agent: Of course, take your time.",
    "Customer: Sorry, my connection dropped for a moment.",
    "Agent: No problem, I am still here.",
]


def build_faq_match(n: int) -> tuple[dict, dict]:
    rng = rng_for("l01")
    target = ("change", "delivery address")
    pool = [
        (v, o)
        for v in FAQ_VERBS
        for o in FAQ_OBJECTS
        if (v, o) != target and not (o == "delivery address" and v in ("update", "transfer"))
    ]
    picks = rng.sample(pool, 229)
    picks.insert(141, target)
    criteria = {f"faq_{k:03d}": f"{v} {o}" for k, (v, o) in enumerate(picks)}
    filler = "\n".join(CHAT_FILLER[i % len(CHAT_FILLER)] for i in range(n))
    state = (
        f"Chat transcript:\n{filler}\nCustomer: I moved to a new apartment last month. The order I placed "
        "yesterday still shows my old street. How can I make sure the parcel is sent to my new place instead?"
    )
    question = {
        "type": "choice",
        "instructions": "Which help-center article answers the customer's request?",
        "criteria": criteria,
    }
    return {"state": state, "question": question}, {"expected": "faq_141"}


def build_delivery_feed(n: int) -> tuple[dict, dict]:
    rng = rng_for("l02")
    events = []
    for i in range(n):
        oid = rng.randint(40000, 49999)
        while oid == 48213:
            oid = rng.randint(40000, 49999)
        events.append(
            {
                "order": str(oid),
                "event": rng.choice(["label created", "shipped", "out for delivery", "delivered", "delayed"]),
                "city": rng.choice(CITIES),
                "time": f"2024-07-{rng.randint(1, 28):02d} {rng.randint(6, 21):02d}:00",
            }
        )
    if n:
        events.insert(n // 3, {"order": "48213", "event": "shipped", "city": "Denver", "time": "2024-07-09 10:00"})
        events.insert(
            2 * n // 3 + 1, {"order": "48213", "event": "delivered", "city": "Boston", "time": "2024-07-12 15:00"}
        )
    state = {"source": "carrier tracking feed", "account": "store-east", "events": events}
    question = {"type": "noul", "instructions": "According to the feed, was order 48213 delivered?"}
    return {"state": state, "question": question}, {"expected": "true" if n else "false"}


HEALTH_LEVELS = [
    "healthy: no errors",
    "minor issues: a few isolated errors",
    "degraded: frequent errors affecting many requests",
    "outage: the service is mostly failing",
]


def build_service_health(n: int) -> tuple[dict, dict]:
    log = server_log(rng_for("l03"), n, error_weights={s: 1.0 for s in SERVICES}, error_rate=0.35)
    state = "Application log, last hour:\n" + "\n".join(log)
    question = {
        "type": "score",
        "instructions": "How healthy is the application in this log?",
        "criteria": HEALTH_LEVELS,
    }
    return {"state": state, "question": question}, {"expected": "2 (about 35% of lines are errors)"}


DEPARTMENTS = {
    "hardware_repair": "Devices that are physically broken or will not power on, including warranty repairs",
    "software_support": "Installing, updating or configuring software",
    "billing": "Charges, invoices and payment methods",
    "shipping": "Delivery times and tracking",
    "returns": "Returning unwanted items for a refund",
    "accounts": "Login and account settings",
    "sales": "Pre-purchase questions and quotes",
    "accessibility": "Assistive features and accommodations",
    "business_accounts": "Volume orders for companies",
    "other": "Anything else",
}


def build_laptop_thread(n: int) -> tuple[dict, dict]:
    rng = rng_for("l04")
    turns = [
        "Customer: My laptop (model ZX-14, bought 7 months ago) will not turn on at all. No lights, no fan, "
        "nothing when I press the power button. It is still under the one-year warranty."
    ]
    agent = [
        "Agent: Please hold the power button for 30 seconds with the charger unplugged, then try again.",
        "Agent: Could you try a different wall outlet and check whether the charger LED lights up?",
        "Agent: Thank you. Do you see any physical damage on the charging port?",
        "Agent: Let us try one more reset: unplug everything, wait one minute, then connect only the charger.",
    ]
    customer = [
        "Customer: I tried that, still completely dead.",
        "Customer: The charger LED is on but the laptop " "shows nothing.",
        "Customer: No visible damage, it was just sitting on my desk.",
        "Customer: Same result, it does not power on.",
    ]
    for i in range(n):
        turns.append(rng.choice(agent if i % 2 == 0 else customer))
    state = "Support chat (newest last)\n" + "\n".join(turns)
    question = {
        "type": "choice",
        "instructions": "Which department should take over this case?",
        "criteria": DEPARTMENTS,
    }
    return {"state": state, "question": question}, {"expected": "hardware_repair"}


TEAMS = [
    "Northside Rovers",
    "Lakeview United",
    "Harbor City",
    "Pine Valley",
    "Eastgate Athletic",
    "Millbrook Town",
    "Westfield Rangers",
    "Stonebridge FC",
]


def build_league(n: int) -> tuple[dict, dict]:
    rng = rng_for("l05")
    table = {t: {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "Pts": 0} for t in TEAMS}
    lines = []
    strength = {t: 1.0 + 0.25 * k for k, t in enumerate(reversed(TEAMS))}  # Northside strongest
    for i in range(n):
        home, away = rng.sample(TEAMS, 2)
        hg = min(6, int(rng.random() * 2.2 * strength[home] / 2))
        ag = min(6, int(rng.random() * 2.2 * strength[away] / 2))
        lines.append(f"Matchday {i // 4 + 1}: {home} {hg} - {ag} {away}")
        for team, gf, ga in ((home, hg, ag), (away, ag, hg)):
            row = table[team]
            row["P"] += 1
            row["GF"] += gf
            row["GA"] += ga
            res = "W" if gf > ga else "D" if gf == ga else "L"
            row[res] += 1
            row["Pts"] += {"W": 3, "D": 1, "L": 0}[res]
    order = sorted(TEAMS, key=lambda t: (-table[t]["Pts"], -(table[t]["GF"] - table[t]["GA"]), t))
    standings = ["Pos Team P W D L GF GA Pts"] + [
        f"{k + 1} {t} {table[t]['P']} {table[t]['W']} {table[t]['D']} {table[t]['L']} {table[t]['GF']} "
        f"{table[t]['GA']} {table[t]['Pts']}"
        for k, t in enumerate(order)
    ]
    state = "Season results:\n" + "\n".join(lines) + "\n\nStandings after the last matchday:\n" + "\n".join(standings)
    question = {
        "type": "choice",
        "instructions": "Which team is top of the standings?",
        "criteria": {t: None for t in TEAMS},
    }
    return {"state": state, "question": question}, {"expected": order[0]}


def long_specs() -> list[PromptSpec]:
    return [
        PromptSpec("l01_faq_match_230opt", "long", build_faq_match, 3300),
        PromptSpec("l02_delivery_feed_json", "long", build_delivery_feed, 2600),
        PromptSpec("l03_service_health", "long", build_service_health, 3600),
        PromptSpec("l04_laptop_department", "long", build_laptop_thread, 2300),
        PromptSpec("l05_league_leader", "long", build_league, 3900),
    ]


# ----------------------------------------------------------------------------------------------
# XL rows (4097-8192 tokens)
# ----------------------------------------------------------------------------------------------


def build_log_most_errors(n: int) -> tuple[dict, dict]:
    weights = {"payments": 6.0, "auth": 1.0, "catalog": 1.0, "search": 1.0, "notifications": 1.0}
    log = server_log(rng_for("x01"), n, error_weights=weights, error_rate=0.12)
    counts = {s: sum(1 for line in log if f"ERROR [{s}]" in line) for s in SERVICES}
    state = "Combined service log, 14 June 2024:\n" + "\n".join(log)
    question = {
        "type": "choice",
        "instructions": "Which service logged the most ERROR lines?",
        "criteria": {s: f"the {s} service" for s in SERVICES},
    }
    return {"state": state, "question": question}, {"expected": max(counts, key=counts.get), "error_counts": counts}


def build_faq_returns(n: int) -> tuple[dict, dict]:
    rng = rng_for("x02")
    combos = [(v, o) for v in FAQ_VERBS for o in FAQ_OBJECTS]
    rng.shuffle(combos)
    entries = []
    for i in range(n):
        v, o = combos[i % len(combos)]
        entries.append(
            f"Q: How do I {v} my {o}?\nA: Open Settings, select '{o.title()}', choose '{v.title()}' and "
            f"follow the steps on screen. The change is applied within {rng.randint(1, 48)} hours."
        )
    entries.insert(
        n // 2,
        "Q: How long do I have to return an item?\nA: Unused items in their original packaging "
        "can be returned within 30 days of delivery for a full refund to the original payment method.",
    )
    state = "Help center FAQ (all entries)\n\n" + "\n\n".join(entries)
    question = {
        "type": "noul",
        "instructions": "According to the FAQ, can unused items be returned within 30 days of delivery?",
    }
    return {"state": state, "question": question}, {"expected": "true"}


def build_reviews_negative(n: int) -> tuple[dict, dict]:
    reviews = review_list(rng_for("x03"), n, {1: 0.4, 2: 0.4, 3: 0.15, 4: 0.05}, "rain jacket")
    state = "Customer reviews for the lightweight rain jacket:\n" + reviews
    question = {
        "type": "score",
        "instructions": "Overall, how satisfied are these customers with the rain jacket?",
        "criteria": STAR_LEVELS,
    }
    return {"state": state, "question": question}, {"expected": "0 or 1"}


def build_customer_lookup(n: int) -> tuple[dict, dict]:
    rng = rng_for("x04")
    target = "Priya Okafor"
    others = [f"{f} {l}" for f in FIRST for l in LAST if f"{f} {l}" != target]
    names = rng.sample(others, 24)
    names.insert(9, target)
    forced = {3 * n // 4: ["9821", "2024-09-03", target, "Tampa", "teal yoga mat", "2", "58.00", "delivered"]}
    table = order_csv(rng_for("x04-orders"), n, forced=forced)
    state = "Order export, all regions:\n" + table
    question = {
        "type": "choice",
        "instructions": "Which customer placed order 9821?",
        "criteria": {name: None for name in names},
    }
    return {"state": state, "question": question}, {"expected": target}


def xl_specs() -> list[PromptSpec]:
    return [
        PromptSpec("x01_log_most_errors", "xl", build_log_most_errors, 6000),
        PromptSpec("x02_faq_returns", "xl", build_faq_returns, 5000),
        PromptSpec("x03_reviews_negative", "xl", build_reviews_negative, 7000),
        PromptSpec("x04_customer_lookup", "xl", build_customer_lookup, 7900),
    ]


def all_specs() -> list[PromptSpec]:
    return short_specs() + medium_specs() + long_specs() + xl_specs()


# ----------------------------------------------------------------------------------------------
# Length fitting
# ----------------------------------------------------------------------------------------------


def fit_spec(spec: PromptSpec, tok: AppTokenizer, n_max: int = 4096) -> dict:
    """Pick the smallest ``n`` whose prompt reaches ``spec.target`` tokens; reject and rebuild
    (smaller ``n``) anything over the band (and therefore anything over 8192)."""
    lo_band, hi_band = BANDS[spec.band]

    def length(n: int) -> tuple[int, dict, dict, list[int]]:
        row, meta = spec.build(n)
        ids = tok.input_ids(row)
        return len(ids), row, meta, ids

    if spec.target is None:
        n = 0
    else:
        lo, hi = 0, 1
        while length(hi)[0] < spec.target:
            lo, hi = hi, hi * 2
            if hi > n_max:
                raise ValueError(f"{spec.id}: cannot reach {spec.target} tokens")
        while hi - lo > 1:  # invariant: length(lo) < target <= length(hi)
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if length(mid)[0] >= spec.target else (mid, hi)
        n = hi if length(0)[0] < spec.target else 0
    seq, row, meta, ids = length(n)
    while seq > min(hi_band, MAX_LENGTH) and n > 0:  # rejected: rebuild smaller
        n -= 1
        seq, row, meta, ids = length(n)
    if not lo_band <= seq <= hi_band:
        raise ValueError(f"{spec.id}: {seq} tokens is outside band {spec.band} {BANDS[spec.band]}")
    count = tok.count(row)
    return {
        "id": spec.id,
        "type": row["question"]["type"],
        "band": spec.band,
        "n": n,
        "row": row,
        "meta": meta,
        "count": count,
        "seq_len": seq,
        "bucket": bucket_of(seq),
        "input_ids": ids,
    }


def build_prompt_set(tok: AppTokenizer | None = None) -> list[dict]:
    tok = tok or AppTokenizer()
    rows = [fit_spec(spec, tok) for spec in all_specs()]
    per_band = {b: sum(r["band"] == b for r in rows) for b in BANDS}
    if per_band != BAND_COUNTS:
        raise AssertionError(f"band counts {per_band} != {BAND_COUNTS}")
    if len({r["id"] for r in rows}) != len(rows):
        raise AssertionError("duplicate row ids")
    return rows


def main() -> None:
    tok = AppTokenizer()
    print(json.dumps(tok.verify_answer_codes()))
    rows = build_prompt_set(tok)
    for i, r in enumerate(rows):
        print(
            f"{i:2d} {r['id']:<26} {r['type']:<6} band={r['band']:<6} n={r['n']:<4} count={r['count']:<4} "
            f"seq={r['seq_len']:<5} bucket={r['bucket']:<5} expected={r['meta']['expected']}"
        )


if __name__ == "__main__":
    main()
