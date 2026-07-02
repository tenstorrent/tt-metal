"""OP CODE -> op_class map (PLAN section 4.2).

`OP CODE` is TT-NN's closed kernel vocabulary; models compose ops, never invent
op codes. First substring match wins; no match -> `other` (+ a coverage-lint
warning so the gap is visible, never silent).

This is DATA, not logic: extending coverage = adding one line to OP_CLASS_MAP.
"""

from __future__ import annotations

# First match wins (PLAN section 4.2). Order matters only for overlap; the TT-NN
# vocabulary is disjoint enough that substring collisions are not expected.
OP_CLASS_MAP: list[tuple[tuple[str, ...], str]] = [
    (("Matmul", "Linear"), "matmul"),
    (
        (
            "SDPA",
            "ScaledDotProduct",
            "FlashDecode",
            "PagedAttention",
            "NlpCreateHeads",
            "NlpConcatHeads",
            "RotaryEmbedding",
        ),
        "attention",
    ),
    (
        ("LayerNorm", "RMSNorm", "GroupNorm", "Softmax", "Reduce", "ArgMax", "TopK", "Moreh"),
        "reduction",
    ),
    (("BinaryNg", "Binary", "Unary", "Eltwise", "Where"), "eltwise"),
    (
        (
            "Reshape",
            "Tilize",
            "Untilize",
            "Typecast",
            "Transpose",
            "Permute",
            "Concat",
            "Slice",
            "Pad",
            "Copy",
            "Move",
            "InterleavedToSharded",
            "ShardedToInterleaved",
            "Reshard",
        ),
        "datamove",
    ),
    (("Embedding",), "embedding"),
    (("Conv", "Halo", "Pool", "GridSample", "Upsample"), "conv_pool"),
    (("AllGather", "ReduceScatter", "AllReduce", "LineAllGather"), "ccl"),
]

UNCLASSIFIED = "other"

# Signpost marker rows emitted by the profiler (not device ops) — dropped early.
SIGNPOST_CODES = frozenset({"start", "stop"})

# The closed set of classes the parser can emit (for coverage lint, section 4.5 rule 6).
EMITTABLE_OP_CLASSES = frozenset([cls for _, cls in OP_CLASS_MAP] + [UNCLASSIFIED])


def base_op_code(op_code: str) -> str:
    """Strip the shape suffix tt-perf-report appends (e.g. 'Matmul... 512 x 1024')."""
    return op_code.split(" ", 1)[0].strip()


def classify_op(op_code: str) -> str:
    """Return the op_class for an OP CODE; `other` when no substring matches."""
    base = base_op_code(op_code)
    for needles, cls in OP_CLASS_MAP:
        if any(n in base for n in needles):
            return cls
    return UNCLASSIFIED


def is_classified(op_code: str) -> bool:
    """True if the op_code matched a known class (i.e. did not fall to `other`)."""
    return classify_op(op_code) != UNCLASSIFIED
