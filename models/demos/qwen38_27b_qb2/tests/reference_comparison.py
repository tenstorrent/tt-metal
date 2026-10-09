# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for teacher-forced diagnostic inputs and vector comparisons."""

import math


def vector_metrics(actual, expected):
    if not actual or len(actual) != len(expected):
        raise ValueError("Comparison requires nonempty vectors with equal lengths")
    if not all(math.isfinite(v) for values in (actual, expected) for v in values):
        raise ValueError("Comparison contains non-finite values")
    n = len(actual)
    dot = math.fsum(a * b for a, b in zip(actual, expected))
    a2 = math.fsum(a * a for a in actual)
    b2 = math.fsum(b * b for b in expected)
    error2 = math.fsum((a - b) ** 2 for a, b in zip(actual, expected))
    mean_a, mean_b = math.fsum(actual) / n, math.fsum(expected) / n
    centered_a = [a - mean_a for a in actual]
    centered_b = [b - mean_b for b in expected]
    covariance = math.fsum(a * b for a, b in zip(centered_a, centered_b))
    variance_a = math.fsum(a * a for a in centered_a)
    variance_b = math.fsum(b * b for b in centered_b)
    return dict(
        elements=n,
        rms_error=math.sqrt(error2 / n),
        relative_rms_error=math.sqrt(error2 / b2) if b2 else None,
        max_absolute_error=max(abs(a - b) for a, b in zip(actual, expected)),
        cosine=dot / math.sqrt(a2 * b2) if a2 and b2 else None,
        pcc=covariance / math.sqrt(variance_a * variance_b) if variance_a and variance_b else None,
    )


def validate_teacher_forcing(rows, prompt, *, layers, hidden_size, vocab_size):
    """Validate tensor metadata before any model allocation or hardware access."""
    if not rows or not prompt or layers < 1:
        raise ValueError("Reference requires a prompt, decoder layers and at least one step")
    previous_token = None
    for index, row in enumerate(rows):
        expected_ids = prompt if index == 0 else [previous_token]
        ids = row["input_ids"].tolist()
        if row.get("step") != index or ids != [expected_ids]:
            raise ValueError("Reference must contain consecutive HF teacher-forced inputs")
        if any(type(t) is not int or not 0 <= t < vocab_size for t in expected_ids):
            raise ValueError("Reference token lies outside the vocabulary")
        states = row["layer_last_hidden"]
        # Transformers capture_outputs ties hidden_states[-1] to final norm.
        # Entries 0..layers-1 are decoder inputs; the last is not raw layer output.
        if len(states) != layers + 1 or any(tuple(t.shape) != (1, hidden_size) for t in states):
            raise ValueError("Reference requires every decoder input and final normalized state")
        logits = row["logits"]
        if tuple(logits.shape) != (1, vocab_size):
            raise ValueError("Reference logits have the wrong vocabulary shape")
        previous_token = int(logits.argmax(-1).item())
    return len(prompt) + len(rows) - 1
