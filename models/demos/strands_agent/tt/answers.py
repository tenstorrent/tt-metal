# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# Prompt and answer logic adapted from strands-labs/strands-decider (Apache-2.0).

import torch
import torch.nn.functional as F

from models.demos.strands_agent.tt.prompting import EncodedQuestion

MASK_VALUE = -1e4


def option_log_probs(logits: torch.Tensor, n_valid: int, temperature: float) -> torch.Tensor:
    """Log-softmax over the first n_valid slots of one row (fp32); rest -inf."""
    logits = logits.reshape(-1).float()
    if temperature != 1.0:
        logits = logits / max(float(temperature), 1e-6)
    valid = torch.arange(logits.numel()) < n_valid
    masked = logits.masked_fill(~valid, MASK_VALUE)
    return F.log_softmax(masked, dim=-1).masked_fill(~valid, float("-inf"))


def derive_confidence(probabilities: list[float]) -> float:
    n = len(probabilities)
    if n <= 1:
        return 1.0
    p_max = max(probabilities)
    return float(max(0.0, min(1.0, (n * p_max - 1.0) / (n - 1.0))))


def derive_score_confidence(probabilities: list[float], *, ordinal_smoothing: float = 0.0) -> float:
    n = len(probabilities)
    if n <= 1:
        return 1.0
    total = sum(probabilities)
    if total <= 0:
        return 0.0
    p = [x / total for x in probabilities]
    mean = sum(i * pi for i, pi in enumerate(p))
    sigma = sum(pi * (i - mean) ** 2 for i, pi in enumerate(p)) ** 0.5
    sigma_max = (n - 1) / 2.0
    sigma_floor = ordinal_smoothing**0.5 if ordinal_smoothing > 0 else 0.0
    if sigma_max <= sigma_floor:
        return 1.0 if sigma <= sigma_floor else 0.0
    return float(max(0.0, min(1.0, (sigma_max - sigma) / (sigma_max - sigma_floor))))


def build_answer(
    q: EncodedQuestion,
    logits: torch.Tensor,
    temperature_by_kind: dict,
    temperature: float,
    ordinal_smoothing: float,
) -> dict:
    n = len(q.labels)
    t = float(temperature_by_kind.get(q.kind, temperature))
    probs = option_log_probs(logits, n, t).exp()[:n].tolist()

    if q.kind == "noul":
        return {"type": q.kind, "noul": round(probs[q.labels.index("true")], 4)}

    if q.kind == "choice":
        by_name = {k: float(p) for k, p in zip(q.labels, probs)}
        best = max(by_name, key=lambda k: by_name[k])
        return {
            "type": q.kind,
            "choice": best,
            "probabilities": {k: round(v, 4) for k, v in by_name.items()},
            "confidence": round(derive_confidence(list(by_name.values())), 4),
        }

    by_level = {k: float(p) for k, p in zip(q.labels, probs)}
    expected = sum(int(level) * p for level, p in by_level.items())
    ordered = {str(i): by_level[str(i)] for i in range(len(by_level))}
    legend = dict(sorted(zip(q.labels, q.descriptions), key=lambda kv: int(kv[0])))
    return {
        "type": q.kind,
        "score": round(float(expected), 4),
        "legend": legend,
        "probabilities": {k: round(v, 4) for k, v in ordered.items()},
        "confidence": round(derive_score_confidence(list(ordered.values()), ordinal_smoothing=ordinal_smoothing), 4),
    }
