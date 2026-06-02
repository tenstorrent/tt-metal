import numpy as np
import torch


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def golden_outputs(reference_module, example_inputs: tuple) -> torch.Tensor:
    reference_module.eval()
    with torch.no_grad():
        return reference_module(*example_inputs)


def drift_metrics(golden_logits, fused_logits) -> dict:
    g = np.asarray(golden_logits)
    f = np.asarray(fused_logits)
    T = min(len(g), len(f))
    g, f = g[:T], f[:T]
    matches = np.argmax(g, axis=-1) == np.argmax(f, axis=-1)
    token_match_rate = float(matches.mean())
    mism = np.where(~matches)[0]
    first_divergence_step = int(mism[0]) if len(mism) else T
    err = np.linalg.norm(g - f, axis=-1)
    slope = float(np.polyfit(np.arange(T), err, 1)[0]) if T > 1 else 0.0
    return {
        "token_match_rate": token_match_rate,
        "first_divergence_step": first_divergence_step,
        "drift_slope": slope,
        "horizon": T,
    }


def perf_gain_pct(naive_ms: float, fused_ms: float) -> float:
    return (naive_ms - fused_ms) / naive_ms * 100.0


def perf_gate_pass(naive_ms: float, fused_ms: float, min_gain_pct: float) -> bool:
    return perf_gain_pct(naive_ms, fused_ms) >= min_gain_pct
