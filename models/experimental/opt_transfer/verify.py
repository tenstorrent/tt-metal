import torch


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def golden_outputs(reference_module, example_inputs: tuple) -> torch.Tensor:
    reference_module.eval()
    with torch.no_grad():
        return reference_module(*example_inputs)


import numpy as np


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
