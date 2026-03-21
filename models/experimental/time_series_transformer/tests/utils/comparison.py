import torch
import numpy as np

def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.flatten().float().numpy()
    b_flat = b.flatten().float().numpy()
    if np.std(a_flat) == 0 or np.std(b_flat) == 0:
        return 1.0 if np.allclose(a_flat, b_flat) else 0.0
    return float(np.corrcoef(a_flat, b_flat)[0, 1])

def compare_tensors(actual, expected, pcc_threshold = 0.99, atol = 1e-3, rtol = 1e-3):

    pcc = compute_pcc(actual, expected)
    allclose = torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol)
    max_abs_diff = (actual.float() - expected.float()).abs().max().item()

    passed = pcc >= pcc_threshold
    return passed, {
        "pcc": pcc,
        "pcc_threshold": pcc_threshold,
        "allclose": allclose,
        "max_abs_diff": max_abs_diff,
        "shape_match": actual.shape == expected.shape,
    }