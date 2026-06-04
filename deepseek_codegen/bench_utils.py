# Vendored from tt-xla tests/benchmark/utils.py (compute_pcc only) so pcc.py
# can run inside the tt-metal repo without the tt-xla tests tree on disk.
import torch


def compute_pcc(golden_output: torch.Tensor, device_output: torch.Tensor) -> float:
    golden_flat = golden_output.to(torch.float32).flatten()
    device_flat = device_output.to(torch.float32).flatten()
    golden_centered = golden_flat - golden_flat.mean()
    device_centered = device_flat - device_flat.mean()
    denom = golden_centered.norm() * device_centered.norm()
    if denom == 0:
        if torch.allclose(golden_flat, device_flat, rtol=1e-2, atol=1e-2):
            return 1.0
        raise ValueError("PCC failed: denom zero but tensors not close")
    pcc = ((golden_centered @ device_centered) / denom).item()
    return max(-1.0, min(1.0, pcc))
