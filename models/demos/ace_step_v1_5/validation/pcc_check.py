from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch


@dataclass(frozen=True)
class PccResult:
    pcc: float
    mean_ref: float
    std_ref: float
    mean_tt: float
    std_tt: float
    max_abs_diff: float


def compute_pcc(torch_ref: torch.Tensor, torch_tt: torch.Tensor) -> float:
    """Pearson correlation coefficient on flattened float32 tensors."""
    a = torch_ref.detach().float().flatten()
    b = torch_tt.detach().float().flatten()
    if a.numel() == 0:
        raise ValueError("Cannot compute PCC on empty tensors")
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def summarize(torch_ref: torch.Tensor, torch_tt: torch.Tensor) -> PccResult:
    ref = torch_ref.detach().float()
    tt = torch_tt.detach().float()
    max_abs_diff = torch.max(torch.abs(ref - tt)).item()
    return PccResult(
        pcc=compute_pcc(ref, tt),
        mean_ref=ref.mean().item(),
        std_ref=ref.std(unbiased=False).item(),
        mean_tt=tt.mean().item(),
        std_tt=tt.std(unbiased=False).item(),
        max_abs_diff=max_abs_diff,
    )


def print_report(
    name: str,
    torch_ref: torch.Tensor,
    torch_tt: torch.Tensor,
    *,
    pcc_threshold: float = 0.99,
) -> Tuple[bool, Dict[str, Any]]:
    r = summarize(torch_ref, torch_tt)
    passed = r.pcc >= pcc_threshold
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")
    print(f"  mean(ref)={r.mean_ref:.6g}  std(ref)={r.std_ref:.6g}")
    print(f"  mean(tt) ={r.mean_tt:.6g}  std(tt) ={r.std_tt:.6g}")
    print(f"  max|diff|={r.max_abs_diff:.6g}")
    print(f"  PCC={r.pcc:.6f}  (threshold={pcc_threshold})")
    return passed, r.__dict__
