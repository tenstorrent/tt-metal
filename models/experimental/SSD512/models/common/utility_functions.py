from __future__ import annotations

import torch
from typing import Tuple

import ttnn


def torch_to_tt_tensor_rm(t: torch.Tensor, device=None, put_on_device: bool = True) -> ttnn.Tensor:
    if isinstance(t, ttnn.Tensor):
        return t
    if put_on_device:
        pass
    return ttnn.from_torch(t)


def tt_to_torch_tensor(t: ttnn.Tensor) -> torch.Tensor:
    if isinstance(t, ttnn.Tensor):
        return t.tt_tensor
    return t


def comp_allclose(a: torch.Tensor, b: torch.Tensor, atol: float = 1e-6, rtol: float = 1e-5) -> bool:
    a_t = a.detach().cpu()
    b_t = b.detach().cpu()
    try:
        return torch.allclose(a_t, b_t, rtol=rtol, atol=atol)
    except Exception:
        diff = (a_t - b_t).abs()
        return bool((diff <= (atol + rtol * b_t.abs())).all())


def comp_pcc(a: torch.Tensor, b: torch.Tensor, threshold: float = 0.99) -> Tuple[bool, float]:
    a_t = a.detach().cpu().float().reshape(-1)
    b_t = b.detach().cpu().float().reshape(-1)
    if a_t.numel() == 0 or b_t.numel() == 0:
        return False, 0.0
    # subtract mean
    a_t = a_t - a_t.mean()
    b_t = b_t - b_t.mean()
    denom = torch.norm(a_t) * torch.norm(b_t)
    if denom == 0:
        return False, 0.0
    pcc = float((a_t * b_t).sum() / denom)
    return (pcc >= threshold, pcc)
