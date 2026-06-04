# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PoC LoRA-aware Linear layer with runtime adapter swapping.

Hosts a bank of (A, B, scale) adapters on device alongside the base weight.
Supports two execution paths so they can be compared head-to-head:

  1. Runtime delta (set_active(idx)):
       y = base(x) + scale * (x @ A) @ B           [3 matmuls per forward]
       swap = host-side int assignment             [microseconds]

  2. Fused base (fuse_into_base(idx)):
       W' = W + scale * (A @ B), stored in place
       y = base(x; W')                             [1 matmul per forward]
       swap = on-device matmul + add               [milliseconds]

set_active(None) reverts to pure base behavior with zero overhead.

Scope (v0):
  - Replicated Linear only — no Col/Row parallel sharding
  - No bias deltas (diff_b) or full-param deltas (diff)
  - Caller supplies A, B in PyTorch LoRA convention:
        A: [rank, in_features]
        B: [out_features, rank]

See docs/plans/wan22_dynamic_lora_swap.md for the full design.
"""
from dataclasses import dataclass

import torch
import ttnn

from ...layers.linear import Linear
from ...utils.tensor import bf16_tensor


@dataclass
class LoRAAdapter:
    name: str
    A: ttnn.Tensor    # TT-internal layout: [in_features, rank]
    B: ttnn.Tensor    # TT-internal layout: [rank, out_features]
    rank: int
    scale: float


class LoRALinear(Linear):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lora_bank: list[LoRAAdapter | None] = []
        self.active_idx: int | None = None
        self._fused_idx: int | None = None
        self._original_W_pytorch: torch.Tensor | None = None

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "weight" in state and self._original_W_pytorch is None:
            # Snapshot in PyTorch [out, in] layout before super() transposes.
            self._original_W_pytorch = state["weight"].detach().clone()
        super()._prepare_torch_state(state)

    # ------------------------------------------------------------------
    # Bank management
    # ------------------------------------------------------------------
    def register_lora(
        self,
        A_torch: torch.Tensor,
        B_torch: torch.Tensor,
        scale: float = 1.0,
        name: str = "",
    ) -> int:
        rank = A_torch.shape[0]
        if A_torch.shape != (rank, self.in_features):
            raise ValueError(
                f"A must be [rank, in_features]; got {tuple(A_torch.shape)} "
                f"expected (*, {self.in_features})"
            )
        if B_torch.shape != (self.out_features, rank):
            raise ValueError(
                f"B must be [out_features, rank]; got {tuple(B_torch.shape)} "
                f"expected ({self.out_features}, {rank})"
            )

        A_tt = A_torch.transpose(0, 1).contiguous()  # [in, r]
        B_tt = B_torch.transpose(0, 1).contiguous()  # [r, out]

        A_dev = bf16_tensor(A_tt, device=self.mesh_device)
        B_dev = bf16_tensor(B_tt, device=self.mesh_device)

        adapter = LoRAAdapter(name=name, A=A_dev, B=B_dev, rank=rank, scale=float(scale))
        self.lora_bank.append(adapter)
        return len(self.lora_bank) - 1

    def unregister_lora(self, idx: int) -> None:
        adapter = self.lora_bank[idx]
        if adapter is None:
            return
        ttnn.deallocate(adapter.A)
        ttnn.deallocate(adapter.B)
        self.lora_bank[idx] = None
        if self.active_idx == idx:
            self.active_idx = None

    def set_active(self, idx: int | None) -> None:
        if idx is not None:
            if not (0 <= idx < len(self.lora_bank)) or self.lora_bank[idx] is None:
                raise IndexError(f"invalid lora slot {idx}")
        self.active_idx = idx

    # ------------------------------------------------------------------
    # Forward — runtime delta path
    # ------------------------------------------------------------------
    def forward(self, x: ttnn.Tensor, **kwargs) -> ttnn.Tensor:
        base = super().forward(x, **kwargs)
        if self.active_idx is None:
            return base

        adapter = self.lora_bank[self.active_idx]
        ax = ttnn.matmul(x, adapter.A)        # [..., r]
        delta = ttnn.matmul(ax, adapter.B)    # [..., out]
        if adapter.scale != 1.0:
            delta = ttnn.multiply(delta, adapter.scale)
        return ttnn.add(base, delta)

    # ------------------------------------------------------------------
    # Fuse-on-swap path — for comparison benchmarks
    # ------------------------------------------------------------------
    def fuse_into_base(self, idx: int) -> None:
        """
        On-device fuse: W' = W + scale * (A @ B). After this, forward executes
        a single matmul. active_idx is forced to None.
        """
        adapter = self.lora_bank[idx]
        if adapter is None:
            raise IndexError(f"invalid lora slot {idx}")

        if self._fused_idx is not None:
            self.restore_base()

        delta_W = ttnn.matmul(adapter.A, adapter.B)  # [in, out]
        if adapter.scale != 1.0:
            delta_W = ttnn.multiply(delta_W, adapter.scale)

        new_W = ttnn.add(self.weight.data, delta_W)
        ttnn.deallocate(delta_W)

        self.weight.deallocate()
        self.weight.data = new_W
        self._fused_idx = idx
        self.active_idx = None

    def restore_base(self) -> None:
        """Re-uploads the original base weight from the host-side backup."""
        if self._fused_idx is None:
            return
        if self._original_W_pytorch is None:
            raise RuntimeError("no original-weight backup; restore_base() unavailable")
        self.weight.deallocate()
        # Stored in PyTorch [out, in]; TT-internal layout is [in, out].
        w_tt = self._original_W_pytorch.transpose(0, 1).contiguous()
        self.weight.load_torch_tensor(w_tt)
        self._fused_idx = None
