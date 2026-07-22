"""CausalConditionalCFM — Euler ODE solver + CFG.

Orchestrates the UNet1D estimator over n_timesteps=10 Euler steps with
classifier-free guidance (batch=2: conditioned + unconditioned).

Stage 2.3: supports both host-side (UNetEstimator) and device (UNetEstimatorTtnn).
When using UNetEstimatorTtnn, the CFM loop uses device tracing for steps 1-9
after a warmup step 0, eliminating per-op dispatch overhead.

Reference: cosyvoice/flow/flow_matching.py::CausalConditionalCFM
"""

from __future__ import annotations

from typing import Union

import torch

from models.demos.cosyvoice.tt.flow.unet_estimator import UNetEstimator
from models.demos.cosyvoice.tt.model_config import FLOW


class CausalConditionalCFM:
    """Euler ODE solver with CFG for flow matching (non-streaming)."""

    def __init__(self, estimator: Union[UNetEstimator, "UNetEstimatorTtnn"], n_timesteps: int | None = None):
        self.estimator = estimator
        self.n_timesteps = n_timesteps if n_timesteps is not None else FLOW.decoder.n_timesteps
        self.inference_cfg_rate = FLOW.decoder.inference_cfg_rate
        self._use_trace = hasattr(estimator, "init_trace")

    def _assemble_input(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        T: int,
    ) -> torch.Tensor:
        """Assemble [x, mu, spks_expanded, cond] → [B, T, 320] in channels-last."""
        B = x.shape[0]
        x_btc = x.permute(0, 2, 1).contiguous()
        mu_btc = mu.permute(0, 2, 1).contiguous()
        spks_exp = spks.unsqueeze(2).expand(-1, -1, T).permute(0, 2, 1).contiguous()
        cond_btc = cond.permute(0, 2, 1).contiguous()
        return torch.cat([x_btc, mu_btc, spks_exp, cond_btc], dim=-1)

    @torch.no_grad()
    def solve_euler(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Euler ODE solver with classifier-free guidance.

        Args:
            x: [1, 80, T] initial noise
            t_span: [n_timesteps+1] cosine-scheduled time points
            mu: [1, 80, T] conditioning mel
            mask: [1, 1, T]
            spks: [1, 80] speaker embedding
            cond: [1, 80, T] prompt mel condition

        Returns:
            [1, 80, T] final mel
        """
        t = t_span[0].unsqueeze(0)
        dt = t_span[1] - t_span[0]
        T_orig = x.size(2)

        x_in = torch.zeros([2, 80, T_orig], device=x.device, dtype=spks.dtype)
        mask_in = torch.zeros([2, 1, T_orig], device=x.device, dtype=spks.dtype)
        mu_in = torch.zeros([2, 80, T_orig], device=x.device, dtype=spks.dtype)
        t_in = torch.zeros([2], device=x.device, dtype=spks.dtype)
        spks_in = torch.zeros([2, 80], device=x.device, dtype=spks.dtype)
        cond_in = torch.zeros([2, 80, T_orig], device=x.device, dtype=spks.dtype)

        trace_initialized = False
        trace_active = False

        for step in range(1, len(t_span)):
            x_in[:] = x
            mask_in[:] = mask
            mu_in[0] = mu
            t_in[:] = t.unsqueeze(0)
            spks_in[0] = spks
            cond_in[0] = cond

            if trace_active:
                T_pad = ((T_orig + 31) // 32) * 32
                x_cat = self._assemble_input(x_in, mu_in, spks_in, cond_in, T_pad)
                dphi_dt = self.estimator.forward_traced(x_cat, t_in)
            else:
                dphi_dt = self.estimator.forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in, streaming=False)

                if self._use_trace and not trace_initialized:
                    trace_initialized = True
                    trace_active = self.estimator.init_trace(2, T_orig, mask_in)

            dphi_dt_cond, dphi_dt_uncond = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt_cond - self.inference_cfg_rate * dphi_dt_uncond

            x = x + dt * dphi_dt
            t = t + dt
            dt = t_span[step + 1] - t if step < len(t_span) - 1 else dt

        if trace_active:
            self.estimator.release_trace()

        return x

    @torch.no_grad()
    def inference(
        self,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Full CFM inference: noise → mel via Euler solver.

        Args:
            mu: [1, 80, T]
            mask: [1, 1, T]
            spks: [1, 80]
            cond: [1, 80, T]

        Returns:
            [1, 80, T] generated mel
        """
        x = torch.randn_like(mu)
        t_span = torch.linspace(0, 1, self.n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if FLOW.decoder.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(x, t_span, mu, mask, spks, cond)
