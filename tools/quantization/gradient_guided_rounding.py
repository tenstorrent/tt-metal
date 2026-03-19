# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Gradient-guided rounding correction for bfloat16 weights.

When casting float32 weights to bfloat16, the default round-to-nearest-even rule
introduces quantization error. This module uses gradient information from a float32
reference model to decide whether each weight element should round to the next higher
or lower bfloat16 representable value, minimizing the output MSE between the reference
and quantized models.
"""

import torch
from typing import Callable, List, Tuple

# (param_name, getter() -> torch.Tensor in param shape, setter(torch.Tensor) -> None)
WeightMapping = List[Tuple[str, Callable[[], torch.Tensor], Callable[[torch.Tensor], None]]]


def bf16_gradient_round(
    fp32_weight: torch.Tensor,
    grad: torch.Tensor,
    grad_percentile: float = 50.0,
) -> torch.Tensor:
    """
    For each fp32 weight element, choose between its two nearest bf16
    representable neighbors based on the gradient direction.

    Only changes weights where *both* conditions hold:
      1. The gradient disagrees with the default round-to-nearest-even choice.
      2. The gradient magnitude is in the top ``(100 - grad_percentile)``
         percent for this tensor — i.e. it is large enough to be a reliable
         signal rather than numerical noise.

    Args:
        fp32_weight: Original float32 weight values.
        grad: Gradient tensor, same shape as fp32_weight.
        grad_percentile: Only trust gradients whose absolute value exceeds
            this percentile of ``|grad|`` within the tensor.  Higher values
            are more conservative (fewer switches).  Default 90 means only
            the top 10 % strongest gradients can trigger a switch.

    Returns:
        bfloat16 tensor where each element is one of the two bf16 neighbors
        of the corresponding fp32 value.
    """
    neg_inf = torch.tensor(-float("inf"), dtype=torch.bfloat16)
    pos_inf = torch.tensor(float("inf"), dtype=torch.bfloat16)

    W_near = fp32_weight.to(torch.bfloat16)
    rounding_error = W_near.float() - fp32_weight

    # The other bf16 neighbor: opposite direction from the rounding.
    W_other = torch.where(
        rounding_error > 0,
        torch.nextafter(W_near, neg_inf),
        torch.where(
            rounding_error < 0,
            torch.nextafter(W_near, pos_inf),
            W_near,
        ),
    )

    # Switch when default rounding went against the gradient preference:
    #   rounded up  (error > 0) but gradient wants lower (grad > 0) → switch
    #   rounded down (error < 0) but gradient wants higher (grad < 0) → switch
    grad_bf16 = grad.to(torch.bfloat16)
    should_switch = ((rounding_error > 0) & (grad_bf16 > 0)) | ((rounding_error < 0) & (grad_bf16 < 0))

    # Only act on gradients large enough to be a reliable signal.
    abs_grad = grad.abs().float()
    threshold = torch.quantile(abs_grad, grad_percentile / 100.0)
    should_switch = should_switch & (abs_grad >= threshold)

    return torch.where(should_switch, W_other, W_near)


def gradient_guided_bf16_rounding(
    reference_module: torch.nn.Module,
    target_forward_fn: Callable[[torch.Tensor], torch.Tensor],
    dummy_input: torch.Tensor,
    weight_mapping: WeightMapping,
    num_iterations: int = 1,
) -> List[float]:
    """
    Correct bf16 quantization rounding using gradient information.

    Uses the float32 reference model as a differentiable proxy to *suggest*
    which bf16 rounding direction is better for each weight element.  Each
    suggestion is then validated against the actual quantized model: if
    applying the gradient-suggested rounding to a parameter doesn't reduce
    MSE, the change is reverted.  This makes the algorithm robust to
    proxy–target mismatch (e.g. different accumulation arithmetic).

    Args:
        reference_module: Torch module with float32 weights.
        target_forward_fn: Callable that takes a float32 torch input tensor
            and returns the quantized model's output as a float32 torch tensor.
            Must read the latest weights from the quantized model on each call
            (so that iterative updates take effect).
        dummy_input: Float32 calibration input tensor.
        weight_mapping: List of (torch_param_name, getter, setter) tuples.
            - torch_param_name: key from reference_module.named_parameters()
            - getter() -> torch.Tensor: returns current quantized weight as a
              bfloat16 torch tensor *in the same shape as the torch parameter*.
              Transpositions or reshaping must be handled internally.
            - setter(torch.Tensor) -> None: writes an updated bfloat16 torch
              tensor back to the quantized model. Reverse any transpositions
              or reshaping internally.
        num_iterations: Number of correction rounds. Each round re-evaluates
            the quantized forward pass and recomputes gradients at the current
            weight values, allowing corrections to cascade across interacting
            weights.

    Returns:
        List of MSE values (one per iteration, measured before that iteration's
        corrections), for monitoring convergence.
    """
    reference_module.eval()
    mse_history: List[float] = []

    with torch.no_grad():
        ref_output = reference_module(dummy_input).detach()

    saved_fp32: dict[str, torch.Tensor] = {}
    for name, param in reference_module.named_parameters():
        saved_fp32[name] = param.data.clone()

    for _ in range(num_iterations):
        baseline_output = target_forward_fn(dummy_input)
        baseline_mse = torch.nn.functional.mse_loss(ref_output, baseline_output).item()
        mse_history.append(baseline_mse)

        # Snapshot current quantized weights so the proxy forward pass
        # operates at the same point as the real quantized model.
        current_ttnn: dict[str, torch.Tensor] = {}
        for torch_name, getter, _ in weight_mapping:
            current_ttnn[torch_name] = getter().float()

        # Proxy forward/backward at the current operating point.
        for name, param in reference_module.named_parameters():
            if name in current_ttnn:
                param.data = current_ttnn[name]
            else:
                param.data = param.data.to(torch.bfloat16).to(torch.float32)
            param.requires_grad_(True)

        reference_module.zero_grad()
        proxy_output = reference_module(dummy_input)
        loss = torch.nn.functional.mse_loss(proxy_output, ref_output)
        loss.backward()

        gradients = {}
        for name, param in reference_module.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()

        # Restore original fp32 weights.
        for name, param in reference_module.named_parameters():
            param.data = saved_fp32[name]
            param.grad = None

        # Greedy per-parameter update: apply gradient-suggested rounding,
        # validate against the actual target model, revert if no improvement.
        running_mse = baseline_mse
        for torch_name, getter, setter in weight_mapping:
            if torch_name not in gradients:
                continue

            prev_weight = getter()
            proposed = bf16_gradient_round(saved_fp32[torch_name], gradients[torch_name])
            setter(proposed)

            new_output = target_forward_fn(dummy_input)
            new_mse = torch.nn.functional.mse_loss(ref_output, new_output).item()

            if new_mse < running_mse:
                running_mse = new_mse
            else:
                setter(prev_weight)

    return mse_history
