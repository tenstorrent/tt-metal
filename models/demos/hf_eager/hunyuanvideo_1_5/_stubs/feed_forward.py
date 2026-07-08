# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `feed_forward` of tencent/HunyuanVideo-1.5.

Reference submodule: `context_embedder.token_refiner.refiner_blocks.0.ff`, a
diffusers `FeedForward` whose `net` ModuleList is:

    net[0] : LinearActivation(proj=Linear(dim, inner), activation=SiLU())
    net[1] : Dropout(0.0)                                   # identity in eval
    net[2] : Linear(inner, dim)

forward: `for module in net: x = module(x)`  ->  linear_1 -> act -> linear_2.
(Note: the sibling `transformer_blocks.*.ff` instead uses net[0]=GELU(approximate
="tanh"); this port detects the activation from net[0] rather than assuming one.)

Input/output: (B, L, dim). Native ttnn: two `matmul + bias` around the detected
activation (SiLU / GELU-tanh / GELU-erf / ReLU / ...). Dropout is an eval-time
identity and is skipped. Float32 math with a HiFi4 config.
"""

from __future__ import annotations

import ttnn

HF_MODEL_ID = "tencent/HunyuanVideo-1.5"


def _resolve_proj_and_activation(act_wrapper):
    """Return (proj_linear, activation_fn) for net[0].

    diffusers FeedForward wraps the input projection + its activation in a single
    module (GELU / GEGLU / ApproximateGELU / SwiGLU / LinearActivation). All expose
    the projection as `.proj`; the activation is either implicit in the class
    (GELU) or a `.activation` submodule (LinearActivation)."""
    import torch

    name = type(act_wrapper).__name__
    proj = getattr(act_wrapper, "proj", None)

    # diffusers GELU: proj then F.gelu(approximate=...)
    if name == "GELU":
        approx = str(getattr(act_wrapper, "approximate", "none"))
        variant = ttnn.GeluVariant.Tanh if approx == "tanh" else ttnn.GeluVariant.Accurate
        return proj, (lambda t: ttnn.gelu(t, variant=variant))

    # LinearActivation (and similar): proj then a torch/diffusers activation module.
    act_mod = getattr(act_wrapper, "activation", None)
    if act_mod is not None:
        return proj, _map_activation_module(act_mod)

    # Fallback: treat as bare projection with no activation.
    if isinstance(act_wrapper, torch.nn.Linear):
        return act_wrapper, (lambda t: t)
    raise RuntimeError(f"feed_forward: unsupported net[0] activation wrapper `{name}`")


def _map_activation_module(mod):
    """Map a torch/diffusers activation module to the matching ttnn op."""
    name = type(mod).__name__.lower()
    if "silu" in name or "swish" in name:
        return lambda t: ttnn.silu(t)
    if "gelu" in name:
        approx = str(getattr(mod, "approximate", "none"))
        variant = ttnn.GeluVariant.Tanh if approx == "tanh" else ttnn.GeluVariant.Accurate
        return lambda t: ttnn.gelu(t, variant=variant)
    if "relu" in name:
        return lambda t: ttnn.relu(t)
    if "mish" in name:
        return lambda t: ttnn.mish(t)
    if "sigmoid" in name:
        return lambda t: ttnn.sigmoid(t)
    if "tanh" in name:
        return lambda t: ttnn.tanh(t)
    raise RuntimeError(f"feed_forward: unsupported activation module `{type(mod).__name__}`")


def build(device, torch_module):
    """Bind the two FeedForward projections + activation; return a native forward."""
    import torch

    ff = torch_module
    net = ff.net

    lin1, activation_fn = _resolve_proj_and_activation(net[0])

    # Final projection: the last Linear in net.
    lin2 = None
    for module in reversed(list(net)):
        if isinstance(module, torch.nn.Linear):
            lin2 = module
            break
    if lin1 is None or lin2 is None:
        raise RuntimeError("feed_forward: could not locate the two Linear projections in net")

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def f32(t):
        return ttnn.from_torch(t.contiguous().float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    w1 = f32(lin1.weight.detach().t())  # (dim, inner)
    b1 = f32(lin1.bias.detach().reshape(1, -1)) if lin1.bias is not None else None
    w2 = f32(lin2.weight.detach().t())  # (inner, dim)
    b2 = f32(lin2.bias.detach().reshape(1, -1)) if lin2.bias is not None else None

    def _to_f32_device(t):
        if isinstance(t, ttnn.Tensor):
            if t.get_dtype() != ttnn.float32:
                t = ttnn.typecast(t, ttnn.float32)
            return t
        return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    def forward(x, *args, **kwargs):
        x = _to_f32_device(x)
        h = ttnn.matmul(x, w1, compute_kernel_config=compute_config)
        if b1 is not None:
            h = ttnn.add(h, b1)
        h = activation_fn(h)
        h = ttnn.matmul(h, w2, compute_kernel_config=compute_config)
        if b2 is not None:
            h = ttnn.add(h, b2)
        return h

    return forward


def feed_forward(*args, **kwargs):
    raise RuntimeError(
        "feed_forward requires build(device, torch_module) to bind the projections; "
        "the bare callable has no parameters."
    )
