# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resident composition of the VAE ports.

Each block port (causal conv, RMS norm, residual / mid / up / attention block, resample) is a thin
adapter around a tt_dit Wan body: its __call__ takes a replicated BCTHW tensor, partitions it over the
mesh, runs the body and gathers the result. Inside the encoder / decoder the activation is ALREADY
partitioned (BTHWC shards), so the port is entered through `forward_sharded`, which runs the same
body on the shards and skips the adapter.

`around(port_cls, body, ...)` builds a port instance that owns an existing body (no second copy of
the weights) and routes the body's forward through the port. The encoder3d / decoder3d ports call
`attach_block_ports()` once at build time, so every child of their Wan stack runs as the matching
graduated port.
"""

from __future__ import annotations


class ResidentPort:
    """Mixin for a port whose body lives inside an encoder/decoder stack."""

    BODY_ATTR = "block"

    @classmethod
    def around(cls, body, device, parallel_config=None, ccl_manager=None, **extra):
        self = cls.__new__(cls)
        self.device = device
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        setattr(self, cls.BODY_ATTR, body)
        for k, v in extra.items():
            setattr(self, k, v)
        self._body_forward = type(body).forward
        # late-bound so a subclass swapped in after attach (e.g. an invocation counter) is honoured
        body.forward = lambda *a, **k: self.forward_sharded(*a, **k)  # the Wan stack now calls the port
        return self

    def forward_sharded(self, *args, **kwargs):
        """The graduated body on the already-partitioned activation (BTHWC shards)."""
        return self._body_forward(getattr(self, self.BODY_ATTR), *args, **kwargs)


def ports_of(ports, bodies):
    """The ports wrapping `bodies` (e.g. a Wan ModuleList), in the bodies' order.

    A tt_dit ModuleList keeps its children in `_children`, which a generic walk does not open; this
    plain list of the same port objects is how the stack stays visible (it owns nothing new)."""
    by_body = {id(getattr(p, p.BODY_ATTR)): p for p in ports}
    return [by_body[id(b)] for b in bodies if id(b) in by_body]


def walk(module):
    """Depth-first over a tt_dit Module tree (the root included)."""
    yield module
    for _, child in module.named_children():
        yield from walk(child)


def attach_block_ports(stack, device, parallel_config, ccl_manager, torch_stack=None, sink=None):
    """Route every child of a Wan encoder/decoder `stack` through its graduated port.

    torch_stack: the HF QwenImageEncoder3d / QwenImageDecoder3d, used to build the op-level ports
    (ZeroPad2d, QwenImageUpsample) that sit inside each resample. sink: optional list collecting the
    ports that were created (in stack order).
    """
    from models.tt_dit.layers.normalization import RMSNorm
    from models.tt_dit.models.vae.vae_wan2_1 import (
        WanAttentionBlock,
        WanCausalConv3d,
        WanMidBlock,
        WanResample,
        WanResidualBlock,
        WanUpBlock,
    )
    from models.tt_dit.pipelines.qwen_image_edit_vae._stubs.qwen_image_attention_block import TtQwenImageAttentionBlock
    from models.tt_dit.pipelines.qwen_image_edit_vae._stubs.qwen_image_causal_conv3d import TtQwenImageCausalConv3d
    from models.tt_dit.pipelines.qwen_image_edit_vae._stubs.qwen_image_mid_block import TtQwenImageMidBlock
    from models.tt_dit.pipelines.qwen_image_edit_vae._stubs.qwen_image_r_m_s import TtQwenImageRMSNorm
    from models.tt_dit.pipelines.qwen_image_edit_vae._stubs.qwen_image_resample import TtQwenImageResample
    from models.tt_dit.pipelines.qwen_image_edit_vae._stubs.qwen_image_residual_block import TtQwenImageResidualBlock
    from models.tt_dit.pipelines.qwen_image_edit_vae._stubs.qwen_image_up_block import TtQwenImageUpBlock

    # op-level ports for the resamples, in stack order (HF: resample = Sequential(op, Conv2d))
    torch_ops = []
    if torch_stack is not None:
        for m in torch_stack.modules():
            if type(m).__name__ == "QwenImageResample" and m.mode in (
                "downsample2d",
                "downsample3d",
                "upsample2d",
                "upsample3d",
            ):
                torch_ops.append(m.resample[0])

    table = [
        (WanUpBlock, TtQwenImageUpBlock),
        (WanMidBlock, TtQwenImageMidBlock),
        (WanResidualBlock, TtQwenImageResidualBlock),
        (WanAttentionBlock, TtQwenImageAttentionBlock),
        (WanCausalConv3d, TtQwenImageCausalConv3d),
        (RMSNorm, TtQwenImageRMSNorm),
    ]
    ports = sink if sink is not None else []
    op_i = 0
    for mod in list(walk(stack))[1:]:
        if isinstance(mod, WanResample):
            op = torch_ops[op_i] if op_i < len(torch_ops) else None
            op_i += 1
            ports.append(TtQwenImageResample.around_resample(mod, device, parallel_config, ccl_manager, op))
            continue
        for body_cls, port_cls in table:
            if type(mod) is body_cls:
                ports.append(port_cls.around(mod, device, parallel_config, ccl_manager))
                break
    return ports
