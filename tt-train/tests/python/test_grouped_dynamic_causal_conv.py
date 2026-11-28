# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""``ttml.ops.conv.grouped_dynamic_causal_conv`` against z-lab DFlash 2's convolution, in the shapes post-training uses.

The torch reference is upstream's ``_grouped_dynamic_convolve`` and ``GroupedDynamicCausalConv``
(https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py, MIT), run per
block: a DFlash 2 drafter convolves each draft block on its own, so training puts one block per batch row and
ttml takes the same rows flattened, ``[1, 1, blocks * block_size, C]``.

The tests go from the op alone, to upstream's module (``kernel_projection`` + ``prepare`` + ``finish``), to a DFlash 2
decoder layer with attention swapped for its output projection, to a few AdamW steps on that layer.
"""

import numpy as np
import pytest
import torch
from torch.nn import functional as F

import ttnn
import ttml
from ttml.modules import AbstractModuleBase, Parameter

pytestmark = pytest.mark.requires_device

BF16 = ttnn.DataType.BFLOAT16
FP32 = ttnn.DataType.FLOAT32
TILE = ttnn.Layout.TILE

BLOCK = 8
GROUP = 16
KERNEL = 2
EPS = 1e-5

# bf16 compute: the bound is relative to the tensor's largest magnitude; PCC guards small entries.
REL_OF_MAX = 2e-2
MIN_PCC = 0.999


@pytest.fixture(autouse=True)
def _reset_graph():
    yield
    ttml.autograd.AutoContext.get_instance().reset_graph()


# --- torch reference (upstream, formatting only) -------------------------------------------------------------------


def _grouped_dynamic_convolve(hidden, dynamic, base, group_size):
    batch, length, hidden_size = hidden.shape
    groups = hidden_size // group_size
    blocks = hidden.view(batch, length, groups, group_size)
    dynamic = dynamic.view(batch, length, base.shape[0], groups, 1)
    output = torch.zeros_like(blocks)
    for offset in range(base.shape[0]):
        values = blocks if offset == 0 else F.pad(blocks[:, :-offset], (0, 0, 0, 0, offset, 0))
        kernel = base[offset].view(1, 1, groups, group_size).to(hidden.dtype)
        output = output + kernel * values
        output = torch.addcmul(output, dynamic[:, :, offset], values)
    return output.view_as(hidden)


def _rmsnorm(x, gamma):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * gamma


# --- helpers ---------------------------------------------------------------------------------------------------------


def _bf16(shape, scale=1.0, *, generator):
    """Uniform values rounded to bf16, so torch and the device start from the same numbers."""
    t = (torch.rand(shape, generator=generator) * 2 - 1) * scale
    return t.to(torch.bfloat16).to(torch.float32)


def _ttml(t, requires_grad=True):
    """torch [..., C] -> ttml [1, 1, rows, C] TILE bf16 (weights [1, 1, out, in])."""
    a = t.detach().numpy()
    a = a.reshape(1, 1, -1, a.shape[-1]) if a.ndim != 4 else a
    tensor = ttml.autograd.Tensor.from_numpy(np.ascontiguousarray(a), layout=TILE, new_type=BF16)
    tensor.set_requires_grad(requires_grad)
    return tensor


def _np(tensor, shape):
    return tensor.to_numpy(FP32).reshape(shape)


def _grad(tensor, shape):
    return tensor.get_grad_tensor().to_numpy(FP32).reshape(shape)


def _assert_close(actual, expected, what):
    expected = expected.detach().numpy() if isinstance(expected, torch.Tensor) else expected
    tolerance = REL_OF_MAX * np.abs(expected).max()
    max_err = np.abs(actual - expected).max()
    assert max_err <= tolerance, f"{what}: max abs err {max_err} > {tolerance}"
    pcc = np.corrcoef(actual.ravel(), expected.ravel())[0, 1]
    assert pcc > MIN_PCC, f"{what}: pcc {pcc} <= {MIN_PCC}"


def _backward(out, grad_out):
    out.set_grad_from_tensor(_ttml(grad_out, requires_grad=False))
    out.backward(False)


def _conv(x, dynamic, base):
    return ttml.ops.conv.grouped_dynamic_causal_conv(x, dynamic, base, BLOCK, GROUP)


# --- the op ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "blocks, hidden",
    [(64, 512), (512, 6144)],
    ids=["64x512", "b512x6144"],
)
def test_op_matches_upstream(blocks, hidden):
    g = torch.Generator().manual_seed(0)
    groups = hidden // GROUP
    x = _bf16((blocks, BLOCK, hidden), generator=g).requires_grad_()
    dyn = _bf16((blocks, BLOCK, KERNEL * groups), generator=g).requires_grad_()
    base = _bf16((KERNEL, hidden), generator=g).requires_grad_()
    grad_out = _bf16((blocks, BLOCK, hidden), generator=g)

    out = _grouped_dynamic_convolve(x, dyn, base, GROUP)
    out.backward(grad_out)

    x_t, dyn_t, base_t = _ttml(x), _ttml(dyn), _ttml(base.view(1, 1, KERNEL, hidden))
    out_t = _conv(x_t, dyn_t, base_t)
    _assert_close(_np(out_t, out.shape), out, "out")
    _backward(out_t, grad_out)
    _assert_close(_grad(x_t, x.shape), x.grad, "grad input")
    _assert_close(_grad(dyn_t, dyn.shape), dyn.grad, "grad dynamic kernel")
    _assert_close(_grad(base_t, base.shape), base.grad, "grad base kernel")


# --- upstream's module: kernel_projection, prepare, finish -----------------------------------------------------------


class TorchConv(torch.nn.Module):
    """Upstream ``GroupedDynamicCausalConv``."""

    def __init__(self, hidden, g):
        super().__init__()
        groups = hidden // GROUP
        self.base_kernel = torch.nn.Parameter(_bf16((2, KERNEL, hidden), 0.5, generator=g))
        self.kernel_projection = torch.nn.Linear(hidden, 2 * KERNEL * groups, bias=False)
        with torch.no_grad():
            self.kernel_projection.weight.copy_(_bf16(self.kernel_projection.weight.shape, hidden**-0.5, generator=g))

    def prepare(self, hidden):
        groups = hidden.shape[-1] // GROUP
        dynamic = self.kernel_projection(hidden).view(*hidden.shape[:-1], 2, KERNEL, groups)
        return (
            _grouped_dynamic_convolve(hidden, dynamic[..., 0, :, :], self.base_kernel[0], GROUP),
            dynamic[..., 1, :, :],
        )

    def finish(self, hidden, dynamic):
        return _grouped_dynamic_convolve(hidden, dynamic, self.base_kernel[1], GROUP)


class TtmlConv:
    """The same module on ttml. ``kernel_projection`` is held as its prepare and finish halves (rows ``[:K*G]`` and
    ``[K*G:]`` of upstream's weight), each a linear, so no autograd slice is needed between them."""

    def __init__(self, ref: TorchConv):
        hidden = ref.base_kernel.shape[-1]
        half = ref.kernel_projection.weight.shape[0] // 2
        w = ref.kernel_projection.weight.detach()
        self.projection = [_ttml(w[:half].view(1, 1, half, hidden)), _ttml(w[half:].view(1, 1, half, hidden))]
        b = ref.base_kernel.detach()
        self.base = [_ttml(b[0].view(1, 1, KERNEL, hidden)), _ttml(b[1].view(1, 1, KERNEL, hidden))]

    def prepare(self, hidden):
        dyn = [ttml.ops.linear.linear(hidden, w, None) for w in self.projection]
        return _conv(hidden, dyn[0], self.base[0]), dyn[1]

    def finish(self, hidden, dynamic):
        return _conv(hidden, dynamic, self.base[1])

    def check_grads(self, ref: TorchConv, what):
        hidden = ref.base_kernel.shape[-1]
        half = ref.kernel_projection.weight.shape[0] // 2
        grad_w = ref.kernel_projection.weight.grad
        for i, (w, rows) in enumerate(zip(self.projection, (slice(0, half), slice(half, None)))):
            _assert_close(_grad(w, (half, hidden)), grad_w[rows], f"{what} kernel_projection half {i}")
        for i, b in enumerate(self.base):
            _assert_close(_grad(b, (KERNEL, hidden)), ref.base_kernel.grad[i], f"{what} base_kernel[{i}]")


def test_module_prepare_finish_matches_upstream():
    blocks, hidden = 64, 512
    g = torch.Generator().manual_seed(1)
    ref = TorchConv(hidden, g)
    mix = _bf16((hidden, hidden), hidden**-0.5, generator=g).requires_grad_()  # stands in for attention
    x = _bf16((blocks, BLOCK, hidden), generator=g).requires_grad_()
    grad_out = _bf16((blocks, BLOCK, hidden), generator=g)

    prepared, dynamic = ref.prepare(x)
    out = ref.finish(prepared @ mix.T, dynamic)
    out.backward(grad_out)

    conv = TtmlConv(ref)
    x_t, mix_t = _ttml(x), _ttml(mix.view(1, 1, hidden, hidden))
    prepared_t, dynamic_t = conv.prepare(x_t)
    out_t = conv.finish(ttml.ops.linear.linear(prepared_t, mix_t, None), dynamic_t)
    _assert_close(_np(out_t, out.shape), out, "out")
    _backward(out_t, grad_out)
    _assert_close(_grad(x_t, x.shape), x.grad, "grad input")
    _assert_close(_grad(mix_t, mix.shape), mix.grad, "grad mix")
    conv.check_grads(ref, "conv")


# --- a DFlash 2 decoder layer, attention replaced by its output projection -------------------------------------------


class TorchLayer(torch.nn.Module):
    """Upstream's decoder layer order around the convolutions (``dflash/model.py``), with ``o_proj`` in place of
    attention: attention has its own fused fw/bw ops in ttml and does not touch the convolution's gradients."""

    def __init__(self, hidden, intermediate, g):
        super().__init__()
        s = hidden**-0.5
        self.input_norm = torch.nn.Parameter(1 + _bf16((hidden,), 0.1, generator=g))
        self.post_norm = torch.nn.Parameter(1 + _bf16((hidden,), 0.1, generator=g))
        self.o_proj = torch.nn.Parameter(_bf16((hidden, hidden), s, generator=g))
        self.gate = torch.nn.Parameter(_bf16((intermediate, hidden), s, generator=g))
        self.up = torch.nn.Parameter(_bf16((intermediate, hidden), s, generator=g))
        self.down = torch.nn.Parameter(_bf16((hidden, intermediate), intermediate**-0.5, generator=g))
        self.attention_conv = TorchConv(hidden, g)
        self.mlp_conv = TorchConv(hidden, g)

    def forward(self, x):
        normed, dynamic = self.attention_conv.prepare(_rmsnorm(x, self.input_norm))
        h = x + self.attention_conv.finish(normed @ self.o_proj.T, dynamic)
        normed, dynamic = self.mlp_conv.prepare(_rmsnorm(h, self.post_norm))
        mlp = (F.silu(normed @ self.gate.T) * (normed @ self.up.T)) @ self.down.T
        return h + self.mlp_conv.finish(mlp, dynamic)


class TtmlLayer(AbstractModuleBase):
    def __init__(self, ref: TorchLayer):
        super().__init__()
        hidden = ref.o_proj.shape[0]
        self.input_norm = Parameter(_ttml(ref.input_norm.view(1, 1, 1, hidden)))
        self.post_norm = Parameter(_ttml(ref.post_norm.view(1, 1, 1, hidden)))
        self.o_proj = Parameter(_ttml(ref.o_proj.view(1, 1, *ref.o_proj.shape)))
        self.gate = Parameter(_ttml(ref.gate.view(1, 1, *ref.gate.shape)))
        self.up = Parameter(_ttml(ref.up.view(1, 1, *ref.up.shape)))
        self.down = Parameter(_ttml(ref.down.view(1, 1, *ref.down.shape)))
        self.attention_conv = TtmlConv(ref.attention_conv)
        self.mlp_conv = TtmlConv(ref.mlp_conv)
        for name, conv in (("attention_conv", self.attention_conv), ("mlp_conv", self.mlp_conv)):
            for i, w in enumerate(conv.projection):
                setattr(self, f"{name}_projection_{i}", Parameter(w))
            for i, b in enumerate(conv.base):
                setattr(self, f"{name}_base_{i}", Parameter(b))

    def forward(self, x):
        linear = ttml.ops.linear.linear
        normed, dynamic = self.attention_conv.prepare(ttml.ops.rmsnorm.rmsnorm(x, self.input_norm.tensor, EPS))
        attn = self.attention_conv.finish(linear(normed, self.o_proj.tensor, None), dynamic)
        h = ttml.ops.binary.add(x, attn)
        normed, dynamic = self.mlp_conv.prepare(ttml.ops.rmsnorm.rmsnorm(h, self.post_norm.tensor, EPS))
        mlp = ttml.ops.swiglu.swiglu(normed, self.gate.tensor, self.down.tensor, self.up.tensor)
        return ttml.ops.binary.add(h, self.mlp_conv.finish(mlp, dynamic))


def test_decoder_layer_matches_torch():
    blocks, hidden, intermediate = 64, 512, 1024
    g = torch.Generator().manual_seed(2)
    ref = TorchLayer(hidden, intermediate, g)
    x = _bf16((blocks, BLOCK, hidden), generator=g).requires_grad_()
    grad_out = _bf16((blocks, BLOCK, hidden), generator=g)
    out = ref(x)
    out.backward(grad_out)

    layer = TtmlLayer(ref)
    x_t = _ttml(x)
    out_t = layer.forward(x_t)
    _assert_close(_np(out_t, out.shape), out, "out")
    _backward(out_t, grad_out)
    _assert_close(_grad(x_t, x.shape), x.grad, "grad input")
    for name in ("input_norm", "post_norm", "o_proj", "gate", "up", "down"):
        p = getattr(ref, name)
        _assert_close(_grad(getattr(layer, name).tensor, p.shape), p.grad, f"grad {name}")
    layer.attention_conv.check_grads(ref.attention_conv, "attention_conv")
    layer.mlp_conv.check_grads(ref.mlp_conv, "mlp_conv")


def test_adamw_steps_reduce_loss():
    """A few optimizer steps through the layer: every parameter, including both convolutions' kernels, receives a
    usable gradient, so fitting a fixed target drives the loss down."""
    blocks, hidden, intermediate = 64, 512, 1024
    g = torch.Generator().manual_seed(3)
    layer = TtmlLayer(TorchLayer(hidden, intermediate, g))
    x_np = _bf16((1, 1, blocks * BLOCK, hidden), generator=g).numpy()
    target_np = _bf16((1, 1, blocks * BLOCK, hidden), generator=g).numpy()
    params = layer.parameters()
    before = {name: p.to_numpy(FP32) for name, p in params.items()}
    cfg = ttml.optimizers.AdamWConfig.make(lr=3e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0)
    opt = ttml.optimizers.AdamW(params, cfg)
    ctx = ttml.autograd.AutoContext.get_instance()

    losses = []
    for _ in range(30):
        opt.zero_grad()
        x = ttml.autograd.Tensor.from_numpy(x_np, layout=TILE, new_type=BF16)
        target = ttml.autograd.Tensor.from_numpy(target_np, layout=TILE, new_type=BF16)
        loss = ttml.ops.loss.mse_loss(layer.forward(x), target)
        losses.append(float(loss.to_numpy(FP32).ravel()[0]))
        loss.backward(False)
        opt.step()
        ctx.reset_graph()

    assert all(np.isfinite(losses)), losses
    assert losses[-1] < 0.5 * losses[0], f"loss {losses[0]:.4f} -> {losses[-1]:.4f}"
    unchanged = [name for name, p in params.items() if np.array_equal(p.to_numpy(FP32), before[name])]
    assert not unchanged, f"parameters never updated: {unchanged}"
