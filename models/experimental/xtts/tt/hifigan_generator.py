"""TTNN port of the XTTS HiFiGAN vocoder (HifiganGenerator) — Phase 3.

Reference: TTS/vocoder/models/hifigan_generator.py. Maps GPT latents (b, 1024, T)
+ speaker embedding g (b, 512, 1) to a waveform (b, 1, T*256):

    o = conv_pre(x) + cond_layer(g)
    for i in range(4):                       # upsample stages
        o = lrelu(o, 0.1); o = ups[i](o)     # ConvTranspose1d
        o = o + conds[i](g)                  # speaker conditioning
        o = mean_j ResBlock1[i*3+j](o)       # 3 residual blocks averaged
    o = lrelu(o, 0.01); o = conv_post(o); o = tanh(o)

Implementation notes:
  * Conv1d -> ttnn.conv1d (input [N,1,L,C], PyTorch-format weight [out,in,1,k]).
  * ConvTranspose1d -> ttnn.conv_transpose2d with height 1.
  * Weight-norm is folded by reading the module's computed `.weight` (PyTorch
    evaluates the parametrization on access), so no manual g*v/||v||.
  * k=1 cond convs are linear over channels (single time step), broadcast over T.
  * Device must be opened with l1_small_size>0 (conv ops need the L1-small region).
"""

import torch
import ttnn

LRELU = 0.1


def _bf16(t, layout=ttnn.ROW_MAJOR_LAYOUT, device=None):
    return ttnn.from_torch(t.to(torch.bfloat16), layout=layout, device=device, dtype=ttnn.bfloat16)


def _conv_w(conv):
    """Folded weight [out,in,1,k] + bias [1,1,1,out] for a (weight-normed) Conv1d."""
    w = conv.weight.detach()  # [out, in, k]  (parametrization applied)
    out, cin, k = w.shape
    b = conv.bias.detach() if conv.bias is not None else torch.zeros(out)
    return (
        w.reshape(out, cin, 1, k).contiguous(),
        b.reshape(1, 1, 1, out).contiguous(),
        dict(cin=cin, cout=out, k=k, pad=conv.padding[0], dil=conv.dilation[0]),
    )


def _convT_w(conv):
    """ConvTranspose1d weight [in,out,1,k] + bias for ttnn.conv_transpose2d."""
    w = conv.weight.detach()  # [in, out, k]
    cin, out, k = w.shape
    b = conv.bias.detach() if conv.bias is not None else torch.zeros(out)
    return (
        w.reshape(cin, out, 1, k).contiguous(),
        b.reshape(1, 1, 1, out).contiguous(),
        dict(cin=cin, cout=out, k=k, stride=conv.stride[0], pad=conv.padding[0], opad=conv.output_padding[0]),
    )


def load_generator_params(gen, device):
    """Extract configs + folded weights from the reference HifiganGenerator module."""

    def conv(c):
        w, b, cfg = _conv_w(c)
        return {"w": _bf16(w), "b": _bf16(b), **cfg}

    def convT(c):
        w, b, cfg = _convT_w(c)
        return {"w": _bf16(w), "b": _bf16(b), **cfg}

    def pointwise(c):  # k=1 conv as linear over channels
        w = c.weight.detach().squeeze(-1)  # [out, in]
        b = c.bias.detach()
        return {
            "w": _bf16(w.t().contiguous(), ttnn.TILE_LAYOUT, device),
            "b": _bf16(b.reshape(1, 1, 1, -1), ttnn.TILE_LAYOUT, device),
        }

    n_up = len(gen.ups)
    n_kern = len(gen.resblocks) // n_up
    resblocks = []
    for rb in gen.resblocks:
        resblocks.append(
            {
                "convs1": [conv(c) for c in rb.convs1],
                "convs2": [conv(c) for c in rb.convs2],
            }
        )
    return {
        "conv_pre": conv(gen.conv_pre),
        "cond_layer": pointwise(gen.cond_layer),
        "ups": [convT(u) for u in gen.ups],
        "conds": [pointwise(c) for c in gen.conds],
        "resblocks": resblocks,
        "conv_post": conv(gen.conv_post),
        "n_up": n_up,
        "n_kern": n_kern,
        "ckc": ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
        # cap the conv activation block height so L1 circular buffers stay bounded
        # regardless of sequence length (after 256x upsampling L gets large)
        "conv_cfg": ttnn.Conv2dConfig(act_block_h_override=32),
    }


def _rm(x):
    return ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT) if x.layout != ttnn.ROW_MAJOR_LAYOUT else x


def _conv1d(x, p, device, L, ckc=None, conv_cfg=None):
    """x: [1,1,L,Cin] -> [1,1,Lout,Cout]."""
    out = ttnn.conv1d(
        input_tensor=_rm(x),
        weight_tensor=p["w"],
        bias_tensor=p["b"],
        in_channels=p["cin"],
        out_channels=p["cout"],
        device=device,
        kernel_size=p["k"],
        stride=1,
        padding=p["pad"],
        dilation=p["dil"],
        batch_size=1,
        input_length=L,
        groups=1,
        compute_config=ckc,
        conv_config=conv_cfg,
    )
    t = out[0] if isinstance(out, (tuple, list)) else out
    t = ttnn.to_layout(ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG), ttnn.TILE_LAYOUT)
    Lout = L + 2 * p["pad"] - p["dil"] * (p["k"] - 1)
    return ttnn.reshape(t, (1, 1, Lout, p["cout"])), Lout


def _convT(x, p, device, L, ckc=None, conv_cfg=None):
    out = ttnn.conv_transpose2d(
        input_tensor=_rm(x),
        weight_tensor=p["w"],
        bias_tensor=p["b"],
        in_channels=p["cin"],
        out_channels=p["cout"],
        device=device,
        kernel_size=(1, p["k"]),
        stride=(1, p["stride"]),
        padding=(0, p["pad"]),
        output_padding=(0, p["opad"]),
        batch_size=1,
        input_height=1,
        input_width=L,
        groups=1,
        compute_config=ckc,
        conv_config=conv_cfg,
    )
    t = out[0] if isinstance(out, (tuple, list)) else out
    t = ttnn.to_layout(ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG), ttnn.TILE_LAYOUT)
    Lout = (L - 1) * p["stride"] - 2 * p["pad"] + (p["k"] - 1) + p["opad"] + 1
    return ttnn.reshape(t, (1, 1, Lout, p["cout"])), Lout


def _pointwise(g_BC, p):
    """g_BC: [1,1,1,Cin] -> [1,1,1,Cout] (k=1 conv == linear over channels)."""
    return ttnn.linear(g_BC, p["w"], bias=p["b"])


def _resblock(x, p, device, L, ckc, conv_cfg):
    for c1, c2 in zip(p["convs1"], p["convs2"]):
        xt = ttnn.leaky_relu(x, LRELU)
        xt, _ = _conv1d(xt, c1, device, L, ckc, conv_cfg)
        xt = ttnn.leaky_relu(xt, LRELU)
        xt, _ = _conv1d(xt, c2, device, L, ckc, conv_cfg)
        x = ttnn.add(xt, x)
    return x


def hifigan_generator(x, g, p, device):
    """x: GPT latents [1, 1024, T]; g: speaker emb [1, 512, 1] -> wav [1, 1, T*256]."""
    ckc, cfg = p["ckc"], p["conv_cfg"]
    T = x.shape[-1]
    o = _bf16(x.permute(0, 2, 1).reshape(1, 1, T, x.shape[1]), device=device)  # [1,1,T,1024]
    g_BC = _bf16(g.reshape(1, 1, 1, g.shape[1]), ttnn.TILE_LAYOUT, device)  # [1,1,1,512]

    o, L = _conv1d(o, p["conv_pre"], device, T, ckc, cfg)
    o = ttnn.add(o, _pointwise(g_BC, p["cond_layer"]))  # broadcast over L

    for i in range(p["n_up"]):
        o = ttnn.leaky_relu(o, LRELU)
        o, L = _convT(o, p["ups"][i], device, L, ckc, cfg)
        o = ttnn.add(o, _pointwise(g_BC, p["conds"][i]))
        z = _resblock(o, p["resblocks"][i * p["n_kern"]], device, L, ckc, cfg)
        for j in range(1, p["n_kern"]):
            z = ttnn.add(z, _resblock(o, p["resblocks"][i * p["n_kern"] + j], device, L, ckc, cfg))
        o = ttnn.multiply(z, 1.0 / p["n_kern"])

    o = ttnn.leaky_relu(o, 0.01)  # default slope (matches reference)
    o, L = _conv1d(o, p["conv_post"], device, L, ckc, cfg)
    o = ttnn.tanh(o)
    return ttnn.to_torch(o).float().reshape(1, L, 1).permute(0, 2, 1)  # [1, 1, L]
