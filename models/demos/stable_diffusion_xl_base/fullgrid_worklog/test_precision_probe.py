# Synthetic precision probe: which op / fidelity systematically shrinks its output (std ratio < 1)?
# Shapes are the SDXL 1024x1024 transformer-block shapes. Compares device output against torch fp32.
import os

import pytest
import torch

import ttnn

torch.manual_seed(0)


def stats(name, out, ref):
    out = out.float().reshape(ref.shape)
    e = out - ref
    # least-squares gain: how much the device output is scaled vs the reference
    gain = (out * ref).sum() / (ref * ref).sum()
    pcc = torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1]
    print(
        f"PROBE {name:60s} std_ratio={out.std()/ref.std():.5f} gain={gain:.5f} bias={e.mean():+.6f} "
        f"rms={e.pow(2).mean().sqrt():.5f} rel_rms={(e.pow(2).mean().sqrt()/ref.std()):.5f} pcc={pcc:.6f}"
    )


def ckc(fid, fp32, l1acc=True, approx=False):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, fid),
        math_approx_mode=approx,
        fp32_dest_acc_en=fp32,
        packer_l1_acc=l1acc,
    )


@pytest.mark.parametrize("shape", [(1, 20, 1024, 64), (1, 10, 4096, 64)])
def test_sdpa(device, shape):
    B, H, S, D = shape
    # realistic magnitudes: ln output ~ unit, q/k after projection ~ N(0, 1)
    q = torch.randn(shape) * float(os.environ.get("QK_STD", "1.0"))
    k = torch.randn(shape) * float(os.environ.get("QK_STD", "1.0"))
    v = torch.randn(shape)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    tq, tk, tv = [ttnn.from_torch(t, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT) for t in (q, k, v)]
    # bf16-rounded torch reference isolates the compute error from the input quantization
    ref_bf = torch.nn.functional.scaled_dot_product_attention(*(t.to(torch.bfloat16).float() for t in (q, k, v)))
    stats(f"sdpa{shape} torch-bf16-inputs", ref_bf, ref)
    kchunk = 1024 if S == 1024 else 512
    for exp_approx in (False, True):
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(11, 10), q_chunk_size=128, k_chunk_size=kchunk, exp_approx_mode=exp_approx
        )
        for fid in ("LoFi", "HiFi2", "HiFi4"):
            for fp32 in (False, True):
                out = ttnn.transformer.scaled_dot_product_attention(
                    tq,
                    tk,
                    tv,
                    is_causal=False,
                    program_config=pc,
                    compute_kernel_config=ckc(fid, fp32),
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                stats(f"sdpa{shape} {fid} fp32={int(fp32)} exp_approx={int(exp_approx)}", ttnn.to_torch(out), ref)
                ttnn.deallocate(out)


@pytest.mark.parametrize("mkn", [(1024, 1280, 1280), (1024, 1280, 5120), (4096, 640, 640)])
def test_matmul(device, mkn):
    M, K, N = mkn
    a = torch.randn(1, 1, M, K)
    w = torch.randn(1, 1, K, N) * (1.0 / K**0.5)
    ref = a @ w
    ta = ttnn.from_torch(a, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    for wdt in ("bfloat8_b", "bfloat16"):
        tw = ttnn.from_torch(w, getattr(ttnn, wdt), device=device, layout=ttnn.TILE_LAYOUT)
        stats(f"mm{mkn} weights-only {wdt} (torch)", a @ ttnn.to_torch(tw).float(), ref)
        for fid in ("LoFi", "HiFi2", "HiFi4"):
            for fp32 in (False, True):
                for l1acc in (True, False):
                    out = ttnn.matmul(
                        ta, tw, compute_kernel_config=ckc(fid, fp32, l1acc), memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
                    stats(f"mm{mkn} w={wdt} {fid} fp32={int(fp32)} l1acc={int(l1acc)}", ttnn.to_torch(out), ref)
                    ttnn.deallocate(out)


def test_layernorm(device):
    x = torch.randn(1, 1, 1024, 1280) * 3 + 0.5
    g = torch.randn(1280) * 0.5 + 1
    b = torch.randn(1280) * 0.1
    ref = torch.nn.functional.layer_norm(x, (1280,), g, b, 1e-5)
    tx = ttnn.from_torch(x, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tg = ttnn.from_torch(g, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tb = ttnn.from_torch(b, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    for fid in ("HiFi2", "HiFi4"):
        for fp32 in (False, True):
            out = ttnn.layer_norm(
                tx,
                weight=tg,
                bias=tb,
                epsilon=1e-5,
                compute_kernel_config=ckc(fid, fp32),
                program_config=ttnn.LayerNormDefaultProgramConfig(legacy_reduction=True, legacy_rsqrt=True),
            )
            stats(f"ln {fid} fp32={int(fp32)} legacy", ttnn.to_torch(out), ref)
            out = ttnn.layer_norm(tx, weight=tg, bias=tb, epsilon=1e-5, compute_kernel_config=ckc(fid, fp32))
            stats(f"ln {fid} fp32={int(fp32)} default", ttnn.to_torch(out), ref)


def test_gelu_mul(device):
    x = torch.randn(1, 1, 1024, 5120)
    g = torch.randn(1, 1, 1024, 5120)
    ref = x * torch.nn.functional.gelu(g)
    tx = ttnn.from_torch(x, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tg = ttnn.from_torch(g, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    for fast in (True, False):
        gg = ttnn.gelu(tg, fast_and_approximate_mode=fast)
        stats(f"gelu fast={int(fast)}", ttnn.to_torch(gg), torch.nn.functional.gelu(g))
        for fam in (True, False):
            out = ttnn.mul(tx, gg, fast_and_approximate_mode=fam)
            stats(f"x*gelu gelu_fast={int(fast)} mul_fast={int(fam)}", ttnn.to_torch(out), ref)


def _round_mantissa(t, bits):
    """Round a bf16-representable tensor to `bits` explicit mantissa bits (round to nearest even)."""
    t32 = t.float().contiguous()
    i = t32.view(torch.int32)
    drop = 23 - bits
    half = 1 << (drop - 1)
    mask = ~((1 << drop) - 1)
    lsb = (i >> drop) & 1
    i = (i + half - 1 + lsb) & mask
    return i.view(torch.float32)


def _trunc_mantissa(t, bits):
    t32 = t.float().contiguous()
    i = t32.view(torch.int32)
    drop = 23 - bits
    mask = ~((1 << drop) - 1)
    return (i & mask).view(torch.float32)


@pytest.mark.parametrize("shape", [(1, 20, 1024, 64)])
def test_sdpa_preround(device, shape):
    """Does rounding q/k/v (RNE) to N mantissa bits before a LoFi SDPA remove the shrinkage? If LoFi = truncation
    of the operands to N bits, the pre-rounded inputs pass through untouched and the gain returns to ~1."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(shape) for _ in range(3))
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(11, 10), q_chunk_size=128, k_chunk_size=1024, exp_approx_mode=False
    )
    for bits in (7, 6, 5, 4, 3):
        qr, kr, vr = (_round_mantissa(t, bits) for t in (q, k, v))
        ref_r = torch.nn.functional.scaled_dot_product_attention(qr, kr, vr)
        stats(f"preround{bits} torch(rounded inputs) vs torch", ref_r, ref)
        tq, tk, tv = [ttnn.from_torch(t, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT) for t in (qr, kr, vr)]
        out = ttnn.transformer.scaled_dot_product_attention(
            tq,
            tk,
            tv,
            is_causal=False,
            program_config=pc,
            compute_kernel_config=ckc("LoFi", False),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        stats(f"preround{bits} LoFi vs torch(rounded inputs)", ttnn.to_torch(out), ref_r)
        stats(f"preround{bits} LoFi vs torch", ttnn.to_torch(out), ref)
    # torch emulation of operand truncation: truncate q,k to N bits, softmax in fp32, truncate p and v to N bits
    for bits in (5, 4, 3):
        qt, kt, vt = (_trunc_mantissa(t.to(torch.bfloat16), bits) for t in (q, k, v))
        s = (qt @ kt.transpose(-1, -2)) / 8.0
        p = torch.softmax(s, -1)
        o = _trunc_mantissa(p.to(torch.bfloat16), bits) @ vt
        stats(f"emulate trunc{bits} (q,k,p,v truncated) vs torch", o, ref)
        o2 = p.to(torch.bfloat16).float() @ vt
        stats(f"emulate trunc{bits} (q,k,v truncated; p bf16) vs torch", o2, ref)


@pytest.mark.parametrize("mkn", [(1024, 1280, 1280)])
def test_matmul_preround(device, mkn):
    M, K, N = mkn
    torch.manual_seed(0)
    a = torch.randn(1, 1, M, K)
    w = torch.randn(1, 1, K, N) * (1.0 / K**0.5)
    ref = a @ w
    for bits in (7, 6, 5, 4, 3):
        ar, wr = _round_mantissa(a, bits), _round_mantissa(w, bits)
        ta = ttnn.from_torch(ar, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        tw = ttnn.from_torch(wr, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        out = ttnn.matmul(ta, tw, compute_kernel_config=ckc("LoFi", True, False), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        stats(f"mm preround{bits} LoFi fp32 vs torch(rounded inputs)", ttnn.to_torch(out), ar @ wr)
        stats(f"mm preround{bits} LoFi fp32 vs torch", ttnn.to_torch(out), ref)
    # only one operand rounded: which side is truncated how much?
    for bits in (5, 4):
        for side in ("a", "w"):
            ar = _round_mantissa(a, bits) if side == "a" else a
            wr = _round_mantissa(w, bits) if side == "w" else w
            ta = ttnn.from_torch(ar, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
            tw = ttnn.from_torch(wr, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
            out = ttnn.matmul(
                ta, tw, compute_kernel_config=ckc("LoFi", True, False), memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            stats(
                f"mm preround{bits} only {side} LoFi fp32 vs torch(rounded)",
                ttnn.to_torch(out),
                ar.to(torch.bfloat16).float() @ wr.to(torch.bfloat16).float(),
            )


@pytest.mark.parametrize("cin,cout,hw", [(320, 320, 64), (320, 32, 128), (32, 320, 128)])
def test_conv(device, cin, cout, hw):
    """conv2d 3x3 gain per compute config (model uses HiFi2 with fp32 on/off, l1acc on/off; conv_in/out use fp32 off + l1acc off)."""
    torch.manual_seed(0)
    x = torch.randn(1, cin, hw, hw)
    w = torch.randn(cout, cin, 3, 3) / (9 * cin) ** 0.5
    b = torch.randn(cout) * 0.1
    ref = torch.nn.functional.conv2d(x, w, b, padding=1)
    tx = ttnn.from_torch(
        x.permute(0, 2, 3, 1).reshape(1, 1, hw * hw, cin), ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
    )
    for wdt in ("bfloat16", "bfloat8_b"):
        for fid, fp32, l1 in (
            ("HiFi2", True, False),
            ("HiFi2", False, True),
            ("HiFi2", False, False),
            ("LoFi", True, False),
            ("HiFi4", True, False),
            ("HiFi4", False, False),
        ):
            tw = ttnn.from_torch(w, ttnn.bfloat16)
            tb = ttnn.from_torch(b.reshape(1, 1, 1, -1), ttnn.bfloat16)
            out = ttnn.conv2d(
                input_tensor=tx,
                weight_tensor=tw,
                bias_tensor=tb,
                in_channels=cin,
                out_channels=cout,
                device=device,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                batch_size=1,
                input_height=hw,
                input_width=hw,
                conv_config=ttnn.Conv2dConfig(
                    weights_dtype=getattr(ttnn, wdt), shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED
                ),
                compute_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=getattr(ttnn.MathFidelity, fid),
                    math_approx_mode=True,
                    fp32_dest_acc_en=fp32,
                    packer_l1_acc=l1,
                ),
                dtype=ttnn.bfloat16,
            )
            o = ttnn.to_torch(out).float()[..., :cout].reshape(1, hw, hw, cout).permute(0, 3, 1, 2)
            stats(f"conv {cin}->{cout} @{hw} w={wdt} {fid} fp32={int(fp32)} l1acc={int(l1)}", o, ref)


def test_matmul_kblocks(device):
    """Gain vs number of K blocks for the no-fp32 / no-l1acc path (partials spilled to L1 as bf16 and reloaded)."""
    torch.manual_seed(0)
    M, K, N = 256, 1280, 1280
    a = torch.randn(1, 1, M, K)
    w = torch.randn(1, 1, K, N) * (1.0 / K**0.5)
    ref = a @ w
    ta = ttnn.from_torch(a, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tw = ttnn.from_torch(w, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    for ibw in (40, 20, 10, 5, 2, 1):
        pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=ibw,
            out_subblock_h=1,
            out_subblock_w=5,
            per_core_M=1,
            per_core_N=5,
            transpose_mcast=False,
            fused_activation=None,
        )
        for fid, fp32, l1 in (
            ("HiFi2", False, False),
            ("HiFi2", False, True),
            ("HiFi4", False, False),
            ("HiFi2", True, False),
        ):
            sb = 5 if not fp32 else 1
            pc2 = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=ibw,
                out_subblock_h=1,
                out_subblock_w=sb,
                per_core_M=1,
                per_core_N=5,
                transpose_mcast=False,
                fused_activation=None,
            )
            out = ttnn.matmul(
                ta,
                tw,
                program_config=pc2,
                compute_kernel_config=ckc(fid, fp32, l1),
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            stats(
                f"kblocks={40//ibw:2d} (in0_block_w={ibw:2d}) {fid} fp32={int(fp32)} l1acc={int(l1)}",
                ttnn.to_torch(out),
                ref,
            )
            ttnn.deallocate(out)
