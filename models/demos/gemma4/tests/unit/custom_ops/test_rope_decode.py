# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""RopeDecode (custom op) vs the composite decode RoPE and the torch golden, single chip."""

import pytest
import torch

import ttnn
from models.demos.gemma4.custom_ops.rope_decode import RopeDecode
from models.demos.gemma4.tt.attention.operations import apply_rope_decode_peruser

SINGLE_CHIP = [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")]


def _cos_sin(B, D, partial=None):
    pos = torch.randint(0, 4096, (B,))
    inv = 1.0 / (10000 ** (torch.arange(0, D, 2).float() / D))
    freqs = pos[:, None].float() * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    cos, sin = emb.cos(), emb.sin()
    if partial:  # Gemma global layers: rotary on the first `partial` dims only (cos 1, sin 0 elsewhere)
        cos[:, partial:] = 1.0
        sin[:, partial:] = 0.0
    return cos.reshape(1, B, 1, D).to(torch.bfloat16), sin.reshape(1, B, 1, D).to(torch.bfloat16)


@pytest.mark.parametrize("mode", ["identity", "cos_only", "sin_only"])
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_rope_decode_isolating_cases(mesh_device, device_params, mode):
    """identity (cos=1, sin=0) must return x exactly; cos_only / sin_only isolate the two products."""
    torch.manual_seed(1)
    B, heads, D = 32, 16, 256
    x = torch.randn(1, B, heads, D, dtype=torch.bfloat16)
    cos, sin = _cos_sin(B, D)
    if mode == "identity":
        cos, sin = torch.ones_like(cos), torch.zeros_like(sin)
    elif mode == "cos_only":
        sin = torch.zeros_like(sin)
    else:
        cos = torch.zeros_like(cos)
    golden = RopeDecode.golden(x.float(), cos.float(), sin.float())

    def dev(t):
        return ttnn.from_torch(
            t, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    out = ttnn.to_torch(RopeDecode.op(dev(x), dev(cos), dev(sin))).float()
    diff = (golden - out).abs()
    # error map per tile column (8 columns of 32) and per row block, to localize a reader/compute bug
    cols = diff.reshape(-1, D).reshape(-1, D // 32, 32).amax(dim=(0, 2))
    print(f"[{mode}] max|diff| {diff.max():.4f}  per tile-column max: {[round(float(c), 3) for c in cols]}")
    pcc = torch.corrcoef(torch.stack([golden.flatten(), out.flatten()]))[0, 1].item()
    print(f"[{mode}] pcc {pcc:.6f}")
    if mode == "identity":
        assert diff.max() == 0, f"identity must return x exactly: max|diff|={diff.max()}"
    assert pcc > 0.9999, f"{mode}: pcc={pcc}"


@pytest.mark.parametrize(
    "B, heads, D, partial",
    [
        (32, 16, 256, None),
        (32, 8, 256, None),
        (32, 32, 256, None),
        (32, 16, 512, 256),
        (8, 16, 256, None),
        (32, 2, 256, None),
        (32, 1, 256, None),
        (32, 1, 512, None),
    ],
    ids=["b32h16d256", "b32h8d256", "b32h32d256", "b32h16d512p256", "b8h16d256", "b32h2d256", "b32h1d256", "b32h1d512"],
)
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_rope_decode_matches_composite(mesh_device, device_params, B, heads, D, partial):
    torch.manual_seed(0)
    x = torch.randn(1, B, heads, D, dtype=torch.bfloat16)
    cos, sin = _cos_sin(B, D, partial)
    golden = RopeDecode.golden(x.float(), cos.float(), sin.float())

    def dev(t):
        return ttnn.from_torch(
            t, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    tx, tc, ts = dev(x), dev(cos), dev(sin)
    ref = ttnn.to_torch(apply_rope_decode_peruser(tx, tc, ts)).float()
    out = ttnn.to_torch(RopeDecode.op(tx, tc, ts)).float()
    assert torch.isfinite(out).all()
    for name, other in (("golden", golden), ("composite", ref)):
        pcc = torch.corrcoef(torch.stack([other.flatten(), out.flatten()]))[0, 1].item()
        print(f"[{B}x{heads}x{D} partial={partial}] pcc vs {name}: {pcc:.6f} max|diff| {(other - out).abs().max():.4f}")
        assert pcc > 0.9999, f"RopeDecode diverges from the {name}: pcc={pcc}"
    # program-cache hit on freshly allocated inputs (new buffer addresses must be re-applied as runtime args)
    out2 = ttnn.to_torch(RopeDecode.op(dev(x), dev(cos), dev(sin))).float()
    assert torch.equal(out2, out), f"cache-hit call differs: max|diff| {(out2 - out).abs().max()}"


@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_rope_decode_perf_vs_composite(mesh_device, device_params):
    """Both ops on the same b32h16d256 inputs, ITERS times each, between tracy signposts. Device time per call comes
    from the device profiler run (see the module docstring of ``test_decode_roofline.py``):
    ``TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r -v -m pytest <this file> -k perf`` then
    ``python tools/tracy/process_ops_logs.py --date`` and sum DEVICE KERNEL DURATION per signpost block / ITERS.
    Outside tracy it still prints the host wall time per call (dispatch-bound for ops this small)."""
    import time

    try:
        from tracy import signpost
    except ImportError:  # not under python -m tracy: still a valid timing run

        def signpost(*_):
            pass

    ITERS = 20
    torch.manual_seed(0)
    B, heads, D = 32, 16, 256
    x = torch.randn(1, B, heads, D, dtype=torch.bfloat16)
    cos, sin = _cos_sin(B, D)

    def dev(t):
        return ttnn.from_torch(
            t, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    tx, tc, ts = dev(x), dev(cos), dev(sin)
    results = {}
    for name, fn in (
        ("composite", lambda: apply_rope_decode_peruser(tx, tc, ts)),
        ("rope_decode", lambda: RopeDecode.op(tx, tc, ts)),
    ):
        for _ in range(3):  # warm (kernel compile, program cache)
            fn()
        ttnn.synchronize_device(mesh_device)
        signpost(f"{name}_start")
        t0 = time.perf_counter()
        for _ in range(ITERS):
            out = fn()
        ttnn.synchronize_device(mesh_device)
        host_us = (time.perf_counter() - t0) / ITERS * 1e6
        signpost(f"{name}_end")
        results[name] = host_us
        print(f"[perf b32h16d256] {name}: host {host_us:.1f} us/call over {ITERS} calls", flush=True)
        ttnn.deallocate(out)
    golden = RopeDecode.golden(x.float(), cos.float(), sin.float())
    out = ttnn.to_torch(RopeDecode.op(tx, tc, ts)).float()
    pcc = torch.corrcoef(torch.stack([golden.flatten(), out.flatten()]))[0, 1].item()
    assert pcc > 0.9999, f"perf run output diverged: pcc={pcc}"
