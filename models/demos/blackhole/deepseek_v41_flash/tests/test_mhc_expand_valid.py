"""Spec A gate: the valid-row mHC expand (mhc_expand3 into a mhc_stream_pool buffer, DSV41_MHC_EXPAND_VALID=1) is BIT-IDENTICAL to mhc_expand2 over the WHOLE
output tile tensor (padding rows included) for T = 1..16, 3 seeds, the three y variants of the decode layer; the padding stays exactly 0 after 100 chained
calls (attn / ffn roles as in the layer); a NaN-poisoned fresh buffer shows the pool is needed (padding keeps the NaNs, only the valid pieces are written)."""

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import mhc_stream_pool
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_expand2 import mhc_expand2
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_expand3 import mhc_expand3

D = 5120


def full(t):
    """the WHOLE tile data of a tile tensor, padding rows included: the tensor viewed with its padded shape ([T,1,32,D]); device 0"""
    return ttnn.to_torch(ttnn.get_device_tensors(t.reshape(t.padded_shape))[0]).float()


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 100_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_mhc_expand_valid(mesh_device):
    md = mesh_device
    rep = ttnn.ReplicateTensorToMesh(md)

    def up(t, dt=ttnn.float32):
        return ttnn.from_torch(
            t.contiguous(),
            device=md,
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )

    def stream(T):
        s = torch.zeros(T, 1, 4, D)
        s[:] = torch.randn(T, 1, 4, D)
        return up(s)  # TILE layout pads rows 4..31 with zeros

    bad = tot = 0
    for seed in range(3):
        torch.manual_seed(100 + seed)
        for T in range(1, 17):
            x = stream(T)
            post, comb = up(torch.rand(T, 1, 4, 1) * 2), up(torch.rand(T, 1, 4, 4))
            variants = {
                "ybf_y2": (up(torch.randn(1, 1, T, D), ttnn.bfloat16), up(torch.randn(1, 1, T, D))),
                "yf_tok": (up(torch.randn(T, 1, 1, D)), None),
                "yf_row_y2": (up(torch.randn(1, 1, T, D)), up(torch.randn(1, 1, T, D))),
            }
            for name, (y, y2) in variants.items():
                if T % 4 and T > 8:
                    pass  # all T are supported by the kernels; odd T goes through the padded wrapper in the layer, tested at op level here
                ref = full(mhc_expand2(y, x, post, comb, y2))
                buf = mhc_stream_pool.get(md, T, D, f"test{seed}", name)
                got = full(mhc_expand3(y, x, post, comb, buf, y2))
                tot += 1
                if not torch.equal(ref, got):
                    bad += 1
                    print("DIFF T", T, name, (ref - got).abs().max().item(), flush=True)
    print(f"VALID_EXPAND differing {bad} of {tot}", flush=True)
    assert bad == 0
    mhc_stream_pool.reset(md)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 100_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_mhc_expand_valid_chain(mesh_device):
    """100 chained attn/ffn expands through the pool (the layer's data flow: x -> A -> B -> A ...): padding exactly 0, valid rows equal the plain expand chain;
    plus the NaN-poisoned buffer check."""
    md = mesh_device
    rep = ttnn.ReplicateTensorToMesh(md)

    def up(t, dt=ttnn.float32):
        return ttnn.from_torch(
            t.contiguous(),
            device=md,
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )

    torch.manual_seed(3)
    for T in (1, 4, 8, 16):
        x0 = torch.randn(T, 1, 4, D)
        xr, xp = up(x0), up(x0)
        pad_bad = val_bad = 0
        for step in range(100):
            role = "attn" if step % 2 == 0 else "ffn"
            post, comb = up(torch.rand(T, 1, 4, 1) * 0.5), up(torch.rand(T, 1, 4, 4) * 0.2)
            y, y2 = up(torch.randn(1, 1, T, D), ttnn.bfloat16), up(torch.randn(1, 1, T, D))
            xr = mhc_expand2(y, xr, post, comb, y2)  # reference chain (fresh buffers)
            buf = mhc_stream_pool.get(md, T, D, "chain", role)
            xp = mhc_expand3(y, xp, post, comb, buf, y2)
            a, b = full(xr), full(xp)
            val_bad += int(not torch.equal(a, b))
            pad_bad += int(b[:, :, 4:, :].abs().max().item() != 0.0)  # padding rows of the pooled chain are exactly 0
        print(f"CHAIN T={T}: differing steps {val_bad}/100", flush=True)
        assert val_bad == 0 and pad_bad == 0
        # NaN-poisoned fresh buffer: padding survives (only valid pieces written) -> a pool with zero init is required
        nan_buf = ttnn.multiply(
            up(x0), float("nan")
        )  # element-wise ops run over whole tiles: the padding rows become NaN too
        post, comb = up(torch.rand(T, 1, 4, 1)), up(torch.rand(T, 1, 4, 4))
        y, y2 = up(torch.randn(1, 1, T, D), ttnn.bfloat16), up(torch.randn(1, 1, T, D))
        o = full(mhc_expand3(y, up(x0), post, comb, nan_buf, y2))
        ref = full(mhc_expand2(y, up(x0), post, comb, y2))
        nan_pad = int(torch.isnan(o[:, :, 4:]).any().item()) and int(not torch.isnan(o[:, :, :4]).any().item())
        print(
            f"NAN_POISON T={T}: padding keeps the NaNs = {bool(nan_pad)} (a zero-initialised pool is required), valid rows equal the plain expand = {bool(torch.equal(o[:, :, :4], ref[:, :, :4]))}",
            flush=True,
        )
        assert torch.equal(o[:, :, :4], ref[:, :, :4])
        assert nan_pad == 1
    mhc_stream_pool.reset(md)
