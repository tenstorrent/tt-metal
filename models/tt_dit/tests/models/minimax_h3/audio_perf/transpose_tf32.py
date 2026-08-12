"""Is ttnn.transpose the actual source of the fp32 -> TF32 truncation?

A bit-pattern probe showed ttnn.concat at C=8 returns `ref & 0xFFFFE000` -- the low 13 mantissa
bits zeroed, i.e. TF32, truncated rather than rounded. concat itself is a pure NOC copy and its CB is
Float32, so it cannot do that. But concat.cpp:186 `build_non_aligned_last_dim_concat` routes last-dim
concats through a `ttnn.transpose(-2,-1)` round trip whenever

    padded_shape[dim] * element_size % buffer()->alignment() != 0

and Blackhole DRAM alignment is 64 bytes -- so C=8 (32B) and C=24 (96B) take the transpose path while
C=16 (64B) and C=32 (128B) go direct. That matches the observed pass/fail exactly.

If transpose is the culprit then the blast radius is far wider than concat: every fp32 transpose
silently loses 13 mantissa bits. This tests transpose on its own, and checks the mask exactly.
"""

import torch

import ttnn
from models.tt_dit.models.audio_vae.minimax_h3 import decoder_minimax_h3_audio  # noqa: F401

TF32_MASK = 0xFFFFE000


def check(name, got, ref):
    exact = torch.equal(got, ref)
    d = float((got - ref).abs().max())
    masked = (ref.view(torch.int32).to(torch.int64) & 0xFFFFFFFF & TF32_MASK).to(torch.int32).view(torch.float32)
    is_tf32 = torch.equal(got, masked)
    print(f"  {name:<34} exact={str(exact):<5} maxdiff={d:<11.3e} == fp32&0xFFFFE000: {is_tf32}")


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
    try:
        for layout, lname in ((ttnn.ROW_MAJOR_LAYOUT, "ROW_MAJOR"), (ttnn.TILE_LAYOUT, "TILE")):
            print(f"\n=== ttnn.transpose(-2, -1), fp32, {lname} ===")
            for shape in [(1, 1024, 8), (1, 1024, 16), (1, 64, 64), (1, 32, 32)]:
                torch.manual_seed(0)
                x = torch.randn(*shape)
                xd = ttnn.from_torch(x, dtype=ttnn.float32, layout=layout, device=device)
                try:
                    got = ttnn.to_torch(ttnn.transpose(xd, -2, -1)).float()
                except Exception as exc:  # noqa: BLE001
                    print(f"  {str(shape):<34} FAILED {str(exc).splitlines()[0][:40]}")
                    continue
                check(str(shape), got, x.transpose(-2, -1).contiguous())

        # bf16 control: if the mask theory is right, bf16 has only 7 mantissa bits so TF32
        # truncation is a no-op and bf16 transpose should be exact.
        print("\n=== ttnn.transpose(-2, -1), bfloat16 (control: TF32 mask is a no-op here) ===")
        for shape in [(1, 1024, 8), (1, 1024, 16)]:
            torch.manual_seed(0)
            x = torch.randn(*shape).bfloat16().float()
            xd = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            got = ttnn.to_torch(ttnn.transpose(xd, -2, -1)).float()
            ref = x.transpose(-2, -1).contiguous()
            print(f"  {str(shape):<34} exact={torch.equal(got, ref)}  maxdiff={float((got-ref).abs().max()):.3e}")

        # And the corollary: bf16 concat at C=8 should be exact even though it takes the same
        # non-aligned fallback, because the lossy step cannot hurt bf16.
        print("\n=== ttnn.concat(dim=-1) bfloat16 at the widths that fail in fp32 ===")
        for C in (8, 24):
            torch.manual_seed(0)
            a = torch.randn(1, 1024, C).bfloat16().float()
            b = torch.randn(1, 1024, C).bfloat16().float()
            ad = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            bd = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            got = ttnn.to_torch(ttnn.concat([ad, bd], dim=2)).float()
            ref = torch.cat([a, b], dim=2)
            print(f"  C={C:<32} exact={torch.equal(got, ref)}  maxdiff={float((got-ref).abs().max()):.3e}")
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
