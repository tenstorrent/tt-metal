"""Verify the fixes for the fp32 last-dim concat corruption, and pick one for `_folded_ab`.

Two call sites in audio_ops.py concat on the last dim in fp32 at widths whose row is not a multiple
of the 64B buffer alignment, so both were silently TF32-truncated:

  _pad_channels_to_aligned   fixed by switching to ttnn.pad -- re-verified here
  _folded_ab                 repeats alpha/beta `fold` times along C; needs a repeat, not a pad, so
                             ttnn.repeat is tested as the replacement

`_run_chunks` also concats on the last dim but only ever at chunk in (128, 64, 32) -- rows of 512/256/
128 bytes, all multiples of 64 -- so it is correct today. It is checked here anyway to keep that
assumption honest.
"""

import torch

import ttnn
from models.tt_dit.layers.audio_ops import _pad_channels_to_aligned
from models.tt_dit.models.audio_vae.minimax_h3 import decoder_minimax_h3_audio  # noqa: F401

T = 20701


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
    try:
        print("=== _pad_channels_to_aligned, after the ttnn.pad fix ===")
        print(f"{'C':>4} {'padded':>7} {'exact':>7} {'maxdiff':>12}")
        print("-" * 34)
        for C in (8, 16, 24, 32, 64):
            torch.manual_seed(0)
            x = torch.randn(2, T, C) * 0.3
            xd = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            out = ttnn.to_torch(_pad_channels_to_aligned(xd, device, channel_align=32)).float()
            ref = torch.zeros(2, T, out.shape[2])
            ref[:, :, :C] = x
            print(f"{C:>4} {out.shape[2]:>7} {str(torch.equal(out, ref)):>7} {float((out-ref).abs().max()):>12.3e}")

        print("\n=== _folded_ab replacement: ttnn.repeat vs the concat it replaces, fp32 (1,1,C) ===")
        print(f"{'C':>4} {'fold':>5} {'method':<8} {'exact':>7} {'maxdiff':>12}")
        print("-" * 44)
        for C in (8, 16, 24, 32, 64, 128):
            for fold in (2, 4):
                torch.manual_seed(0)
                a = torch.randn(1, 1, C)
                ad = ttnn.from_torch(a, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
                ref = a.repeat(1, 1, fold)
                got_c = ttnn.to_torch(ttnn.concat([ad] * fold, dim=2)).float()
                print(
                    f"{C:>4} {fold:>5} {'concat':<8} {str(torch.equal(got_c, ref)):>7} "
                    f"{float((got_c-ref).abs().max()):>12.3e}"
                )
                try:
                    got_r = ttnn.to_torch(ttnn.repeat(ad, ttnn.Shape([1, 1, fold]))).float()
                except Exception as exc:  # noqa: BLE001
                    print(f"{C:>4} {fold:>5} {'repeat':<8}  FAILED {str(exc).splitlines()[0][:32]}")
                    continue
                ok = tuple(got_r.shape) == tuple(ref.shape) and torch.equal(got_r, ref)
                d = float((got_r - ref).abs().max()) if tuple(got_r.shape) == tuple(ref.shape) else float("nan")
                print(f"{C:>4} {fold:>5} {'repeat':<8} {str(ok):>7} {d:>12.3e}")

        print("\n=== _run_chunks assumption: last-dim concat at the chunk widths it uses ===")
        print(f"{'chunk':>6} {'exact':>7} {'maxdiff':>12}")
        print("-" * 28)
        for chunk in (32, 64, 128):
            torch.manual_seed(0)
            parts = [torch.randn(2, 1024, chunk) for _ in range(3)]
            pds = [ttnn.from_torch(p, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device) for p in parts]
            got = ttnn.to_torch(ttnn.concat(pds, dim=2)).float()
            ref = torch.cat(parts, dim=2)
            print(f"{chunk:>6} {str(torch.equal(got, ref)):>7} {float((got-ref).abs().max()):>12.3e}")
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
