"""Is `_pad_channels_to_aligned` corrupting the live decode?

`concat_vs_slice.py` showed `ttnn.concat(dim=2)` disagrees with `torch.cat` at C=8.
`_pad_channels_to_aligned` (audio_ops.py:944) makes exactly that call and runs wherever C < 32, which
is the audio tail.

`test_decode` would not have caught it despite scoring against the CPU reference: at ~1e-03 the error
sits far under a whole-model tolerance that passes at 42.9 dB against a 28 dB gate. So this checks the
one op on its own, at the widths the model actually presents.

It also carries the minimal repro for the underlying bug -- a bare two-tensor concat with no audio
machinery involved -- so an upstream issue can be filed from it directly.
"""

import torch

import ttnn
from models.tt_dit.layers.audio_ops import _pad_channels_to_aligned
from models.tt_dit.models.audio_vae.minimax_h3 import decoder_minimax_h3_audio  # noqa: F401

T = 20701
WIDTHS = [8, 16, 24, 32, 64]


def audit(device):
    print("=== _pad_channels_to_aligned vs CPU ===")
    print(f"{'C':>4} {'padded':>7} {'exact':>7} {'maxdiff':>12}   note")
    print("-" * 52)
    for C in WIDTHS:
        torch.manual_seed(0)
        x = torch.randn(2, T, C) * 0.3
        xd = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        try:
            out = ttnn.to_torch(_pad_channels_to_aligned(xd, device, channel_align=32)).float()
        except Exception as exc:  # noqa: BLE001
            print(f"{C:>4} {'-':>7}  FAILED {str(exc).splitlines()[0][:34]}")
            continue
        padded = out.shape[2]
        ref = torch.zeros(2, T, padded)
        ref[:, :, :C] = x
        exact = torch.equal(out, ref)
        d = float((out - ref).abs().max())
        note = "no-op (already aligned)" if padded == C else ""
        print(f"{C:>4} {padded:>7} {str(exact):>7} {d:>12.3e}   {note}")


def upload_control(device):
    """The control `concat_vs_slice.py` never ran.

    That script blamed concat, but its slice arm only ever read back tensors of width 2C -- 16, 32,
    64 -- so a narrow tensor was never round-tripped on its own. If `from_torch`/`to_torch` is itself
    lossy at C=8, concat is innocent and merely the first op to expose it.

    The magnitude argues for this reading: 9.764e-04 against data of scale ~0.3 is bf16 rounding, not
    the garbage a misaligned copy produces.
    """
    print("\n=== control: from_torch -> to_torch round trip, no op at all ===")
    print(f"{'C':>4} {'layout':>10} {'exact':>7} {'maxdiff':>12}")
    print("-" * 40)
    for C in (8, 16, 32):
        for lname, layout in (("ROW_MAJOR", ttnn.ROW_MAJOR_LAYOUT), ("TILE", ttnn.TILE_LAYOUT)):
            torch.manual_seed(0)
            a = torch.randn(1, 1024, C)
            d = ttnn.from_torch(a, dtype=ttnn.float32, layout=layout, device=device)
            got = ttnn.to_torch(d).float()
            print(f"{C:>4} {lname:>10} {str(torch.equal(got, a)):>7} " f"{float((got - a).abs().max()):>12.3e}")


def minimal_repro(device):
    """Smallest form of the bug, with no audio code in the path -- for an upstream issue."""
    print("\n=== minimal repro: bare ttnn.concat vs torch.cat, fp32 ROW_MAJOR ===")
    print(f"{'C':>4} {'rows':>8} {'exact':>7} {'maxdiff':>12}")
    print("-" * 36)
    for C in (8, 16):
        for rows in (32, 1024, 82806):
            torch.manual_seed(0)
            a = torch.randn(1, rows, C)
            b = torch.randn(1, rows, C)
            ad = ttnn.from_torch(a, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            bd = ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            got = ttnn.to_torch(ttnn.concat([ad, bd], dim=2)).float()
            ref = torch.cat([a, b], dim=2)
            d = float((got - ref).abs().max()) if got.shape == ref.shape else float("nan")
            print(f"{C:>4} {rows:>8} {str(torch.equal(got, ref)):>7} {d:>12.3e}")
            # Attribute the error: is it in the uploaded halves or introduced by the concat?
            if not torch.equal(got, ref):
                ua = ttnn.to_torch(ad).float()
                print(
                    f"     {'upload a':>8} {str(torch.equal(ua, a)):>7} "
                    f"{float((ua - a).abs().max()):>12.3e}   <- input, pre-concat"
                )
                # Is the output exactly bf16(ref)? A silent downcast and a misaligned copy both
                # produce "wrong", but only a downcast produces *this* wrong.
                bf = ref.bfloat16().float()
                print(
                    f"     {'== bf16':>8} {str(torch.equal(got, bf)):>7} "
                    f"{float((got - bf).abs().max()):>12.3e}   <- vs bf16-rounded reference"
                )


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
    try:
        audit(device)
        upload_control(device)
        minimal_repro(device)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
