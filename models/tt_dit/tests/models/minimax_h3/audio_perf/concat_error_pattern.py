"""What exactly does ttnn.concat get wrong at C=8?

Established so far: concat is guilty (from_torch/to_torch round trips exact at every width), and the
output is not a bf16 downcast (it does not equal bf16(ref), and sits further from bf16 than from the
truth). Both mechanism guesses were wrong, so this stops inferring from magnitudes and reads the
error directly:

  which columns   if only the second tensor's half is wrong, it is the `l1_write_addr += page_size`
                  step in the WIDTH_CONCAT loop; if scattered, it is precision
  which rows      all rows, or a periodic subset (a per-page or per-core boundary effect)
  bit patterns    how many low mantissa bits differ -- names the format if it is a truncation
  data dependence rerun with values the size of a float32 ulp ladder, so a truncation shows up as an
                  exact power-of-two mask rather than a fuzzy magnitude
"""

import torch

import ttnn
from models.tt_dit.models.audio_vae.minimax_h3 import decoder_minimax_h3_audio  # noqa: F401

C, ROWS = 8, 1024


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
    try:
        torch.manual_seed(0)
        a = torch.randn(1, ROWS, C)
        b = torch.randn(1, ROWS, C)
        ad = ttnn.from_torch(a, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        bd = ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        got = ttnn.to_torch(ttnn.concat([ad, bd], dim=2)).float()
        ref = torch.cat([a, b], dim=2)
        bad = got != ref

        print(f"shape {tuple(got.shape)}   wrong elements {int(bad.sum())} / {bad.numel()}")

        print("\nwrong count per column (0-7 = tensor a, 8-15 = tensor b):")
        for c in range(2 * C):
            n = int(bad[0, :, c].sum())
            mark = " <-- b" if c >= C else ""
            print(f"  col {c:>2}  {n:>5} / {ROWS}{mark}")

        rows_bad = bad.any(dim=2)[0]
        idx = torch.nonzero(rows_bad).flatten()
        print(f"\nrows with any error: {len(idx)} / {ROWS}")
        if len(idx):
            print(f"  first 16 row indices: {idx[:16].tolist()}")
            if len(idx) > 1:
                d = (idx[1:] - idx[:-1]).unique()
                print(f"  unique row strides:   {d[:8].tolist()}")

        # Bit-level: how do the wrong words differ from the right ones?
        gi = got.view(torch.int32)[bad]
        ri = ref.view(torch.int32)[bad]
        if gi.numel():
            x = (gi ^ ri).to(torch.int64) & 0xFFFFFFFF
            highest = torch.tensor([int(v).bit_length() for v in x])
            print(f"\nXOR of wrong words: highest differing bit (0=LSB of mantissa)")
            print(f"  max {int(highest.max())}, min {int(highest.min())} (bit 23 = mantissa top, 24+ = exponent)")
            print(f"  sample xor: {[hex(int(v)) for v in x[:6]]}")
            print(f"  sample got: {[hex(int(v) & 0xFFFFFFFF) for v in gi[:6]]}")
            print(f"  sample ref: {[hex(int(v) & 0xFFFFFFFF) for v in ri[:6]]}")

        # Is it a copy of the *wrong element* rather than a corrupted one? Check whether each wrong
        # value appears anywhere in the source row -- a data-movement offset bug would show that.
        print("\nis each wrong value present elsewhere in the same source row?")
        hits = 0
        checked = 0
        rr = torch.cat([a, b], dim=2)[0]
        for r in idx[:64].tolist():
            for c in range(2 * C):
                if bad[0, r, c]:
                    checked += 1
                    if (rr[r] == got[0, r, c]).any():
                        hits += 1
        print(f"  {hits} / {checked} wrong values also occur in the correct row (relocation, not corruption)")
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
