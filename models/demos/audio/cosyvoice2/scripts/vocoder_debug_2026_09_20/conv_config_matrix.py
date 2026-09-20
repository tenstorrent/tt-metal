"""The live warning: Conv1d(128->128, k=11) at length 18560 (resblocks.5, the 128-ch k=11 block).
For each of its 6 convs, run the 2x2 {prepared, raw weight} x {accurate, safe config} on device and compare EVERY
variant to float64 torch truth -- which axis disagrees, and is the shipped fallback (raw+safe) actually right?"""
import torch

import ttnn
from models.demos.audio.cosyvoice2.tt.checkpoint import load_checkpoint_file
from models.demos.audio.cosyvoice2.tt.hifigan.conv import extract_conv_weights
from models.demos.audio.cosyvoice2.tt.hifigan.generator import TorchHiFTDecodeRef, TtHiFTDecoder

L = 18560
hift_sd = load_checkpoint_file("hift.pt")
ref = TorchHiFTDecodeRef.from_checkpoint(hift_sd)
g = torch.Generator().manual_seed(0)
x_t = torch.randn(1, L, 128, generator=g) * 0.5  # [1,L,C] channels-last

dev = ttnn.open_device(device_id=0, l1_small_size=65536)
try:
    dec = TtHiFTDecoder(dev, ref, dtype=ttnn.float32)
    rb, rrb = dec.resblocks[5], ref.resblocks[5]
    print(f"resblocks.5: {len(rb.convs1)} convs1 + {len(rb.convs2)} convs2\n")
    hdr = f"{'conv':10s} {'dil':>3s} | " + " | ".join(
        f"{n:>17s}" for n in ("prep+acc(fast)", "prep+safe", "raw+acc", "raw+safe(fallbk)")
    )
    print(hdr + "    [each: rel err vs fp64 | max|out|]   truth max|out|")
    for grp, tt_list, ref_list in (("convs1", rb.convs1, rrb.convs1), ("convs2", rb.convs2, rrb.convs2)):
        for i, (conv, rc) in enumerate(zip(tt_list, ref_list)):
            w, b = extract_conv_weights(rc)
            with torch.no_grad():
                truth = torch.nn.functional.conv1d(
                    x_t.transpose(1, 2).double(),
                    w.double(),
                    b.double(),
                    stride=1,
                    padding=int(rc.padding[0]),
                    dilation=int(rc.dilation[0]),
                )  # [1,C,L]
            truth = truth.transpose(1, 2)
            x_dev = ttnn.from_torch(x_t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
            wp, bp = conv._prepared(x_dev, L, 1)
            variants = {
                "prep+acc": (wp, bp, conv.compute_config),
                "prep+safe": (wp, bp, conv._safe_compute_config),
                "raw+acc": (conv.weight, conv.bias, conv.compute_config),
                "raw+safe": (conv.weight, conv.bias, conv._safe_compute_config),
            }
            cells = []
            for name, (ww, bb, cfg) in variants.items():
                out, _ = conv._conv(x_dev, ww, bb, L, 1, cfg)
                got = ttnn.to_torch(out).float().reshape(1, -1, 128).double()
                n = min(got.shape[1], truth.shape[1])
                rel = float((got[:, :n] - truth[:, :n]).norm() / truth[:, :n].norm())
                cells.append(f"{rel:8.4f} | {float(got.abs().max()):6.2f}")
                ttnn.deallocate(out)
            print(
                f"{grp}[{i}]  {conv.dilation:>3d} | "
                + " | ".join(f"{c:>17s}" for c in cells)
                + f"    {float(truth.abs().max()):.2f}"
            )
            ttnn.deallocate(x_dev)
finally:
    ttnn.close_device(dev)
