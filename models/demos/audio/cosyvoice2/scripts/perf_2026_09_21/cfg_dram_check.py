"""Does COSYVOICE2_CONV_CONFIG_IN_DRAM actually take effect, per conv TYPE?  (upstream #33284: for conv_transpose the
flag was once not propagated to the halo op.) For each op type, call it at four DIFFERENT lengths, printing L1_SMALL
and DRAM allocated per bank after each call. The flag "took effect" only if L1_SMALL stays flat across lengths --
passing the flag proves nothing. Run once with the env var 0 and once with 1."""
import os
import sys

import torch

import ttnn
from models.demos.audio.cosyvoice2.tt.flow.decoder import TtCausalConv1d
from models.demos.audio.cosyvoice2.tt.flow.encoder import TtPaddedConv1d
from models.demos.audio.cosyvoice2.tt.hifigan.conv import TtConv1d, config_tensors_in_dram
from models.demos.audio.cosyvoice2.tt.hifigan.istft import TtIStft
from models.demos.audio.cosyvoice2.tt.hifigan.stft import TtStft
from models.demos.audio.cosyvoice2.tt.hifigan.upsample import TtConvTranspose1d

print(f"COSYVOICE2_CONV_CONFIG_IN_DRAM -> config_tensors_in_dram() = {config_tensors_in_dram()}")
dev = ttnn.open_device(device_id=0, l1_small_size=65536)
g = torch.Generator().manual_seed(0)


def mem():
    l1 = ttnn.get_memory_view(dev, ttnn.BufferType.L1_SMALL).total_bytes_allocated_per_bank / 1024
    dr = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank / 1024
    return l1, dr


def cl(L, C):
    return ttnn.from_torch(torch.randn(1, L, C, generator=g) * 0.5, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)


def run(name, build, call, lengths):
    print(f"\n== {name}")
    try:
        op = build()
    except Exception as e:
        print("  BUILD FAILED:", str(e)[:150]); return
    l1_0, dr_0 = mem()
    rows = []
    for L in lengths:
        try:
            call(op, L)
            ttnn.synchronize_device(dev)
        except Exception as e:
            print(f"  L={L}: FAILED {str(e)[:120]}"); break
        l1, dr = mem()
        rows.append(l1)
        print(f"  L={L:6d}  L1_SMALL {l1:7.2f} KB (+{l1 - l1_0:6.2f})   DRAM {dr:9.1f} KB (+{dr - dr_0:7.1f})")
    if len(rows) >= 2:
        print(f"  --> L1_SMALL growth across lengths: {rows[-1] - rows[0]:+.2f} KB  ({'FLAT' if abs(rows[-1] - rows[0]) < 0.5 else 'GROWS'})")


def w(o, i, k):
    return torch.randn(o, i, k, generator=g) / (i * k) ** 0.5


run("hifigan TtConv1d 128->128 k=11", lambda: TtConv1d(dev, w(128, 128, 11), torch.zeros(128), padding=5, dtype=ttnn.float32),
    lambda op, L: op(cl(L, 128), L, 1), [1000, 1500, 2100, 3000])
run("hifigan TtConvTranspose1d 256->128 k=11 s=5", lambda: TtConvTranspose1d(dev, torch.randn(256, 128, 11, generator=g) * 0.02, torch.zeros(128), stride=5, padding=3, dtype=ttnn.float32),
    lambda op, L: op(cl(L, 256), L, 1), [200, 300, 420, 600])
run("hifigan TtStft", lambda: TtStft(dev, 16, 4, dtype=ttnn.float32),
    lambda op, L: op(cl(L, 1), L, 1), [4800, 9600, 14400, 24000])
run("hifigan TtIStft", lambda: TtIStft(dev, 16, 4, dtype=ttnn.float32),
    lambda op, n: op(ttnn.from_torch(torch.randn(1, 9, n, generator=g), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev),
                     ttnn.from_torch(torch.randn(1, 9, n, generator=g), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)), [500, 900, 1300, 2000])
run("flow decoder TtCausalConv1d 256->256 k=3", lambda: TtCausalConv1d(dev, w(256, 256, 3), torch.zeros(256), dtype=ttnn.float32),
    lambda op, L: op(cl(L, 256), L, 1), [300, 450, 620, 760])
run("flow encoder TtPaddedConv1d 512->512 k=3", lambda: TtPaddedConv1d(dev, w(512, 512, 3), torch.zeros(512), pad=(2, 0), dtype=ttnn.float32),
    lambda op, L: op(cl(L, 512), L, 1), [150, 230, 310, 400])
ttnn.close_device(dev)
