"""Exact DRAM bytes per chip of every head / embedding variant (allocator view before/after the build), plus the host-side formula."""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "head_device.py")).read().split("for v in VARIANTS:")[0])
import gc

dev = md.get_devices()[0] if hasattr(md, "get_devices") else md


def alloc():
    mv = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
    return (
        (
            mv.total_bytes_allocated_per_bank * mv.num_banks
            if hasattr(mv, "num_banks")
            else mv.total_bytes_allocated_per_bank
        ),
        mv,
    )


b0, mv = alloc()
res(
    event="mem_attrs",
    attrs=[a for a in dir(mv) if not a.startswith("_")],
    total_per_bank=mv.total_bytes_per_bank,
    alloc_per_bank=mv.total_bytes_allocated_per_bank,
)


def per_bank():
    return ttnn.get_memory_view(dev, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank


banks = int(dev.num_dram_channels()) if hasattr(dev, "num_dram_channels") else int(os.environ.get("DRAM_BANKS", "8"))
res(event="banks", banks=banks)
base = per_bank()
for hd in ["bf16", "bfp8", "bfp4", "fp32"]:
    h = Hy4DeviceHead(md, mc, ccl, W, head_dtype=hd, out_dtype="fp32", fidelity="HiFi4")
    ttnn.synchronize_device(md)
    full = per_bank() - base
    # weight only: build the weight tensor alone to separate it from the constants
    res(
        event="bytes_head",
        head_dtype=hd,
        alloc_per_bank_delta=full,
        per_chip_bytes_est=full * banks,
        formula_bytes=head_bytes_per_chip(hd),
    )
    del h
    gc.collect()
    ttnn.synchronize_device(md)
E = torch.load(f"{LMH}/embed.pt")["embed"]
for mode in ["dshard", "vshard"]:
    base2 = per_bank()
    e = Hy4DeviceEmbedding(md, mc, ccl, E, mode=mode)
    ttnn.synchronize_device(md)
    d = per_bank() - base2
    res(
        event="bytes_emb",
        mode=mode,
        alloc_per_bank_delta=d,
        per_chip_bytes_est=d * banks,
        formula_bytes=VOCAB * DIM * 2 // 8,
    )
    del e
    gc.collect()
ttnn.close_mesh_device(md)
