"""Is the op's bfloat4_b output as good as the FORMAT allows?

Reference = the same tensor converted to bfloat4_b entirely on the HOST by
ttnn.from_torch (a different code path; used here only as a numerical oracle).
If the host conversion lands at the same PCC, the ~0.98 readings are the format's
floor on this data, not the op's packer."""
import importlib, torch, ttnn

M = importlib.import_module("ttnn.operations.tilize.tilize")

device = ttnn.open_device(device_id=0)


def pcc(a, b):
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    va, vb = a - a.mean(), b - b.mean()
    d = va.norm() * vb.norm()
    return float((va * vb).sum() / d) if d > 0 else 1.0


for seed in (0, 1, 2, 3, 4):
    torch.manual_seed(seed)
    x = torch.randn((1, 1, 50, 50)).bfloat16()
    padded = torch.nn.functional.pad(x, (0, 14, 0, 14), value=5)
    # op path
    tt = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    got = M.tilize(tt, dtype=ttnn.bfloat4_b, pad_value=5).cpu().to_torch_with_padded_shape()
    # host-only reference conversion of the ALREADY-PADDED tensor
    host = ttnn.to_torch(ttnn.from_torch(padded, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT))
    print(
        f"seed={seed}  op_pcc={pcc(got, padded):.6f}  host_pcc={pcc(host, padded):.6f}  "
        f"op_vs_host_identical={torch.equal(got, host)}",
        flush=True,
    )
ttnn.close_device(device)
