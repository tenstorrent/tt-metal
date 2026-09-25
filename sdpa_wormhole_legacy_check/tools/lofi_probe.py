# usage: lofi_probe.py gx gy q k fidelity s [causal]   -- one legacy dense SDPA call (noncausal default)
import sys, torch, ttnn
gx, gy, q, k = map(int, sys.argv[1:5]); fid = sys.argv[5]; s = int(sys.argv[6]); causal = len(sys.argv) > 7
dev = ttnn.open_device(device_id=0)
g = torch.Generator().manual_seed(20260921)
xs = [torch.randn((1, 2, s, 128), generator=g).bfloat16() for _ in range(3)]
t = [ttnn.from_torch(x, device=dev, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG) for x in xs]
pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(gx, gy), q_chunk_size=q, k_chunk_size=k)
cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=True,
                                       fp32_dest_acc_en=False, packer_l1_acc=False, dst_full_sync_en=False)
o = ttnn.to_torch(ttnn.transformer.scaled_dot_product_attention(*t, is_causal=causal, program_config=pc, compute_kernel_config=cfg))
r = torch.nn.functional.scaled_dot_product_attention(*[x.float() for x in xs], is_causal=causal)
p = torch.corrcoef(torch.stack([o.float().flatten(), r.flatten()]))[0, 1].item()
print("PROBE_OK", sys.argv[1:], "pcc", p, flush=True)
ttnn.close_device(dev)
