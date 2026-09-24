### Issue
Qwen3-4B prefill on P150 pays two pure data-movement passes per layer around attention: the QKV projection writes `[B*S, (H+2*Hkv)*d]`, a head-split op re-reads it and writes `[B, H, S, d]` Q/K/V (214 MB moved per layer at batch 32); after SDPA a concat pass turns `[B, H, S, d]` back into `[B*S, H*d]` for the output projection (142 MB). Under a sustained load the P150 power manager holds AICLK at ~1.1 GHz and ~80% of the batch-32 iteration is bound by DRAM traffic, so these passes are ~10% of the 450 ms sustained forward (about 5 ms of 124 at batch 8).

Both are tile-id remaps in tile layout, not data reshuffles:
- output side: tile `(row_tile = b*S/32 + s, col_tile = h*d/32 + j)` of a `[B*S, H*d]` result is tile `((b*H + h)*S/32 + s)*d/32 + j` of `[B, H, S, d]`;
- input side: the same mapping read in reverse lets the output projection consume `[B, H, S, d]` as if it were `[B*S, H*d]`.

### Expected
`minimal_matmul` (and/or `ttnn.linear`) options to (a) write the output in head-major `[B, H, S, d]` tile order, and/or (b) read in0 from a `[B, H, S, d]` tensor as `[B*S, H*d]`, given `H` and `d`. With (a) the head-split op only has to RMSNorm+RoPE Q/K in place (V untouched, ~120 MB/layer saved at batch 32); with (b) the concat pass disappears (142 MB/layer). Same PCC as today; no extra kernel time beyond the remap.

### Unit test (random data, one P150)
```python
import torch, ttnn
B, S, H, d, K = 8, 512, 32, 128, 2560
D = ttnn.open_device(device_id=0, l1_small_size=32768)
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
                                       fp32_dest_acc_en=False, packer_l1_acc=True)
cfg = ttnn.MinimalMatmulConfig(M_block_size=8, K_block_size=8, N_block_size=8, subblock_h=1, subblock_w=8,
                               compute_with_storage_grid_size=ttnn.CoreCoord(12, 10))
try:
    x = ttnn.from_torch(torch.randn(1, 1, B * S, K), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=D)
    w = ttnn.from_torch(torch.randn(1, 1, K, H * d) * 0.02, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=D)
    y = ttnn.experimental.minimal_matmul(x, w, compute_kernel_config=ckc, config=cfg, dtype=ttnn.bfloat8_b)
    ref = ttnn.to_torch(y).float().reshape(B, S, H, d).permute(0, 2, 1, 3)  # what a head-major output must equal
    # (a) with the head-major output option, ttnn.to_torch(y_headmajor).reshape(B, H, S, d) must equal `ref` bit for bit
    # (b) with the head-major in0 option, minimal_matmul(y_headmajor, w2) must equal minimal_matmul(concat_heads(y_headmajor), w2)
    print("reference shape", tuple(ref.shape))
finally:
    ttnn.close_device(D)
```
