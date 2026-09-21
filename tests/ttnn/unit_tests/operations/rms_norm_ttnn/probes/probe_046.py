# Exact per-head sequence: [1,H,8,256] TILE -> reshape flat -> rms_norm -> reshape back.
# Then EXPOSE the padding by viewing the padded row count as logical rows.
import torch, ttnn
dev = ttnn.open_device(device_id=0, l1_small_size=8192)
try:
    eps = 1e-6
    H, S, D = 16, 8, 256
    torch.manual_seed(0)
    x_t = torch.randn(1, H, S, D, dtype=torch.bfloat16)
    w_t = torch.randn(1, 1, D // 32, 32, dtype=torch.bfloat16)
    x = ttnn.from_torch(x_t, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w = ttnn.from_torch(w_t, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    print("x shape", tuple(x.shape), "padded", tuple(x.padded_shape), flush=True)
    flat = ttnn.reshape(x, (1, 1, H * S, D))
    print("flat shape", tuple(flat.shape), "padded", tuple(flat.padded_shape), flush=True)
    for side, fn in (("gen", ttnn.rms_norm), ("nat", ttnn._native_rms_norm)):
        y = fn(flat, weight=w, epsilon=eps, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        back = ttnn.reshape(y, (1, H, S, D))
        print(f"{side}: y shape {tuple(y.shape)} padded {tuple(y.padded_shape)} | back shape {tuple(back.shape)} padded {tuple(back.padded_shape)}", flush=True)
        # expose padding: view the padded rows of y as logical
        py = y.padded_shape
        try:
            full = ttnn.to_torch(ttnn.reshape(y, (1, 1, py[-2], py[-1]))).float()
            logical = ttnn.to_torch(y).float()
            print(f"{side}: exposed rows={full.shape[-2]} logical rows={logical.shape[-2]}", flush=True)
            if full.shape[-2] > logical.shape[-2]:
                pad = full[..., logical.shape[-2]:, :]
                print(f"{side}: PADDING absmax={pad.abs().max().item():.4e} nan={torch.isnan(pad).sum().item()} "
                      f"nonzero={(pad != 0).sum().item()}/{pad.numel()}", flush=True)
        except Exception as e:
            print(f"{side}: expose failed: {e}", flush=True)
        # padding via the per-head view: to_torch of back vs a downstream softmax over S
        bt = ttnn.to_torch(back).float()
        # downstream: q@k^T over padded S then softmax -- use back as both q and k
        scores = ttnn.matmul(back, ttnn.permute(back, (0, 1, 3, 2)))
        sm = ttnn.softmax(scores, dim=-1)
        print(f"{side}: back->scores shape {tuple(scores.shape)} padded {tuple(scores.padded_shape)} "
              f"softmax rowsum(logical)={ttnn.to_torch(sm).float().sum(-1).mean().item():.6f}", flush=True)
    print("PROBE_OK", flush=True)
finally:
    ttnn.close_device(dev)
