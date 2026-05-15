# LTX-2 Host-Side Operations

Operations that remain on host (CPU) and cannot currently be moved to device. See [`BRINGUP.md`](BRINGUP.md) for bringup history and the performance optimization backlog.

## Split RoPE Fallback (`attention_ltx.py:_apply_split_rope_host`)

**Why on host:** TTNN elementwise ops (multiply, subtract, add) fail with "Invalid subtile broadcast type" when the last tensor dimension is < 64 (the minimum tile width). This affects audio attention with `D_half = 32` (dim=2048, heads=32, head_dim=64).

**What it does:** Reads Q/K/cos/sin from each device shard, computes split-style rotation on host using torch ops, then re-shards and pushes back to device.

**Permanent fix:** TTNN kernel improvement to handle subtile broadcast for dimensions < 64.

## Per-Step Input Conversion (`ltx_transformer.py:inner_step`)

**Why on host:** The denoising loop keeps latents as torch tensors between steps. Each step converts `torch.Tensor → ttnn.Tensor` via `bf16_tensor()`.

**What it does:** `bf16_tensor(video_latent, device=mesh, shard_dim=-2)` converts and SP-shards the latent once per step.

**Permanent fix:** Keep latents on device between steps (requires device-side Euler step and guidance computation).

## Euler Step (`pipeline_ltx.py:euler_step`)

**Why on host:** The Euler step `x_next = x + velocity * dt` runs on host to match the reference pipeline's exact float32 precision.

**What it does:** `(sample.float() + velocity * dt).to(sample.dtype)` — float32 computation then cast back.

**Permanent fix:** Move to device using `ttnn.add` + `ttnn.multiply` with fp32 accumulation. Requires keeping latents on device.

## Per-Head Gate (`attention_ltx.py:_compute_gate`)

**Why on host:** bf16 device matmul over K=4096 with near-zero gate weights accumulates error; compounds over 48 layers × N denoise steps (latent PCC dropped to 0.92). Host `F.linear` in fp32 gives PCC 0.997+.

**What it does:** Reads hidden states from device, computes `2 * sigmoid(F.linear(x))` on host, pushes bf16 gate tensor back for on-device multiply in BHNE space before `to_out`.

**Permanent fix:** On-device gate with fp32 accumulation, or batched async readback to hide latency.
