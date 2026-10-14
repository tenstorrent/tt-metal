# Repo map: deepseek_v41_flash_d_p

| Need | Where it is |
|---|---|
| Run DeepSeek's own inference code (checkpoint `inference/model.py`) on the CPU, unmodified: GPU kernels ported, experts read per use, engram tables memory-mapped | `models/demos/deepseek_v41_flash_d_p/reference/hf_oracle.py` (`inference_module`: model.py with `reference/kernels.py` as its `kernel` module and tiled `F.linear` / `torch.einsum`; `build_oracle`; `LazyExpert`, `MemmapEngramEmbedding`, per-block `taps`) |
| CPU port of the DeepSeek-V4.x tilelang kernels (fp8 act quant ue8m0, fp4 e2m1 quantize-dequantize with ue8m0 / e4m3 scales, fp8 / fp4 GEMMs, sparse attention with sink, mHC Sinkhorn) | `models/demos/deepseek_v41_flash_d_p/reference/kernels.py` |
| Checkpoint access without loading shards: index + safetensors header + memory map; fp8 32x32 block and fp4 (packed e2m1, ue8m0 per 32) dequant; engram row lookup | `models/demos/deepseek_v41_flash_d_p/reference/weights.py` (`Checkpoint`, `fp8_block_dequant`, `fp4_dequant`, `EngramTable`) |
| Engram n-gram hashing on the host, chunked (carried 3-token tail), bit-equal to the checkpoint's `inference/engram.py` | `models/demos/deepseek_v41_flash_d_p/reference/ngram.py:EngramHasher` |
| CPU reference: CSA2 attention (window ring, compressor, indexer with candidate pool, shared KV / top-k across layers), single-pass mHC, MoE, chunked | `models/demos/deepseek_v41_flash_d_p/reference/ds41_ref.py` (`Config` layer roles / `sources`, `State`, `block_steps`, `DS41Reference`) |
