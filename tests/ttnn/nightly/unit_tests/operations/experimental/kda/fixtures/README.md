# Real KDA inverse regression input

`layer13_head50_chunk6.safetensors` contains one 32-token chunk from the layer-13 decoder-stream-derived Kimi-K3 probe, head50, chunk6 (tokens192–223). It reproduces issue#55420 without model files: original PS4 inverse max error4.773934, strictly-lower PCC0.550148.

Payload: q/k/v/g are BF16[1,32,128] (8,192 bytes each); beta is FP32[1,1,32,1] (128 bytes). Total tensor payload32,896 bytes; file33,472 bytes. SHA256:e87fd5798928d2da0f6881d67497684c2fd26d00998ac320f34f3dd31d3d31d7. Expected outputs are computed with the existing FP64 causal oracle, not stored.

Provenance: `k3_vllm_code_debug_1M` preceding decoder output, Kimi-K3 layer13 checkpoint weights; the probe applies RMSNorm, projections and causal convolution, then slices head50/chunk6. q/k/v are post-convolution, g is the transformed gate and beta the sigmoid output. This is derived from real decoder data, not a claim of exact replay of the original attention-input boundary. The optional trace probes document/reproduce that construction; no checkpoint or large trace is required for these unit tests.

The scratch-wrap regression repeats this same fixture for32 work items per device core, covering transaction boundaries that a single chunk cannot exercise. It checks final_decay independently of inverse accuracy.
