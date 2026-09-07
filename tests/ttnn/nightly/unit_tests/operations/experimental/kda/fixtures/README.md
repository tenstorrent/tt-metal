# Real KDA inverse regression input

`layer13_head50_chunk6.safetensors` contains one 32-token chunk from the layer-13 decoder-stream-derived Kimi-K3 probe, head 50, chunk 6 (tokens 192–223). It reproduces issue #55420 without model files: the original PS4 inverse has maximum error 4.773934 and strictly-lower PCC 0.550148. The four-block Horner inverse reduces these to 0.005516410 and 0.999988.

| Tensor | Shape | Dtype | Payload bytes |
| --- | --- | --- | ---: |
| q | [1, 32, 128] | BF16 | 8,192 |
| k | [1, 32, 128] | BF16 | 8,192 |
| v | [1, 32, 128] | BF16 | 8,192 |
| g | [1, 32, 128] | BF16 | 8,192 |
| beta | [1, 1, 32, 1] | FP32 | 128 |

Total tensor payload: 32,896 bytes; file: 33,472 bytes. SHA256: `e87fd5798928d2da0f6881d67497684c2fd26d00998ac320f34f3dd31d3d31d7`. Expected outputs are computed with the existing FP64 causal oracle.

Provenance: `k3_vllm_code_debug_1M` preceding decoder output and Kimi-K3 layer 13 checkpoint weights. The probe applies RMSNorm, projections and causal convolution, then slices head 50/chunk 6. q/k/v are post-convolution, g is the transformed gate and beta the sigmoid output. This is derived from real decoder data; it does not claim exact replay of the original attention-input boundary. The optional trace probes reproduce that construction. No checkpoint or large trace is required for these unit tests.

The scratch-wrap regression repeats this fixture for 32 work items per device core, covering transaction boundaries that a single chunk cannot exercise. It checks final_decay independently of inverse accuracy.
