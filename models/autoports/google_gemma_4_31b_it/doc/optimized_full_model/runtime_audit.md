# Runtime path audit

The measured4096/128 path uses the default constructor and fixed-step buffered generation. Full-model timing/readiness runners set ttnn.CONFIG.throw_exception_on_fallback=True. No vLLM adapter is implemented or imported.

Model trace: persistent uint32 token input → embedding/scale → device position/RoPE lookup → unchanged60-layer TP4 stack → final norm/vocabulary-sharded head/softcap → persistent sampler logits. Position increments on device, skipping inactive negative positions.

Buffered sampling trace: common TTSampling with physical local32 candidates and semantically greedy parameters, candidate collectives, tt_out_tok bound to persistent decode token input, seed advance, indexed history write and device history-index increment. The local vocabulary shard is exactly65536, a power of two; no padded vocabulary IDs exist. No full-vocabulary gather or host argmax is in this path.

All three traces and their token/position/page/cache/logits/history/seed buffers are prepared before capture. Acknowledgment scopes and corruptible tensor exemptions are not used. The history's1024-token default capacity is a construction-time output-buffer choice, separate from262144-token model context; callers can configure it before capture. Other generation lengths retain the existing streaming API. Each buffered request overwrites the rows it returns; stale history beyond max_new_tokens is never returned.

The steady loop submits model and buffered-sampling traces nonblocking. It does not allocate tensors, rebuild tokens/masks/positions/pages, copy host inputs, or read/synchronize per token. Timing performs one first-token read to establish host TTFT and one final history read; normal generate_buffered performs only the latter. Page copies occur only when mappings change. Request setup/reset, sampling parameters and initial position are explicitly outside the loop.

Teacher forcing and explicit host_sampling compatibility remain separate caller-driven paths. Teacher forcing necessarily refreshes reference tokens; EOS-sensitive streaming reads caller-visible tokens. These modes are not the no-readback autoregressive performance result. Fixed-step generate/generate_batch select buffering by default; buffered=False supplies the streaming comparison.

Decoder dtype/fidelity, BFP8 KV/cache mapping, activation/CCL policy, selected residual layout and shared persistent CCL workspaces are unchanged. Embedding's barrier-protected gather is retained because the current persistent-output API disables that required barrier; prior repeatability repair and source references are in work_log.md. The head is split only along vocabulary, retaining every K element and ascending token order.
