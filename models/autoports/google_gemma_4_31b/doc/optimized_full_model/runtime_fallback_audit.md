# Runtime Fallback Audit

Measured path: `Gemma4Generator.benchmark_token_out_no_readback`, batch 1, exact 149-token AIME24 prompt, 100 output tokens, semantic greedy (`top_k=1`, `top_p=0`, temperature 1), four P150b devices.

| Boundary | Measured behavior | Disposition |
| --- | --- | --- |
| Model selection | `Gemma4FullModel` owns 60 production `MultichipDecoder` layers | No demo, single-chip, CPU, or replicated decoder fallback. |
| Embedding/residual | TTNN embedding and preserved replicated BF16 inter-layer residual | No host tensor boundary between layers. |
| Cache/page tables | Explicit persistent BFP8 KV and stable per-kind tables | Copies occur only for a changed allocation/generation; unchanged replay is copy-free. |
| Position/RoPE | Persistent device position, inactive-row mask, and RoPE positions | Two setup writes; fixed-step decode advances on device. |
| LM head/logits | TP-local BF16/HiFi2 eight-view DRAM-sharded projection and device softcap | No full-vocabulary all-gather or host logits. |
| Greedy sampler | `Gemma4GreedyTP4Sampler` local winners plus tiny TP pair exchange | Exact lowest-token tie rule; no generic `TopKDeviceOperation`, force-argmax, or host argmax. |
| Non-greedy contract | `Sampling1D` remains available for top-k/top-p/temperature modes | Preserved but not substituted into the greedy performance result. |
| Token feedback | Sampler writes the persistent tensor used by the next model replay | No Python token-feedback loop or per-replay token write. |
| Trace replay | Nonblocking model trace followed by sampler trace on one queue | One synchronization after all steady replays, not one per token. |
| Readback | One sampled prefill token seeds trace capture | No readback during the 98 steady replays and no full-logit readback. |
| Reset/reuse | Both traces release before cache/input/sampler mutation | Mixed prompts, fixed slots, inactive rows, and changed tables retain explicit ownership. |

Source branches for `host_sampling_compat=True`, eager teacher-forcing logits, and explicit `force_argmax=True` remain compatibility/test interfaces. They are not active in the benchmark. Prompt-based `generate` reads sampled tokens to return inspectable text and apply EOS; it is qualitative evidence, not the no-readback token-out performance harness. The full benchmark JSON counters and source inspection are consistent; no runtime fallback was observed on the measured path.
