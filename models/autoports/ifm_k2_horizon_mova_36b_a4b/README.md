# IFM/K2-Horizon-MoVA-36B-A4B TTNN autoport

**Warmed primary vLLM: TTFT 87.223 ms; decode 62.561 tokens/s/user** for
128 input / 128 output tokens, one request, server capacity 1, on a 1×4 Blackhole
p300c mesh with `max_model_len=524288`. Before optimization: 88.734 ms and
62.574 tokens/s/user for the same workload. Decode is effectively unchanged;
the small TTFT difference comes from a single measured request after one warmup.

The secondary CI serving burst completes 32 requests of 100 input / 100 output
tokens at **130.040 output tokens/s aggregate**, with server capacity 32
(before: 130.031). This is capacity evidence, not headline decode per user.

The [optimized vLLM report](doc/optimized_vllm/README.md) records matched benchmark
commands, sampling and mixed-shape trace checks, quality controls, limitations,
and review evidence. The implementation uses the real TT plugin through
[generator_vllm.py](tt/generator_vllm.py), async device-sampled decode, the common
split greedy sampler, and the selected datatype configuration. Context and
non-aligned prompt support are preserved. The optimization captures the prefill
terminal head and reads one replicated token result instead of four.

**Standalone selected default: `head4_ccl4_norm4_lofi`. Warmed queued token-out decode:
62.330 tokens/s/user; TTFT: 95.199 ms.** This uses all 48 layers, batch 1,
128 input / 128 generated tokens, a 1×4 Blackhole p300c mesh and a 524,288-token
cache allocation. Decode includes sampling and device token/position/RoPE
feedback, with 127 queued steps, two boundary synchronizations, and no token
readback or per-token synchronization. TTFT is the paired warmed first-token
measurement. Caller-read decode measures 62.134 tokens/s/user.

The datatype sweep selects by traced teacher forcing: **61.803 tokens/s/user**,
TTFT **523.136 ms**, with 156 input / 100 generated AIME24 chat tokens. That
regime includes explicit HF-token feedback and token readback. Accuracy is
**91%/99%/100%** top-1/top-5/top-100; prefill is **92%/100%/100%**. Both pass the
90%/98%/100% gates. Comparisons against standalone decode use the post-selection token-out control
above, with its workload and measurement boundary, rather than teacher forcing.

The required [precision artifact](doc/datatype_sweep/selected_precision_config.json)
is consumed by normal `K2HorizonModel` and `build_generator` construction. It keeps
BFP4/LoFi decoder weights, the existing first-three-layer BFP8 prefill-attention
exception, BFP8 KV and BF16 residuals. The head, decode CCL and normalized
attention/MLP activations use BFP4. `K2_PRECISION_CONFIG=baseline` restores the
previous policy. Missing selected artifacts fail construction explicitly.

The [datatype-sweep report](doc/datatype_sweep/README.md) contains all 28 full-model
measurements, five physical-DRAM rejections, Pareto plots, observed runtime policy,
qualitative controls and final validation status. The [context contract](doc/context_contract.json)
retains the advertised 524,288 context and aggregate cache-token budget, with up
to 32 fixed slots. Capability test scopes and limitations are stated in the report.
The selected policy is also used by the real vLLM TT plugin adapter.

The [optimized full-model report](doc/optimized_full_model/README.md) preserves
Stage 7 profiling and historical measurements. The [full-model report](doc/full_model/README.md)
retains the generator contract and pinned serving-interface inventory.
