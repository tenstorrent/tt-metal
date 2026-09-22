# Closest repository canonical policy

The checkout has no separate Qwen3.8-27B canonical model. The closest architecture
is models/demos/blackhole/qwen36, which uses Qwen3_5 modules. Its mlp.py:149–153
loads BFP4 gate/up and BFP8 down; gated_attention.py:26–35 and
 gdn/gated_deltanet.py:38–47 use LoFi with FP32 destination accumulation.
attention/tp.py:71–113 and gdn/tp.py:102–140 load BFP8 projections.
attention/tp.py:401–402 allocates BF16 cache; an optional BF8 SDPA path also exists.

canonical_qwen36_mixed_lofi maps those material groups to the autoport's packed
attention/GDN input, output, packed gate/up and down projections, retaining the
autoport geometry, paged ownership, BF16 residual/CCL, and BFP8/HiFi2 terminal.
It is a mapped precision comparison, not an execution of the Qwen36 model or
its optional vLLM route. Packed gate/up must have equal dtype and fidelity.

canonical_decoder_bfp8_lofi and canonical_decoder_bfp8_hifi2 are uniform BFP8
controls, not claims that the closest canonical defaults are uniformly BFP8.
The inherited optimized baseline is already BFP4/LoFi for all64 decoder layers;
inner_mlp_bfp4_lofi_outer_bfp8 tests a more conservative first/last-layer policy.
