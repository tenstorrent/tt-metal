# Graph-fusing pattern audit

Each entry maps the installed graph-fusing skill to the actual Ornith layer graph. Measurements and
source snapshots are in work_log.md and the candidate JSON files. Baseline dedicated ops remain part
of the measured fused implementation; no runtime dispatch returns to FunctionalDecoder.

| Skill pattern | Assessment and action |
|---|---|
| Dedicated elementwise activation | Baseline already uses SiLU/sigmoid/softplus. Merge applicable activations into convolution, minimal matmul and binary inputs; measured. |
| Dedicated softmax | Full attention already uses paged SDPA, including scaled/stable softmax. No primitive exp/sum/div subgraph remains. |
| Dedicated RMSNorm | Main norms already dedicated. Replace Q/K square/sum/rsqrt/multiply with scaled FP32 RMSNorm; subsequently fuse normalization into flat-input GDN preparation. Both measured. |
| Distributed RMSNorm | No multi-device gather/norm/slice exists; single chip is the stage contract. |
| Dedicated SDPA | Existing prefill chunked SDPA and paged decode SDPA preserved. |
| Split QKV and heads | Repack Q/gate columns at setup, pack Q/gate/K/V projection, use dedicated prefill splitter. Measured and kept. Linear Q/K/V widths differ; dedicated convolution returns the three unequal widths directly. |
| Decode QKV heads | Tried with packed/reordered weights. Adapted the 16-head slice to interleaved layout after the first shard error. Correct but slower than packed projection plus direct reshape; rejected for decode. |
| Decode concat heads | Tried and adapted to explicit height sharding because GQA SDPA rejects sharded output. Adapted path measured including conversions. |
| Prefill concat heads | Used for full attention; also used after head-major GDN output normalization. Measured. |
| Dedicated RoPE | Partial 64-dimensional rotate-half uses rotary_embedding. Decode token_index=0 broadcasts the runtime-selected cos/sin row. Trim padded heads before concatenating the untouched 192 dimensions. Measured. |
| TopK | No sort, selection, logits, or sampling in this layer. |
| RepVGG conv-sum | No parallel 3x3/1x1 convolution branches. Four-tap depthwise temporal convolution is handled by its dedicated op instead. |
| Shared-LHS matmul | Pack four linear-attention input projections and three full-attention projections. Pack MLP gate/up; compare split projections, fused activation and dedicated packed SwiGLU. Measured. |
| Spatial mean | No spatial reduction; the Q/K last-dimension reduction is absorbed into GDN normalization. |
| Permute/reshape/permute | Remove unconditional layout round trips with native reshape. Use flat GDN inputs and head-major output through norm/head concat to remove structural movement. |
| Conv + bias | No convolution bias in checkpoint. |
| Conv + scale | Per-channel tap scales are inputs to the dedicated causal-convolution kernel; no adjacent independent scale remains. |
| Conv + activation | Dedicated four-tap causal convolution includes SiLU and Q/K/V split. Measured. |
| Matmul + activation | Dedicated minimal matmul fuses SwiGLU at prefill. Separate gate matmul with fused SiLU tested at decode and rejected against the best packed candidate. |
| Input activation + binary | SiLU gating, full-attention sigmoid gating and FP32 softplus/decay multiplication merged and measured. |
| Matmul + bias | Projection matrices have no biases. dt_bias precedes softplus and cannot be moved through it or folded into a bias-free packed projection without changing FP32 rounding. |
| Transpose + matmul | Full attention's QK transpose is inside dedicated SDPA. Other remaining transposes are head/sequence ordering, not independent RHS matrix transposes. |
| Slice after matmul | All logical projection columns are consumed. Convolution reads directly from the packed projection, avoiding the full QKV slice; only the last three history rows are sliced for n>=3. Padding columns are internal. |
| BatchNorm + conv | No BatchNorm in the model. |
| Pad + pooling/conv | No pooling or conv2d. Temporal fused convolution requires tile-aligned T; pad internally, update history using logical T, and set GDN tail beta=0/g=0. No public alignment restriction. |
| Stable softmax merge | Already internal to paged SDPA. |
| Reduction + reshape | No reduction with keepdim=False remains. L2 reduction is fused into GDN. |
| Scaled sum -> mean | L2 sum equivalence was tested as RMSNorm with epsilon/128 and weight=1/sqrt(128), then fused into GDN. No other sum/reciprocal pair. |
| RoPE decode reshape | Decode-mode rotary uses token_index=0 after device lookup of the actual position. Packed row-major lookup is prepared at setup; no full-table runtime untilize. |

Additional repository-discovered candidates:

- `experimental.kda.qkv_causal_conv1d_silu`: kept after adapting config and tail padding.
- `experimental.minimal_matmul(fuse_swiglu=True)`: prefill kept, default and adapted decode configurations rejected.
- Flat `chunk_gated_delta_rule`: fuses Q/K normalization, GQA expansion, head addressing and scale.
- `output_head_major=True`: carry GDN output through per-head norm, then concatenate heads once.
- Recurrent ring-state output alias: measured and correct, then superseded by the faster padded chunk decode.
- Use the flat chunk kernel for one logical decode token, with inactive tail rows: measured against the recurrent wrapper.
- `experimental.kda.sigmoid_gated_rms_norm` computes a sigmoid gate rather than SiLU. An adapter could multiply its result by z. After the final head-major GDN change this is a plausible replacement for norm + head concat + SiLU binary (three ops to two), requiring padding/packed-gate adaptation and fresh PCC evidence. Not tried after Stop Rule 2; measured benefit is unknown. Carried as a hand-off experiment.
- Residual-input RMSNorm cannot eliminate the residual addition here: the same unnormalized residual is needed at the MLP output. The exposed RMSNorm returns only normalized output, not both values.

The profile's DRAM-sharded advice was tested and selected as a realization of the same packed projection graph,
with unchanged BF16/HiFi4 precision. No optimized-decoder or later-stage file is created.
