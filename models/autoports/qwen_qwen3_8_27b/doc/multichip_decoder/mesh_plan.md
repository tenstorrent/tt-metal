# Stage04 plan before implementation

Baseline: `tt/optimized_decoder.py` at 7c620da56e3475f64e8656d70cde0a9b6464bb20, BFP4/LoFi projections, BF16 activations, FP32 recurrence, BFP8 KV. Dense model: 48 linear-attention and 16 full-attention layers; no MoE/gate-selected experts. Model hidden5120/intermediate17408, context262144. Four Blackhole p300c ASICs, two links per physical edge; `cluster.yaml` and successful2x2 FABRIC_2D open establish connectivity. Candidate target is1x4 TP ring over all four chips (open pending). No smaller mesh support is planned.

| Tensor | Single chip | TP4 local logical shape | Packing/padding / device sharding |
|---|---|---|---|
| residual | B,S,5120 | replicated B,S,5120 or hidden-sharded B,S,1280 | BF16; norm must use full hidden statistics |
| full Q and output gate | 24x256 each | 6x256 each | independently shard heads before Q,K,V,gate packing |
| full K/V | 4x256 each | 1x256 each | local paged cache `[pages,1,32,256]`, shared page IDs/positions |
| full mix weight | 5120,14336 | 5120,3584 | BFP4, DRAM width shard over8 banks;512-channel alignment for2 readers |
| linear Q/K | 16x128 each | 4x128 each | local aligned group of3 value heads per key head |
| linear V/Z | 48x128 each | 12x128 each | mix packs Q512,K512,V1536,Z1536,a32,b32 =4160, padded4608 |
| linear a/b | 48 each | 12 each | pad each scalar projection to32; slice12 after matmul |
| convolution history | B,3,10240 | B,3,2560 | BF16 row-major; tap channels follow Q/K/V ownership |
| recurrent state | B,48,128,128 | B,12,128,128 | FP32;786432 bytes/user plus15360 history =801792 |
| attention out | 6144,5120 | 1536,5120 | row parallel; partial sum collectively reduced |
| gate/up | 5120,17408 each | 5120,4352 each | BFP4; separate prefill pads output4608; packed decode8704 stays aligned |
| down | 17408,5120 | 4352,5120 | row parallel; partial sum collectively reduced |
| norm weights | 5120 |5120 replicated or1280 sharded | BF16; Q/K norm256 and delta output norm128 replicated |
| page tables/positions/RoPE | request metadata | replicated | setup upload only, persistent device inputs during trace |

Initial projection storage configs: mix/gate/up use80 input cores,K2; out uses16 cores,K3 (1536/16/32); down uses17 cores,K8 (4352/17/32). Prefill K-block10 for hidden5120, K-block16 for out1536, K-block8 for down4352. These are legal starting candidates, not measured winners. Prefill grid11x10 with calculated output blocks. Tail padding remains internal; baseline logical-length and continuation contracts remain.

Let A=2*B*ceil(S/32)*32*5120 BF16 bytes per chip (327680 at decode B1). Ring volume estimates per chip, per row-parallel boundary:

| Family | Residual before/after | Next consumer | Collective bytes | Persistent state | Decision |
|---|---|---|---|---|---|
| local matmul + all-reduce | replicated/replicated | local norm | 1.5A | ping-pong CCL semaphores | correctness/perf control |
| reduce-scatter, delayed gather | sharded/sharded | distributed norm then column projection | .75A RS plus .75A later AG | semaphores; independent live trace buffers | test as coherent full-layer candidate |
| fused all-gather-matmul | sharded/sharded | projection consumes gathered norm | .75A AG, can overlap matmul | fused-op buffers | inspect actual Blackhole support and adapt |
| fused matmul-reduce-scatter | sharded/sharded | residual add/distributed norm | .75A RS | fused-op buffers | inspect actual supported shape/dtype |
| replicated attention + TP MLP | replicated/replicated | local norm | MLP only1.5A | larger replicated attention weights/cache | reject as final if full TP works; misses attention bandwidth |
| TP2/DP2 | group-local replicated | next layer within group | smaller2-chip collective | two weight replicas | weaker B1 parallelism; compare only if TP4 loses materially |

Final residual topology requires measurements. Shared MLP1D supplies the column/row parallel pattern and CCL helpers; Attention1D's full-head Meta RoPE contract does not directly express partial64-channel RoPE plus Qwen output gate, nor delta recurrence. Reuse optimized math and adapt ownership model-locally. RMSNorm1D local decode assumes gathered activation; sharded candidate needs distributed statistics or gather at the consuming projection, not incorrect local normalization.

Memory planning at TP4: full KV544 bytes/token/device/layer including BFP8 exponent headers;142606336 bytes at262144. All16 full layers:2281701376 bytes per device for one max-context user;48 linear states:38486016 bytes. Agent estimate32x32768 tokens:9126805504 KV bytes +1231552512 recurrence/history bytes/device. Weight/storage/padding/trace/activation ledger must be computed from actual setup before acceptance. Native context is not reduced. Exact allocation peaks, persistent CCL/trace reserves and capacity remain unvalidated.

## Measured selection

Both physical mesh probes succeeded. Use the 1x4 fabric ring across all four ASICs, with two links per edge. No subset mesh is supported. The geometry sweeps (`geometry_results.json`, `extra_geometry_results.json`) use real checkpoint weights and identical BFP4/LoFi compute across candidates. The mixed projection's 40-core/K4 input layout improves both layer types relative to the initial 80-core/K2 layout. Out retains 16-core/K3, down 17-core/K8, and gate/up the optimized baseline geometry.

The selected phase contract is hidden-sharded prefill `[B,S,1280]` in/out per device and replicated decode `[B,1,5120]`. Within prefill, RMSNorm gathers per-shard sum-of-squares statistics before normalization; the next column projection gathers normalized activations. Row projections reduce-scatter directly into the sharded residual. Separate prefill gate/up projections share their gathered input. Decode uses local RMSNorm and RS+AG at row projections. State ownership is unchanged between phases. A full-model caller must supply the documented phase layout; consecutive decoder layers consume the previous layer's output directly.

Phase-specific 4096+128 trials (`phase_mix40_l0.json`, `phase_mix40_l3.json`) passed against the single-chip baseline, including full-attention page remapping. The fused CCL family was repaired after an isolated AutoDebug diagnosis of worker-core enumeration across a row boundary; using one worker/link avoids the order mismatch. It is measured with sharded residuals through the subsequent consumers, not forced replication. See `AUTODEBUG.md`, `AUTOFIX.md` and the final alternatives validation before accepting its rejection.

## Final packed-prefill refinement

The single-chip separate-prefill choice was remeasured at TP4's local MLP width. Ten warmed4096-token prefill replays give packed/separate medians8481.95/8557.71us for linear attention and10786.09/10915.79us for full attention; all128 advancing decode comparisons and local states pass (`packed_validation.json`). Select packed prefill and decode, retaining the phase-specific residual contract. Only the `[5120,8704]` local packed gate/up weight is allocated by default; the two separate `[5120,4608]` physical copies are absent. This saves26,542,080 bytes/device/layer,1,698,693,120 bytes/device over64 layers. Measured setup is57,197,056 bytes for linear and54,345,728 for full attention. The updated all-stack feasibility ledger is18,465,972,224 bytes/device for32×32768, including the same reserves. The separate-prefill branch remains an explicit test candidate, not the default.
