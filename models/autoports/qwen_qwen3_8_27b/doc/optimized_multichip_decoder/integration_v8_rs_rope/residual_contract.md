# Inter-layer residual and workspace contract

Selected default for a1x4 tensor-parallel mesh of four Blackhole p300c ASICs, FABRIC_1D_RING, axis1, two links. This is partitioned model execution: mix/gate-up own local output channels, output/down consume local input channels, and each ASIC owns one KV head or12 linear-attention value heads. Replicated residual activations do not imply replicated weights or single-chip fallback. Final acceptance artifacts validate this contract separately.

|Phase / logical batch|Per-device input and output|Boundary storage|Internal row collective|
|---|---|---|---|
|Prefill, all supported batches|[B,S,1280], hidden-sharded across ASICs|BF16 DRAM|BFP8 RS payload/output followed by BF16 residual; native fused RS at B1×4096 working rows|
|Decode B1|[1,1,5120], replicated residual|BF16 L1 width-sharded8x1, shard[32,640]|Direct asynchronous all-reduce, BFP8 payload|
|Decode B2–7|[B,1,5120], replicated residual|BF16 L1 width-sharded10x8, shard[B*32,64]|Same direct all-reduce|
|Decode B8–32|[B,1,1280], hidden-sharded residual|BF16 interleaved L1|Native DRAM-weight matmul+reduce-scatter, BF16 payload/output|

Call `decode_residual_layout(batch)` during setup and place the initial residual accordingly. `decode_implementation(batch)` returns a shared-weight view; it performs no conversion. A legacy replicated5120-channel decode input remains supported through the TP4 all-reduce path, but final optimized measurements use the advertised layout. `DecoderStack` passes each output directly to the next decoder: no wrapper gather, reshard or all-reduce. Norm/projection helpers own internal working-layout conversions. A future generator must produce each phase's initial residual in its advertised layout. If it reuses a residual across phases, convert once at that phase boundary; no conversion is required between decoder layers.

For linear-attention decodeB8–32, four-row temporal windows combine each user's three raw history rows and current projected QKV row. One convolution processes the packed sequence; only each fourth result is retained. Assembly/extraction uses row-major layout and internally pads nonaligned batches. `packed_conv_exact.json` compares Q/K/V and raw history exactly against the prior loop on all devices. No persistent state format changes.

Sharded decode uses distributed RMSNorm, transient L1 statistics/hidden all-gathers, compact column inputs[1,B,5120], and packed gate/up. Fused row projections return compact[1,1,B,1280] before the internal expansion to public[B,1,1280]. Residual additions retain the fused output memory. B represents logical users; tile padding rows never represent users. Prefill owns logical sequence padding, masking, page updates and slicing, including nonaligned4095/4097 and native context. No caller alignment rule is added.

## One owner for the mesh lifetime

Construct the first decoder and pass its `ar_workspace` as `ccl_workspace=` to every subsequent decoder. The owner holds one TT_CCL semaphore context, two BFP8 AR scratch slots and native-RS role buffers for every logical batch8–32. All buffers are allocated at construction before any trace capture. Views and layers share them; none are multiplied by layer count.

The AR pair uses1,392,640 bytes/device across8 cores (174,080 bytes/core). For each of25 native-RS batch signatures, two roles each own one DRAM intermediate and one interleaved L1 output. DRAM total is16,384,000 bytes/device. Bank-granular L1 reservation is4096 bytes/bank/signature, hence102,400 bytes/bank or11,264,000 bytes/device across110 banks. Physical M is32 even for logical batch8. These are independent of the number of layers and live traces. The memory ledger subtracts the shared DRAM pool from per-layer setup before multiplying48/16 layer counts, then adds it once.

All users enqueue complete attention-out/MLP-down pairs in the same order on the same CQ. Two slots are necessary: the intervening collective proves all ranks completed the previous use of the other role. One scratch/output slot is not safe merely because the next launch has begun. Both native RS and AR enforce this pairing while graphs are built. The final layer output is a separate residual-add result, not a retained alias of shared row scratch. Separate concurrently executing CQs/subdevices need separate owners or external serialization. A partial failed enqueue invalidates the owner.

Keep the owner, buffers and semaphores alive until all referencing traces retire and queued work completes. Construct every decoder, request/state/input/output buffer, and warm every supported execution signature before capture. No per-token host fence is required. This decoder provides no scheduler or generation layer. Final synthetic workspace, real mixed-batch trace, two-layer stack and watcher artifacts are the ownership acceptance evidence; full-model/vLLM execution is not claimed.

Source: tt/decode_workspace.py, tt/fused_decode.py, tt/multichip_decoder.py and tests/run_multichip_decoder.py:DecoderStack. Candidate memory controls and rejected private-buffer options remain in topology_audit.md and work_log.md.

Standalone prefill collectives use8 workers/link, BFP8 hidden-AG/RS payload and actual BFP8 inputs for mix and separate gate/up. Distributed-norm statistics and residuals stay BF16. Full-attention mix AG writes directly into interleaved L1 only when its padded working row count is<=4096; its matmul consumes that allocation directly. Larger row counts use DRAM. This is an internal transient choice, not an inter-layer layout change. The shared prefillRS pool owns55,705,600 DRAM bytes/device for out/down, each with4096×5120 intermediate and4096×1280 output, BFP8 tiles. RS itself needs no extra weight copy; separate prefill gate/up weights are accounted below. Allocation precedes all captures; both fused and separate rows participate in the owner's complete out/down prefill protocol. Errors invalidate the owner; concurrent CQs require separate owners or explicit serialization.

## Collective API settings

All selected CCL paths use the TP4 ring and two links; hidden/statistics collectives use tensor dim3, and direct all-reduce uses mesh axis1. Actual row times and input/output memory appear in final runtime_contract.json and batched_runtime_contract.json.

| Path | Payload | Input/output memory | Workers/link; chunks/sync; buffers/channel | Persistent output / intermediate |
|---|---|---|---|---|
| B1 decode direct AR | BFP8 input, BF16 output | L1 width-sharded | Direct-AR API does not expose these three knobs | Separate transient output; shared two-slot persistent scratch |
| B8–32 decode norm/hidden AG | BF16 | L1 interleaved | 2;10;2 | Transient; persistent candidate measured slower |
| B8–32 decode fused MM/RS | BF16 | L1 activation, DRAM MM intermediate, L1 RS output | Auto:2 workers/direction; nullopt; nullopt | Both preallocated per batch/role, shared across layers |
| Prefill norm-stat AG | BF16 | DRAM interleaved | 8;10;2 | Transient; persistence tried |
| Prefill hidden AG | BFP8 | DRAM input; DRAM output or full-mix L1 output at<=4096 padded rows | 8;10;2 | Transient; persistence tried |
| B1x4096 prefill fused MM/RS | BFP8 MM/RS output, BF16 activation | DRAM activation/intermediate/output | Auto:4 workers/direction; nullopt; nullopt | Both preallocated per role, shared across layers |
| Other prefill row RS | BFP8 | DRAM interleaved | 8;10;2 | Transient; persistence tried |

For fused RS the public wrapper does not expose standalone CCL worker/chunk/buffer knobs: matmul_reduce_scatter_async_device_operation.cpp:175–193 forces persistent buffers and one worker/link, leaving chunk/buffer options nullopt for lower-level selection. Do not interpret the standalone8-worker policy as a fused-kernel setting. The direct-AR scratch/output distinction likewise avoids claiming persistent outputs where only scratch is preallocated. Candidate comparisons and rejection reasons are in topology_audit.md and prefill_ccl_findings.md.

V4 prefill projection update: distributed post norm gathers the hidden input once in BF8 DRAM for separate gate/up matmuls; gate applies fused SiLU. The independent BF16 product releases gathered input/gate output before down. Inter-layer residual and shared workspace contracts above are unchanged. Additional gate/up weights are layer-owned25,067,520bytes/device/layer; packed gateup remains for decode and one-row prefill. No extra inter-layer collective or conversion is introduced.

Fused RS worker accounting: the wrapper serializes a worker field of1, but matmul_reduce_scatter_async_program_factory.cpp passes nullopt for workers, chunks and buffers to the minimal-RS builder. Its size/placement heuristic chooses4 workers/direction for4096-row prefill (20CCL cores at offset0,8) and2 for physicalM32 decode (12CCL cores). Thus current reported core counts84prefill/20decode include64/8 matmul cores plus CCL workers/mux cores; they are not pure matmul grid sizes. Larger10x8 and11x8 prefill MM grids are being compared at the same CCL placement.
