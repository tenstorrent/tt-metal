# Qwen2.5-Coder-32B optimized multichip decoder

Status: complete. Independent `$stage-review` returned `clean-pass`. This stage optimizes the completed `MultichipDecoder` in place. It does not add a full model, generator, LM head, vLLM, or serving path.

## Outcome

The selected path is the real TP4 decoder on four Blackhole p300c devices in a `MeshShape(1,4)` `FABRIC_1D_RING`. It is neither a single-chip nor a replicated fallback.

| Real layer 32, batch 32, logical sequence 17 | Starting default | Final default | Change |
| --- | ---: | ---: | ---: |
| Prefill PCC | 0.993392 | 0.992527 | accepted, above 0.99 |
| Decode PCC | 0.994006 | 0.993698 | accepted, above 0.99 |
| Warmed prefill median | 3.577945 ms | 3.541916 ms | 1.01% faster |
| Warmed traced decode median | 0.791976 ms | 0.758020 ms | 4.29% faster |
| Trace measurement | 7 trials | 7 trials | 100 replays/trial; bitwise stable |

The final numbers are from `results/final_default.json`, the final source defaults, and the final seven-trial/100-replay run. The independent TTNN reference artifact is `/tmp/qwen2_5_coder_32b_optimized_baseline.pt`; all real weights are revision `381fc969f78efac66bc87ff7ddeadb7e73c218a7` of `Qwen/Qwen2.5-Coder-32B-Instruct`.

The final default is:

- TP=4 on mesh axis 1; each rank owns 1280 hidden channels, 10 Q heads, 2 KV heads, and 6912 logical MLP channels.
- BFP8/LoFi attention weights, BFP4/LoFi MLP weights, BF16 activations/residual/CCL, and BFP8 KV cache.
- Packed QKV and one packed gate+up decode matmul. Gate+up uses 32 cores; QKV/down use 16; O uses 8; decode SDPA uses 8x4.
- Two hidden all-gathers and two row-parallel reduce-scatters per layer, with persistent two-link async CCL buffers shared by sequential layers.
- No layer-boundary collective, gather, all-reduce, reduce-scatter, or reshard.

Qwen2.5-Coder-32B is a dense model, so MoE active-expert requirements do not apply. This decoder has no terminal LM-head/sampler path.

## Inter-layer residual contract

Full-model bringup must preserve this contract:

- Decode producer output: BF16 TILE TP fracture with logical shape `[1,32,1,1280]` per rank in L1 `WIDTH_SHARDED` memory.
- Canonical consumer view after the metadata-only reshape: `[1,1,32,1280]`, 10x2 core grid, physical shard `[32,64]`.
- A reshape may attach ND-shard metadata, so compatibility is defined by physical memory layout, buffer type, shard grid, shard shape, and orientation. Equal physical placement must not trigger `to_memory_config`.
- Prefill producer output remains the same BF16 TP fracture in DRAM and is consumed directly by the next decoder; no boundary collective or reshard is inserted.
- Public or foreign decode inputs are converted once if they do not already satisfy the placement. Homogeneous decoder-to-decoder handoff is a view only.

The two-layer synthetic gate directly passes one decoder's output into the next and checks the canonical L1 placement before the second layer. It achieves PCC 0.990737 for the decode handoff and 0.990702 for the non-aligned prefill handoff. The compiler-provenance replicated-boundary family is numerically identical but slower when traced: 0.842635 ms versus 0.758101 ms for the selected fractured family. Therefore no inter-layer collective remains.

## Operation-topology audit and actions

This audit was performed before tuning. The retained Tracy reports and final source were then used to close every material item.

| Topology area | Audit finding | Action and evidence |
| --- | --- | --- |
| Layer boundary | Decode converted the final L1 fracture to DRAM and restored L1 at the next layer | Removed the return conversion; added placement-aware view handling and a direct two-layer gate. L1 candidate was 0.786000 ms versus 0.791976 ms before. |
| Repeated same-input matmuls | Gate and up separately read the same 5120-wide normalized activation | Packed into one physical `32x5120x14336` BFP4/LoFi matmul and split on device. Full retained-family result was 0.782612 ms. A 16-core first-CB failure was adapted to 32 cores. |
| Input norm/QKV movement | Hidden AG, norm placement, and QKV reshard are material | Distributed-statistics norm was correct but 0.795022 ms and 4.651343 ms prefill. Fused AG+QKV was adapted across rank-4 weights, 2-link/1-link CCL, 8x4/8x1 grids, persistent gathered buffers, and packed/unpacked projections; its final-policy 7x100 coherent family was 0.930669 ms and rejected. |
| Attention head padding | Local 10Q/2KV decode requires a legal physical SDPA layout | Model owns padding: two logical five-Q GQA groups are internally expanded to 16 each, concatenated to 32, then sliced back. SDPA 8x4 was retained; group width 8 was 0.772048 ms and rejected. Public lengths/heads stay logical. |
| O/down row-parallel output | Matmul output conversion plus reduce-scatter is material | DRAM-sharded matmuls retained. Fused matmul+RS compiled and was correct but 0.885514 ms. O/down core and block families were measured; no variant beat the selected coherent default. |
| Collective placement | Two AG and two RS are required by the full-K/full-N projections | Persistent BF16 async collectives retained. Replicated boundary with two all-reduces was 0.842635 ms traced; selected fractured AG/RS family was 0.758101 ms at PCC 1.0. |
| Collective dtype | BF16 payload might be reducible | BFP8 exact AG/RS focused traces passed after adapting persistent-buffer placement. Crossed with the final policy and 8x4 SDPA at 7x100, it gave PCC 0.992494/0.993622, prefill 3.708916 ms, decode 0.787956 ms. Rejected. |
| Persistent buffers | Allocation/rebinding may dominate trace | Disabling persistent CCL buffers under the final policy/8x4 family was correct but 0.771160 ms versus 0.758020 ms final. Retained shared ping-pong buffers. |
| Precision/fidelity | Attention and MLP precision dominate DRAM traffic | BFP8/LoFi attention plus BFP4/LoFi MLP was the fastest policy above the 0.99 PCC floor. All-BFP4 variants reached 0.758270-0.780703 ms but prefill PCC fell to 0.989796-0.989919, so they were rejected. |
| Activation dtype | BFP8 matmul outputs could reduce local movement | The old global probe was followed by final-policy group isolation: attention-only was 0.771944 ms; MLP-only was 0.759567 ms at 7x100 and PCC 0.992527/0.993653. Both lost to 0.758020 ms BF16. |
| KV dtype | Cache bandwidth may favor BFP8 | BF16 KV control was 0.785161 ms; BFP8 remains selected. Both contiguous and paged BFP8 caches pass. |
| Prefill projections/CCL | Packed gate/up, four collectives, and cache fills dominate | 10x10/block-limit-10 retained at 3.571924 ms control; 8x10 was 3.680131 ms. Block 20 failed CB capacity, then adapted 8x10 block-20 and block-16 retries also failed exact CB capacity. Final seven-trial default is 3.541916 ms. |

The complete machine-readable ledger is `results/candidate_summary.csv`; every row links to its JSON artifact. Material first errors were adapted or sent through AutoFix rather than treated as final rejections.

## Coherent family comparisons

Lower-movement changes were evaluated as families, without immediately restoring the prior replicated residual:

| Family | Best valid evidence | Decision |
| --- | --- | --- |
| Residual placement | L1 fractured boundary, 0.786000 ms; direct two-layer handoff passes | retain |
| Collective placement | Fractured `2 AG + 2 RS` 0.758101 ms versus replicated `2 all-reduce` 0.842635 ms, PCC 1.0 | retain fractured |
| Fused CCL+matmul | Adapted one-link persistent fused gate AG plus direct-up consumer, final-policy 7x100 0.930669 ms | reject |
| Packed versus separate projections | Packed gate+up 0.782612 ms versus split starting path 0.791976 ms | retain packed |
| Activation/CCL dtype | Final-policy attention-only BFP8 0.771944 ms, MLP-only BFP8 0.759567 ms, and BFP8 CCL 0.787956 ms | retain BF16 |
| Persistent buffers | Final-policy disabled 0.771160 ms versus enabled 0.758020 ms | retain persistent |
| DRAM-sharded versus advisor L1 projections | Current family 291.6 us versus advisor family 381.2 us across QKV/O/gate/down | retain DRAM-sharded |

The shard-advisor capture produced 16 ops, 15 choices, and zero L1 spills. Isolated projection measurements were QKV 46.588/43.342 us, O 32.812/35.087 us, gate 139.139/148.627 us, and down 73.047/154.139 us for current/advisor respectively. The advisor's QKV win did not offset losses in the coherent projection family.

## tt-perf-report findings

Compact CSV and text reports live under `tracy/layer32/`; `tracy/layer32/provenance.json` records source, weights, command, raw hashes, and compact hashes. Raw Tracy/profile tables were omitted from git after report generation.

Decode's merged four-device report contains 72 device ops and 595 us of device work. The warmed trace benchmark, not the profiler's merged host-gap total, is the end-to-end authority.

| Decode role | Physical matmul / op | Device time | Selected program/layout |
| --- | --- | ---: | --- |
| QKV | `32x5120x2048`, BF16 x BFP8, LoFi | 26 us | 16-core DRAM-sharded weight; local logical N=1792 |
| SDPA | local padded 32Q, BFP8 cache | 19 us | 8x4, logical 10Q/2KV |
| O | `32x1280x5120`, BF16 x BFP8, LoFi | 19 us | 8-core DRAM-sharded |
| Hidden AG | two BF16 async Ring AG | 11 us each | persistent, two links |
| O/down RS | BF16 async Ring RS | 23/20 us | persistent, two links |
| Packed gate+up | `32x5120x14336`, BF16 x BFP4, LoFi | 134 us | 32-core program; dominant device op |
| Down | `32x7168x5120`, BF16 x BFP4, LoFi | 67 us | 16-core DRAM-sharded |

The report's actionable matmul advice was exercised through gate 16/32/64-core, down 16/32-core, QKV/O core sweeps, six K-block variants, advisor layouts, packed/split projections, and fused AG/RS paths. Its fidelity advice was exercised by the precision sweep and accepted only above the PCC gate. Its head-padding/layout gaps were exercised with SDPA 8x4, 8x8, and group-width variants.

Prefill's merged report contains 222 device ops and 1992 us of device work. Key device costs are packed gate+up 356 us, down 185 us, QKV 72 us, O 57 us, hidden AG 83/81 us, and RS 87/90 us. The measured prefill grid/block and fused-collective families cover those costs.

The per-rank decode projection/cache read floor is approximately 80.35 MB. At an optimistic 512 GB/s this is 0.157 ms, versus 0.595 ms merged device work and 0.758 ms end-to-end traced latency. The report's modeled overall DRAM utilization is 121 GB/s (23.7%), consistent with the remaining head-layout, collective, and mixed-op overhead rather than one untried local matmul knob.

## Correctness, stress, and safety gates

- Real layer 32: prefill/decode PCC 0.992527/0.993698; K/V PCC 0.999598-0.999847.
- Synthetic logical sequence 31: prefill 0.996387, decode 0.996491, two-layer prefill/decode 0.990702/0.990737.
- Contiguous versus paged prefill/decode and logical K/V layouts: PCC 1.0.
- Trace positions 32 and 33: output PCC 0.996469/0.996472; K/V PCC at least 0.999783; ten-replay output PCC 1.0 and bitwise deterministic.
- Separate in-place paged trace-table/position refresh: passed.
- Final performance trace: 7x100 replays and bitwise-equal repeated output.
- Runtime fallback audit: all owned prefill/decode/collective methods are free of `torch`, `from_torch`, `to_torch`, and `super()` runtime fallback tokens.
- Valid non-aligned public lengths remain supported. The sequence-31 gate proves internal padding/masking/slicing ownership; no aligned-only public requirement was added.
- Capacity: 12224 passes with 16,593,920 DRAM bytes/device free; 12225 pads to 12288 and fails with a measured 1,606,418,432-byte allocation request. `doc/context_contract.json` retains the largest physical value and points to adjacent evidence.
- Watcher: the final real layer-32 path passed with worker/dispatch monitoring on all four devices and no fault-pattern matches. Full ETH instrumentation was attempted first but its 27,920-byte active-Ethernet Ring program exceeds the runtime's 25,600-byte kernel-config buffer before model execution; the retained gate therefore disables ETH monitoring only. This is a tooling-capacity exception, not a model failure.

## Artifacts

- `results/before_default.json`, `results/final_default.json`: reproducible before/final PCC and latency.
- `results/candidate_summary.csv` and `results/sweep_*.json`: candidate evidence and rejection provenance.
- `results/topology_family_benchmark.json`: fractured versus replicated topology.
- `results/synthetic_correctness.json`, `results/paged_trace_refresh.json`: non-aligned, stacked, paged, and trace checks.
- `results/capacity_seq12224.json`, `results/capacity_seq12225.json`: hard context boundary.
- `results/watcher_clean.json`, `results/watcher_clean.log`: monitored final correctness run.
- `validation/fused_ag_autofix.md`, `validation/bfp8_ccl_autofix.md`: adapted material-candidate investigations.
- `validation/advisor_projection_compare.json`, `shard_advise/result/report.json`: compiler advisor and device comparison.
- `tracy/layer32/decode_perf_report.{csv,txt}`, `prefill_perf_report.{csv,txt}`, and `provenance.json`: compact profiler evidence.
- `work_log.md`: exact command/provenance log, including recovery and review history.

Known non-model environment notes are the harmless recurring MPI `/dev/shm` warning and the Watcher ETH instrumentation limit described above. No optimization is deferred, and no full-model or vLLM work was started.
