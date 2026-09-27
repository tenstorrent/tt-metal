# Fused CCL component experiment plan

Status: Ring component probes completed; see `AUTOFIX_fused_ccl.md` for results and Linear MMRS triage. This is a
shape-faithful synthetic component experiment, not a layer accuracy or speed claim.
Read alongside `mesh_plan.md`. The target is FABRIC_1D, Linear, one link, 1x4
Blackhole. These candidates consume fractured activations directly.

| Candidate | Local input and weight | Local output and immediate consumer | Bytes per rank, algorithmic estimate |
| --- | --- | --- | --- |
| WO matmul + reduce-scatter | Sliding [1,1,32,1024] x [1,1,1024,2816]; full K=2048 | [1,1,32,704] -> distributed RMSNorm -> hidden-sharded residual add | RS 270336 FP32 / 135168 BF16, plus small stats gather |
| Gather + QKV matmul | Normalized [1,1,32,704] -> gathered H=2816; weight [1,1,2816,2048] sliding, N=3072 full | Local QKV output, suitable for local head splitting | AG 270336 FP32 / 135168 BF16, plus preceding stats gather |

Estimates use .75 * 32 * 2816 * element_bytes; topology schedules and headers
are excluded. Both cases use BFP8 weights and HiFi2 with FP32 destination
accumulation. FP32 activations/output is the first candidate. `--dtype bf16`
provides a deliberate precision alternative rather than silently changing the
contract. Actual model precision acceptance still needs real-weight whole-layer
PCC. QKV synthetic columns model local physical widths, including duplicated
full-attention KV storage; they do not assert head-packing correctness.

The probe allocates persistent MMRS intermediate [1,1,32,2816] and output
[1,1,32,704], or AG buffer [1,1,32,2816], before warmup/capture. Buffers are
DRAM interleaved. Separate semaphore sets belong to RS, AG, and norm-stat AG.
The all-core subdevice leaves explicit room for CCL at offset (0,6), while
matmul uses a configurable grid width and six-row envelope. Decode M=32
uses one tile row, so the active compute set is narrow; that underutilization
is a performance consideration, not a reason to omit the experiment.

Initial config: 2D multicast, grid (8,6), K-block=2 tiles, M/core=1,
N/core=ceil(N_tiles/8), subblock1x1. MMRS N/core=11; QKV N/core=8 sliding or12
full. No hidden padding is needed: H/4=704, local WO K=1024/2048, and QKV
N=2048/3072 are all tile aligned. The H-shard is22 tiles, so AG synchronization
and K-block choices must respect that boundary. Config adaptation knobs are
`--grid-x`, `--block-k`, and `--out-block-w`; a first config failure does not
reject the family. Grid11 gives MMRS N/core8 and K-block choices must divide
actual K tiles. Start with supported 2D configs before experimenting with AG's
1D variant.

`tests/probe_multichip_fused_ccl.py` compares each candidate to separate native
TT ops with identical quantized input/weights and matmul configs. Both MMRS
paths include distributed norm and residual; neither immediately gathers the
reduced hidden activation. Both AGMM paths include the preceding distributed
norm. Host conversion happens only outside measured/captured paths. It checks
PCC>=.995 and identical repeated trace output, then reports median warmed trace
host latency. These are component host timings, not device timings or the
4096/128 headline workload. Profiling and real-layer integration are later gates.

Example, only after exclusive device ownership is handed over:

```bash
HF_HUB_OFFLINE=1 python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_multichip_fused_ccl --family mm_rs --layer sliding_attention --output models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/fused_mmrs_sliding.json
```

Run each family and meaningful layer type in a separate process. After any
failed multichip process, follow device-usage reset and health-check procedures
before the next attempt. For a hang, capture triage before terminating. An API
or program-config failure requires adaptation and retry; it is not evidence
that the topology family is slow or unsupported.

## Source contracts inspected

- `tech_reports/LLMs/llms.md`, section3.3: row-parallel outputs can stay
  reduce-scattered; a later column projection gathers its inputs.
- `ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/device/matmul_reduce_scatter_async_device_operation.cpp:35`:
  scatter dim3; only `MatmulMultiCoreReuseMultiCastProgramConfig` (2D multicast).
- Same family `matmul_reduce_scatter_async_program_factory.cpp:84`: calls
  ring-named RS artifact builder and passes selected topology. The name alone
  does not prove Linear unsupported. The builder in
  `reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_program.cpp:339`
  uses optional forward/backward peers and requires even ring size (four meets it).
- `all_gather_matmul_async/device/all_gather_matmul_async_device_operation.cpp:25`:
  rank4, dim3, batch prefix[1,1], 1D or2D multicast; if AG buffer is core-sharded,
  its shard count must equal ring size. DRAM-interleaved persistence avoids this
  unrelated shard-count constraint.
- `tests/ttnn/unit_tests/operations/ccl/test_new_matmul_reduce_scatter.py` and
  `tests/nightly/t3000/ccl/test_minimal_all_gather_matmul_async.py`: public API,
  persistent buffers, subdevice, semaphores, and compute/CCL separation examples.

No hard blocker or performance rejection is established by this preparation.

A recursive architecture-guard search in both fused-family source directories
and the RS artifact implementation found no Blackhole/Wormhole architecture
branch or explicit exclusion. This is source compatibility evidence only;
delegated matmul/CCL compilation and runtime support remain unverified.

The executed adaptation adds `--topology ring` (FABRIC_1D_RING plus Ring CCL)
and `--subblock-w`. Ring initialization and all eight component runs passed.
Tuned MMRS uses grid11,K16,subblock4; tuned AGMM uses grid8,K22,subblock4.
AGMM shows a component host-latency win; MMRS is slower than the equivalent
separate producer/consumer path. Final residual choice still requires actual
whole-layer comparison.

Real-weight integration is now measured at4096/128 for both kinds, including
logical S1, paged cache, runtime fallback guard, and deterministic trace.
See `AUTOFIX_fused_ccl.md`: adapting fused AGMM to the optimized 1D matmul
and L1 output gives only a small full-attention improvement, and loses on
sliding. Complete residual-contract selection belongs to the stage report.
