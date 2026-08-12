# Mistral-Small-24B multichip decoder mesh plan

Status: frozen before implementation on 2026-07-17; acceptance outcomes appended after measurement.

## Target hardware and topology

The final path targets the complete local Blackhole p300c mesh: logical `1x4`, tensor-parallel degree 4 on mesh axis 1, two fabric links per device. `tt-smi -ls --local` found four healthy devices, and a real `ttnn.open_mesh_device(ttnn.MeshShape(1, 4))` smoke passed with device IDs `[3, 2, 1, 0]`, compute grid `11x10`, eight DRAM banks, and `34,178,731,008` DRAM bytes per device. The system mesh descriptor is physically `2x2`; the model uses a logical `1x4` fabric so every model tensor has one unambiguous TP axis. The initial collective baseline is the compiler-emitted replicated-residual plan with an all-reduce after attention output and MLP down projection. Final measurement selected Linear topology with two links: 0.114246 ms versus Ring at 0.117872 ms for the actual 327,680-byte BF16 decode payload.

This choice is fixed before final-path coding because it uses all available chips, exactly matches the authoritative `1x4` TTNN IR provenance, evenly divides all heads and channel dimensions, and keeps the decoder stack input/output contract unchanged. The code is intentionally not required to support smaller meshes.

## Tensor contract and per-device shapes

TTNN weights use `[in_features, out_features]`. The packed QKV tensor must be packed independently per rank as `q_rank || k_rank || v_rank`; naively sharding a globally packed `Q || K || V` tensor would assign the wrong head ownership.

| Tensor | Global TTNN shape | Distribution | Per-device shape | Physical dtype |
| --- | --- | --- | --- | --- |
| input/output residual | `[1, batch, seq, 5120]` | replicated baseline | same | BF16 |
| input/post-attention RMS weights | `[5120]` | replicated | `[5120]` | BF16 |
| packed QKV weight | `[5120, 6144]` | column/heads, TP4 | `[5120, 1536]` = Q 1024 + K 256 + V 256 | BFP4_B |
| query activation | `[batch, 32, seq, 128]` | heads, TP4 | `[batch, 8, seq, 128]` | BF16 |
| key/value activation | `[batch, 8, seq, 128]` | heads, TP4 | `[batch, 2, seq, 128]` | BF16 |
| output weight | `[4096, 5120]` | row/input channels, TP4 | `[1024, 5120]` | BFP4_B |
| packed prefill gate/up weight | `[5120, 65536]` | column/intermediate, TP4 | `[5120, 16384]` | BFP4_B |
| decode gate and up weights | `[5120, 32768]` each | column/intermediate, TP4 | `[5120, 8192]` each | BFP4_B |
| down weight | `[32768, 5120]` | row/intermediate, TP4 | `[8192, 5120]` | BFP4_B |
| contiguous K/V cache | `[batch, 8, max_seq, 128]` | KV heads, TP4 | `[batch, 2, max_seq, 128]` | BFP8_B |
| paged K/V cache | `[blocks, 8, block_size, 128]` | KV heads, TP4 | `[blocks, 2, block_size, 128]` | BFP8_B |
| page table / current positions / RoPE tables | logical public shape | replicated | same | INT32/BF16 |

Logical sequence lengths are never rounded in the public API. Tile padding and MLP chunk padding are implementation details. A tail chunk is sliced back to its logical length, and page tables/current positions remain indexed in logical tokens.

## Collective and stacked-layer strategy

Each rank computes local Q/K/V heads, local attention, and a row-parallel output projection. Its partial hidden tensor is reduced across TP axis 1. RMSNorm, residual addition, and the next column-parallel projection consume the resulting replicated hidden tensor. The MLP similarly computes a local 8192-wide gated activation, a row-parallel down projection, and a second hidden all-reduce. Thus the baseline has two all-reduces per decoder layer in both prefill and decode and requires no all-gather for activations.

The decoder remains stack-compatible: it accepts and returns `[1, batch, seq, 5120]` replicated on all four devices. K/V caches remain head-sharded between layers and calls. Page tables and positions are replicated because every TP rank processes the same users and logical tokens.

Collective candidate audit and final disposition:

| Candidate | Expected benefit | Acceptance test |
| --- | --- | --- |
| local matmul + synchronous all-reduce | exact emitted IR, simplest trace ownership | selected; trace safe and fastest measured consumer chain; Linear/two-link topology |
| local matmul + asynchronous all-reduce with persistent semaphores | overlap/less dispatch | rejected: no supported Python async API with explicit trace-lifetime buffers for this path; synchronous replay already removes dispatch overhead |
| matmul + reduce-scatter, hidden-sharded residual | half of a reduce-scatter/all-gather pipeline can lower traffic | rejected by actual consumer chain: 0.155569 ms versus replicated 0.102029 ms |
| reduce-scatter followed by all-gather | preserves replicated contract but adds a round trip | measured only through distributed RMSNorm and QKV, never as an isolated round trip; rejected with the chain above |
| fused all-gather+matmul / matmul+reduce-scatter | hides communication in adjacent GEMM | rejected: no applicable trace-safe fused Blackhole binding for these shapes; even perfect overlap of the ~20 us final gather cannot recover the measured 53.54 us deficit |

Persistent collective buffers would be required for an async trace path. The winning synchronous `ttnn.all_reduce` owns its scratch internally and exposes no persistent Python semaphore/buffer contract; no artificial buffer is retained.

## Capacity calculation for the full-model handoff

The estimate uses TT tile storage rather than nominal scalar widths: BFP4_B is 576 bytes/tile, BFP8_B is 1088 bytes/tile, and BF16 is 2048 bytes/tile. Per device:

- one decoder layer's TP4 BFP4 matrix weights: `78,151,680` bytes;
- all 40 decoder layers: `3,126,067,200` bytes;
- all replicated layer norms: `26,214,400` bytes; TP4 BF16 embedding and untied LM head: `335,544,320` bytes each; final norm: `327,680` bytes;
- complete fixed full-model decode-weight lifetime: `3,823,697,920` bytes per device;
- batch-32 BFP8 K/V cache across all 40 layers: `696,320` bytes per logical token;
- cache at 32,768 tokens: `22,817,013,760` bytes;
- shared tiled prefill plus row-major decode RoPE at 32,768: `33,554,432` bytes; shared position indices: `131,072` bytes; page table: `131,072` bytes; persistent decode positions: `128` bytes;
- complete steady state at the advertised context: `26,674,528,384` bytes, leaving `7,504,202,624` bytes before the runtime reserve.

The context-scaled lifetime is `697,352` bytes/token: K/V `696,320`, both RoPE representations `1,024`, position indices `4`, and page-table entries `4`. Reserving a physically allocated 4 GiB for activations, traces, programs, and collective scratch yields a tile/block-aligned calculated cache ceiling of 37,344 tokens, above the Hugging Face 32,768-token contract. At 32,768 the measured arithmetic leaves another `3,209,235,328` bytes beyond that reserve. Therefore TP4 memory does not justify reducing advertised context. A single materialized BF16 residual for batch 32 and 32,768 tokens is 10,737,418,240 bytes per rank, so full-context prefill must be streamed/chunked rather than materializing every full-width layer intermediate concurrently. Decode and paged-cache capacity are the binding full-model handoff contract.

The physical gate instantiates two real decoders sharing one immutable RoPE bundle, releases their duplicate prefill matrices, allocates the other 38 layers' exact local decode matrix/norm shapes and shard specs, allocates TP embedding/head/final norm, all 40 local K/V pairs, page table and position vector, then reserves 4 GiB per rank and executes paged decode at position 32,767. This is a cache/decode-handoff capacity result, not a claim that batch-32 32K prefill can materialize a full residual.

## TP-local decode geometry selection

The first profiler showed DRAM-sharded matmuls as the largest tunable group. A bounded real-weight sweep held precision, collectives, cache policy, and output fixed, used 50 replay iterations per process, and required PCC at least 0.9999 against the original geometry. The original `(attention 10,12,4,8,10,4; MLP 40,32,40,4,8)` decode median was `0.609677 ms` over three fresh runs. The selected `(attention 10,12,16,8,10,4; MLP 10,32,40,16,16)` median was `0.579591 ms`, a `4.934%` improvement, with PCC `0.999989`. Wider attention/MLP core grids and narrower intermediate variants were slower; all raw runs are in `evidence/geometry_sweep.csv` and their command logs.

The asymmetric MLP layout intentionally maps the input across 10 cores, the local 8192-wide intermediate across 32, and the hidden output across 40. `MultichipDecoder._mlp_forward` makes that down-projection output shard explicit, eliminating TTNN's otherwise-correct runtime layout override.

## Rejected mesh/model alternatives

- TP1 and TP2 leave available devices idle, retain too many weights and KV heads per device, and do not match the emitted graph.
- Data parallelism replicates all weights and caches and cannot accelerate a single batch-32 decoder layer.
- Sequence parallelism as the primary split complicates causal prefill and does not reduce decode weight traffic; it is not the emitted plan.
- Expert parallelism is inapplicable: this Mistral variant is dense and contains no MoE layers.
- A public aligned-only sequence contract is rejected. Tests pass logical sequence lengths 17, 18, and 32; internal tile/block padding remains invisible.

## Post-plan acceptance

The plan was implemented without changing the target mesh or advertised context. The complete 40-layer steady-state envelope described above passed paged decode at position 32,767 with the physical 4 GiB reserve resident. Real layer-20 checkpoint comparison against the optimized TP1 TTNN baseline passed at 0.999994 prefill PCC, 0.999990 decode PCC, and 1.0 K/V PCC. The final exact-code timing produced 2.222466× warmed decode speedup and 55.562% TP4 efficiency; the three-run geometry finalist median was 0.579591 ms. Full evidence and limitations are in `README.md` and `evidence/README.md`.
