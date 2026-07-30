# Composite-operation evidence

The exact-final profiler capture in `../tracy_final/ops.csv` contains
`SDPAOperation` for prefill and `SdpaDecodeDeviceOperation` for decode. The
decode row records paged attention, `q_chunk_size=32`, `k_chunk_size=32`,
`max_cores_per_head_batch=1`, HiFi2, and BFP8 K/V inputs. This is the native
TTNN composite path; decomposed QK-softmax-V would add intermediates.

Packed projection and cache alternatives were executed at both required
batches in `topology_cache_split.log`:

| Candidate | B1 mean | B32 mean | Decision |
| --- | ---: | ---: | --- |
| BF16 cache, packed QKV/gate-up | 0.488666 ms | 0.648506 ms | Reject: B32 regression |
| BFP8 cache, packed QKV/gate-up | 0.488911 ms | 0.632711 ms | Select |
| BFP8 cache, split Q/K/V and gate/up | 0.566527 ms | 0.712643 ms | Reject |

The generic rotary path documented by the completed functional decoder has a
width-divisible-by-64 contract and cannot represent Phi's 96-wide rotate-half
split. The newer Llama-specific composite requires its own transformation
matrix and Llama cache contract, rather than Phi's short/long
position-dependent cache pair. Substitution is not semantics preserving for
Phi LongRoPE. The selected device-only slice/concat/multiply/add topology is
validated at ordinary and LongRoPE positions by `correctness_final.log`, with
no host conversion.
