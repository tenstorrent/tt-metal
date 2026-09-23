# Explicit streaming SDPA precision recipes

`ttnn.transformer.scaled_dot_product_attention(..., precision=...)` selects a
qualified numerical recipe independently of the compute grid. Omitting
`precision` preserves existing defaults, configurations and feature dispatch.
No existing caller is migrated by this change.

## Choosing a recipe

| `ttnn.SDPAPrecision` | Frozen ID | QK / PV fidelity | Destination / recurrent state | Input preparation |
| --- | --- | --- | --- | --- |
| `FAST` | A | HiFi2 / HiFi2 | BF16 / uncompensated BF16 | None |
| `COMPENSATED` | B | HiFi2 / HiFi2 | BF16 / compensated BF16 | None |
| `BALANCED` | C | HiFi4 / HiFi2 | FP32 / FP32 | None |
| `ACCURATE` | D | HiFi4 / HiFi4 | FP32 / FP32 | None |
| `LOW_PRECISION` | E_bf16 | LoFi / LoFi | BF16 / compensated BF16 | Q RNE7, KV RNE5 |
| `LOW_PRECISION` | E_bfp8 | Same | Same | Q RNE7, KV RNE5 then BFP8 packing |
| `LOW_PRECISION` | E_bfp4 (formerly G) | Same | Same | Q RNE7, KV BFP4-grid RNE with saturation |

All return BF16. Names describe numerical choices, not universal accuracy or
speed guarantees. `FAST` retains the original streaming approximation and
long-context accumulation limitations. B reduces recurrent-state loss but
does not make HiFi2 matmuls exact. C uses the matched, biased cubic exponential
and cheaper subtraction. D retains full-FP32 score subtraction and the
unbiased, refined exponential. FP32 state updates multiply before the separate
L1 addition; neither the rounding point nor the two-iteration reciprocal is
replaced with a fused approximation.

B/E keep high/low BF16 recurrent state and combine two local numerator chunks
when possible. Changed maxima force the qualified fold; an odd final chunk is
flushed, and validity flags reset for each Q block. E still uses HiFi2 for
recurrent/output scaling: using LoFi there would discard state bits even for
an identity multiplier. B requires **no special input rounding**.

Large common components and outliers remain stress cases, especially for low
precision KV. These recipes do not center or rotate inputs or apply fitted
output corrections. The qualification suite records L2, PCC, row-error tails
and absolute errors; PCC alone is not an acceptance criterion. See the
[measured results](sdpa_precision_qualification.md).

## Current support and rejection rules

The current PR2 implementation supports Blackhole devices and uniform SPMD
meshes, dense noncausal unmasked attention, and:

- Q `[B, Hq, Q, 128]`, K/V `[B, Hkv, K, 128]`, with positive dimensions
  and `Hq` divisible by `Hkv` (including grouped-query attention);
- arbitrary positive sequence lengths; fixed Q256/K512 blocks;
- standard 32x32 tiles, minimal sequence tile padding, interleaved DRAM
  inputs/output, and no padding in the other dimensions;
- BF16 Q, BF16 KV for A-D, matching BF16/BFP8/BFP4 KV for E;
- a rectangular origin-based grid with at least one core per batch/query head;
- `scale=None` or the default `1 / sqrt(128)` represented as FP32. A BF16-rounded
  scale is a different value and is rejected; the kernel always uses the default.

The default grid is the device's compute grid. The host assigns a per-head KV
forwarding chain, capped by `max_cores_per_head_batch` (default 16) and available
Q blocks. Unequal chain job counts are supported. Q is double buffered; KV
has one slot for C/D and two slots for A/B/E, matching the frozen schedules.

Explicit recipes reject causal/masked/windowed/sink attention, alternative
scales, sub-core grids, other head dimensions, core-sharded/L1 tensors,
and unsupported architectures. They also reject an explicit
`compute_kernel_config` or `exp_approx_mode=False`: the recipe already owns
those numerical decisions. They never silently fall back to legacy attention.
Do not increase logical K/V lengths to bypass these checks; that changes the
softmax denominator. Physical tile padding is masked by the implementation.

Mesh execution applies the same local operation independently on each device.
Replicated, head-sharded and query-sharded tensors are qualified; KV must be
complete for each local query. This is not sequence-parallel ring attention.

## Joint attention

`joint_scaled_dot_product_attention` accepts the same `precision` and
`inputs_prepared` arguments. Apply LOW_PRECISION preparation separately to both
Q/K/V segments. Omitting `precision` preserves the existing joint implementation.

The joint adapter supports `joint_strategy="rear"` and the same device,
dtype, D128, batch/GQA and grid restrictions above. Each segment must have positive
logical lengths. Segment boundaries and the end of the concatenated sequence
may occur inside a Q or K chunk; Q and K lengths may differ.

The reader maps virtual concatenated tile addresses to the two input segments,
retaining each segment's minimal tile padding;
the writer maps output tiles back to their respective tensors. There is no
materialized concatenation or output slicing. The final partial chunk is zero-filled
by the reader, nonexistent output tiles are discarded, and padded **key** score
columns are replaced with negative infinity before the row maximum and exponential.
Zero-filling K/V alone would incorrectly increase the denominator. This mask is
compiled out for aligned K lengths; valid-score arithmetic, CB depths and the
softmax-state lifetime remain unchanged. The same handling covers sub-tile tails
in dense and joint attention. Preparation clears padding before quantization.

## Ring attention

`ring_joint_scaled_dot_product_attention` accepts the same recipe arguments.
The existing ring reader, active-step scheduler and CCL transport are reused.
Primary/joint KV may be replicated or sequence-sharded as supported by the
existing ring API; prepare E inputs **before** caching or communication.

B/C/D/E execute the shared streaming recipe with one recurrent state across all
active ring contributions. Releasing Q no longer implies final normalization.
Single-Q workers retain state in L1; multi-Q workers checkpoint raw tile bytes
to an internal DRAM buffer. C/D retain FP32 numerator/denominator; B/E retain
both BF16 components, unfinished local groups and global chunk parity. Only the
last active contribution normalizes. A retains the existing ring streaming loop.

Current ring scope is Blackhole, noncausal D128, Q256/K512, batch/GQA, scalar
logical lengths, and the existing `rear` joint strategy. Physical local primary
Q/KV sequence extents must be tile-aligned; `logical_n` masks a possibly
sub-tile global KV tail. Q shorter than local KV requires `is_cross=True`.
Two connected devices are qualified, including unequal worker chains, skipped
shards and replicated/sharded joint inputs. Larger ring topologies are not yet
qualified. Causal/balanced, indexed/paged/chunked-cache, sliding-window, sink,
MLA and device-tensor logical lengths remain legacy-only and reject explicit recipes.
The third returned tensor is internal scratch, **not a supported LSE result**.

`WanAttention` has opt-in `sdpa_precision` and `sdpa_kv_dtype` constructor
arguments for self-attention. E preparation happens after norm/RoPE and before
ring communication; ping-pong KV buffers use the selected storage dtype.
Omitting these arguments retains the original model behavior. Fresh pretrained
attention-block tests qualify this integration, not generated-video quality or
a change to model defaults.

## Examples

```python
cfg = ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=(8, 8),
    q_chunk_size=256,
    k_chunk_size=512,
)
output = ttnn.transformer.scaled_dot_product_attention(
    q, k, v,
    is_causal=False,
    program_config=cfg,
    precision=ttnn.SDPAPrecision.ACCURATE,
)
```

Choose a grid that fits the actual device. For low-precision KV, prepare from
the original BF16 tensors explicitly, before caching or communication:

```python
prepare = ttnn.transformer.prepare_sdpa_input
prepared_q = prepare(q, is_query=True)  # RNE7, BF16 storage
prepared_k = prepare(k, is_query=False, dtype=ttnn.bfloat4_b)
prepared_v = prepare(v, is_query=False, dtype=ttnn.bfloat4_b)
output = ttnn.transformer.scaled_dot_product_attention(
    prepared_q, prepared_k, prepared_v,
    is_causal=False,
    program_config=cfg,
    precision=ttnn.SDPAPrecision.LOW_PRECISION,
    inputs_prepared=True,
)
```

Use `bfloat16` or `bfloat8_b` instead for the other E storage choices. RNE7/RNE5
count significant bits **including the leading bit**, not fraction bits.
BFP4 uses a shared exponent for each native group of 16 values and saturates
rounded magnitudes to the grid maximum. An ordinary dtype cast is not the same
operation. Preparation runs on device, returns new tensors and does not mutate
the originals. There is no CPU copy or hidden attention-side preparation.

`inputs_prepared=True` is a caller assertion, not tensor provenance tracking.
An external producer may implement the same rounding contract; dtype alone
cannot verify it. Prepare static KV once, and keep preparation outside repeated
attention traces unless the original input changes. Preparation itself supports
program-cache hits and trace replay.

The preparation value domain is finite normal BF16 or zero, with no RNE5/7
overflow. For BFP4, nonzero groups require maximum exponent in `[-124, 106]`.
Subnormal/nonfinite behavior is not qualified. Value-domain constraints are
caller responsibilities, not an implicit host scan.

## Implementation and review map

- `sdpa_precision_policy.hpp`: numerical identities, not scheduling knobs.
- `sdpa_numerics.cpp`: conflicts and legacy defaults.
- `sdpa_recipe.cpp`: eligibility, grid/chain assignment, CB formats/capacities,
  and ordinary cached program descriptors.
- `compute/sdpa_recipe.cpp`: recipe specialization; A reuses the existing
  streaming implementation. B/C/D/E share `streaming/recipe_streaming.hpp`.
- `streaming/recipe_sfpu.hpp`, `compensated_sfpu.hpp`, `compensated_group.hpp`:
  selected exponential and state arithmetic. `fp32_state.hpp` is shared by
  attention and its independent component tests, not a separate test implementation.
- `sdpa_input_preparation.cpp`, `compute/prepare_*`: explicit device preparation.
- `tests/.../sdpa/recipe_accuracy_baseline.json`: compact frozen input/output
  digests and metrics, with source hashes; no experimental kernels or media.

Watcher builds use size optimization to fit instrumentation in the instruction
buffer. Release recipes retain their selected optimization settings. Do not use
Watcher timings for performance claims.

No legacy feature coverage is removed. Unqualified configurations remain on
their existing dispatch until they have tested streaming replacements.
Pretrained FLUX/Wan evidence from the research branch is historical, distinct
from the fresh PR2 Wan attention-block qualification.
