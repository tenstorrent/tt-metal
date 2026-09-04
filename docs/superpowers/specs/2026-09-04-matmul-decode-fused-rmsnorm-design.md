# Matmul Decode Fused RMSNorm Design

## Goal

Add optional fused RMS normalization to the full-width-sharded
`ttnn.experimental.matmul_decode` path. The implementation reduces only one
sum-of-squares statistic per output row from each producer core, computes the
global RMS scale on one producer-grid hub, and multicasts that scale back to
the producers. It does not gather the full matmul result.

## Public API

The C++, nanobind, device-operation, and Python descriptor APIs add:

- `rms_norm: bool = false`
- `rms_norm_gamma: std::optional<float> = std::nullopt`
- `rms_norm_epsilon: float = 1e-6`

`rms_norm_gamma` is a scalar applied uniformly over N. When `rms_norm` is
enabled, gamma must be present. No gamma tensor is added to the operation.

When `rms_norm` is false, these additions do not change the selected kernels,
CB allocation, synchronization, output placement, or numerical behavior.

## Supported Configuration

The first implementation supports only the full-width-sharded program
factory. Enabling RMSNorm with partial-width-sharded or batched matmul is a
validation error.

Fused RMSNorm is incompatible with `all_gather` and `ring_gather`. It supports
both resident and Global Circular Buffer weight paths and both the general
matmul and Blackhole `custom_mm` compute paths.

The output placement remains unchanged:

- Without `output_core_grid`, the normalized output is width-sharded over the
  weight/producer grid.
- With `output_core_grid`, normalization completes on the producer grid first,
  then the existing output multicast writes a full normalized `[M, N]` replica
  to every destination core.
- Both direct producer multicast and `output_mcast_two_hub` remain supported.

## Numerical Definition

For every logical output row:

```text
x = matmul(input_a, input_b)
sum_sq = sum(x_i * x_i for i in [0, N))
scale = gamma / sqrt(sum_sq / N + epsilon)
y_i = x_i * scale
```

Local sum-of-squares, cross-core accumulation, epsilon addition, reciprocal
square root, and scale generation use FP32 intermediate storage and
accumulation. The normalized output retains the requested matmul output dtype.

`N` is the full logical output width, not the per-core shard width or padded
width. Existing matmul output padding rules continue to apply, and padded
elements must not contribute to the statistic.

## Device Dataflow

When fused RMSNorm is enabled, matmul writes each producer's local
`[M, N_c]` result into a scratch CB instead of directly aliasing the final
width-sharded output.

For each output row:

1. The producer compute kernel retains the local matmul tiles and reduces
   their valid elements to one FP32 local `sum_sq` statistic.
2. The producer publishes the statistic to its data-movement kernel.
3. Producer data-movement kernels unicast their statistics to a designated RMS
   hub: the first row-major core of the weight/producer grid.
4. The hub waits for one statistic from every producer and publishes the
   gathered statistic tiles to its compute kernel.
5. The hub compute path sums the local statistics, divides by full logical N,
   adds epsilon, computes reciprocal square root, multiplies by scalar gamma,
   and publishes one FP32 scale per output row.
6. The hub data-movement kernel multicasts the scale tiles to every producer,
   including itself, and advances the destination CB state on all producers.
7. Every producer waits for its scale, multiplies its retained local output
   tiles by the broadcast scalar, and packs normalized tiles into the final
   local output CB.
8. The existing output writer either leaves that CB as the width-sharded
   output or multicasts its normalized N shard into `output_core_grid`.

The stats synchronization uses dedicated CBs and semaphores so it does not
reuse the activation-gather or output-multicast semaphore IDs.

## Validation

Host validation rejects:

- `rms_norm=true` without `rms_norm_gamma`
- `rms_norm=true` with partial-width-sharded or batched selection
- `rms_norm=true` with `all_gather` or `ring_gather`
- non-finite gamma
- non-finite or negative epsilon
- output widths whose partial final tile cannot be excluded correctly from the
  sum-of-squares reduction

The initial implementation may require N to be divisible by 32 if the chosen
reduction primitive cannot mask a partial final output tile. That requirement
must be explicit in validation and tests rather than silently including padded
values.

## Testing

Tests are added before production changes and must first fail because the new
API or behavior is absent.

Coverage includes:

- Existing full-width output with `rms_norm=false`
- Width-sharded fused RMSNorm against a PyTorch reference
- A non-default epsilon
- Scalar gamma other than one
- Global-CB streamed weights
- Blackhole `custom_mm`
- `output_core_grid` direct multicast after normalization
- `output_core_grid` two-hub multicast after normalization
- Validation failures for missing gamma, unsupported factories, incompatible
  gather modes, and invalid gamma/epsilon

Numerical tests compare against:

```python
x = torch.matmul(a, b)
expected = x * gamma * torch.rsqrt(torch.mean(x.float().square(), dim=-1, keepdim=True) + epsilon)
```

The C++ and CMake changes require a successful `ttnn` build and install.
On-device tests require attached Blackhole hardware; results must distinguish
host compilation from silicon verification.
