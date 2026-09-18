# Certified matmul registry

The matmul registry selects measured program and compute-kernel configurations
for exact TTNN matmul calls. It is disabled by default, and a lookup miss keeps
the existing TTNN behavior.

## Runtime modes

Set `matmul_registry_mode` through `TTNN_CONFIG_OVERRIDES` or
`ttnn.CONFIG.matmul_registry_mode`:

- `off` skips registry lookup.
- `shadow` performs lookup without applying the result.
- `on` applies an exact match and otherwise uses the existing TTNN path.

For example:

```bash
TTNN_CONFIG_OVERRIDES='{"matmul_registry_mode":"on"}' pytest ...
```

Use the existing fallback guard when a test must prove that every relevant call
has a registry entry:

```bash
TTNN_CONFIG_OVERRIDES='{"matmul_registry_mode":"on","throw_exception_on_fallback":true}' pytest ...
```

The model CI workflows expose the same `off`, `shadow`, and `on` choice, so the
same model command can be compared without model-specific registry code.

## Current coverage

The dense table contains 60,221 exact Blackhole entries over 1,302 matrix
shapes. It accepts single-device, interleaved tensors with rank at least two
when every dimension before the final matrix dimensions is `1`. Calls with
sharded tensors, mesh-wide tensors, explicit program configs or core grids, or
active trace capture use the existing TTNN path.

The all-gather matmul table contains 36 eight-device entries and 104
thirty-two-device entries. An exact match owns both the program config and the
compute-kernel config; a miss leaves the caller's configs unchanged.

## Updating the tables

Sweep results and the generators live in the `tt-matmul-codegen` repository.
From that checkout, install freshly generated tables into a TT-Metal checkout:

```bash
scripts/pipeline.sh registry ~/tt-metal
scripts/pipeline.sh multichip-registry ~/tt-metal
```

Commit the generated C++ together with any generator change. TT-Metal builds do
not run Python or regenerate the tables.

After updating a table:

1. Run the focused registry tests and build the TTNN tests.
2. Run the same model pipeline with the registry `off` and `on`.
3. Use strict `on` mode when claiming registry coverage rather than simple
   compatibility.
4. Use `off` versus `shadow` to isolate lookup overhead from recipe changes.
