# Warmup before `begin_trace_capture`

## Rule

Any op whose program factory parks a device buffer in
`ProgramArtifacts::op_owned_tensors` (page-mapping tables, config tensors,
lookup tables, pad constants, …) **must be run at least once before
`begin_trace_capture`**. Otherwise the op's first post-capture program-cache
miss may allocate that op-owned buffer at an address the trace's recorded
dispatch commands will write to, and `execute_trace` will corrupt it.

## Why

Buffers allocated inside a capture are deallocated when it ends, but their
addresses stay baked into the recorded command stream. Once the trace is
registered as active (`end_trace_capture`), the DRAM / L1 free list is still
free to hand those addresses back out — the allocator only *warns* via
`verify_safe_allocation`. A later cache miss on, e.g., `ttnn.reshape` can land
its cached mapping tensor on one of those addresses. `execute_trace` then
overwrites the mapping; the next reshape (cache hit) reads garbage page
indices and hangs or returns wrong numerics. See
[#61727](https://github.com/tenstorrent/tt-metal/issues/61727).

## Idiom

```python
# Warmup: compile every op you will dispatch after the trace.
_ = ttnn.reshape(x, out_shape)
_ = ttnn.sort(y, dim=-1)

trace_id = ttnn.begin_trace_capture(device, cq_id=0)
# ... ops captured in the trace ...
ttnn.end_trace_capture(device, trace_id, cq_id=0)

# Safe: reshape / sort below are cache hits; their op-owned buffers were
# allocated before the trace registered as active.
ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
ttnn.reshape(x, out_shape)
```

## Known offenders (as of this doc)

Ops that currently cache an op-owned device buffer and will hang / corrupt
without warmup:

- `ttnn.reshape` (tiled) — `reshape_tiled_program_factory.cpp`
- `ttnn.experimental.quasar.reshape` (tiled) —
  `reshape_tiled_metal2_program_factory.cpp`
- `ttnn.sort` — `sort_program_factory.cpp`
- `ttnn.pad` (RM reader/writer) — op-owned pad-value tensor
- `ttnn.conv2d`, `ttnn.halo`, `ttnn.pool` with `config_tensor(s)_in_dram=True`

L1_SMALL placements (conv/halo/pool defaults, upsample sharded) are
structurally safer because L1_SMALL is a separate reserved per-core region
outside the main free list, but **warmup is still recommended** — trace
replay can target any physical address.

## Flags

| Env var | Effect |
|---|---|
| `TT_METAL_FAIL_ON_UNSAFE_TRACE_ALLOC=1` | Throws instead of warning on every unsafe allocation. Use in CI so the hang becomes a named failure. |
| `TT_METAL_TRACE_ALLOC_TRACKING=1` | Populates the allocation context stack so the warning / fatal message names the originating op and its attributes. Also enables `trace_allocation_tracker` for programmatic inspection. |

Running the two flags together is the recommended CI configuration for any
nightly that exercises `execute_trace`.
