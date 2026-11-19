# AR worker placement: coherent residual-grid adaptation

Prepared source-only after the original custom-worker jobs failed role overlap at(5,3). This is an adaptation retry, not rejection based on the first API error. Original files/results remain unchanged; no hardware was opened or run.

## Why the original placement overlaps

`experimental/ccl/llama_common.cpp:9–32` chooses custom workers(5,3),(6,3) for two links. The original residual output grid(0,0)..(7,3) includes both. `all_reduce_async_program_factory.cpp:220–260` reserves the output-grid bounding box and removes it from automatic-worker availability, but the custom branch bypasses that availability filter. The same core is then assigned output reduction-receiver and worker-reader roles. This is a layout incompatibility, not evidence that the collective algorithm is inherently slower or numerically wrong.

## Primary candidate: shifted8x4

New `tests/multichip_ar_residual_grid_candidate.py` defines a setup-owned shared workspace and decoder subclass. Residual/output/receive grids move to(0,4)..(7,7),32 cores. Residual shards remain `[32,128]`, ROW_MAJOR WIDTH_SHARDED BF16 L1. The two AR receive buffers stay `[1,1,32,16384]` with `[32,512]` shards on the same shifted grid. Custom workers remain(5,3),(6,3), now disjoint from the output bounding box. Automatic and custom variants use the same two-link topology, NoC policy, precision, workspaces, and residual layout.

RMSNorm supports this offset in current source:

- `normalization/shard_spec_validation.cpp:36–46` subtracts the shard bounding-box start before validating it against program-grid dimensions.
- `layernorm_device_operation.cpp:379–390` validates bounding-box extent, not an origin-zero absolute end. `block_h=1`, `block_w=4`, `subblock_w=2` match32 physical rows and128 columns/core.
- `sharded_layernorm_factory_helpers.cpp:170–195` records `grid_offset`; lines452–461 apply it to sender/reduction-worker ranges.
- `layernorm_op_multi_core_sharded.cpp:284–291` applies the offset when mapping multicast NoC coordinates.

Thus norm program extent remains8×4 even though its physical origin is row4. CPU-only binding construction of memory configs and norm config passed. This does not replace a hardware test of multicast or the new AR route.

## Carried layout and unavoidable movement

Both inherited `decode_norm` calls use `self.ar_mem` and the matching norm config; residual adds consume that layout. AR returns it directly. `decode_batch_packed` asserts its output still has this layout, so the next layer's usual `to_memory_config(x,self.ar_mem)` is an identity conversion, not a restore. There is no conversion back to the original origin8×4 residual contract after either collective or layer.

Projection matmuls retain their measured DRAM-sharded weight and input/output configurations. `decode_linear` still converts norm output into each existing matmul input layout. That internal projection conversion is necessary and included in whole-stack timing; it is not an inter-layer residual restore. Existing sharded row-projection outputs can feed AR directly because its reader enumerates input shard coordinates (`all_reduce_async_program_factory.cpp:301,554–557`) independently of output reduction workers. Only unsharded inputs are converted to the new residual layout before AR.

The source supports public B1–32 unchanged. Physical rows remain32, width4096 is exactly32×128, and RMSNorm normalizes each row independently. No extra logical user or public alignment requirement is introduced. Existing ragged-prefill and cache/state handling is inherited unchanged.

## Persistent ownership and memory

The runner's `MultichipCollectives` reference is replaced process-locally before setup, so it allocates only the new workspace, once per serial stack. Both decoder layers receive that same instance; existing runner identity assertions verify shared buffer/semaphore lists. No old-grid receive buffers are constructed and retained. A standalone constructor creates the same workspace if none was passed; concurrent stacks require separate instances.

Two BF16 receive buffers total2097152 bytes/chip (1 MiB each), unchanged; one physical residual is262144 bytes. Receive shard shape32×512 contains the required four-chip intermediate volume for output32×128. `all_reduce_async_device_operation.cpp:62–71` validates buffer-grid containment and per-shard capacity. The existing full-compute-grid CCL semaphore allocation is retained, covering both alternate residual and custom worker cores. Every layer still issues two decode reductions, preserving ping-pong reuse order. No state/KV dtype, persistent size, or advertised context changes.

## Comparison jobs and fallback geometry

`doc/optimized_multichip_decoder/ar_shifted8x4_retry_jobs.json` contains five automatic-worker and five custom-worker jobs: B1/B32 short with warmed/traced timing,64K context, Watcher B32, and1024-answer long. Same HF references and unchanged .995/rank/state gates are retained. Compare shifted-auto versus shifted-custom to isolate placement; compare shifted-auto versus original-auto to expose the cost of changing residual layout. Do not attribute the entire latter difference to worker placement.

An optional `--residual-grid compact4x8` uses(0,0)..(3,7),same32 cores and shard sizes, matching4×8 norm extent. This also excludes both custom workers. `ar_compact4x8_retry_jobs.json` contains the same paired matrix as an isolated follow-up if shifted-grid behavior fails or another coherent layout comparison is needed. It is not an automatic runtime fallback: the command selects one grid and must pass its own gates. Shifted support exists in source, so the primary retry should remain shifted8×4.

Python compilation and CPU memory/norm descriptor construction passed. `validation.json` records grids, disjointness, allocation arithmetic and source hashes. Hardware compile, Watcher, trace safety, final quality, and performance remain pending root's serialized runner. No current production or original candidate file was modified.
