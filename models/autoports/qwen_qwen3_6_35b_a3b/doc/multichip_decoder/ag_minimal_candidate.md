# Adapted Linear-fabric AG + minimal matmul candidate

CPU inspection only; this code has not been run on devices. It leaves FABRIC_1D / `Topology.Linear` unchanged. The earlier `all_gather_matmul_async` proposal is blocked on this topology by `all_gather_async_default_program_factory.cpp:220`, which explicitly rejects `Linear && fuse_op`. That restriction does **not** apply to the separate strided implementation below.

## Evidence that the adapted path implements Linear

- `strided_all_gather_minimal_matmul_async_program.cpp:55–79` derives neighbors with the requested topology and passes that topology to `MinimalMatmulFusedOpSignaler`.
- Its separate `strided_all_gather_async_minimal_default_helper` receives the topology, axis and explicit core offset; it does not call the rejecting default AG helper.
- `strided_all_gather_async/device/kernels/minimal_default_reader.cpp:88–95` explicitly calculates forward/backward slice and forwarding counts for `Topology::Linear`.
- `minimal_default_writer.cpp:237–242` has corresponding Linear forwarding counts. Endpoints suppress nonexistent neighbors.
- `minimal_default_mm_signal_aggregator.cpp:56` also has a Linear branch, although the initial candidate disables aggregators to reduce resources and semaphore complexity.
- Existing Blackhole strided-AG/MM test cases inspected use Ring, so source support is **not** claimed as existing runtime proof for Linear on this machine.

## Pasteable candidate

The public input remains BF16 `[1,1,1,512]`; local BFP8 weights are `[1,1,2048,2560]`. The conservative first probe internally pads the pure projection to64 logical rows and returns only row0. This satisfies the minimum2-row compute grid without relying on an empty second M partition. No padded row may be sent to KV-cache updates. After a correct run, measure a32-row variant as a separate geometry candidate;64 is not an asserted minimum supported M.

`MinimalMatmulConfig` block sizes are in **tiles**, not elements. This candidate uses grid8x2, one M tile/core, K block8 tiles=256 elements (two blocks per device's512-wide slice), N block10 tiles=320 elements (eight cores cover80 output tiles), subblock1x2. All compute cores occupy rows0–1; AG workers begin at row2. With one link and one worker per direction, AG needs four cores; aggregators are disabled. FP32 accumulation is explicitly enabled; output inherits BF16 from the input because this fused API has no `dtype=` parameter.

```python
import os
import torch
import ttnn


class LinearAGMinimalQKVG:
    """Setup once, warm before capture, retain until every using trace is released."""

    def __init__(self, mesh, weight, *, physical_rows=64, k_tiles=8,
                 num_links=1, fp32_acc=True):
        assert tuple(mesh.shape) == (1, 4)
        assert tuple(weight.shape) == (1, 1, 2048, 2560)
        assert weight.dtype == ttnn.bfloat8_b
        assert weight.layout == ttnn.TILE_LAYOUT
        assert physical_rows in (32, 64)
        assert k_tiles in (1, 2, 4, 8, 16)
        assert num_links in (1, 2)
        # The reader-signaled path emits one signal per chunk, so it cannot
        # satisfy an environment request for multiple M bands per chunk.
        assert os.environ.get("IN0_SUB_CHUNKS", "1") == "1"
        assert os.environ.get("TT_METAL_SLOW_DISPATCH_MODE") is None
        grid = mesh.compute_with_storage_grid_size()
        assert grid.x >= 8 and grid.y >= 3
        self.mesh = mesh
        self.rows = physical_rows
        self.links = num_links
        self.weight = ttnn.to_memory_config(weight, ttnn.DRAM_MEMORY_CONFIG)
        self.config = ttnn.MinimalMatmulConfig(
            M_block_size=1,
            K_block_size=k_tiles,
            N_block_size=10,
            subblock_h=1,
            subblock_w=2,
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 2),
        )
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_acc,
            packer_l1_acc=True,
        )
        all_cores = ttnn.CoreRangeSet({ttnn.CoreRange(
            ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1)
        )})
        # Off mode requires two out-ready semaphores, one per direction.
        # Do not pass a barrier_semaphore: this API has no such keyword.
        self.semaphores = [ttnn.create_global_semaphore(mesh, all_cores, 0)
                           for _ in range(2)]
        self.gather_buffer = ttnn.from_torch(
            torch.zeros(1, 1, physical_rows, 2048, dtype=torch.bfloat16),
            device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    def __call__(self, norm_local):
        assert tuple(norm_local.shape) == (1, 1, 1, 512)
        assert norm_local.dtype == ttnn.bfloat16
        assert norm_local.layout == ttnn.TILE_LAYOUT
        x = ttnn.to_memory_config(norm_local, ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.pad(x, [(0, 0), (0, 0), (0, self.rows - 1), (0, 0)],
                     0.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gathered, projected = (
            ttnn.experimental.strided_all_gather_minimal_matmul_async(
                x, self.weight,
                persistent_output_buffer=self.gather_buffer,
                dim=3,
                multi_device_global_semaphore=self.semaphores,
                strided_all_gather_core_grid_offset=ttnn.CoreCoord(0, 2),
                num_links=self.links,
                memory_config_ag=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Linear,
                cluster_axis=1,
                config=self.config,
                memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute,
                num_workers_per_link=1,
                num_buffers_per_channel=1,
                read_local_slice_from_input=False,
                chunks=1,
                mm_signal_aggregator_mode=ttnn.MMSignalAggregatorMode.Off,
            )
        )
        # The returned gathered tensor is useful for comparing to the unfused
        # projection and for branches that need the same normalized full input.
        return gathered[:, :, :1, :], projected[:, :, :1, :]
```

Setup on the current decoder: reshape its already loaded local QKVG weight to `[1,1,2048,2560]`, instantiate the candidate once, and route the **distributed norm's local result** into it. Remove the standalone activation AG for this candidate. Return the projected logical row to the existing Q/K/V/gate split. The input and output residual contract stays hidden-sharded; do not gather the residual simply to exercise this API.

The first isolated control should compare the fused output to `ttnn.linear(gathered_row, existing_qkvg_weight, dtype=ttnn.bfloat16, compute_kernel_config=attention_compute)`, then compare the complete decoder to the single-chip optimized baseline. Readback/checks are outside measured execution.

## Resources, precision and follow-up measurements

- The persistent gather buffer is256KiB/chip at64 rows,128KiB at32 rows. Weight is interleaved BFP8; no DRAM-sharded matmul configuration is passed. No new weight packing is required beyond rank4 view.
- The compute factory uses8x2 cores. MinimalMatmul validation explicitly requires both grid axes >=2 on Blackhole (`minimal_matmul_device_operation.cpp:376–386`); an8x1 grid is not a valid adaptation. The conservative64-row pad gives both M partitions an actual row tile.
- One link/worker with aggregators Off uses `2 * (1 + 1)=4` mux/worker cores after the offset. Two links uses8, still fitting row2 when grid.x>=8. These resources are separate from the16 MM cores.
- A later aggregator-On candidate needs two additional aggregator cores **and** `2 + 2*num_links*num_workers_per_link` global semaphores. Allocate sufficient rows from the offset rather than borrowing active MM cores. Do not silently use Auto and misattribute a fallback to the intended mode.
- `fp32_acc=True` selects Float32 intermediate storage; `False` selects Float16_b in `minimal_matmul_fabric_bound_program_factory.cpp:233–238`. The kernel explicitly enables L1 pack accumulation across K blocks (`fabric_bound_compute.cpp`, after the first K block). A caller flag alone does not establish the effective pack behavior.
- Keep HiFi2/FP32 accumulation for the correctness/control candidate. BF16 accumulation (`fp32_acc=False`) is a separate performance/accuracy candidate. Compare real model inputs/weights through the whole decoder; do not reject it solely from unrelated synthetic-weight failures. This API still outputs BF16 in either case.
- The minimal kernel's final intermediate-to-output copy and topology-dependent K order can differ from the original projection's rounding. This is a PCC candidate, not a bit-equality claim. All chips must agree on the subsequent router inputs under the intended residual reduction.
- Warm/capture the pad, fused operation and final slices together. Keep gather buffer and semaphore backing allocations live for all captured traces. The strided reader resets its out-ready semaphore at the end (`minimal_default_reader.cpp:236`), but repeated replay and changing input evidence are still required. Do not overlap two invocations that use the same gather buffer/semaphore set; allocate separate slots when concurrent lifetimes demand them.
- After the first pass, compare32 versus64 internal rows, K blocks4/8/16, links1/2, and aggregatorOff/On where needed. Include padding and slicing in decoder timing. Persistent resources and precision must stay matched between variants.

No Linear topology prohibition was found in this adapted family. The exact unresolved question is device correctness/performance for the proposed1x4 Linear signature; it must be measured under the existing fabric configuration.
