# AutoDebug: undersized persistent Linear reduce-scatter scratch

**Proven defect; leading diagnosis:** `_collective` allocates half the scratch required by the selected Linear reduce-scatter kernel. Backward traffic indexes beyond the supplied tensor. Fix this allocation first. Inspection does **not** identify the overwritten allocation or prove that this repair alone restores trace PCC.

Investigation was inspection-only: source comparison, artifact hashes and host-only page arithmetic. No device code, launcher, resets or source edits were performed. The snapshot, log and command manifest were preserved.

**Evidence and revision.** The [command manifest](/home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/family3_persistent_l0_b32_s33.command.json) selects layer 0, B32, prefill S33, `--trace`, and `{"flat_finish":true,"flat_prefill_correction":true,"persistent_ccl":true}`. The [snapshot](/home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/debug_persistent/decoder_snapshot.py:339) hashes to the manifest's `f89fb449…`; the current runner and `optimized_decoder.py` also match their manifest hashes. Current decoder additions are unrelated guarded flat-prefill/recurrence hooks; the persistent branch is unchanged.

The [log](/home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/family3_persistent_l0_b32_s33.log) reaches trace after the initial eager baseline checks, then fails the first checked replay (`original`, repeat 0): output PCC **0.637431591297329**, conv **0.799329237755364**, recurrent **0.9666407531562667**. These values identify detection time, not the first corrupting execution. Four later eager trace-case decodes save unchecked snapshots; two final eager decodes are also unchecked ([runner](/home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/tests/run_multichip_decoder.py:307)). Including the initial decode, seven eager decodes execute fourteen undersized RS calls before capture.

**Caller → dispatch → kernel.** Decode processes one token for 32 users, so the collective input is `[1,1,32,W]`. S33 is prefill history; its flattened height 1056 excludes persistent mode. The decoder explicitly chooses Linear topology, `dim=3`, `cluster_axis=1`, BF16 and interleaved L1 input/output. Width `5120/4=1280` is tile-aligned, so the wrapper takes native Linear RS, not composite or Ring staging.

| Decode order | Native input | Persistent allocation reused |
|---|---|---|
| Input norm AG | `[1,1,32,1280]` | L1 output `[1,1,32,5120]` |
| Attention RS | `[1,1,32,5120]` | DRAM scratch; L1 output `[1,1,32,1280]` |
| Post-attention norm AG | Same AG shape | Same AG output |
| MLP RS | Same RS shape | Same RS pair |

The contract contradiction is direct:

- [Snapshot:367](/home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/debug_persistent/decoder_snapshot.py:367) creates RS scratch with `empty_like(x, memory_config=DRAM)`: **`[1,1,32,5120]`**. The wrapper interprets buffer 0 as the intermediate.
- Native [output-spec construction](/home/ttuser/qwen36-27b/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op_device_operation.cpp:222) starts from the **padded** input shape and doubles dimension zero for Linear: **`[2,1,32,5120]`**. [Creation](/home/ttuser/qwen36-27b/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op_device_operation.cpp:256) returns supplied tensors without resizing them.
- [Validation](/home/ttuser/qwen36-27b/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_common/reduce_scatter_validate_utils.cpp:111) checks format/memory properties but omits Linear shape/capacity. Cache-hit validation checks the intermediate's mesh only. Thus the wrong buffer is accepted.
- The selected [Linear writer](/home/ttuser/qwen36-27b/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/line_reduce_scatter_minimal_async_writer.cpp:284) adds `input_num_pages` for backward traffic, then performs remote writes at those page IDs (323–358). The [reader](/home/ttuser/qwen36-27b/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/line_reduce_scatter_minimal_async_reader.cpp:155) documents and uses that second half.

Concrete bounds: `Ht=1`, `Wt=160`, `slice_Wt=40`, input pages `P=160`, BF16 page size 2048 bytes. Supplied payload is **327,680 bytes**; native scratch requires **655,360 bytes**. Across worker partitions, backward rank 1→0 writes pages **160–199**, rank 2→1 **160–239**, rank 3→2 **160–279**. The supplied tensor ends at page 159. Dynamic allocation supplies the missing half with the same kernel indexing. The [Blackhole persistent setup](/home/ttuser/qwen36-27b/tt-metal/tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/nightly/test_minimal_reduce_scatter_async_bh.py:84) independently allocates twice the volume. An existing [Wormhole regression](/home/ttuser/qwen36-27b/tt-metal/tests/nightly/t3000/ccl/test_minimal_reduce_scatter_async.py:1537) explicitly guards a neighboring sentinel against this class of overflow; it is corroboration, not a reproduction here.

**Why eager success does not clear this bug.** Out-of-range sender/receiver addresses can still exchange correct RS data while damaging other storage. Allocation placement and lifetime determine when that damage becomes visible. Conv/recurrent updates precede both RS calls, so later writes can corrupt state, saved restore snapshots or other operands. The artifacts contain no address mapping that identifies a victim. The first divergence could precede capture; claiming a proven trace-only race would exceed the evidence.

**Other potential issues, adjudicated.**

- **Shape-only cache/disabled barriers:** both roles really share each buffer pair. However, AG output is copied to DRAM before normalization ([norm](/home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/tt/optimized_decoder.py:623)); each RS result feeds a fresh residual add. For in-bounds collective execution, the sequence `AG0 → GDN → RS0 → add → AG1 → MLP → RS1` provides cross-rank dependencies that appear sufficient for these particular lifetimes: reaching AG1 requires every rank's GDN contribution after AG0 consumption; reaching RS1 requires every rank's AG1 publication after RS0 completion. Persistent mode suppresses startup barriers, but no independent race is established in this graph. This is not a general barrier guarantee; one CQ alone would not prove safety.
- **Program-cache overrides:** [the adapter](/home/ttuser/qwen36-27b/tt-metal/ttnn/api/ttnn/mesh_device_operation_adapter.hpp:703) refreshes workload tensor bindings before factory overrides. [AG](/home/ttuser/qwen36-27b/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_default_program_factory.cpp:695) declares input/output bindings; its override refreshes semaphores and height. [RS](/home/ttuser/qwen36-27b/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_program.cpp:493) explicitly refreshes every tensor-role address and semaphore. [Trace recording](/home/ttuser/qwen36-27b/tt-metal/tt_metal/impl/program/dispatch.cpp:3244) copies per-launch runtime arguments; later cache hits do not overwrite earlier captured bindings. No concrete cache-rebinding defect was found.
- **API discrepancy:** [public documentation](/home/ttuser/qwen36-27b/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/reduce_scatter_minimal_async_nanobind.cpp:115) incorrectly says Linear scratch is input-shaped. The allocation helper is Ring-only. `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE` exempts allocation accounting; it does not disable program caching. Tracking was disabled in this run.

**Minimal repair and verification boundary.** Replace only the persistent Linear scratch allocation with the native padded-shape rule; retain cache keys, output placement and barrier behavior for the first controlled test:

```python
intermediate_shape = list(x.padded_shape)
intermediate_shape[0] *= 2  # This decoder explicitly uses Linear topology.
intermediate = ttnn.empty(
    intermediate_shape, dtype=x.dtype, layout=x.layout,
    device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

Also validate Linear scratch capacity and addressing compatibility on cache misses and hits, and correct the misleading documentation. Require the native 2× padded capacity without unnecessarily rejecting equivalent interleaved shapes used by existing tests. Splitting buffers by role first would change neighboring allocations and could mask the overflow.

Follow-up requiring hardware: rerun the exact command after the sizing-only correction, compare eager warmup cases against baseline before capture, then check both replay rounds. The runner synchronizes before capture and uses blocking correctness replays. These completion boundaries cannot repair previously corrupted buffers or snapshots. If failure remains, test correctly sized persistent RS and persistent AG separately, then examine synchronization. **The undersizing is proven; the exact corruption victim and sufficiency of the repair remain unverified.**
