# AutoDebug Report: Gemma 4 31B Fused Matmul + Reduce-Scatter Hang

Scope: inspection-only diagnosis of the Stage 05 `google/gemma-4-31B` fused matmul/reduce-scatter model-shape hang recorded in:

- `models/autoports/google_gemma_4_31b/doc/optimized_multichip_decoder/candidates/fused_mmrs_model_shape_retry1.log`
- `models/autoports/google_gemma_4_31b/doc/optimized_multichip_decoder/candidates/fused_mmrs_hang_triage.txt.gz`

No implementation files were changed and no TT hardware reproduction was run during this investigation.

## Verdict

The strongest root cause is a topology/program-factory mismatch in `MatmulReduceScatterAsyncProgramFactory`: it resolves and stores `Topology::Linear`, but unconditionally builds and later overrides the **ring** reduce-scatter program. The exact repro runs fabric as `FABRIC_1D` and requests `Topology.Linear`, while all repository coverage of this fused API runs `FABRIC_1D_RING` with `Topology.Ring`.

This mismatch explains the delayed device hang:

1. The fused matmul completes enough to release the reduce-scatter readers from their matmul-ready handshake.
2. Those readers then block in ring collective semaphore waits for directional data that a line endpoint cannot produce.
3. The host sees no API validation error because the fused device op validates the matmul and scatter dimension, but does not validate that the selected collective program matches the resolved topology.

The material candidate is therefore **not rejected**. First retry the exact Gemma shape with `FABRIC_1D_RING` + `Topology.Ring`. If the stage must retain linear fabric, the fused factory needs a real line branch in both program creation and runtime-argument override, plus a two-volume persistent intermediate buffer required by the line implementation.

## Repro Lowering

The model test at `models/autoports/google_gemma_4_31b/tests/test_multichip_decoder.py:954-980` calls the shared helper with:

- TP4 mesh, `M=32`, global `K=N=5376`;
- local input `[1, 1, 32, 1344]` and local weight `[1, 1, 1344, 5376]`;
- BF16 activation, BFP8_B weight, BF16 output;
- two links, `Topology.Linear`, DRAM interleaved buffers;
- 2D multicast matmul grid `(8, 6)` and reduce-scatter offset `(0, 6)`;
- `in0_block_w=3`, `per_core_M=1`, `per_core_N=21`, and adapted legal `out_block_w=7`.

The 168 output tiles split into eight 21-tile blocks, so only the first matmul row (`x=0..7, y=0`) has output work. The reduce-scatter workers begin at logical `y=6`; triage reports 44 collective cores in `y=6..9`. The two grids do not overlap.

The Python API resolves the requested topology with `get_usable_topology` and passes it into `MatmulReduceScatterAsyncDeviceOperation` (`matmul_reduce_scatter_async.cpp:36-56`). On this full 1x4 tensor mesh, explicitly requested ring can remain ring; the current request remains linear.

## Headline Finding

### H1. The fused factory always selects the ring collective program, even for resolved linear topology

Confidence: high; confirmed source discrepancy that directly matches the kernel names and wait state in triage.

The fused factory reads the resolved topology at `matmul_reduce_scatter_async_program_factory.cpp:45`, and uses it to compute optional forward/backward neighbors at lines 62-70. However, at lines 80-103 it unconditionally calls:

```cpp
build_ring_reduce_scatter_minimal_async_program_artifacts(..., topology, ...)
```

Its runtime-argument override is likewise unconditionally ring-specific at `matmul_reduce_scatter_async_program_factory.cpp:150-169`.

That differs from the standalone minimal reduce-scatter operation. `ReduceScatterMinimalAsyncDeviceOperation::select_program_factory` explicitly returns `RingReduceScatterMeshWorkloadFactory` for ring and `LineReduceScatterMeshWorkloadFactory` for linear (`reduce_scatter_minimal_async_op_device_operation.cpp:15-23`). The two builders implement different algorithms:

- ring builder: `reduce_scatter_minimal_async_program.cpp:338-910`;
- line builder: `reduce_scatter_minimal_async_program.cpp:981-1555`.

The distinction is semantic, not naming only. The ring builder requests `ring_size - 1` multicast distance in both directions and launches ring reader/writer kernels. The line builder computes endpoint-specific `num_targets_forward` and `num_targets_backward`, detects first/last chips, and adjusts reduction steps for each direction (`reduce_scatter_minimal_async_program.cpp:1028-1067, 1422-1436`).

The captured execution proves that the mismatched lower is active:

- The test fixture initializes `FabricConfig::FABRIC_1D` (`test_multichip_decoder.py:325-330`) and the candidate requests `Topology.Linear` (`test_multichip_decoder.py:975`).
- The only fused API regression parametrization uses `FabricConfig::FABRIC_1D_RING` and `Topology.Ring` on 1x8 (`tests/ttnn/unit_tests/operations/ccl/test_new_matmul_reduce_scatter.py:380-427`). There is no fused linear case or exact TP4 Blackhole model-shape case.
- Triage shows `MatmulReduceScatterAsyncDeviceOperation` still `RUNNING` on all four devices and all listed collective kernels are `ring_reduce_scatter_minimal_async_reader`, `ring_reduce_scatter_minimal_async_writer`, and `ring_reduction` (decompressed `fused_mmrs_hang_triage.txt.gz`, lines 192-201 and 280-523).
- Many readers are stopped at `ring_reduce_scatter_minimal_async_reader.cpp:215/219`, waiting for collective direction semaphores. Those lines occur after the fused matmul-ready wait at reader line 81. Thus these workers received the matmul notification and entered the collective protocol; the visible hang is not simply an unstarted matmul.
- Endpoint/device states differ: some workers have exited or reached final writer synchronization while others wait for direction data. That is the expected failure shape when a ring protocol is launched with missing wraparound neighbors, not a uniform compute-kernel stall.

The first valid intervention boundary is topology/program selection. Changing matmul geometry, dtype, or output block width does not repair this protocol mismatch.

## Secondary Confirmed Contract Gaps

These are real discrepancies that must be handled for a true linear fused implementation, but they are not the immediate cause of this run because the current fused factory never enters the line builder.

### S1. The shared Python helper underallocates the persistent intermediate buffer for linear reduce-scatter

The standalone reduce-scatter spec doubles the intermediate tensor's leading extent for `Topology.Linear` (`reduce_scatter_minimal_async_op_device_operation.cpp:64-89`). For the fused matmul output `[1, 1, 32, 5376]`, the line intermediate must have the capacity represented by `[2, 1, 32, 5376]`.

The shared test helper always allocates `single_batch_input_shape == [1, 1, 32, 5376]` (`test_new_matmul_reduce_scatter.py:72-87`), because that helper was written and covered only for ring. Merely changing the fused factory from `build_ring...` to `build_line...` without doubling this persistent buffer would expose a second invalid contract and is not a complete linear retry.

### S2. Fused output-spec derivation uses the matmul input and returns the intermediate spec

`MatmulReduceScatterAsyncDeviceOperation::compute_output_specs` correctly computes `matmul_output_specs`, but then constructs reduce-scatter inputs from `tensor_args.input` rather than the matmul output and returns `reduce_scatter_output_specs[0]` (`matmul_reduce_scatter_async_device_operation.cpp:52-69`). In the standalone operation, element 0 is the intermediate spec and element 1 is the reduce-scattered output spec.

The present call returns the caller-supplied persistent output directly (`matmul_reduce_scatter_async_device_operation.cpp:72-80`), so this spec bug does not explain the semaphore hang. It is nevertheless a validation/test gap and should be corrected or explicitly validated before making linear fused MMRS production code.

## Hypotheses Ruled Down

### Core-grid overlap: unlikely

The configured matmul grid occupies `y=0..5`, with actual output work only on `y=0`. The reduce-scatter offset is `(0, 6)`. Triage lists the 44 collective cores in `y=6..9`; it does not show an overlap with matmul workers.

### Two links / worker capacity: unlikely as primary cause

The operation successfully creates and launches all 44 collective workers. The full 11x4 region above the offset exactly contains them. No core-range, L1-capacity, or program-creation assertion is reported. A one-link retry is useful only after topology is made coherent.

### Subdevice setup: unlikely

The helper creates a single subdevice spanning the entire compute grid, loads it, places it in the stall group, and creates all three global semaphores on that range (`test_new_matmul_reduce_scatter.py:52-70`). The op launches on all devices and the worker kernels run. There is no evidence of a missing subdevice or dispatch boundary.

### First invalid `out_block_w`: ruled out for this retry

The original value 10 was changed to legal 7. For this shape, `per_core_N=21`, so 7 divides the output block geometry. Retry1 gets past host validation and program creation on every device. The hang is later at collective synchronization.

### Generic Blackhole numeric race: no supporting evidence

This failure is a deterministic synchronization hang with explicit ring readers waiting on semaphores, not a PCC drift or nondeterministic output. Triage reports ARC heartbeats, clean CB inactivity and NoC location/status checks, and the operation still resident. Firmware 19.9.0 being newer than the fully tested 19.5.0 is worth recording but is weaker than the concrete topology mismatch. There is no Blackhole TP4/linear fused-MMRS regression in the repository to establish support.

## Exact Next Retries

### Retry A: coherent ring candidate (smallest bounded retry)

Keep the exact Gemma shape, dtypes, `out_block_w=7`, two links, DRAM buffers, and `(0, 6)` offset, but open the mesh with:

```python
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
```

and pass:

```python
rs_topology=ttnn.Topology.Ring
```

Run one non-traced iteration first. This matches the only covered fused program family while preserving the model shape. If it completes, check both matmul and reduce-scatter PCC before any timing run. This is the most likely usable Stage 05 candidate.

### Retry B: coherent ring with one link

Only if Retry A still hangs, repeat the same coherent ring setup with `num_links=1`. This separates a two-link worker/fabric issue from the topology bug. Capture triage again if it hangs; do not use a first failure as a rejection.

### Retry C: true linear fused support

If the final stage must use `FABRIC_1D`/Linear, make one coherent implementation change:

1. Branch program creation on resolved topology: ring calls `build_ring...`; linear calls `build_line...`.
2. Branch runtime-argument override in the same way.
3. Allocate/validate the line persistent intermediate at twice the matmul-output volume (`[2, 1, 32, 5376]` for this repro).
4. Correct fused reduce-scatter spec derivation to use the matmul output and select the actual output spec.
5. Add a focused TP4 linear regression that compares fused and non-fused PCC before integrating into the decoder.

The line implementation already accepts a fused-op signaler, so this is a bounded adaptation rather than a new algorithm. It requires a source fix and rebuild; simply toggling `Topology.Linear` is exactly the broken path captured here.

## Review of Claims

The headline claim accounts for every important observation:

- why a host-side API/shape assertion does not fire;
- why ring-named kernels execute despite a linear request;
- why the matmul-ready boundary is passed but collective direction semaphores remain unsatisfied;
- why all existing fused coverage can pass while this case hangs (coverage is ring-only);
- why grid/dtype/block-width changes are not causal interventions.

The doubled line intermediate and output-spec findings were deliberately kept secondary because the failing execution uses the ring builder. They become mandatory only after correcting program selection for true linear support.
