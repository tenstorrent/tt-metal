# Direct native slice of sharded router outputs

Source/CPU-only control on compact serial baseline `v4_serial_l1_compact.py.snapshot`, SHA256 `deb1fa8f3fb3020bfb0f983cc04c43e45f96f245535e0e8e7b5741fe417cf4ad`. Candidate `v4_router_direct_slice.py.snapshot` has SHA256 `ec354a71ce36a576ff147354d3b0b6f63b64398acfe6201a7b88ad46fc1a2a4c`. Production and prior probes are unchanged. This candidate is independent of the serial-prefill RM placement controls.

The only changes are the two native gate-output adapters. Replace each full `to_memory_config(gate_output,DRAM)` followed by `[:,0,:8]` with:

```python
ttnn.slice(gate_output, (0,0,0), (1,1,8), memory_config=DRAM)
```

The existing score reshape/tilize to `(1,8)` and subsequent TILE routing reshape to `(1,8,1,1)` remain. Ids still become RM UINT16 `(1,1,1,8)`. The base already elides identity casts; those stay elided. This does not adopt the previously tested delayed score-tilization geometry. It removes two full sharded-to-interleaved transfers if native direct slicing lowers as the source specifies; both necessary slices remain.

## Native contracts

Actual native router outputs are logical `(1,32,32)` HEIGHT-sharded RM, BF16 scores and UINT16 ids. The selected eight elements are the first eight columns of row0. Their direct-slice result `(1,1,8)` has the same element order as the old integer-index result `(1,8)`; the retained explicit reshape resolves the rank difference.

`data_movement/slice/slice.cpp:50–67` routes an RM HEIGHT-sharded input through a composite only when its input H/W are not tile multiples and it cannot stay on the sharded path. Here both are32, so this fallback is not needed. The explicit DRAM output avoids inherited sharding. `device/slice_device_operation.cpp:436–455` selects generic `SliceRmProgramFactory` for RM input and interleaved output, while its specialized sharded factory requires sharded output as well. `device/slice_program_factory_rm.cpp` binds the input tensor metadata and handles physical page geometry. The reader `device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp:41,107–109` uses `TensorAccessor` and `noc_async_read_sharded`; its HEIGHT-sharded case performs a direct NoC read. No native restriction requires an intermediate interleaved input for this aligned case.

These are current-source contracts. Installed-binary execution and actual lowering still require the parent-owned probe; source legality alone is not device evidence or a speedup.

## Dedicated probe and fixture provenance

`tests/v4_router_direct_slice_probe.py` extracts the actual four adapter assignments from each snapshot and records their AST hashes. It checks that `_moe` is the only changed method. Both adapters receive outputs of the real native generalized gate with the production sharded shape/dtypes. Fixture cases are:

- Checkpoint-derived CPU BF16 logits for seeded hidden rows42/154/913, matching the previous router adapter controls. These are seeded checkpoint fixtures, not captured model router inputs.
- Actual historical TT router logits captured at seed154 position5991 in `routing_seed154.json.pos5991.pt`, separately hashed and labeled. This tests a real numerically sensitive routing case without claiming the historical failing decoder itself is a correct baseline.

Every final routing score and expert id must match bitwise with the baseline, with unchanged shape/dtype/layout and DRAM placement. Native gate outputs must remain immutable. Warmed eager and traced timings include both full post-gate adapters, every slice, reshape, tilize and memory placement. The common native gate is outside adapter timing; full decoder timing remains necessary. First/last trace outputs and eager outputs are rechecked. Source/probe/dependency/capture hashes are recorded, existing outputs cannot be overwritten, physical chip0 is the only opened device, and closure is in `finally`.

CPU preparation passes all four fixture/AST cases in `v4_router_direct_slice_cpu.json`. No hardware was executed by the preparer. Hardware-owner command:

```bash
HF_HOME=/home/ttuser/qwen36-35b-a3b/hf python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v4_router_direct_slice_probe
```

If exact and faster, benchmark whole linear and full-attention decoders because both use this decode router. Retention still requires the unchanged correctness and performance gates. There is no production-selection or Stop Rule claim.

## Parent-run device result

[v4_router_direct_slice_probe.json](v4_router_direct_slice_probe.json) has SHA256 `c9e421051b7ac857bdb2aa6a93e54f1c8e62b621de321417fd72c90bcff78e1e`. All four cases pass exact output, native-input immutability, layout/dtype and first/last replay checks; the device closed successfully. Complete traced adapter results are:

| Case | Baseline microseconds | Direct slice microseconds |
|---|---:|---:|
| checkpoint_seed42 | 14.475 | 10.632 |
| checkpoint_seed154 | 14.464 | 10.599 |
| checkpoint_seed913 | 14.477 | 10.596 |
| captured_seed154_pos5991 | 14.470 | 10.637 |

The adapter win includes every post-gate transformation. Whole-decoder comparisons for both layer kinds remain required; this isolated result does not select production.
