# AutoDebug: TP4 DRAM-sharded decode dispatch

Inspection-only diagnosis, 2026-09-11. No implementation changes, hardware
commands, installations, or performance measurements were made by this agent.
The parent supplied a fresh isolated context as required by AutoFix.

## Headline finding

The inherited three-reader configuration reaches a helper that explicitly
rejects multi-device meshes. One reader per bank avoids that helper and is the
smallest model-local control. Two readers do not avoid it. The standard tensor
shard/view APIs do not turn mesh tensors into tensors owned by unit meshes, so
iterating `get_device_tensors()` cannot preserve the three-reader path.

## Observations and complete causal chain

- Original command: `bash models/autoports/qwen_qwen3_8_27b/tests/run_multichip_experiment.sh replicated_fixed_l0 --layer 0`.
- `replicated_fixed_l0.log` records `PREFILL_DONE`, then an exception on the
  first eager decode, before trace capture. The stack names
  `linear_attn.packed`, `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory`,
  `get_dram_bank_reader_assignments`, and `get_worker_noc_hop_distance`.
- `multichip_decoder.py:31-35` inherits `DEFAULT_POLICY`; the five projection
  roles have three readers (`optimized_decoder.py:22-69`). Its `_linear`
  delegates to the frozen baseline (`multichip_decoder.py:135-136`).
- With sequence length 1, the inherited `_linear` selects DRAM sharding,
  constructs `num_workers_per_dram_bank=3`, and invokes `ttnn.linear`
  (`optimized_decoder.py:320-351`). With the repro's prefill length 128, this
  branch is not selected: the short-prefill threshold is 32. This explains
  why prefill can complete before decode fails.
- The DRAM factory gets `IDevice* device = a.device()` at
  `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:979`
  and passes it into reader assignment at lines 123-124. `Tensor::device()`
  returns the owning MeshBuffer's mesh (`ttnn/core/tensor/tensor.cpp:507-511`),
  which here is the 1x4 mesh.
- Reader assignment returns early for one reader
  (`ttnn/cpp/ttnn/operations/matmul/device/utilities/matmul_utilities.cpp:415-419`).
  Otherwise it searches secondary worker locations and calls the
  coordinate-free hop helper at lines 433-447.
- That helper rejects `mesh->num_devices() != 1`
  (`tt_metal/impl/device/experimental/device.cpp:17-21`). A coordinate-aware
  overload exists at lines 44-52, but this factory does not call it.

This is a host-side descriptor-construction incompatibility. The observed
exception does not establish a CCL, NoC kernel, recurrent-state, numerical,
or trace bug. It also does not establish that later decode operations pass.

## Concrete first-projection parameters

Layer 0's TP-local packed weight is `[5120,4160]`: Q512, K512, V1536,
Z1536, B32, A32. B/A each have 12 real heads padded to 32. B1 decode becomes
logical `[1,1,1,5120]`, with 32 physical rows, BF16 input/output, BFP4 weights,
LoFi, FP32 destination accumulation. Activation storage uses 80 cores with
`[32,64]` shards, `in0_block_w=2`, `per_core_M=1`.

The setup formula in `multichip_decoder.py:115-125` makes each of eight DRAM
bank shards `[5120,576]` with three readers, versus `[5120,544]` with one.
The respective widths are 18 and 17 tiles; both produce storage
`per_core_N=2` on 80 cores. Change reader policy at construction so weights
and program configuration agree. No dtype or fidelity change is needed.

## Focused verify/refute experiments

1. **One-reader control: supported branch avoids the observed assertion.**
   Exact command, launched by the parent after its device recovery:

   ```bash
   bash models/autoports/qwen_qwen3_8_27b/tests/run_multichip_experiment.sh reader1_l0 --layer 0 --policy '{"attention_readers":1,"output_readers":1,"gate_readers":1,"up_readers":1,"down_readers":1}'
   ```

   Artifacts: `reader1_l0.log` and, on success, `reader1_l0.json` in this
   directory. The parent reports this experiment passed: prefill PCC
   0.9999953, decode PCC 0.9999997, recurrent-state PCC 0.999957, convolution
   PCC 1; changed-input replay is bitwise equal. Measured traced decode is
   0.7023 ms versus the parent's 0.8268 ms single-chip baseline. These runtime
   results were supplied by the parent, not measured by this investigation.
   The assertion disappearing verifies the predicted branch-level mechanism.
   The parent is repairing an allocation warning in the harness by preparing
   refresh buffers/eager controls before capture. Reverify that final harness,
   then extend to a full-attention layer and broader state/cache coverage.

2. **Native interleaved control: retain precision while changing matmul family.**
   Existing policy `--policy '{"dram":false}'` bypasses DRAM-sharded decode
   and uses the interleaved weights already retained for prefill
   (`optimized_decoder.py:395-401`). For the original S128 prefill this does
   not change its projection branch. Compare exact output and traced latency
   against reader1. It provides a model-only alternative if reader1 performs
   poorly; no performance advantage is established by source inspection.

3. **Minimal matmul: another native candidate, requiring measurement.**
   Existing policy `--policy '{"minimal_prefill_min":1}'` selects `_minimal`
   before the DRAM branch, including decode (`optimized_decoder.py:301-309`).
   `_minimal` uses interleaved weights, BF16 output, unchanged compute
   configuration and `M_block_size=1` for decode (lines 404-428). This broad
   control also changes some prefill projections. If promising, use a
   decode-only selection in `multichip_decoder.py` and retain the existing
   output/down collectives; prove each change separately. Score warmed trace
   latency for the full layer, including layout conversions and collectives.

4. **Mesh-local three-reader dispatch: simple view workaround is refuted.**
   `get_device_tensors()` constructs coordinate-restricted `DeviceStorage`
   views (`ttnn/core/distributed/api.cpp:77-84`). The constructor retains the
   same MeshTensor holder (`ttnn/core/tensor/storage.cpp:134-136`), hence
   `view.device()` remains the parent four-device mesh and the same assertion
   follows. A metadata probe of `view.device().get_num_devices()` can confirm
   this, but there is no reason to run the failing matmul merely to rediscover
   it. `ttnn.to_device(view, unit_submesh)` also cannot rebind the buffer:
   `ttnn/core/tensor/tensor_ops.cpp:137-140` rejects moving device tensors
   between different MeshDevice objects. `combine_device_tensors()` requires
   the exact same MeshTensor allocation (`storage.cpp:242-250`), so separately
   allocated unit-mesh outputs cannot simply be assembled into the parent
   tensor for CCL. No supported Python-only zero-copy rebind route was found.

## Scope boundary and remaining uncertainty

There is no demonstrated blocker to a model-local solution: the parent's
reader1 layer-0 run passed, and interleaved native matmuls offer further
source-supported candidates. Broader coverage and the fastest supported
policy remain runtime questions.

Preserving **this exact multi-reader DRAM factory on the existing parent-mesh
tensors** is blocked in native C++: its config exposes only block sizes,
activation, and reader count, not reader assignments or a mesh coordinate
(`matmul_program_config_types.hpp:75-81`). Python monkey-patching the exposed
hop helper cannot intercept the factory's compiled call. The proper native
repair would need coordinate-aware assignment/descriptor handling or a
validated representative-device policy; neither is authorized in this stage.

Creating independent unit meshes is not an established drop-in solution:
besides tensor ownership/reassembly above, trace capture is owned by each
mesh's command queue (`tt_metal/distributed/mesh_device.cpp:1393-1415`). It
would require separate evidence for parent CCL ordering and capture/replay,
not just successful eager local matmuls. Host tensor round trips would violate
the runtime contract and do not qualify as a workaround.

The batched DRAM program is also not a config substitution: its validator
requires height-sharded A/B/output and batch-owned full matrices, whereas
these projections use width sharding
(`matmul_device_operation.cpp:1375-1423`). Prefer measuring reader1,
interleaved, and minimal native paths before redesigning projection ownership.
