# Read-only audit: native slice's already-RM layout warning

Audited frozen production `tt/fused_decoder.py`, SHA256 `b1633d51621b4c45d9514a66fad21faa57ad562185502342c3d87bd5ebc2701c`. No source, harness, or hardware changes were made.

**The warning does not indicate incorrect final placement.** The two decode router slices request DRAM output from already-row-major, height-sharded L1 gate tensors. Native slice's internal layout-preparation call ignores the placement request because its input is already RM, but the following slice primitive separately receives and honors DRAM output placement. It would be incorrect to claim that this intermediate input was already DRAM: it is deliberately L1. The completed adapter outputs are DRAM, as established by source and a prior device probe with explicit memory-config assertions.

## Exact likely call chain

Current `_moe` lines693–694 call:

```python
ttnn.slice(gate_scores, (0, 0, 0), (1, 1, 8), memory_config=DRAM)
ttnn.slice(gate_ids,    (0, 0, 0), (1, 1, 8), memory_config=DRAM)
```

Setup lines253–269 creates both preallocated gate outputs as RM `[1,32,32]` in one-core height-sharded L1. The native generalized gate receives and fills these buffers. Scores are BF16 and IDs UINT16.

For either slice, `slice.cpp:233` sets `rm_only` because input layout is RM. The composite-staging predicates at lines53–89 do not intervene: source H/W are32-aligned, the input is height-sharded rather than width/block-sharded, and requested output is interleaved DRAM. The native wrapper therefore reaches `slice.cpp:382`:

```cpp
input = ttnn::to_layout(input, Layout::ROW_MAJOR, std::nullopt, memory_config);
```

That call supplies requested DRAM while `input` is already RM L1. `to_layout_op.cpp:62–81` detects equal layout, warns when memory configs differ, and directly returns `tensor_arg`. It performs no dtype conversion, memory movement, device allocation, or kernel launch. The message accurately describes this internal call; it does not describe the eventual slice output.

Next, `slice.cpp:432–450` calls `ttnn::prim::slice` with the independently retained `output_memory_config=DRAM`. `SliceDeviceOperation::compute_output_specs` constructs the output TensorSpec using that config (`slice_device_operation.cpp:354–410`). RM inputs without stepping route to `SliceRmProgramFactory` when output is interleaved (`:431–455`). Its input and output buffer bindings distinguish sharded-L1 input from DRAM output. The final metadata view retains that output storage. The following score tilization / ID reshape preserves DRAM.

Thus the native slice itself is a necessary real data-movement operation that crops eight values and places them in DRAM. The preceding `to_layout` is a redundant host-side layout check, **not an extra device layout conversion**. Trace replay contains the actual slice and required downstream work; it does not rerun this host warning. This distinction matters for the final graph accounting.

## Closed-log evidence and attribution limit

Each latest process segment in `logs/stress_0.log` and `logs/stress_extended_0.log` contains exactly six such warnings: two close together consistent with warm decode, another pair consistent with capture construction, and a final pair after the three replay-complete messages. `stress_fused_decoder.py:118–127` executes one eager warm decode and one captured decode; its continuation uses33 tokens, ending in a singleton tail under `prepare_prefill`, explaining the third pair. Trace replays produce no repeated host warnings.

This matches two router tensors per host-executed singleton step. The logs do not contain a native stack trace, so attribution of each individual line is a source/shape/count inference rather than a recorded backtrace. It is nevertheless supported by the exact call path above. The direct model-level explicit-memory `to_layout` calls in serial PF are not a competing placement bug: PF input is TILE before RM conversion, each singleton projection returns TILE before RM conversion, and final concat requests DRAM before the final tilize. If concat already returned TILE, its own explicit output config would already be DRAM.

Audited complete-log hashes (logs include preserved earlier process segments):

- `stress_0.log`: `430029c6d1565109b1d0f09f25feba7c33f903f571003072f0ff7200ba8adf22`.
- `stress_extended_0.log`: `23ab50238285dec0a7e6f1ffc47c111b2bf7a25e63a908a83ef27a960b12912f`.

## Existing device evidence proves final DRAM placement

`endurance/v4_router_direct_slice_probe.json`, SHA256 `c9e421051b7ac857bdb2aa6a93e54f1c8e62b621de321417fd72c90bcff78e1e`, reports exact completion on all four real/checkpoint-derived fixture cases with clean close. The probe at lines153–172 asserts native gate outputs are sharded RM `[1,32,32]`, then explicitly asserts:

```python
routing.memory_config() == ids.memory_config() == ttnn.DRAM_MEMORY_CONFIG
```

It also checks unchanged output shapes/dtypes/layouts, native-input immutability, and first/last replay values. The current four adapter assignments plus helper have AST SHA256 `63aace3632718993510b68c5cd4f0b14fe4f76d4c34e1ee72afc2a6c3124becd`, exactly equal to the candidate adapter hash in that completed report. The current probe file hash also matches the report's source hash. This is direct evidence of the requested final placement, not an inference from numeric accuracy alone.

No additional device check is required to resolve this placement concern while the frozen adapters and native path remain unchanged. If a future source change alters either route, the minimal check is to rerun the existing exact adapter probe with its explicit DRAM assertions; suppressing the warning or adding an unconditional model-level reshard would not establish correctness. No such edits are proposed here.
