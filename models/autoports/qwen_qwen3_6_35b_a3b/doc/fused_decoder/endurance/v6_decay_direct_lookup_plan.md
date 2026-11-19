# Exact direct decay lookup: component evidence and standalone candidates

The selected linear decoder reads a per-head decay table generated during setup from the exact existing Torch gate arithmetic. The previous decode adapter bitcasts BF16 projection codes, computes compact indices with integer unary/binary TTNN operations, transposes/untilizes indices, gathers each head's value, and transposes/tilizes the FP32 output. This candidate performs the same integer index mapping and table reads in one model-local dataflow operation. It evaluates no exp/softplus, changes no table bits, and adds no persistent copy.

## Frozen source and algorithm

`tests/v6_decay_direct_lookup_probe.py`, SHA256 `591cad1578e85497c788f7e72b11077878014e1d1981199e1d7d16052feba0f0`, was frozen after pinned autoflake/isort/Black. CPU setup and literal reader source are retained in `v6_decay_direct_lookup_setup.json` and its sidecars. The native adapter block and three table-math assignments are AST-extracted from ab28 (`v6_corrected_fma_full_v5_triangular_compact_silu_formatted_integration.py.snapshot`), whose SHA is checked before use.

The reader accesses the original BF16 TILE projection directly. For raw code `c`, magnitude `m=c&32767`, and existing setup-derived plateau bounds `lo/hi`, compact index is:

```
clamp(m,lo,hi)-lo + max(m,32640)-32640 + ((c&32768)>>2)
```

The full-table mode uses raw code directly. For each of32 heads, the reader fetches the aligned64-byte DRAM block containing the FP32 entry, selects its exact32-bit word, and writes that word directly into the output TILE layout. Table rows remain separate by head. It emits existing DRAM FP32 TILE `[1,S,32]` layout, with zeroed physical padding. The component supports positive logical S including33/65; the decoder integration is restricted to public decode. At singleton decode it uses one worker and12KiB temporary circular-buffer storage, with no persistent L1 addition. No compute arithmetic or intermediate gather/layout tensors are used.

Layer34's bounds13056/17366 come from the unchanged table construction, not kernel constants specific to an observed input. The CPU test reconstructs every one of65,536 codes for each of32 heads from the compact table and compares all2,097,152 FP32 bit patterns. `v6_decay_direct_lookup_table_inventory.json` records1,023 subnormal outputs,8,128 NaNs, and zero negative-zero outputs in this full table, so the tests do not avoid exceptional table values. Native tilize's source selects its documented lossless FP32 mode (`ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize.cpp`, `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`); device evidence below establishes actual equality rather than relying only on that source description.

## Complete bit gate and timing

The parent-owned first actual-input probe and expanded43-case gate passed (`v6_decay_direct_lookup_actual_device.json`, `v6_decay_direct_lookup_all_device.json`). Cases comprise four saved layer34 projections, four rotated sign/NaN/boundary fixtures, seeded random S1/33/65, and32 exhaustive chunks. Each exhaustive chunk has2,048 rows and all32 heads receive each code; together they cover all65,536 codes for every head. Compact-reader output, full-reader output, native adapter output, and stored table values have zero differing FP32 bits in every case. No tolerance, NaN masking, or underflow waiver is applied.

The expanded report includes clean complete-adapter timing with five warmups, three eager groups of20 calls, and three trace groups of100 replays. Inputs and original tables are already resident, as in the decoder. Native indexing, transposes, untilize, gather, and tilize are all measured. Output placement is identical. First/last replay outputs and determinism have zero bit differences.

| Complete decay adapter | Native, ms | Direct compact, ms |
|---|---:|---:|
| Warmed eager median | 0.1407947624 | 0.02027174924 |
| Traced median | 0.05306608044 | 0.00883941073 |

These are isolated complete-decay-adapter results. Full-decoder gain is not inferred from operation count or these timings.

## Explicit physical code-domain coverage

Device-assisted `from_torch(..., TILE)` can choose a numeric tilize route. To prove the intended code domain reaches the reader unchanged, a separate frozen wrapper constructs the BF16 TILE tensor on the host, copies it to the device, bitcasts the resident tensor to UINT16, and verifies every uploaded bit pattern before executing the same native/direct adapter comparison.

`tests/v6_decay_direct_lookup_rawcode_probe.py`, SHA256 `4c9148b3ac5be6dd5194c302d7dfe5d6bbbad15190a13a09057100f2fc11b33b`, passed all43 cases in `v6_decay_direct_lookup_rawcode_all_device.json`. Every raw UINT16 upload matches, including subnormals, signed zero, both infinities, and all NaN payloads. Every native/compact/full/stored-table FP32 comparison remains bitexact. `v6_decay_direct_lookup_rawcode_transfer.json` verifies that the runtime class, kernel literals, native adapter, table math, and fixture ASTs are identical to the timed591cad component. Only diagnostic setup upload/readback changes. No host operation appears in the measured or decoder runtime path.

Reproduction commands, using fresh output paths:

```bash
PYTHONPATH=. python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_decay_direct_lookup_probe.py --run-device --time --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_decay_direct_lookup_all_device.json
PYTHONPATH=. python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_decay_direct_lookup_rawcode_probe.py --run-device --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_decay_direct_lookup_rawcode_all_device.json
```

## Exact decoder integration and remaining gates

`tests/v6_build_decay_lookup_snapshot.py` emits standalone candidates containing static helper/class and literal kernel definitions. It adds a nonallocating setup object referencing the existing compact table (or existing full table) and the existing computed lo/hi. Original CPU table generation, compact reconstruction assertion, and all allocations remain unchanged. Source inspection confirms lo/hi are not reassigned later in `from_state_dict`.

The runtime guard is `decode and cache_boundary`. Inside that branch, `g=None` and direct lookup supplies `decay_value`; the original adapter block is the exact else branch. Therefore all prefill calls, including public/internal singleton calls with `cache_boundary=False`, retain their original path. Full-attention setup and all public method ASTs remain unchanged. No public alignment, sharding, dtype, or capacity contract is reduced.

- Independent ab28 candidate: `v6_ab28_decay_lookup_integration.py.snapshot`, SHA256 `ef066e5d3a034ab9399cb9060372b39a1709c48e38691124469cca2c078c2597`; exact proof `v6_ab28_decay_lookup_integration_build.json`.
- Composed candidate on803685e1 (transport, convolution, norm-terminal scan): `v6_transport_conv_norm_decay_lookup_integration.py.snapshot`, SHA256 `44f87c04349836a4ad8b09c7a6c573fe3afe1e558af285e0482cce2c0fe5b2b3`; exact proof `v6_transport_conv_norm_decay_lookup_integration_build.json`.

Both snapshots passed pinned autoflake/isort/Black before freezing. Their proofs reverse only the precise inserted setup statement, new runtime conditional, and three appended helper definitions, restoring the complete original module AST. `v6_decay_lookup_composition_transfer.json` independently proves the new setup, new runtime conditional including original else, and all appended helpers are identical between both candidates. The builders' rawcode-gate reminder is satisfied by the supplemental transfer artifact above.

The parent owns independent/composed whole-decoder every-token accuracy, warmed/traced performance, repeated-run, context, watcher, and final review gates. These snapshots are not production changes or final acceptance claims. There are no rejected numerical alternatives for this candidate: the initial direct-reader component passed; the supplemental wrapper strengthened input-domain provenance without changing runtime behavior. Native mapping/table generation remains the contract.
