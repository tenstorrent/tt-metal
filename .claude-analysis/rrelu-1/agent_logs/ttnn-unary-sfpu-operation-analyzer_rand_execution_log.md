# Execution Log: ttnn-unary-sfpu-operation-analyzer (rand)

## Metadata
- **Operation**: rand
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Status**: SUCCESS
- **Output file**: `.claude-analysis/rrelu-1/rand_analysis.md`

## Input Interpretation
| Field | Value | Confidence |
|-------|-------|------------|
| Operation name | `rand` | HIGH |
| Output directory | `.claude-analysis/rrelu-1/` | HIGH (explicit override) |
| Output filename | `rand_analysis.md` | HIGH (explicit override) |

## Execution Timeline

### Phase 1: Dispatch Tracing
- Searched for `RAND` / `rand` in `unary_op_utils.hpp` / `.cpp` -- not found. `rand` is NOT a standard `UnaryOpType`.
- Discovered dedicated program factory at `ttnn/cpp/ttnn/operations/rand/device/rand_program_factory.cpp`.
- Traced compute kernel: `ttnn/cpp/ttnn/operations/rand/device/kernels/compute_uniform.cpp`.
- Identified tile-level calls: `rand_tile_init(seed)` and `rand_tile(0, from, scale)`.
- Traced through API header (`rand.h`) and macro expansions to core SFPU implementation.

### Phase 2: Kernel Source Analysis
- Read Wormhole B0 and Blackhole versions of `ckernel_sfpu_rand.h`.
- Classified as Style A (inline comments) -- raw TTI instructions but NO condition code manipulation.
- Identified key architectural difference: Wormhole requires SFPNOP after 2-cycle instructions, Blackhole does not.
- Identified PRNG access mechanism via `SFPMOV` with `mod1=8, VC=9` (RS[9] = PRNG Counter).

### Phase 3: External Research
- DeepWiki: Unavailable (repository not indexed).
- Confluence ISA page (1170505767): Successfully consulted for SFPMOV/PRNG semantics, SFPSETSGN, SFPSETEXP, SFPADDI specifications.
- Confirmed PRNG is 32-bit LFSR with XNOR taps at 31, 30, 10, 0.

### Phase 4: Verification
- All function names verified via grep.
- All file paths verified via `ls`.
- All SFPU instructions verified to exist in the ckernel_sfpu_rand.h files.

## Non-Standard Discovery
`rand` does NOT use `UnaryProgramFactory`. It has its own `RandDeviceOperation::ProgramFactory` with:
- No reader kernel (generates data, not transforms)
- FP32 intermediate circular buffer (CB c_24)
- `fp32_dest_acc_en = true` for precision
- Writer kernel handles FP32-to-BF16 conversion in software when needed

## Artifacts
| File | Action |
|------|--------|
| `.claude-analysis/rrelu-1/rand_analysis.md` | Created |
| `.claude-analysis/rrelu-1/agent_logs/ttnn-unary-sfpu-operation-analyzer_rand_execution_log.md` | Created |

## Deviations
- The `rand` operation is not a standard unary op (no `UnaryOpType` enum value, no `UnaryProgramFactory`). Analysis adapted by tracing the dedicated program factory instead.
- The output filename was overridden to `rand_analysis.md` per the caller's instructions.
