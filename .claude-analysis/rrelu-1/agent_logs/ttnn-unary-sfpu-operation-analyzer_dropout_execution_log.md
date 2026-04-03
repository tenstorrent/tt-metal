# Execution Log: ttnn-unary-sfpu-operation-analyzer (dropout)

## Metadata
- **Operation**: dropout
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Status**: SUCCESS
- **Output file**: `.claude-analysis/rrelu-1/dropout_analysis.md`

## Input Interpretation
- **Operation name**: dropout (HIGH confidence -- explicitly provided)
- **Output location**: `.claude-analysis/rrelu-1/` (HIGH confidence -- explicitly provided)
- **Naming collision**: None -- `dropout_analysis.md` did not exist

## Execution Timeline

### Phase 1: Dispatch Tracing
- Searched `unary_op_utils.cpp` for DROPOUT -- not found
- Discovered DROPOUT exists as UnaryOpType enum but is NOT dispatched through standard UnaryProgramFactory
- Found dedicated program factory at `ttnn/cpp/ttnn/operations/experimental/dropout/`
- Identified compute kernel: `dropout_kernel.cpp` with direct calls to `dropout_tile()` and `dropout_kernel_init()`

### Phase 2: Abstraction Layer Tracing
- API header: `dropout.h` uses `SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS` macro
- LLK dispatch: `ckernel_sfpu_dropout.h` (thin wrapper in llk_api layer)
- Core SFPU: `ckernel_sfpu_dropout.h` (in tt_llk layer)
- Parameters dispatch: standard `_llk_math_eltwise_unary_sfpu_params_` with VectorMode::RC

### Phase 3: SFPU Kernel Analysis
- Identified kernel style: B_raw_TTI (raw TTI instructions with CC manipulation)
- Analyzed all 8 SFPU instructions in the per-iteration loop
- Deep-dived into PRNG mechanism via SFPMOV(InstrMod=8, VC=9) from RS[9]
- Analyzed CC state machine: SFPIADD sets CC for dropout comparison, SFPMOV is CC-guarded

### Phase 4: CC State Investigation
- Extensive investigation into CC.En initial state
- ISA spec says CC.En defaults to 0 (unconditional) on reset
- All WH B0/BH raw-TTI kernels (relu, dropout, typecast, etc.) assume CC.En=1 without explicitly setting it
- Concluded CC.En must be 1 at kernel entry on WH B0/BH hardware
- Documented this as an important observation in the CC State Machine section

### Phase 5: External Research
- DeepWiki: unavailable (repo not indexed) -- 2 queries attempted, both failed
- Confluence SFPU ISA page: consulted 7 sections (SFPMOV, PRNG, RS View, SFPSETSGN, SFPIADD, SFPENCC, Predicated Execution)
- Glean: not needed

## Recovery Summary
No errors requiring recovery.

## Artifacts
- Created: `.claude-analysis/rrelu-1/dropout_analysis.md`

## Key Findings
1. **Non-standard dispatch**: Dropout does NOT use UnaryProgramFactory. It has its own DropoutProgramFactory in the experimental path.
2. **PRNG mechanism**: Uses hardware LFSR accessed via SFPMOV from RS[9] status register
3. **CC pattern**: SFPIADD with inverted subtraction for probability comparison, followed by CC-guarded zero
4. **WH/BH identical**: Both architectures have identical SFPU kernel implementation
5. **Approximation mode**: Template parameter exists but is never branched on -- same code path always executes
6. **CC.En assumption**: The kernel assumes CC.En=1 at entry, consistent with all other raw-TTI kernels
