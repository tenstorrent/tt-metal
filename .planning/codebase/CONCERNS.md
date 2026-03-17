# Codebase Concerns: tt-metal Fabric Router

**Analyzed:** 2026-03-12
**Focus:** Technical debt, known issues, performance bottlenecks, fragile areas

---

## Active Refactoring (In-Progress)

### Per-VC Indexing Conversion
- **Location:** `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp`
- **Status:** ~10 remaining tasks to convert flat sender channel arrays to per-VC template accessors
- **Risk:** Partial conversion leaves two indexing paradigms in same file — easy to introduce off-by-one bugs at boundary

---

## Known Bugs

| Issue | Description | Workaround |
|-------|-------------|------------|
| #29073 | Two-ERISC receiver TXQ hang | Disabled; single-ERISC workaround active |
| #36811 | 9 disabled descriptor merger tests for multi-host scenarios | Tests skipped |

---

## Performance Bottlenecks

### Single ERISC Bottleneck
- Current ceiling: ~31.5 GB/s (single active_erisc core)
- Root cause: Dual-ERISC path (active + subordinate_active) not yet enabled for speedy path
- Recommendation: Enabling dual-ERISC is the next major throughput unlock

### Volatile Pointer Credit State
- `credit_epoch_array[]` is file-static and accessed via volatile pointer
- Prevents compiler from register-caching credit values across iterations
- Workaround: epoch_accumulator on stack reduces writes; full elimination not possible

### Stack Struct Register Pressure
- `LineSenderState` / `LineReceiverState` structs passed by reference via FORCE_INLINE
- Compiler promotes to registers when structs are stack-local — sensitive to inlining decisions
- Risk: Future changes to FORCE_INLINE functions may degrade register allocation

---

## Fragile Areas

### Speedy Path Inline Optimization
- 9-cycle micro-optimization in `fabric_erisc_router_speedy_path.hpp` tightly coupled to RISC-V compiler register allocation behavior
- Assembly must be verified after any change to the hot path
- Location: `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_speedy_path.hpp`

### Per-VC Context Switching Isolation
- Each VC must maintain independent sender/receiver state
- No focused unit tests for isolation — failures surface as subtle data corruption
- Risk increases as per-VC indexing refactor progresses

### Hardcoded Handshake Timeouts
- Timeout values hardcoded at build time
- Cannot be adjusted at runtime for different network topologies or latency profiles

---

## Test Coverage Gaps

| Area | Gap | Impact |
|------|-----|--------|
| Kernel functions | No unit tests for kernel (3,608 line monolith) | Regressions only caught by integration tests |
| Per-VC isolation | No tests verifying VCs don't bleed state | Silent data corruption risk |
| Chaos/fault injection | No fault injection tests | Unknown failure modes |
| Receiver credit loop | No instrumentation tests | Hard to debug credit starvation |

---

## Missing Features / Tech Debt

- **Configurable credit amortization:** `epoch_accumulator` frequency hardcoded — no runtime tuning
- **Deadlock detection/recovery:** No mechanism to detect or recover from credit deadlocks
- **Receiver credit instrumentation:** No visibility into receiver-side credit loop timing
- **Monolithic kernel file:** `fabric_erisc_router.cpp` at 3,608 lines — difficult to navigate and test in isolation

---

## Recommendations

1. **Complete per-VC indexing refactor** before adding new VC features — the hybrid state is the highest debt item
2. **Add per-VC isolation tests** alongside the refactor to catch regression early
3. **Verify assembly after every hot-path change** — use `erisc-disasm-explorer` at `/home/snijjar/tt-scaleout-dev-tooling-internal/scripts/erisc-disasm-explorer`
4. **Fix Issue #29073** (TXQ hang) before enabling dual-ERISC path

---

*Last updated: 2026-03-12 via gsd:map-codebase*
