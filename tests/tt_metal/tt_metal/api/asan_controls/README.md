# ASAN reachability control tests

These tests answer a different question from the existing per-check death tests
in `tests/tt_metal/tt_metal/api/test_*.cpp`.

The existing tests ask: **does the check's logic work?**
They typically reach *inside* the sanitizer and call its host function directly
from the kernel source. For example `OOB_Tensor_Gap_DRAM_SanityCheck` does:

```cpp
extern "C" uint8_t* __emule_dram_ptr(uint64_t offset);
void kernel_main() {
    volatile uint32_t* bad_ptr = (volatile uint32_t*)__emule_dram_ptr(addr);
    *bad_ptr = 0x777;
}
```

That is a valid test of the *comparison*, but it proves nothing about whether a
real kernel can ever get there. `__emule_dram_ptr` currently has **zero call
sites** in emule — the DRAM access path was migrated to
`__emule_resolve_noc_addr` for multi-bank correctness (see
`api/tensor/noc_traits.h`), and the check did not move with it. So that death
test is green while the check is unreachable in production.

These tests ask instead: **can a violation committed through the PUBLIC kernel
API reach the check?** They use only APIs a real op would use —
`cb_reserve_back`, `get_write_ptr`, `noc_async_write`, tensor accessors — and
never name an `__emule_*` internal.

## Naming convention

- `*_Reachable` — a violation via public API **must** abort. Regression fence
  against a check being orphaned by a refactor.
- `*_Unreachable` — a violation via public API is currently **not** detected.
  These are documentation of a live coverage gap, not assertions that the
  behaviour is correct. Each carries a comment explaining the cause and what
  would have to change. Flip them to `_Reachable` when the gap is closed.

## Why this folder exists

A per-check tally on real Blaze ops (see
`~/asan_logs/ALL_CHECKS_BLAZE_AUDIT_2026-07-29.md`) showed several checks with
zero evaluations. Two very different causes were behind that, and only a
reachability test distinguishes them:

1. the check does not fit how Blaze touches memory (e.g. §4 OOB-L1 is
   short-circuited by `cb_resolve` because all Blaze data lives in CBs), or
2. **emule** stopped routing through the function that hosts the check
   (§4 OOB-DRAM), which is an emule-side problem, not a check-design problem.

Distinguishing those is the whole point.
