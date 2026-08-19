# fresh_cpp/ — canonical semantic C++ bodies (one op per header)

Convention (the storm contract):
- One header per op: fresh_cpp/<op>.h, body named calculate_<op>_fresh_cpp
  (template params per the established precedents in fresh_cpp_operations.h).
- PLAIN TYPED C++ ONLY: sfpi::dst_reg loads/stores, plain vFloat/vInt locals,
  constexpr constants. NO l_reg pinning, NO raw TTI_*, NO markers/annotations,
  NO magic beyond the golden math itself.
- Semantics derived from the op's MATHEMATICAL DEFINITION (PyTorch reference
  semantics / published formula), matched to the production test's golden
  contract (same golden, same tolerance) — independent derivation, never a
  transcription of the production body.
- Wired as full2x2: fresh body = sem arm, production body = hand arm;
  schedule=weekly; collect-verified nodes (no silent skips).
- Legacy note: pre-storm bodies live in ../fresh_cpp_operations.h pending
  migration here; new bodies NEVER go there.
