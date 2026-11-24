Write the LLK functions used by the tests listed below from scratch, using ONLY the HAL interface.
Target architecture: Blackhole.

Build ONE shared library: implement each LLK function once, in headers under
tt_metal/tt-llk/tests/hal_llk/, and use that same implementation from every test that needs
it. Do not write per-test variants of a function.

## Tests
| Original test source | Python test | Test source you write | Python test for your source | Command (from tt_metal/tt-llk/tests) |
|---|---|---|---|---|
| tests/sources/eltwise_binary_test.cpp | tests/python_tests/test_eltwise_binary.py | tests/sources/eltwise_binary_hal_test.cpp | tests/python_tests/test_eltwise_binary_hal.py | `pytest python_tests/test_eltwise_binary_hal.py -k 'add'` |

Scope: every variant the python test selects with the command shown in the table.

## What you can see
- The tests themselves: the original test sources in tt_metal/tt-llk/tests/sources/, the
  python tests in tests/python_tests/ and the helpers in tests/helpers/. These tell you what
  each LLK function must achieve and how it is called.
- The HAL documentation: tt_metal/tt-llk/docs/hal/ (start from README_DOCS.md, which lists
  every document, where it comes from and whether it is known to be outdated).
- A spec of the LLK library: tt_metal/tt-llk/llk_spec/tt_llk_blackhole/ mirrors
  llk_lib/ and common/inc/ with every function SIGNATURE and doc comment, but no
  function bodies and no raw instruction encodings. Use it to learn the exact names,
  template parameters, signatures and the types/enums you must use.

## What you cannot see
- The original LLK implementations and everything else under tt_llk_blackhole/
  (including the HAL source in tt_llk_blackhole/llk_lib/hal/).
  Reading them is blocked. Include headers by name as the spec and the HAL docs show,
  but do not try to read the originals in any other way.
- You therefore cannot compare against the original implementations. Correctness
  is verified only by the tests passing.

## What to produce
- HAL-based implementations of every LLK function the tests use, in new headers under
  tt_metal/tt-llk/tests/hal_llk/, with exactly the names and signatures from the spec,
  so the test bodies stay the same.
- For every test in the table, the HAL test source named in the table: a copy of the
  original test source that is identical except that it includes your headers
  (e.g. #include "../hal_llk/...") instead of the original llk_*.h headers.
- The HAL python tests in the table already exist (the harness generated them); they run
  your test sources. Do not change them.

## How to use HAL
- Design each function from what it must do, not from how it was done before.
- Use the highest-level HAL APIs available, as the documentation intends. HAL is
  responsible for grouping and ordering the underlying instructions and config writes;
  let it make those decisions.
- Prefer higher-level calls over writing individual cfg fields one by one. Use per-field
  cfg writes only when no higher-level API exists, and list those cases in the report.
- Do not use raw TT instructions or macros (TTI_*, TT_OP_*, direct cfg register writes through
  a pointer such as `cfg[ADDR] = value`). `hal::cfg::write<hal::cfg::Access::MMIO>` is a HAL API
  and is allowed.
- If something truly cannot be expressed through HAL, write the minimal non-HAL code for
  that piece only, mark it with a `// NOT_IN_HAL:` comment explaining why, and continue.

## Instruction cost of per-tile code
The per-tile functions (`_llk_unpack_AB_`, `_llk_math_eltwise_binary_`, `_llk_pack_`) run once
for every tile, so every instruction they emit is multiplied by the number of tiles. They must
not emit more instructions per tile than the work needs.

How `hal::cfg::write` reaches the hardware:
- `Access::TensixCfgUnit` with a compile-time value: immediate instructions (TTI). A state-CFG
  word gets one RMWCIB per byte the write touches; a thread-CFG word gets one SETC16.
- `Access::TensixCfgUnit` with a runtime value: the same instructions, but the RISC-V core
  builds each one at runtime and stores it to the instruction buffer (TT). A 32-bit state-CFG
  field therefore costs 4 TT instructions.
- `Access::TensixCfgUnit` with a GPR source (`cfg::gpr<N>()`): one WRCFG for the whole word,
  ordered with the Tensix instruction stream. The GPR has to be loaded first.
- `Access::MMIO`: one RISC-V store for a full-word field (a read-modify-write for a narrower
  field). It is not ordered with the Tensix instruction stream, so a STALLWAIT does not delay it.

Rules for per-tile code:
- Do not write a value that changes on every tile (for example an L1 address) into a state-CFG
  field with `Access::TensixCfgUnit` and a runtime value. Use `Access::MMIO`, or a GPR + WRCFG
  write.
- An `Access::MMIO` write must not change configuration that a unit may still be using. Make it
  safe without stalling that unit on every tile: for example, write the next tile's settings into
  a configuration context the unit is not using, and synchronize with a semaphore. HAL exposes the
  unpacker contexts (`Unpacker[u].Cntx[c].Base`, `UnpackMiscCfg::CfgContextOffset_*`) and
  semaphores (`hal::sync::semaphore`).
- The access path is your choice; HAL does not choose it for you.

## Rules
- Do NOT modify anything outside tests/hal_llk/ and the HAL test sources named in the table.
- If the device hangs, reset it with `tt-smi -r` and continue.
- Do not run commands in the background or schedule wake-ups. Run the final
  verification in the foreground, wait for it, and end with the final report.

## Verification
- Run each HAL test from tt_metal/tt-llk/tests with the command in the table.
- The results must match the original: every test must pass with the same tolerance.
- Do not finish until every test passes.
- Before finishing, run every HAL test once more, each one from a clean device: reset the
  device with `tt-smi -r` right before each test and run nothing else in between. All of
  them must pass from that clean state.
- Every test must pass both with and without a device reset before it. Your code must
  leave the device in a state in which the original tests also pass right after your HAL tests.
- Before finishing, run this sequence with nothing else in between and give each result line
  in the final report:
  1. `tt-smi -r`, then every HAL test;
  2. without a reset, every original test;
  3. without a reset, every HAL test again;
  4. `tt-smi -r`, then every HAL test.
  All of them must pass.

## Sources for every finding
For every gap, contradiction or problem in your final report, name exactly where it comes from:
the document or header file (and section) that says what, any compiler message that shows what
the HAL headers actually accept, and what you observed in a test. Then classify it as one of:
documentation gap (the docs are missing or wrong), implementation gap (HAL lacks or
mishandles something), contradiction between sources (two documents, or a document and the code,
disagree), or other. Do not report a finding without its source.

## Final report
At the end, list:
1. every LLK function you implemented, which HAL modules it uses and which tests use it,
2. every `NOT_IN_HAL` piece and why HAL could not express it,
3. every place where you had to write individual configuration fields instead of a higher-level construct,
   and, for every cfg write in a per-tile function, its access path and how many instructions it emits per tile,
4. anything in the reference material or the spec that was unclear, missing or wrong,
5. how you worked out what each function must do without the original implementation,
6. the final result of every test in the table (passed / failed / skipped counts).
