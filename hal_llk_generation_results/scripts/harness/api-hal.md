@@API_TASK
using ONLY the HAL interface
@@API_SEE
- The HAL documentation: tt_metal/tt-llk/docs/hal/ (start from README_DOCS.md, which lists
  every document, where it comes from and whether it is known to be outdated).
@@API_CANNOT_SEE

  (including the HAL source in tt_llk_blackhole/llk_lib/hal/, in any form)
@@API_INCLUDE_GUIDE
and the HAL docs
@@API_PRODUCE
HAL-based
@@API_HOW
## How to use HAL
- Design each function from what it must do, not from how it was done before.
- Use the highest-level HAL APIs available, as the documentation intends. HAL is
  responsible for grouping and ordering the underlying instructions and config writes;
  let it make those decisions.
- Prefer higher-level calls over writing individual cfg fields one by one. Use per-field
  cfg writes only when no higher-level API exists, and list those cases in the report.
- Do not use raw TT instructions or macros (TTI_*, TT_OP_*, direct cfg register writes).
- If something truly cannot be expressed through HAL, write the minimal non-HAL code for
  that piece only, mark it with a `// NOT_IN_HAL:` comment explaining why, and continue.

## You know HAL only from its documentation
- Do not try to learn anything about the HAL implementation in any other way. That includes
  probing it through the compiler: writing deliberately wrong code so that the compiler prints
  HAL types, enum values, candidate lists or "did you mean" names, using `__has_include`, or
  guessing header or API names that the documentation does not give.
- Use only the headers and names the documentation gives.
- Compiler output has HAL implementation details removed: lines from HAL headers are shown as
  `[HAL implementation detail hidden]` and unknown names as `<hidden>`. Do not try to work
  around that.
- If the documentation is not enough to write something, report it as a documentation gap
  (with the page and what is missing). If it cannot be done with the documented API, use a
  `NOT_IN_HAL` piece as described above.
@@API_COMPILER
any compiler error in your own code
@@API_NAME
HAL
@@API_PARTS
HAL modules
@@API_REPORT_2
every `NOT_IN_HAL` piece and why HAL could not express it
