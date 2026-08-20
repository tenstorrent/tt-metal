# Metal 2.0 Python ProgramSpec playground — shared recipe

## Goal
Find out what the pybound Metal 2.0 surface can and cannot express. We are AUTHORING new ops
from scratch in Python, NOT porting C++ ops. Record findings in four buckets:

- **BLOCKED** — I wanted to do X and there is no way to express it.
- **UGLY** — I can do X, but the expression is bad / verbose / surprising.
- **BROKE THE MODEL** — I could only do X by deliberately defeating a Metal 2.0 invariant
  (smuggling an address through an RTA, faking an endpoint, `.id` extraction, etc).
- **WIN** — genuinely better than ProgramDescriptor: a class of bug that can't happen now,
  or code that is materially cleaner.

Each finding: what you tried, the exact error/diff, and a one-line verdict.

## Device discipline (NON-NEGOTIABLE)
- Device access serializes on a flock across all of us. **Batch aggressively**: one probe file
  that exercises 10 things beats 10 probe files.
- `source python_env/bin/activate` first.
- Run ONLY via `scripts/run_safe_pytest.sh [--dev] [--run-all] <file>` or `scripts/tt-probe.sh`.
- NEVER `python3` directly on device code. NEVER `run_in_background: true`. NEVER `tt-smi -r`.
- Kernel code is JIT-built at runtime — do NOT rebuild metal for kernel edits.

## The invocation shape
```python
spec, run_args = build(...)                       # ttnn.ProgramSpec, ttnn.ProgramRunArgs
ttnn.generic_op([a, b, out], spec, run_args, {TP_A: 0, TP_B: 1, TP_OUT: 2})
#                ^io list             ^ TensorParameter name -> index into the io list
```
`ttnn.CONFIG.validate_program_args = True` turns on the host legality checks. Use it.

## Reference material
- Working examples: `ttnn/ttnn/operations/toy_spec_mul/` (interleaved eltwise, 3 kernels),
  `ttnn/ttnn/operations/toy_spec_mcast/` (cross-core), `ttnn/ttnn/mcast_spec.py` (host mcast helper).
- Tests: `tests/ttnn/unit_tests/operations/toy_spec_{mul,mcast}/`.
- TT_KERNEL named-arg kernels: `tests/tt_metal/tt_metal/test_kernels/{compute,dataflow}/tt_kernel_named_args_*.cpp`.
- Full Python surface: `ttnn/ttnn/types.py:122-160`.
- Upstream (C++) docs, mirrored: `tt_metal/third_party/tt_ops_code_gen/references/external/metal2/`.
  These describe PORTING C++ ops. Useful for semantics + validator rules, NOT for our ergonomics.
- Generators (ground truth for what tokens exist): `tt_metal/jit_build/genfiles.cpp`
  (`write_kernel_bindings_generated_header` ~line 105, `write_kernel_args_generated_header` ~line 255),
  `tt_metal/jit_build/kernel_signature_parser.cpp:461` (the TT_KERNEL shim).

## Known facts (don't re-derive)
- `dfb::name` is a `constexpr DFBBindingToken` with an implicit `constexpr operator uint32_t()`,
  so it passes straight into any LLK / kernel_lib helper taking a cb id.
- Validator: every DFB needs >=1 PRODUCER **and** >=1 CONSUMER; every TensorParameter needs
  >=1 binding (exception: one named by a DFB's `borrowed_from`).
- `TT_KERNEL` syntax = CTAs as template params, RTA/CRTA as fn params. It CANNOT express varargs.
- `if constexpr` does NOT save you from an unbound `dfb::x` — non-dependent name lookup happens
  at template definition time. Verified.
- Compute (TRISC) kernels cannot bind a TensorAccessor (no NoC includes in the TRISC build).

## Write findings to
`tests/ttnn/unit_tests/operations/metal2_play/FINDINGS_<yourarea>.md`
