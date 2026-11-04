---
description: 'Metal 2.0 Host API Test organization'
applyTo: 'tests/tt_metal/tt_metal/api/metal2_host_api/**,tt_metal/api/tt-metalium/experimental/metal2_host_api/**,tt_metal/impl/metal2_host_api/**'
excludeAgent: "cloud-agent"
---

# Metal 2.0 Host API Review

Metal 2.0 Host APIs have complex machinery underneath them, especially around validation.
This necessitates a sizable test suite.
In order to keep this test suite maintainable and navigable,
a rule set requires new PRs to follow the folder structure of the Metal 2.0 Host API test directory.
This instruction file highlights particular rules that PR review agents should validate.

Use `tests/tt_metal/tt_metal/api/metal2_host_api/README.md` and the other READMEs in its nested folders as the source of truth for policies.
The rules below highlight guidelines all PRs should follow if they wish to update Metal 2.0 and its tests.

The folder is mostly organized by these 3 test categories:
1. Unit tests.
   These are tests that are self-contained and fast to run;
   do not require a physical device or emulator to be present;
   nor custom kernel source code that requires the JIT system to compile it.
2. Kernel Compilation Tests.
   Like unit tests, these tests should be self-contained and relatively fast to run;
   do not require a physical device or emulator to be present;
   but can contain custom kernel source code that requires the JIT system to perform compilation as part of its validity checks.
   Note that these kernels are not run, they're merely compiled.
   These tests usually involve compile-time assertions or sanity checks on the kernel-side setup.
3. Integration Tests.
   These are the most heavyweight tests in the Metal 2.0 test suite.
   They require physical hardware or full emulation to run, and custom kernel source code to compile.
   They generally assert on the end-to-end result of device output.

Whether a test needs a device or the JIT system is not yet reflected in its test harness,
as the test suite has not evolved to take advantage of these categories.
To decide which category a test belongs to,
please judge semantically.

## 🔴 CRITICAL

- **Wrong test category**:
   Flag tests that do not belong to the directory (and thus the category) they are being checked into.
   When providing feedback, cite the category details listed above and suggest which path and file a test should be migrated to.
- **Modifying invariants, semantics or layouts of existing Metal 2.0 APIs**:
  - Update invariants listed in the README:
    Most of the Metal 2.0 headers have their invariants (classified as local vs structural) listed in:
    `tests/tt_metal/tt_metal/api/metal2_host_api/unit_tests/invariant_tests/{header_name}/README.md`.
    There, every field, and the class as a whole, has its invariants (if any) documented as part of the test README.
    This clarifies the validation rules of all the Metal 2.0 constructs.
    When a Metal 2.0 API changes (including new fields, refactoring, changed invariants or changed behaviors),
    these READMEs should be kept in sync.
    When new headers are introduced, their respective folders should also be created with the documented invariants.
  - Adding or changing checks within the test directory:
    When any fields, semantics or invariants are changed, they should come with accompanying test changes.
  - Communication:
    As a review agent, please mention that Metal 2.0 has complicated verification rules,
    which makes maintaining the navigability of its tests and invariant checks challenging;
    thus, it's critical to keep the respective invariant-checking READMEs in sync.
- **Missing from `sources.cmake`**: every new test `.cpp` under
  `tests/tt_metal/tt_metal/api/metal2_host_api/` must be listed in
  `tests/tt_metal/tt_metal/api/metal2_host_api/sources.cmake`. A file that is not listed is
  never compiled into `unit_tests_api`.
- **Maintaining folder organization**:
   There are patterns of folder organization within the subdirectories of the Metal 2.0 Host API test folder.
   Please make sure the checked-in tests are consistent with the subfolder's organization.
   Each subfolder generally has a minimal README describing what it is for.
   When new files or directories are created, suggest a short,
   one-paragraph summary that could be used to document the desired organization.

## 🟡 IMPORTANT

- **`CPU_` prefix**: tests in `unit_tests/` and `kernel_compilation_tests/` must use the `CPU_`
  name prefix. Integration tests must not.
- **Suggest cheaper test category**:
  When new tests are introduced,
  please attempt to come up with a semantically equivalent cheaper version instead (e.g. Integration -> Unit).
  This may not be possible for all tests.

## Review Checklist

- [ ] Test is in the correct category (`unit_tests/` / `kernel_compilation_tests/` / `integration_tests/`)
- [ ] New test `.cpp` is listed in `metal2_host_api/sources.cmake`
- [ ] New or changed `program_spec.cpp` validation is in a README's Listed invariants and has a matching invariant test
- [ ] Listed invariants match the fields of the header structs they mirror
- [ ] Host-only tests use the `CPU_` prefix; integration tests do not
- [ ] Kernel-compilation tests use `static_assert` and do not run kernels
