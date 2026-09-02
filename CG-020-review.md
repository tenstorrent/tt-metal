# Review of CG-020 (naming convention for safety mechanisms)

Rule CG-020, in Section 2.3 of the *Coding and Architecture Guidelines* (Draft v0.1),
requires the suffix `_sm` on any function, class or variable that implements or interacts
with a safety mechanism, and `_diag` on diagnostic-only tooling. It traces to finding FM-012:
diagnostic tooling is silently relied on as a safety mechanism.

This note answers two questions: whether CG-020 is standard practice in safety applications,
and if not, what should take its place.

## Summary

Having a naming convention is a standard requirement. Encoding safety relevance in the
identifier itself is not, and none of the five safety projects examined below does it. The
rule also cannot be checked by any tool, because nothing in the source code says whether a
symbol belongs to a safety mechanism.

Two findings are specific to this codebase. The suffix `_sm` already means softmax here, so
the rule would collide with existing names. And Section 2.3 sets no rule for how a name is
spelled, leaving capitalization, macro prefixes and namespace leakage all unconstrained, even
though the `clang-tidy` configuration in the repository could enforce all three today.

**Recommendation: delete CG-020 and adopt recommendations 1, 2 and 4 below. Recommendation 3
applies only if CG-020 is kept.**

## 1. What the safety standards require

ISO 26262-6:2018 Table 1 lists method 1g, use of style guides, and method 1h, use of naming
conventions, as two of the coding guideline topics a project must address. DO-178C likewise
requires a Software Code Standards document and leaves its content to the applicant.

Where identifier rules do appear in the language standards, they cover distinctness only:

- MISRA C:2012 Rules 5.1 to 5.9: external identifiers distinct, no shadowing, macro
  identifiers distinct, typedef and tag names unique.
- AUTOSAR C++14 Rules A2-10-1, A2-10-4, A2-10-5, A2-10-6 and M2-10-1.

Neither set says anything about what a name should mean.

## 2. What other safety projects do

Naming conventions in safety code serve uniqueness, readability and traceability. None of the
five projects below uses one to mark safety relevance. The first three are C, the last two C++.

**AUTOSAR Classic Platform.** Requirement SWS_BSW_00101 fixes a module abbreviation (`EcuM`,
`CanIf`, `Com`), and SWS_BSW_00102 forms the module implementation prefix as
`<Ma>[_<vi>_<ai>]`, where `<vi>` is the vendor identifier and `<ai>` the vendor API infix. The
same abbreviation appears inside the requirement identifiers as `[SWS_<Ma>_nnnn]`, so one
token ties the requirement, the file and the symbol together.

**Zephyr RTOS**, targeting IEC 61508 SIL 3. Every public interface is prefixed by its
subsystem (`k_` kernel, `sys_` system, `net_` networking, `bt_` Bluetooth). Style follows the
Linux kernel and is enforced automatically by the `checkpatch` tool, not left to reviewers to
catch.

**BARR-C:2018**, a freely published embedded C standard, is the most prescriptive on
capitalization:

| Rule | Text |
|---|---|
| 5.1.a | New data types shall use only lowercase and internal underscores, and end with `_t` |
| 6.1.e | No function name shall contain any uppercase letters |
| 6.1.f | No macro name shall contain any lowercase letters |
| 6.1.i | Public function names shall be prefixed with their module name and an underscore |
| 7.1.f | No variable name shall contain any uppercase letters |
| 7.1.j to 7.1.o | Globals begin with `g`, pointers `p`, booleans `b`, handles `h`, in that order |

All three keep safety information outside the identifier. AUTOSAR records it in the ARXML
files that describe the components, and Zephyr expresses it through which set of coding rules
a file must satisfy, recorded outside the file.

All three are also C, where a prefix is the only way to separate symbols. Two modern C++
projects with the same safety intent show what carries across. **Eclipse S-CORE** is an
automotive platform built to ISO 26262 from the start, and **Eclipse iceoryx** is the C++
middleware used in automotive and in ROS 2. iceoryx fixes capitalization across seven
categories of identifier: files `lower_snake_case`, types `UpperCamelCase`, methods and
variables `lowerCamelCase`, compile-time constants and enum values `UPPER_SNAKE_CASE`, members
prefixed `m_`, namespaces `lower_snake_case`, aliases suffixed `_t`. Neither project marks
safety relevance in a name, and neither prefixes C++ symbols with a module token, because
namespaces already separate them. S-CORE's own comparison of the Google Style Guide, the C++
Core Guidelines and AUTOSAR C++14 finds that the three conflict on capitalization for types and
functions, but agree without exception that macros need ALL_CAPS and a project-specific prefix.
Recommendation 4 follows that split.

## 3. Why CG-020 cannot be checked automatically

Enforcing CG-020 requires a hand-maintained list of which symbols belong to a safety mechanism.
Once that list exists, it already provides the mapping from code to architecture element that
the naming was meant to supply. The naming becomes a second copy of the list rather than a
substitute for it, and the two can then disagree.

No tool can build that list, because the fact CG-020 encodes lives in the architecture
description, not in the code. A checker can read a translation unit; it cannot read a design
document. A name therefore goes stale when the architecture description or the hazard analysis
changes, and nobody editing the C++ file sees that happen.

In contrast, capitalization rules don't have this problem, because they constrain only the
spelling of a name so no outside information is required to verify them.

## 4. Two findings in this codebase

**Both suffixes are already taken.** `_sm` means softmax. The softmax operation's own
documentation at
[Softmax.md:363-367](ttnn/cpp/ttnn/operations/normalization/softmax/docs/Softmax.md#L363-L367)
lists five reader kernels whose names all carry `_sm`, among them
`reader_unary_sharded_sm.cpp`, whose name carries both `sharded` and `_sm`. A reader meeting
that file would have to guess whether `_sm` marks softmax or a safety mechanism, and the same
suffix would carry two unrelated meanings in one repository.

The `_diag` suffix is already in use with two different meanings: matrix diagonals (`is_diag`,
`q_diag`, `try_skip_causal_above_diag`) and a genuine diagnostic
(`dram_ncrisc_run_deferred_diag`). Adopting CG-020 would mean renaming the diagonal uses, which
is churn with no safety benefit, while the one real diagnostic already complies without the
rule.

**Nothing enforces a naming style.** Section 2.3 has rules about what a name *means* and none
about how a name is *spelled*, the opposite emphasis from every project in Section 2. Three
gaps follow, all of them cheap to close:

- **Capitalization.** Class and struct names in the public headers under `ttnn/api` split 71
  UpperCamelCase to 17 snake_case.[^1] The `clang-tidy` check `readability-identifier-naming`
  is already enabled, because the check list at [.clang-tidy:39-40](.clang-tidy#L39-L40) begins
  with `*` and the only `readability-identifier-*` check switched off is
  `readability-identifier-length` at [.clang-tidy:115](.clang-tidy#L115). No `CheckOptions`
  entry configures a naming style, so the check runs today and reports nothing. This is a
  common pitfall rather than an oversight here: S-CORE's central baseline enables the same
  check by wildcard and carries the comment "CheckOptions are yet subject to be provided",
  so it reports nothing either.
- **Macro prefixes.** C++ namespaces cannot contain macros, because the preprocessor runs before
  the compiler sees a namespace, so a macro name collides across the whole translation unit.
  `ttnn/` defines 538 macros, among them names as generic as `FACE_WIDTH`, `FACE_SIZE` and
  `INDEX_TILE_SIZE`.
- **Namespace leakage.** A namespace only separates symbols while callers leave it in place.
  645 files under `ttnn/` have a file-scope `using namespace`, 16 of them in headers, where it
  reaches every file that includes them. The check that would catch this,
  `google-build-using-namespace`, is switched off at [.clang-tidy:73](.clang-tidy#L73).

## 5. Recommendations

1. **Prevent FM-012 in the build rather than with a naming rule.** If diagnostic tooling is
   excluded from the production build target, relying on it becomes a link error instead of
   something a reviewer must notice. A register of safety mechanisms with stated coverage makes
   undeclared reliance visible as a missing entry.

2. **Separate safety code structurally, and let naming reflect that.** Put safety-relevant code
   in its own module, directory, build target and memory partition, which is what
   ISO 26262-9:2018 Clause 6 asks for when it discusses coexistence of elements. A namespace or
   a dedicated directory gives the same visible marking CG-020 wants, in a form the compiler
   checks, `grep` finds and the build system enforces.

3. **If CG-020 is kept, narrow it.** Apply it to functions and classes, not variables: the set
   of variables that interact with a safety mechanism has no clear boundary, and renaming them
   is costly. Choose suffixes the codebase has not already claimed.

4. **Add spelling rules to Section 2.3, scoped to what C++ namespaces do not already cover.**
   A capitalization rule for all identifiers; a prefix rule for macros only, since namespaces
   already separate every other kind of symbol; and a ban on `using namespace` in headers, so
   the namespaces already in place hold. A blanket prefix rule on C++ symbols would duplicate
   the namespaces and cause renaming for no gain. All three are enforceable by the `clang-tidy`
   configuration already in the repository, as Section 4 shows.

   Apply the capitalization rule to new code only. iceoryx encoded the convention quoted in
   Section 2 as `clang-tidy` options, then backed the enforcement out. Its configuration
   enables `readability-*` near the top of the check list but then disables the naming check with
   [`-readability-identifier-naming`](https://github.com/eclipse-iceoryx/iceoryx/blob/08bf71b0384fc6092b8f910f90a5f2832d5612ea/.clang-tidy#L36)
   further down, and because the list is evaluated in order the later entry wins. The reason is
   recorded a few lines below as
   ["Temporarily disabled because massive API changes"](https://github.com/eclipse-iceoryx/iceoryx/blob/08bf71b0384fc6092b8f910f90a5f2832d5612ea/.clang-tidy#L46-L47),
   and the convention itself is still in the file at
   [lines 196 to 209](https://github.com/eclipse-iceoryx/iceoryx/blob/08bf71b0384fc6092b8f910f90a5f2832d5612ea/.clang-tidy#L196-L209),
   but those CheckOptions have no effect while the check is off.
   Given the 71 to 17 split in the public headers here, enabling the check across this
   repository would reach the same result. Limiting `clang-tidy` to changed lines enforces the
   rule on new code without a repository-wide rename.

### Note on CG-021

CG-021, the traceability naming rule, would be better served by the AUTOSAR pattern: give each
architecture element one token, use it as the prefix for its files and public symbols, and use
the same token in the requirement identifiers. That meets CG-021's goal and leaves safety
marking in the architecture metadata, where the other projects keep it.

[^1]: Unique names from
`grep -rhoE '^[[:space:]]*(class|struct)[[:space:]]+[A-Za-z_][A-Za-z0-9_]*' --include=*.hpp ttnn/api | sed -E 's/^[[:space:]]*(class|struct)[[:space:]]+//' | sort -u`

## References

- ISO 26262-6:2018 Table 1 (methods 1g and 1h); ISO 26262-9:2018 Clause 6
- MISRA C:2012 Rules 5.1 to 5.9; DO-178C Software Code Standards
- [Zephyr Coding Guidelines](https://docs.zephyrproject.org/latest/contribute/coding_guidelines/index.html)
  and [Naming Conventions](https://docs.zephyrproject.org/latest/contribute/style/naming.html)
- [BARR-C:2018 Embedded C Coding Standard](https://barrgroup.com/embedded-c-coding-standard)
- [Eclipse S-CORE C++ policies](https://github.com/eclipse-score/score_cpp_policies) and its
  [C++ style comparison discussion](https://github.com/orgs/eclipse-score/discussions/657)
- [Eclipse iceoryx naming conventions](https://github.com/eclipse-iceoryx/iceoryx/blob/main/CONTRIBUTING.md)
  and its [clang-tidy configuration](https://github.com/eclipse-iceoryx/iceoryx/blob/08bf71b0384fc6092b8f910f90a5f2832d5612ea/.clang-tidy)
- [C++ Core Guidelines SF.7](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md)
  (do not write `using namespace` at global scope in a header file)
- [AUTOSAR CP General Specification of Basic Software Modules R24-11](https://www.autosar.org/fileadmin/standards/R24-11/CP/AUTOSAR_CP_SWS_BSWGeneral.pdf)
- AUTOSAR C++14 rule text as listed by
  [MathWorks Polyspace documentation](https://www.mathworks.com/help/bugfinder/autosar-c-14.html)
