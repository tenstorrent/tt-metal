# Metalium API review policy and automation

The objective is to enforce routine API rules automatically and leave API owners
with explicit design, compatibility, and exception decisions. This policy covers
the host API under `tt_metal/api/`; it does not extend stability guarantees to
TTNN or device kernel APIs.

## Stability boundaries

Header locations below are relative to `tt_metal/api/`.

| Header location | Audience | Allowed API dependencies |
| --- | --- | --- |
| `tt-metalium/`, excluding `experimental/` | Stable public API | Stable |
| `tt-metalium/experimental/` | External users accepting change | Stable, experimental |
| `internal/` | Tenstorrent components only | Stable, experimental, internal |

Internal headers belong in `tt_metal/api/internal/`, never `tt_metal/api/tt-metalium/internal/`.
See the [internal API README](../../tt_metal/api/internal/README.md) for the
difference between internal infrastructure and experimental functionality
intended for productization.

The boundary applies to transitive dependencies and exposed types as well as
direct includes. Moving an include into a helper header or replacing it with a
forward declaration must not expose a less stable type through a public
signature, alias, or base class. Private opaque implementation types and the
approved experimental free-function/friend pattern remain permitted.

## Checks implemented today

Run from the repository root:

```sh
python3 scripts/validate_api/validate_api.py tt_metal/api
python3 -m unittest discover -s scripts/validate_api -p 'test_*.py'
```

The `validate-metalium-api` pre-commit hook runs all API validation
through this command. The `all-static-checks.yaml`
workflow runs on PR updates and merge groups. Checker failures fail its
`Run Pre-commit Hooks` job. The checker tests also run through pre-commit when
files under `scripts/validate_api/` change.

The shared parser checks includes in C++ sources and headers for angle-bracket
spelling, approved standard headers and prefixes, banned heavyweight headers,
and the frozen UMD header allowlist. Unused allowed prefixes are also reported.
The legacy include-style skip list remains unchanged; it does not exempt headers
from the new guard and stability-boundary checks.

For all `.h`, `.hpp`, `.hh`, and `.hxx` files under `tt_metal/api/`, it also checks:

- Unconditional `#pragma once` before declarations and other directives.
- Direct literal includes that violate the table above.
- Headers or includes using the incorrect `tt-metalium/internal/` location.
- Nonliteral includes whose API tier cannot be checked lexically.

Comments and raw string contents do not count as directives. Continued lines,
quoted relative includes, and normalized paths are handled. Quoted paths are
resolved for boundary diagnostics even though include-style validation rejects
them. Every conditional branch is checked, regardless of the current host
architecture.

**Coverage limit:** this first checker does not invoke the C++ preprocessor or
walk a transitive include graph. It recognizes API-root and quoted relative
includes even when the target is absent. Existing targets take precedence in
include-search order; otherwise quoted paths are source-relative, except the
canonical `tt-metalium/` and `internal/` API-root spellings.
It cannot classify custom compiler include roots or
dependencies hidden in external/helper headers. It does not prove source or
binary compatibility. Compiler-based analysis is follow-up work.

### Existing include exceptions

`scripts/validate_api/header_hygiene_exceptions.json` records the five existing
forbidden include edges at initial adoption. Each entry names an exact source
and target, an owner, a reason, and a removal condition. The checker rejects
wildcards, duplicates, malformed entries, and stale exceptions. Remove an entry
in the same PR that repairs its include or removes its source header.

These entries preserve existing behavior during cleanup; they do not approve
new API exposure. Adding or changing an exception requires API-owner review,
enforced through CODEOWNERS on `scripts/validate_api/`. Do not add exceptions just
to make CI pass. Existing indirect exposure through an excepted edge still needs
review until transitive analysis is implemented.

## Same-PR deprecation requirements

The PR introducing a deprecation must add the replacement, migrate every
production usage in this repository, and annotate the old interface with a
removal notice and specific migration instructions. Do not add new usages of
already-deprecated interfaces.

Compatibility tests intentionally exercising the old interface and necessary
compatibility-shim implementation may have narrowly scoped exceptions and
warning suppressions. Broad suppression of deprecation warnings is not a
substitute for migrating callers.

Preserve the old interface for downstream consumers for at least four weeks
after the deprecation PR merges to `main`, and honor any longer promised
deadline. Remove it in a separate PR. Extend `.github/deprecations.json` and its
reaper as automated symbol-level validation is added. The lexical hygiene check
does **not** yet enforce deprecation requirements.

## Next automation increments

1. **Compiler-based hygiene:** resolved transitive include chains; exposed-type
   boundaries; standalone consumer compilation for each public header; global
   namespace pollution; and implementation placement. Ordinary runtime bodies
   belong in source files. Allow defaulted/deleted functions, trivial accessors,
   small forwarding wrappers, and definitions required for templates or constant
   evaluation. `inline` alone is not a reason to keep implementation in a header.
   Preserve deprecated global aliases until their compatibility window expires.
2. **Compatibility and deprecation:** compare compiler-derived API inventories
   for the base and proposed merge result; compile unchanged consumer fixtures;
   resolve usages of exact deprecated symbols/overloads; verify the introducing
   PR and merge time; and enforce same-PR migration and separate-PR removal.
   Report unsupported configurations explicitly. Define the ABI and supported
   release contract before adding binary-compatibility enforcement.
3. **AI-assisted resolution:** feed deterministic findings, implementation
   changes, tests, and approval evidence into an API skill that prepares fixes,
   migrations, documentation, and cleanup proposals. Produce one report with
   verified violations and specific unresolved decisions. AI cannot waive a
   deterministic failure or substitute for Runtime graduation/design approval.
4. **Selective specialist review:** validate against historical PRs and seeded
   violations, measuring missed breaks, false positives, and reviewer time.
   Change ownership/routing only after the checks demonstrate adequate coverage.
   A successful hygiene check alone does not establish that a PR needs no API
   specialist review.
