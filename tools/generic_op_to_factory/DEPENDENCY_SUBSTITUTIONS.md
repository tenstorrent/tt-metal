# Explicit substitutions for missing shared headers

The default flow requires recorded dependencies. If an in-run canonical helper
was not ingested, do not invent its contents or silently borrow another run's
version. First attempt deterministic recovery of the original. A user may then
authorize a specifically audited substitute, accepting that the runtime is no
longer an exact historical reproduction.

This optional mechanism only **adds missing headers** below
`ttnn/cpp/ttnn/kernel_lib/`. It refuses tracked destinations, existing files,
symlinks, duplicate destinations and path traversal. It does not replace an
existing helper, modify frozen exports/preparations, or relax golden, cache or
independent-review gates.

1. Export and verify the donor run with the normal read-only exporter. Keep that
   export separate from the candidate's original export and validation workspace.
2. Audit the donor header, its includes, macros, relevant compile-time defines
   and numerical/profiling behavior against the candidate's actual consumers.
3. Obtain explicit user approval for the substitution and its limitations.
4. Supply the following list as `dependency_substitutions` in both baseline and
   native-validation JSON configurations. Use a new evidence workspace.

```json
[
  {
    "export": "/absolute/verified-donor-export",
    "source_path": "source/kernels/shared_helper.hpp",
    "destination": "ttnn/cpp/ttnn/kernel_lib/shared_helper.hpp",
    "approval": "Record the actual user authorization",
    "reason": "The selected run did not preserve this shared dependency",
    "limitations": "Describe audited behavior differences and unverified historical identity"
  }
]
```

For target installation, save the same list in a JSON file and pass
`--dependency-substitutions /absolute/substitutions.json` to
`python3 -m tools.generic_op_to_factory.prepare_target`. The baseline driver
installs approved additions during its `install` checkpoint. Direct
`prepare_baseline` remains strict and does not install substitutions.

The source must be a materialized `.h`, `.hpp` or `.inl` file in a verified donor
export. Plans and installation receipts record its DB identity, run/table/row,
original name, exact hash, donor snapshot hash, destination, approval and
limitations. Approval prose is an attestation, not authenticated authorization;
the initiating agent/user remains responsible for actual consent.

Resume re-verifies donor exports and installed bytes. Changing either, changing
the approval/configuration, or changing implementation requires a new evidence
workspace. Interrupted partial installs remain visible; do not delete receipts
or overwrite files to force a retry.

Baseline state/comparison and native completion evidence retain an explicit
`baseline_scope` saying that substituted dependencies are **not exact historical
runtime reproduction**. Matching golden outcomes does not prove dependency or
profiling identity. Historical failures remain failures; substitution approval
does not authorize hiding them, weakening tests, enabling new defines, or making
unmeasured performance claims.
