# MakeProgramFromSpec output tests

These tests check what `MakeProgramFromSpec` (declared in `program.hpp`) produces from a valid spec: lowered
hardware configs, kernel argument layouts, resource slots and scopes. They inspect the Program's internals through
`program.impl()` on a mock device. Whether a spec is accepted at all is tested in
[`../invariant_tests/`](../invariant_tests/).
