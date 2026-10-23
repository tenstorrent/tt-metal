# MakeProgramFromSpec output tests

What `MakeProgramFromSpec` builds from a valid spec: lowered hardware configs, kernel argument layouts, resource
slots and scopes. The tests inspect the Program through `program.impl()` on a mock device. Whether a spec is
accepted at all is tested in `../invariant_tests/`.
