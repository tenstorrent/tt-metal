# MakeProgramFromSpec output tests

What `MakeProgramFromSpec` builds from a valid spec: lowered hardware configs, kernel argument layouts, resource
slots and scopes. The tests inspect the Program through `program.impl()` on a mock device. Whether a spec is
accepted at all is tested in `../invariant_tests/`.

`scratchpad_allocations.cpp` covers placement overrides, shared allocation and CRTA delivery,
including allocation on selected derived nodes, excluded nodes, multi-binding combinations,
L1 limits and address updates after DFB resizing.

`dfb_placement.cpp` covers slot assignment from inferred placement and backing-memory allocation from
placement overrides. It checks separate slots for shared endpoint kernels with matching or different
metadata, slot reuse for disjoint endpoint nodes, alias allocation, and memory available to scratchpads.
