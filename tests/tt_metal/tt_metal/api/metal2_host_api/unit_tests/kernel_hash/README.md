# Kernel hash tests

Kernels whose `compute_hash()` is equal share one JIT-cached binary. A spec change that alters the generated code or
compile-time arguments must change the hash, or a stale binary runs; a change supplied only at run time must not.
Each test builds Programs from specs that differ in one field and compares the kernels' hashes, with one file per
struct whose field is varied.
