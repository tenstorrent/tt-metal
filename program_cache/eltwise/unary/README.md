Unary OP program cache review

Summary

- Reviewed `ttnn/cpp/ttnn/operations/eltwise/unary` (interleaved and sharded program factories).
- No cache-hit issues found. Overrides correctly update input/output buffer addresses; sharded variant updates CB dynamic addresses.

Key observations

- Interleaved: `UnaryProgramFactory::override_runtime_arguments` iterates cores and updates reader/writer arg[0] with fresh buffer base addresses while leaving hashed compile-time args intact.
- Sharded: `UnaryShardedProgramFactory::override_runtime_arguments` calls `UpdateDynamicCircularBufferAddress` for input/output CBs each run.

Conclusion

- Cache overrides are complete and aligned with kernel arg order established at creation. Marked reviewed with no issues.
