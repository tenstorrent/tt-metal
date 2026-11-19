# Before-run source comparability

The initial `before_l{0,1,3}` runs use the exact accepted stage4 decoder source.
Before the real-fixture and length pairs, the shared-expert expression was extracted
unchanged into `MeshMoe.shared_forward` to permit candidate replacement. The same
operations, weights, precision, collective placement and return order remain.
This adds one Python helper call to eager execution and no device operation to
traced decode. No optimized kernel or projection was selected for these runs.

`before_source_delta.patch` reconstructs that entire source change from accepted
HEAD79582226. Its source digest matches the already-recorded real-before provenance:
`e4f63391a796313ae2d1622820476c3002e0e7c65f7fa1fdc0f8c2a94fc54c46`. This was independently reconstructed from the accepted Git object
and the recorded 20:38:37 source edit; it is not inferred from similar timings.
Final real-fixture comparisons use the same logical workload and references.
