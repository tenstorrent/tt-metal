# moreh/moreh_linear_backward (bias_add_backward) program cache review

Finding:
- Single-core and multi-core factories only override buffer addresses for reader[0] and writer[0]. No scalar runtime args changed on cache-hit. Indices align with create-time ordering.
- No cache-hit bugs identified.

No failure tests added. Reviewed and marked as reviewed.
