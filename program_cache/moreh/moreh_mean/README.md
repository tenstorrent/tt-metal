# moreh/moreh_mean program cache review

Factories: H, W, and NC variants
- Override updates only input/output buffer addresses; all scalar rt args remain constant on cache-hit.
- Indices match create-time ordering across variants.

No cache-hit issues identified. Reviewed and marked as reviewed.
