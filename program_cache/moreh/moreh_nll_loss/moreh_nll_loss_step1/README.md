# moreh/moreh_nll_loss/moreh_nll_loss_step1 program cache review

- Override updates `target`, `weight`, `output` addresses; scalar args like `ignore_index`, `channel_size` are hashed and constant on cache-hit.
- Reader/writer arg indices align with create-time ordering; no optional deref issues.

No cache-hit issues identified. Reviewed and marked as reviewed.
