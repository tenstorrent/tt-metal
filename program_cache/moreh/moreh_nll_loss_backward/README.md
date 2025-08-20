# moreh/moreh_nll_loss_backward program cache review

Variants: 2D, 3D, 4D reader paths
- Override updates target/output_grad/weight/divisor/input_grad buffer addresses; scalar rt args (ignore_index, sizes) remain constant and hashed.
- Indices align with create-time ordering across variants.

No cache-hit issues identified. Reviewed and marked as reviewed.
