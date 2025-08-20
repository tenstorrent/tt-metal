# moreh/moreh_layer_norm_backward program cache review

Factories reviewed:
- input_grad: override guards optional gamma; indices match create-time.
- gamma_beta_grad: override guards optional outputs; potential minor typo: writer override checks `gamma_grad_buffer` twice (should check `beta_grad_buffer` for index 1).

Suggested fix:
- In `moreh_layer_norm_backward_gamma_beta_grad_program_factory.cpp`, line ~302: replace the second `if (gamma_grad_buffer != nullptr)` with `if (beta_grad_buffer != nullptr)` before writing index 1.

Repro/regression tests are covered in upstream unit tests; no cache-hit failure observed here.
