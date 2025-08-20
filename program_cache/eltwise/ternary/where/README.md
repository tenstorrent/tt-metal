Where (ternary) OP program cache review

Summary

- Reviewed `ttnn/cpp/ttnn/operations/eltwise/ternary/where` program cache behavior for TTT, TTS, and TST variants.
- No cache-hit issues found. Overrides update predicate/value tensor buffer addresses, output addresses, per-core tile counts and start ids, and scalar runtime args for mixed tensor-scalar cases.

Key observations

- Program factory stores reader/writer/compute kernel ids and grid size in `shared_variables`.
- `WhereProgramFactory::override_runtime_arguments(...)` reuses `set_or_update_runtime_arguments(...)` with an updater to overwrite per-core runtime arg vectors in-place.
- Variant-specific argument ordering preserved across create/override:
  - TTS: reader args `[pred_addr, value_true_addr, 0, tiles, start_id]`; compute includes packed `value_false_scalar`.
  - TST: reader args `[pred_addr, value_false_addr, 0, tiles, start_id]`; compute includes packed `value_true_scalar`.
  - TTT: reader args include both tensor addresses; compute has only `[tiles]`.

Conclusion

- Cache-hit path mirrors creation-time argument layout and updates all runtime-only values. Marked as reviewed with no issues.
