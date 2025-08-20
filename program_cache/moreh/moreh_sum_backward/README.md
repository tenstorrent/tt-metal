Moreh Sum Backward — Program Cache Review

- Status: Reviewed — no program-cache issues found

Summary
- Reader args: output_grad base address then per-core counts/offset/shape dims for broadcasting. Writer args: input_grad base address and tile metadata.
- Override updates reader[0] and writer[0] with new buffer base addresses across cores.

Key references
- Factory create: `device/moreh_sum_backward_program_factory.cpp` around L206–L218.
- Cache-hit override updates: around L244–L251.

Notes
- All per-core tile counts/offsets/dims are derived from hashed shapes and remain stable across cache hits.
