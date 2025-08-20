Moreh Softmax Backward — Program Cache Review

- Status: Reviewed — no program-cache issues found (representative H-small variant inspected)

Summary
- Reader args: output, output_grad base addresses then per-core counts/offsets, Ht/Wt, scaler, mask. Writer args: input_grad base address and tile metadata.
- Override updates DRAM base addresses for both reader (two inputs) and writer (output) across all cores.

Key references
- Factory create: `ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/softmax_backward_h_small/softmax_backward_h_small.cpp`
  - Reader arg order (output addr, output_grad addr, ...): around L137–L145.
  - Writer arg order (input_grad addr, ...): around L147–L151.
- Cache-hit override updates:
  - Reader base addresses updated: around L171–L174.
  - Writer base address updated: around L176–L178.

Notes
- Compute args and constants are tied to hashed attributes; only base addresses vary run-to-run and are correctly overridden.
