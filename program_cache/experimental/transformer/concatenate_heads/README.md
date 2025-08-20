### concatenate_heads program cache review

- OP: `ttnn/cpp/ttnn/operations/experimental/transformer/concatenate_heads`

Findings:
- Per-core reader/writer runtime args include only source and destination buffer base addresses and tile-id offsets derived from shapes. Override updates base addresses per core; hashed properties (batch size, layout, shapes) remain constant across cache hits.

No issues identified.
