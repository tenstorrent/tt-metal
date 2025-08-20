Moreh Sum — Program Cache Review

- Status: Reviewed — no program-cache issues found (NC/H/W variants inspected)

Summary
- All variants set reader args with input base address and per-core indices/counts plus masks where applicable; writer args set output base address and per-core metadata.
- Overrides for NC/H/W factories update reader[0] and writer[0] with new buffer addresses per core; other runtime args remain derived from hashed shapes/dims.

Key references
- H variant create/override: `device/moreh_sum_h_program_factory.cpp` around L206–L224 (create) and L249–L257 (override).
- W variant create/override: `device/moreh_sum_w_program_factory.cpp` around L205–L224 (create) and L248–L255 (override).
- NC variant create/override: `device/moreh_sum_nc_program_factory.cpp` around L154–L169 (create) and L191–L199 (override).

Notes
- Scalars/masks are constants for a given hash; only DRAM base addresses vary and are correctly overridden.
