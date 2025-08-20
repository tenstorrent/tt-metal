Upsample operations program cache review

Status: Reviewed — no program cache issues found.

Reviewed files
- `device/upsample_program_factory_multicore_interleaved.cpp`
- `device/upsample_program_factory_multicore_sharded.cpp`
- `device/upsample_bilinear_program_factory_multicore.cpp`

Findings
- Interleaved: override updates reader arg[0] (src base) and writer arg[0] (dst base) per core.
- Sharded: runtime addresses are bound via dynamic CBs; override updates `cb_src0` and `out_cb` to current buffers.
- Bilinear sharded: override rebuilds halo input and updates `cb_src0` and `out_cb` dynamic CB addresses accordingly.
- All compile-time args (scales, shapes, layout, tiling, block sizes) are selected at creation and are stable across cache hits for the same hashed attributes.

Recommendation
- No changes required.
