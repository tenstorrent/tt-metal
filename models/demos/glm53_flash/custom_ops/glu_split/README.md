# `glu_split` -- the shared expert's slices + clamped SwiGLU as one generic op (decode ledger C11)

`tt/mlp.py::TtClampedSwiGLU.forward` computes the fused gate|up matmul `gu = x @ [gate_c | up_c]` and then cut it
with two `ttnn.slice` ops before `ttnn.clamped_silu_glu` (3 ops, 1.6 + 1.6 + 4.8 us plus three launch gaps per
shared expert, `_logs/mesh/report_prof_ep32_ep_b4` ids 305-307). This op reads the gate tile `(r, c)` and the up
tile `(r, It + c)` straight out of `gu` and applies the fused SFPU `clamped_silu_glu_tile` (the same op the expert
stream kernels run in DST; limit 10 = `swiglu_limit`, compile-time in the LLK):

    h[t, c] = silu(min(gu[t, c], 10)) * clamp(gu[t, I + c], -10, 10)

| tensor | shape | dtype / layout | role |
|---|---|---|---|
| `gu` | `[1,1,T,2I]` (padded rows) | bf16 TILE interleaved (DRAM or L1) | the fused gate\|up matmul output |
| `out` | `[1,1,T,I]` | bf16 TILE interleaved, pre-allocated (`allocate_output`) | the activation |

Work unit = one output tile, `Rt x It` units dealt row-major over the compute grid (16 units on 16 cores for the
shared expert at decode: I = 512 per column; 96 units for the dense layers' 3072). Kernels: `reader_glu_split.cpp`
(NCRISC: the two tiles per unit), `compute_glu_split.cpp` (gate -> DST 0, up -> DST 1, `clamped_silu_glu_tile(0, 1, 2)`,
fp32 DST, pack), `writer_glu_split.cpp` (BRISC). Both data-movement kernels end with `noc_async_full_barrier()`.
Compile-time args: reader `[CB_G, CB_U, It, tile_bytes] + acc(gu)`, writer `[CB_OUT, tile_bytes] + acc(out)`,
compute `[CB_G, CB_U, CB_OUT]`; runtime per core `[start, count]` (compute `[count]`), common `[gu addr]` / `[out addr]`.

Knob: `GLM53_SHARED_GLU=fused` (T <= 32: the decode shapes) | `fused_all` (every T) | `composite` (default, today's
three ops). Numerics: the SFPU implementation of the activation is the expert kernels' (not the composite's
MINIMUM -> SILU -> CLAMP -> multiply chain): expect bf16-ulp-level differences, not bit identity.

Tests: `tests/test_moe_roofline_host.py` (golden, work split on 13x10 / 8x8, kernel ABI);
`tests/test_moe_roofline_mesh.py::test_glu_split_chip0` (device A/B vs the composite, PCC and max ulp).
