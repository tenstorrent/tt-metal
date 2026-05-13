# Why `matmul_block` and `matmul_kloop_pack` Are Kept Separate

## Bottom line

Merging them into one C++ function is mechanically possible. We chose not to because the two helpers encode **different contracts about who owns the outer loop**, and a merged helper would surrender the contract narrowness that gives `compute_kernel_lib` helpers their safety value. The decision belongs in the "could merge → would harm callers" category, not the "physically impossible" category.

## The merge question collapses into three positions:

1. **Cannot be merged at all.** False. One C++ function with a `bool helper_owns_outer_loop` template parameter could subsume both. The shared FMA atom is already factored out as `detail::MatmulSubblockStep` — a single `ckernel::matmul_block` call plus hygiene — so sharing at *that* level is already done.
2. **Could merge, but the result would be aesthetically ugly.** Partially true. The merged `.inl` would be two disjoint code paths under one `if constexpr`, sharing only the FMA primitive. But aesthetics is the weaker objection.
3. **Could merge, but doing so widens each helper's contract past the point where it can absorb caller mistakes — defeating the purpose of having helpers at all.** This is the load-bearing reason.

## The two contracts, in pseudo code

Both helpers ultimately call `ckernel::matmul_block`. What differs is what each helper takes ownership of.

### `matmul_block` — helper owns the K-loop end to end

Used by 11 callers (standard matmul, SDPA, conv2d, DRAM-sharded, etc.). The caller hands the helper a shape and gets a finished matmul.

```
matmul_block(in0, in1, out, interm, shape, ...):
    init_matmul_state()
    for batch in shape.batch:
        for m_sb in shape.in0_num_subblocks:           # M sub-block sweep
            for n_sb in shape.in1_num_subblocks:       # N sub-block sweep
                tile_regs_acquire()                     # === DST OPEN per output sub-block
                for k in shape.num_k_blocks:            # K-loop
                    in0.wait_front(); in1.wait_front()
                    if k > 0 and not packer_l1_acc:
                        reload interm -> DST
                    matmul_block_FMA(in0, in1) -> DST   # fixed FMA call
                    if k < last and not packer_l1_acc:
                        spill DST -> interm
                    in0.pop_front(); in1.pop_front()
                tile_regs_commit(); tile_regs_wait()
                out.reserve_back(); pack DST -> out; out.push_back()
                tile_regs_release()                     # === DST CLOSE
```

### `matmul_kloop_pack` — caller owns the outer loop, helper owns only DST + segmented K-loop + pack scope

Used by 4 ring-aware MoE / DeepSeek MLA callers (`mla_wo`, `moe_gate_mm`, `moe_compute`, `moe_gpt`). The caller supplies a per-step functor and a pack body; the helper supplies the DST scope and the in1 wait/pop skeleton.

```
# Caller's outer loop:
for ring_position / chunk / etc.:
    caller_prework()
    k_step = KStepWithRing{...}                         # mutable functor with caller state
    matmul_kloop_pack(in1, shape, k_step, pack_body, post_k_body)
    caller_postwork()

matmul_kloop_pack(in1, shape, k_step, pack_body, post_k_body):
    tile_regs_acquire()                                 # === DST OPEN once for whole K-loop
    for block in shape.num_blocks:
        in1.wait_front(tiles_per_block)
        for k in 0..tiles_per_block step ct_dim:
            k_step(...)                                 # caller defines what one step means:
                                                        #   - FMA from in0
                                                        #   - bias FMA from ones_buf
                                                        #   - padding skip (no FMA)
                                                        #   - pop+wait on a third CB then FMA
        in1.pop_front(tiles_per_block)
    post_k_body()                                       # optional MATH-thread SFPU on accumulator
    tile_regs_commit()
    pack_body()                                         # caller-defined; multi-CB OK,
                                                        # pack-thread SFPU OK, owns tile_regs_wait
    tile_regs_release()                                 # === DST CLOSE
```

The structural differences worth noting:

- **DST scope.** `matmul_block` opens DST per output sub-block — many DST cycles per call. `matmul_kloop_pack` opens DST once per call.
- **CB rhythm.** `matmul_block` has one in0 + one in1 wait/pop per K-block, paired and homogeneous. `matmul_kloop_pack` has in1 wait/pops at a coarser granularity (per outer block, not per FMA step), touches in0 only through the caller's functor, and may pop/wait on a third CB mid-K from inside the functor.
- **Inner body.** `matmul_block`'s inner FMA is one fixed call; per-K-block hooks adjust *indices* only. `matmul_kloop_pack`'s inner body is itself a parameter — an iteration can be a regular FMA, a bias FMA from a different CB, a padding skip, or a CB-pop-and-wait followed by an FMA.

## What merging would cost on the caller-facing surface

A merged helper would force every existing parameter into a "meaningful in mode A / meaningful in mode B / meaningful in both" classification:

| Parameter | Meaningful in `matmul_block` mode | Meaningful in `kloop_pack` mode |
|---|---|---|
| `MatmulBlockShape.num_k_blocks` | Yes — one K-tile per block | No — superseded by `num_blocks × tiles_per_block` |
| `SegmentedKLoopShape.num_blocks` | No — implicitly 1 per K-block | Yes — in1 wait/pop granularity |
| `KBlockInnerDimFn` / `In0SourceFn` / `In1BaseOffsetFn` | Yes — index shifts | No — per-step functor already controls this |
| `LastBlockTarget` / `pack_relu` / `pin_interm_to_captured_base` / `caller_owns_pack_target` | Yes — helper owns the pack | No — `pack_body` owns the pack |
| `In0Policy` / `In1Policy` | Yes — at K-block granularity | Partial — different rhythm; in0 not touched by helper |
| `init_mode` | Yes — helper issues `mm_block_init` | No — caller's outer init covers state |
| `k_step` functor | No — FMA is fixed | Yes — required |
| `pack_body` functor | No — helper packs | Yes — required |
| `post_k_body` | No | Yes — optional MATH-thread SFPU slot |

Roughly a third of the parameters end up as "only meaningful when other parameters are set to X" — meaning the helper's `ASSERT`s can no longer be tight (they must permit both modes' shapes), the docstring grows to cover the cross-product, and every caller has to mentally branch on mode before evaluating relevance.

## Why this matters for the helper library specifically

`docs/kernel-helpers.md` is explicit that wrong policy-enum selection is the dominant helper-misuse failure mode today — the wrong enum compiles cleanly but produces silent corruption or a CB-wait hang. The whole point of the helper library is to absorb that surface (CB ordering, DST capacity, init state, CB aliasing) so that kernels built from helpers tend to work the first time.

A helper's safety value is exactly proportional to the narrowness of its contract:

- `matmul_block`'s narrow contract — "I own the K-loop; you give me a shape and CBs" — lets it absorb the spill/reload / L1_ACC / init-dispatch / pack-target-dispatch / per-K-block-hook surface. Eleven callers don't have to reason about any of it.
- `matmul_kloop_pack`'s narrow contract — "You own the outer loop and the inner step; I own the DST scope and the in1 skeleton" — lets it absorb the DST-scope / segmented-K-loop / pack-scope surface for callers whose outer loop is structurally non-rectangular (ring-aware, semaphore-interleaved, multi-CB-pack).

A merged helper would absorb neither, because its contract would no longer be narrow enough to enforce either invariant.
