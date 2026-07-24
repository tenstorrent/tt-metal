# Qwen3-8B BFP8 Prefetcher Corruption — LLK Analysis

**Reviewer:** Staff LLK engineer (Tenstorrent)
**Customer report:** `predator2k/sglang@tenstorrent-p1` / `tt_qwen3_8b_prefetcher_UPSTREAM_BUG_REPORT_2026-05-26.md`
**Customer fork:** `predator2k/tt-metal@tenstorrent-p1` (`b723bd4648c`)
**Target HW:** Blackhole P150a (mesh 1×2), FW 19.6.0.0
**Workload:** `MatmulMultiCoreReuseMcast1DProgramFactory::process_gather_in0` with `use_global_cb=true`, BFP8 weights via `dram_prefetcher` → `GlobalCircularBuffer`

---

## TL;DR

**The bug is not in LLK.** The customer's pin on `llk_unpack_AB_matmul.h:333-336` is a mislocalization: that branch emits a single data‑format‑agnostic `TTI_UNPACR(SrcA, …)` whose behavior is wholly determined by THCON CFG state — which the customer has already verified is correct (U43). The same code path also serves the BFP4 case that works.

The most plausible failure surface is **above LLK**: in the program‑factory ↔ JIT‑build ↔ producer‑side CB metadata plumbing for the global‑CB path. I found one concrete latent bug in that surface and pushed a candidate fix.

## What I ruled in / out on the LLK side

| Customer hypothesis | Verdict | Reasoning |
|---|---|---|
| `_llk_unpack_AB_matmul_` (BH) full‑face full‑tile branch is BFP8‑broken | **Ruled out** | `llk_unpack_AB_matmul.h:333‑336` is a single `TTI_UNPACR(SrcA, 0, …)`. The encoded instruction carries no data‑format/geometry/stride. All format state lives in THCON CFG (`Out_data_format`, `TileDescriptor`) — which the customer reports is correctly programmed in both the BFP4 (working) and BFP8 (broken) cases. There is no kernel‑side way for this branch to discriminate by data format. |
| MOP replay buffer tile‑stride mismatch | **Ruled out kernel‑side** | The MOP loop reads `THCON_SEC0_REG3_Base_address`, adds the `TILE_SIZE_A` GPR, and writes it back per iteration. `TILE_SIZE_A` is programmed by `_llk_unpack_hw_configure_` (llk_unpack_common.h:60) from `get_local_cb_interface(unpA_id).fifo_page_size`, which is the consumer CB's `set_page_size(in1_single_tile_size)`. For BFP8 this is 1088 (fits in the 16‑bit `LOWER_HALFWORD` slot), for BFP4 576 — both correct. No stale‑GPR scenario unless an earlier program left it dirty in the same trace, and the reproducer parameterizes BFP4/BFP8 across separate processes. |
| ADC counter leakage between tiles | **Unlikely** | `_llk_unpack_AB_matmul_init_` does `TTI_SETADCZW(0b011, 0,0,0,0, 0b1111)` and either `TT_SETADCXX(UNP_A, …)` or `config_unpacker_x_end<UNP_A>()` on every init. Full‑face path also issues `TTI_SETADCZW(UNP_A, …, 0b0101)` in the partial‑face leg. State is reset per init. No data‑format dependence. |
| MOP replay buffer captured wrong tile geometry from a previous ELF | **Inapplicable on BH** | The BH MOP path uses `load_replay_buf(…, lambda …)` which records into a dedicated replay buffer each time `_llk_unpack_AB_matmul_mop_config_` runs (i.e. once per `mm_block_init`). There is no cross‑program persistence. (WH uses `lltt::record`; same lifecycle.) |
| BH vs WH `WRCFG` race vs WH's `REG2FLOP` | **Possible but unproven; not the simplest explanation** | BH replaces WH's single `TTI_REG2FLOP` with `TTI_STALLWAIT(STALL_CFG, THCON) + TTI_WRCFG`. The stall is correct in placement and meaning. Would equally affect BFP4. |

Bottom line: every LLK‑level instruction emitted on this path is format‑blind, fed by CFG/GPR state that is either trivially correct or already verified correct by the customer's own probes. There is no kernel‑side switch I can find that the format would tickle.

## The latent bug I did find (one layer above LLK)

In **both** `MatmulMultiCoreReuseMcast1DProgramFactory::process_gather_in0_program_and_create_override_variables` and `process_agmm_fusion_program_and_create_override_variables` (the llama_1d_mm_fusion AGMM variant), the `use_global_cb=true` branch sets the **local** CB index of the remote/global CB pair without `.set_tile_dims()`:

```cpp
// matmul_multicore_reuse_mcast_1d_program_factory.cpp, around line 2091 (main)
remote_cb_config.remote_index(remote_cb_index)
    .set_page_size(in1_block_size_bytes)
    .set_data_format(in1_data_format);
remote_cb_config.index(src1_cb_index)
    .set_page_size(in1_single_tile_size)
    .set_data_format(in1_data_format);
//  ^^ missing .set_tile_dims(in1_tile)  — non-global path 30 lines below DOES set it
```

`ProgramImpl::set_cb_tile_dims` (`tt_metal/impl/program/program.cpp:1476`) only emits the per‑CB `unpack_tile_r_dim` / `unpack_tile_c_dim` / `unpack_num_faces` / `unpack_partial_face` / `unpack_face_r_dim` / `unpack_narrow_tile` constexpr arrays into the JIT genfiles when `circular_buffer->tile(buffer_index).has_value()`. Without `set_tile_dims`, it falls through to the `else` branch (program.cpp:1495) and only sets `buf_tile_size` (correctly, from the CB's data format). The geometry arrays stay at the `tt_hlk_desc` constructor defaults (`hlk_desc.hpp:43`): 32×32, 4 faces, face_r_dim=16, partial_face=0, narrow_tile=0.

For the customer's exact repro (32×32 4‑face BFP8 weight tile) **those defaults happen to match** the actual tile geometry — so this is a latent rather than immediate bug for this specific case. But it is a real divergence between the global and non‑global paths, it breaks any non‑standard tile (transposed faces, partial face, 16×32), and it surfaces precisely the kind of "U41 metadata looks fine" symptom the customer reports — because metadata *does* look fine when the defaults align.

**Fix pushed:** `origin/ncvetkovic/qwen3_bfp8_global_cb_tile_dims_fix` — propagates `in1_tile` on both factories. 2 files, 8 insertions, 2 deletions. Worth landing for correctness regardless of whether it moves the needle on this specific report.

```
https://github.com/tenstorrent/tt-metal/pull/new/ncvetkovic/qwen3_bfp8_global_cb_tile_dims_fix
```

I want to be clear: **I do not believe this fix is the root cause of the customer's BFP8 corruption.** Defaults match the test's tile geometry, and the customer's U41 probe confirms metadata is correct. I am pushing it because it is a real bug found during this investigation and a free correctness win.

## Where I think the actual bug is

The corruption pattern — magnitudes 2^60–2^109, BFP8‑only, byte‑exact L1 contents, correct CFG, correct CB metadata — is *not* mantissa corruption. Those magnitudes are the signature of **exponent corruption**: the unpacker is reading the per‑face 8‑bit shared exponent from the wrong byte offset, so each datum's mantissa gets paired with a wildly off scale.

For BFP8 on Blackhole the per‑tile layout is `aligned_exp_size = round_up(face_r_dim × num_faces, L1_ALIGNMENT) = round_up(64, 64) = 64` bytes of exponents followed by 1024 bytes of mantissas. For BFP4 the mantissa block halves (512 B) but the exponent block stays 64 B. The unpacker locates the exponent block via `THCON_SEC0_REG0_TileDescriptor` + tile_size (`fifo_page_size`).

This focuses suspicion on three places I'd ask the customer to instrument before suspecting LLK silicon:

1. **Producer‑side L1 layout for BFP8 through `dram_prefetcher`.** U37 demonstrates byte‑equality between producer write and consumer read — that proves nothing about *layout*. If the prefetcher's `dram_prefetcher_program_factory.cpp` chooses `max_tile_size_df` and `max_tile_size` over the tensor set (lines 100–103), and any tensor in the set has a different dtype or tile shape from `in1` on the consumer, then bytes can be byte‑identical between writer and reader yet the exponent block sits at the wrong offset relative to the consumer's mantissa block. Worth dumping the actual L1 contents of one full tile and asking "is the exp block where THCON `TileDescriptor` says it is?" — not "do the bytes match what the producer wrote".

2. **`Tile::get_tile_size` ↔ `BFLOAT8_B_TILE_HW` divergence.** `Tile::get_tile_size` (`tt_metal/impl/data_format/tile.cpp:70`) computes `aligned_exp_size = round_up(face_shape[0] × num_faces, l1_alignment)`. On BH `l1_alignment=64`, BFP8 → 1088 ✓, BFP4 → 576 ✓. But the `tt_hlk_desc` constructor's `buf_tile_size_arr` default is unconditionally `BFLOAT8_B_TILE_HW = 1088`. Anywhere the BFP4 path inherits the default and the BFP8 path computes it explicitly (or vice versa), you can get an asymmetric, dtype‑specific corruption that only fires in one direction. Worth grep'ing the customer's stack for `buf_tile_size_arr` reads that bypass `set_buf_tile_size`.

3. **`Prefetcher` Python orchestration on `tenstorrent-p1`.** The repro test uses `models/tt_transformers/tt/prefetcher.Prefetcher`. If that class caches a `GlobalCircularBuffer` sized for one dtype and reuses it across pytest parameterizations, or sizes the CB by the *first* tensor's tile size rather than the per‑tensor `in1_single_tile_size`, you get a strict subset of "bytes look fine, layout doesn't". The fact that BFP8 specifically (the larger tile) breaks and BFP4 (the smaller) passes is also consistent with a CB sized to the smaller and overflowing the larger — but the customer would have caught that.

## Cross‑check vs working Llama‑3 70B Galaxy

The customer's report notes Llama‑3 70B works on Galaxy with BFP8 + the same allocation pattern. Galaxy = Wormhole. Qwen3‑8B fails on Blackhole.

Two arch deltas that change between WH and BH on this exact path:

- **L1 alignment**: WH=16, BH=64. Affects `aligned_exp_size` rounding (no effect for 4×16 = 64, which is already aligned both ways) but does affect other sizes if num_faces or face_r_dim differ.
- **MOP base‑address update**: WH uses `TTI_REG2FLOP`; BH uses `TTI_STALLWAIT(STALL_CFG, THCON) + TTI_WRCFG`. Functionally equivalent; not data‑format‑dependent.

Neither delta is a smoking gun for BFP8‑only corruption. This corroborates "not LLK" — if it were LLK, Llama‑3 70B BFP8 would be broken too, and the BH‑specific WRCFG path would have to discriminate by data format somehow, which it doesn't.

## Verification status in tt‑metal `work2`

I did not run the test — it requires a Blackhole 1×2 mesh I don't have access to. I did:

- Audit `tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h` (entire file)
- Audit `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_unpack_AB_matmul_api.h`, `llk_unpack_common_api.h`
- Audit `tt_metal/impl/buffers/circular_buffer_config.cpp`, `tt_metal/impl/program/program.cpp` (set_cb_tile_dims)
- Audit `tt_metal/jit_build/jit_build_options.cpp`, `hlk_desc.hpp`
- Audit `matmul_multicore_reuse_mcast_1d_program_factory.cpp::process_gather_in0` and `llama_1d_mm_fusion.cpp::process_agmm_fusion`
- Audit `ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/dram_prefetcher_program_factory.cpp`
- Diff WH vs BH `llk_unpack_AB_matmul.h`

The fix branch is at `origin/ncvetkovic/qwen3_bfp8_global_cb_tile_dims_fix`. It applies cleanly on top of `origin/main` (`d77027f0134`).

## Recommended next steps for the customer

1. **Stop bisecting `tt-llk` for this**, especially the Feb–Mar 2025 CFGSHIFTMASK PRs. The LLK on the BFP8 path is silently format‑agnostic — there is nothing in those PRs that would have asymmetric BFP4/BFP8 effects on `process_gather_in0`'s in1/SrcA path. Bisect time would be better spent elsewhere.
2. **Add a probe of *layout*, not bytes.** U37 proves NoC delivery is byte‑exact. The next test should: (a) dump 1 BFP8 weight tile from L1 as raw 1088 bytes, (b) compare the *first* 64 bytes against the host‑side reference's exp block, (c) compare bytes 64–1087 against the host‑side mantissa block — separately. If those pass independently but the unpacked SrcA values are still nonsense, the bug is in CFG `TileDescriptor`/`Out_data_format` programming order, not in layout. If (a) or (b) fails, the bug is producer‑side.
3. **Pull the fix on `origin/ncvetkovic/qwen3_bfp8_global_cb_tile_dims_fix`** as a baseline. Even if it doesn't fix the corruption, it removes one variable from the diff.
4. **Disassemble the MOP replay buffer at runtime** as the customer's report already suggested — but specifically capture `TILE_SIZE_A` GPR value at the instant the failing matmul runs, not from a separate probe. The kernel can `TTI_STOREREG` it to L1 just before `TT_MOP(0, …)` and the host can read it back.
5. **If Llama‑3 70B Galaxy is genuinely working** with BFP8 + `process_gather_in0` + global CB, run a side‑by‑side diff of:
   - `in1_tile.get_tile_shape() / get_face_shape() / get_num_faces() / get_partial_face() / get_narrow_tile() / get_transpose_of_faces()` for both
   - `in1_single_tile_size` and `in1_block_size_bytes`
   - `global_cb.size()` / `num_global_cb_receivers` / `in0_block_w` / `per_core_N`
   - Compute‑kernel defines (`PACKER_L1_ACC`, `FP32_DEST_ACC_EN`, throttle level, `dst_full_sync_en`)
   - `MathFidelity`

The customer's report attributes Galaxy‑works to the dtype path being correct. Equally plausible: it's a `dst_full_sync_en + packer_l1_acc + LoFi` interaction that only manifests with the BH dest accumulator quirks. That's not LLK either, but it'd be a much shorter bug.

---

## Addendum: cross‑check against `tt-isa-documentation`

After writing the analysis above I re‑verified each LLK / Tensix ISA claim against the local copy of `tt-isa-documentation` (Blackhole A0 and Wormhole B0 trees). Refs are repo‑relative paths.

| Claim in this report | ISA source | Verdict |
|---|---|---|
| `TTI_UNPACR(SrcA, 0, …)` carries no data‑format / geometry in its encoding; all behavior is driven by THCON CFG state | `WormholeB0/TensixTile/TensixCoprocessor/UNPACR_Regular.md` (BH file is a stub pointing at WH); lines 62, 84–101 — `InDataFormat`, `OutDataFormat`, `DatumSizeBytes`, `XDim`/`YDim`/`ZDim`/`WDim` all read from `ConfigState.THCON_SEC[…]` not from the instruction | **Confirmed** |
| BFP8 → BF16 and BFP4 → BF16 share the same FPU path under `MathFidelity::LoFi` | `WormholeB0/TensixTile/TensixCoprocessor/SrcASrcB.md:96‑97` — fidelity table. BFP8 *can* use 2 phases at full precision, but LoFi = phase 0 only for both | **Confirmed** |
| The BH MOP body `TTI_RDCFG → TTI_ADDDMAREG → TTI_STALLWAIT(STALL_CFG, THCON) → TTI_WRCFG → TTI_NOP` is correct ordering and is data‑format‑independent | `BlackholeA0/TensixTile/TensixCoprocessor/STALLWAIT.md` — WRCFG blocks under B7 (STALL_CFG), `THCON` is C0 ("ThCon has outstanding requests"). `WRCFG.md` line 3 — "behavior is identical between Wormhole and Blackhole". `ConfigurationUnit.md` line 19, 24, 35 — WRCFG is 2 cycles, must be followed by a cycle of separation before the new value is consumed (matches the trailing `TTI_NOP`) | **Confirmed** |
| WH's `TTI_REG2FLOP` and BH's `TTI_WRCFG` write to the same THCON CFG storage; routing differs (ThCon vs Config Unit) | `WormholeB0/TensixTile/TensixCoprocessor/REG2FLOP_Configuration.md` line 3 — "In most cases, the `WRCFG` instruction should be used instead, unless explicitly trying to avoid contention around the Configuration Unit." Same `ThConCfgBase` storage written by both | **Confirmed**; report wording is correct ("functionally equivalent"), routing nuance noted |
| MOP replay buffer cannot "capture geometry from a previous ELF" | `WormholeB0/TensixTile/TensixCoprocessor/MOPExpander.md` — the expander has no model of data format or tile geometry; it is a pure instruction macro. Replay buffer holds the literal instructions recorded by `lltt::record` / `load_replay_buf` at init. Customer's hypothesis #3 in the original report is impossible at the hardware level | **Customer hypothesis disproven** |
| `TILE_SIZE_A` GPR is loaded from CB `fifo_page_size` in `_llk_unpack_hw_configure_` and is the sole stride source the MOP body uses for advancing `THCON_SEC0_REG3_Base_address` | `tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_common.h:60`; `llk_unpack_AB_matmul.h:64` & `:94`. No ISA contradiction | **Confirmed** |
| BFP8 always reads the exponent section regardless of `NoBFPExpSection`; BFP4 only reads it when `NoBFPExpSection=false` | `WormholeB0/TensixTile/TensixCoprocessor/UNPACR_Regular.md:126‑133` — `if (InDataFormat == BFP8 ‖ InDataFormat == BFP8a ‖ !ConfigDescriptor.NoBFPExpSection)` | **Confirmed** — and notably this is the *one* place in the unpack pipeline where BFP4 and BFP8 take asymmetric branches. If `NoBFPExpSection` were wrongly set on the BFP8 path, BFP8 would still read the exp section (no change), but the *advance* of `InAddr` past the exp section is gated on the same condition — so if for some reason `NoBFPExpSection=true` *and* the layout was BFP4 with no exp section, BFP8 would read past where the producer wrote. Worth probing |

### Two ISA caveats worth flagging to the customer

These came up while verifying and are *not* yet captured in the customer's debug list:

1. **`REG2_Force_shared_exp` + `UNP[…].FORCED_SHARED_EXP_shared_exp`** (`UNPACR_Regular.md:126, 304‑305`). When set, the unpacker substitutes a single hardcoded exponent for **every datum**, ignoring the per‑face exponent block in L1. If this register is true on the BFP8 path but false on BFP4 (or holds a stale value from an earlier kernel), the symptom is precisely "magnitudes 2^60–2^109" — random scale applied to correct mantissas. **The U43 probe in the report inspects THCON_SEC0/SEC1; it likely does not cover `UNP[…]` regs.** This is the single most plausible silicon‑side cause that's consistent with every diagnostic the customer ran. Worth a 30‑minute probe of `UNP0_FORCED_SHARED_EXP` and `UNP1_FORCED_SHARED_EXP` plus the `REG2_Force_shared_exp` bit per unpacker before anything else.

2. **Blackhole `UnpackRowWidth` for BFP2/4 is documented as "not yet characterized"** (`UNPACR_Regular.md:210‑214`). Quote: *"`UnpackRowWidth = (DatumSizeBytes <= 1) ? 16 : 32;` // Note: behavior not yet characterized for BFP2/4 formats"*. For the customer's repro both BFP4 (`DatumSizeBytes=0.5`) and BFP8 (`DatumSizeBytes=1`) compute to 16, so this is *not* the failing axis — but the ISA team has formally not committed to BFP4 silicon behavior matching the spec. If the customer's "BFP4 works" baseline is itself coincidentally correct rather than spec‑guaranteed, the comparison to BFP8 may be unreliable. I'd recommend also re‑running the BFP4 baseline with an output‑value sanity check, not just a "no crash" check.

3. **SrcA burst alignment on 16‑row boundaries is `UndefinedBehavior()`** (`UNPACR_Regular.md:336‑341`). Quote: *"for SrcA, a burst that spans a 16‑row set boundary may not commit all of its datums per the formula below. This spec marks the bank‑edge boundary cases as UndefinedBehavior() pending characterization."* This is format‑independent, so it's not BFP8‑specific, but it's a hazard for any matmul with `out_subblock_h × out_subblock_w` straddling 16‑row banks. Worth a glance at the customer's `out_subblock_h=1, out_subblock_w=8` configuration to confirm no straddle.

### What I did *not* re‑verify

- WH/BH unpacker silicon RTL — out of scope; ISA doc is the contract.
- The `Prefetcher` Python orchestration on the customer's `tenstorrent-p1` branch — I'd need to clone their fork to inspect.

The conclusions in the body above stand after this cross‑check. The single substantive update is **suspect (1) above (`Force_shared_exp` + `FORCED_SHARED_EXP`) as the most plausible silicon‑side root cause** — it explains the symptom shape (random scale, BFP8‑specific in practice when the bit is set per‑unpacker, byte‑exact L1, correct THCON cfg) better than anything else I have, and it's testable in well under an hour by the customer.

---

*If after (2) the layout dumps are clean, the `FORCED_SHARED_EXP` probe is also clean, and the unpacked SrcA is still garbage, I'll change my mind and look at silicon. Until then I'd bet against an LLK or hardware bug.*
