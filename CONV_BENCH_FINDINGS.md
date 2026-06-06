# GH #45995 — Conv2d matmul-helper: cross-arch baselines + helper_trm analysis

**Branch:** `wransom/conv_bench` @ `839878b0d58`
**Devices:** Wormhole n150 L and Blackhole p100a (single chip each)
**Measurement:** Tracy `DEVICE KERNEL DURATION [ns]`, warm (2nd of `run_twice`) `Conv2dDeviceOperation` row, ≥2 reps / median (device timing spread 0.0–0.4%).
**Modes** (`TT_CONV_BENCH_MODE`): `main` (main's verbatim hand-written kernel), `helper_sbm` (matmul-helper kernel, SubblockMajor), `helper_trm` (matmul-helper kernel, TileRowMajor subblock relaxation).
**Data:** `conv_bench_data.csv` (WH), `conv_bench_data_bh.csv` (BH). Submodules clean. conv2d only.

---

## TL;DR

- **Migrating conv onto the matmul helper (`main` → `helper_sbm`, same tuner-picked subblock) is safe and a real win on heavyweight BLOCK_SHARDED + `packer_l1_acc` convs.** WH mean **−0.90%** (range −6.8%…+1.3%); BH mean **−1.95%** (range −10.5%…+1.4%). No meaningful regression anywhere; BH win ≥ WH win on 23/35 convs.
- **Answer A (why the helper wins with the *same* subblock):** identical math, less orchestration. `pin_interm_to_captured_base` turns the matmul-partials CB from a churned FIFO (per-subblock reserve/push + a per-K-block full-block L1_ACC drain in `main`) into one reserve + fixed-offset packs + one push. The win tracks `packer_l1_acc=ON × deep-K × BLOCK_SHARDED`, and amplifies on Blackhole because that fixed overhead is a larger fraction of BH's faster math. See §1(a), §4.2.
- **Answer B (why we don't use `helper_trm` for conv):** its only lever is unlocking DST-stranded subblocks, and on-device that lever is empty for real convs — BH: it engages but lands in the noise (+0.07%); WH: it can't engage at the real shape (odd `per_core_M`) and OOMs n150's L1 where it would. It also forfeits the Answer-A `pin` win. See §1(b), §5.
- **Helper capability (most-asked):** the helper **can** drive TileRowMajor for conv today — verified on-device (PCC 0.9997) after a *conv-side* CB-sizing fix. It is **not** incapable. The only genuine helper feature-gap, relevant only if a future arch/trace makes TRM worth it, is the absence of a **TileRowMajor-compatible `pin`**. Everything else needed to flip conv to TRM is conv-side wiring, not a helper deficiency. See §6.
- **Design question raised (for the team):** conv's `pin` is the *one* exception to the matmul helper's CB contract (which otherwise forbids CB dual-use + manual pointer manipulation — matmul callers do zero). Keeping it is what buys the migration win above; **dropping it for a consistent/robust CB model would forfeit that win — `helper_sbm` falls back to `main` — not merely cost L1.** See §9.
- **New to this / found it confusing?** §10 is a plain-language FAQ for the points that trip people up (SubblockMajor vs row-major, who needs a reorder, why only TRM drops pin, "measured on TILE" vs "tested ROW_MAJOR").

---

## 1. The two original questions, answered

### (a) Why is `helper_sbm` faster than `main` at the identical subblock shape?

Both kernels issue the **identical** inner block-matmul (`ckernel::matmul_block`, same operands, same strides, same DEST 4-phase lifecycle) → bit-identical PCC and the same `(per_core_M, per_core_N, out_subblock)`. The only thing that differs is how the **matmul-partials circular buffer** is driven across the K-loop.

- **`main` (hand-written):** per output subblock, per K-block, it `reserve_back`/`push_back`s the partials CB; and with `packer_l1_acc` ON it drains the *entire* output block through that CB on (almost) every K-block — `wait_front(out_block_num_tiles)` + `pop_front(out_block_num_tiles)` — plus manual `fifo_rd_ptr`/`fifo_wr_ptr` juggling.
- **`helper_sbm` (`pin_interm_to_captured_base=true`):** **one** `reserve_back(out_block)` at K-loop entry; each K-block's subblocks packed to **fixed tile offsets** within that reservation (`pack_tile<true>`); reloads read those offsets with **no** `wait_front`/`pop_front`; the per-K-block L1_ACC drain is **skipped entirely** (the packer integrates L1_ACC per-address, so the FIFO bookkeeping isn't needed); **one** `push_back(out_block)` at exit.

Net: the helper removes ≈`(#subblocks × #K-blocks)` CB reserve/push pairs **and** ≈`#K-blocks` full-block drains per output block. That is pure orchestration overhead — CB semaphore traffic, pointer math, drain stalls — with zero change to the matmul itself.

The data is consistent with exactly this:
- The big wins are `packer_l1_acc=ON` + BLOCK_SHARDED + deep reduction (ResNet50 512←512 14²/7², 2048←1024; vanilla 512←512 BS) — they maximize subblocks × K-blocks (so the most ops removed) while the per-op matmul is modest (so the removed overhead is a large fraction of runtime).
- SDXL (`packer_l1_acc=OFF`, hugely compute-bound, K up to 3072) is uniformly neutral — the removed FIFO ops are a rounding error against that much matmul.
- Bigger on BH: BH runs the math 1.5–4.7× faster, but the removed overhead is fixed cost → a larger fraction of BH's shorter kernel (Amdahl).

This is **not** a "fewer reconfigs" story — the helper actually does per-subblock pack-format reconfig where `main` does it per-K-block. The decisive saving is CB FIFO traffic + the L1_ACC drain.

> **Measurement caveat:** the bench `main` file is upstream main's kernel *minus* one split-reader datatype-reconfig micro-opt that upstream main has and the helper kernel keeps. On split-reader convs this slightly inflates the measured gap vs true upstream main — one reconfig per K-block, `packer_l1_acc`-independent, far smaller than the drain, so it cannot explain the multi-percent BS wins.

### (b) Why don't we use `helper_trm` (TileRowMajor) for conv?

`helper_trm`'s *only* purpose is to let the tuner pick a **larger subblock** than the SubblockMajor correctness gate (`out_subblock_w == per_core_N || out_subblock_h == 1`, §4.1) allows. On-device that lever is empty for real convs, for two independent reasons (§5):

1. **It forfeits `pin`.** `pin` is SubblockMajor-only (its offset arithmetic is subblock-aligned), so TileRowMajor must run unpinned — i.e. it gives back the entire Answer-A win.
2. **The subblock-size benefit isn't there.** A taller subblock only helps when weights are the expensive operand (fp32/bf16) *and* the shape is DST-stranded; even on the single best candidate it measured **in the noise** on BH, and on WH it either can't form a taller subblock (odd `per_core_M`) or doesn't fit n150's L1.

So `helper_trm` is excluded from the migration dataset (§3); the authoritative signal is `main` vs `helper_sbm`. Note this is *not* a helper limitation — see §6 for the capability question.

---

## 2. Methodology — what is held vs varied

**Two distinct experiments, on two different output formats — don't conflate them:**

1. **The migration (§3 — the headline result).** `main` vs `helper_sbm` — *both* SubblockMajor, same tuner-picked subblock — at each conv's **real config** (real `weights_dtype`, `fp32_accum`, fidelity, `output_layout`, `packer_l1_acc`, shard, bias). Real configs are **TILE output**, so the migration win is a *TILE-output* result. `helper_trm` is **not** run here — the factory gates it to ROW_MAJOR (§4.6). `main` and `helper_sbm` give identical PCC per conv; only duration differs.
2. **The TRM relaxation (§5 — a forced side-study).** `helper_sbm` vs `helper_trm` — SubblockMajor vs TileRowMajor, gate-violating subblock — in **forced ROW_MAJOR output** (the only format the factory lets TRM run in). A factorial (`TT_CONV_BENCH_SUBBLOCK_H/W`) separates the pure subblock lever from the TRM-machinery cost.

So in this doc, *"measured on TILE"* refers to experiment 1 (the migration) and *"tested only on ROW_MAJOR"* refers to experiment 2 (TRM) — different comparisons on different formats, not a contradiction. The three earlier bench forcings (ROW_MAJOR out, `packer_l1_acc` off, fp32 weights) were lifted for experiment 1 (commit `ff9b84c95e0`); every forcing is bench-gated, so non-bench convs are byte-identical.

Per-family real configs (experiment 1):
- **ResNet50** — bf8 weights, LoFi, `fp32_accum=False`, **`packer_l1_acc` ON**, bias, TILE out (batch 20).
- **SDXL UNet** — bf8 weights, HiFi2, `fp32_accum=False`, **`packer_l1_acc` OFF**, BS, bias, TILE out (batch 1).
- **vanilla UNet** — bf16 weights, LoFi, `fp32_accum=False`, **`packer_l1_acc` ON**, **no bias**, HS/BS, TILE out (batch 1).

> **Cross-arch note:** WH n150 and BH p100a shard the same conv onto different grids, so `per_core_M` / `per_core_N` / the tuner-picked subblock differ by arch. The BH Δ column in §3 is therefore the **arch-native** delta (same conv, BH's own config), not the same subblock as WH.

---

## 3. Per-conv deltas — Wormhole and Blackhole

`Δ = (helper_sbm − main)/main`; negative = helper faster. Per-conv absolute `ns` for both arches are in the CSVs (`conv_bench_data.csv` = WH, `conv_bench_data_bh.csv` = BH).

### Cross-model summary

| model | convs (fit both) | WH Δ range | WH mean | BH Δ range | BH mean |
|---|---|---|---|---|---|
| ResNet50 | 11 | −6.8% … +1.3% | −1.87% | −10.5% … +1.4% | −3.1% |
| SDXL UNet | 12 | −0.9% … +0.3% | −0.23% | −2.8% … −0.2% | −1.0% |
| vanilla UNet | 12 | −2.6% … +0.2% | −0.69% | −10.2% … +0.3% | −1.8% |
| **all** | **35** | **−6.8% … +1.3%** | **−0.90%** | **−10.5% … +1.4%** | **−1.95%** |

### ResNet50 (bf8, LoFi, fp32_accum=F, packer_l1_acc ON, bias, TILE)

| conv (out←in, H×W, k/stride) | shard | WH subblock | WH Δ | BH Δ |
|---|---|---|---|---|
| 64←64 56² 3×3 | HS | 1×2 | −0.4% | −0.9% |
| 128←128 56² 3×3 s2 | HS | 2×4 | −0.8% | −1.7% |
| 128←128 28² 3×3 | HS | 2×4 | −1.0% | −1.6% |
| 256←256 28² 3×3 s2 | BS | 8×1 / (BH 1×1) | −0.9% | −2.1% |
| 256←256 14² 3×3 | BS | 8×1 / (BH 1×1) | −0.7% | −2.3% |
| **512←512 14² 3×3 s2** | BS | 4×2 | **−6.8%** | **−10.4%** |
| **512←512 7² 3×3** | BS | 4×2 | **−5.8%** | **−10.5%** |
| 256←64 56² 1×1 s2 | HS | 1×8 | −0.1% | +1.4% |
| 1024←512 28² 1×1 s2 | BS | 2×4 / (BH 1×3) | −0.1% | +1.2% |
| **2048←1024 14² 1×1 s2** | BS | 1×8 / (BH 1×6) | **−5.3%** | **−7.4%** |
| stem 64←16 115² 4×4 | HS | 1×2 / (BH 4×2) | +1.3% | +0.0% |

### SDXL UNet (bf8, HiFi2, fp32_accum=F, packer_l1_acc OFF, BS, bias, TILE)

| conv (out←in, H×W, stride) | subblock | WH Δ | BH Δ |
|---|---|---|---|
| 768←1152 64² | 2×3 / (BH 1×3) | −0.3% | −1.1% |
| 1536←1536 16² | 1×6 / (BH 1×5) | −0.4% | −0.4% |
| 1536←1536 32² | 1×6 / (BH 1×5) | −0.0% | −1.0% |
| 1536←2304 32² | 1×6 / (BH 1×5) | +0.3% | −0.7% |
| 1536←3072 16² | 1×6 / (BH 1×5) | −0.2% | −0.2% |
| 1536←3072 32² | 1×6 / (BH 1×5) | −0.5% | −0.6% |
| 768←384 64² | 2×3 / (BH 1×3) | −0.1% | −2.8% |
| 1536←768 32² | 1×6 / (BH 1×5) | −0.2% | −1.8% |
| 768←768 64² | 2×3 / (BH 1×3) | −0.9% | −1.3% |
| 1536←1536 32² s2 | 1×6 / (BH 1×5) | −0.4% | −0.3% |
| 384←384 128² s2 | 4×2 / (BH 1×2) | +0.1% | −1.0% |
| 768←768 64² s2 | 2×3 | −0.2% | −1.1% |
| 1536←1536 64² *(BH-only fit)* | 1×5 | (OOM) | −0.6% |
| 768←1536 64² *(BH-only fit)* | 1×3 | (OOM) | −0.8% |
| 384←384 128² *(BH-only fit)* | 4×2 | (OOM) | −1.1% |

### vanilla UNet (bf16, LoFi, fp32_accum=F, packer_l1_acc ON, no bias, TILE)

| conv (out←in, H×W) | shard | subblock | WH Δ | BH Δ |
|---|---|---|---|---|
| 32←3 480×640 | HS | 6×1 / (BH 8×1) | −0.3% | +0.3% |
| 64←32 240×320 | HS | 2×2 | −1.1% | −0.4% |
| 64←64 240×320 | HS | 2×2 | −0.5% | −0.5% |
| 128←64 120×160 | HS | 2×4 | −1.4% | −2.3% |
| 128←128 120×160 | HS | 2×4 | −0.8% | −1.1% |
| 256←128 60×80 | HS | 1×8 | −0.6% | −1.3% |
| 256←256 60×80 | HS | 1×8 | −0.3% | −0.2% |
| 288←288 60×80 | HS | 1×3 | −0.2% | −0.5% |
| 512←256 30×40 | HS | 1×8 | −0.2% | −0.2% |
| **512←512 30×40** | BS | 4×2 / (WH 1×2) | **−2.6%** | **−10.2%** |
| 256←512 60×80 | BS | 1×1 / (BH 5×1) | +0.2% | −3.8% |
| 32←64 256×256 | HS | 8×1 / (BH 1×1) | −0.4% | −1.5% |
| 128←256 120×160 *(BH-only fit)* | HS | 2×4 | (OOM) | −0.1% |
| SDXL VAE 512←512 64×64 *(BH-only fit)* | BS | 1×2 | (n/a) | −3.6% |

---

## 4. Mechanism / perf tradeoffs (the concepts in detail)

### 4.1 Output tile order, the SubblockMajor gate, and what reads conv's output

A matmul's per-core output is an `M×N` grid of tiles, computed in `out_subblock_h × out_subblock_w` chunks (subblocks) because DEST holds only a few tiles at once. **Pack order** = the order tiles land in the output CB:

- **SubblockMajor** (conv/bmm default): subblock (0,0)'s tiles, then (0,1)'s, … Compute packs sequentially per subblock.
- **TileRowMajor** (SDPA / subblock-growing matmul): tile-row order — (0,0),(0,1),…,(0,N−1),(1,0),… at absolute offsets.

A sharded output tensor wants **row-major tile order**. SubblockMajor order equals row-major order *only* when `out_subblock_w == per_core_N` (one N-subblock) **or** `out_subblock_h == 1` (each subblock is a one-tile-tall strip). That is exactly the conv op's sharded-output gate (`conv2d_device_operation.cpp`). Any other shape (`h>1` and `w<per_core_N`) interleaves columns wrong.

> **Plain-language (this confused us — FAQ §10):** "SubblockMajor" and "row-major" are **not** two competing final orders. *Under the gate they are the same layout.* In the common `out_subblock_h==1` case (1×8, 1×3, …) each subblock is a single tile-row, so packing subblocks in order *is* row-major. So when conv finishes, its output **is** in row-major tile order — "SubblockMajor" only names the packing *strategy* that produces it. The gate is precisely the condition that makes that strategy land row-major; that is *why* the tuner is constrained to gate-legal subblocks.

**What reads conv's output:** for the common TILE-output case the matmul packs *directly into the globally-allocated output CB* — the pack position **is** the final tile position in the sharded tensor; there is no separate writer kernel doing a reorder. So the compute must pack in the order the tensor's layout expects. SubblockMajor-under-the-gate delivers that for free, which is why the tuner is constrained to gate-legal subblocks.

### 4.2 `pin` — the source of the Answer-A win, and why TRM can't keep it

Conv allocates the matmul-partials CB to **alias the output CB** in L1 (`partials_cb_uses_output`). A normal FIFO would, across the K-loop, advance the partials pointers and eventually wrap into already-written output. `pin_interm_to_captured_base` solves this by treating partials as a **fixed-address scratchpad**: reserve the whole block once, pack/reload each K-block at fixed subblock offsets, never advance the CB pointers, push once at exit (§1(a)). The perf benefit is the elimination of per-subblock reserve/push and the per-K-block L1_ACC drain.

`pin` requires SubblockMajor — its offset is `(in0_subblock·in1_num_subblocks + in1_subblock)·out_num_tiles`, which is subblock-contiguous. TileRowMajor uses row-strided offsets, so the helper static-asserts `pin ⇒ SubblockMajor`; conv sets `pin = !tile_pack_row_major`. **Turning on TRM turns off pin → you pay back exactly the overhead pin removed.** On a heavyweight `l1_acc` conv that is a real regression, not a gain.

### 4.3 DST capacity, subblock volume, and the weight/activation re-read tradeoff

DEST holds **8** tiles (`fp32_accum=False`) or **4** (`fp32_accum=True`); a subblock's `h·w` must fit. Operand re-reads from L1 follow the loop nesting:

- weight (in1) tile-reads ∝ `per_core_M / out_subblock_h` → **taller `h` ⇒ fewer weight reads**
- activation (in0) tile-reads ∝ `per_core_N / out_subblock_w` → **wider `w` ⇒ fewer activation reads**

Within a fixed volume (e.g. 1×8 ↔ 2×4 ↔ 4×2, all 8 tiles) DST utilization is identical and you only slide the weight/activation re-read split — so reshaping within a volume is a wash-to-loss and the fast-path tuner already picks the best (it puts `1×8` first). The lever that *could* matter is **raising the volume** when the gate strands DST below capacity (e.g. `per_core_N=9` forces `1×3` = 3/8 because 9 has no divisor near 8). Going `1×3 → 2×3` fills DST 3/8→6/8 and halves weight reads.

But that only *pays* when weights are the expensive operand:
- **fp32 weights** (4 KB/tile): the matmul is weight-bandwidth-bound; halving weight re-reads is a real win (measured ≈6–9% in the forced regime).
- **bf16 weights** (2 KB/tile): partial — the lever points the right way but is weaker.
- **bf8 weights** (block-float, ~1 KB/tile): the weight-read lever is essentially inert. Empirically bf8 *prefers* `1×8` even though it re-reads weights most, and the ordering is non-monotonic (`4×2` beats `2×4` at equal volume) — a uniform-across-RISC matmul-issue/pipeline effect, **not** a bandwidth effect (the byte arithmetic predicts the opposite). Pinning that down would need LLK-level issue-rate profiling; for this work the takeaway is just that the bf8 regime gets no benefit from a taller subblock.

Real heavyweight convs are bf8 (ResNet/SDXL) or bf16 (vanilla), TILE-output, and their gate-legal subblock is already optimal — so there is nothing for the relaxation to unlock.

### 4.4 Reorder / untilize / reblock — and why TRM needs no reorder

- **TILE output:** TileRowMajor packs row-major *directly* into the sharded output — **no reorder needed** (TRM order = the order the tensor wants).
- **ROW_MAJOR output:** an untilize step converts tiles → row-major bytes. SubblockMajor needs `reblock_and_untilize` (a gather, SubblockMajor-only by static_assert); TileRowMajor uses the plain `untilize` helper because its row strip is already contiguous.
- A reorder is only ever needed for **SubblockMajor with a gate-violating subblock** — which is not a TRM scenario.

**Mind the direction (this flipped on us repeatedly — FAQ §10):** it is **SBM** that needs the *complex* `reblock` gather; **TRM** uses the *simple* plain `untilize` (it is already row-ordered). And **TRM never needs a reorder — not even a gate-violating subblock on TILE output** — because it packs row-major directly for *any* subblock. The reorder is *SBM's* problem, and on TILE there is no untilize step to perform it, which is exactly why gate-violating **SBM** is simply illegal on TILE while **TRM** sidesteps it. So at no point does TRM require a reorder the helper lacks (and a conv-side reorder, were one ever needed, would be conv's responsibility, not a helper deficiency — §6).

### 4.5 The two TileRowMajor deadlocks (both conv-side wiring, now understood)

1. **TRM + fused bias on a shared CB:** conv's untilize+bias path aliases `out_buf == partials_buf`; the TRM bias branch does `wait_front(row_group)` then `reserve_back(row_group)` on that same CB before the pop → when `per_core_M == out_subblock_h` the reserve can never find space. Worked around by routing the bias path through SubblockMajor. The clean fix is a *distinct* output CB (the helper's CB contract already requires in/out to be distinct) — conv-side.
2. **TRM + software-reload + multi-K-block (the one fixed this session):** the partials CB was sized to exactly `per_core_out_ntiles`, so a non-last K-block's per-subblock spills fill it and the last K-block's per-row-group `reserve_back` can never get space → TRISC2's packer holds the pack engine, TRISC0's unpack stalls on it → cross-RISC deadlock (device timeout / FW-init failure). Fixed in `conv2d_op_program_factory_common.cpp` by sizing MATMUL_PARTIALS `+row_group_tiles`, gated to the bench path — **conv-side** (a program-factory CB-sizing change), see §6.

### 4.6 The significance of output format — why the bench restricts `helper_trm` to ROW_MAJOR

The conv program factory hard-gates `helper_trm` to `untilize_out=true` (ROW_MAJOR output): `TT_FATAL(untilize_out, "helper_trm … only applies to ROW_MAJOR output (untilize_out=true); got tiled output")`. The two output formats consume the matmul output differently, and that is what decides where TileRowMajor is meaningful:

- **TILE output (`untilize_out=false`; the common/real case).** The matmul packs straight into the **globally-allocated** output CB — the pack position **is** the final tile position in the sharded tensor (no separate writeback reorders it, §4.1). The tensor wants row-major tile order, and **both** SubblockMajor-under-the-gate and TileRowMajor produce row-major — so on a gate-legal subblock they are **byte-identical (TRM is a no-op)**. TileRowMajor differs here only if you pick a gate-violating (taller) subblock, and that path — per-row-group writeback into the aliased output, with `pin` necessarily off — was never wired or validated in the bench, so the factory guard-fatals it. This is also why the globally-allocated aliasing makes `pin` valuable (§4.2), and `pin` is SubblockMajor-only.
- **ROW_MAJOR output (`untilize_out=true`).** The matmul packs to a **separate** interm CB, then an `untilize` step converts tiles → row-major bytes. This is the one path where the pack order becomes a distinct code path: SubblockMajor interm must go through `reblock_and_untilize` (the gather, correct only under the gate), while TileRowMajor interm goes through the plain `untilize` helper (correct for *any* subblock, because the row strip is already contiguous). It is also where the relaxation's only measured win lives (fp32 weight re-read halving, §7).

So the ROW_MAJOR-only restriction is a **bench scoping decision**: it confines `helper_trm` to the single path where (a) TileRowMajor is a genuinely distinct code path from `helper_sbm` (plain `untilize` vs `reblock_and_untilize`) and (b) the relaxation had a measured benefit. **It is not fundamental.** TileRowMajor packs row-major tile order, which is exactly what a TILE sharded output also wants, so a TILE-output conv *could* use TileRowMajor — it just needs the conv-side wiring of §6 (a non-aliasing output CB / per-row-group writeback) and accepts the `pin` forfeit. Output format gates *where the bench exercises TRM*, not whether TRM is possible for conv.

**Net — the (pack × subblock × output) space collapses to two meaningful behaviors.** Two of the four (pack × subblock) combinations are degenerate:
- **SBM is always gate-legal.** The gate exists precisely to keep SubblockMajor's tile order correct; gate-violating SBM scrambles the output (garbage PCC, blocked by `validate()`). So "SBM" effectively means "SBM, gate-legal."
- **TRM gate-legal is pointless** — it yields the same result as SBM with extra steps (the factory rejects it via the `weight_num_subblocks>1` no-op guard). So "meaningful TRM" effectively means "TRM, gate-violating" — the relaxation.

That leaves **two distinct, meaningful behaviors**:
1. **SBM gate-legal** (× both output formats) — the real world: the entire §3 dataset (TILE) plus the PoC baseline (ROW_MAJOR, where SBM goes through `reblock_and_untilize`).
2. **TRM gate-violating** (× both output formats) — the relaxation. We tested it on **ROW_MAJOR** (§5: lever in the noise on BH, OOM / odd-`per_core_M` on WH); the **TILE** variant is possible but **unwired** (factory guard-fatals it — the §6 case). We tested ROW_MAJOR and not TILE because that is where TRM has a clean home: the matmul packs into a *separate* interm CB that the plain `untilize` consumes with ordinary FIFO semantics, whereas TILE would make TRM pack per-row-group straight into the *globally-allocated* output shard with `pin` off — an untested writeback path on the very buffer `pin` exists to manage. The relaxation's matmul win would be identical in both formats; ROW_MAJOR is simply the format whose downstream plumbing is trivial.

---

## 5. `helper_trm` on-device PoC — confirming it is not worth the larger test

Candidate = the single DST-stranded, weight-bound shape in the corpus: **vanilla 288←288, 60×80, HS, bf16, no bias** (`per_core_N=9`, which 288=9×32 produces; the SubblockMajor gate forces `1×3` = DST 3/8; TRM can relax to `2×3` = 6/8, same `w=3` so a *pure* weight-read halving with no activation counterweight). Studied in forced ROW_MAJOR output with `l1_acc` off — the most TRM-favorable setting (pin's biggest benefit, the per-K-block L1_ACC drain skip, is absent there, so the SBM baseline's `pin` advantage is small; the factorial below isolates the pure lever regardless).

**Blackhole p100a (per_core_M=2):** factorial, warm ns —

| config | subblock | warm ns | vs baseline |
|---|---|---|---|
| `helper_sbm` (baseline) | 1×3 | 67,024 | — |
| `helper_trm` (the lever) | 2×3 | 67,069 | **+0.07%** |
| `helper_trm` (machinery only) | 1×3 | 67,186 | +0.24% |

Pure subblock lever (trm 2×3 vs trm 1×3) = **−0.17%**; noise floor (cold vs warm) = **0.21%**. **Every effect is at/below noise.** Reason: at the real per-core shape `per_core_M=2`, the matmul is a tiny slice of the 67 µs kernel (row-major untilize + tilize + dispatch dominate), so halving weight re-reads on a 2-row matmul is invisible. This is the best candidate; if it's noise here it's noise everywhere real.

**Wormhole n150:** the lever cannot be brought to bear at all —

| batch | `per_core_M` | TRM's relaxed pick | outcome |
|---|---|---|---|
| 1 (real) | 3 (odd) | 1×3 (= SubblockMajor) | runs, **no-op** — `h=2` illegal (3 not divisible by 2) |
| 2 | 5 (odd) | 5×1 | **L1 OOM** |
| 4 | 10 (even) | 2×3 (clean lever) | **L1 OOM** — needs 1.93 MB > n150's 1.5 MB |
| 8 | 19 (odd) | 1×3 | no-op |

n150's smaller L1 is the binding constraint: the matmul-bound shape that *would* show a lever (`per_core_M=10`) doesn't fit the (pin-off, larger-CB) TRM path, and every fitting shape has an odd `per_core_M` that makes a taller subblock illegal. (`helper_sbm` 1×3 fits at all these batches; only the TRM path overflows.)

**Cross-arch conclusion:** on the single best candidate, the relaxation yields no usable win on either architecture — BH proves it's negligible where it runs; WH proves it can't even be brought to bear. The "different bottlenecks" on WH are real (smaller L1), but they make TRM *less* viable, not more. This closes the question: there is no real-conv `helper_trm` win to chase.

> A note on generality (why this is the *only* candidate): every other real conv has a gate-friendly `per_core_N` (a divisor near DST) so the relaxed and constrained subblock coincide → TRM is a literal no-op; or it's bf8 (lever inert, §4.3); or it's a heavyweight `l1_acc` conv where TRM would forfeit the pin win (§4.2).

---

## 6. Can the helper support TileRowMajor for conv? (capability vs conv-side work)

**Yes — the helper is capable today; it is *not* truly incapable.** This session ran `helper_trm` end-to-end on a real tiled-reduction conv with **PCC 0.9997**: `matmul_block` packs TileRowMajor, `add_bias_bcast_rows` has a TileRowMajor branch, and the plain `untilize` helper consumes the row strip. What blocked it initially was a **conv-side** CB-sizing bug, fixed in the conv program factory (§4.5).

**Production evidence the helper does TRM (bias included):** the matmul kernel `bmm_large_block_zm_fused_bias_activation.cpp` already runs TileRowMajor — a factory `TILE_PACK_ROW_MAJOR` define flips its `output_layout` to `TileRowMajor`, threaded through both `matmul_block` *and* `add_bias_bcast_rows`. It works with fused bias precisely because it uses **distinct** partials and output CBs, so the TRM+bias self-deadlock that forces conv to SBM (§4.5) never arises. (The `…gathered` variant hardcodes SubblockMajor — its writer needs subblock order.) So conv's SBM-on-bias is purely conv's CB-aliasing quirk, not a helper limit.

To flip conv to TileRowMajor in the future, the required work is almost entirely **conv-side (not helper deficiencies):**
- un-gate the conv op `validate()` for the chosen path;
- size MATMUL_PARTIALS for the TRM reload granularity (the §4.5 fix — a program-factory change);
- give the fused-bias path a *distinct* output CB (the helper's CB contract already requires distinct in/out; conv's aliasing is the issue);
- **no reorder is needed** for tiled TRM (TRM packs row-major directly, §4.4) — and even if one were, a conv-side reorder is conv's responsibility, not a helper gap.

**The only genuine helper feature-gap — flag this for the future:**
- **No TileRowMajor-compatible `pin`.** `pin_interm_to_captured_base` is SubblockMajor-only (§4.2). So TileRowMajor necessarily runs unpinned and forfeits the partials-CB-traffic elimination that is the entire Answer-A win. If a future architecture or trace ever made the TRM subblock relaxation worth using on conv, and you did not want to surrender the pin advantage to get it, the helper would need a **TileRowMajor-aware pin** (fixed-base packs/reloads computed with row-major, not subblock-major, offsets). This does not exist today. It is a *performance-parity* feature, not a correctness blocker — TRM functions without it (we ran it).

**One optional helper robustness improvement** (not a blocker): on the `packer_l1_acc=false` TRM path the helper spills non-last K-blocks per-subblock but packs the last per-row-group — the granularity mismatch that forced the conv-side CB over-size (§4.5). Making that spill uniformly row-grouped would let callers size the interm CB normally. Minor; the conv-side bump already works around it.

**Bottom line:** helper-capable now (with conv-side wiring); the only thing the *helper* would need for a future TRM-on-conv that also keeps today's pin win is a TileRowMajor-compatible `pin`.

(For the *separate* design question — whether conv should keep `pin` at all, since it is the lone exception to the matmul helper's CB contract — see §9.)

---

## 7. Background — the forced regime (prior sessions)

To make `helper_trm` runnable before the real-config rewire, the harness forced ROW_MAJOR output + `packer_l1_acc` off + **fp32 weights**, and two real issues were found and fixed: the validate() `out_subblock_h>1` gate (relaxed for the HelperRowMajor path) and the TileRowMajor+bias self-deadlock (§4.5, routed through SubblockMajor). In that forced fp32-weights regime the relaxation gives ≈6–9% (a taller subblock halves fp32-weight re-reads, §4.3) — but fp32 weights + ROW_MAJOR output is not how models run convs, which is why the real-config dataset is authoritative and the forced win is mechanism only.

---

## 8. Coverage / out of scope

- **Collected (heavyweight, eligible, fits single-chip):** ResNet50, SDXL UNet, vanilla UNet — 35 convs both arches; BH's larger L1 additionally fits 5 WH-OOM cases + one SDXL VAE conv.
- **Structurally ineligible:** yolo/mobilenet/efficientnet heavy convs are depthwise (different kernel); segformer/swin heavy convs are depthwise/width-sharded. Only low-FLOP non-depthwise stems are eligible.
- **Eligible but does not fit single-chip even on BH:** the 512²/1024²-spatial SDXL VAE convs and the 128×128 high-channel SDXL convs — genuinely need DRAM activation slicing, which this single-chip harness does not express.
- **Out of scope:** conv1d / conv_transpose2d / conv3d (separate ops, no bench wiring).

---

## 9. Pinning, the CB contract, and the open design question

The matmul helper enforces a **CB contract**: `in0` / `in1` / `out` must be **distinct** CBs — in-place aliasing "is NOT supported and will silently corrupt FIFO state" (`matmul_block_helpers.hpp`; `ASSERT(in0_cb_id != out_cb_id)` in `.inl`). The broader intent is that **no caller manually manipulates a CB's FIFO pointers**, so every other user of that CB can trust the pointers point where the CB API says they do. The contract names **exactly one exception**: *"the one supported aliasing is interm_buf overlaying out_buf in L1 (conv2d's `partials_cb_uses_output` path); opt in via `pin_interm_to_captured_base`."*

**Conv is that exception, and it is not consistent with matmul:**

| | matmul callers (`zm_fused`, gathered, CCL) | conv (`conv_bmm_tilize.cpp`) |
|---|---|---|
| in/out CBs | **distinct** | partials **aliases** out (`partials_cb_uses_output`) |
| `pin_interm_to_captured_base` | **false** (every caller) | **true** |
| manual `fifo_rd/wr_ptr` pokes | **0** | **~9** (partials capture/reset) |

So conv uses exactly the CB-dual-use + manual-pointer-manipulation the contract otherwise forbids — behind a documented, asserted carve-out, driven by conv's L1 pressure (overlaying partials on the output buffer saves a whole output-block of L1, and conv is L1-bound).

**Two things the team should weigh:**

1. **Dropping `pin` from conv forfeits the migration win — not just L1.** The −5…−10.5% in §3 *is* the pin win (Answer A, §1a). Without pin, `helper_sbm` reverts to `main`'s per-subblock churn + L1_ACC drain → the migration goes **perf-neutral** (`helper_sbm` ≈ `main`, possibly a hair worse from per-subblock reconfig) **and** costs more L1 (separate partials buffer). So "drop the exception for a robust CB model" trades **both** the perf win and L1 — a legitimate choice, but the cost is two-axis, not L1-only.
2. **There are two separable "abuses."** (a) the CB dual-use (aliasing partials/out), and (b) `pin`'s pointer manipulation. Dropping *only* (a) — give partials its own CB, keep pin — **keeps the win** and costs L1, removing the dual-use but not the pointer manipulation. Dropping pin removes **both** but loses the win. Which lever to pull depends on which objection you are targeting.

**On TRM+pin specifically:** not directly tested — it does not exist (the helper has no TileRowMajor-compatible pin). But the §5 factorial isolated the *only* thing TRM+pin would add over today's SBM+pin — the pure subblock lever, both pin-off — at −0.17% (noise). Pinning is shared with SBM, so TRM+pin's marginal benefit on real convs is ~nil; building a row-strided pin is poor ROI unless a future arch/trace makes the lever real.

**Open design question (no decision recorded here):** keep conv's `pin` exception (it *is* the migration win, and it minimizes L1 on an L1-bound op) or drop it (a consistent, more robust CB model, at the cost of the win + extra L1, with L1 pressure pushed elsewhere / onto newer HW)?

---

## 10. Common confusions, clarified (plain-language FAQ)

**Q: Don't we want row-major tile order, not subblock-major?** Yes — and **under the gate, SubblockMajor *is* row-major.** They are not competing orders. With `out_subblock_h==1` (the common 1×8 / 1×3 / … case) each subblock is one tile-row, so packing subblocks in order is literally row-major. "SubblockMajor" names the packing *strategy*; under the gate the *result* is row-major. (§4.1)

**Q: Isn't SBM just TRM for the subblock shapes we actually pick?** For gate-legal subblocks (all the tuner picks for real convs): **yes — byte-identical output.** TRM only diverges for *gate-violating* shapes, which real convs never pick. Even at the same shape, though, **SBM is the faster of the two** because it keeps `pin` and TRM cannot. (§4.6, §5)

**Q: Which untilize does each use (ROW_MAJOR output)?** **SBM → `reblock_and_untilize`** (the complex *gather*). **TRM → plain `untilize`** (simple), because TRM is already row-ordered. Don't flip these — the *simple* one goes with TRM. (§4.4)

**Q: If we violate the gate with TRM (even on TILE output), don't we need a reorder?** **No — TRM never needs a reorder, for any subblock.** It packs row-major directly, which is exactly what the output wants. The reorder is *SBM's* problem; on TILE there is no untilize step to perform it, which is why gate-violating *SBM* is illegal there — but TRM sidesteps the whole issue. (§4.4)

**Q: Why does only TRM have to drop `pin`, not SBM?** `pin` packs each subblock to a *subblock-contiguous* fixed offset and reloads it as a contiguous run — that layout **is** SubblockMajor. TRM's tiles are row-strided (a subblock's tiles are not contiguous), so pin's offset math cannot address them. Pin is a SubblockMajor-shaped feature: SBM keeps it natively, TRM cannot use it. (§4.2)

**Q: We "measured on TILE" but "tested only ROW_MAJOR" — which is it?** Two different experiments: the **migration** (`main` vs `helper_sbm`, both SBM) ran at each conv's real config = **TILE output** → that is the headline win; the **TRM relaxation** (`helper_sbm` vs `helper_trm`) was forced to **ROW_MAJOR** (the only format the factory lets TRM run in) → that is the null lever. Different comparisons, different formats — not a contradiction. (§2)

**Q: Will real models (TILE output) see the helper win?** Yes — the migration win **is** a TILE result (the entire §3 dataset is at real TILE configs). It does not depend on TRM, and TRM is moot for real convs regardless of output format. (§2, §3)

**Q: Does dropping conv's `pin` just cost L1?** No — it also **erases the migration win** (`helper_sbm` falls back to `main`, §1a/§9). The win comes *from* pin, so dropping pin costs perf **and** L1. (§9)

---

## Appendix — reproduction

Per-conv, env-driven (defaults match real usage: TILE out, l1_acc on). Migration baseline example (ResNet50 L4, BH or WH):
```
CB_BATCH=20 CB_OUT_CH=512 CB_IN_CH=512 CB_H=14 CB_W=14 CB_FILTER=3 CB_STRIDE=2 CB_PAD=1,1,1,1 CB_SHARD=BS \
CB_WEIGHTS_DTYPE=bfloat8_b CB_OUT_DTYPE=bfloat16 CB_FP32_ACCUM=false CB_FIDELITY=LoFi CB_L1_ACC=true CB_OUT_LAYOUT=tile \
CB_BIAS=true  TT_CONV_BENCH_MODE=<main|helper_sbm> python -m tracy -r -p -v -m pytest \
  tests/ttnn/unit_tests/operations/conv/test_conv_bench.py
```
Read warm duration from the newest `generated/profiler/reports/*/ops_perf_results_*.csv`, col `DEVICE KERNEL DURATION [ns]`, 2nd `Conv2dDeviceOperation` row.

`helper_trm` PoC (forced ROW_MAJOR so TRM pays no pin-loss; factorial isolates the lever). The 288←288 candidate:
```
CB_OUT_CH=288 CB_IN_CH=288 CB_H=60 CB_W=80 CB_FILTER=3 CB_STRIDE=1 CB_PAD=1,1,1,1 CB_SHARD=HS \
CB_WEIGHTS_DTYPE=bfloat16 CB_OUT_DTYPE=bfloat16 CB_IN_DTYPE=bfloat16 CB_FP32_ACCUM=false CB_FIDELITY=LoFi \
CB_BIAS=false CB_OUT_LAYOUT=row_major CB_L1_ACC=false CB_BATCH=<1|4> \
TT_CONV_BENCH_MODE=<helper_sbm|helper_trm> [TT_CONV_BENCH_SUBBLOCK_H=h TT_CONV_BENCH_SUBBLOCK_W=w] \
python -m tracy -r -p -v -m pytest tests/ttnn/unit_tests/operations/conv/test_conv_bench.py
```
Gate `helper_trm` behind `bash scripts/run_safe_pytest.sh ...` first — TileRowMajor can hang or OOM at lever-engaging shapes; recover a wedged board with `tt-smi -r` then a `ttnn.open_device/close_device` health check.
