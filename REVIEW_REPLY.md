Thanks — fair question, and the answer is "it depends on fidelity", which the PR should have said. I re-ran
the whole thing; numbers below are **device cycles**, not wall-clock. (Wall-clock µs is not trustworthy here:
AICLK moves with load and the bs16 shape sits between two clock states, so the same config re-measured a few
minutes later moved 9%. Device cycles reproduce to 0.2–0.6%, and the same arm measured twice under two names
agrees to 0.03 pp.)

### 1. Workload and configuration

pplx-embed-4b batched prefill on Blackhole P150b. Q `[B, 32, 512, 128]`, K/V `[B, 8, 512, 128]`, bfp8,
non-causal, `output_concat_heads`, `reuse_kv` with q128/k512 (one K chunk), 12×8 at bs8 and 12×10 at bs16/32,
K/V in L1. **LoFi**, `math_approx_mode=False`, `exp_approx_mode=True`, `fp32_dest_acc_en=False` (so the
streaming kernel), `packer_l1_acc=True`.

| fidelity | bs8 | bs16 | bs32 |
|---|---|---|---|
| **LoFi** | **−5.3%** | **−6.6%** | **−6.4%** |
| HiFi2 | +7.1% | +7.3% | +7.4% |
| HiFi4 | +8.7% | +8.8% | +8.6% |

Two other axes move it as much as fidelity does, at LoFi / bs16:

| | |
|---|---|
| d = 64 / 128 / 256 | −8.8% / −6.6% / **+4.2%** |
| `exp_approx_mode` True / False | −6.6% / −6.1% |
| K/V in L1 vs. all DRAM | −6.6% / −1.6% |

The last row matters for how the PR described this: the original bs32 figure was taken all-DRAM, where the op
is bandwidth-bound and the compute saving is hidden. It is a compute gain, so quote it with the placement.

### 2. Matched PACK-vs-MATH profiles

`sdpa_kernel_variants.py zconly` (phase zones, compute-only), pplx bs8 on 12×8, mean cycles per work unit over
3072 units, flag off / on:

| phase | LoFi MATH | LoFi PACK | HiFi2 MATH | HiFi2 PACK |
|---|---|---|---|---|
| P1 QK+exp rowgroup #2 | 5443 → **4555** | 6005 → **4833** | 6183 → 6174 | 6176 → 6160 |
| P2 exp rg1 + PV rg0 | 5691 → **4555** | 5670 → **4556** | 5981 → 5970 | 5933 → 5683 |
| P2 normalize #1 | 1440 → **1828** | 3637 → **4024** | 1604 → **2618** | 5871 → **6870** |
| P2 normalize #2 | 1428 → **1812** | 1438 → **1830** | 1502 → **2493** | 1504 → **2495** |
| **unit total** | **20295 → 18753 (−7.6%)** | | **25252 → 27113 (+7.4%)** | |

And the kernel's own zones (`zfconly`, one work unit per core), per `sub_exp_block_bcast_cols` call — that is
8 score tiles, a 2×4 subblock:

| thread / zone | LoFi off → on | HiFi2 off → on |
|---|---|---|
| MATH `SUB` — mostly the `tile_regs_acquire` wait on PACK | **701 → 390** | **293 → 293** |
| MATH `Q@KT MM+Pack` | 1353 → 1367 | 1903 → 1903 |
| PACK `EXP` | 558 → 556 | 608 → 676 |
| PACK `PACK SUB_EXP` (incl. the 8 L1-acc row-sum packs) | **486 → 147** | **479 → 146** |
| MATH `NORM_MATMUL_RECIP` (per row group) | 459 → 1357 | 414 → 1850 |
| MATH `TRISC-KERNEL` | **25456 → 23943** | **30178 → 31915** |

That is the whole mechanism:

- The flag removes the **same ~339 cycles** of row-sum packs from PACK at both fidelities (486→147, 479→146).
  The pack-thread saving does not depend on fidelity.
- What changes is whether MATH was **waiting for it**. At LoFi MATH idles 701 cycles per call in
  `tile_regs_acquire`; the flag cuts that to 390, so almost all of PACK's saving turns into real time. At
  HiFi2 MATH's wait is 293 and **does not move at all** — its own Q·Kᵀ has gone 1353 → 1903 cycles and is now
  the long pole, so PACK's saving lands in slack.
- The normalize matmul the flag *adds* to MATH costs +898 (LoFi) / +1436 (HiFi2) cycles per row group. At LoFi
  that is less than MATH recovers; at HiFi2 it is more, and it is more precisely because the normalize matmul
  itself runs at HiFi2.

So the predictor is MATH's Q·Kᵀ cycles per score tile against PACK's exp + row-sum cycles per score tile — which
is also why `d` moves it (d=256 loses at LoFi) and why `exp_approx_mode` barely does (the flag does not remove
the exp from PACK, only the row-sum packs, and those cost the same either way).

### 3. Why Blackhole only, and your premise about #22312

Your premise is right, and it is the reason this is fidelity-dependent rather than always-wrong. #22312 put the
row sums on PACK when the kernel ran `exp_tile` on the **math** thread (`compute_common.hpp`), so moving the
reduction off MATH was the right call then. The streaming kernel (#38838) moved exp onto the **pack** thread
(`exp_packthread_tile`) for FPU/SFPU overlap, so PACK now carries the exp *and* the L1-accumulated row-sum
packs — and at low fidelity it has become the pacing thread. At HiFi2/HiFi4 the original reasoning still holds,
which is exactly what the table above shows.

On packer bandwidth: that is not what this relieves, so Blackhole's ~2× packer bandwidth over Wormhole does not
argue against it. What it removes is pack-*thread* time — issuing ~57-cycle single-tile L1-accumulate packs —
not packed bytes.

**Blackhole-only is a scope choice, not an architectural claim.** The batched normalize matmul relies on
Blackhole's `matmul_block` behaviour (`rt_dim` over the row group, `kt_dim` as the row stride, plus a 1×1
re-init afterwards to avoid the V-matmul hang) and was only validated there. A Wormhole port on the WH no-MOP
matmul (`bb3abd495f3`) is plausible but unmeasured, and given the above I would expect it to help only in the
same LoFi / small-`d` / compute-bound corner.

### 4. Accuracy, and why it stays opt-in

Single-call PCC vs. fp32 torch moves only in the 5th decimal (0.99925 -> 0.99923 at the pplx shape). But the
error is systematic, so it accumulates over a model. All three SDXL 1024x1024 module PCC tests, flag forced on:

| test / case | what the flag touches | off | on | bar | on |
|---|---|---|---|---|---|
| `transformermodel` `down_blocks.1` (4096 tok, k512) | 77-tok cross-attn only | 0.9990458 | 0.9990448 | 0.998 | passes |
| `transformermodel` `down_blocks.2` (1024 tok, k1024) | LoFi self-attn | **0.997167** | **0.996846** | 0.997 | **fails** |
| `crossattnmidblock2d` (1024 tok, k1024) | LoFi self-attn | **0.9972232** | **0.9966934** | 0.9968 | **fails** |
| `crossattnupblock` case 0 = `up_blocks.0` (1024 tok, k1024) | **HiFi2** self-attn | 0.9897058 | 0.9847438 | 0.968 | passes |
| `crossattnupblock` case 1 = `up_blocks.1` (4096 tok, k512) | 77-tok cross-attn only | 0.9930153 | 0.9928541 | 0.993 | **fails** |

Rows 2 and 3 reproduce the "0.9972 -> 0.9968" I quoted, and both fall below their CI bar. Row 1 is the clean
control: 8 K chunks there, so the flag only reaches that block's 77-token cross-attention and the block moves
1e-6. Row 5 is the one that corrected me — same 77-token cross-attention as row 1, yet the block moves 1.6e-4
and crosses its 0.993 bar (it passes by only 1.5e-5 with the flag off, so it is a knife-edge test rather than
a big regression, but my "a few K tiles make no measurable difference" was too strong and the docstring no
longer says that).

Given all of that, plus +7%/+9% at HiFi2/HiFi4 and +4% at d=256, it stays **opt-in**. The docstring now spells
out the regime, the config behind the speed numbers, and what the accuracy costs.

One correction to something I had assumed: SDXL *does* have a HiFi2 single-K-chunk self-attention —
`up_blocks.0` is HiFi2 and gets `1024_K` at 1024 tokens. Measured there the flag is about +2%, i.e. mildly
harmful, consistent with everything above.
