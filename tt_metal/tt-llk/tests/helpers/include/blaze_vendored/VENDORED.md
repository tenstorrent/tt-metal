# blaze_vendored — vendored tt-blaze SFPU kernels (lane FD, 2026-08-21)

Source repo: `tenstorrent/tt-blaze`, branch `nkapre/sfpi`, commit
`69b8782e2c0446af07b2c9c73df996de1ce6e03a` (lane EX tip = lane EW census +
lifts + lane EX bridge lifts).  The tree below mirrors the blaze repo root so
the files' in-repo `"blaze/kernels/..."` include spellings resolve against
`-Ihelpers/include/blaze_vendored` (added in helpers/test_config.py).
Harness-side only — the pristine tt-llk LLK trees are untouched (R7).

Back-port note (blaze -> pin-18 tt-llk harness): the vendored files compile
against the harness's include environment unmodified.  The blaze-env
requirements dissolve here: `COMPILE_FOR_TRISC` is a standard harness define,
`lltt::record/replay` and `load_replay_buf<ExecBool>` exist in the pin-18
canon, and the metal-layer helpers (`_sfpu_sigmoid_`, `_sfpu_tanh_fp32_
accurate_`, `calculate_exponential`, `horizontal_reduce`) are include-order
satisfied by the test drivers (which also supply the tt-metal JIT thread
define `TRISC_MATH` the typed bodies gate on, and the accurate-tanh
programmable constants the blaze PACK-thread callers own in-repo).

## ORIGINALS — byte-exact (sha-verified against the source commit)

| file (under blaze/kernels/) | class |
|---|---|
| sfpu/clamped_silu_sfpu.hpp | TYPED (5 entries) |
| sfpu/logit_softcap_sfpu.hpp | TYPED (TRISC_PACK-gated) |
| sfpu/silu_scaled.hpp | TYPED |
| sfpu/sparse_k_filter_sfpu.hpp | TYPED (Int32) |
| sfpu/zero_pad_sfpu.hpp | TYPED |
| kernel_includes/.../sfpu/experimental/ckernel_sfpu_rope.h | RAW-TTI |
| kernel_includes/.../sfpu/experimental/ckernel_sfpu_sdpa_reduce_row.h | RAW-TTI |
| kernel_includes/.../sfpu/experimental/ckernel_sfpu_softmax_k.h | RAW-TTI |
| kernel_includes/.../sfpu/experimental/ckernel_sfpu_generic_moe_gate_topk.h (+_top8.h, _top16.h) | RAW-TTI |
| kernel_includes/.../sfpu/experimental/ckernel_sfpu_deepseek_top32_rm.h | RAW-TTI |
| kernel_includes/.../sfpu/experimental/ckernel_sfpu_deepseek_moe_gate_topk_single_face.h | RAW-TTI |
| kernel_includes/.../sfpu/experimental/ckernel_sfpu_sdpa_exp_unclamped.h | TYPED |
| kernel_includes/.../metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_add_rsqrt.h | TYPED |
| kernel_includes/.../metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sampling.h | TYPED (vendored for the manifest; NOT raced — it calls the blaze-pin `_reciprocal_compat_` value-form recip API which pin-18 canon does not have; the in-tree sampling row races the evolved canon twin) |

NOT vendored (named exclusions, see the blaze sweep rows' notes):
`ckernel_sfpu_topk_xl.h` (deep phases gated on the FPU face-transpose
MOVD2B/TRNSPSRCB choreography; the lifted phases are delivery-drop utilities
without a self-contained Dst e2e contract; in-source craq-sim fidelity gap
tt-blaze#2475), `ops/dsv4_mhc_sinkhorn/.../sinkhorn_4x4_sfpu.h` (census-level
REFUSED, outside the blaze/kernels scope).

## LIFTS — lanes EW/EX typed semantic bodies (blaze/kernels/sfpu/semantic/)

sfpu_bridge.hpp, rope.hpp, sdpa_reduce_row.hpp, logit_softcap.hpp,
softmax_k.hpp, generic_moe_gate_topk.hpp, deepseek_top32_rm.hpp,
deepseek_moe_gate_topk_single_face.hpp.

The lifts are OUR bodies (not blaze-owned) and carry LANE FD EXECUTION FIXES
made at registration — this was the first time the lane-EX bridge lifts ever
EXECUTED (lane EX was compile-gated only).  Each fix is marked in-file with a
"LANE FD EXECUTION FIX" comment; upstreaming to tt-blaze nkapre/sfpi is the
recorded follow-up.  Findings:

1. sfpu_bridge.hpp `indexed_swap`/`swap_mod`: the SFPSWAP builtin's operand
   contract is arg0 = VD (min under mod1=1), arg1 = VC (max), select(N) =
   argN's register result — proven by sfpi_lib.h's silicon-proven `min_max`.
   Lane EX's assembly-print probe concluded the opposite; the un-fixed bridge
   made every bridge lift sort INVERTED (the moe-gate lift returned the
   BOTTOM-8 on the pinned sim; byte-identical stimuli, hand arm correct).
2. deepseek_moe_gate_topk_single_face.hpp window discipline: LaneConfig
   ENABLE_DEST_INDEX is STORE-VISIBLE state — with the bit set, a bf16 value
   SFPSTORE preserves the Dst word's low 16 bits (craq-sim SFPSTORE bf16 arm,
   lane_config bit 2), which the originals' packed idx|score scheme depends
   on across phase boundaries.  Lane EX's F3/F4 window moves ("no instruction
   in between reads that bit") are refuted; fixed for sum_top2 (window ON for
   the final interm stores, ON at exit) and sort_top4 (ON at exit).
3. deepseek single_face `sort_top4_groups` lift remains EXECUTION-REFUTED on
   the pinned sim after fixes 1-2 (wrong winner set; hybrid-phase bisection
   via -DBLAZE_HYBRID_PHASES in sources/deepseek_moe_gate_test.cpp).  The
   registered sem arm is a PARTIAL lift (sum_top2 + top8 lifted, sort_top4
   original); completing it is a lane-EX follow-up.

## Consumers (all test-side)

- sources/sfpu_blaze_test.cpp + test_sfpu_blaze.py (typed-8 causal rows;
  rope + sdpa_reduce_row full2x2 via BLAZE_IMPL)
- sources/sfpu_generic_moe_gate_topk_test.cpp, sources/sfpu_softmax_k_test.cpp,
  sources/deepseek_moe_gate_test.cpp (+ their python tests): BLAZE_IMPL axis
  (0 = in-tree arm, 1 = vendored blaze original, 2 = vendored lift)
- corpus/sweep_2x2_ops.tsv `blaze-*` rows
