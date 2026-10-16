## Files read
- agent_orch/WORKER.md, campaign.yaml: the rules, allowed paths and accuracy gate.
- $DREAM_HOME/routed-expert-prefill/history.md: the index of all 44 nodes.
- ttnn/.../unified_routed_expert_ffn/device/kernels/compute/fused_swiglu.cpp (whole file):
  - matmul_phase (down);
  - matmul_phase_fused_gu: non-last K-block walk, owner tilize lambda, last-block LW pipeline with mm_last;
  - kernel_main: init/reconfig order, per-chunk per_core_M, RE_HW zones.
- ttnn/.../compute/re_hwprof.hpp: the parent's perf-counter instrumentation. Trimmed here for size.
- tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h:
  - `reuse_a = ct >= rt`;
  - in0 -> SrcB, in1 -> SrcA;
  - with !reuse_a, the SrcA tile is unpacked once per column t, and the MOP streams rt SrcB tiles with stride TMP_LO = TILE_SIZE_B * KT_DIM (MULDMAREG once per call);
  - SrcB address for t: tile_size_a * tile_index_a, with no t offset.
- tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_matmul.h:
  - the same MVMUL replay in both modes, only the CLR_A vs CLR_B choice differs;
  - the dst index for !reuse_a is t + rut * ct = m * ct + j.
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_unpack_AB_matmul_api.h: the API wrapper (operand swap, CB page sizes in 16 B units).
- tt_metal/hw/inc/api/compute/matmul.h:
  - matmul_block_init / matmul_block;
  - BH dynamic throttle: re-inits with the current ct/rt when the FW word is odd, so mode-agnostic.
- tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_common.h: TILE_SIZE_B is set at hw configure and on a format-changing SrcB reconfig.
- tech_reports/GEMM_FLOPS/GEMM_FLOPS.md: BH bf4 LoFi big GEMMs (default subblock (4,2), i.e. SrcB streaming) reach ~70-82% of the 16-cycle/tile ideal.
- tt_metal/impl/program/program.cpp ~L3630: the kernel config buffer check.

## Nodes consulted
- All 44 reflections, via git show of every tag. The ones that shaped this node:
  - **r03-b02-a03 (parent):**
    - FPU 16.1 cycles/mult; SrcA unpack 17.5 cycles/mult, serialized with the FPU;
    - throttle never set;
    - next #1 is this mechanism.
  - **r01-b02-a04, r01-b03-a03, r01-b04-a03:** subblock-shape changes inside `ct >= rt` don't change per-mult unpack work. Consistent with this, since here the streamed operand changes.
  - **r03-b01-a01 / r03-b02-a02:** kernel-config-buffer overflows on h7168 (71056-72752 B vs 70656 B), and the size recipes. Hence the out-of-line helpers and a net size reduction.
  - **r01-b04-a02:** in0_block_w is an accuracy knob. Not touched: same K-blocks, same per-tile K order.
  - **r03-b03-a03:** "more precise" numerics fail the mean_abs gate. Here numerics are bit-identical by construction.
  - **r02-b01-a03 / r02-b03-a04 / r03-b04-a03:** the last-block PACK-thread activation pipeline. Kept as is; CS only covers non-last blocks.
  - **r02-b02-a01 / r02-b04-a01 / r03-b03-a03:** layouts and per_core_M per shape (mistral 4 x 2-row groups at pcM 3, gpt-oss 3/3/2/2 at pcM 3-4, dsv3/glm phase A at pcM 3).

## Docs / external references
- Local size harness: /tmp/r03b02a04/sz (copied from /tmp/r03b02a03/sz). kc.sh compiles one TRISC with a dsv3-like CTA; sz.sh sizes all three.
- Measured .o text sizes, profiler on, parent -> this node (T0 / T1 / T2):
  - SiLU 19464/14632/14956 -> 21008/13728/13824;
  - SwiGLU-OAI 19660/14280/14272 -> 21212/12916/13124;
  - SiTU-GLU 19660/14280/13652 -> 21212/12916/12468.
- Also compiled clean: FP32_DEST_ACC_EN=1, TILE-layout x (x_is_row_major=0), and without the profiler.
