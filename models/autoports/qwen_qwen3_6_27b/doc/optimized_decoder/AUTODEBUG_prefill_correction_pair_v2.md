# Three-tile output subblocks for the packed prefill pair

Parent measured pair133: K4/W33/SW1 warmed prefill B1/B32 = 7.755/30.289 ms; K8/W33/SW1 = 7.089/28.515 ms, versus tuned separate 6.715/26.81 ms. Those passing but slower rows motivate testing better DST utilization before rejecting the packed topology.

`candidates/prefill_correction_pair_v2.py` differs from v1 only in its Python guard: `sw in (1, 2, 4)` becomes `1 <= sw <= 4`. Its generator and manifest are also immutable new files. Defaults and all math, phase routing, weight setup, tail handling and runtime operations remain unchanged.

Native source supports a 1x3 output subblock. `matmul_device_operation.cpp:615-634` requires block/subblock divisibility; 3 divides both 33 and 99. At lines662-682 the 2D configuration uses half-sync destination capacity and requires subblock area to fit; three FP32 tiles fit the available four. There is no power-of-two width restriction. The compute kernel `bmm_large_block_zm_fused_bias_activation_metal2.cpp:457` passes `effective_subblock_w` directly into `matmul_block`; width3 is not transformed into width4. The writer handles the final two-real-tiles subblock through its existing last-subblock width and padded-tile skip logic. Final N worker width98 therefore remains legal.

Recommended next rows:

| K block | Output block W | Subblock W | Main CB bytes | Assessment |
|---|---:|---:|---:|---|
| 4 | 33 | 3 | 438,784 | Legal baseline with better DST usage |
| 8 | 33 | 3 | 742,400 | Priority: fastest measured K setting |
| 4 | 99 | 3 | 1,283,584 | Useful full-width control; check actual allocation |
| 8 | 99 | 3 | 2,161,664 | Main CB alone exceeds 1,572,864-byte L1 |

The CB estimates follow `matmul_multicore_reuse_mcast_2d_program_factory.cpp:411-434,758-803`: double-buffer BF16 A and BFP8 B, plus one output block of FP32 partials/output. Output and partials share storage here because both are FP32, output is interleaved, and untilize is false. Thus `2*K*W*1088 + 2*K*2048 + W*4096` bytes. Ancillary buffers and reserved L1 are additional. The earlier report used a conservative two-buffer partial/output bound; this source inspection resolves the aliasing and tightens it.

CPU checks assert v2 is exactly the one guard change, the three-tile geometry passes the stated divisibility/capacity constraints, and `py_compile` succeeds. Artifact: `prefill_correction_pair_v2_cpu_checks.json`. Hardware correctness, whole-layer prefix-state sensitivity and total warmed prefill must still be measured. No device was opened for this follow-up.
