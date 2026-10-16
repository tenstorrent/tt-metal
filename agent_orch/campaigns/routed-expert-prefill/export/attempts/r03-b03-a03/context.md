## Files read
- agent_orch/WORKER.md, campaign.yaml, history.md — rules, metric, allowed paths, every node's index.
- unified_routed_expert_ffn/device/kernels/dataflow/unified_routed_expert_ffn_reader.cpp (r03-b02-a02 version) — gate/up and down K-loops, x owner multicast, sender handshakes.
- .../dataflow/unified_routed_expert_ffn_writer.cpp — UP_SPLIT read + IN1_WRITER_MCAST multicast loop.
- .../compute/fused_swiglu.cpp — matmul_phase_fused_gu (tilize owner, last-block PACK pipeline), multiply_phase, main loop, RE-GU/RE-ACT/RE-DN zones.
- .../compute/binary_act_sfpu.h — op-local SwiGLU-OAI / SiTU-GLU / clamped SiLU-GLU bodies (template for the new SiLU-GLU).
- tt_metal/hw/ckernels/blackhole/.../ckernel_sfpu_silu.h, ckernel_sfpu_sigmoid.h, ckernel_sfpu_clamped_silu_glu.h — stock silu = x * _sfpu_sigmoid_<false>(x) rounded to bf16; sigmoid_init.
- tt_metal/hw/inc/api/compute/compute_kernel_api.h — silu_tile_pack / silu_tile_init_pack.
- tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h — in0->srcB, in1->srcA, one non-reused unpack per tile-mult.
- program factory — ring depth (2/3 slots), L1 fit, activation defines, intermediate formats (partials bf16, intermediates bf8).
- device_operation.cpp — bias allowed only for SwiGluOai / SituGlu; SiTU/clamped BH-only.
- weight_runs.hpp, models/demos/deepseek_v3_d_p/tt/moe/tt_routed_expert.py — ND-shard weight layout (1 K-row x per_core_N shard, round-robin over banks).
- test_routed_expert_prefill.py — shapes, LoFi config (fp32_dest_acc_en False), routing.
## Nodes consulted
- All 40 nodes via history.md; full proposal/reflection for r03-b03-a01/a02, r03-b02-a01/a02, r03-b04-a01/a02, r03-b01-a02, r01-b04-a03, r01-b01-a02, r01-b03-a02; "next" sections of r02-b01-a02/a03, r02-b02-a04, r02-b03-a01, r02-b04-a01/a04.
- r03-b03-a02 (parent): all-NoC-0 gate/up reads lose ~10% — reverted by taking r03-b02-a02's code.
- r03-b02-a02 (best): base code; its profile (reports/r03-b02-a02) gave RE-ACT ≈ 5 µs/expert on SiLU shapes, ~0 on binary shapes.
- r02-b04-a04: SiLU body treatment (instruction count) — different mechanism.
## Analysis scripts
- /tmp/r03b03a03_bw.py — per-shape weight bytes, achieved GB/s, compute floors.
- /tmp/r03b03a03_tl.py — per-row RE-GU/RE-ACT/RE-DN timeline from a profile CSV.
- /tmp/r03b03a03_tx.py — per-core zone end grid.
- /tmp/rep-routed-r03b03a02/route.py — test routing (per-expert tile counts).
