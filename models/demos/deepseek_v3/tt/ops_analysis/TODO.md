# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

# DeepSeek V3 TT Ops — Combined TODO (Docs + Tests)

Start here
- Read `PROCESS_GUIDE.md` in this folder first. It lists compulsory reading and the exact process for matching operator signatures.
- Then take the next unchecked item below and begin.
- Tip: Review completed items (e.g., LMHead) to mirror structure, rigor, and style.

Deliverables per module
- Ops analysis markdown in this folder: list all TTNN ops used in forward, with shapes derived from HF config + mesh, and exact non‑tensor args (memory_config, program_config, compute_kernel_config, topology, etc.).
- Isolated per‑op unit tests under `models/demos/deepseek_v3/tests/` that reconstruct the same memory/program configs and compare to Torch/CPU with PCC.

Checklist (each module has two tasks: Ops Doc and Op Tests)
- LMHead (`tt/lm_head.py`)
  - [x] Ops Doc → `LMHead_ops.md`
  - [x] Op Tests → `tests/test_lm_head_ops.py`
- Embedding1D (`tt/embedding_1d.py`)
  - [x] Ops Doc → `Embedding1D_ops.md`
  - [x] Op Tests → `tests/test_embedding_1d_ops.py`
- DistributedRMSNorm (`tt/rms_norm/distributed_rms_norm.py`)
  - [ ] Ops Doc → `DistributedRMSNorm_ops.md`
  - [ ] Op Tests → `tests/test_distributed_rms_norm_ops.py`
- RMSNorm (`tt/rms_norm/rms_norm.py`)
  - [ ] Ops Doc → `RMSNorm_ops.md`
  - [ ] Op Tests → `tests/test_rms_norm_ops.py`
- MLA1D (`tt/mla_1d.py`)
  - [ ] Ops Doc → `MLA1D_ops.md`
  - [ ] Op Tests → `tests/test_mla_1d_ops.py`
- DecoderBlock (`tt/decoder_block/decoder_block.py`)
  - [ ] Ops Doc → `DecoderBlock_ops.md`
  - [ ] Op Tests → `tests/test_decoder_block_ops.py`
- MoEDecoderBlock (`tt/decoder_block/moe_decoder_block.py`)
  - [ ] Ops Doc → `MoEDecoderBlock_ops.md`
  - [ ] Op Tests → `tests/test_moe_decoder_block_ops.py`
- MLP NonExpert (`tt/mlp/non_expert.py`)
  - [ ] Ops Doc → `MLP_NonExpert_ops.md`
  - [ ] Op Tests → `tests/test_mlp_non_expert_ops.py`
- MLP SharedExpert (`tt/mlp/shared_expert.py`)
  - [ ] Ops Doc → `MLP_SharedExpert_ops.md`
  - [ ] Op Tests → `tests/test_mlp_shared_expert_ops.py`
- MLP 1D (`tt/mlp/mlp_1d.py`)
  - [ ] Ops Doc → `MLP_1D_ops.md`
  - [ ] Op Tests → `tests/test_mlp_1d_ops.py`
- MLP 1D Dequant (`tt/mlp/mlp_1d_dequant.py`)
  - [ ] Ops Doc → `MLP_1D_Dequant_ops.md`
  - [ ] Op Tests → `tests/test_mlp_1d_dequant_ops.py`
- MoE (`tt/moe.py`)
  - [ ] Ops Doc → `MoE_ops.md`
  - [ ] Op Tests → `tests/test_moe_ops.py`
- MoE Gate (`tt/moe_gate.py`)
  - [ ] Ops Doc → `MoE_Gate_ops.md`
  - [ ] Op Tests → `tests/test_moe_gate_ops.py`
- Experts (`tt/experts.py`)
  - [ ] Ops Doc → `Experts_ops.md`
  - [ ] Op Tests → `tests/test_experts_ops.py`
- Model1D (`tt/model_1d.py`)
  - [ ] Ops Doc → `Model1D_ops.md`
  - [ ] Op Tests → `tests/test_model1d_ops.py`
- Generator (`tt/generator.py`)
  - [ ] Ops Doc → `Generator_ops.md`
  - [ ] Op Tests → `tests/test_generator_ops.py`
- Rope (`tt/rope.py`)
  - [ ] Ops Doc → `Rope_ops.md`
  - [ ] Op Tests → `tests/test_rope_ops.py`
- CCL1D (`tt/ccl_1d.py`)
  - [ ] Ops Doc → `CCL1D_ops.md`
  - [ ] Op Tests → `tests/test_ccl1d_ops.py`

Notes
- Always read `PROCESS_GUIDE.md` first; then mirror the source module’s construction path (weight sharding, memory configs, program configs, compute kernel configs) in both the analysis doc and tests.
- Prefer per‑device shapes (on the target mesh) and explicitly state memory configs.
- For composite helpers, enumerate and test the underlying TTNN primitives with identical topologies and arguments.
