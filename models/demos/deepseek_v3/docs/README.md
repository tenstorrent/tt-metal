# DeepSeek V3 MoE Documentation

Comprehensive documentation for the DeepSeek V3 Mixture of Experts (MoE) implementation in TT-Metal.

## Documentation Index

### 1. [MoE Tensor Flow and Parallelism Architecture](moe_tensor_flow_deepseek.md)
**Main Technical Documentation** (~3000 words)

This is the primary technical reference covering:
- Executive summary of DeepSeek V3's hierarchical routing innovation
- Complete tensor journey from input through all MoE stages to output
- Detailed explanation of the GroupedTopK router with 256 experts
- Expert Parallelism (EP) and Tensor Parallelism (TP) strategies
- Device mesh configurations for TG (4×8) and QUAD (16×8)
- Communication patterns and memory optimization
- Key differences from GPT-OSS implementation

**Start here for deep technical understanding of the architecture.**

### 2. [Quick Reference Guide](moe_quick_reference_deepseek.md)
**Practical Commands and Configuration**

Essential reference for running tests and debugging:
- Environment setup commands (one-liners and step-by-step)
- Test commands for different configurations
- Complete configuration parameters table
- Tensor shape reference at each pipeline stage
- Device mapping formulas
- Common errors and fixes
- Debugging checklist

**Use this for day-to-day development and testing.**

### 3. [Parallelism Diagrams and Visualizations](moe_parallelism_diagrams_deepseek.md)
**Visual Architecture Guide**

ASCII art diagrams and visualizations including:
- Device mesh layouts for TG and QUAD configurations
- Expert distribution patterns across devices
- Grouped routing visualization (8 groups × 32 experts)
- Two-stage selection process diagrams
- All-to-all communication flow patterns
- Complete tensor flow pipeline visualization
- Scaling analysis and optimization opportunities

**Reference this for visual understanding of parallelism and communication.**

## Key Architecture Highlights

### DeepSeek V3 Innovations
- **256 routed experts + 1 shared expert**: Hybrid architecture for specialization with knowledge retention
- **Hierarchical GroupedTopK routing**: Two-stage selection (groups then experts) for efficiency
- **8 experts per token**: Higher capacity than typical MoE architectures
- **Scaling factor of 2.5**: Ensures proper gradient flow through routing

### Parallelism Strategy
- **Expert Parallelism (EP)**: Distributes experts across device rows
  - TG: EP=4 (64 experts per row)
  - QUAD: EP=16 (16 experts per row)
- **Tensor Parallelism (TP)**: Shards weights across device columns
  - Both systems: TP=8 (hidden_size/8 per device)

### Test Configuration
```bash
# Quick test command
cd /home/ntarafdar/tt-moe/tt-metal
source python_env/bin/activate
export PYTHONPATH=$PWD TT_METAL_HOME=$PWD MESH_DEVICE=TG
export DEEPSEEK_V3_HF_MODEL=/data/deepseek/DeepSeek-R1-0528
export DEEPSEEK_V3_CACHE=/tmp/deepseek_cache_test && mkdir -p $DEEPSEEK_V3_CACHE
pytest models/demos/deepseek_v3/tests/test_decoder_block.py::test_moe_decoder_block_2d -xvs
```

## Quick Links

### Implementation Files
- **Unified MoE Block**: `models/tt_moe/moe_block.py`
- **GroupedTopK Router**: `models/tt_moe/components/routers/grouped_topk_router.py`
- **Routed Experts**: `models/tt_moe/components/experts/routed_experts.py`
- **Test Entry Point**: `models/demos/deepseek_v3/tests/test_decoder_block.py`

### Model Parameters
- Hidden size: 7168
- Routed experts: 256
- Shared experts: 1
- Experts per token: 8
- Intermediate size (routed): 2048
- Intermediate size (shared): 10752

### Performance Targets
- Accuracy: > 99.99% PCC
- Supported batch sizes:
  - TG Decode: 128
  - QUAD Decode: 512
  - Prefill: Variable (memory limited)

## Getting Started

1. **Set up environment**: Follow the [Quick Reference Guide](moe_quick_reference_deepseek.md#environment-setup)
2. **Understand the architecture**: Read the [main technical documentation](moe_tensor_flow_deepseek.md)
3. **Visualize the parallelism**: Review the [diagrams](moe_parallelism_diagrams_deepseek.md)
4. **Run tests**: Use commands from the [Quick Reference](moe_quick_reference_deepseek.md#test-commands)

## Documentation Maintenance

Last updated: March 2026
Version: 1.0
Authors: TT-MoE Team

For questions or updates, see the main TT-Metal documentation or contact the MoE team.
