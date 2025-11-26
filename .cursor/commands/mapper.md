# Mapper

## Overview
Map all functions listed in `compute_kernel_api_reference.md` to their corresponding `init` and `uninit` calls. Each operation in the compute API has associated initialization and uninitialization functions that need to be identified and connected. Output should be mapped_sentinel.md.

## Steps
1. **Understand the compute API operations**
   - Read `compute_kernel_api_reference.md` to get the list of all functions
   - Use deepwiki MCP to get explanations for each operation in the compute API
   - Understand what each operation does and its purpose

2. **Identify init and uninit patterns**
   - Review example kernel implementations (e.g., `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_kernel.cpp`)
   - Look for init and uninit call patterns in the compute folder of operation kernels
   - Understand the naming conventions and structure

3. **Map operations to init/uninit calls**
   - For each operation in the reference, identify its corresponding `init` function
   - For each operation in the reference, identify its corresponding `uninit` function
   - Document the mapping clearly

4. **Verify and validate mappings**
   - Cross-reference with actual kernel implementations
   - Ensure all operations have corresponding init/uninit pairs
   - Check for consistency across similar operations

## Resources
- **Reference document**: `docs/compute_kernel_api_reference.md`
- **Example kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_kernel.cpp`
- **Kernel location**: `ttnn/cpp/ttnn/operations/*/device/kernels/compute/`
- **MCP tool**: Use deepwiki MCP for operation explanations and anything that you are not sure of
- **DeepWiki Repositories**: `tenstorrent/tt-metal` - main repository for firmware and compute API questions, `tenstorrent/tt-isa-documentation` - Instruction information and architecure overview questions

## Notes
- Each operation should have both an `init` and `uninit` call, if not find what is used in that case
- Look for patterns in similar operation types (e.g., eltwise unary operations)
- Kernel implementations in the compute folder are the source of truth for init/uninit usage
