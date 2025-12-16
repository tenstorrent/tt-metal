# ComfyUI-Tenstorrent Integration: Project Summary

## Initial 3 Options for ComfyUI-Tenstorrent Integration

The project began with three architectural approaches:

### Option 1: ComfyUI as HTTP Client (Simplest)
- Create custom ComfyUI nodes that make HTTP requests to existing `tt-inference-server`
- Pure network communication with JSON serialization
- **Trade-off**: 10-50ms latency overhead, limited to server's API contract
- **Timeline**: Fastest implementation, works immediately

### Option 2: Deep Native Integration (Most Complex)
- Implement complete Tenstorrent backend directly in ComfyUI core
- Modify `model_management.py` to add `CPUState.TENSTORRENT`
- Reimplement all model architectures (SDXL, SD3.5, Flux) from scratch
- **Trade-off**: Zero latency, but 4-6 months development + maintenance burden

### Option 3: Hybrid Bridge Approach (CHOSEN - Recommended)
- Create ComfyUI-compatible bridge server using Unix domain sockets
- Wrap existing proven `TTSDXLGenerateRunnerTrace` from tt-metal
- Lightweight client in ComfyUI connecting to separate bridge service
- **Trade-off**: 1-5ms latency, 2.5-3.5 months development, reuses battle-tested code

**Why Bridge Was Chosen**: Option 3 provided the optimal balance—reusing proven implementations while achieving near-native performance, with clear architectural boundaries for debugging.

## Implementation Journey & Key Success

### The Bridge-Owned Loop Implementation

The team implemented a **full bridge-owned denoising loop** approach (~650 lines across 6 files):

**Core Architecture**:
- Bridge handles the entire sampling loop internally (replicating proven `tt-media-server` design)
- All operations stay in bfloat16 throughout denoising
- Single precision conversion only at the final output
- ComfyUI sends one request with all parameters; bridge returns final denoised latent

**Critical Discovery**: The root cause wasn't configuration—it was **numerical precision mismatch**:
- TT-Metal uses bfloat16; ComfyUI expects float32
- ComfyUI's `to_d()` formula `d = (x - denoised) / sigma` amplifies errors by **33x at small sigma values** (σ < 0.5)
- This created failure at denoising steps 16-20 where sigma becomes small

### Solution E: Hybrid Sigma-Dependent Strategy (100% Success)

**Implementation**: Intelligent sigma threshold (0.8) switches strategies mid-denoising
- **Large sigma (σ ≥ 0.8)**: Return epsilon directly
- **Small sigma (σ < 0.8)**: Compute denoised in float64 for precision, then convert back

**Results**: All 20 denoising steps completed successfully, clear high-quality images generated (~95s total including model load)

## Successes & Key Takeaways

### Infrastructure Achievements
- Unix socket bridge with <5ms latency
- Zero-copy shared memory tensor transfer
- Successfully loads SDXL, SD3.5, SD1.4 models on Tenstorrent hardware
- Custom ComfyUI nodes (TT_CheckpointLoader, TT_FullDenoise, TT_ModelInfo)

### Key Insights
1. **Precision boundaries are critical** - bfloat16↔float32 conversions compound through feedback loops
2. **Systematic investigation > random fixes** - Phased approach (1→2→3→Investigation→Solution E) was essential
3. **Hybrid strategies bridge architectural mismatches** - Target problem areas rather than fixing precision everywhere
4. **TT-Metal's core design is sound** - tt-media-server proves the hardware works; issue was integration

## Final Blockers

Despite Solution E's success, **fundamental architectural incompatibility** remained:

### The Core Problem: Loop Control Ownership Conflict

**tt-media-server (Working)**:
- Bridge owns the complete denoising loop
- Epsilon stays internal, never exposed

**ComfyUI-TT Integration (Problematic)**:
- ComfyUI owns the loop, calls bridge per-step
- Expects denoised output but receives epsilon
- Two systems with incompatible state management

### Why Resolution Was Difficult

1. **Scheduler State Conflicts**: TT scheduler initialized for internal loop control conflicts with ComfyUI's external control (IndexError after N calls)
2. **Numerical Instability at Small Sigma**: At σ=0.03, any precision error amplifies 33x catastrophically
3. **bfloat16 Precision Loss**: TT hardware's reduced precision doesn't round-trip cleanly through float32 conversions

### Trade-off Accepted

The working solution (bridge-owned loop) sacrifices ComfyUI's sampler flexibility (no ControlNet, IP-Adapter, custom samplers) for guaranteed quality matching tt-media-server. This became a "TT-only" execution path outside ComfyUI's broader ecosystem, but delivered reliable, high-quality results.

## References

For detailed implementation history, see:
- `docs/comfyui_integration.md` - Initial integration plan
- `docs/COMFYUI_INTEGRATION_IMPLEMENTATION_SUMMARY.md` - Implementation summary
- `old_comfyui_markdowns/` - Complete debugging and implementation journey
