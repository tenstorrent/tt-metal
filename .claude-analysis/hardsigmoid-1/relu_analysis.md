# SFPU Kernel Analysis: relu

## Operation
- **Name**: relu (and variants: relu_min, relu_max, leaky_relu)
- **Definition**: relu(x) = max(0, x); relu_max(x) = min(threshold, max(0, x))

## Architecture Layers

### Layer 1: SFPU Kernel (`ckernel_sfpu_relu.h`)
```cpp
template <bool APPROXIMATION_MODE>
inline void relu_min(uint uint_threshold) {
    vFloat threshold = Converter::as_float(uint_threshold);
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[0];
        v_if(a < threshold) { a = threshold; }
        v_endif;
        dst_reg[0] = a;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void relu_max(uint uint_threshold) {
    vFloat threshold = Converter::as_float(uint_threshold);
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[0];
        v_if(a > threshold) { a = threshold; }
        v_endif;
        v_if(a < 0.0f) { a = 0.0f; }
        v_endif;
        dst_reg[0] = a;
        dst_reg++;
    }
}
```

**Key patterns**:
- Uses SFPI `v_if`/`v_endif` conditional construct for branching
- `Converter::as_float()` converts uint parameter to vFloat
- `dst_reg[0]` reads/writes the current tile element
- `dst_reg++` advances to next element in the tile

### Layer 2: LLK Dispatch
Relu operations are declared via macros in the compute kernel API directly.

### Layer 3: Compute API
Declared directly in `compute_kernel_api.h` (legacy path) and separately in `api/compute/eltwise_unary/relu.h`.

## Relevance to hardsigmoid
- **v_if/v_endif pattern**: Simple conditional assignment pattern
- **Converter::as_float**: Parameter conversion pattern
- **dst_reg access pattern**: Standard read-modify-write loop
- **relu_max is conceptually similar**: It clamps both lower (0) and upper bounds, like hardsigmoid
