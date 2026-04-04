# Reference Operation Selection for hardsigmoid

## Target Operation
- **Name**: hardsigmoid
- **Definition**: max(0, min(1, x/6 + 0.5))
- **Component operations identified**: linear transform (x/6 + 0.5), clamp/min/max (clamping to [0,1])

## Selected References (ranked by relevance)

### 1. hardtanh
- **Why selected**: Hardtanh implements clamp(x, min_val, max_val) = min(max(x, min_val), max_val), which is structurally identical to the clamping portion of hardsigmoid. Uses TTI_SFPLOAD/TTI_SFPSWAP for efficient min/max.
- **Relevance**: high -- the clamping logic can be directly adapted for hardsigmoid's max(0, min(1, ...)) pattern

### 2. relu
- **Why selected**: Relu implements max(0, x) which is the outer max(0, ...) in hardsigmoid. Also demonstrates conditional patterns with v_if and Converter::as_float for parameter handling.
- **Relevance**: high -- demonstrates simple conditional clamping to 0 and general SFPU kernel structure

### 3. clamp
- **Why selected**: Clamp implements min(max(x, min_val), max_val) using the unary_max_min primitives. Shows how to compose max and min operations sequentially on SFPU registers.
- **Relevance**: high -- demonstrates composition of max/min with parameterized bounds

### 4. heaviside
- **Why selected**: Heaviside uses v_if/v_elseif/v_else branching to assign constant values based on input comparisons. Hardsigmoid similarly needs to output constant 0 or 1 in certain ranges.
- **Relevance**: medium -- demonstrates conditional constant assignment pattern and simple SFPU iteration loop

### 5. silu
- **Why selected**: Silu (x * sigmoid(x)) demonstrates the full LLK stack wiring: ckernel_sfpu -> llk_math_eltwise_unary -> compute_kernel_api. Uses _sfpu_sigmoid_ helper and shows fp32/bf16 rounding patterns.
- **Relevance**: medium -- provides template for full integration stack (LLK header, compute API registration, SfpuType enum entry)
