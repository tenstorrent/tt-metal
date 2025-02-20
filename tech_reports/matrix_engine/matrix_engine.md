# Matrix Engine

## Introduction

The matrix engine supports the following operations: matrix mult, reduction, eltwise add/sub/mul, and transpose_xy.

## Operations

>[!NOTE]
>All numbers and values apply to both Wormhole and Blackhole devices.

### Matrix Mult

The Wormhole matrix engine performs 8x16 x 16x16 = 8x16 in a single cycle. \
This is 2*8\*16\*16 = 4096 muladds in a single cycle. At 1GHz, this is 4 TFLOPS per matrix engine. \
The 8x16 is the smallest matrix that can be fed into in0, and 16x16 is the
smallest matrix that can be fed into in1.

If the input matrices fed into the engine are "shorter" than 8x16, for example 1x16, the engine will still perform 8x16 x 16x16 = 8x16, but the effective throughput will be 1/8.
Thus, for 1x16 x 16x16 matrices, the effective throughput is 0.5 TFLOPS per matrix engine.

MATH_FIDELITY is used for higher precision, and TFLOPS are calculated by dividing by the MATH_FIDELITY value.

LoFi ->  4 TFLOPS \
HiFi2 -> 2 TFLOPS \
HiFi3 -> 1.33 TFLOPS \
HiFi4 -> 1 TFLOPS

### Reduction: Max/Average/Sum
The Wormhole matrix engine performs 16x16 reduce max/average/sum operations in a single cycle. \
This is 2*16\*16 multiply + adds in a single cycle. At 1GHz, this is 0.512 TFLOPS per matrix engine.

Reduce max does not use MATH_FIDELITY; however reduce average/sum does use MATH_FIDELITY for higher precision, and TFLOPS are calculated by dividing by the MATH_FIDELITY value.

LoFi ->  0.512 TFLOPS \
HiFi2 -> 0.256 TFLOPS \
HiFi3 -> 0.171 TFLOPS \
HiFi4 -> 0.128 TFLOPS

### Eltwise: Add, Sub, Mul
The WH matrix engine performs 8x16 elementwise addition/subtraction/multiplication in a single cycle. \
This is 8\*16 (multiply or adds, not both) in a single cycle. At 1GHz, this is 0.128 TFLOPS per matrix engine. \
Elementwise addition and subtraction do not use MATH_FIDELITY; however, elementwise multiplication does use MATH_FIDELITY for higher precision, and TFLOPS are calculated by dividing by the MATH_FIDELITY value.

LoFi ->  0.128 TFLOPS \
HiFi2 -> 0.064 TFLOPS \
HiFi3 -> 0.043 TFLOPS \
HiFi4 -> 0.032 TFLOPS

## Configurations

```
struct WormholeComputeKernelConfig {
    MathFidelity math_fidelity = MathFidelity::LoFi;
    bool math_approx_mode = true;
    bool fp32_dest_acc_en = false;
    bool packer_l1_acc = false;
};

```

### Math Fidelity

Wormhole multipliers are 5b x 7b multipliers, this means it uses 5 bits from SrcA, and 7bits from SrcB register (Commonly known as operand 0 & operand 1).

Math Fidelity specifies the number of times an operation is run to consume the full precision of the inputs. Math Fidelity has 4 values: LoFi, HiFi2, HiFi3, HiFi4.

LoFi -> SrcA register: uses 1 hidden bit + 4 most significant bits of the mantissa (MSB of the mantissa), SrcB register: uses 1 hidden bit + 6 MSB of the mantissa \
HiFi2 -> SrcA register: uses 1 hidden bit + next 4 bits of LSBs of the mantissa, SrcB register: uses 1 hidden bit + 6 MSB of the mantissa \
HiFi3 -> SrcA register: uses 1 hidden bit + 4 MSB of the mantissa, SrcB register: Uses 1 hidden bit + next 6 LSB of the mantissa \
HiFi4 -> SrcA register: uses 1 hidden bit + next 4 bits of LSBs of the mantissa, SrcB register: Uses 1 hidden bit + next 6 LSB of the mantissa

### Math Approx Mode

Some SFPU operations come in approximate mode, this means the operation can either be run as high precision and low performance, or high performance and lower precision.

Not all SFPU operations support this. But some examples include exponential, gelu, sqrt, etc. In order to enable high performance but lower precision for these sfpu ops, then `math_approx_mode` needs to be set.

### Fp32 Dest Acc (or DST_ACCUM_MODE)

Wormhole can have the FPU operate with Float16/Float16_b values, or Float32 values. In order to set Float32 values, `fp32_dest_acc_en` must be set.

Warning: If this flag is set, the math destination register can fit as half as many tiles as Float16_b. So if using DstSync::Half, then Float16_b can fit 8 tiles, while Float32 can only fit 4.

### Packer L1 acc

Wormhole has the ability to do accumulation in the L1 memory, the packer will read the input address, and accumulate it with the values read from dest, then write back into the same address.
This feature is useful for accumulations in higher precision, and then a final pack call can be done to convert into lower precision (for example accumulate in fp32, then final output as float16_b).
In order to enable this feature, `packer_l1_acc` must be set.
