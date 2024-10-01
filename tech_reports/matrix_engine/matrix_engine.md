# Matrix Engine

## Introduction

The matrix engine supports the following operations: matrix mult, dot product, reduction, eltwise add/sub/mul, and tranpose_xy.

## Matrix Mult 

The WH matrix engine performs 8x16 x 16x16 = 8x16 in a single cycle. \
This is 2*8\*16\*16 = 4096 muladds in a single cycle. At 1GHz, this is 4 TFLOPs per matrix engine. \
The 8x16 is the smallest matrix that can be fed into in0, and 16x16 is the 
smallest matrix that can be fed into in1.

If the input matrices fed into the engine are "shorter" than 8x16, for example 1x16, the engine will still perform 8x16 x 16x16 = 8x16, but the effective throughput will be 1/8. 
Thus, for 1x16 x 16x16 matricies, the effective throughput is 0.5 TFLOP per matrix engine.

## Dot Product

## Reduction: Addition and Max
The WH matrix engine performs 16x16 reduce max/average operations in a single cycle. \
This is 2*16\*16 multiply + adds in a single cycle. At 1GHz, this is 0.512 TFLOPs per matrix engine. 


## Eltwise: Add, Sub, Mul
The WH matrix engine performs 8x16 elementwise addition/subtraction/multiplication in a single cycle. \
This is 8\*16 (multiply or adds, not both) in a single cycle. At 1Ghz, this is 0.128 TFLOPs per matrix engine. \
Elementwise addition and subtraction do not use MATH_FIDELITY; however, Elementwise multiplication does use MATH_FIDELITY, and TFLOPs are calculated by dividing by the MATH_FIDELITY value.

LoFi -> 0.128 TFLOPs \
HiFi2 -> 0.064 TFLOPs \
HiFi3 -> 0.043 TFLOPs \
HiFi4 -> 0.032 TFLOPs
