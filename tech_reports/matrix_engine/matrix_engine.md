# Matrix Engine

## Introduction

The matrix engine supports the following operations: matrix mult, dot product, reduction, eltwise add/sub/mul, and tranpose_xy.

## Matrix Mult 

The WH matrix engine performs 8x16 x 16x16 = 8x16 in a single cycle. 
This is 2*8*16*16 = 4096 muladds in a single cycle. At 1GHz, this is 4 TFLOPs per matrix engine.
The 8x16 is the smallest matrix that can be fed into in0, and 16x16 is the 
smallest matrix that can be fed into in1.

If the input matrices fed into the engine are "shorter" than 8x16, for example 1x16, the engine will still perform 8x16 x 16x16 = 8x16, but the effective throughput will be 1/8. 
Thus, for 1x16 x 16x16 matricies, the effective throguhput is 0.5 TFLOP per matrix engine.

## Dot Product

## Reduction: Addition and Max


## Eltwise: Add, Sub, Mul