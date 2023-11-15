.. _Getting started for SFPU Kernel Operator devs:

SFPU Operators
===============
**Note**: This document is Work In Progress

SFPU provides resources for single-cycle fused-multiply add (MAD), floating-point specific operations on the Tenstorrent Tensix cores.
It exposes the underlying hardware registers and assembly language of the SFPU in each Grayskull (GS) or Wormhole B0 (WH) architectures as
simple C API. SFPU can be thought of as a math co-processor which has specialize math, vector, SIMD units.

This section of manual shows how to develop DNN operators, activation functions, trigonometric and hyperbolic functions for
various applications on the Tenstorrent Tensix cores by leveraging the SFPU.

**True North** for the TT Metal project is to have a ecosystem of operator implementations suitable for eager mode execution of neural-network graphs,
general computational **DAG**\s, or provide a backend of hosting a compiler for either of above uses.

Requirements for TT Metal
-------------------------
We require all operator implementations on Tensix (Tenstorrent SFPU core) to support:

- Grayskull (GS) architecture
- Wormhole B0 (WH) architecture
- performant implementation on device
- support single core Tensix, multi-core Tensix implementation
- support row-major (RM) and tile (Tile) layout of Tensor I/O to operators
- support broadcast with scalar arguments when sensible across dimension of other input Tensors
- support profiling operator on device
- accurate implementation on device
- provide Python bindings for operators
- test operators on CPU and device
    - test Python API
    - test TT Metal API


Operator Implementation Methodology
------------------------------------

1. Operators should be selected from the list maintained by the Tenstorrent team;
2. General idea of operator implementation is to get:
    -  accurate implementation of operator on device
    -  performant implementation of operator on device
    -  support the functionality for requisite datatypes (BF16, FP32, FP16, FP8, BF8, UINT32, UINT16, INT32) as required

2. Operators should have reference PyTorch, Theano, or Tensorflow implementation
    -  we can usually look at the open-source PyTorch, Theano or Tensorflow implementation of the operator and use a similar numerical approximation method to implement the kernel
    -  we typically don't have to invent anything new for kernel support
    -  good understanding of the

3. Operators should have various tests:
    - all tests compare against CPU implementation for acceptable metrics:
        - Pearson-Correlation Coefficient (PCC) and/or,
        - threshold relative and
        - absolute tolerances
    - sweep testing on device
        - support multiple tensor shapes (rank-4)
        - support multiple data types
        - support any special modes of the operator
            - this can be special flags/attributes of operator etc.
            - for example sort operator could have attribute of sorting ascending or descending passed as boolean value

Code Organization
-----------------
**Note**": Our codebase is being actively developed and the operator infrastructure is subject to an uplift.

.. list-table:: Code Organization
   :widths: 25 25 50
   :header-rows: 1

   * - Folder
     - File
     - Description
   * - tt_eager/tt_dnn/op_library/
     - tt_eager/tt_dnn/op_library/your_operator/\*.cpp, \*.hpp
     - TT DNN implementation of operator referencing SFPU
   * - tt_eager/tt_dnn/op_library/eltwise_unary/ or tt_eager/tt_dnn/op_library/eltwise_binary/
     - tt_eager/tt_dnn/op_library/eltwise_(unary|binary)/\*.cpp, \*.hpp
     - TT DNN implementation of basic unary, binary ops referencing SFPU
   * - tt_eager/tt_dnn/op_library/composite/
     - tt_eager/tt_dnn/op_library/composite/\*.cpp, \*.hpp
     - TT DNN implementation of composite ops based on SFPU
   * - tt_metal/hw/ckernels/grayskull/
     - tt_metal/hw/ckernels/grayskull/common/inc/ckernel_sfpu.h,
     - GraySkull SFPU Kernel implementation programming registers of SFPU
   * - tt_metal/hw/ckernels/wormhole_b0/
     - tt_metal/hw/ckernels/wormhole_b0/common/inc/ckernel_sfpu.h,
     - *Wormhole B0* SFPU Kernel implementation programming registers of SFPU

Composite Operators
-------------------
We use a technique called *composite ops* which help us to write composite kernels for cases where peak performance
can be a secondary goal but getting operator to run on device is primary goal. The notion of composite ops is to
write a mathematical description quickly in compliance with the above principles:

Sample implementation of a hard swish operator (product of elementwise input tensor with hard sigmoid) is implemented as follows,

::

     /Ref: PyTorch
     //hard swish(x) = x*hardsigmoid(x,scale,shift)
     Tensor hardswish(const Tensor& a,float scale,float shift) {
         Tensor a_sigmoid = hardsigmoid(a,scale,shift);
         Tensor result_sq = mul(a_sigmoid,a);
         return std::move(result_sq);
     }
