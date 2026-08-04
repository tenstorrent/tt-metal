.. _custom_sfpu:

Writing Custom SFPU Operations using SFPI
=========================================

The SFPU (Special Function Processing Unit) is a programmable vector engine designed to perform more complicated mathematical operations than the matrix engine. It is used internally for many standard compute functions, such as ``sin``, ``cos``, ``exp``, ``relu``, and ``tanh``. Programming the SFPU directly allows you to implement custom mathematical operations beyond those provided by the standard library, which can be useful for specialized workloads.

The SFPU operates on 32-wide vectors and supports FP32 and INT32 arithmetic, with full instruction predication. Note that FP32 arithmetic on the SFPU does not fully comply with the IEEE 754 standard, but it is a practical implementation for most use case, and is more compliant than the matrix engine (FPU)

SFPI (SFPU Interface) is a library and compiler for writing custom SFPU programs in C++. SFPI provides a high-level abstraction over the SFPU's instruction set, making it easier to develop and maintain code running on the SFPU.

Please also see the :ref:`Compute Engines and Data Flow within Tensix<compute_engines_and_dataflow_within_tensix>`, the :ref:`LLK<llk>` documentation, and the :ref:`Internal structure of a Tile<internal_structure_of_a_tile>` documentation for more context on how the SFPU fits into the overall architecture.

.. toctree::
   :maxdepth: 1

   custom_sfpi_add
   custom_sfpi_smoothstep
