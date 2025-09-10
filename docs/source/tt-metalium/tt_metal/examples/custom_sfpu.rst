.. _custom_sfpu:

Writing Custom SFPU Operations
==============================

The SFPU (Special Function Processing Unit) is a programmable vector engine for efficient mathematical computation. Many standard compute functions (e.g., ``sin``, ``cos``, ``exp``, ``relu``, ``tanh``) use the SFPU internally. Direct SFPU programming allows you to implement custom mathematical operations not covered by the standard library, enabling support for specialized workloads and many other applications.

The SFPU is a 32-wide vector engine, capable of performing FP32 and INT32 arithmetics and supports full instruction predication (FP32 on the SFPU does not fully comply with the IEEE 754 standard, but it provides a very good implementation).

Please also see the :ref:`Compute Engines and Data Flow within Tensix<compute_engines_and_dataflow_within_tensix>`, the :ref:`LLK<llk>` documentation, and the :ref:`Internal structure of a Tile<internal_structure_of_a_tile>` documentation for more context on how the SFPU fits into the overall architecture.

.. toctree::
   :maxdepth: 1

   custom_sfpu_add
