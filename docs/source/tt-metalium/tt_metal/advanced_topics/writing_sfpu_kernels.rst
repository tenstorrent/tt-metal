.. _writing_sfpu_kernels:

Writing SFPU kernels
====================

The SFPU (vector engine) is designed to perform complex mathematical operations. As a vector engine, writing SFPU code feels like classicial GPU programming - with one important catch - the SFPU itself does not have the ablity to execute control flow, instead the Math core is responsible to control the exact instructions that is dispatched to the vector engine. And per-lane flow is done by lane masking.

This documentation provides guidance on wiring your own operations using the SFPU.

.. warning::

    The width, built-in variables and supported operations of the SFPU may change between generations. Making SFPU kernels (potentially) not portable across generations of Tensix Processors.

The anatomy of SFPU operations
------------------------------

As the vector engine only have access to the ``Dst`` registers and they are exposed via the ``dst_reg`` global variable.
