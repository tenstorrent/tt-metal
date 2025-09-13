.. _Custom_SFPI_Smoothstep:

Smoothstep using SFPI
=====================

Smoothstep is a commonly used function in graphics and procedural generation to interpolate smoothly between two values. It is defined as the following piecewise function. It is a simple, familiar and widely used yet non-trivial function that can benefit from acceleration on the SFPU (and non-SFPU) hardware.

.. math::

    \operatorname{smoothstep}(e_0, e_1, x) =
    \begin{cases}
    0, & x \leq e_0, \\[6pt]
    1, & x \geq e_1, \\[6pt]
    \left( \dfrac{x - e_0}{e_1 - e_0} \right)^2 \bigl(3 - 2 \tfrac{x - e_0}{e_1 - e_0}\bigr),
    & e_0 < x < e_1 .
    \end{cases}

Please refer to the `OpenGL documentation <https://registry.khronos.org/OpenGL-Refpages/gl4/html/smoothstep.xhtml>`_ for more details on the smoothstep function.

The full source code for this example is available under the ``tt_metal/programming_examples/custom_sfpi_smoothstep`` directory.

Building the example can be done by adding a ``--build-programming-examples`` flag to the build script or adding the ``-DBUILD_PROGRAMMING_EXAMPLES=ON`` flag to the cmake command and results in the ``metal_example_custom_sfpi_smoothstep`` executable in the ``build/programming_examples`` directory. For example:

.. code-block:: bash

    export TT_METAL_HOME=</path/to/tt-metal>
    ./build_metal.sh --build-programming-examples
    # To run the example
    ./build/programming_examples/metal_example_custom_sfpi_smoothstep

.. warning::

    Tenstorrent does not guarantee backward compatibility for user-implemented SFPI functions. Keep your implementations up to date with the latest Metalium releases. APIs that call low-level SFPI functions may change without notice, and SFPI specifications may also change in future hardware versions.
