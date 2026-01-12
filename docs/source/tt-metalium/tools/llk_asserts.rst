LLK Asserts
===========

Overview
--------

LLK Asserts provide validation checks within the tt-llk codebase that implements core tensor operations. These asserts validate critical assumptions about tensor dimensions, data formats, and hardware configuration parameters at the lowest level of the compute stack.

LLK asserts complement Lightweight Kernel Asserts and the Watcher tool by providing specialized validation for the low-level kernel library that handles:

- Tensor unpacking (moving data from L1 memory into source registers)
- Math operations (matrix multiplication, element-wise operations)
- Tensor packing (moving results from destination register back to L1 memory)
- Tilization and untilization (data layout transformations)

Enabling
--------

LLK Asserts are controlled independently from Lightweight Kernel Asserts and the Watcher. To enable them, set the following environment variable:

.. code-block:: bash

   export TT_METAL_LLK_ASSERTS=1  # Enable LLK Asserts. Default is `0` (disabled).

**Important:** LLK Asserts require either Lightweight Kernel Asserts or the Watcher to be enabled for proper failure reporting. It is recommended to enable at least one of these mechanisms:

.. code-block:: bash

   # Option 1: Enable LLK Asserts with Lightweight Kernel Asserts
   export TT_METAL_LLK_ASSERTS=1
   export TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1

   # Option 2: Enable LLK Asserts with Watcher (more comprehensive debugging)
   export TT_METAL_LLK_ASSERTS=1
   export TT_METAL_WATCHER=1

When an LLK assert fails, it triggers an ``ebreak`` instruction causing the kernel to hang. You can then use ``tt-triage`` to analyze the failure state.

What LLK Asserts Validate
--------------------------

LLK asserts perform runtime validation of low-level kernel operations. Common checks include:

**Tensor Dimension Validation**
   - Number of faces (must be 1, 2, or 4)
   - Tile dimensions match expected values (TILE_R_DIM, TILE_C_DIM)
   - Face dimensions are valid (FACE_R_DIM, FACE_C_DIM)

**Data Format Checks**
   - Pack/unpack source and destination formats are compatible
   - Partial face configurations are valid for the operation
   - Narrow tile parameters are supported by the operation

**Matrix Multiplication Constraints**
   - Input tile dimensions are standard (TILE_R_DIM × TILE_C_DIM)
   - Partial face operations are not used with certain configurations
   - Transpose and dimension parameters are compatible

**Memory and Configuration**
   - Block dimensions divide evenly into full dimensions
   - Address mode configurations are valid
   - Unused parameters are not set (defensive programming)

Example Assertions from tt-llk Code
------------------------------------

Here are some representative LLK asserts from the codebase:

**From llk_unpack_tilize.h:**

.. code-block:: cpp

   LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4,
              "num_faces must be 1, 2, or 4");

**From llk_math_matmul.h:**

.. code-block:: cpp

   LLK_ASSERT(
       (in0_tile_r_dim == TILE_R_DIM) &&
       (in0_tile_c_dim == TILE_C_DIM) &&
       (in1_tile_r_dim == TILE_R_DIM) &&
       (in1_tile_c_dim == TILE_C_DIM) &&
       !partial_face,
       "Matmul requires standard tile dimensions and no partial faces");

Location in Codebase
---------------------

LLK asserts are defined and used throughout the tt-llk library under:

.. code-block::

   tt_metal/third_party/tt_llk/
   ├── tt_llk_blackhole/
   │   ├── llk_lib/
   │   │   ├── llk_assert.h          # Assert macro definitions
   │   │   ├── llk_unpack_*.h        # Unpack operation asserts
   │   │   ├── llk_math_*.h          # Math operation asserts
   │   │   ├── llk_pack*.h           # Pack operation asserts
   │   │   └── ...
   │   └── common/inc/
   │       ├── cunpack_common.h      # Common unpack asserts
   │       ├── cpack_common.h        # Common pack asserts
   │       └── ...
   ├── tt_llk_wormhole_b0/
   │   └── (similar structure)
   └── tt_llk_quasar/
       └── (similar structure)

The ``LLK_ASSERT`` macro is defined in ``llk_assert.h`` and expands to either:

- An ``ebreak`` instruction (when used standalone in low-level infrastructure)
- The standard ``ASSERT`` macro (when compiled within tt-metal context)

CI/CD Integration
-----------------

LLK asserts are fully integrated into the tt-metal CI/CD system through the ``enable-llk-asserts`` parameter. This parameter is available in some workflow files and can be enabled during testing (via checkbox).

**Available Workflows**

The following workflows support LLK asserts:

- ``apc-select-tests.yaml`` - Selective APC test execution

When to Use LLK Asserts
------------------------

As a user, you typically don't need to add LLK asserts—they are already present in the system library code. However, you should enable them during development and testing to catch:

1. **Incorrect tensor operation parameters** - Catches dimension mismatches early
2. **Unsupported operation configurations** - Validates that hardware capabilities aren't exceeded
3. **Library integration issues** - Ensures operations are called with valid parameters

Performance Impact
------------------

LLK asserts have zero overhead when disabled (compiled out as no-ops). When enabled, they add conditional checks at critical points in low-level operations. The performance impact is generally acceptable for development and testing builds but should be disabled in production for maximum performance.

Debugging Failed LLK Asserts
-----------------------------

When an LLK assert fails:

1. The kernel hangs at the assertion point (``ebreak`` instruction)
2. Run ``tt-triage`` to analyze the device state
3. If Lightweight Kernel Asserts are enabled, use ``dump_lightweight_asserts.py`` to see call stacks
4. Check the assertion message to understand what constraint was violated
5. Review your operation parameters (tile dimensions, formats, etc.)

Common failure scenarios:

- Attempting operations on tiles with unsupported dimensions
- Passing incompatible data formats between operations
- Using narrow tiles or partial faces where not supported
- Dimension mismatches in matrix multiplications

Related Tools
-------------

- **Lightweight Kernel Asserts** - For user-written compute kernel assertions
- **Watcher** - Comprehensive device monitoring and debugging
- **tt-triage** - Post-mortem analysis tool for hung kernels
- **Kernel Print** - Debug output from device to host

See Also
--------

- :doc:`lightweight_kernel_asserts` - User-level kernel assertions
- :doc:`watcher` - Device monitoring and debug
- :doc:`triage` - Failure analysis and debugging
