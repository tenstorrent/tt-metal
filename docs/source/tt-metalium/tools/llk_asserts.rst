LLK Asserts
===========

Overview
--------

LLK Asserts provide runtime validation checks within the tt-llk codebase that implements core tensor operations. These asserts validate critical assumptions about tensor dimensions, data formats, and hardware configuration parameters within the device-side compute stack, executing on the accelerator hardware itself.

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

Lightweight Kernel Asserts at present provide more detailed information about the assertion failure.

It is recommended to enable at least one of these mechanisms for comprehensive debugging:

.. code-block:: bash

   # Option 1: Enable LLK Asserts with Lightweight Kernel Asserts
   export TT_METAL_LLK_ASSERTS=1
   export TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1

   # Option 2: Enable LLK Asserts with Watcher (more comprehensive debugging)
   export TT_METAL_LLK_ASSERTS=1
   export TT_METAL_WATCHER=1

   # Option 3: Enable LLK Asserts only
   export TT_METAL_LLK_ASSERTS=1

When an LLK assert fails, it triggers:

1. An ``ebreak`` instruction (in case of Lightweight Kernel Asserts or LLK Asserts only)
2. Watcher assertion (in case of Watcher)

This causes the kernel to hang.
If Watcher is used, the assertion message will be printed to stderr and the watcher log file.
If Lightweight Kernel Asserts are used or LLK Asserts only are used, use ``tt-triage`` to analyze the failure state.

LLK_ASSERT Macro
----------------

There is a single macro used throughout the tt-llk library:

.. code-block:: cpp

   LLK_ASSERT(condition, message)

Its runtime behavior depends on the compile context, which is determined by flags injected during JIT build (``tt_metal/jit_build/build.cpp``):

- **Disabled** (``ENABLE_LLK_ASSERT`` not defined): the condition is evaluated only as an unevaluated ``sizeof`` expression — fully type-checked and name-resolved at compile time, but with zero runtime overhead.

- **Assert-only context** (``ENV_LLK_INFRA`` or ``ENABLE_LLK_ASSERT_ONLY`` defined): if the condition is false, an ``ebreak`` instruction is executed directly and the TRISC hangs. In JIT builds, ``ENABLE_LLK_ASSERT_ONLY`` is set when ``TT_METAL_LLK_ASSERTS=1`` and **neither** ``TT_METAL_WATCHER`` **nor** ``TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS`` is enabled.

- **tt-metal context** (``ENABLE_LLK_ASSERT`` defined but neither of the above): delegates to the standard ``ASSERT(condition)`` macro from ``api/debug/assert.h``. This covers two sub-cases:

  - *Lightweight Kernel Asserts enabled* (``TT_METAL_LLK_ASSERTS=1`` + ``TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1``, no Watcher): ``ASSERT`` triggers ``ebreak`` via the lightweight-assert path, allowing ``dump_lightweight_asserts.py`` to retrieve the call stack and local variables.
  - *Watcher enabled* (``TT_METAL_LLK_ASSERTS=1`` + ``TT_METAL_WATCHER=1``): ``ASSERT`` reports the failure through the Watcher mechanism, printing a message to stderr and the watcher log file.

The macro is defined in ``tt_metal/tt-llk/common/llk_assert.h``.

What LLK Asserts Validate
--------------------------

LLK asserts perform runtime validation of low-level kernel operations. Common checks include:

**Hardware Configuration (most common assert trigger)**
   - Unpacker format conversion is supported for the source/destination data format pair and FP32 accumulation setting (``is_unpacker_format_conversion_supported_fp32_acc``, ``is_unpacker_format_conversion_supported_dest``)
   - Packer-to-L1 data format conversion is supported for the given source/destination pair (``is_packer_to_L1_conversion_supported``)
   - Multi-tile pack respects per-format tile count limits (e.g. ≤ 4 tiles for Float32, ≤ 8 for Float16/Float16_b)
   - ``face_r_dim`` and ``narrow_tile`` parameters are not set in contexts where they are unused (defensive correctness)
   - Face row dimension is one of the valid values: 1, 2, 4, 8, or ``FACE_R_DIM``

**Tensor Dimension Validation**
   - Number of faces is 1, 2, or 4 (``num_faces``, ``unpA_num_faces``, ``unpB_num_faces``)
   - Tile dimensions match expected values (TILE_R_DIM, TILE_C_DIM)
   - Tile shape is valid for tile-dependent operations (``validate_tensor_shape_tile_dependent_ops_``)

**Matrix Multiplication Constraints**
   - MM throttling requires full 32×32 tiles with partial faces disabled
   - 16×16 × 16×16 matmul (1-face × 1-face) is not supported
   - Transpose with 32×16 input tiles is not supported

**Broadcast and Transpose Constraints**
   - Column broadcast requires full 32×32 tiles (``num_faces == 4``)
   - Broadcast with 32×16 (narrow) tiles is not supported for column or row modes
   - Scalar broadcast is not compatible with transpose of faces
   - Transpose requires ``face_r_dim == 16`` and ``num_faces`` of 4 or 1

**Math Pipeline Constraints**
   - Math fidelity higher than LoFi is only valid with element-wise multiply (``ELWMUL``)
   - Element-wise binary type must be ``ELWADD``, ``ELWSUB``, or ``ELWMUL``
   - Reduce narrow tile: ``num_faces`` of 4 implies full-width 32×32, not a narrow tile

**Destination Register Addressing**
   - Destination tile index does not exceed ``get_dest_max_tiles()`` for SFPU unary, binary, and ternary operations
   - Destination address does not overflow half-dest in TopK operations (``addr < DEST_REGISTER_HALF_SIZE``)
   - Face index is in range (``face_index < 4``)

**L1 Memory**
   - Packer destination address is within the valid L1 memory region (``is_valid_L1_address``)
   - Unpacker base address is within the valid L1 memory region

**Synchronization**
   - Semaphore index is within bounds (``index < semaphore::NUM_SEMAPHORES``)
   - Semaphore value is not already at max before increment
   - Semaphore value is not already at zero before decrement

Examples from tt-llk Code
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

.. code-block:: text

   tt_metal/tt-llk/
   ├── common/
   │   └── llk_assert.h              # Assert macro definition (single source of truth)
   ├── tt_llk_blackhole/
   │   ├── llk_lib/
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

CI/CD Integration
-----------------

LLK asserts are fully integrated into the tt-metal CI/CD system through the ``enable-llk-asserts`` input parameter. When set to ``true``, the ``setup-job`` composite action exports ``TT_METAL_LLK_ASSERTS=1`` into the test environment, causing JIT-compiled kernels to include ``LLK_ASSERT`` checks at runtime.

**Workflows Accepting ``enable-llk-asserts``**

The following key workflow files accept the ``enable-llk-asserts`` boolean input and can be triggered manually via ``workflow_dispatch`` with the checkbox enabled:

- ``sanity-tests.yaml`` — broad sanity suite (fast dispatch, models, ops, TTNN, profiler)
- ``sanity-tests-debug.yaml`` — nightly debug run (also the scheduled nightly LLK assert run)
- ``blackhole-post-commit.yaml`` — Blackhole-specific test matrix (models, ops, TTNN, UMD, multi-card)

Other workflows also accept this parameter (e.g. ``ttnn-post-commit.yaml``, ``ops-post-commit.yaml``, ``models-post-commit.yaml``, ``tt-metal-l2-nightly.yaml``, and others), but the three above are the most useful entry points for validating new asserts.

**Nightly Runs with LLK Asserts**

The ``sanity-tests-debug.yaml`` workflow runs automatically every night **at 2:00 AM UTC** with LLK asserts enabled. It spawns four test jobs — Wormhole and Blackhole on both Ubuntu 22.04 and Ubuntu 24.04:

- ``wormhole-nightly-debug-ubuntu-22`` — runs ``sanity-tests.yaml`` with ``enable-llk-asserts: true``
- ``blackhole-nightly-debug-ubuntu-22`` — runs ``blackhole-post-commit.yaml`` with ``enable-llk-asserts: true``
- ``wormhole-nightly-debug-ubuntu-24`` — runs ``sanity-tests.yaml`` with ``enable-llk-asserts: true``
- ``blackhole-nightly-debug-ubuntu-24`` — runs ``blackhole-post-commit.yaml`` with ``enable-llk-asserts: true``

The same workflow can also be triggered on demand via ``workflow_dispatch`` with the ``enable-llk-asserts`` checkbox.

Adding New LLK_ASSERT — Required Procedure
-------------------------------------------

After adding a new ``LLK_ASSERT`` to the codebase, you must validate it does not trigger false positives on supported configurations. Follow this procedure:

1. **Run CI with LLK asserts enabled**

   Trigger either (or both) of these workflows manually via ``workflow_dispatch`` with the ``enable-llk-asserts`` checkbox enabled:

   - `Sanity tests <https://github.com/tenstorrent/tt-metal/blob/main/.github/workflows/sanity-tests.yaml>`_ — exercises a broad set of configurations including fast dispatch, models, ops, TTNN, and profiler regression.
   - `Blackhole post-commit <https://github.com/tenstorrent/tt-metal/blob/main/.github/workflows/blackhole-post-commit.yaml>`_ — validates Blackhole-specific code paths (models, ops, TTNN, UMD, multi-card).

2. **For each test failure, choose one of two resolutions:**

   a. **Fix the failure (preferred)** — When an ``LLK_ASSERT`` fires, the root cause is almost always in **kernel code**, not in the test. The assert is catching an invalid parameter or configuration being passed to the LLK API. Fix the kernel code to pass valid values.

   b. **Skip with tracking** — If the failure cannot be fixed immediately, mark the test to be skipped when LLK asserts are enabled, with a reference to a tracking issue:

      .. code-block:: python

         from models.common.utility_functions import skip_with_llk_assert

         @skip_with_llk_assert("Hit LLK_ASSERT for unpacker data format conversion. Issue: #XXXXX")
         def test_something(...):
             ...

      The ``skip_with_llk_assert`` decorator (from ``models/common/utility_functions.py``) is a ``pytest.mark.skipif`` that activates whenever ``TT_METAL_LLK_ASSERTS=1``. Always include the GitHub issue number in the reason string for tracking.

3. **Ensure nightly coverage** — Once all failures are resolved or skipped, the new assert will be continuously validated by the nightly ``sanity-tests-debug.yaml`` run at 2:00 AM UTC.

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
3. If Lightweight Kernel Asserts are enabled or LLK Asserts only are enabled, use ``dump_lightweight_asserts.py`` to see call stacks and local variables
4. Check the assertion message to understand what constraint was violated

.. note::

   When an LLK assert fires, the root cause is almost always in **kernel code** — the assert is validating that the LLK API was called with legal parameters. Look at the kernel code that is calling the LLK function, not at the test itself.

Common failure scenarios:

- Attempting operations on tiles with unsupported dimensions
- Passing incompatible data formats between operations
- Using narrow tiles or partial faces where not supported
- Dimension mismatches in matrix multiplications

Real-World Examples
-------------------

The following issues and PRs illustrate the most common patterns of LLK assert hits observed in practice and how they were resolved.

**Issue** `#38346 <https://github.com/tenstorrent/tt-metal/issues/38346>`_ — *Create green runs of Sanity/BPC workflows when LLK_ASSERTS are enabled*

The umbrella tracking issue that motivated systematic LLK assert work. When LLK asserts were first enabled across the CI suite, a large number of pre-existing tests failed — revealing that many kernels had been silently misusing the LLK API. The resolution strategy defined was: fix kernel code where the LLK API was misused, skip tests (with ``skip_with_llk_assert``) where further investigation was needed, and make nightly runs with LLK asserts enabled a reliable green baseline going forward.

**Issue** `#39184 <https://github.com/tenstorrent/tt-metal/issues/39184>`_ — *Fix tests which hit LLK_ASSERT during unpack configuration verification*

A detailed description of the most common HW configure assert pattern: the unpacker is configured during HW configure (or a reconfigure phase) with specific ``face_r_dim`` / ``num_faces`` values, but a different configuration is then passed to ``init`` or the execution block — causing the assert to fire on the mismatch. The issue includes a full reproduction recipe and an example ``dump_lightweight_asserts.py`` callstack:

.. code-block:: text

   #0 llk_unpack_tilizeA_B_init<false, 1, false, true>()
      at llk_unpack_tilize_api.h:97
   #1 ckernel::tilizeA_B_reduce_init<false, true>()
      at tilize.h:79
   #2 chlkc_unpack::unpack_main()
      at compute_pool_2d.cpp:87

   Arguments: operandA=1, operandB=2, ct_dim=1, num_faces=2,
              unpA_face_r_dim=4, unpB_face_r_dim=1

The callstack clearly points to the compute kernel (``compute_pool_2d.cpp``) as the source of the problem, not the test.

**PR** `#39945 <https://github.com/tenstorrent/tt-metal/pull/39945>`_ — *Fix LLK_ASSERT hits for sdpa tests* *(kernel code fix)*

Addressed LLK assert failures in SDPA/MLA kernels caused by unpacker configuration mismatches. The fixes were entirely in kernel code:

1. **Fixed operand ordering** — ``reconfig_data_format(in0, in1)`` in ``sdpa_flash_decode.cpp`` was reordered to ``reconfig_data_format(in1, in0)`` to match the ``state_configure(in1, in0)`` call used by the matmul path.
2. **Added missing reconfigure calls** — ``reconfig_data_format`` and ``pack_reconfig_data_format`` calls were added in ``compute_common.hpp`` to keep unpacker/packer configuration consistent before matmul init and execution in the SDPA inner loop.

After the kernel fixes, all ``skip_with_llk_assert`` decorators were removed from the SDPA/MLA test files.

**PR** `#40282 <https://github.com/tenstorrent/tt-metal/pull/40282>`_ — *Enrich LLK API with more customized hw (re)configure options* *(LLK API extension)*

Addressed LLK assert failures in pooling-related kernels (``compute_pool_2d.cpp``) where the kernel legitimately needs to change ``face_r_dim`` or ``num_faces`` dynamically — for example, using different geometry for the last block. Because the existing LLK HW configure/reconfig API always derived these values from the circular buffer metadata, there was no way to express the kernel's intent without triggering an assert.

The fix added new LLK pack/unpack reconfig API overloads that accept explicit ``face_r_dim`` and ``num_faces`` parameters, allowing kernels to configure hardware with geometry that differs from the CB metadata when there is a documented reason to do so. After the API was extended, ``skip_with_llk_assert`` was removed from multiple pool and reduction test files (``test_avgpool2d.py``, ``test_upsample.py``, ``test_rotate.py``, ``test_grid_sample.py``, ``test_adaptive_pool2d.py``, ``test_reduction.py``).

See Also
--------

- :doc:`lightweight_kernel_asserts` - User-level kernel assertions
- :doc:`watcher` - Device monitoring and debug
- :doc:`triage` - Failure analysis and debugging
