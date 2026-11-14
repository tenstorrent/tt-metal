Compute Kernel API Ordering Checker
===================================

Overview
--------

The Compute Kernel API Ordering Checker (``tt_metal/tt-llk/infra/check_compute_api_order.py``) is a static
typestate and dataflow analysis tool that validates compute-kernel and LLK API ordering rules across all
code paths without requiring Tenstorrent hardware.

Compute kernels drive the unpacker (``UNPACK``), math engine (``MATH``: ``Fpu`` and ``Sfpu``), and packer
(``PACK``) of each Tensix core through ``tt_metal/hw/inc/api/compute/``, which wraps the architecture-specific
LLK APIs (``tt_metal/hw/ckernels/<arch>/metal/llk_api/``). Because these APIs are stateful, calling an
operation without its matching init, calling an operation with compile-time parameters that differ from
its preceding init, clobbering an active init with a different init or data-format reconfig, or omitting
a required ``*_uninit`` call causes silent data corruption or device hangs.

Unlike runtime checks (the LLK Sanitizer and ``TT_METAL_LLK_ASSERTS=1``), which observe only the branches
executed by a hardware test, ``check_compute_api_order.py`` inspects the static per-thread call graph across
both Wormhole (``wormhole_b0``) and Blackhole (``blackhole``).

Diagnostic Rules
----------------

The checker reads the declarative rule table in ``tt_metal/tt-llk/infra/compute_api_rules.yaml`` and reports
six rule categories per kernel, architecture, and Tensix thread (``UNPACK``, ``MATH``, ``PACK``):

1. ``missing-init`` (error):
   An operation is executed with no preceding matching initialization call on some control-flow path
   reaching it.
2. ``arg-mismatch`` (error):
   An init call's compile-time arguments or template parameters differ from those of the operation it
   precedes (for example, ``#59276`` ``AccumulateToDest`` mismatch between ``llk_unpack_A_init`` and
   ``llk_unpack_A``, ``BroadcastType`` mismatch, ``EltwiseBinaryType`` mismatch, or ``#39945``
   ``reconfig_data_format(in0, in1)`` operand-order reversal before ``matmul_tiles(in0, in1, ...)``).
   Arguments that are not known at compile time are recorded as ``unknown(...)`` and never guessed.
3. ``init-conflict`` (error):
   An init (or state-invalidating reconfiguration) for a different operation on the same execution unit
   (``Unpack``, ``Fpu``, ``Sfpu``, ``Pack``) intervenes between an init and its operation without
   re-initialization (for example, ``#61146`` ``pack_reconfig_data_format`` after ``tilize_init`` on
   Blackhole, ``#39184`` unpacker mode overwrite before ``unpack_tilizeA_B_block``, or ``#50381`` SFPU
   programmable-constant overwrite between ``exp_tile_init`` and ``exp_tile``).
4. ``missing-uninit`` (error):
   An operation from the curated set requiring uninitialization (``tilize``, ``untilize``,
   ``pack_untilize``, ``fast_tilize``, ``reduce``, ``reduce_block_max_row``) transitions to another
   operation or exits the kernel without its matching ``*_uninit`` call (``#52175``).
5. ``deprecated-init`` (error):
   Use of a deprecated compute or LLK init/operation API (cross-referenced with ``.github/deprecations.json``
   and ``[[deprecated]]`` header annotations, such as ``untilize_init_short`` or ``init_bcast``).
6. ``loop-invariant-init`` (warning only):
   A hoistable init (``Hoistable::Yes`` in ``sanitizer/operation.h``) with loop-invariant arguments is
   called inside a loop body that contains no competing init or uninit on the same execution unit.

Running the Checker Locally
---------------------------

Check one or more compute kernel files across both Wormhole and Blackhole:

.. code-block:: bash

   python3 tt_metal/tt-llk/infra/check_compute_api_order.py \
       tt_metal/programming_examples/eltwise_binary/kernels/compute/tiles_add.cpp

Check a single architecture or emit structured JSON output:

.. code-block:: bash

   python3 tt_metal/tt-llk/infra/check_compute_api_order.py \
       --arch blackhole \
       --json \
       ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_kernel.cpp

Scan the entire repository corpus (``ttnn/``, ``models/``, ``tests/``, ``tt-train/``) in non-blocking
advisory mode:

.. code-block:: bash

   python3 tt_metal/tt-llk/infra/check_compute_api_order.py --corpus --advisory

Supply a ``compile_commands.json`` database (generated with ``TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1`` or
downloaded from the ``kernel-clang-tidy`` workflow artifacts) to provide per-kernel ``-D`` defines and
``-I`` include paths:

.. code-block:: bash

   python3 tt_metal/tt-llk/infra/check_compute_api_order.py \
       --compile-commands compile_commands.json \
       --corpus

Run the unit and regression test suite (including the ``sanitizer/operation.h`` agreement check):

.. code-block:: bash

   pytest tt_metal/tt-llk/infra/tests/test_check_compute_api_order.py

Adding or Updating a Rule
-------------------------

All API ordering rules live in ``tt_metal/tt-llk/infra/compute_api_rules.yaml``—never hardcoded in the
checker script:

1. **LLK Sanitizer alignment (``sanitizer_operations``)**:
   When adding or modifying an ``Operation*`` struct in ``tt_metal/tt-llk/tools/include/sanitizer/operation.h``,
   update the corresponding entry under ``sanitizer_operations`` in ``compute_api_rules.yaml`` with its
   ``exu`` (``Unpack``, ``Fpu``, ``Pack``), ``hoistable`` flag, ``state_fields``, and ``source`` line.
   ``test_rule_table_matches_llk_sanitizer_operation_header`` enforces 1-to-1 agreement.
2. **LLK or SFPU operation rules (``llk_operations`` / ``sfpu_operations``)**:
   Add the operation family under ``llk_operations`` (for ``Unpack``, ``Fpu``, ``Pack``) or
   ``sfpu_operations`` (for ``Sfpu`` on ``MATH``), specifying:

   - ``thread`` (``UNPACK``, ``MATH``, ``PACK``) and ``exu`` (``Unpack``, ``Fpu``, ``Sfpu``, ``Pack``)
   - ``exec_calls`` and ``required_inits``
   - ``matching_args`` (mapping ``init_template_index`` / ``init_arg_index`` to ``exec_template_index`` /
     ``exec_arg_index`` and their defaults)
   - ``requires_uninit`` and ``uninit_calls`` (only for operations that leave sticky hardware state)
   - ``source`` (must point to an existing repository header, doc file, or issue number)

3. **Compute API wrappers (``compute_api_wrappers``)**:
   Map high-level wrapper functions in ``tt_metal/hw/inc/api/compute/*.h`` to their per-thread LLK calls
   and parameter placeholders (``$tpl0``, ``$arg0``, etc.).
