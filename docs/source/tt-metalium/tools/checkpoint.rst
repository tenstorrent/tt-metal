Debug Checkpoints
=================

.. note::
   Tools are only fully supported on source builds.

Overview
--------

Debug checkpoints provide synchronized inspection points for fused kernels. When a checkpoint is
hit, all active RISCs on a core (BRISC, NCRISC, TRISC0, TRISC1, TRISC2) halt together, dump
circular buffer state via DPRINT, then proceed in unison.

This generalizes ``dprint_tensix_dest_regs``, which only synchronizes the three TRISC cores and
only dumps destination registers. Checkpoints synchronize all five RISCs and dump the full
circular buffer state visible to each one.

Enabling
--------

Checkpoints are automatically enabled when DPRINT is enabled. Set the ``TT_METAL_DPRINT_CORES``
environment variable as described in :doc:`kernel_print`:

.. code-block:: bash

    export TT_METAL_DPRINT_CORES=0,0

When DPRINT is disabled, all checkpoint macros are no-ops with zero overhead.

Usage
-----

Include ``api/debug/checkpoint.h`` in every kernel (reader, writer, and compute) that
participates in the checkpoint. All active RISCs must call ``DEBUG_CHECKPOINT`` with the same ID
at the corresponding micro-op boundary.

**Compute kernel:**

.. code-block:: c++

    #include "api/debug/checkpoint.h"

    void kernel_main() {
        // ... stage 1: unpack and compute ...

        DEBUG_CHECKPOINT(1);  // all RISCs synchronize and dump CB state

        // ... stage 2: pack output ...
    }

**Reader kernel (NCRISC):**

.. code-block:: c++

    #include "api/debug/checkpoint.h"

    void kernel_main() {
        // ... read tiles from DRAM into input CB ...

        DEBUG_CHECKPOINT(1);  // must match the compute kernel's checkpoint ID
    }

**Writer kernel (BRISC):**

.. code-block:: c++

    #include "api/debug/checkpoint.h"

    void kernel_main() {
        DEBUG_CHECKPOINT(1);  // synchronize before consuming output CB

        // ... write tiles from output CB to DRAM ...
    }

Every active RISC must call the checkpoint. If a RISC is active but does not call
``DEBUG_CHECKPOINT``, the barrier will hang.

Knobs
-----

``DEBUG_CHECKPOINT_EX`` provides compile-time knobs to control what gets dumped:

.. code-block:: c++

    DEBUG_CHECKPOINT_EX(id, num_cbs, words_per_cb, dump_dest)

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``id``
     - ``uint8_t``
     - (required)
     - Checkpoint identifier. All RISCs must use the same value.
   * - ``num_cbs``
     - ``uint8_t``
     - 0
     - Number of CBs to dump. 0 means all configured CBs.
   * - ``words_per_cb``
     - ``uint16_t``
     - 0
     - Number of uint32 words of L1 data to hex-dump per CB. 0 means metadata only.
   * - ``dump_dest``
     - ``bool``
     - false
     - If true, TRISC1 (Math) dumps destination register contents instead of skipping.

Examples:

.. code-block:: c++

    // Dump metadata for all configured CBs
    DEBUG_CHECKPOINT(1);

    // Dump first 4 CBs, 8 words of L1 data each, plus dest registers
    DEBUG_CHECKPOINT_EX(2, 4, 8, true);

Output Format
-------------

Each RISC prints a header line followed by CB metadata:

.. code-block:: text

    === CKPT 1 RISC 0 ===
    CB0 sz=128 rd=1024 wr=1152 ack=0 rcv=1
    CB16 sz=128 rd=2048 wr=2048 ack=0 rcv=0
    === CKPT 1 RISC 1 ===
    CB0 sz=128 rd=1024 wr=1152 ack=0 rcv=1
    === CKPT 1 RISC 3 ===
    (math thread, no CB access)

The RISC indices are:

- 0: BRISC
- 1: NCRISC
- 2: TRISC0 (Unpack)
- 3: TRISC1 (Math)
- 4: TRISC2 (Pack)

When ``words_per_cb > 0``, L1 data at the read pointer is printed in hex:

.. code-block:: text

    CB0 sz=128 rd=1024 wr=1152 ack=0 rcv=1
      [0] 3f800000 40000000 40400000 40800000
      [4] 40a00000 40c00000 40e00000 41000000

The CB metadata fields are:

- **sz**: FIFO size (in address-shifted units)
- **rd**: read pointer
- **wr**: write pointer
- **ack**: tiles acked (consumed)
- **rcv**: tiles received (produced)

Comparison with dprint_tensix_dest_regs
---------------------------------------

.. list-table::
   :header-rows: 1

   * - Feature
     - ``dprint_tensix_dest_regs``
     - ``DEBUG_CHECKPOINT``
   * - RISCs synchronized
     - TRISC0, TRISC1, TRISC2 only
     - All active RISCs (BRISC, NCRISC, TRISC0/1/2)
   * - What is dumped
     - Destination register contents
     - CB metadata from all RISCs (+ optional dest regs, + optional L1 data)
   * - Callable from
     - Compute kernels only
     - Any kernel, but all active RISCs must participate
   * - BRISC/NCRISC involvement
     - None
     - Full participation: they print their CB view

Use ``dprint_tensix_dest_regs`` when you only need to inspect compute output in dest registers.
Use ``DEBUG_CHECKPOINT`` when you need a consistent snapshot of the entire dataflow + compute
pipeline at a micro-op boundary.

How It Works
------------

1. **Firmware init**: Before each kernel launch, BRISC writes the ``enables`` bitmask (which
   RISCs are active) to a 12-byte struct at ``MEM_LLK_DEBUG_BASE`` in L1.

2. **Entry barrier**: Each RISC sets its bit in ``arrived_mask``. BRISC spins until all bits
   match ``participant_mask``, then sets a ``proceed`` counter. Subordinate RISCs spin on
   ``proceed``.

3. **Dump**: Each RISC prints its CB state via DPRINT. DPRINT's built-in back-pressure
   handles the 204-byte-per-thread buffer limit automatically.

4. **Exit barrier**: A second barrier ensures all RISCs finish dumping before any proceeds.

Files
-----

.. list-table::
   :header-rows: 1

   * - File
     - Purpose
   * - ``tt_metal/hw/inc/api/debug/checkpoint.h``
     - User API, barrier logic, CB dump, dest reg dump
   * - ``tt_metal/jit_build/build.cpp``
     - Adds ``-DDEBUG_CHECKPOINT_ENABLED`` when DPRINT is on
   * - ``tt_metal/hw/firmware/src/tt-1xx/brisc.cc``
     - Calls ``debug_checkpoint_init()`` before kernel launch
