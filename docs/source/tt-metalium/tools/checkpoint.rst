Debug Checkpoints
=================

.. note::
   Tools are only fully supported on source builds.

Overview
--------

Debug checkpoints provide synchronized inspection points for fused kernels. When a checkpoint is
hit, all active RISCs halt together, dump circular buffer state via DPRINT, then proceed in
unison. Two levels are available:

- **Single-core** (``DEBUG_CHECKPOINT``): synchronizes all 5 RISCs on one core.
- **Global** (``DEBUG_CHECKPOINT_GLOBAL``): synchronizes all RISCs on all tensix cores.

This generalizes ``dprint_tensix_dest_regs``, which only synchronizes the three TRISC cores and
only dumps destination registers.

Enabling
--------

**Checkpoints** (synchronized barriers) are enabled with:

.. code-block:: bash

    export TT_METAL_CHECKPOINT=1

Without a print backend, checkpoints act as barriers only (no dump output). To get CB dump
output, also enable DPRINT or DEVICE_PRINT:

.. code-block:: bash

    export TT_METAL_CHECKPOINT=1
    export TT_METAL_DPRINT_CORES=0,0

**Standalone dump utilities** (``debug_dump_cb``, ``debug_dump_l1``, etc.) require only DPRINT
or DEVICE_PRINT — no ``TT_METAL_CHECKPOINT`` needed:

.. code-block:: bash

    export TT_METAL_DPRINT_CORES=0,0

When neither is set, all dump functions and checkpoint macros are no-ops with zero overhead.

.. note::
   ``TT_METAL_CHECKPOINT`` is read at JIT compile time. If you toggle it, clear the kernel
   cache to force recompilation: ``rm -rf ~/.cache/tt-metal-cache``

Single-Core Checkpoints
-----------------------

Usage
^^^^^

Include ``api/debug/checkpoint.h`` in every kernel (reader, writer, and compute) that
participates in the checkpoint. All active RISCs must call ``DEBUG_CHECKPOINT`` with the same ID
at the corresponding point within the op.

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
^^^^^

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

How it works
^^^^^^^^^^^^

**L1 state.** Before each kernel launch, BRISC writes a 20-byte checkpoint struct to
``MEM_LLK_DEBUG_BASE`` in L1 (shared by all RISCs on the core):

.. code-block:: text

    participant_mask = 0x1F     // bits 0-4: BRISC, NCRISC, TRISC0, TRISC1, TRISC2
    proceed          = 0        // monotonically increasing epoch counter
    arrived[0..4]    = 0        // per-RISC arrival flag (one byte each)
    orchestrator_idx = 0        // lowest active RISC (BRISC)

**Entry barrier.** Each RISC hits ``debug_checkpoint_barrier()`` at its own pace:

1. Each RISC reads the current ``proceed`` epoch (0) and computes ``next_epoch = 1``.
2. Each RISC writes ``arrived[my_idx] = 1`` — its own byte, no contention with other RISCs.
3. The **orchestrator** (lowest active RISC, typically BRISC) polls all ``arrived[]`` bytes,
   spinning with ``invalidate_l1_cache()`` until all match ``next_epoch``.
4. All **other RISCs** spin on ``proceed``, waiting for it to equal ``next_epoch``.
5. Once the orchestrator sees all arrivals, it sets ``proceed = 1``, releasing everyone.

At this point all RISCs are synchronized — no RISC proceeds until every active RISC has arrived.

**Dump.** CB interfaces are shared L1 — all RISCs on a core see the same data. To avoid
redundant output, only one RISC prints each type of information:

- **BRISC** prints CB metadata (size, read/write pointers, acked/received counts) and
  optionally hex-dumps L1 data at the read pointer. DPRINT's built-in back-pressure handles
  the 204-byte buffer limit automatically.
- **TRISC1 (Math)** prints destination register contents if ``dump_dest=true`` (only Math can
  access dest regs). Otherwise it prints nothing.
- **NCRISC, TRISC0, TRISC2** print nothing — they still participate in the barriers but skip
  the dump since BRISC already covers the CB state.

**Exit barrier.** Same mechanism with ``next_epoch = 2``. This ensures no RISC moves past the
checkpoint until every RISC has finished printing — without it, a fast RISC could modify CBs
before a slow RISC finishes reading them.

**Why per-byte flags (not a shared bitmask).** The original design used
``arrived_mask |= (1 << my_idx)`` — a read-modify-write on a shared ``uint32_t``. If two RISCs
read the same stale value from L1 cache, they overwrite each other's bit. Per-byte flags avoid
this: each RISC writes only ``arrived[my_idx]``, a distinct byte. The orchestrator reads all
bytes but never writes to another RISC's byte.

**Why an epoch counter (not a simple flag).** If we used a 0/1 flag, the exit barrier would see
``proceed`` already at 1 (from the entry barrier) and skip the wait. The monotonically increasing
epoch (0 → 1 → 2 → ...) ensures each barrier waits for a unique value.

Global Checkpoints (Cross-Core)
-------------------------------

``DEBUG_CHECKPOINT_GLOBAL`` extends checkpoints to synchronize **all RISCs on all tensix cores**.
This is needed when a fused kernel spans multiple cores and you want a consistent snapshot of CB
state across the entire grid.

Usage
^^^^^

.. code-block:: c++

    DEBUG_CHECKPOINT_GLOBAL(id, sem_id, barrier_coord_x, barrier_coord_y, num_cores)

.. list-table::
   :header-rows: 1

   * - Parameter
     - Description
   * - ``id``
     - Checkpoint identifier (for DPRINT output labeling)
   * - ``sem_id``
     - Semaphore ID allocated by host via ``CreateSemaphore``
   * - ``barrier_coord_x``, ``barrier_coord_y``
     - Physical NOC coordinates of the coordinator core for the cross-core barrier.
       These only affect synchronization, not what gets printed.
   * - ``num_cores``
     - Total number of cores participating

**Host setup:**

.. code-block:: c++

    // Allocate semaphore on all participating cores
    CoreRange cores({0, 0}, {0, 1});  // 2 cores
    uint32_t sem_id = CreateSemaphore(program, cores, 0);

    // Get coordinator's physical NOC coordinates (for the barrier, not for printing)
    CoreCoord barrier_coord = device->worker_core_from_logical_core({0, 0});

    // Pass to all kernels as runtime args
    SetRuntimeArgs(program, kernel, core, {
        ...,
        sem_id, barrier_coord.x, barrier_coord.y, num_cores
    });

**Kernel usage** (all RISCs on all cores must call with the same args):

.. code-block:: c++

    #include "api/debug/checkpoint.h"

    void kernel_main() {
        uint32_t sem_id = get_arg_val<uint32_t>(3);
        uint32_t barrier_coord_x = get_arg_val<uint32_t>(4);
        uint32_t barrier_coord_y = get_arg_val<uint32_t>(5);
        uint32_t num_cores = get_arg_val<uint32_t>(6);

        // ... work ...

        // All cores synchronize here, then each core prints its OWN local CB state
        DEBUG_CHECKPOINT_GLOBAL(1, sem_id, barrier_coord_x, barrier_coord_y, num_cores);
    }

How it works
^^^^^^^^^^^^

The global checkpoint layers a cross-core NOC semaphore barrier around the single-core
intra-core barriers described above:

.. code-block:: text

    DEBUG_CHECKPOINT_GLOBAL:
      ┌─ intra-core barrier ──── all 5 RISCs on THIS core sync ─────┐
      │  ┌─ cross-core barrier ── BRISC on ALL cores sync ────────┐  │
      │  │                        (NOC semaphore)                 │  │
      │  └────────────────────────────────────────────────────────┘  │
      ├─ intra-core barrier ──── BRISC releases other RISCs ────────┤
      │                                                              │
      │  DUMP: each RISC prints its own local CB state               │
      │                                                              │
      ├─ intra-core barrier ──── all 5 RISCs finish dumping ────────┤
      │  ┌─ cross-core barrier ── BRISC on ALL cores sync ────────┐  │
      │  └────────────────────────────────────────────────────────┘  │
      └─ intra-core barrier ──── final release ─────────────────────┘

The single-core ``DEBUG_CHECKPOINT`` is the same structure but without the cross-core
barrier steps. Only BRISC participates in the NOC cross-core operations — TRISC and NCRISC
threads wait via the intra-core barriers that bracket the cross-core phase.

**Step-by-step (example with 2 cores):**

1. **Intra-core barrier.** Each core runs the single-core barrier independently
   (per-byte ``arrived[]`` flags + epoch counter at ``MEM_LLK_DEBUG_BASE``). After this,
   all 5 RISCs on each core are halted together — but the two cores are NOT yet
   synchronized with each other.

2. **Cross-core barrier (BRISC only).** Only BRISC on each core participates. The other
   4 RISCs are blocked at the next intra-core barrier (step 3), waiting for BRISC.

   - Both BRISCs reset their local semaphore to 0 (clears stale values from any
     previous global checkpoint).
   - Both BRISCs atomically increment the **coordinator's** semaphore via NOC:
     ``noc_semaphore_inc(coordinator_sem_addr, 1)``. Both target the same physical L1
     address on the coordinator core. ``noc_semaphore_inc`` is a hardware atomic.
   - **Coordinator BRISC:** Its local semaphore IS the one being incremented.
     Calls ``noc_semaphore_wait_min(local_sem, num_cores)`` — a local spin, no NOC reads.
   - **Non-coordinator BRISC:** Polls the coordinator's semaphore via
     ``noc_async_read`` into its own local semaphore copy, checking until the value
     reaches ``num_cores``.

3. **Intra-core barrier.** BRISC has returned from the cross-core barrier. The other
   4 RISCs were spinning here. BRISC's arrival advances the epoch, releasing them.
   All 10 RISCs (5 per core × 2 cores) are now synchronized.

4. **Dump.** On each core, BRISC prints the CB state (once per core — CBs are shared L1).
   If ``dump_dest=true``, TRISC1 also prints dest registers. Other RISCs print nothing.

5. **Intra-core barrier.** Ensures all RISCs on each core finish printing.

6. **Cross-core barrier.** Same mechanism as step 2. Ensures core 0 doesn't proceed
   past the checkpoint while core 1 is still printing.

7. **Final intra-core barrier.** Releases all RISCs after the cross-core exit barrier.

Output Format
-------------

BRISC prints a header line followed by CB metadata (once per core):

.. code-block:: text

    === CKPT 1 CBs ===
    CB0 sz=128 rd=1024 wr=1152 ack=0 rcv=1
    CB16 sz=128 rd=2048 wr=2048 ack=0 rcv=0

When ``dump_dest=true``, TRISC1 (Math) also prints destination register contents:

.. code-block:: text

    === CKPT 1 dest regs ===
    ...

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

Standalone Dump Utilities
-------------------------

In addition to the full checkpoint barrier, standalone dump functions are available for quick
inspection of individual CBs or arbitrary L1 memory. These can be called from any kernel at any
point — no barrier or synchronization required. Include ``api/debug/dump.h``.

``debug_dump_cb(cb_id, num_words)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Prints CB metadata and optionally raw hex data starting at the read pointer. Available on
BRISC, NCRISC, TRISC0, and TRISC2 (not TRISC1/Math, which cannot access CB interfaces).

.. code-block:: c++

    #include "api/debug/dump.h"

    debug_dump_cb(0);       // CB0 metadata only
    debug_dump_cb(0, 8);    // CB0 metadata + 8 hex words from read pointer
    debug_dump_cb(16, 4);   // CB16 metadata + 4 hex words

Output:

.. code-block:: text

    CB0 sz=128 rd=1024 wr=1152 ack=0 rcv=1
      [0] 3f800000 40000000 40400000 40800000
      [4] 40a00000 40c00000 40e00000 41000000

``debug_dump_cb_typed(cb_id, tile_idx)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Prints tile data interpreted according to the CB's data format, showing actual float/int values.
Uses TileSlice internally. DPRINT only (not supported with DEVICE_PRINT).

Available on TRISC0 (Unpack), TRISC2 (Pack), BRISC, and NCRISC. On BRISC/NCRISC, an additional
``cb_type`` parameter specifies whether the CB is an input or output.

.. code-block:: c++

    // On TRISC0 (Unpack) or TRISC2 (Pack):
    debug_dump_cb_typed(0, 0);              // CB0, tile 0, untilized

    // On BRISC or NCRISC (need to specify input vs output):
    debug_dump_cb_typed(0, 0, TSLICE_INPUT_CB);   // CB0 as input CB
    debug_dump_cb_typed(16, 0, TSLICE_OUTPUT_CB);  // CB16 as output CB

``debug_dump_l1(addr, num_words)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Hex-dumps arbitrary L1 memory. Available from all RISCs. Useful for inspecting semaphores,
scratch space, or any L1 region.

.. code-block:: c++

    debug_dump_l1(0x100000, 16);   // 16 words starting at L1 address 0x100000

Output:

.. code-block:: text

    L1[0x100000] 16 words:
      [0x100000] 3f800000 40000000 40400000 40800000
      [0x100010] 40a00000 40c00000 40e00000 41000000
      [0x100020] 41100000 41200000 41300000 41400000
      [0x100030] 41500000 41600000 41700000 41800000

All three functions are no-ops when DPRINT/DEVICE_PRINT is not enabled.

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
pipeline at a specific point within a large op.

Files
-----

.. list-table::
   :header-rows: 1

   * - File
     - Purpose
   * - ``tt_metal/hw/inc/api/debug/checkpoint.h``
     - Checkpoint API: single-core barrier, cross-core barrier, CB dump, dest reg dump
   * - ``tt_metal/hw/inc/api/debug/dump.h``
     - Standalone dump utilities: ``debug_dump_cb``, ``debug_dump_cb_typed``, ``debug_dump_l1``
   * - ``tt_metal/jit_build/build.cpp``
     - ``TT_METAL_CHECKPOINT`` env var enables ``-DDEBUG_CHECKPOINT_ENABLED``
   * - ``tt_metal/hw/firmware/src/tt-1xx/brisc.cc``
     - Calls ``debug_checkpoint_init()`` before kernel launch
