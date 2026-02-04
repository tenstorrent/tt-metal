
Multicast for Improved Data Reuse in Multi Core Matrix Multiplication
=====================================================================

Introduction
------------

In Lab 2 you implemented **multi core matrix multiplication with data reuse within each core**. Each core read tiles of input matrices from DRAM into its own circular buffers (CBs) and reused them locally across multiple multiply-accumulate steps. However, **data was not reused across cores**: each core independently read its own tiles from DRAM, even when neighbor cores needed the exact same data.

Ideally, each piece of data should be fetched from DRAM only once and then reused by all cores that need it. On Tenstorrent devices, cores do not have direct access to each other's L1 CBs, but they are connected by a 2D on-chip **Network-on-Chip (NOC)**. The NOC supports **multicast**, which allows a sender core to write the same data to multiple destination cores in a single NOC operation. In this lab you will:

* Use simple multicast to send tiles from one "coordinator" core to multiple receiver cores.
* Understand how semaphores, device coordinates, and multicast addressing work together.
* Apply multicast to your Lab 2 multi core matmul so that tiles of A and B are reused **across cores**, not just within a single core.

High level motivation
---------------------

In Lab 2, you already reduced DRAM bandwidth by reusing tiles locally. However, consider a row of cores that all work on different columns of the same row block of matrix A. Each core separately loaded the same A tiles from DRAM into its L1. This is wasteful: DRAM bandwidth and energy are spent multiple times on identical data.

The natural next step is to **load a tile from DRAM once and share it across cores**. For example, the leftmost core in each row could read a tile of A and then forward it to all other cores in that row that also need it. Similarly, in each column, the topmost core could read a tile of B and forward it to all cores below.

Because cores cannot peek into each other's CBs, this sharing must happen explicitly through the NOC. The idea is:

* For any given tile, **one core is responsible for reading it from DRAM**.
* That core stores the tile in its own CBs (for its own computation), and also **multicasts it to other cores** that need the same tile.
* All receiving cores place the tile into their own input CBs and then proceed with computation.

In the context of multi core matmul:

* Tiles of A can be read once per row of cores and multicasted **down the column**.
* Tiles of B can be read once per column of cores and multicasted **across the row**.
* Every core receives the A and B tiles it needs from its row/column multicast senders, while also performing its share of compute.

In the rest of this lab, you will first work through a **standalone multicast example**, then retrofit the same ideas into your Lab 2 matrix multiplication solution.

Lab objectives
--------------

By the end of this lab you should be able to:

* Describe, at a high level, how the Tenstorrent NOC and multicast work.
* Use **device coordinates** (NOC coordinates) together with `worker_core_from_logical_core` to configure multicast.
* Use semaphores and NOC APIs to synchronize a sender core with multiple receivers.
* Explain the interaction between multicast and double buffering.
* Modify your multi core matmul with intra core reuse so that tiles of A and B are **multicasted across cores**.

Prerequisites
-------------

This lab assumes that you have:

* Completed Lab 1 and are comfortable with:
  * Tiled matmul on a single core.
  * Using CBs, dataflow kernels, and compute kernels.
* Completed Lab 2 and are comfortable with:
  * Multi core matmul with data reuse inside each core.
  * Setting up core ranges, splitting work across cores, and using CBs for double buffering.

Background: Tenstorrent NOC and Multicast
-----------------------------------------

NOC overview
~~~~~~~~~~~~

Tenstorrent devices organize Tensix cores in a **2D grid** connected by a Network-on-Chip. Each core has **NOC coordinates** of the form `(x, y)` indicating its position in this grid. The NOC allows:

* Point-to-point (unicast) data transfers, e.g., from one core to exactly one other core.
* **Multicast** transfers, where a single source sends the same payload to multiple destination cores in a single NOC operation.

Conceptually, you can think of the NOC as a grid of cores, each with an L1 memory:

.. code-block:: text

   y
   ^
   |   (0,2)   (1,2)   (2,2)   (3,2)
   |   +-----+ +-----+ +-----+ +-----+
   |   |     | |     | |     | |     |
   |   +-----+ +-----+ +-----+ +-----+
   |   (0,1)   (1,1)   (2,1)   (3,1)
   |   +-----+ +-----+ +-----+ +-----+
   |   |     | |     | |     | |     |
   |   +-----+ +-----+ +-----+ +-----+
   |   (0,0)   (1,0)   (2,0)   (3,0)  -> x
   |   +-----+ +-----+ +-----+ +-----+
   |   |     | |     | |     | |     |
   +   +-----+ +-----+ +-----+ +-----+

Multicast uses this structure to **replicate a stream of tiles from one core to many**. For example, a sender at `(0,0)` could multicast a tile to receivers `(1,0)`, `(2,0)`, and `(3,0)` in one operation.

Logical vs device coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In TT-Metalium:

* **Logical coordinates** are used on the host side to describe how you want to assign work to cores in a program. For example, you might define:

  .. code-block:: c++

     CoreRange all_cores_logical({0, 0}, {3, 0});

  even if the physical layout of cores on the device is more complex.

* **Device coordinates** (sometimes called NOC or worker coordinates) are the actual coordinates used by the hardware NOC. These are the coordinates that device kernels need when they call NOC APIs such as multicast.

The host code always uses logical coordinates when creating kernels and CBs, but kernels that perform NOC operations must know device coordinates. To map between the two, you use:

.. code-block:: c++

   CoreCoord sender_core_device =
       mesh_device->worker_core_from_logical_core(sender_core_logical);

This conversion lets you write host code in a device independent way, while still supplying correct NOC addresses to the kernels at runtime.

Multicast and double buffering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this lab, multicast is combined with **double buffering** in the CBs:

* On the sender, double buffering allows overlapping:
  * Reading the next tile from DRAM into L1.
  * Multicasting the previous tile to receivers.
* On each receiver, double buffering allows overlapping:
  * Receiving a tile via multicast into its input CB.
  * Computing on previously received tiles.

Double buffering still works with multicast, as long as:

* You do not reuse a CB slot until all NOC operations that write to that memory have completed.
* You maintain a consistent pattern of `cb_reserve_back`, `cb_push_back`, `cb_wait_front`, and `cb_pop_front` across sender and receivers, so that everyone agrees which L1 addresses are used for which tile at each step.

You will see this pattern in the standalone multicast example first, then reuse it in the matmul case.

Standalone multicast example overview
-------------------------------------

Overview of provided files
~~~~~~~~~~~~~~~~~~~~~~~~~~

The lab includes a standalone multicast example with:

* Host program:
  * ``lab_multicast.cpp``
* Dataflow kernels:
  * ``kernels/dataflow/mcast_sender.cpp``
  * ``kernels/dataflow/mcast_receiver.cpp``
  * ``kernels/dataflow/write_tiles.cpp``
* Compute kernel:
  * ``kernels/compute/tiles_copy.cpp``

High level behavior:

* The host creates a 2D tensor (in DRAM) and fills it with random data.
* One **sender core** reads tiles of this tensor from DRAM and multicasts them to several **receiver cores**.
* Each receiver core:
  * Receives tiles into its input CB.
  * Passes tiles through a simple compute pipeline (copy).
  * Writes its own copy of the full tensor back to DRAM.
* The host reads back all receiver outputs and verifies that they match the original tensor.

Core roles in the basic multicast example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The multicast example uses a single row of logical cores:

.. code-block:: text

   Logical coordinates:

     (0,0)   (1,0)   (2,0)   (3,0)
     +---+   +---+   +---+   +---+
     | S |   | R |   | R |   | R |
     +---+   +---+   +---+   +---+

* Core ``(0,0)`` is the **sender core**:
  * Reads tiles from DRAM into its own CB.
  * Multicasts these tiles to cores ``(1,0)``, ``(2,0)``, and ``(3,0)``.
* Cores ``(1,0)``, ``(2,0)``, and ``(3,0)`` are **receiver cores**:
  * Do not read the input tensor from DRAM.
  * Receive tiles via multicast from the sender.
  * Run compute and writeback kernels.
* Any other cores on the device do **nothing** in this example. This is fine: you only need a subset of cores for this demonstration, and leaving the rest idle simplifies reasoning about NOC behavior.

Note that in this example the sender core only multicasts data; it does not currently run the same compute pipeline as receivers. You will change that in Exercise 1.

High level pseudocode for multicast pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before looking at code, it is helpful to describe the multicast protocol at a high level.

Sender core (per tile):

.. code-block:: text

   for each tile index t:
       1. Reserve an input CB slot.
       2. Read tile t from DRAM into that slot.
       3. Mark the tile as available in the CB (push_back).
       4. Wait until all receivers signal that they are ready for tile t.
       5. Multicast the tile from the CB slot to all receiver cores.
       6. Flush NOC writes to ensure the multicast command has been sent.
       7. Multicast a semaphore update to tell receivers that tile t has been sent.
       8. Wait for the multicast to complete before freeing the CB slot.
       9. Pop the tile from the CB (free the slot for a future tile).

Receiver cores (per tile):

.. code-block:: text

   for each tile index t:
       1. Reserve an input CB slot for the incoming tile.
       2. Reset local "tile_sent" semaphore to INVALID.
       3. Increment the sender's "receivers_ready" semaphore to say:
          "this receiver is ready for the next tile".
       4. Wait for the sender to multicast "tile_sent" semaphore to VALID.
       5. At this point, the tile has arrived in the CB slot.
       6. Push the tile in the CB so the compute kernel can consume it.

Compute kernel (per tile):

.. code-block:: text

   for each tile index t:
       1. Wait for a tile to appear at the front of the input CB.
       2. Load the tile into a compute register.
       3. Perform the desired computation (copy in this example).
       4. Reserve an output CB slot, write result tile, and push it.

Writer kernel (per tile):

.. code-block:: text

   for each tile index t:
       1. Wait for a tile at the front of the output CB.
       2. Write the tile to DRAM at this receiver's region of the output tensor.
       3. Pop the CB slot.

This protocol uses semaphores to coordinate when each tile is safe to multicast and when it is safe to consume, while CB operations manage local double buffering.

New concepts and APIs in the multicast example
----------------------------------------------

This section focuses on **new constructs** that were not needed in Labs 1 and 2. You should refer to the example code as you read.

Device coordinates and `worker_core_from_logical_core`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Host code uses logical coordinates to create kernels and CBs:

.. code-block:: c++

   CoreRange all_cores_logical({0, 0}, {3, 0});
   CoreCoord sender_core_logical = {0, 0};
   CoreRange receiver_cores_logical({1, 0}, {3, 0});

To perform NOC operations in device kernels, you need **device coordinates**. The host converts logical coordinates to device coordinates:

.. code-block:: c++

   CoreCoord sender_core_device =
       mesh_device->worker_core_from_logical_core(sender_core_logical);

   CoreRange receiver_cores_device(
       mesh_device->worker_core_from_logical_core(receiver_cores_logical.start_coord),
       mesh_device->worker_core_from_logical_core(receiver_cores_logical.end_coord));

The sender's runtime arguments then include the device coordinates of the receiver range:

.. code-block:: c++

   SetRuntimeArgs(
       program,
       mcast_sender_id,
       sender_core_logical,
       {
           static_cast<uint32_t>(receiver_cores_device.start_coord.x),
           static_cast<uint32_t>(receiver_cores_device.start_coord.y),
           static_cast<uint32_t>(receiver_cores_device.end_coord.x),
           static_cast<uint32_t>(receiver_cores_device.end_coord.y),
           receivers_ready_semaphore,
           tile_sent_semaphore,
           src_mesh_buffer->address(),
           n_tiles,
           num_dests
       });

Key points:

* On the host:
  * You always pass **logical coordinates** to `CreateKernel`, `CreateCircularBuffer`, and `SetRuntimeArgs`.
* In device kernels:
  * You must use **device coordinates** when calling NOC APIs that address other cores (e.g., multicast).
  * Device coordinates are passed into kernels through runtime arguments that the host constructs using `worker_core_from_logical_core`.

`tt_l1_ptr` macro
~~~~~~~~~~~~~~~~~

In the sender and receiver kernels, semaphore pointers are declared using the `tt_l1_ptr` macro:

.. code-block:: c++

   volatile tt_l1_ptr uint32_t* sender_sem_ptr =
       reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receivers_ready_semaphore_addr);

`tt_l1_ptr` expands to a compiler attribute indicating that the pointer refers to **L1 memory**. This helps the compiler:

* Optimize address calculations.
* Avoid unnecessary loads/stores.
* Potentially use specialized addressing modes.

You should use `tt_l1_ptr` for pointers to L1 memory that are accessed from kernels. The macro does not change program semantics, but it enables better compiler optimizations.

NOC semaphores and `noc_semaphore_wait`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Semaphores are used for coordination between cores. The sender and receivers share two semaphores:

* `receivers_ready_semaphore`:
  * Stored in the sender's L1 (but accessible over NOC).
  * Receivers increment it to indicate they are ready for the next tile.
* `tile_sent_semaphore`:
  * Also stored in the sender's L1.
  * The sender multicasts its value to receivers to indicate that a tile has been sent.

In the sender kernel:

.. code-block:: c++

   volatile tt_l1_ptr uint32_t* sender_sem_ptr =
       reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receivers_ready_semaphore_addr);

   // Wait for all receivers to signal they are ready for the next tile.
   noc_semaphore_wait(sender_sem_ptr, num_receivers);
   noc_semaphore_set(sender_sem_ptr, 0);

`noc_semaphore_wait(sender_sem_ptr, num)` blocks until the value stored at `sender_sem_ptr` becomes equal to `num`. In this example:

* Each receiver calls `noc_semaphore_inc` on the sender's "receivers_ready" semaphore to increment it by 1.
* When all `num_receivers` receivers have incremented it, the sender proceeds, then resets the semaphore to 0 with `noc_semaphore_set`.

On the receiver side, for the "tile sent" semaphore:

.. code-block:: c++

   volatile tt_l1_ptr uint32_t* tile_sent_sem_ptr =
       reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_sent_semaphore_addr);

   // Reset tile_sent semaphore to INVALID before signaling ready
   noc_semaphore_set(tile_sent_sem_ptr, INVALID);

   // ...

   // Wait for sender to multicast the tile (semaphore becomes VALID)
   noc_semaphore_wait(tile_sent_sem_ptr, VALID);

Here `noc_semaphore_wait(tile_sent_sem_ptr, VALID)` waits until the sender multicasts a "VALID" update to this semaphore.

`noc_async_write_multicast`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The sender multicasts tiles using:

.. code-block:: c++

   uint64_t tile_mcast_addr =
       get_noc_multicast_addr(receiver_start_x, receiver_start_y,
                              receiver_end_x, receiver_end_y,
                              l1_read_addr);

   noc_async_write_multicast(l1_read_addr, tile_mcast_addr, tile_size_bytes, num_receivers);

Key points:

* `noc_async_write_multicast` is **non blocking**:
  * It enqueues a multicast transfer on the NOC, then **returns control to the kernel immediately**.
  * The hardware performs the tile transfer in the background.
* Multicast is far more efficient than issuing `num_receivers` separate `noc_async_write` calls:
  * One command instead of many.
  * Less contention and overhead on the NOC command FIFOs.
  * Simpler synchronization, because all receivers observe the same sequence of operations.

TODO: Explain Why ``noc_async_write_multicast`` needs ``num_dests`` parameter?
Why isn't that automatically calculated from ``dst_noc_addr_multicast``, which encodes the range of destinations?

TODO: document limitations of ``noc_async_write_multicast``:

In short:

They **cannot** be an arbitrary set of cores.

For `noc_async_write_multicast` the documented and implemented constraints are:

1. **Rectangular grid only (base API)**
   The targets must be all Tensix worker cores in a **contiguous rectangle** `[x_start..x_end] × [y_start..y_end]` on the NoC, all at the same L1 address.
   The docs say explicitly:

   - “The destination nodes can only be a set of Tensix cores + L1 memory address.”
   - “The destination nodes must form a **rectangular grid**.”

   So: not “same row” or “same column” only; but **any axis‑aligned rectangle** is fine.

2. **L‑shaped variant = rectangle minus rectangle**
   There is a separate API (multicast with exclude region) where you specify a rectangle and then subtract a rectangular “exclusion zone” to get an **L‑shaped** pattern, but even that is *“rect grid minus sub‑rect grid”*, not an arbitrary subset.

3. **Same L1 address on all destinations**
   All destination cores must use the **same L1 address**; the multicast NOC address encodes one local address plus the rectangular coord range, not per‑core offsets.

4. **Sender cannot be in the destination set (for this API)**
   The base `noc_async_write_multicast` excludes the sender core; if you want the sender included you must use the `*_loopback_src` variant.

5. **Cardinality is otherwise unconstrained**
   Aside from “non‑zero” and “<= number of cores − 1”, the number of destinations can be as large as “full chip rectangle.”

So if you need to hit an **arbitrary set of scattered cores** (e.g. “this triangle” or a few disjoint islands), you have to implement that as:

- multiple multicast calls to different rectangles / L‑shapes, or
- fall back to multiple unicast writes.

A single `noc_async_write_multicast` call always targets a **rectangular (or L‑shaped via the exclude API) contiguous region**, not an arbitrary mask of cores.



Ordering with `noc_async_writes_flushed`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because `noc_async_write_multicast` is non blocking, you must ensure that:

* The multicast command is actually **sent onto the NOC** before you signal receivers that the tile is ready.

This is handled using:

.. code-block:: c++

   noc_async_writes_flushed();

   // Signal receivers that tile has been sent by multicasting VALID to receiver semaphore
   *receiver_sem_ptr = VALID;
   noc_semaphore_set_multicast(tile_sent_semaphore_addr, receiver_sem_mcast_addr, num_receivers);

`noc_async_writes_flushed()` does **not** wait for multicast transfers to complete. It only ensures that all outstanding enqueued `noc_async_write` calls issued on the current core have **departed** (have been pushed into the NOC) before the function returns.

Why is it safe to send the semaphore update before the transfers complete?

* The sender:
  * Issues the multicast command.
  * Calls `noc_async_writes_flushed()` to ensure the command has been sent into the NOC.
  * Then multicasts the "tile_sent" semaphore value.
* Each receiver:
  * Waits for the "tile_sent" semaphore to become `VALID` before using the tile.
  * The NOC ensures that the data transfer and the semaphore update are observed in the correct order.
* Additionally, the sender calls:

  .. code-block:: c++

     noc_async_write_barrier();

  before popping the CB slot, ensuring that the tile transfer has completed before the sender reuses the CB memory location.

TODO: Explain that this is a simplified view of the reality and that for optimal performance on some architectures we may do things differently.
Also, NOC is customizable and may behave differently if customizations are made.


`get_noc_multicast_addr` and CB address synchronization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To multicast data, the sender needs to know **where** in each receiver's L1 to write the tile. This is the job of `get_noc_multicast_addr`:

.. code-block:: c++

   uint64_t tile_mcast_addr =
       get_noc_multicast_addr(receiver_start_x, receiver_start_y,
                              receiver_end_x, receiver_end_y,
                              l1_read_addr);

The last argument passed to `get_noc_multicast_addr` is a **memory address at the destination**. In this example:

* Sender and receivers use the same CB index (e.g., input CB).
* All cores reserve CB slots and push/pop tiles in the **same pattern**.
* For each tile index:
  * The write pointer in the sender CB points to the **same offset** within the corresponding CB as the write pointer in the receivers.
  * Therefore, using the sender's local CB address as the "destination address" in `get_noc_multicast_addr` is correct.

You can think of the return value of `get_noc_multicast_addr` as a packed 64 bit encoding of:

* The destination core rectangle: `(x_start, y_start, x_end, y_end)`.
* The destination L1 address.

In general, there is no direct way for receivers to "tell" the sender which exact L1 address they are using for a CB slot. Instead, the design ensures that **all cores run the same CB protocol**:

* Each core executes the same sequence of:
  * `cb_reserve_back`
  * `get_write_ptr`
  * `cb_push_back`
  * `cb_wait_front`
  * `get_read_ptr`
  * `cb_pop_front`
* Because CB sizes and page sizes are identical, the CB write pointer for a given tile index is the same on the sender and all receivers.
* This makes it possible for the sender to use its own CB write pointer as the destination address in `get_noc_multicast_addr`.

TODO: Example code still has variables like sender_sem_ptr, receiver_sem_ptr, and receiver_sem_mcast_addr, rather than referenceing receivers_ready and tile_sent semaphore names.


Exercise 1: Extending the standalone multicast example
------------------------------------------------------

In the provided example, the sender core only forwards data, while receivers perform the copy and writeback. In this exercise you will:

1. Build and run the example to verify it works as is.
2. Extend it so that **the sender core also participates in the same computation**, using the same multicast data.
3. Update the host side verification to include the sender's results.

Step 1: Build and run the base example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Build the example host program and kernels, following the same pattern as in Labs 1 and 2.
2. Run the executable.
3. Confirm that:
   * The program completes without errors.
   * All receiver cores pass the verification step.
   * Log messages indicate successful multicast and correct number of tiles.

You should see output similar to:

.. code-block:: text

   [PASS] Receiver 1 received correct tensor (400 tiles)
   [PASS] Receiver 2 received correct tensor (400 tiles)
   [PASS] Receiver 3 received correct tensor (400 tiles)
   [PASS] All 3 receivers received correct tensor data
   Test Passed

Step 2: Have the sender core also compute on the data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Currently the sender core:

* Reads tiles from DRAM into input CB `c_in0`.
* Multicasts tiles to receivers.
* Does not pass these tiles to the compute kernel on the sender core.

Your goal is to modify the program so that:

* The sender core also runs the **same compute and writer kernels** as receivers (copying tiles and writing them to its own region in the output tensor).
* After the change, output of the program will be a tensor that contains four copies of the input tensor; one from the sender core and three from the receiver cores.

The simplest way to achieve this is for the sender core to run the same compute kernel as the receivers.
This means that it will use the same CB that all other cores use for input tiles, which is `c_0`.
Observe that this is the same CB that is used by the sender as a source of data for multicast.
The `mcast_sender` kernel code can stay largely the same, with one key difference: it should not
perform any `cb_wait_front` or `cb_pop_front` operations, because the compute kernel will be doing this work.


 already configured to multicast tiles from `c_0` to the receiver cores.

TODO: A figure would come handy here!!!

The main required change to achieve the objective of this exercise is to update the host program to include the sender
core in the core range when creating the compute and writer kernels and pass the appropriate runtime arguments.
The writer kernel itslef does not need to change at all.
Circular buffers are already created on all cores, so there is no change required to the circular buffer setup.
The existing semaphore protocol already supports sending tiles from sender to receivers.
The sender acting as a "local receiver" for compute does **not** require introduction of additional semaphores.
This is because the sender already knows when a tile is in its local CB (right after the DRAM read completes).
     * The compute kernel on the sender simply waits for tiles in `c_in0` in the same way as receivers.




Follow these steps to complete the exercise:

#. Update the host program to:
   * Include the sender core in the core range when creating the compute and writer kernels.
   * Include the sender core in the core range when setting up the runtime arguments for the writer kernel.
   * Update `output_data` and related variables to account for the additional copy from the sender core.
     Make sure that each core writes to a unique region of the output tensor.
   * If you created a new folder for this exercise, make sure to update `CreateKernel` with paths to kernels in the new folder.

#. Update the `mcast_sender` kernel code to not perform any `cb_wait_front` or `cb_pop_front` operations.
   Since the code won't perform `cb_wait_front`, this also means that it cannot call `get_read_ptr`.
   Instead, the source address for multicast should be the same address that was used for writing the tile to the CB.

#. Ensure that the sender core's CB usage matches receivers:
   * The sender already uses `c_in0` to hold input tiles.
   * For the compute and writer kernels, ensure they are created to run on:
     * All cores that should produce output (including the sender).
   * Make sure `c_out0` (or the CB used for output tiles) exists and has the same size on the sender as on receivers.

High level pseudocode for host side changes:

.. code-block:: text

   define logical cores:
       sender_logical = (0, 0)
       receiver_cores_logical = (1,0) .. (3,0)
       all_compute_cores = (0,0) .. (3,0)   # now includes sender

   create CBs (input/output) on all_compute_cores

   create mcast_sender kernel on sender_logical
   create mcast_receiver kernel on receiver_cores_logical

   create tiles_copy compute kernel on all_compute_cores
   create write_tiles kernel on all_compute_cores

   for each core in all_compute_cores:
       assign a unique receiver_idx (0..3) for write_tiles runtime args

#. Adjust semaphores:
   * The existing semaphore protocol already supports sending tiles from sender to receivers.
   * The sender acting as a "local receiver" for compute does **not** need additional semaphores:
     * The sender already knows when a tile is in its local CB (right after the DRAM read completes).
     * The compute kernel on the sender simply waits for tiles in `c_in0` in the same way as receivers.
   * Ensure CB sizes are sufficient for:
     * Overlapping DRAM reads and NOC transfers.
     * Overlapping NOC transfers and compute on all participating cores.

Step 3: Update verification to include the sender's output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The host currently verifies only the receiver outputs. Modify verification so that:

1. The sender's output data is included in the DRAM tensor that you read back.
2. The reference comparison checks **all 4 copies**:

   * If you use a layout where each core writes `n_tiles` tiles to a consecutive region:
     * Core index `0` writes tiles `0..n_tiles-1`.
     * Core index `1` writes tiles `n_tiles..2*n_tiles-1`.
     * And so on.

3. The verification logic should report whether:

   * The sender's copy matches the reference.
   * Each receiver's copy matches the reference.

At the end of Exercise 1, you should have:

* A working multicast example where:
  * One core reads from DRAM.
  * All four cores (including the sender) receive the same tiles and perform identical computation.
* A clear understanding of how:
  * NOC multicast distributes data across cores.
  * Semaphores synchronize sender and receivers.
  * CBs and double buffering fit into the pipeline.

Applying multicast to multi core matmul
---------------------------------------

In Lab 2 you implemented multi core matmul where:

* Each core:
  * Loaded tiles of A and B from DRAM into its own CBs.
  * Reused its local tiles across multiple compute steps.
* Cores did **not** share tiles across the NOC.

In this section you will conceptually extend that design using multicast.

Motivation for multicast in matmul
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Recall the tiling pattern from Lab 2:

* The output matrix C is divided into tiles.
* Each core computes a block of tiles in C.
* For a given core:
  * It needs a sequence of tiles of A (corresponding to its rows).
  * It needs a sequence of tiles of B (corresponding to its columns).

Now consider a grid of cores arranged as:

.. code-block:: text

   Top row:
       (0,0)   (1,0)   (2,0)   ...   (Nc-1,0)

   Other rows:
       (0,1)   (1,1)   (2,1)   ...   (Nc-1,1)
       (0,2)   (1,2)   (2,2)   ...   (Nc-1,2)
       ...
       (0,Nr-1) ...                  (Nc-1,Nr-1)

For each core `(i,j)`:

* It needs A tiles for **row i**.
* It needs B tiles for **column j**.

Without multicast:

* Each core reads its own A and B tiles from DRAM, even when:
  * All cores in the same row need the same A tiles.
  * All cores in the same column need the same B tiles.

With multicast:

* The **leftmost column** cores (j = 0) load A tiles from DRAM.
* The **topmost row** cores (i = 0) load B tiles from DRAM.
* These cores then **multicast**:

  * A tiles down their column.
  * B tiles across their row.

Roles of different cores
~~~~~~~~~~~~~~~~~~~~~~~~

We will use four roles:

1. **Top-left core `(0,0)`**:
   * Special core that:
     * Reads both A tiles and B tiles from DRAM.
     * Multicasts A tiles down the first column.
     * Multicasts B tiles across the first row.
     * Also performs its share of matmul computation.

2. **Top row cores (i = 0, j > 0)**:
   * Read only **B tiles** from DRAM.
   * Multicast B tiles across their row (to cores below them).
   * Perform matmul computation for their subset of C tiles.

3. **Left column cores (i > 0, j = 0)**:
   * Read only **A tiles** from DRAM.
   * Multicast A tiles down their column (to cores to the right).
   * Perform matmul computation for their subset of C tiles.

4. **Interior cores (i > 0, j > 0)**:
   * Do **not** read A or B directly from DRAM.
   * Receive both A and B tiles via multicast:
     * A from the left column core in their row.
     * B from the top row core in their column.
   * Perform matmul computation for their subset of C tiles.

This pattern ensures that:

* Each distinct tile of A is read from DRAM exactly once per **core row**.
* Each distinct tile of B is read from DRAM exactly once per **core column**.
* All cores use the multicasted data in their local matmul loops.

ASCII diagram of roles
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Roles:

   T = Top-left core (A- and B-source)
   R = Top row B-source cores
   C = Left column A-source cores
   I = Interior compute-only cores

           j = 0    1      2      3
         +------+------+------+------+
   i = 0 |  T   |  R   |  R   |  R   |
         +------+------+------+------+
       1 |  C   |  I   |  I   |  I   |
         +------+------+------+------+
       2 |  C   |  I   |  I   |  I   |
         +------+------+------+------+
       3 |  C   |  I   |  I   |  I   |
         +------+------+------+------+

For each tile index in the K dimension:

* Core `(0,0)`:
  * Reads one tile of A and one tile of B from DRAM.
  * Multicasts A down the first column.
  * Multicasts B across the first row.
* Each top row core `(j > 0)`:
  * Reads its own B tile from DRAM.
  * Multicasts B down its column.
* Each left column core `(i > 0)`:
  * Reads its own A tile from DRAM.
  * Multicasts A across its row.
* Interior cores:
  * Receive the appropriate A tile from the left.
  * Receive the appropriate B tile from above.

In your implementation, you should choose a consistent convention:

* For example:
  * A tiles are multicast **horizontally** (from left to right).
  * B tiles are multicast **vertically** (from top to bottom).

The important property is that **all cores in a row** agree on which core reads and multicasts A tiles, and **all cores in a column** agree on which core reads and multicasts B tiles.

Granularity of DRAM reads and multicast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Just as in Labs 1 and 2:

* The natural granularity of data movement is **one tile** at a time.
* A tile is the smallest unit that the compute kernel operates on.

For matmul with multicast:

* DRAM reads:
  * Typically one tile of A or B at a time per source core.
  * You can extend this later to load multiple tiles at once for better throughput, but that is not required in this lab.
* Multicast:
  * One tile of A or B at a time.
  * Each tile is multicast to all cores that need it in the corresponding row or column.

Keeping DRAM and multicast granularity at one tile allows you to reuse the same double buffering patterns you saw in the standalone multicast example and in Lab 2.

Interaction between multicast and double buffering in matmul
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same rules that applied in the standalone example apply in matmul:

* Each core uses CBs (e.g., `c_in0` for A and `c_in1` for B) with at least two tiles per CB to enable double buffering.
* For A-source and B-source cores:
  * DRAM reads fill CBs.
  * Multicast operations forward CB tiles to other cores.
* For receiving cores:
  * NOC writes fill CBs.
  * Compute kernels consume tiles and write results to output CBs.

To safely combine multicast and double buffering:

1. Ensure all cores that share A (or B) tiles:
   * Use the same CB configuration for that operand:
     * Same CB index.
     * Same page size (tile size).
     * Same number of tiles in the CB.
2. Ensure all cores perform CB operations in the **same order**:
   * For each tile:
     * `cb_reserve_back` before writing or receiving.
     * `cb_push_back` once the tile is ready.
     * `cb_wait_front` before reading.
     * `cb_pop_front` after the tile is consumed.
3. On multicast sender cores:
   * Use `noc_async_write_multicast` to forward A or B tiles.
   * Call `noc_async_writes_flushed()` before updating semaphores that tell receivers the tile is ready.
   * Use NOC barriers (e.g., `noc_async_write_barrier()`) before `cb_pop_front` to avoid reusing CB slots while transfers are still in progress.
4. On receivers:
   * Use semaphores (`noc_semaphore_wait`) to ensure that tiles are fully received before making them visible to compute kernels with `cb_push_back`.

Exercise 2 will guide you through applying these ideas to your Lab 2 matmul solution.

Exercise 2: Multi core matmul with multicast
--------------------------------------------

In this exercise you will extend your Lab 2 multi core matmul implementation to:

* Use multicast to share A and B tiles across cores.
* Maintain double buffering for performance.
* Preserve correctness of the computed C matrix.

You will work from your **Exercise 2 solution from Lab 2** ("Multi Core Matrix Multiplication with Data Reuse").

Step 0: Review your Lab 2 solution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before you start changing code, identify in your Lab 2 program:

1. How work is split across cores:
   * Which core range (logical coordinates) is used.
   * Which tiles of C each core computes.
2. Where A and B tiles are loaded from DRAM into CBs:
   * Which CB indices are used for A and B.
   * How double buffering is configured.
3. Where compute kernels consume A and B tiles to produce C tiles.
4. Where C tiles are written back to DRAM.

You will modify the **data movement for A and B tiles** while leaving the overall compute structure as unchanged as possible.

Step 1: Define core roles for matmul
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose a rectangular grid of logical cores for your matmul, for example:

.. code-block:: c++

   CoreRange core_grid_logical({0, 0}, {Nc - 1, Nr - 1});

Adopt the four roles described earlier:

* `(0,0)` is the combined A and B source.
* Top row `(i = 0, j > 0)` are B-source cores.
* Left column `(i > 0, j = 0)` are A-source cores.
* All other cores `(i > 0, j > 0)` are interior compute-only cores.

In code, you can classify core roles using their logical coordinates.

Step 2: Ensure CB layout is consistent across cores
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each operand:

* A tiles:
  * Choose one CB index (e.g., `c_in0`) for A across **all** cores that will hold A tiles.
  * Configure this CB with the same number of tiles and page size everywhere.
* B tiles:
  * Choose another CB index (e.g., `c_in1`) for B across all cores that will hold B tiles.
  * Configure this CB similarly.

This uniformity ensures that:

* All cores use the same sequence of CB operations for A and for B.
* `get_noc_multicast_addr` can use local CB addresses to generate correct destination addresses for multicast.

Step 3: Introduce semaphores for A and B multicast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each operand (A and B), you will need semaphores to coordinate source and receivers. One simple design is:

* For A:
  * `A_receivers_ready_sem`: each A receiver in a row increments this when ready for the next A tile.
  * `A_tile_sent_sem`: A source core multicasts this to receivers when the A tile is ready.
* For B:
  * `B_receivers_ready_sem`: each B receiver in a column increments this when ready for the next B tile.
  * `B_tile_sent_sem`: B source core multicasts this to receivers when the B tile is ready.

In the host program:

* Allocate these semaphores on all cores that participate in A or B multicast.
* Pass the appropriate semaphore indices to the data movement kernels for A and B.

The exact number and placement of semaphores can vary, but the pattern should match what you saw in the standalone multicast example:

* Receivers use `noc_semaphore_inc` to signal "ready".
* Senders use `noc_semaphore_wait` and `noc_semaphore_set` / `noc_semaphore_set_multicast` to synchronize.

Step 4: Implement A multicast data movement kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create data movement kernels for A:

1. A source kernel (runs on left column cores):
   * Reads A tiles from DRAM into `c_in0`.
   * Multicasts them along its row to cores that need those tiles.
   * Uses device coordinates derived from `worker_core_from_logical_core` to build multicast addresses.
   * Uses semaphores and `noc_async_writes_flushed` as in the standalone example to ensure correct ordering.

2. A receiver kernel (runs on cores that consume A but do not read it from DRAM):
   * Reserves CB slots in `c_in0` for incoming A tiles.
   * Signals readiness to the A source using a NOC semaphore increment.
   * Waits on a "tile sent" semaphore to become `VALID`.
   * Pushes tiles into `c_in0` for compute kernels.

High level pseudocode for A source kernel (per tile and per row):

.. code-block:: text

   for each K tile index k needed for this row:
       cb_reserve_back(A_cb, 1)
       l1_addr = get_write_ptr(A_cb)
       read A tile(k) from DRAM into l1_addr
       noc_async_read_barrier()
       cb_push_back(A_cb, 1)

       wait until all A receivers in this row have incremented A_receivers_ready_sem
       reset A_receivers_ready_sem to 0

       cb_wait_front(A_cb, 1)
       l1_read = get_read_ptr(A_cb)
       compute multicast address for A using get_noc_multicast_addr
       noc_async_write_multicast(l1_read, mcast_addr, tile_size, num_receivers)

       noc_async_writes_flushed()  # ensure multicast commands are sent
       multicast "tile_sent = VALID" to receivers via A_tile_sent_sem

       noc_async_write_barrier()   # wait for multicast completion before reusing CB slot
       cb_pop_front(A_cb, 1)

High level pseudocode for A receiver kernel (per tile):

.. code-block:: text

   for each K tile index k:
       cb_reserve_back(A_cb, 1)
       set local A_tile_sent_sem to INVALID

       # signal ready to source
       noc_semaphore_inc(A_receivers_ready_sem_addr, 1)

       # wait for source to indicate tile has been sent
       noc_semaphore_wait(A_tile_sent_sem_ptr, VALID)

       # tile is now in the CB write pointer location
       cb_push_back(A_cb, 1)

Step 5: Implement B multicast data movement kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Repeat the same pattern for B tiles, but along columns:

1. B source kernel (runs on top row cores):
   * Reads B tiles from DRAM into `c_in1`.
   * Multicasts them down its column to cores that need those tiles.
   * Uses device coordinates and semaphores in the same way as the A source kernel.

2. B receiver kernel (runs on cores that consume B but do not read it from DRAM):
   * Reserves CB slots in `c_in1` for incoming B tiles.
   * Signals readiness to the B source.
   * Waits for "B tile sent" semaphore to become `VALID`.
   * Pushes tiles into `c_in1` for compute kernels.

You can reuse much of the logic from the A multicast implementation, changing only:

* The dimension along which you multicast (rows vs columns).
* The CB index (e.g., `c_in1` for B).
* The semaphores used for B.

Step 6: Integrate with compute and writeback kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Your existing compute kernel from Lab 2 likely expects:

* A tile of A in an input CB (e.g., `c_in0`).
* A tile of B in another input CB (e.g., `c_in1`).
* It multiplies them and accumulates into the C tile in registers or in a dedicated CB.

You should not need to change the compute kernel logic:

* It still performs the same `cb_wait_front` / `cb_pop_front` pattern to consume A and B tiles.
* The only difference is the **source of those tiles**:
  * Instead of coming from DRAM via local dataflow kernels, they now come via multicast from an A source or B source.

Similarly, the C writeback kernel can remain unchanged:

* It reads C tiles from its CB and writes them to DRAM.
* Its runtime arguments should still point to the correct region of the output tensor for each core.

Double check that:

* All cores that perform compute have both A and B input kernels running appropriately:
  * A-source cores have:
    * A source dataflow kernel.
    * B receiver dataflow kernel (unless also B-source).
  * B-source cores have:
    * B source dataflow kernel.
    * A receiver dataflow kernel (unless also A-source).
  * Interior cores have:
    * A receiver kernel.
    * B receiver kernel.
* The top-left core `(0,0)` may run both A source and B source kernels.

Step 7: Verification
~~~~~~~~~~~~~~~~~~~~

Verification should follow the same pattern as Lab 2:

1. On the host:
   * Compute a reference C matrix on the host using standard matmul (e.g., in C++ or Python).
2. After running the device program:
   * Read back the full C tensor from DRAM.
   * Compare every element (or tile) against the reference.
3. Report:
   * Whether the final C matches the reference.
   * Optionally, per core or per tile diagnostics if mismatches occur.

Because parallelism and data movement do not change the mathematical matmul, any mismatch indicates a bug in:

* Core role classification.
* Semaphore synchronization.
* Multicast addressing.
* CB management / double buffering.

Conclusion
----------

In this lab you:

* Learned how the Tenstorrent NOC can be used to **multicast tiles** from a single source core to multiple receivers.
* Studied the roles of:
  * Device coordinates and `worker_core_from_logical_core`.
  * NOC semaphores and `noc_semaphore_wait`.
  * `noc_async_write_multicast`, `noc_async_writes_flushed`, and barriers.
  * `get_noc_multicast_addr` for encoding destination cores and L1 addresses.
* Extended a simple multicast example so that the sender core also participates in computation.
* Applied multicast to your multi core matmul implementation from Lab 2 so that:
  * A tiles are reused across rows of cores.
  * B tiles are reused across columns of cores.
  * Double buffering coexists correctly with multicast.

From here, natural next steps include:

* Batching multiple tiles in each DRAM read and multicast to amortize semaphore overhead.
* Exploring alternative multicast patterns (e.g., 2D rectangles, hierarchical multicast).
* Measuring and comparing performance and DRAM bandwidth usage:
  * Without reuse.
  * With intra core reuse only.
  * With intra core reuse plus cross core reuse via multicast.

These topics build directly on the concepts you have implemented in this lab and are good candidates for further experiments or project work.
