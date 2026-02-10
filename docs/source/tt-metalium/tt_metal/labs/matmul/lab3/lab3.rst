Lab 3: Multicast for Improved Data Reuse in Multi Core Matrix Multiplication
############################################################################

Introduction
************

In Lab 2 you implemented multi core matrix multiplication with data reuse within each core.
Each core read tiles of input matrices from DRAM into its own circular buffers (CBs) and
reused them locally across multiple multiply-accumulate steps. However, data was not reused across cores:
each core independently read its own tiles from DRAM, even when neighboring cores needed the exact same data.

Ideally, each piece of data should be fetched from DRAM only once and then reused by all cores that need it.
On Tenstorrent devices, cores do not have direct access to each other's circular buffers (CBs), but they are
connected by a2D **Network-on-Chip (NOC)** allowing the cores to pass data to each other.
While sending data over the NOC is more efficient than reading data from DRAM multiple times,
it still introduces some overhead. Therefore, we would like to minimize this overhead.
The NOC supports **unicast** and **multicast** operations.
Unicast allows a sender core to write data to a single destination core.
Multicast allows a sender core to write the same data to multiple destination cores in a single NOC operation,
minimizing the overhead when the same data needs to be sent to multiple cores.

In this lab you will:

* Learn how to use simple multicast to send tiles from one sender core to multiple receiver cores.
* Understand how semaphores, device coordinates, and multicast addressing work together.
* Apply multicast to your Lab 2 multi core matrix multiplication so that tiles of A and B are reused across cores,
  not just within a single core.

High Level Motivation
=====================

Consider an example matrix multiplication shown in Figure 1.

.. figure:: images/data_reuse_no_multicast.png
   :alt: Example matrix multiplication on a 3x3 core grid
   :width: 900
   :align: center

   Figure 1: Example matrix multiplication on a 3x3 core grid

Each square in Figure 1 represents a tile, and dimensions of matrices are ``9x6`` tiles
for ``A`` and ``6x9`` tiles for ``B``, resulting in a ``9x9`` tile output matrix ``C``.
The squares in the middle of the figure represent the core grid, with each square labeled
with its core coordinates ``(x, y)``. The core coordinates are also shown over the output
matrix ``C`` to indicate the core that computes the corresponding rectangular block of tiles.

From the basic matrix multiplication algorithm, we know that computing an element of the output matrix ``C``
requires all elements of the corresponding row of ``A`` and the corresponding column of ``B``.
Same applies when computing tiles or rectangular blocks of tiles of ``C``.
This means that all cores in the same row need the same rows of tiles of ``A``,
and all cores in the same column need the same columns of tiles of ``B``.

Arrows in Figure 1 indicate reads of tiles from DRAM into the cores' on-chip SRAM,
illustrating the fact that all cores in the same row read the same rows of tiles of ``A``,
and all cores in the same column read the same columns of tiles of ``B``.

Since DRAM bandwidth is limited, this is inefficent because same data is read multiple times from DRAM.
Instead, we would like to load a tile from DRAM once and share it across all cores that need it through the NOC.
A possible way to achive this is shown in Figure 2.

.. figure:: images/data_reuse_with_multicast.png
   :alt: Example matrix multiplication on a 3x3 core grid with multicast
   :width: 900
   :align: center

   Figure 2: Example matrix multiplication on a 3x3 core grid with multicast

In the example in Figure 2, only the leftmost core in each grid row reads tiles of ``A`` from DRAM,
depicted by thin arrows in the figure.
Each leftmost core stores the tiles of ``A`` into its own CBs for its own computation, just as it did in Lab 2.
However, it now also multicasts the tiles of ``A`` it read from DRAM to all the other cores in the same row.
Similarly, only the topmost core in each grid column reads tiles of ``B`` from DRAM, storing them into
its own CBs for its own computation, and multicasts these tiles to all the other cores in the same column.
The multicast operations are depicted by thick arrows in the figure.

In the rest of this lab, you will first work through a simple example program demonstrating NOC and
multicast features, then retrofit your Lab 2 matrix multiplication solution to use multicast.


Background: Tenstorrent NOC and Multicast
*****************************************

The Network-on-Chip (NoC) is a 2D mesh interconnect that connects:
* All Tensix cores
* DRAM controllers
* PCIe interfaces
* Ethernet cores (for multi-device systems)

NOC is used to transfer data between different components of the device, including transferring
data between DRAM and on-chip SRAM. As you have seen in the preceding labs, TT-Metalium programmer
doesn't need to understand all the details of the underlying hardware to use the NOC.
In this lab, we will expand our use of the NOC to include multicast operations to transfer data between cores.
For more detailed information about the NOC, refer to the resources listed in the Additional Information
section at the end of this lab.

In TT-Metalium, NoC multicast is a data movement operation where one core writes directly into the
on-chip SRAM of multiple other cores with a single command. The sender core specifies a group of
destination cores and a destination memory address, and the NoC hardware delivers the data to that
address on every destination core.
Unlike a "pull" model where receivers issue read requests, multicast is a "push" model: the sender
pushes the data into the receivers' on-chip memories.

From the receiving core's point of view, a multicast operation writes tiles straight into its on-chip SRAM,
typically into a location it has already set aside in a circular buffer (CB). The receiver does not
need to perform any explicit read or copy for the data itself; it only needs to prepare space and
indicate that it is ready to accept a tile (for example, by reserving a CB slot). Once multicast
completes, the tile is simply present in the CB, ready to be consumed by the compute or writer
kernels just like any other locally produced data.

We will illustrate the multicast operation with a simple example program in the next section.

Example Multicast Program
=========================

The main program for the example multicast program is located in the file ``ttnn/examples/lab_multicast/lab_multicast.cpp``.

The program creates a 2D tensor and fills it with random data.
One **sender core** uses a reader kernel to read tiles of this tensor from DRAM and also multicasts them to three **receiver cores**.
The flow of data is shown in Figure 3.

.. figure:: images/data_flow_multicast.png
   :alt: Data flow in the multicast example program
   :width: 700
   :align: center

   Figure 3: Data flow in the multicast example program

Core ``(0,0)`` is the **sender core** and cores ``(1,0)``, ``(2,0)``, and ``(3,0)`` are **receiver cores**:
Receiver cores do not read the input tensor from DRAM, but receive tiles via multicast from the sender.
Each receiver core has three kernels:

* A reader kernel that manages CB and signals to the sender core when it is ready for the next tile.
* A compute kernel, which simply copies each tile to the output CB. In a real application, this is where computation would happen.
* A writer kernel that writes each tile int an appropriate region of the output tensor in DRAM.

The host reads back all receiver outputs and verifies that the output matches expectations,
which is a tensor that contains three copies of the original tensor, stacked vertically.
Note that number of tiles in Figure 3 is symbolic and doesn't accurately represent the number of tiles in the actual program.

Logical vs. Device Coordinates
==============================

The first new thing in the example program not previously seen in Labs 1 and 2 is the use of **device coordinates**.
So far, we have been using logical coordinates to describe how you want to assign work to cores in a program.
Logical coordinates make a simplifying assumption that the physical layout of Tensix cores on the device is the same as the logical layout.
However, a typical Tensix device also contains multiple DRAM controllers, multiple Ethernet cores and a PCIe interface.
Since NOC interconnects these components, it needs to know the actual coordinates of each component on the device
to send the data to the correct destination.

Tenstorrent architecture actually defines more than two different coordinate systems, but
for the purpose of TT-Metalium programming, we only need to consider logical and device coordinates.
Note that device coordinates are also referred to as *virtual coordinates* in the Tenstorrent architecture documentation.

The host code always uses logical coordinates (e.g. when creating kernels and CBs), and the compiler takes care of converting
them to device coordinates when needed, making the program easier to write and understand.
However, to maximize performance, we want to avoid performing such conversions in device kernels as much as possible.
Therefore, device kernels performing NOC operations must use device coordinates to generate the correct NOC addresses.
To facilitate this, TT-Metalium provides the ``worker_core_from_logical_core`` function that is called on the host to
convert logical coordinates to device coordinates before passing them to the device kernels as either compile-time or runtime arguments.
For example, to convert the logical coordinates of the sender core to device coordinates, you can use the following code:
to device coordinates, you can use the following code:

.. code-block:: cpp

   CoreCoord sender_core_device =
       mesh_device->worker_core_from_logical_core(sender_core_logical);

This conversion lets you write host code in a device independent way, while still supplying correct NOC addresses to the kernels,
which use device coordinates to generate the correct NOC addresses.

Synchronization with Semaphores
===============================

Given that multicast uses a "push" model where the sender writes data directly into receivers' on-chip SRAM,
it is important to coordinate the execution between the sender and receivers to avoid data corruption and
race conditions. This coordination is done using semaphores.
In general, a semaphore is a small shared variable used to coordinate execution between different pieces
of code that run concurrently. For example, a semaphore can be used to signal
when the receivers are ready to have data written to their memory (i.e., when it is safe to do so, because
the receivers have reserved a CB slot for the incoming tile). Another semaphore can be used to signal
that data has been sent (i.e., written to receivers' memory).

In TT-Metalium, a semaphore is an integer value stored in on-chip SRAM that multiple cores can read and update.
Typical use cases for semaphores include:

* A core **increments or sets** a semaphore to signal that some condition is now true
  (for example, "a receiver is ready" or "a tile has been sent").
* A core **waits until the semaphore reaches a target value** before proceeding, ensuring it does not
  read data or start an action too early (e.g., before the data has been written to memory).

A semaphore can be created on one or more Tensix cores using the `CreateSemaphore` host-side API,
which allocates and initializes a semaphore in on-chip SRAM and returns a semaphore ID.
For example, in ``lab_multicast.cpp``, there are two semaphores created:

.. code-block:: cpp

   uint32_t receivers_ready_semaphore = CreateSemaphore(prog_state.program, all_cores_logical, 0);
   uint32_t tile_sent_semaphore = CreateSemaphore(prog_state.program, all_cores_logical, INVALID);

where ``prog_state.program`` is the TT-Metalium ``Program`` these semaphores belong to,
and ``all_cores_logical`` is the logical core range on which to create the semaphore
(in this example, all four cores). Finally, the last argument is the initial value of the
semaphore on each core. In this example, ``0`` is used for the ``receivers_ready_semaphore``
and ``INVALID`` is used for the ``tile_sent_semaphore``. ``INVALID`` is just a constant integer
value defined in the TT-Metalium API to make the code more readable.
``tile_sent_semaphore`` is intialized to ``INVALID`` because initially the sender core has not sent any tiles.

The IDs returned by `CreateSemaphore` are then passed as kernel arguments so that kernels can use
them to access the semaphore on the core they are running on.

Note that this example program creates both semaphores on all cores, although not all cores will use both semaphores.
This is because overhead of creating a semaphore is minimal, so it is easier and less error prone to create semaphores
on all cores rather than creating them on a subset of cores.
Having said that, number of semaphores on each core is limited in hardware, so in cases where there are many semaphores,
it may be necessary to create them on a subset of cores to avoid running out of semaphore hardware resources.

High Level Multicast Protocol
=============================

Before looking at code in more detail, it is helpful to describe the multicast protocol at a high level, which is shown in Figure 4.

.. figure:: images/multicast_protocol.png
   :alt: Multicast Protocol
   :width: 1200
   :align: center

   Figure 4: Multicast Protocol

Figure 4(a) shows the multicast protocol near the beginning of kernel code execution,
with all semaphores at their initial values.
The sender core has just read a tile from DRAM into its input CB, and is ready to multicast it
to other cores. However, it must wait until all receivers signal that they are ready for the tile.
The way this is achieved is by waiting for the ``receivers_ready`` semaphore,
which resided **in the senders on-chip SRAM**, to reach the number of receivers.
Waiting on a semaphore is a blocking call and does not involve any NOC traffic since the
semaphore is in the local on-chip SRAM.

Receivers for their part must allocate space in their input CB for the incoming tile and then
signal that they are ready to receive the next tile. They do so by calling ``noc_semaphore_inc``
on the ``receivers_ready`` semaphore **in the sender's on-chip SRAM**, to increment it by 1.
This is shown in Figure 4(b).
Note that this increment operation requires a NOC transaction, since we wish to update the ``receivers_ready``
semaphore is in the sender's on-chip SRAM.
Each receiver core sends an independent increment transaction to the sender core, so order
of increments is not guaranteed. However, incrementing a semaphore is an atomic operation, so
the sender will eventually see the correct number of receivers ready for the tile.
Having indicated its readiness to receive a tile, each receiver core then waits for the sender to multicast
the tile to it. This is done by waiting on the ``tile_sent`` semaphore **in the receiver's on-chip SRAM**.
This wait operation also does not involve any NOC traffic since the semaphore is in the local on-chip SRAM.

Once the sender core has seen the correct number of receivers ready for the tile,
it can multicast the tile to all receiver cores, using the ``noc_async_write_multicast`` function.
This is illustrated in Figure 4(c).
The sender core also resets the ``receivers_ready`` semaphore to ``0`` to avoid accidental reuse of
the same semaphore value, and in preparation for the next tile.
Since the ``receivers_ready`` semaphore is in the sender's on-chip SRAM, this does not require any NOC traffic.

Having sent the tile to all receiver cores, the sender core must signal to the receivers that the tile has been sent.
This is done by calling ``noc_semaphore_set_multicast`` on the ``tile_sent`` semaphore **in the receiver's on-chip SRAM**,
to set it to ``VALID``. This is illustrated in Figure 4(d).
Since we wish to update the ``tile_sent`` semaphore on **all receiver cores**, this requires a NOC multicast command.

Finally, once a receiver core observes that its ``tile_sent`` semaphore has been set to ``VALID``,
it can proceed to consume the tile. Once the tile has been consumed, the receiver core calls ``noc_semaphore_set``
on the ``tile_sent`` semaphore in its own on-chip SRAM, to set it to ``INVALID`` to prepare for the next tile.
This is illustrated in Figure 4(e).
As can be seen, the state of all the semaphores is now the same as at the beginning of the protocol,
ready for the next tile to be multicast.

While this high-level protocol description is helpful to understand the overall flow of the multicast operation,
there are some API details worth exploring in more depth, which is described in the following sections.

* The source address of the tile in the sender's on-chip SRAM.
* The destination address of the tile in the receiver's on-chip SRAM.
* The size of the tile in bytes.
* The number of receivers to multicast to.

This is shown in Figure 4(c).
Note that this multicast operation requires a NOC transaction to be sent to all receiver cores.

 then waits until all receivers signal that they are ready for the tile.
Note that the protocol is the same for all tiles, so the same code is executed for each tile.

The sender core reads a tile from DRAM into its input CB, then waits until all receivers signal that they are ready for the tile.
Once all receivers are ready, the sender core multicasts the tile to all receiver cores.
The sender core then flushes the NOC writes to ensure the multicast command has been sent.
The sender core then multicasts a semaphore update to tell receivers that the tile has been sent.
The sender core then waits for the multicast to complete before freeing the CB slot.
The sender core then pops the tile from the CB (free the slot for a future tile).

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

This protocol uses semaphores to coordinate when each tile is safe to multicast and when it is safe to consume,
while CB operations manage local double buffering.




uint32_t receivers_ready_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(4));
uint32_t tile_sent_semaphore_addr       = get_semaphore(get_arg_val<uint32_t>(5));
```

`get_semaphore` maps the index back to the L1 address of the semaphore on that core.
The sender kernel then uses that local address with `noc_semaphore_wait` and `noc_semaphore_set`.
Receiver kernels use `get_noc_addr(sender_x, sender_y, receivers_ready_semaphore_addr)` to build
a remote NOC address that points to the sender's copy of the same semaphore,
so they can increment it when they are ready.





You may have noticed that the sender core doesn't specify any compute or writer kernels.
While this is acceptable, it is not the most efficient way to use the sender core.
In a real application, the sender core would also perform computation and writeback.

Therefore, exercise ...


Note that in this example the sender core only multicasts data; it does not currently run the same compute pipeline as receivers.
You will change that in Exercise 1.



Multicast and Double Buffering
==============================

In this lab, multicast is combined with **double buffering** in the CBs:

* On the sender, double buffering allows overlapping:
  * Reading the next tile from DRAM into on-chip SRAM.
  * Multicasting the previous tile to receivers.
* On each receiver, double buffering allows overlapping:
  * Receiving a tile via multicast into its input CB.
  * Computing on previously received tiles.

Double buffering still works with multicast, as long as:

* You do not reuse a CB slot until all NOC operations that write to that memory have completed.
* You maintain a consistent pattern of ``cb_reserve_back``, ``cb_push_back``, ``cb_wait_front``,
  and ``cb_pop_front`` across sender and receivers, so that everyone agrees which memory addresses
  are used for which tile at each step.

You will see this pattern in the standalone multicast example first, then reuse it in the matmul case.



Core Roles in the Basic Multicast Example
-----------------------------------------




New Concepts and APIs in the Multicast Example
==============================================

This section focuses on **new constructs** that were not needed in Labs 1 and 2. You should refer to the example code as you read.

Device Coordinates and ``worker_core_from_logical_core``
--------------------------------------------------------

Host code uses logical coordinates to create kernels and CBs:

.. code-block:: cpp

   CoreRange all_cores_logical({0, 0}, {3, 0});
   CoreCoord sender_core_logical = {0, 0};
   CoreRange receiver_cores_logical({1, 0}, {3, 0});

To perform NOC operations in device kernels, you need **device coordinates**. The host converts logical coordinates to device coordinates:

.. code-block:: cpp

   CoreCoord sender_core_device =
       mesh_device->worker_core_from_logical_core(sender_core_logical);

   CoreRange receiver_cores_device(
       mesh_device->worker_core_from_logical_core(receiver_cores_logical.start_coord),
       mesh_device->worker_core_from_logical_core(receiver_cores_logical.end_coord));

The sender's runtime arguments then include the device coordinates of the receiver range:

.. code-block:: cpp

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
  * You always pass **logical coordinates** to ``CreateKernel``, ``CreateCircularBuffer``, and ``SetRuntimeArgs``.
* In device kernels:
  * You must use **device coordinates** when calling NOC APIs that address other cores (e.g., multicast).
  * Device coordinates are passed into kernels through runtime arguments that the host constructs using ``worker_core_from_logical_core``.

``tt_l1_ptr`` Macro
-------------------

In the sender and receiver kernels, semaphore pointers are declared using the ``tt_l1_ptr`` macro:

.. code-block:: cpp

   volatile tt_l1_ptr uint32_t* sender_sem_ptr =
       reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receivers_ready_semaphore_addr);

``tt_l1_ptr`` expands to a compiler attribute indicating that the pointer refers to **on-chip (L1) memory**. This helps the compiler:

* Optimize address calculations.
* Avoid unnecessary loads/stores.
* Potentially use specialized addressing modes.

You should use ``tt_l1_ptr`` for pointers to on-chip (L1) memory that are accessed from kernels. The macro does not change program semantics, but it enables better compiler optimizations.


TODO: Explain that semaphores don't need to be created on all cores, but it is common to do so because overhead is minimal.
Having said that, number of sempahores on each core is lmited, so may need to create it on a subset of cores if there are many semaphores.

Semaphores are used for coordination between cores. The sender and receivers share two semaphores:

* ``receivers_ready_semaphore``:
  * Stored in the sender's on-chip SRAM (but accessible over NOC).
  * Receivers increment it to indicate they are ready for the next tile.
* ``tile_sent_semaphore``:
  * Also stored in the sender's on-chip SRAM.
  * The sender multicasts its value to receivers to indicate that a tile has been sent.

In the sender kernel:

.. code-block:: cpp

   volatile tt_l1_ptr uint32_t* sender_sem_ptr =
       reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receivers_ready_semaphore_addr);

   // Wait for all receivers to signal they are ready for the next tile.
   noc_semaphore_wait(sender_sem_ptr, num_receivers);
   noc_semaphore_set(sender_sem_ptr, 0);

``noc_semaphore_wait(sender_sem_ptr, num)`` blocks until the value stored at ``sender_sem_ptr`` becomes equal to ``num``. In this example:

* Each receiver calls ``noc_semaphore_inc`` on the sender's "receivers_ready" semaphore to increment it by 1.
* When all ``num_receivers`` receivers have incremented it, the sender proceeds, then resets the semaphore to 0 with ``noc_semaphore_set``.

On the receiver side, for the "tile sent" semaphore:

.. code-block:: cpp

   volatile tt_l1_ptr uint32_t* tile_sent_sem_ptr =
       reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_sent_semaphore_addr);

   // Reset tile_sent semaphore to INVALID before signaling ready
   noc_semaphore_set(tile_sent_sem_ptr, INVALID);

   // ...

   // Wait for sender to multicast the tile (semaphore becomes VALID)
   noc_semaphore_wait(tile_sent_sem_ptr, VALID);

Here ``noc_semaphore_wait(tile_sent_sem_ptr, VALID)`` waits until the sender multicasts a "VALID" update to this semaphore.

``noc_async_write_multicast``
-----------------------------

The sender multicasts tiles using:

.. code-block:: cpp

   uint64_t tile_mcast_addr =
       get_noc_multicast_addr(receiver_start_x, receiver_start_y,
                              receiver_end_x, receiver_end_y,
                              l1_read_addr);

   noc_async_write_multicast(l1_read_addr, tile_mcast_addr, tile_size_bytes, num_receivers);

Key points:

* ``noc_async_write_multicast`` is **non blocking**:
  * It enqueues a multicast transfer on the NOC, then **returns control to the kernel immediately**.
  * The hardware performs the tile transfer in the background.
* Multicast is far more efficient than issuing ``num_receivers`` separate ``noc_async_write`` calls:
  * One command instead of many.
  * Less contention and overhead on the NOC command FIFOs.
  * Simpler synchronization, because all receivers observe the same sequence of operations.

TODO: Explain Why ``noc_async_write_multicast`` needs ``num_dests`` parameter?
Why isn't that automatically calculated from ``dst_noc_addr_multicast``, which encodes the range of destinations?

TODO: document limitations of ``noc_async_write_multicast``:

In short:

They **cannot** be an arbitrary set of cores.

For ``noc_async_write_multicast`` the documented and implemented constraints are:

1. **Rectangular grid only (base API)**
   The targets must be all Tensix worker cores in a **contiguous rectangle** ``[x_start..x_end] x [y_start..y_end]``
   on the NoC, all at the same memory address.
   The docs say explicitly:

   - "The destination nodes can only be a set of Tensix cores + a memory address."
   - "The destination nodes must form a **rectangular grid**."

   So: not "same row" or "same column" only; but **any axis-aligned rectangle** is fine.

2. **L-shaped variant = rectangle minus rectangle**
   There is a separate API (multicast with exclude region) where you specify a rectangle and then subtract a rectangular "exclusion zone" to get an **L-shaped** pattern, but even that is *"rect grid minus sub-rect grid"*, not an arbitrary subset.

3. **Same memory address on all destinations**
   All destination cores must use the **same memory address**; the multicast NOC address encodes one local address plus the rectangular coord range, not per-core offsets.

4. **Sender cannot be in the destination set (for this API)**
   The base ``noc_async_write_multicast`` excludes the sender core; if you want the sender included you must use the ``*_loopback_src`` variant.

5. **Cardinality is otherwise unconstrained**
   Aside from "non-zero" and "<= number of cores - 1", the number of destinations can be as large as "full chip rectangle."

So if you need to hit an **arbitrary set of scattered cores** (e.g. "this triangle" or a few disjoint islands), you have to implement that as:

- multiple multicast calls to different rectangles / L-shapes, or
- fall back to multiple unicast writes.

A single ``noc_async_write_multicast`` call always targets a **rectangular (or L-shaped via the exclude API) contiguous region**, not an arbitrary mask of cores.



Ordering with ``noc_async_writes_flushed``
------------------------------------------

Because ``noc_async_write_multicast`` is non blocking, you must ensure that:

* The multicast command is actually **sent onto the NOC** before you signal receivers that the tile is ready.

This is handled using:

.. code-block:: cpp

   noc_async_writes_flushed();

   // Signal receivers that tile has been sent by multicasting VALID to receiver semaphore
   *receiver_sem_ptr = VALID;
   noc_semaphore_set_multicast(tile_sent_semaphore_addr, receiver_sem_mcast_addr, num_receivers);

``noc_async_writes_flushed()`` does **not** wait for multicast transfers to complete. It only ensures that all outstanding enqueued ``noc_async_write`` calls issued on the current core have **departed** (have been pushed into the NOC) before the function returns.

Why is it safe to send the semaphore update before the transfers complete?

* The sender:
  * Issues the multicast command.
  * Calls ``noc_async_writes_flushed()`` to ensure the command has been sent into the NOC.
  * Then multicasts the "tile_sent" semaphore value.
* Each receiver:
  * Waits for the "tile_sent" semaphore to become ``VALID`` before using the tile.
  * The NOC ensures that the data transfer and the semaphore update are observed in the correct order.
* Additionally, the sender calls:

  .. code-block:: cpp

     noc_async_write_barrier();

  before popping the CB slot, ensuring that the tile transfer has completed before the sender reuses the CB memory location.

TODO: Explain that this is a simplified view of the reality and that for optimal performance on some architectures we may do things differently.
Also, NOC is customizable and may behave differently if customizations are made.


``get_noc_multicast_addr`` and CB Address Synchronization
---------------------------------------------------------

To multicast data, the sender needs to know **where** in each receiver's on-chip SRAM to write the tile.
This is the job of ``get_noc_multicast_addr``:

.. code-block:: cpp

   uint64_t tile_mcast_addr =
       get_noc_multicast_addr(receiver_start_x, receiver_start_y,
                              receiver_end_x, receiver_end_y,
                              l1_read_addr);

The last argument passed to ``get_noc_multicast_addr`` is a **memory address at the destination**. In this example:

* Sender and receivers use the same CB index (e.g., input CB).
* All cores reserve CB slots and push/pop tiles in the **same pattern**.
* For each tile index:
  * The write pointer in the sender CB points to the **same offset** within the corresponding CB as the write pointer in the receivers.
  * Therefore, using the sender's local CB address as the "destination address" in ``get_noc_multicast_addr`` is correct.

You can think of the return value of ``get_noc_multicast_addr`` as a packed 64 bit encoding of:

* The destination core rectangle: ``(x_start, y_start, x_end, y_end)``.
* The destination on-chip SRAM address.

In general, there is no direct way for receivers to "tell" the sender which exact memory address they are using for a CB slot.
Instead, the design ensures that **all cores run the same CB protocol**:

* Each core executes the same sequence of:
  * ``cb_reserve_back``
  * ``get_write_ptr``
  * ``cb_push_back``
  * ``cb_wait_front``
  * ``get_read_ptr``
  * ``cb_pop_front``
* Because CB sizes and page sizes are identical, the CB write pointer for a given tile index is the same on the sender and all receivers.
* This makes it possible for the sender to use its own CB write pointer as the destination address in ``get_noc_multicast_addr``.

TODO: Example code still has variables like sender_sem_ptr, receiver_sem_ptr, and receiver_sem_mcast_addr, rather than referenceing receivers_ready and tile_sent semaphore names.


Exercise 1: Extending the Standalone Multicast Example
******************************************************

In the provided example, the sender core only forwards data, while receivers perform the copy and writeback. In this exercise you will:

1. Build and run the example to verify it works as is.
2. Extend it so that **the sender core also participates in the same computation**, using the same multicast data.
3. Update the host side verification to include the sender's results.

Step 1: Build and run the base example
======================================

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
=====================================================

Currently the sender core:

* Reads tiles from DRAM into input CB ``c_in0``.
* Multicasts tiles to receivers.
* Does not pass these tiles to the compute kernel on the sender core.

Your goal is to modify the program so that:

* The sender core also runs the **same compute and writer kernels** as receivers (copying tiles and writing them to its own region in the output tensor).
* After the change, output of the program will be a tensor that contains four copies of the input tensor; one from the sender core and three from the receiver cores.

The simplest way to achieve this is for the sender core to run the same compute kernel as the receivers.
This means that it will use the same CB that all other cores use for input tiles, which is ``c_0``.
Observe that this is the same CB that is used by the sender as a source of data for multicast.
The ``mcast_sender`` kernel code can stay largely the same, with one key difference: it should not
perform any ``cb_wait_front`` or ``cb_pop_front`` operations, because the compute kernel will be doing this work.


 already configured to multicast tiles from ``c_0`` to the receiver cores.

TODO: A figure would come handy here!!!

The main required change to achieve the objective of this exercise is to update the host program to include the sender
core in the core range when creating the compute and writer kernels and pass the appropriate runtime arguments.
The writer kernel itslef does not need to change at all.
Circular buffers are already created on all cores, so there is no change required to the circular buffer setup.
The existing semaphore protocol already supports sending tiles from sender to receivers.
The sender acting as a "local receiver" for compute does **not** require introduction of additional semaphores.
This is because the sender already knows when a tile is in its local CB (right after the DRAM read completes).
     * The compute kernel on the sender simply waits for tiles in ``c_in0`` in the same way as receivers.




Follow these steps to complete the exercise:

#. Update the host program to:
   * Include the sender core in the core range when creating the compute and writer kernels.
   * Include the sender core in the core range when setting up the runtime arguments for the writer kernel.
   * Update ``output_data`` and related variables to account for the additional copy from the sender core.
     Make sure that each core writes to a unique region of the output tensor.
   * If you created a new folder for this exercise, make sure to update ``CreateKernel`` with paths to kernels in the new folder.

#. Update the ``mcast_sender`` kernel code to not perform any ``cb_wait_front`` or ``cb_pop_front`` operations.
   Since the code won't perform ``cb_wait_front``, this also means that it cannot call ``get_read_ptr``.
   Instead, the source address for multicast should be the same address that was used for writing the tile to the CB.

#. Ensure that the sender core's CB usage matches receivers:
   * The sender already uses ``c_in0`` to hold input tiles.
   * For the compute and writer kernels, ensure they are created to run on:
     * All cores that should produce output (including the sender).
   * Make sure ``c_out0`` (or the CB used for output tiles) exists and has the same size on the sender as on receivers.

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
     * The compute kernel on the sender simply waits for tiles in ``c_in0`` in the same way as receivers.
   * Ensure CB sizes are sufficient for:
     * Overlapping DRAM reads and NOC transfers.
     * Overlapping NOC transfers and compute on all participating cores.

Step 3: Update verification to include the sender's output
==========================================================

The host currently verifies only the receiver outputs. Modify verification so that:

1. The sender's output data is included in the DRAM tensor that you read back.
2. The reference comparison checks **all 4 copies**:

   * If you use a layout where each core writes ``n_tiles`` tiles to a consecutive region:
     * Core index ``0`` writes tiles ``0..n_tiles-1``.
     * Core index ``1`` writes tiles ``n_tiles..2*n_tiles-1``.
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


Remind students to use ``tt-smi -r`` to reset the device after encountering hangs/issues.


Applying Multicast to Multi Core Matmul
***************************************

In Lab 2, you reduced DRAM traffic in multi core matrix multiplication by:

* Dividing the output tiles into **blocks** ``C_block`` and assigning each block to a core.
* Splitting the inner ``K`` dimension into **K-blocks** of size ``K_block_tiles``, so that ``Kt = K / TILE_WIDTH`` is divided into ``num_k_blocks = Kt / K_block_tiles`` equal chunks.
* For each core and each K-block index ``b`` (``0 .. num_k_blocks - 1``), defining:

  * ``A_slab(b)``: a **slab** of tiles from ``A`` of size ``M_block_tiles x K_block_tiles``, covering the rows of that core's ``C_block`` and the K indices in block ``b``.
  * ``B_slab(b)``: a **slab** of tiles from ``B`` of size ``K_block_tiles x N_block_tiles``, covering the columns of that core's ``C_block`` and the K indices in block ``b``.

Each core in Lab 2:

* Loaded its own ``A_slab(b)`` into an input CB (e.g., CB0),
* Loaded its own ``B_slab(b)`` into another input CB (e.g., CB1),
* Used those slabs to update all tiles in its ``C_block`` for that K-block,
* Repeated this for every K-block, with partial results stored in an intermediate CB.

The key observation is:

* For a **given row** of cores (fixed ``y`` in the core grid) and a fixed K-block index ``b``, **all cores in that row need the same ``A_slab(b)``**:

  * They share the same range of output rows (same ``M_block_tiles`` rows of ``C``),
  * They differ only in which columns of ``C`` they cover (different ``N_block_tiles`` regions),
  * So their required tiles of ``A`` for that K-block are identical.

* Similarly, for a **given column** of cores (fixed ``x``) and K-block index ``b``, **all cores in that column need the same ``B_slab(b)``**.

In Lab 2 each core independently loaded its own copy of these slabs from DRAM, even though:

* Every core in row ``y`` for K-block ``b`` needed the same ``A_slab(b)``,
* Every core in column ``x`` for K-block ``b`` needed the same ``B_slab(b)``.

In this lab, we keep the same blocked and slab-based algorithm, but change how slabs are brought into CBs:

* For each row ``y`` and K-block ``b``, exactly one core in that row reads **one copy of** ``A_slab(b)`` from DRAM and multicasts every A tile in that slab to the other cores in that row.
* For each column ``x`` and K-block ``b``, exactly one core in that column reads **one copy of** ``B_slab(b)`` from DRAM and multicasts every B tile in that slab to the other cores in that column.

Compute kernels and writer kernels remain unchanged; only the **slab loading stage** (how ``A_slab(b)`` and ``B_slab(b)`` end up in CBs) is modified.

Core Roles and Slabs on a 5x5 Grid
==================================

Assume a rectangular grid of logical cores of size ``core_grid.x`` by ``core_grid.y``. As in Lab 2, we divide the tiled output matrix ``C`` into blocks:

* Each core ``(x, y)`` is responsible for a block of output tiles ``C_block(x, y)``:

  * ``M_block_tiles`` rows of C tiles,
  * ``N_block_tiles`` columns of C tiles.

The mapping is chosen so that:

* Cores with the same ``y`` index cover **different columns** but the same set of **rows** in C.
* Cores with the same ``x`` index cover **different rows** but the same set of **columns** in C.

For each K-block index ``b``:

* Every core in row ``y`` needs **the same ``A_slab(b)``** (same rows of A for the K indices in that block).
* Every core in column ``x`` needs **the same ``B_slab(b)``** (same columns of B for the K indices in that block).

TODO: Add a figure here that shows arrows from sender cores to receiver cores.


On a 5x5 grid, we assign four roles:

* **Top-left core** ``(0, 0)``:
  * Computes ``C_block(0, 0)``.
  * Reads ``A_slab(b)`` for row 0 and ``B_slab(b)`` for column 0 from DRAM.
  * Multicasts A tiles horizontally to all cores in row 0.
  * Multicasts B tiles vertically to all cores in column 0.

* **Top row B-source cores** ``(x, 0)``, ``x > 0``:
  * Compute ``C_block(x, 0)``.
  * For each K-block, read ``B_slab(b)`` for column ``x`` from DRAM and multicast its B tiles down column ``x``.
  * Receive A tiles by multicast from the left column core in their row (``(0, 0)`` for row 0).

* **Left column A-source cores** ``(0, y)``, ``y > 0``:
  * Compute ``C_block(0, y)``.
  * For each K-block, read ``A_slab(b)`` for row ``y`` from DRAM and multicast its A tiles across row ``y``.
  * Receive B tiles by multicast from the top row core in their column (``(0, 0)`` for column 0).

* **Interior cores** ``(x, y)`` with ``x > 0`` and ``y > 0``:
  * Compute ``C_block(x, y)``.
  * Do not read A or B directly from DRAM.
  * For each K-block:

    * Receive ``A_slab(b)`` by multicast from the A-source core at ``(0, y)``.
    * Receive ``B_slab(b)`` by multicast from the B-source core at ``(x, 0)``.

A conceptual diagram for a 5x5 grid:

.. code-block:: text

   5x5 core grid roles (per C_block):

       x = 0    1      2      3      4
     +------+------+------+------+------+
   y  |      |      |      |      |      |
   =0 |  T   |  R   |  R   |  R   |  R   |
     +------+------+------+------+------+
   1 |  C   |  I   |  I   |  I   |  I   |
     +------+------+------+------+------+
   2 |  C   |  I   |  I   |  I   |  I   |
     +------+------+------+------+------+
   3 |  C   |  I   |  I   |  I   |  I   |
     +------+------+------+------+------+
   4 |  C   |  I   |  I   |  I   |  I   |

Where:

* ``T`` is the top-left core (A-source and B-source for row 0 and column 0).
* ``R`` are top row B-source cores.
* ``C`` are left column A-source cores.
* ``I`` are interior cores (A- and B-receivers).

For each row ``y`` and K-block ``b``:

* The **A-source core** at ``(0, y)`` reads **one copy** of ``A_slab(b)`` from DRAM and multicasts its tiles to all cores in row ``y``.
* Every core in row ``y``, including the source, ends up with the same ``A_slab(b)`` in its A CB (CB0).

For each column ``x`` and K-block ``b``:

* The **B-source core** at ``(x, 0)`` reads **one copy** of ``B_slab(b)`` from DRAM and multicasts its tiles to all cores in column ``x``.
* Every core in column ``x``, including the source, ends up with the same ``B_slab(b)`` in its B CB (CB1).

Slab Storage in CBs with Multicast
==================================

From Lab 2, recall that:

* Input CBs must be sized to store **full slabs**:

  * A CB (CB0) must store at least ``M_block_tiles * K_block_tiles`` tiles for ``A_slab(b)``.
  * B CB (CB1) must store at least ``K_block_tiles * N_block_tiles`` tiles for ``B_slab(b)``.

* In the Lab 2 version, each core used a reader kernel to fill CB0 with its own ``A_slab(b)`` and CB1 with its own ``B_slab(b)`` directly from DRAM for each K-block.

In Lab 3:

* CB capacities and the **slab organization inside CBs** remain the same.
* What changes is **which kernel fills those slabs**:

  * On A-source cores:
    * A-reader/multicast kernel reads all tiles of ``A_slab(b)`` from DRAM into CB0 and multicasts each tile to the rest of the row.
  * On other cores in the row:
    * A-receiver kernel receives all tiles of ``A_slab(b)`` via multicast into CB0.

  Consequently, every core in the row has a complete **identical** slab ``A_slab(b)`` in CB0.

  * On B-source cores:
    * B-reader/multicast kernel reads all tiles of ``B_slab(b)`` from DRAM into CB1 and multicasts each tile down the column.
  * On other cores in the column:
    * B-receiver kernel receives all tiles of ``B_slab(b)`` via multicast into CB1.

  Consequently, every core in the column has a complete **identical** slab ``B_slab(b)`` in CB1.

The order of tiles inside each slab within the CBs stays **exactly the same** as in Lab 2:

* A slabs are stored in CB0 in **slab row-major order** (over ``i`` and local K index ``k_local``).
* B slabs are stored in CB1 in **slab row-major order** (over ``k_local`` and ``j``).

This guarantees that the compute kernel can continue to index ``A_slab(b)`` and ``B_slab(b)`` tiles using its existing logic (conceptually ``A_slab_tile(i, k_local)`` and ``B_slab_tile(k_local, j)``), without any awareness of multicast.

Pseudocode with Slabs and Multicast
===================================

At slab level, the Lab 2 compute pseudocode remains:

.. code-block:: cpp

   // For every K-block:
   for (b in 0 ..
        num_k_blocks - 1) {

       // After slab loading:
       //  - A_slab(b) is present in CB0 on this core.
       //  - B_slab(b) is present in CB1 on this core.

       for (i in 0 ..
            M_block_tiles - 1) {
           for (j in 0 ..
                N_block_tiles - 1) {

               // Load partial result for C(i, j) if b > 0
               // or initialize acc_tile if b == 0, as in Lab 2.
               ...

               // For each K tile inside this K-block slab:
               for (k_local in 0 ..
                    K_block_tiles - 1) {
                   // Compute tile indices into A_slab(b) and B_slab(b)
                   a_tile = A_slab_tile(i, k_local);
                   b_tile = B_slab_tile(k_local, j);
                   acc_tile += matmul(a_tile, b_tile);
               }

               // Store partial or final result into appropriate CB.
               ...
           }
       }
   }

The only new phase is **how A_slab(b) and B_slab(b) get into CBs** for each core.

For A slabs (row multicast), you can describe the protocols as follows:

*On the A-source core for row y:*

.. code-block:: cpp

   // For A slabs on row y
   for (b in 0 ..
        num_k_blocks - 1) {

       // Read and multicast all tiles in A_slab(b)
       for (t in 0 ..
            M_block_tiles * K_block_tiles - 1) {

           // 1. Read the next A tile of this slab from DRAM
           cb_reserve_back(A_cb, 1);
           uint32_t addr = get_write_ptr(A_cb);
           noc_async_read_tile(global_a_tile_idx_for_row_y_and_kblock(b, t),
                               A_addr_gen, addr);
           noc_async_read_barrier();
           cb_push_back(A_cb, 1);

           // 2. Wait until all row receivers are ready for this tile
           noc_semaphore_wait(A_receivers_ready_sem_ptr[y], num_receivers_in_row[y]);
           noc_semaphore_set(A_receivers_ready_sem_ptr[y], 0);

           // 3. Multicast the tile to all cores in row y
           uint64_t mcast_addr = get_noc_multicast_addr(
               row_start_x, y,
               row_end_x,   y,
               addr);
           noc_async_write_multicast(addr, mcast_addr, tile_size_bytes, num_receivers_in_row[y]);

           // 4. Ensure multicast command is issued before signaling receivers
           noc_async_writes_flushed();

           // 5. Multicast "tile sent = VALID" semaphore
           *A_tile_sent_sem_ptr[y] = VALID;
           noc_semaphore_set_multicast(
               local_A_tile_sent_sem_addr[y],
               mcast_A_tile_sent_sem_addr[y],
               num_receivers_in_row[y]);

           // 6. Wait for multicast completion before reusing this CB slot
           noc_async_write_barrier();
       }

       // After this loop:
       //  - This core's CB0 contains A_slab(b).
       //  - All cores in row y have received A_slab(b) into their CB0.
   }

*On each A-receiver core in the same row y (including optionally the source if you choose to reuse the same CB protocol):*

.. code-block:: cpp

   for (b in 0 ..
        num_k_blocks - 1) {

       for (t in 0 ..
            M_block_tiles * K_block_tiles - 1) {

           // Reserve space for incoming A tile
           cb_reserve_back(A_cb, 1);

           // Reset local "tile sent" semaphore and signal ready to source
           noc_semaphore_set(A_tile_sent_sem_ptr[y], INVALID);
           noc_semaphore_inc(A_receivers_ready_sem_addr[y], 1);

           // Wait until source multicasts "tile sent = VALID"
           noc_semaphore_wait(A_tile_sent_sem_ptr[y], VALID);

           // The tile is now at the CB write pointer
           cb_push_back(A_cb, 1);
       }

       // After this loop, CB0 on this core holds the full A_slab(b),
       // in the same order as on the source core.
   }

For B slabs (column multicast), the structure is analogous, but along columns and using a B-specific set of CBs (e.g., CB1), address generators, and semaphores.

Interaction with Double Buffering
=================================

In Lab 2 you were instructed to size CBs so that:

* Input CBs (for A and B slabs) can hold **full slabs** and use **double buffering**:

  * CB0 has capacity for at least ``2 * M_block_tiles * K_block_tiles`` tiles,
  * CB1 has capacity for at least ``2 * K_block_tiles * N_block_tiles`` tiles.

This allowed the reader kernels to:

* Load slab ``b+1`` while the compute kernel is still processing slab ``b`` for the same core.

In Lab 3, double buffering plays the same role:

* On A- and B-source cores, double buffering allows overlapping:

  * DRAM reads and multicast for slab ``b+1``,
  * With compute and writeback for slab ``b`` on all cores in that row or column.

* On receiver cores, double buffering allows:

  * Receiving the tiles of slab ``b+1`` via multicast,
  * While the compute kernel is still consuming slab ``b``.

To maintain correctness:

* You must not reuse a CB slot (for slab ``b+1``) until:

  * The compute kernel has called ``cb_pop_front`` for the tile held in that slot (freeing it from the CB's perspective), and
  * The source kernel has ensured that all multicast transfers involving that tile have completed (using ``noc_async_write_barrier``).

As long as these two conditions are enforced, double buffering and multicast coexist correctly with slab-based processing.

Exercise 2: Multi Core Matrix Multiplication with Multicast and Slabs
*********************************************************************

In this exercise, you will start from your **Exercise 2 solution from Lab 2** (multi core matrix multiplication with blocked data reuse using slabs) and extend it to:

* Use slab-level multicast for A slabs across rows,
* Use slab-level multicast for B slabs down columns,
* Retain the same blocked compute kernel and writer kernel,
* Preserve correctness and then compare performance to the Lab 2 version.

Use the same matrix sizes and tile sizes as before:

* ``A``: ``640x320``,
* ``B``: ``320x640``,
* ``C``: ``640x640``,
* Tiles: ``TILE_HEIGHT == TILE_WIDTH == 32``.

And test at least the same core grid sizes as in Lab 2, such as:

* ``5x5`` core grid,
* ``10x10`` core grid.

Follow these steps:

#. **Review your Lab 2 implementation**

   Make sure your Lab 2 code:

   * Defines ``M_block_tiles``, ``N_block_tiles``, ``K_block_tiles``, and ``num_k_blocks``.
   * Sizes input CBs to store full slabs (with double buffering):

     * CB0 for ``A_slab(b)`` of size ``M_block_tiles * K_block_tiles`` tiles,
     * CB1 for ``B_slab(b)`` of size ``K_block_tiles * N_block_tiles`` tiles.

   * Loads ``A_slab(b)`` and ``B_slab(b)`` into CBs in slab row-major order.
   * Uses the blocked compute structure shown in Lab 2's pseudocode.

#. **Assign core roles**

   For your chosen core grid:

   * Define roles:

     * Top-left core: ``(0, 0)``,
     * Top row B-source cores: ``(x, 0)`` with ``x > 0``,
     * Left column A-source cores: ``(0, y)`` with ``y > 0``,
     * Interior cores: ``(x, y)`` with ``x > 0`` and ``y > 0``.

   * Verify that:

     * All cores in row ``y`` share the same rows of ``C`` (same ``M_block_tiles``),
     * All cores in column ``x`` share the same columns of ``C`` (same ``N_block_tiles``).

#. **Add semaphores for slab-level multicast**

   For A slabs:

   * For each row ``y``, allocate:

     * A row-specific "receivers ready" semaphore for A,
     * A row-specific "tile sent" semaphore for A.

   For B slabs:

   * For each column ``x``, allocate:

     * A column-specific "receivers ready" semaphore for B,
     * A column-specific "tile sent" semaphore for B.

   These can be created on the corresponding source cores and passed as runtime arguments to all source and receiver kernels that need them.

#. Create code for four types of reader kernels for the four types of cores.

#. Create appropriate kernels on appropriate cores for each of the four types of cores.
   Define ``CoreRange`` objects for each of the four types of cores to be used when calling ``CreateKernel`` for each of the four types of kernels.

#. **Implement A-slab multicast kernels**

   Modify your existing A reader logic from Lab 2:

   * On left column cores (A-source cores):

     * Create or adapt a kernel that:

       * For each K-block ``b``, reads all tiles of ``A_slab(b)`` for this row into CB0 in slab row-major order.
       * For each tile, uses the row-specific semaphores and NOC multicast APIs to:

         * Wait until all cores in the row are ready,
         * Multicast the tile to all cores in that row,
         * Signal that the tile has been sent,
         * Use NOC barriers before reusing CB slots for the next K-block.

   * On other cores in the row (A receivers):

     * Replace their A-reader kernel with an A-receiver kernel that:

       * For each K-block and each tile index in the slab:

         * Reserves CB0 space,
         * Signals readiness to the row's A-source core,
           Observe that receivers no longer read data from DRAM, so they don't need to calculate offsets within the slab.
           All such computations are done by the sender core which pushes data into CB..
         * Waits on the row's A "tile sent" semaphore,
         * Pushes the received tile into CB0.

   At the end of slab loading for K-block ``b``, every core in the row should have the same ``A_slab(b)`` in CB0.


#. **Implement B-slab multicast kernels**

   Apply the same pattern for B:

   * On top row cores (B-source cores):

     * For each K-block ``b``, read all tiles of ``B_slab(b)`` for that column into CB1.
     * For each tile, multicast down the column using column-specific semaphores and NOC multicast.

   * On other cores in the column (B receivers):

     * For each K-block and each tile index in the slab:

       * Reserve CB1 space,
       * Signal readiness to the column's B-source core,
       * Wait for the column's B "tile sent" semaphore,
       * Push the received tile into CB1.

   After slab loading for each K-block, every core in a given column should have the same ``B_slab(b)`` in CB1.

Make sure to account for the fact that number of receivers for A may be different than number of receivers for B in a general case.


#. **Reuse compute and writer kernels**

   Keep your Lab 2 compute and writer kernels unchanged:

   * Compute kernels still:

     * Expect full ``A_slab(b)`` in CB0 and full ``B_slab(b)`` in CB1,
     * Implement the K-blocked accumulation strategy using intermediate CBs.

   * Writer kernels still read ``C_block`` tiles from the output CB in row-major order and write them to the destination tensor in DRAM.

   Because slabs in CBs have the same layout and ordering as before, the compute and writer kernels do not need to know whether slabs arrived via DRAM reads or multicast.

#. **Set per-core runtime arguments**

   Update host code that sets runtime arguments:

   * For each A-source core:

     * Pass DRAM base addresses for A,
     * Row index ``y`` and the number of cores in that row,
     * Row-specific semaphore indices and any device coordinates needed to construct multicast addresses.

   * For each A-receiver core:

     * Pass device coordinates of its row's A-source core,
     * Row-specific semaphore indices,
     * Slab tile count.

   * Similarly for B-source and B-receiver kernels.

   As in Lab 2, iterate over all logical cores in your core grid, determine their role based on ``(x, y)``, and set the appropriate runtime arguments.

   Be careful: kernels are multicasting only within one row or one column, no core ever multicasts across both rows and columns.

   Remember that any coordinates in host code (e.g., in ``CreateKernel``, ``CreateSemaphore``, etc.) must be logical coordinates.
   Be particularly carful about ``SetRuntimeArgs`` function, which is a host function that takes logical coordinates to specify
   which cores will receive the runtime arguments, while any runtime arguments that refer to coordinates must themselves be in device coordinates.
   Similarly, coordinates determining what is leftmost or top are logical coordinates.

   On the other hand, any coordinates passed to kernels as arguments must be device coordinates. This includes the coordinates of the core whose
   logical coordinates are ``(0, 0)`` (i.e. you shouldn't assume that core with logical coordinates ``(0, 0)`` has device coordinates ``(0, 0)``).
   Advice: name all your coordinate variables with logical/device in the name to avoid any ambiguity.

   When converting lofgical to device coordinates, you can use the ``worker_core_from_logical_core`` function.
  Keep in mind that this function accepts only valid logical coordinates. For example, passing a coordinate outside the grid size will throw an error.

   Make sure to perform all arithmetic on coordinates before converting to device coordinates. Add or subtract any offsets
   (e.g., to determine the core to the right of the current core) in logical coordinate space, then convert the final result to device coordinates.|



   Reason: The adjacent physical core may be a non-tensix core, or it could be a harvested core.

#. **Verify correctness and profile performance**

   * Verify correctness by comparing the resulting C tensor to your CPU reference matmul (as in Lab 2).
   * Then enable the device profiler (``TT_METAL_DEVICE_PROFILER=1``) and measure firmware time for:

     * Your Lab 2 data reuse implementation (no multicast),
     * Your Lab 3 multicast implementation (same matrix sizes and core grids).

   * Plot firmware time (or speedup) vs number of cores, and compare:

     * Single-core baseline (Lab 1),
     * Multi core with slab-based reuse only (Lab 2),
     * Multi core with slab-based reuse plus multicast (this lab).

Debugging Advice
================

When debugging this lab, you may run into two broad classes of issues:

#. Program hangs
   This is most commonly because of sempahores not being updated correctly.
   Use DPRINT statements before and after ``noc_semaphore_wait``, ``noc_semaphore_set`` and ``noc_semaphore_inc`` statements.
   Remember that output of DPRINT automatically indicates ``(x, y)`` coordinates of the core.
   However, it doesn't automtically indicate the name of the kernel, so make sure to include brief kernel name
   if you're adding similar DPRINT statements in different kernels.

#. Program doesn't produce correct results
   This is most commonly because of incorrect slab loading, or mistakenly using addresses associated with matrix ``A`` for matrix ``B`` and vice versa.
   Use DPRINT statements to print the values in the CBs to verify that they are correct.
   To make debugging easier, you can reduce matrix and core grid sizes, while respecting the constraints on divisibility of different parameters.
   If you're copy-pasting code segments, it is easy to forget to update all the relevant variables.
   Consider whether you can refactor common code into functions to avoid copy-pasting and thus make your code less error prone
   (of course, in such a case make sure that all parameters passed to the function are correct when the function is called in multiple places with different parameters).

   Could also be due to incorrectly assigned runtime parameters to kernels.
   Make sure to DPRINT the runtime parameters to verify that they are correct.


TODO: Consider adding an exercise where different cores are assigned to different NOCs.

TODO: lab_example still uses confusing semaphore names in a few places. Review every use and fix them.

Conclusion
**********

In this lab, you refined the multi core, slab-based matrix multiplication implementation from Lab 2 by adding **cross-core reuse of slabs via multicast**:

* You kept the blocked structure and slab definitions from Lab 2:

  * Each core still computes a ``C_block`` using ``A_slab(b)`` and ``B_slab(b)`` for multiple K-blocks,
  * Partial results still live in an intermediate CB across K-blocks.

* You observed that for each K-block:

  * All cores in the same row need the **same** A slab, ``A_slab(b)``,
  * All cores in the same column need the **same** B slab, ``B_slab(b)``.

* You applied NoC multicast so that:

  * Each ``A_slab(b)`` is read from DRAM **once per row** and then multicast to all cores in that row,
  * Each ``B_slab(b)`` is read from DRAM **once per column** and then multicast to all cores in that column.

* You integrated multicast with:

  * The existing slab-sized CBs,
  * Double buffering for slab loading,
  * The same compute and writer kernels from Lab 2.

This lab shows how higher-level algorithmic structure (blocked matmul with slabs) can be combined with low-level architectural features (NoC multicast and semaphores) to further reduce DRAM traffic and potentially improve performance, without changing the core mathematical computation.

Additional information about Tenstorrent NOC can be found in the following resources:

* NoC (Network on Chip) Readme: https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/NoC/README.md
* Networks and Communication Lesson: https://github.com/tenstorrent/tt-vscode-toolkit/blob/main/content/lessons/cs-fundamentals-04-networks.md
* Introduction to Data Movement in TT-Metal: https://github.com/tenstorrent/tt-low-level-documentation/blob/main/data_movement_doc/general/intro_to_dm.md
