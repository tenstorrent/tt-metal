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
connected by a 2D **Network-on-Chip (NoC)** allowing the cores to pass data to each other.
While sending data over the NoC is more efficient than reading data from DRAM multiple times,
it still introduces some overhead. Therefore, we would like to minimize this overhead.
The NoC supports **unicast** and **multicast** operations.
Unicast allows a sender core to write data to a single destination core.
Multicast allows a sender core to write the same data to multiple destination cores in a single NoC operation,
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
Instead, we would like to load a tile from DRAM once and share it across all cores that need it through the NoC.
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

In the rest of this lab, you will first work through a simple example program demonstrating NoC and
multicast features, then retrofit your Lab 2 matrix multiplication solution to use multicast.


Background: Tenstorrent NoC and Multicast
*****************************************

The Network-on-Chip (NoC) is a 2D mesh interconnect that connects:
* All Tensix cores
* DRAM controllers
* PCIe interfaces
* Ethernet cores (for multi-device systems)

NoC is used to transfer data between different components of the device, including transferring
data between DRAM and on-chip SRAM. As you have seen in the preceding labs, TT-Metalium programmer
doesn't need to understand all the details of the underlying hardware to use the NoC.
In this lab, we will expand our use of the NoC to include multicast operations to transfer data between cores.
For more detailed information about the NoC, refer to the resources listed in the Additional Information
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
*************************

The main host program for the example multicast program is located in ``ttnn/examples/lab_multicast/lab_multicast.cpp``.
The program creates a 2D tensor and fills it with random data.
One **sender core** uses a reader kernel to read tiles of this tensor from DRAM and also multicasts them to three **receiver cores**.
The flow of data is shown in Figure 3.

.. figure:: images/data_flow_multicast.png
   :alt: Data flow in the multicast example program
   :width: 700
   :align: center

   Figure 3: Data flow in the multicast example program

Core ``(0,0)`` is the **sender core** and cores ``(1,0)``, ``(2,0)``, and ``(3,0)`` are **receiver cores**.
Receiver cores do not read the input tensor from DRAM, but receive tiles via multicast from the sender.
Each receiver core has three kernels:

* A reader kernel that manages CB and signals to the sender core when it is ready for the next tile.
* A compute kernel, which simply copies each tile to the output CB. In a real application, this is where computation would happen.
* A writer kernel that writes each tile int an appropriate region of the output tensor in DRAM.

The host reads back all receiver outputs and verifies that the output matches expectations,
which is a tensor that contains three copies of the original tensor, stacked vertically.
Note that number of tiles in Figure 3 is symbolic and doesn't accurately represent the number of tiles in the actual program.

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

A semaphore can be created on one or more Tensix cores using the ``CreateSemaphore`` host-side API,
which allocates and initializes a semaphore in on-chip SRAM and returns a semaphore ID.
For example, in ``lab_multicast.cpp``, there are two semaphores created:

.. code-block:: cpp

   uint32_t receivers_ready_semaphore = CreateSemaphore(prog_state.program, all_cores_logical, 0);
   uint32_t tile_sent_semaphore = CreateSemaphore(prog_state.program, all_cores_logical, INVALID);

where ``prog_state.program`` is the TT-Metalium ``Program`` these semaphores belong to,
and ``all_cores_logical`` is the logical core range on which to create the semaphore
(in this example, all four cores). Finally, the last argument is the initial value of the
semaphore on each core. In this example, ``0`` is used for the ``receivers_ready_semaphore``
to indicate that none of the receivers are ready to receive a tile.
Similarly, ``tile_sent_semaphore`` is initialized to ``INVALID`` value because initially the sender
core has not sent any tiles.. ``INVALID`` and ``VALID`` are constant integer values defined in the
TT-Metalium API to make the code more readable.

The IDs returned by ``CreateSemaphore`` are then passed as kernel arguments so that kernels can use
them to access the semaphore on the core they are running on.


High Level Multicast Protocol
=============================

Before looking at code in more detail, it is helpful to describe the multicast protocol at a high level, which is shown in Figure 4.

.. figure:: images/multicast_protocol.png
   :alt: Multicast Protocol
   :width: 1900
   :align: center

   Figure 4: Multicast Protocol

Figure 4(a) shows the multicast protocol near the beginning of kernel code execution,
with all semaphores at their initial values.
The sender core has just read a tile from DRAM into its input CB, and is ready to multicast it
to other cores. However, it must wait until all receivers signal that they are ready for the tile.
The way this is achieved is by waiting for the ``receivers_ready`` semaphore,
which resided **in the senders on-chip SRAM**, to reach the value equal to the number of receivers,
which is three in our example program.
Waiting on a semaphore is a blocking call and does **not** involve any NoC traffic since the
semaphore is in the local on-chip SRAM.

Receivers for their part must allocate space in their input CB for the incoming tile and then
signal that they are ready to receive the next tile. They do so by calling ``noc_semaphore_inc``
on the ``receivers_ready`` semaphore **in the sender's on-chip SRAM**, to increment it by 1.
This is shown in Figure 4(b).
Note that this increment **does** require a NoC transaction, since we wish to update the ``receivers_ready``
semaphore is in the sender's on-chip SRAM. These transactions are **unicast transactions**:
each receiver core sends an independent increment transaction to the sender core, so order
of increments is not guaranteed. However, incrementing a semaphore is an atomic operation, so
the sender will eventually see the correct number of receivers ready for the tile.
Of course, the sender core will not send the tile to receivers until all receivers have indicated they are ready.
Having indicated its readiness to receive a tile, each receiver core then waits for the sender to multicast
the tile to it. This is done by waiting on the ``tile_sent`` semaphore **in the receiver's on-chip SRAM**.
This wait operation also does not involve any NoC traffic since the semaphore is in the local on-chip SRAM.

Once the sender core has seen the correct number of receivers ready for the tile,
it can **multicast** the tile to all receiver cores in one operation, using the ``noc_async_write_multicast``
function. This is illustrated in Figure 4(c).
The sender core also resets the ``receivers_ready`` semaphore to ``0`` to avoid accidental reuse of
the same semaphore value, and in preparation for the next tile.
Since the ``receivers_ready`` semaphore is in the sender's on-chip SRAM, this does not require any NoC traffic.

Having sent the tile to all receiver cores, the sender core must signal to the receivers that the tile has been sent.
This is done by calling ``noc_semaphore_set_multicast`` on the ``tile_sent`` semaphore **in the receiver's on-chip SRAM**,
to set it to ``VALID``. This is illustrated in Figure 4(d).
Since we wish to update the ``tile_sent`` semaphore on **all receiver cores**, this requires a NoC **multicast transaction**.

Finally, once a receiver core observes that its ``tile_sent`` semaphore has been set to ``VALID``,
it can proceed to consume the tile. Once the tile has been consumed, the receiver core calls ``noc_semaphore_set``
on the ``tile_sent`` semaphore in its own on-chip SRAM, to set it to ``INVALID`` to prepare for the next tile.
Similar to the earlier ``noc_semaphore_set`` call, this does **not** require any NoC traffic since the
semaphore is in the local on-chip SRAM. This is illustrated in Figure 4(e).
As can be seen, the state of all the semaphores is now the same as at the beginning of the protocol,
ready for the next tile to be multicast.

This high-level protocol description is helpful to understand the overall flow of the multicast operation.
In the following sections, we describe details of TT-Metalium APIs used to implement theese operations.


Overview of Provided Files
==========================

The example multicast program located in ``ttnn/examples/lab_multicast/`` contains the following files:

* Host program:

  * ``lab_multicast.cpp`` - Creates kernels, CBs, and semaphores on appropriate cores and launches kernel execution.

* Dataflow kernels:

  * ``kernels/dataflow/mcast_sender.cpp`` - Reads tiles from DRAM and multicasts them to receiver cores.
    This kernel runs only on the sender core.

  * ``kernels/dataflow/mcast_receiver.cpp`` - Receives tiles via multicast into its input CB.
    This kernel runs only on the receiver cores.

  * ``kernels/dataflow/write_tiles.cpp`` - Writes tiles to DRAM at this receiver's region of the output tensor.
    This kernel runs only on the receiver cores.

* Compute kernel:

  * ``kernels/compute/tiles_copy.cpp`` - Copies tiles from input CB to output CB.
    This kernel runs only on the receiver cores.


Logical vs. Device Coordinates
==============================

The example multicast program uses **device coordinates**, a new concept not previously seen in Labs 1 and 2.
So far, we have been using logical coordinates to describe how you want to assign work to cores in a program.
Logical coordinates assume that the physical layout of Tensix cores is a contiguous grid of compute cores.
However, a typical Tensix device also contains multiple DRAM controllers, multiple Ethernet cores and a PCIe interface.
Since NoC interconnects all these components, it needs a coordinate system that includes all components, not just compute cores.

Tenstorrent architecture actually defines more than two different coordinate systems, but
for the purpose of TT-Metalium programming for this lab, we only need to consider logical and device coordinates.
Note that device coordinates are also referred to as *virtual coordinates* in the Tenstorrent architecture documentation.

The host code always uses logical coordinates (e.g. when creating kernels and CBs), and the compiler takes care of converting
them to device coordinates when needed, making the program easier to write and understand.
However, to maximize performance, we want to avoid performing such coordinate conversions in device kernels.
Therefore, device kernels performing NoC operations must use device coordinates when performing NoC operations.
To facilitate this, TT-Metalium provides the ``worker_core_from_logical_core`` function that is called on the host to
convert logical coordinates to device coordinates before passing them to the device kernels as either compile-time or runtime arguments.
For example, to convert the logical coordinates of the sender core to device coordinates, you can use the following code:

.. code-block:: cpp

   CoreCoord sender_core_device =
       mesh_device->worker_core_from_logical_core(sender_core_logical);

This conversion allows TT-Metalium programmers to write host code using device independent logical coordinates, while still
supplying correct NoC addresses to the kernels, which use device coordinates internally when performing NoC operations.
Remember that host code must pass logical coordinates to all host APIs, such as ``CreateKernel``, ``CreateCircularBuffer``,
``SetRuntimeArgs``, etc.
On the other hand, device kernels must use device coordinates when calling NoC APIs that address other cores (e.g., multicast).


Data Movement Processors and NoC Selection
==========================================

When creating kernels using ``CreateKernel`` in Labs 1 and 2, we always assigned reader kernels to
``DataMovementProcessor::RISCV_0`` and writer kernels to ``DataMovementProcessor::RISCV_1``.
We also set the ``noc`` field of ``DataMovementConfig`` to its default value for the corresponding RISC-V processor,
without discussing it in detail.
On Tensix cores there are two NoC ports, often referred to as NOC0 and NOC1. The default mapping used by TT-Metalium assigns
``DataMovementProcessor::RISCV_0`` to NOC0 and ``DataMovementProcessor::RISCV_1`` to NOC1.
As a result, our simple choice of processor index had an implicit effect: all reader kernels (on ``RISCV_0``) used NOC0,
and all writer kernels (on ``RISCV_1``) used NOC1. We will continue to use this pattern in Lab 3, so that all reader kernels
use NOC0 and all writer kernels use NOC1.

This default assignment is convenient and works well for many examples, but it is not always optimal. If there is significantly
more NoC traffic on readers than on writers (or vice versa), it may be beneficial to rebalance which kernels use NOC0 vs. NOC1,
or to route specific high-traffic kernels through a particular NoC. TT-Metalium allows more complex assignments by explicitly
setting the ``noc`` field in ``DataMovementConfig``, but exploring alternative NoC mappings is beyond the scope of this lab.


Receiver Kernel Overview
========================

The multicast receiver kernel plays a role that is analogous to a reader kernel that reads tiles from DRAM
into a circular buffer. In a DRAM reader kernel, the basic pattern is:

* Reserve space in the CB with ``cb_reserve_back``.
* Initiate an asynchronous DRAM read into the CB write pointer.
* Wait for the read to complete.
* Mark the tile as available with ``cb_push_back``.

The multicast receiver kernel follows the same CB protocol, but instead of initiating a DRAM read,
it relies on the sender to write the tile into its on-chip SRAM via the NoC. For each tile, the receiver:

#. Calls ``cb_reserve_back`` to reserve space in the input CB for the incoming tile.
   This ensures that the CB has a free slot at the write pointer.
#. Resets its local ``tile_sent`` semaphore to ``INVALID`` using ``noc_semaphore_set``.
   This clears any previous state so the kernel can reliably detect when the next tile arrives.
#. Signals to the sender that it is ready for the next tile by incrementing the sender's
   ``receivers_ready`` semaphore by calling ``noc_semaphore_inc``.
#. Waits for the sender to multicast the tile and then mark it as valid by calling
   ``noc_semaphore_wait(tile_sent_sem_ptr, VALID)``. This blocks until the sender has both issued the
   multicast and updated the ``tile_sent`` semaphore.
#. At that point, the tile has been written into the reserved CB slot in on-chip SRAM by the sender's NoC multicast
   operation. The receiver then calls ``cb_push_back`` to mark this tile as available to downstream
   compute kernels, exactly as if it had been read from DRAM locally.


Semaphores: Local vs. Remote Access
-----------------------------------

This kernel also illustrates two ways semaphores are referenced depending on whether they are local or remote.

As discussed earlier, the host passes a semaphore ID (integer) as a kernel argument.
Inside the kernel, ``get_semaphore()`` converts this semaphore ID into a concrete on-chip SRAM address
on the current core, represented by a ``uint32_t`` value.
The next step depends on whether we wish to access the semaphore in local memory or the sempahore on another core.

* To access a **local semaphore** (such as the ``tile_sent`` semaphore on the receiver core), the kernel casts
  this on-chip SRAM address to a pointer:

  .. code-block:: cpp

     volatile tt_l1_ptr uint32_t* tile_sent_sem_ptr =
         reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_sent_semaphore_addr);

  The kernel can now use ``tile_sent_sem_ptr`` with local semaphore APIs such as
  ``noc_semaphore_set`` and ``noc_semaphore_wait`` to manipulate and observe the semaphore value on this core.
  The ``tt_l1_ptr`` qualifier tells the compiler that this pointer refers to a specific section of on-chip SRAM memory.
  This qualifier does not change program semantics, but it enables better compiler optimizations.

* To access a **remote semaphore** (such as the sender's ``receivers_ready`` semaphore), the semaphore doens't need
  to cast the address into a pointer. Rather, it is used to compute the NoC address that points to the remote core's on-chip SRAM.
  To compute the NoC address, the kernel calls ``get_noc_addr``. For example:

  .. code-block:: cpp

     uint64_t receivers_ready_sem_noc_addr =
         get_noc_addr(sender_x, sender_y, receivers_ready_semaphore_addr);

  Here, ``sender_x`` and ``sender_y`` identify the sender core in device coordinates, and
  ``receivers_ready_semaphore_addr`` is the on-chip SRAM address of the semaphore obtained from ``get_semaphore()``.
  It may seem counter-intuitive to use the local semaphore address to compute the NoC address of a remote semaphore.
  This is possible because ``CreateSemaphore`` guarantees that the same semaphore
  ID will always map to the same local on-chip SRAM address on all cores created by one ``CreateSemaphore`` call.
  Therefore, the receiver core can use the local semaphore address, knowing that the same address is used by all cores.
  This convention avoids the need for different cores to pass their local addresses to each other.
  This is the main reason why both semaphores are created on all cores; although the receiver kernel never reads or writes
  its local ``receivers_ready`` semaphore, it needs it to determine its on-chip SRAM address.
  It is worth noting that the overhead of creating a semaphore is minimal.

  ``get_noc_addr`` combines the sender's core coordinates with the semaphore's on-chip SRAM address
  to produce a 64-bit NoC address, which can be used to directly access the same semaphore in a remote core.
  In our example program, the receiver then uses this NoC address with APIs such as ``noc_semaphore_inc`` to update
  the sender's semaphore over the NoC.


Sender Kernel Overview
======================

The multicast sender kernel builds directly on the standard reader pattern you saw in earlier labs.
A regular reader kernel reserves space in a circular buffer (CB), reads a tile from device DRAM into
the CB using an asynchronous NoC read, waits for the read to complete, and then calls ``cb_push_back``
to mark the tile as present in the CB.
The main difference from a standard reader is that the sender kernel in the multicast example program
does not feed a local compute kernel directly; instead, it uses NoC multicast to send the tile to multiple
remote cores.

After loading a tile from DRAM and pushing it into the CB, the sender waits until all receivers have
indicated that they are ready to receive the next tile. It does this by calling ``noc_semaphore_wait``
on its local ``receivers_ready`` semaphore.
Once the semaphore reaches the expected value, indicating that all receivers are ready,
the sender resets it to zero with ``noc_semaphore_set`` so it can later be used for the next tile.

The multicast operation is performed by calling ``noc_async_write_multicast``, which generally requires
the following:

#. The memory address where the source data is located in local memory.
#. The NoC address of the memory in the destination cores.
   To make the process efficient, multicast implicitly assumes that destination memory addresses are
   the same on all destination cores.
#. The number of bytes of data to be multicast. In our example program, this is the number of bytes in a tile.
#. The number of destination cores.
   While this value could technically be decoded from the NoC destination address, it is more efficient to pass it as an argument since this value is known to the code issuing the ``noc_async_write_multicast`` command.
   it is more efficient to pass it as an argument since this value is known to the code issuing the
   ``noc_async_write_multicast`` command.

The memory address of source data is simply the CB read pointer obtained by calling ``get_read_ptr``
after calling ``cb_wait_front``, since the tile has just been pushed into the CB.
The NoC address of the destination memory is more complex to understand, so we discuss it in more detail below.


CB Address Synchronization
--------------------------

To multicast data, the NoC needs to know **where** in each receiver's on-chip SRAM to write the data and what
cores to multicast the data to.
Both types of information are encoded into one 64-bit value by the ``get_noc_multicast_addr`` function:

.. code-block:: cpp

   uint64_t mcast_addr = get_noc_multicast_addr(
       uint32_t noc_x_start,
       uint32_t noc_y_start,
       uint32_t noc_x_end,
       uint32_t noc_y_end,
       uint32_t dest_mem_addr);

The first four arguments specify the coordinates of the opposite corners of a rectangle of cores, which are
the destination for the multicast. To use ``noc_async_write_multicast`` or ``noc_semaphore_set_multicast``,
the destination must be a rectangular grid of cores.

The ``dest_mem_addr`` argument is an **on-chip SRAM address in the destination cores** where the data will be written.
As noted above, it is assumed that all destination cores use the same on-chip SRAM address to receive data.
This is possible because ``CreateCircularBuffer`` guarantees that the same CB index
will always map to the same local on-chip SRAM addresses on all cores created by one ``CreateCircularBuffer`` call.
While the range of addresses is guaranteed to be the same, CBs often have room for multiple tiles, so the
read and write pointers change as tiles are pushed and popped from the CB.
Therefore, all receiver cores must synchronize their CB push and pop operations so that their read and write pointers always point to the
same on-chip SRAM address when receiving a tile through multicast.
Furthermore, if the sender also synchronizes its own CB push and pop operations with receivers, its own read and write pointers
will be in sync with the receivers' pointers.
This approach avoids the need for receiver cores to pass their local addresses to the sender; the sender can simply use its own CB
read and write pointers as the destination address in ``get_noc_multicast_addr`` because they are guaranteed to be in sync.
This can be seen in the example multicast program, where the sender uses its own ``cb_read_addr`` in a call to
``get_noc_multicast_addr``.


Multicast Operation
-------------------

Once the sender kernel has obtained encoded destination address via ``get_noc_multicast_addr``,
it issues ``noc_async_write_multicast`` to send the data to all receivers using the NoC.
``noc_async_write_multicast`` is a **non blocking** operation. It enqueues a multicast transfer on the NoC,
then returns control to the kernel immediately, while the hardware performs the tile transfer in the background.

After issuing the tile multicast operation, the sender needs to inform receivers that the tile has been sent and is valid.
To do this, it performs another multicast operation to update the receiver's ``tile_sent`` semaphore to
``VALID`` by calling ``noc_semaphore_set_multicast``.
On some architectures, NoC operations may be issued to separate command buffer FIFOs and may not be
issued in the order they are called. To ensure the commands are issued in the program order,
the sender calls ``noc_async_writes_flushed()`` before calling ``noc_semaphore_set_multicast``
This ensures that the tile multicast command has been sent into the NoC before the ``noc_semaphore_set_multicast``
command that sets ``tile_sent``.

While this ensures commands are issued in the program order, for multicast protocol correctness it is also necessary
that the commands be completed in the order they were issued. The default NoC configuration used in this lab ensures
the NOC operations issued from one core will be completed in the order they were issued.

Finally, the sender calls ``noc_async_write_barrier()`` to wait until the multicast data transfer completes before reusing the CB slot.
After this barrier the sender calls ``cb_pop_front`` to free the CB entry for the next tile..
This preserves the usual CB producer-consumer protocol by ensuring that multicast data has been sent before any
tile data is overwritten.

Another thing worth noting is that ``noc_semaphore_set_multicast`` takes a pointer to a value to be multicast.
The NOC hardware does a 4-byte read from that address and multicasts those 4 bytes to the receiver cores.
We could pass a pointer to any memory location that holds ``VALID`` (e.g. a pointer to a ``uint32_t``) and
pass that as the source. Instead of using an arbitrary value, we use the ``tile_sent`` semaphore, which is
already allocated and otherwise unused on the sender. This avoids the need for an additional variable and
makes the code more resilient to any future changes to sempahore's internal representation (e.g., if in the
future semaphores hold more than 4 bytes).

It is worth noting that NoC supports other more complex modes of operation, where the order of completion of commands may not match
the order of their issuance. In such cases, it may be necessary to add an additional ``noc_async_write_barrier()`` after the tile multicast
data to ensure that the semaphore set command is issued only after the data transfer completes.


Compute and Writer Kernels
==========================

Compute and writer kernels are similar to the ones used in Labs 1 and 2.
Because they use CBs for their data, they don't need to know whether the data was received via multicast or DRAM read.


Multicast and Double Buffering
==============================

In the multicast example program, multicast is combined with **double buffering** in the CBs.
On each receiver, double buffering allows overlapping:

* Receiving a tile via multicast into its input CB.
* Computing on previously received tiles.

Double buffering still works with multicast, as long as:

* You do not reuse a CB slot until all NoC operations that read or write the memory occupied by the slot have completed.
* You maintain a consistent pattern of ``cb_reserve_back``, ``cb_push_back``, ``cb_wait_front``,
  and ``cb_pop_front`` across sender and receivers, so that senders and receivers consistently use
  the same memory addresses for the same tiles.


Multicast Exclusions
====================

When calling ``noc_async_write_multicast`` or ``noc_semaphore_set_multicast``, the core initiating these operations
is excluded from the multicast operation by default.
This means that the number of destination cores and the destination NoC address passed to these function should not
include the core initiating the operation.
While we will not require it for this lab, it is worth noting that separate functions do exist that include the core
initiating the operation in the multicast operation. These functions are ``noc_async_write_multicast_loopback_src`` and
``noc_semaphore_set_multicast_loopback_src``.


Debugging Hangs with Watcher
****************************

Given that multicast is a relatively complex operation, it is possible to introduce bugs that are difficult to debug.
For example, forgetting to update a semaphore, updating semaphores at wrong points in the code, or passing incorrect
coordinates to NoC APIs can lead to program hanging indefinitely.
while such issues can be debugged using debug features introduced in Lab 1, there is another tool that is particularly
useful for debugging Noc Issues and Hangs.

The **Watcher** tool in TT-Metalium is a debug facility that instruments firmware and kernels and runs a
host-side monitoring thread to catch common programming errors and hangs.
On a fatal error, Watcher stops the program and reports a clear message.
On a hang, the log shows which kernels and cores were active at the time of the hang.

Watcher can be enabled by setting an environment variable before running your program::

    # Enable Watcher with a 10 second polling interval
    export TT_METAL_WATCHER=10

The numeric value is the interval, in seconds, between Watcher status dumps. Small values like 1 give very frequent snapshots and
are convenient while debugging a hang, but introduce a significant performance overhead.
Larger values like 10 or 60 are less intrusive and are better starting point when doing initial debugging.

When enabled, Watcher will print messages such as "Watcher checking device 0" to the terminal and write a log file
to ``generated/watcher/watcher.log``, which summarizes the kernel IDs that were running, as well as the
last **waypoint** string hit on each RISC-V. Waypoints are short markers (up to 4 characters long) that can be inserted into kernel code to tag key
positions like "entered main loop" or "finished writing". Various TT-Metalium APIs already encode waypoints into their code.
For example, if you examine the code for ``noc_semaphore_wait`` in ``tt_metal/hw/inc/api/dataflow/dataflow_api.h``,
you can observe that it encodes the waypoint "NSW" (for "NoC Semaphore Wait") before waiting on a semaphore and "NSD" (for "NoC Semaphore Done") after.
You can also add your own waypoints to the code to tag key positions simply by adding the ``#include "api/debug/waypoint.h"``
nad then using the ``WAYPOINT`` macro at desired points in the code.

.. code-block:: cpp

   #include "api/debug/waypoint.h"

   void kernel_main() {
      WAYPOINT("MYWY");
   }

Ensure that you use unique waypoint strings for each key position in the code, otherwise the watcher output may be misleading.

Since Watcher adds extra checking and bookkeeping code, it adds a performance and code-size cost.
Therefore Watcher should be disabled when doing performance benchmarking or production runs.
In some cases, Watcher can significantly prolong program execution time, making a valid program run appear to hang.
Therefore, for the purpose of thiese labs, it is best to disable Watcher by default and only enable it when debugging.

For more information about the Watcher, refer to Additional resources section at the end of this lab.


Exercise 1: Debugging Multicast Issues Using Watcher
====================================================

In this exercise, you will intentionally introduce errors into the multicast sender and receiver kernels
and use the Tenstorrent watcher and DPRINT to help diagnose the problem.
This is intended to give you hands-on experience with debugging tools for distributed NoC and semaphore issues
in a controlled environment.

Perform the following steps to complete the exercise:

#. If you haven't already done so, from the root of the ``tt-metal`` repository, run the build script ``./build_metal.sh``.

#. Run the multicast example (``./build/ttnn/examples/example_lab_multicast``) and
   verify that it completes successfully and prints a "Test Passed" message on the host.

#. Next, introduce an error in the multicast sender kernel by modifying the destination core range.
   Open ``ttnn/examples/lab_multicast/kernels/dataflow/mcast_sender.cpp`` and find the line where the sender
   precomputes the multicast address for the ``tile_sent`` semaphore:

   .. code-block:: cpp

      uint64_t tile_sent_mcast_addr = get_noc_multicast_addr(
          receiver_start_x, receiver_start_y, receiver_end_x, receiver_end_y, tile_sent_semaphore_addr);

   Change ``receiver_start_x`` to a constant that is outside the valid core coordinate range on your device, such as ``100``.
   This makes the sender attempt to multicast to a non-existent core along the ``x`` dimension.
   Because this is a change in kernel code only, you do not need to rebuild the program;
   the updated kernel will be JIT-compiled the next time the program is run.

#. Run the multicast example again:
   You should see that the program now hangs, running indefinitely without printing a final result or explicit error.
   This kind of hang is typical for incorrect NoC addressing or synchronization errors.

#. Terminate the program (using ``Ctrl + C``) and execute the ``tt-smi -r`` command from command line to reset the device.
   It is always a good idea to reset the device after a hang to ensure that the device is in a known good state.

#. Rerun the program with Watcher enabled with a period of 10 seconds:

   .. code-block:: bash

      TT_METAL_WATCHER=10 ./build/ttnn/examples/example_lab_multicast

   Watcher will periodically inspect the device state. After some time it should detect that the program is not
   making progress and report an error. the error should indicate the logical (e.g. ``core(x= 0,y= 0)``) and
   device (e.g. ``virtual(x= 1,y= 2)``) coordinates of the core that caused the erroneous NoC operation,
   along with a message indicating the type of error. Note that the exact error messages may vary depending on the
   type of error.

#. Revert the sender change by putting ``receiver_start_x`` back into the ``get_noc_multicast_addr`` call
   in ``mcast_sender.cpp``. Reset the device using ``tt-smi -r``, then rerun the multicast example
   with Watcher disabled and confirm that it completes successfully without hanging.

#. Next, introduce a synchronization bug in the receiver kernel by removing a key semaphore update.
   Open ``ttnn/examples/lab_multicast/kernels/dataflow/mcast_receiver.cpp`` and comment the ``noc_semaphore_inc``
   line that signals the sender that this receiver is ready for the next tile.
   Rerun the multicast example once more with Watcher disabled.
   The sender kernel will hang waiting on ``receivers_ready`` semaphore, because receivers no longer
   increment that semaphore.

#. Terminate the program, reset the device using ``tt-smi -r``, then rerun the multicast example, this time
   with Watcher **enabled** with a period of 10 seconds.

   .. code-block:: bash

      TT_METAL_WATCHER=10 ./build/ttnn/examples/example_lab_multicast

   Once program starts, watcher should activate once every 10 seconds and log the state of the device into
   a log file. After several watcher messages, terminate the program (using ``Ctrl + C``) and inspect
   the log file in ``generated/watcher/watcher.log``.

   First, review the legend in the log file to understand the meaning of the subsequent lines.
   Key takeaways:

   * Each Tensix core has 5 RISC-V processors: BRISC, NCRISC, TRISC0, TRISC1, TRISC2, corresponding to
     RISC-V 0 through RISC-V 4 in Tensix Core figure in Lab 1.
     BRISC is considered to be the primary processor, and the other RISC-V processors are considered
     to be subordinate processors.
   * State of each RISC-V processor is indicated either through a single-character code
     (e.g., ``W`` = Waiting, ``R`` = Running, ``D`` = Done), or through a multi-character code,
     (e.g., ``NRW`` = "NOC Read Wait", ``NSW`` = "NOC Semaphore Wait").
   * ``smsg`` shows if subordinate processors are in Go (G) or Done (D) state.
   * ``k_ids`` maps to kernel source files listed at the end of each dump section

   Next, examine a single line of the log file to understand how to interpret the information.
   Consider this example line:

   .. code-block:: text

      Device 0 worker core(x= 1,y= 0) virtual(x= 2,y= 2):  NSW,CWFW,   K,MWDD,   K  rmsg:D0G|BNT h_id:  0 smsg:GGGG k_ids:  5|  6|  7|  7|  7

   Breaking this apart:

   +--------------------------+-------------------------+----------------------------------------------------------+
   | Field                    | Value                   | Meaning                                                  |
   +--------------------------+-------------------------+----------------------------------------------------------+
   | `core(x= 1,y= 0)`        | Logical coords          | This is logical core (1,0)                               |
   +--------------------------+-------------------------+----------------------------------------------------------+
   | `virtual(x= 2,y= 2)`     | Device/Virtual coords   | Used for NOC addressing                                  |
   +--------------------------+-------------------------+----------------------------------------------------------+
   | `NSW`                    | BRISC status            | **N**OC **S**emaphore **W**ait                           |
   +--------------------------+-------------------------+----------------------------------------------------------+
   | `CWFW`                   | NCRISC status           | **C**B **W**ait **F**or **W**rite                        |
   +--------------------------+-------------------------+----------------------------------------------------------+
   | `K`                      | TRISC0 status           | In **K**ernel                                            |
   +--------------------------+-------------------------+----------------------------------------------------------+
   | `MWDD`                   | TRISC1 status           | **M**ath **W**ait **D**ata **D**ependency                |
   +--------------------------+-------------------------+----------------------------------------------------------+
   | `K`                      | TRISC2 status           | In **K**ernel                                            |
   +--------------------------+-------------------------+----------------------------------------------------------+
   | `rmsg:D0G\|BNT`          | Run message             | Dispatch, NOC 0, Go state; BRISC/NCRISC/TRISC enabled    |
   +--------------------------+-------------------------+----------------------------------------------------------+
   | `smsg:GGGG`              | Subordinate message     | NCRISC, TRISC0, TRISC1, TRISC2 in **G**o state (running) |
   +--------------------------+-------------------------+----------------------------------------------------------+
   | `k_ids: 5\|6\|7\|7\|7`   | Kernel IDs              | BRISC=5, NCRISC=6, TRISC0/1/2=7                          |
   +--------------------------+-------------------------+----------------------------------------------------------+

   Kernels are identified through their IDs, and mapping between kernel IDs and source file names is listed
   at the end of each dump section.
   Idle cores where the program hasn't created any kernels can easily be identified by their ``k_ids`` fields all set to 0.

   To help us identify the source of the hang, we observe the **first column of the status** (BRISC status) for the
   cores running our kernels and observe that they are all stuck at ``NSW`` (**N**OC **S**emaphore **W**ait).
   Of course, we need to verify that this is actually a hang and not just a slow operation, which we can do by observing
   that the program state in multiple dumps doesn't change.

   With simple bugs like the one we introduced, this information may be sufficient to diagnose the problem.
   However, in more complex cases, we may need to add additional instrumentation to the kernels to help us diagnose the problem.
   This could be either by adding additional waypoints, or by adding DPRINT statements to the code.

#. Revert the receiver change by uncommenting the ``noc_semaphore_inc`` line in ``mcast_receiver.cpp``.
   Reset the device using ``tt-smi -r``, then rerun the multicast example
   with Watcher disabled and confirm that it completes successfully without hanging.

In this exercise you have intentionally introduced NoC issues and then used Watcher to analyze the resulting behavior.
Taken together, Watcher, waypoints, and DPRINTs provide a powerful set of tools for debugging NoC-related bugs
in TT-Metalium multicast and multi-core programs.


Exercise 2: Extending the Standalone Multicast Example
******************************************************

You may have noticed that the sender core in the multicast example program doesn't specify any compute or writer kernels.
While this is acceptable, it is not the most efficient use of the sender core resources as most of the core is idle.
In a real application, the sender core would also perform computation and writeback.
In this exercise you will extend the example program so that the sender core also participates in the same computation
as the receiver cores.

Perform the following steps to complete the exercise:

#. Start by copying the files from the ``lab_multicast`` directory into a new directory (e.g. ``lab3_exercise2``),
   and rename the copied ``lab_eltwise_binary.cpp`` file to match the directory name (e.g. ``lab1_matmul.cpp``).

#. Update all ``CreateKernel`` calls to point to kernel source files in the new directory.

#. Update ``CMakeLists.txt`` files in the new directory and in the parent directory to include the new executable,
   then build and run the new program to confirm that it works as expected.

#. Update the host program to include the sender core in the core range when creating the compute and writer kernels.
   Don't forget to also pass runtime arguments for all cores where the kernels are created, including the sender core.
   Observe that the compute and writer kernels themselves do not need to change at all.

#. Update ``output_data`` and related variables to account for the additional copy from the sender core.
   After the change, output of the program should be a tensor that contains four copies of the input tensor;
   one from the sender core and three from the receiver cores.
   Make sure that each core writes to a unique region of the output tensor.

#. Update the ``mcast_sender`` kernel code.
   The sender acting as a "local receiver" for compute does **not** require introduction of
   any additional semaphores. This is because the sender already knows when a tile is in its
   local CB immediately after the DRAM read completes. However, there is one change that needs
   to be made to the sender kernel: it should not perform any ``cb_wait_front`` or ``cb_pop_front``
   operations, because the compute kernel will be doing this work.
   This also means that the sender kernel should not call ``get_read_ptr``, since the read pointer
   is valid only between ``cb_wait_front`` and ``cb_pop_front`` calls.
   Instead, the source address for multicast should be the same address that was used for
   writing the tile to the CB, and multicast should be performed **after** the data has been
   read from DRAM (i.e., after the first ``noc_async_read_barrier``).
   Similarly, the CB write address can be used to determine destination address for multicast,
   because all receiver cores use the same CB write address.

#. Ensure that result verification code on the host now also verifies the sender's output.

#. Build and run your program and verify that it completes successfully.
   Make sure that the output indicates correct number of receiver cores and output tiles.

In case you encounter any hangs, don't forget to use the ``tt-smi -r`` command to reset the device
before running the program again.

In this exercise you have extended the multicast example program to include the sender core in the computation.
This is a common pattern in real applications, where we wish to maximize the utilization of the sender core.





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

Exercise 4: Multi Core Matrix Multiplication with Multicast and Slabs
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
       * For each tile, uses the row-specific semaphores and NoC multicast APIs to:

         * Wait until all cores in the row are ready,
         * Multicast the tile to all cores in that row,
         * Signal that the tile has been sent,
         * Use NoC barriers before reusing CB slots for the next K-block.

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
     * For each tile, multicast down the column using column-specific semaphores and NoC multicast.

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
   Be particularly careful about the ``SetRuntimeArgs`` function, which is a host function that takes logical coordinates to specify
   which cores will receive the runtime arguments, while any runtime arguments that refer to coordinates must themselves use device coordinates.
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

Additional information about Tenstorrent NoC can be found in the following resources:

* NoC (Network on Chip) Readme: https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/NoC/README.md
* Networks and Communication Lesson: https://github.com/tenstorrent/tt-vscode-toolkit/blob/main/content/lessons/cs-fundamentals-04-networks.md
* Introduction to Data Movement in TT-Metal: https://github.com/tenstorrent/tt-low-level-documentation/blob/main/data_movement_doc/general/intro_to_dm.md
* Watcher Documentation: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/watcher.html
