# Direct host push: fillers own D2H sockets, movers deleted

Status: IMPLEMENTED (`wip/direct-push`), then made faster than the ring design it replaced by
gather-READ packing -- the filler reads each live run straight to its packed wire offset in staging and
ships a frame as ONE PCIe write. See FINDINGS §N+71 for the measurements.

## What changes

Today every marker crosses DRAM twice:

    worker L1 -> [filler] -> DRAM ring -> [mover] -> staging -> host

Direct push removes the ring and the mover entirely:

    worker L1 -> [filler] -> staging -> host

Each filler owns a D2H socket. Decode-thread count stays a host-side knob: N sockets
can be serviced by 1..N threads, because the receiver's assignment already strides
(`for (i = t; i < streams; i += nthreads)`). In practice 1-2 threads round-robin over
the sockets.

## Why it is possible on Blackhole

The socket's host FIFO is a plain `mmap` pinned through the IOMMU
(`D2HSocket::init_host_buffer`), reached by a full 64-bit NoC/PCIe address. There is
no window budget and no channel-size cap on this path -- those belong to the Wormhole
hugepage fallback (`init_host_buffer_hugepage`), which BH never takes. So socket count
and FIFO size are both free parameters, and the comment in streaming_profiler.hpp
about `SOCKET_WIN_BASE` / 16 windows / "2 is the ceiling" does not describe BH.

That matters because the DRAM ring exists precisely to provide elasticity the WH host
FIFO could not: "the whole reason to stage in DRAM is that this number is not capped by
the TLB window budget the way the 12 MiB host FIFO is". On BH the FIFO can simply be
made large, so the spike absorption moves from device DRAM to host RAM -- one hop closer
to the consumer.

## Wins

- Deletes a DRAM write and a DRAM read per byte.
- Frees the 2 mover DRISCs to be fillers: 6 fillers, ~17 worker cores each instead of 26.
  Revisit drops ~1.5x for free, which is the budget everything else spends.
- Removes the ring-full back-pressure path (filler blocked on ring room) outright.
- The GDDR-DMA ring-placement dilemma disappears: with no ring there is nothing to place,
  so neither side has to give up its local channel.
- Decode-thread count becomes a configuration choice rather than a consequence of the
  mover count.

## Work

| item | size | notes |
|---|---|---|
| Direct push | medium | move the 8 socket call sites from the mover into the filler; `stage_run` targets `pcie_xy_enc` instead of a DRAM bank; ring-room wait becomes socket credit wait |
| 6 fillers / N sockets | medium | `kNFillers` 4 -> 6; drop `kNFillers % kNSockets == 0` (it exists only because "mover m drains fillers m, m+kNSockets") |
| Size the host FIFO | small | deliberate choice, not the inherited 12 MiB; this is where burst absorption now lives |
| Trim the frame header | small | make the head/tail array width arch-specific (24 is Quasar's); used fields land in the first ~11 words, ship 16 instead of 64 -> ~192 B/frame, no repacking |
| Re-derive the lane trigger | measurement | the single knob controlling frame fullness, wasted worker reads, and PCIe write size |

## Delete

Mover kernel, its GDDR DMA read path, the peer handshake, DRAM ring allocation, and the
ring-room back-pressure path.

## Order, and why it matters

1. Direct push + 6 fillers  (revisit 26 -> ~17 cores per filler)
2. Header trim              (independent)
3. Re-derive the lane trigger

Step 3 must be last. Frame fullness and producer headroom are the same budget: at
trigger=128 a lane keeps 384 words of headroom, at 376 it keeps 136. That headroom can
only be spent once revisit has bought it back -- otherwise it reintroduces the stalls the
eager trigger was chosen to remove ("the eager trigger took six of eight devices to ZERO
stalls").

Note the trade also changes shape: today an under-full frame costs DRAM ring space; after,
it costs host FIFO space and a smaller PCIe write. The old measurement does not transfer.

## Risks to size first

- **Six DRAM cores doing host-facing writes.** streaming_profiler.hpp records that
  "exactly two DRAM cores measure safe to host a drainer (row y == 0)". Establish whether
  that constrains PCIe-facing work or only the mover's former role -- it caps socket count
  if the former.
- **Runway.** Today: 64 MiB ring behind a 12 MiB FIFO, ~115 busy sweeps of slack. After:
  whatever the FIFO is sized to. Choose it deliberately.
- **Credit handling moves into the filler**, which also sweeps worker cores; a filler
  blocked on socket credit is not sweeping, so credit stalls now cost revisit directly.
