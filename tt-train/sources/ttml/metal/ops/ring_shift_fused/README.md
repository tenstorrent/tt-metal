# `ring_shift_fused` -- every tensor of a ring shift in one launch

The ring attention moves K and V (and in the backward dK and dV) one chip
along the ring every step. The first direct transport moved one tensor per
shift in four launches: even chips send, odd chips receive, then the other
way round, because a chip's send program has to wait for its neighbour's
receive program, and with separate send and receive launches on one command
queue a full ring of them deadlocks. This op puts both roles in one program
per chip and every tensor in one launch.

| entry point | what it is for |
|---|---|
| `ttml::metal::ring_shift_fused(inputs, send_sockets, recv_sockets)` | the op: the tensors shifted along the sockets' connections |
| `ttml::ttnn_fixed::distributed::ring_shift_many(tensors, axis, direction, Direct)` | what the ring driver calls; builds the sockets, picks the launch mode |

`ring_shift(tensor, ..., Direct)` is `ring_shift_many` of one tensor.
`RingShiftTransport::DirectTwoPhase` is the first direct transport, kept
as the reference the tests compare against; `Fifo` is the original.

## How it works

The sockets are the two-phase transport's two: even chips to odd and odd
to even, one sender core per fabric link on worker row 0 and the matching
receiver core on row 1 (a chip that is both a sender and a receiver of one
socket hangs the fabric handshake). Per chip, one program: the sender cores
read their share of every tensor's pages from DRAM into a packet buffer
(bank-packed, so one fabric packet writes a run of pages that sit next to
each other in one DRAM bank on both chips) and write them straight into the
neighbour's output tensors through the fabric; the receiver cores answer the
sender's one handshake with every output tensor's address, then wait for
the one completion token. The tensors' DRAM layouts are the same on both
chips, so a page's address on the neighbour is the one it would have here.

**Launch mode, chosen by `ring_shift_many` and logged once.** On a single
ring (the 1x8 loudbox) every chip sends and receives at once: one launch
per shift, every link busy. On a mesh of several rings (the 2x4 loudbox
mesh, two rings of four) that launch loses the handshake's answers in the
fabric and hangs, reproducibly and at any size, while each half of the ring
alone -- even chips sending, or odd chips -- completes and lands the right
bytes; two sockets, one link per chip, or staggering the two connections'
opening change nothing. So there the shift is two launches, even chips
sending and then odd chips, each with all the tensors. That is the
two-phase transport's order and bandwidth with fewer launches. The 1x8 ring
is what the trainer uses.

## What it buys

`LoudboxRingSDPATest.DISABLED_TimeTheShift`, 8 chips, median of 7, the
transport's launches only:

| | two-phase | fused, 1x8 ring | fused, 2x4 mesh |
|---|---|---|---|
| bf16, 4 heads x 4096 rows a chip (2.1 MB) | 193 us | 159 us (-18%) | 164 us |
| fp32, same shape (4.2 MB) | 226 us | 168 us (-26%) | 223 us |
| the backward step's set: K, V bf16 + dK, dV fp32, 10 heads x 5632 rows (43 MB a chip) | 1472 us | 907 us (-38%) | 1605 us (+10%) |

On the single ring the gain is the links' other half. On the 2x4 mesh the
one launch (since 22 September, below) sends on one link per chip and
carries the answers on the other, so the bytes move at the two-launch rate
and it is a wash: the backward set measures 1709 us against 1848 for the
two-phase transport on the same day (-8%), within the day-to-day spread.

### Why the full ring hung on the 2x4 mesh, and the fix (22 September)

A fabric router's worker channel takes one connection: the connection
helper wires every worker to sender channel 0, and the worker adapter's
`open()` has no exclusivity check, so two workers on one router silently
share its slot cursor and credit record, and the handshake's answers are
lost. A receiver answers toward the chip that sends to it. On the 1x8 ring a
chip's two neighbours lie in opposite directions and the direction-based
link choice keeps the roles apart; on a ring of four laid along a mesh row
without wrap, the end chips reach both neighbours through one direction,
and their sender and receiver both took link index 0 of that direction, the
same router. Each half of the ring alone works because no chip then has
both roles. The tt-llk noc-sync audit skill found this from the code; the
fix, in the driver, asks the fabric per chip which routers each role would
use and leaves one link to the receiver where they coincide (the fused op
refuses, naming the cause, if a receiver has no router left). With it the
one-launch mode passes the two `RingShiftFused*` tests on the 2x4 mesh.

In the ring step (`DISABLED_CompareStepTimes`, 1x8 ring, zigzag, the
cyclic forward and backward, median of five) and in training
(`training_shakespeare_nanollama3_cp8_char.yaml`, 40 steps):

| | two-phase shifts | fused shifts |
|---|---|---|
| 20/10 heads, 5632 rows a chip, d 64: forward / backward step | 43.3 / 37.3 ms | 41.6 / 33.0 ms (-4% / -12%) |
| 32/8 heads, 5632 rows a chip, d 128: forward / backward step | 93.1 / 113.4 ms | 90.2 / 106.3 ms (-3% / -6%) |
| 4/4 heads, 4096 rows a chip, d 64: forward / backward step | 12.7 / 12.7 ms | 12.1 / 11.2 ms |
| training step, 20/10 heads, four layers | 462 ms | 392 ms (-15%) |

The losses of the training run are the same at every printed step as
before (2.7676 at step 10, 2.5176 at 20, 2.4922 at 30, 2.4688 at 40). Bit-exact with the two-phase transport and with
Fifo (`RingShiftFusedMatchesTwoPhase`, `RingShiftFusedOneTensor`, both
meshes, both directions, bf16 and Float32).

## Testing

`LoudboxRingSDPATest.RingShiftFused*` (the op), `LoudboxRingSDPATest.CyclicForward*`
and `ZigzagCausalBackward*` (the ring driver through it), with
`TTML_LOUDBOX_RING8=1` for the single ring. A hung fabric kernel wedges the
boards: `tt-smi -r` before the next run.
