# experimental/conv3d/device/kernels/writer.cpp — DEFERRED (design gap)

## Verdict: deferred (helper design gap — NOT migrated, file untouched)

In-scope path: only the `WeightShareMode::Mcast` rectangle-mcast (lines 197-253); the Chain/unicast
forwarding and Disabled paths are explicitly out of scope.

The Mcast mode has three roles:
- **McastSender** (199-246): pre-wait `weights_mcast_sender_sem.wait(mcast_num_dests)` + DRAM read +
  EXCLUDE_SRC data mcast (count `mcast_num_dests`) + flag `set_multicast` (count `mcast_num_dests`),
  linked, no barrier (same VC). Structurally a clean `SenderPipe::send()`.
- **McastReceiver** (247-252) and **McastPassive** (138-148): ack `up(...,1)` + `wait(1)` + `set(0)`,
  exactly `ReceiverPipe::receive()` with PRE_HANDSHAKE=true (VALID=1, INVALID=0 confirmed).

BLOCKER (runtime recipient count — known v7 design gap): `mcast_num_dests` is a **RUNTIME arg**
(line 93, `get_arg_val<uint32_t>(argidx++)`) used as BOTH the mcast dest count and the sender's
ack-wait count. The v7 `SenderPipe<NOC_ID, DATA_READY_SEM_ID, NUM_ACTIVE_RECEIVER_CORES, ...>` takes
`NUM_ACTIVE_RECEIVER_CORES` as a **compile-time template param**. A runtime num_dests cannot be passed.
Inexpressible (the documented runtime-per-rect-recipient-count gap; cf. gn_v2/welford deferrals).

The McastReceiver/McastPassive sides ARE individually expressible as ReceiverPipe (no count needed),
but migrating only the receiver/passive halves while the sender stays raw splits the handshake
protocol across helper + open-code for marginal gain — the SenderPipe emitter, the heart of the block,
is the gap. Defer the whole kernel.

Sem ids (weights_mcast_sender_sem_id / receiver_sem_id) ARE compile-time — not the blocker.

## Action: no edit, ledger status=deferred, flag design-gap.
