# persistent_d2h_reader.cpp — DEFERRED (design-gap + coverage-gap)

Kernel: `models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/persistent_d2h_reader.cpp`

Emitter: `ttnn/core/services/d2h_socket_service.cpp`

Tier: 2.16c. Status: deferred. No production code change.

## API-v11 address-model gap

The persistent service core sends a signal-only Counter increment to `transfer_done_sem_addr` on the
worker rectangle, then preserves an atomic barrier before waiting on its separate local write-ack
counter. The emitter obtains `transfer_done_sem_addr` from
`ttnn::global_semaphore::create_global_semaphore(...)`; it is a worker-grid L1 address, not a program
semaphore ID owned by the persistent reader program.

API-v11 `Semaphore<>` resolves a program semaphore ID with `get_semaphore(id)`. `SenderPipe` therefore
cannot target the runtime GlobalSemaphore address. Substituting a program-local semaphore would signal
a different L1 address from the one exposed to workers and would change the service contract. The
socket writer, DRAM reads, atomic completion barrier, and operation-owned write-ack counter remain
outside the multicast pipe.

Apart from address ownership, this protocol is an exact API-v11 fit: a no-pre-handshake Counter
`send_signal()` performs an EXCLUDE-source +1 multicast and the required atomic barrier. This makes
the address binding the sole helper capability gap for D2H.

## Generality and coverage gates

H2D has the same address-binding need, but the two directions are twins within one persistent
host-I/O socket subsystem, not unrelated production families. They do not pass the plan's generality
gate for a new arbitrary-address semaphore face.

The Python D2H service module and C++ worker-sync tests require a Blackhole Galaxy/UBB configuration
(with an additional documented D2H LLK skip). They skip on the current single-chip Blackhole P100a,
so this route cannot satisfy the plan's correctness or matched-performance gates here.

## Claude consultation

Claude independently returned DEFER — DESIGN-GAP + COVERAGE-GAP. It confirmed that the target crosses
program boundaries and cannot be replaced with a program semaphore, that the API-v11 control protocol
otherwise matches exactly, and that the H2D/D2H twins fail the unrelated-family generality gate.

## Verdict

DEFER — DESIGN-GAP + COVERAGE-GAP. Keep the GlobalSemaphore multicast primitive raw and do not expand
the helper API.
