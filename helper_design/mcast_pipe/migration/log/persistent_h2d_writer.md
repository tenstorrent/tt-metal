# persistent_h2d_writer.cpp — DEFERRED (design-gap + coverage-gap)

Kernel: `models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/persistent_h2d_writer.cpp`

Emitter: `ttnn/core/services/h2d_socket_service.cpp`

Tier: 2.16b. Status: deferred. No production code change.

## API-v11 address-model gap

After the backing-tensor DRAM writes complete, the persistent service core increments
`data_ready_sem_addr` on the worker rectangle. The emitter obtains this address from
`ttnn::global_semaphore::create_global_semaphore(...)` and passes the resulting worker-grid L1
address to the kernel at runtime. It is not a semaphore ID allocated in the persistent service-core
program.

API-v11 `Semaphore<>` binds a program semaphore ID through `get_semaphore(id)`, and `SenderPipe`
signals that owned local address. It has no public constructor or send verb for an arbitrary runtime
GlobalSemaphore L1 target. Replacing the target with a service-program semaphore would change which
worker address consumers poll and would not bind the existing GlobalSemaphore object.

The optional metadata multicast is ordered before this worker-ready event, while the completion PCIe
write and service-core consumed counter are separate synchronization and remain outside the pipe. In
particular, an `async_write_barrier()` and the PCIe completion publication occur between metadata and
the worker-ready Counter. `SenderPipe::send()` fuses one data payload directly to its signal, so it
cannot preserve that separation even if the address-binding gap were solved. Migrating metadata alone
would leave the defining worker-ready multicast raw and would not complete the atomic unit.

## Generality and coverage gates

The D2H twin has the same address-binding need, but both directions belong to one persistent host-I/O
socket subsystem. They are not two unrelated production families, so they do not satisfy the plan's
generality gate for an arbitrary-external-semaphore-address extension.

The Python H2D service module and the C++ worker-sync tests require a Blackhole Galaxy/UBB
configuration. They skip on the current single-chip Blackhole P100a, so the exact worker-sync route
cannot be correctness- or performance-validated on this machine. This is a coverage gap in addition
to the binding design gap.

## Claude consultation

Claude independently returned DEFER — DESIGN-GAP + COVERAGE-GAP. It confirmed that neither service
program creates worker-grid program semaphores and identified the data/signal ordering separation as
a second H2D blocker. It also confirmed that the H2D/D2H twins fail the unrelated-family generality
gate.

## Verdict

DEFER — DESIGN-GAP + COVERAGE-GAP. Keep the GlobalSemaphore multicast primitive raw and do not expand
the helper API.
