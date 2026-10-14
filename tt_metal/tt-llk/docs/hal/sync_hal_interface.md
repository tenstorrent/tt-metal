<!-- Ported verbatim from Confluence: "Synchronization (hal::sync)" (https://tenstorrent.atlassian.net/wiki/spaces/LLK/pages/2708635726/Synchronization+hal+sync), last modified 2026-10-09. -->

# Synchronization interface

`hal::sync` provides a small interface for semaphores, mutexes, and waits.

```cpp
#include "hal/sync.h"

using namespace hal::sync;
```

Use:

- **Semaphores** to track available work and coordinate producers and consumers.
- **Mutexes** to give a Tensix thread exclusive ownership of a shared resource.
- **Waits** to delay selected work until its dependencies are ready.

## Choosing a call form

Use template arguments when the selection is known at compile time:

```cpp
semaphore::post<Semaphore::S1>();
mutex::acquire<Mutex::M2>();
```

Use function arguments when the selection is determined at runtime:

```cpp
semaphore::post(selected_semaphore);
mutex::acquire(selected_mutex);
```

Both forms perform the same operation. Template arguments are checked at compile time. Runtime checks require `ENABLE_LLK_ASSERT`.

## Semaphores

Select a semaphore with `Semaphore::S0` through `Semaphore::S7`.

### Initialize

Set the starting count and the threshold used by full-capacity waits:

```cpp
// Start empty; consider the semaphore full when its count reaches 2.
semaphore::init<0 /* Initial */, 2 /* Maximum */, Semaphore::S1>();

// Runtime form.
semaphore::init(Semaphore::S1, 0, 2);
```

Initialize a semaphore before other threads use it. Both values must be in `[0, 15]`.

`Maximum` is a wait threshold; it does not prevent further posting. Use a `WhileMaximum` wait when the producer must respect that threshold.

### Signal and consume

`post` increments the count. `get` decrements it.

```cpp
semaphore::post<Semaphore::S1>();
semaphore::get<Semaphore::S1>();

// Runtime forms.
semaphore::post(Semaphore::S1);
semaphore::get(Semaphore::S1);
```

Neither operation waits automatically. Posting stops increasing the count at 15; getting stops decreasing it at zero. Use a wait when work must pause until space or data is available.

### Select several semaphores

List the semaphores as template arguments, or pass a braced list:

```cpp
semaphore::init<0, 2, Semaphore::S1, Semaphore::S3>();
semaphore::post<Semaphore::S1, Semaphore::S3>();
semaphore::get<Semaphore::S1, Semaphore::S3>();

semaphore::init({Semaphore::S1, Semaphore::S3}, 0, 2);
semaphore::post({Semaphore::S1, Semaphore::S3});
semaphore::get({Semaphore::S1, Semaphore::S3});
```

The selection is handled internally. Callers only work with `Semaphore` values.

### Read the count

```cpp
const std::uint8_t pending = semaphore::read<Semaphore::S1>();

// Runtime form.
const std::uint8_t other = semaphore::read(Semaphore::S3);
```

A read returns the current count. It does not reserve work, and another thread may change the count afterward.

### Choose an access path

`post` and `get` use the Tensix path by default. To access a single semaphore directly from the RISC core, select `Access::MMIO`:

```cpp
semaphore::post<Access::MMIO, Semaphore::S1>();
semaphore::get<Access::MMIO, Semaphore::S1>();

semaphore::post<Access::MMIO>(selected_semaphore);
semaphore::get<Access::MMIO>(selected_semaphore);
```

`Access::Tensix` can also be specified explicitly. Explicit access-path overloads take one semaphore. `read` always uses MMIO; `init` uses Tensix.

## Waits

A wait describes **which work to delay** and **what must become ready**. It applies to the issuing Tensix thread. When that thread reaches work covered by the wait, it pauses until the condition clears.

### Wait for available work

Delay Math work while a semaphore is empty:

```cpp
wait::semaphore<
    StallTarget::Math,
    SemaphoreCondition::WhileZero,
    Semaphore::S1>();
```

### Wait for available capacity

Delay Unpack work while the count is at or above its configured maximum:

```cpp
wait::semaphore<
    StallTarget::Unpack,
    SemaphoreCondition::WhileMaximum,
    Semaphore::S1>();
```

Function-argument forms use the order **targets, semaphores, conditions**:

```cpp
wait::semaphore(
    StallTarget::Math,
    Semaphore::S1,
    SemaphoreCondition::WhileZero);
```

To wait on several semaphores, list them:

```cpp
wait::semaphore<
    StallTarget::Math,
    SemaphoreCondition::WhileZero,
    Semaphore::S1,
    Semaphore::S3>();

wait::semaphore(
    StallTarget::Math,
    {Semaphore::S1, Semaphore::S3},
    SemaphoreCondition::WhileZero);
```

This waits while either selected semaphore is empty. Combining `WhileZero | WhileMaximum` waits while any selected semaphore is empty or at its maximum threshold or above.

A wait does not change the count or reserve a token. Keep waiting and consuming within a protocol appropriate for the participating threads.

### Wait for work to finish

Use `wait::stall` when a later operation depends on earlier work completing:

```cpp
// Delay configuration work until the packer has finished.
wait::stall<StallTarget::Config, StallCondition::PackerIdle>();

// Function-argument form.
wait::stall(StallTarget::Config, StallCondition::PackerIdle);
```

When a semaphore signals completed work, wait for that work before posting:

```cpp
// ... issue unpack work ...

wait::stall<
    StallTarget::Sync,
    StallCondition::Unpacker0Idle | StallCondition::Unpacker1Idle>();

semaphore::post<Semaphore::S1>();
```

Choose the condition for the work being completed. Common choices include `MathIdle`, `SfpuIdle`, `PackerIdle`, `Unpacker0Idle`, and `Unpacker1Idle`.

Targets and conditions support `|` to combine selections. All selected completion conditions must be satisfied before the wait clears.

## Mutexes

Choose from `Mutex::M0`, `Mutex::M2`, `Mutex::M3`, and `Mutex::M4`.

Use `acquire` and `release` around shared work performed by Tensix threads:

```cpp
mutex::acquire<Mutex::M2>();

// ... issue protected Tensix work ...
// ... wait for completion if required before releasing the resource ...

mutex::release<Mutex::M2>();
```

Function-argument forms:

```cpp
mutex::acquire(selected_mutex);
// ... protected Tensix work ...
mutex::release(selected_mutex);
```

`acquire` waits for ownership on the Tensix thread. Release the same mutex from the thread that acquired it. These calls coordinate Tensix work; they do not provide a general-purpose lock around RISC-side C++ code.

## Build an operation without executing it

Use a descriptor's `.operation()` when another interface needs the operation value rather than an immediate call:

```cpp
constexpr std::uint32_t post =
    SemaphorePost{Semaphore::S1}.operation();

constexpr std::uint32_t wait_for_data =
    SemaphoreWait{
        StallTarget::Math,
        {Semaphore::S1, Semaphore::S3},
        SemaphoreCondition::WhileZero
    }.operation();
```

Available descriptors:

| Operation | Descriptor |
| --- | --- |
| Acquire a mutex | `MutexAcquire{selector}` |
| Release a mutex | `MutexRelease{selector}` |
| Initialize semaphores | `SemaphoreInit{semaphores, initial, maximum}` |
| Increment semaphores | `SemaphorePost{semaphores}` |
| Decrement semaphores | `SemaphoreGet{semaphores}` |
| Wait for completion | `StallWait{targets, conditions}` |
| Wait on semaphores | `SemaphoreWait{targets, semaphores, conditions}` |

The corresponding `*_operation()` functions provide the same facility:

```cpp
constexpr auto post = semaphore::post_operation<Semaphore::S1>();
constexpr auto acquire = mutex::acquire_operation<Mutex::M2>();
```

These return a `std::uint32_t` without executing the operation. Invalid selections fail compilation when used in a required constant expression, such as a `constexpr` initializer. Runtime checks require `ENABLE_LLK_ASSERT`.

Operation values are available for Tensix operations. Semaphore reads and MMIO accesses use the normal call interfaces.
