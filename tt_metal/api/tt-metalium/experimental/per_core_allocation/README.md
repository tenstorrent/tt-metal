# Per-Core Allocation (Experimental)

This directory contains the experimental per-core L1 allocation API.

## Overview

Standard L1 allocation uses a single "lockstep" allocator: every core that
holds a shard of a buffer receives the same L1 address.  Per-core allocation
relaxes this constraint by maintaining independent allocators per bank, so
each core can receive a different address.

The feature is gated behind `AllocatorMode::HYBRID`, which is enabled by
setting the environment variable `TT_METAL_ALLOCATOR_MODE_HYBRID=1`.

## Usage

```cpp
#include <tt-metalium/experimental/per_core_allocation/allocator_mode.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>

// Enable hybrid allocator mode via env var before device creation:
//   export TT_METAL_ALLOCATOR_MODE_HYBRID=1

// Query per-core addresses from a buffer
auto addr = experimental::per_core_allocation::get_per_core_address(buffer, core);
```

## Why experimental?

Per-core allocation is not yet general-purpose.  It is currently used by
DeepSeek compressed-tensor inference and has restrictions (L1-only, sharded
layouts, no buffer views, no trace state serialisation).  Once the feature
matures it may be promoted into the stable public API.
