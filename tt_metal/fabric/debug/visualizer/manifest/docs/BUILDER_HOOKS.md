# Builder hooks

The manifest records how fabric's routers were built, and most of that is known only to fabric's builder objects. This
doc covers the few places where those builders hand facts to the manifest: when each one runs, what it hands over, and
why it's shaped the way it is. Everything else in manifest generation reads only what these hooks hand over, plus what
the rest of the system knows at write time.

## Why the manifest needs hooks

Fabric builds its routers in `create_and_compile_tt_fabric_program` (`fabric_init.cpp`), one `FabricBuilder` per
device. Two things about that build shape the hooks:

- **The builders don't outlive the build.** The `FabricBuilder` and its router builders are destroyed once the fabric
program compiles. Anything the manifest needs from them has to be collected before then.
- **The kernel's arguments are never stored.** Each router builder computes its kernel's defines and named compile-time
arguments in `create_kernel` and passes them straight to `CreateKernel`. Those arguments are the best record of what
each router was configured with, so the builder has to keep a copy for the manifest.

The manifest is written later, after the routers are launched, when ControlPlane, the cluster and the HAL can answer  
the questions the builders can't, such as which router is on the other end of each link. So the hooks split the work:  
each router's facts are collected during the build and kept, and the rest is added at manifest-write time.

## When each hook runs

```
FabricFirmwareInitializer::init                         (fabric_firmware_initializer.cpp)
├── remove_stale_fabric_manifest                           delete the last run's file, even with the option off
├── per device: create_and_compile_tt_fabric_program    (fabric_init.cpp)
│   ├── FabricBuilder::create_kernels
│   │   ├── keeps the KernelCreationContext                master router, router count, router channel mask
│   │   └── each router's create_kernel                    records its RouterKernelInputs
│   ├── FabricBuilder::publish_manifest_chip              only with TT_METAL_FABRIC_GENERATE_MANIFEST
│   │   ├── each router's collect_manifest_router          runs the router pass on that router
│   │   └── PublishedChips::publish                        keeps the chip in the builder context
│   └── compile the program                                the builders are destroyed after this
├── configure fabric                                       routers launched
├── write_fabric_manifest                                  only with TT_METAL_FABRIC_GENERATE_MANIFEST
│                                                          the topology pass reads PublishedChips
└── wait for router sync
```

The manifest is written before the host waits for router sync, so a run that hangs during bring-up still leaves one.
On a mock device, init compiles fabric without launching it (`compile_fabric_only`), then writes the manifest the same
way.

## The hooks

| Hook | Where | What it hands over |
| --- | --- | --- |
| `RouterKernelInputs` | Defined in `fabric_router_builder.hpp`; filled by `ComputeMeshRouterBuilder::create_kernel` | What the router passed to `CreateKernel`: the defines every RISC shares, and each RISC's processor and named compile-time arguments |
| The kept `KernelCreationContext` | `FabricBuilder::create_kernels` | The chip-wide values every router's kernel was given: the master router's channel, the router count and the router channel mask. The manifest records them as the chip's `local_sync` |
| `collect_manifest_router(chip_facts)` | Pure virtual on `FabricRouterBuilder`; implemented by `ComputeMeshRouterBuilder` | One `manifest::Router`. The builder gathers what only it can see into a `ManifestRouterInputs` and hands it to the router pass (`fabric_manifest_router_pass.cpp`) |
| `publish_manifest_chip()` | `FabricBuilder` | The chip: its Z port role, its local sync, and every router, sorted by Ethernet channel |
| `get_manifest_chips()` | `FabricBuilderContext` | The `PublishedChips` (`fabric_manifest_router_pass.hpp`) that holds each chip by physical chip id. The builder context lives in `FabricContext`, so it outlives the builders, and the topology pass reads it at write time |
| `remove_stale_fabric_manifest`, `write_fabric_manifest` | Called from `FabricFirmwareInitializer::init`; defined in `fabric_manifest.cpp` | Nothing: they delete last run's file and write this one |

`TT_METAL_FABRIC_GENERATE_MANIFEST` sets `RunTimeOptions::get_generate_fabric_manifest()`, which gates
`publish_manifest_chip` and `write_fabric_manifest`.

## What the router pass reads from a router builder

`ComputeMeshRouterBuilder::collect_manifest_router` fills `ManifestRouterInputs` with:

| Input | From | Used for |
| --- | --- | --- |
| `erisc_builder` | The router's `FabricEriscDatamoverBuilder` | Its config's channel allocator (channel buffers and the L1 left over after them), its channel trimming overrides, the receiver's downstream adapter (the edges to sibling routers), and the semaphore ids three fields read |
| `vc_shape` | The router's `RouterVcShape` | The VC count and the receivers per VC |
| `location` | The router's `RouterLocation` | Its Ethernet channel, the direction it faces, and whether its link is a dispatch link |
| `chip_facts` | The chip's `ChipRoutingFacts`, passed in by `FabricBuilder` | The edge capability of the direction the router faces, which becomes its link's `edge_capability` |
| `addresses_to_clear` | `FabricBuilderContext::get_fabric_router_addresses_to_clear()` | Whether fabric init clears each L1 region (`cleared_by_host`) |
| `kernel` | The `RouterKernelInputs` recorded in `create_kernel` | Every field the field tables read, and the check that every argument and define is accounted for |
| `hal` | `FabricContext`'s HAL | The physical ERISC each RISC runs on |

[ROUTER_PASS.md](ROUTER_PASS.md) covers what the router pass does with these.

## Keeping the footprint small

The hooks are meant to cost fabric as little as possible:

- **Fabric's headers include no manifest headers.** `fabric_router_builder.hpp` forward-declares `manifest::Router`,
`fabric_builder_context.hpp` forward-declares `manifest::PublishedChips` and holds it through a `unique_ptr`, and
`RouterKernelInputs` is plain builder data. Only four fabric `.cpp` files include manifest headers:
`compute_mesh_router_builder.cpp`, `fabric_builder.cpp`, `fabric_builder_context.cpp` and
`fabric_firmware_initializer.cpp`. A change to the manifest doesn't recompile the code that includes fabric's
headers.
- **Each hook lives in the class that has the data.** The router builder knows its ERISC builder, VC shape and kernel
arguments, so it gathers them itself. The manifest needs no friend access, and fabric needs no accessors added just
for it.
- **With the option off, almost nothing runs.** Every router still records its `RouterKernelInputs`, which is a copy of
arguments it already computed, and the builder context holds an empty `PublishedChips`. Nothing is collected or
written.

## Checks at the hooks

- `PublishedChips::publish` fails when a chip is published twice, when its id is out of range, or when it publishes a
different number of routers than its builder initialized
(`FabricBuilderContext::get_num_fabric_initialized_routers`).
- `PublishedChips::get` fails for a chip that was never published. The topology pass treats a local chip that
published nothing as having no routers, which is only valid when ControlPlane lists no active fabric channels for it.
Otherwise resolving its routers fails.
- `collect_manifest_router` must run after `create_kernel`, once connections are wired and each RISC's compile-time
arguments are final. `publish_manifest_chip` is called right after `create_kernels` for that reason.

## Adding a router builder type

A new `FabricRouterBuilder` subclass has to:

1. Implement `collect_manifest_router`. It's pure virtual, so the build fails until it does.
2. Record its `RouterKernelInputs` in `create_kernel`, as it calls `CreateKernel`.
3. Fill a `ManifestRouterInputs` and call the router pass, or describe its router some other way if it isn't an

   ERISC datamover router. The router pass assumes one.
