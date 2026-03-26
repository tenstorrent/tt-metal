# Kernel Reference Pointers

## Canonical reader/compute/writer pattern
`tt_metal/programming_examples/eltwise_binary/`
Three kernels: reader (NOC→CB), compute (CB→CB), writer (CB→NOC). Best starting point.

## Compute-only (no data movement)
`tt_metal/programming_examples/add_2_integers_in_compute/`
Minimal example of compute kernel with compile-time and runtime args.

## Dataflow API
`tt_metal/hw/inc/api/dataflow/dataflow_api.h`
All NOC and CB functions for reader/writer kernels. Read this for exact signatures.

## Compute API
`tt_metal/hw/inc/api/compute/`
All tile operations, FPU/SFPU functions. Read this for exact compute signatures.

## Circular buffer host config
`tt_metal/api/tt-metalium/circular_buffer_config.hpp`
Host-side CB setup API.
