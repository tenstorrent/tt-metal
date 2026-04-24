# Kernels

Topic knowledge for tt-metal kernels — external references only today;
extend with patterns and traps as they surface.

## External references

- **Canonical reader/compute/writer pattern** — `tt-metal/tt_metal/programming_examples/eltwise_binary/`
  Three kernels: reader (NOC→CB), compute (CB→CB), writer (CB→NOC). Best
  starting point.
- **Compute-only (no data movement)** — `tt-metal/tt_metal/programming_examples/add_2_integers_in_compute/`
  Minimal compute kernel with compile-time and runtime args.
- **Dataflow API** — `tt-metal/tt_metal/hw/inc/api/dataflow/dataflow_api.h`
  All NOC and CB functions for reader/writer kernels. Read for exact
  signatures.
- **Compute API** — `tt-metal/tt_metal/hw/inc/api/compute/`
  All tile operations, FPU/SFPU functions. Read for exact compute
  signatures.
- **Circular buffer host config** — `tt-metal/tt_metal/api/tt-metalium/circular_buffer_config.hpp`
  Host-side CB setup API.
