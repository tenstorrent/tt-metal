# Running tests

## Basic examples

FF1 without gelu: `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest tests/didt/test_ff1_matmul.py::test_ff1_matmul -k "without_gelu and 2chips"`

FF1 with gelu: `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest tests/didt/test_ff1_matmul.py::test_ff1_matmul -k "with_gelu and 2chips"`

LM head: `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest tests/didt/test_lm_head_matmul.py::test_lm_head_matmul -k "2chips"`

Resnet Convolution: `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest tests/didt/test_resnet_conv.py::test_resnet_conv -k "2chips"`

## Variations

### Supported systems

We support N150, N300, T3000, 6U Galaxy systems, and single chip Blackhole. To choose the system, pass in the following parametrization ids (as shown in the example commands):
- 1chips
- 2chips
- 8chips
- galaxy

NOTE: If running on Galaxy or Blackhole systems, remove the WH_ARCH_YAML env variable from the command.

### Targetting specific device

On all multi-device systems, you can target a specific device using its ID in the parametrization `logical_chip_{id}_`:

`WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest tests/didt/test_ff1_matmul.py::test_specific_chip_ff1_matmul -k "without_gelu and 8chips and logical_chip_3_"`

Galaxy example
`pytest tests/didt/test_ff1_matmul.py::test_specific_chip_ff1_matmul -k "without_gelu and galaxy and logical_chip_3_"`

### Targetting specific board

On T3000 systems, you can target a specific board (local and remote chip together) using the ID of the local device in the parametrization `board_id_{id}`:

`WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest tests/didt/test_ff1_matmul.py::test_specific_board_ff1_matmul -k "without_gelu and 8chips and board_id_2"`

### Iterations

By default, we run 100000 iterations of the workload loop, but you can override that behavior using `--didt-workload-iterations` option:

`WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest tests/didt/test_ff1_matmul.py::test_ff1_matmul -k "without_gelu and 2chips" --didt-workload-iterations 5000000`

### Determinism

If you wish to check if the output is deterministic, simply pass in the `--determinism-check-interval` option - the option tells on how many iterations we do the determinism check. Example:

`WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest tests/didt/test_ff1_matmul.py::test_ff1_matmul -k "without_gelu and 2chips" --determinism-check-interval 50`

## Blackhole

Tests are only supported with `1chips` ID because multi-chip/board BH has not been brought up yet.

By default the Blackhole workload compute grid is 13x10 (1 column is reserved for fast-dispatch).

### Grid size

You can run matmul on grid size smaller than 8x8 which is the default option. When we decrease the grid size, we need to keep the compute same for all cores, so we just lower the outer input dimensions accordingly.
To use smaller grid sizes, you need to target a different test in the file and add parametrization query for the specific grid size (ix8 or 8xi, where i is between [1,8]).

`WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest tests/didt/test_ff1_matmul.py::test_grid_size_ff1_matmul -k "without_gelu and 2chips and 6x8"`


## Legacy commands

For backwards compatibility, we still support the commands used so far and their old behavior:

FF1 without gelu: `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/experimental/falcon_7b/tests/test_reproduce_hang_matmul.py -k "test_reproduce_matmul_2d_hang and ff1-hang and 8chips"`

FF1 with gelu: `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest tests/didt/test_sharded_ff1.py -k "test_reproduce_matmul_2d_hang and 8chips"`

LM head: `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/falcon7b/tests/test_falcon_hang.py -k "test_reproduce_lm_head_nd_32 and 8chips"`




# Adding another suspected repro test

`tests/didt/op_test_base.py` defines a base class for all tests that encapsulates common behavior - how we run iterations, deallocate, check determinism, sync, etc.  To add a new test, create a new file under the same directory, and then either:
- instantiate object of the base class in case you don't need to change any behavior, just populate dimensions, configs etc (example in `test_ff1_matmul.py`)
- extend the base class to override any behavior that needs to be changed (for now we allow to change the way we generate activations & weights, and setting the seed), and then instantiate object of the new class
