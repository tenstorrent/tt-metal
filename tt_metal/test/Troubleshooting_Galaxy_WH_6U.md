# Troubleshooting Galaxy Wormhole 6U Systems
If you suspect system issues in Wormhole Galaxy 6U systems reference the following troubleshooting methods to diagnose and correct any problems. The [TT-Metal](https://github.com/tenstorrent/tt-metal) repository landing page on GitHub lists models that can be runon 6U systems. Look for models that indicate Galaxy hardware is used.

## TT-SMI System Information Dump
To perform a system information dump see the [TT-SMI README.md](https://github.com/tenstorrent/tt-smi/blob/main/README.md). In the TT-SMI repository there are instructions for building from git, usage, resets, disabling software versions, taking system snapshots, and more.

## Testing System Configurations
Testing any system configuration if required to ensure that code is behaving as intended and the configuration is set up properly. You can build a custom test or run component tests to check for basic functionality of componenets in the AI stack.

### Build a Metal Test
To build a custom test run the following command: `./build_metal.sh --build-test`

### Component Tests
The following tests are short in runtime and loop over all chips to test for basic component functionality:
- `./build/test/tt_fabric/test_system_health` - This test prints Ethernet connnections, link statuses, and performs sanity checks on the system.
- `TT_METAL_SKIP_EHT_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="CommandQueueSingleCardFixture.*"` - This test ensures that basic Command Queue APIs function correctly.
- `TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="CommandQueueSingleCardProgramFixture.*"` - This test ensures that Metal Program APIs function correctly.
- 'TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filetr="CommandQueueSingleCardBufferFixture.ShardedBufferLarge*ReadWrites" - This test ensures that basic Metal Buffers read and write to both L1 and DRAM memory buffers.
- `./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2D*Fixture.*"` - This test sets up the Galaxy as a 2D mesh and uses basic Fabric APIs to read and write between chips.

### Ethernet Bandwidth Tests
The following Ethernet bandwidth tests loop over all active Ethernet links and prints bandwidth. The following tests require a profiler build and therefore take longer to run:
- `./build_metal.sh --build-test --enable-profiler` - This command enables the profiler.
- `pytest tests/tt_metal/microbenchmarks/ethernet/test_all_ethernet_links_bandwidth.py` - This test ensures that Ethernet links are properly sending and receiving data.
- `pytest tests/tt_metal/microbenchmarks/ethernet/test_all_ethernet_links_latency.py` - This test prints latency for all Ethernet links.
