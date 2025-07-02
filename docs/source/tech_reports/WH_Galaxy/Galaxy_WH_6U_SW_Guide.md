# Installing Galaxy Wormhole Systems

## Hardware Setup
Follow the instructions for your specific Wormhole device at: [Wormhole Hardware Setup](https://docs.tenstorrent.com/aibs/wormhole/).

## Software Installation
To install TT-Metalium visit the [TT-Installer](https://github.com/tenstorrent/tt-installer) repository on GitHub for a quick installation package. The [Installation Guide](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md) in the TT-Metal repository has installation instructions for advanced developers.

# Troubleshooting Galaxy Wormhole 6U Systems
If you suspect system issues in Wormhole Galaxy 6U systems reference the following troubleshooting methods to diagnose and correct any problems. The [TT-Metal](https://github.com/tenstorrent/tt-metal) repository landing page on GitHub lists models that can be run on 6U systems. Look for models that indicate Galaxy hardware is used.

## TT-SMI Reset and System Information Dump
Visit the [TT-SMI](https://github.com/tenstorrent/tt-smi) repository on GitHub for instructions on how to perform a reset on a 6U systems.

The following display will appear upon performing a successful system reset:

![](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/common/images/TT_SMI_Successful_Reset.png)

To perform a system information dump see the [TT-SMI README.md](https://github.com/tenstorrent/tt-smi/blob/main/README.md). In the TT-SMI repository there are instructions for building from git, usage, resets, disabling software versions, taking system snapshots, and more.

## Testing System Configurations
Testing any system configuration is required to ensure that code is behaving as intended and the configuration is set up properly. Run component tests to verify basic functionality of components in the AI stack.

### Component Tests
When performing systems tests developers should use the following test suite:

`dockerfile/upstream_test_images/run_upstream_tests_vanilla.sh`

The following tests are short in runtime and loop over all chips to test for basic component functionality:
- `./build/test/tt_metal/tt_fabric/test_system_health` - This test prints Ethernet connnections, link statuses, and performs sanity checks on the system.
  The following expected output will appear upon completion of a successful Ethernet connection test:

![](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/common/images/Ethernet_Link_Status_Test.png)

- `TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="CommandQueueSingleCardFixture.*"` - This test ensures that basic Command Queue APIs function correctly.
  The following expected output will appear upon completion of a successful Command Queue API test:

![](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/common/images/CQ_API_Test.png)

- `TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="CommandQueueSingleCardProgramFixture.*"` - This test ensures that Metal Program APIs function correctly.
  The following expected output will appear upon completion of a successful Program API test:

![](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/common/images/Program_API_Test.png)

- `TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="CommandQueueSingleCardBufferFixture.ShardedBufferLarge*ReadWrites"` - This test ensures that basic Metal Buffers read and write to both L1 and DRAM memory buffers.
  The following expected output will appear upon completion of a successful Memory Buffer test:

![](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/common/images/Memory_Buffer_Test.png)

- `./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2D*Fixture.*"` - This test sets up the Galaxy as a 2D mesh and uses basic Fabric APIs to read and write between chips.
  The following expected output will appear upon completion of a successful Fabric API test:

![](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/common/images/Fabric_API_Test.png)

### Ethernet Bandwidth Tests
The following Ethernet bandwidth tests loop over all active Ethernet links and prints bandwidth. The following tests require a profiler build and therefore take longer to run:
- `./build_metal.sh --build-tests --enable-profiler` - This command enables the profiler.
- `pytest tests/tt_metal/microbenchmarks/ethernet/test_all_ethernet_links_bandwidth.py` - This test ensures that Ethernet links are properly sending and receiving data.
  In the following expected output the test is identifying all mismatching values.

![](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/common/images/Ethernet_Link_Test.png)

> [!NOTE]
> This test can take several hours to complete.

- `pytest tests/tt_metal/microbenchmarks/ethernet/test_all_ethernet_links_latency.py` - This test prints latency for all Ethernet links.
  In the following expected output the test is identifying high latency of the machine:

![](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/common/images/Ethernet_Link_Latency_Test.png)

# FAQ
## What does Metal infra provision vs. what does Cloud provision?
Cloud owns:
- Physical hardware and data center installation.
- Initial boot.
- Any Tenstorrent related health checks like ensuring devices come up and smoke tests.
Infra owns:
- System-dependency provisioning for Metalium.
- Firmware and driver installation if Metal has specific needs.
- Enabling and access of development environment.
- Installation of CI runner.
- Weka MLPerf Mount.
- Metalium infra smoke tests: [sw-hello-world repository](https://github.com/tt-rkim/sw-hello-world)

## How does Metal Infra deal with hardware-related issues?
After we have attempted to revive a machine we will execute the following:
- File and issue with the cloud team.
- If in CI, we will take the machine out of service and mark it with an issue number label describing the problem.
- Post a cloud issue in our cloud support channel.
- The cloud team will assist and further escalate the issue if needed.

## For 6U, what is the difference between POR and C5?
The C5 is an early 6U build that does not meet certain power draw requirements. Lightweight functional bring-up is possible however it is not recommended for tasks like model-development and benchmarking.

## Cloud 6U will not boot up after a reset?
Run the following command from any cloud machine with access to the -oob of the 6U machine:

`ipmitool -C 17 -I lanplus -H <GALAXY_HOST>-oob -U root -P 0penBmc chassis power reset`

## What is -oob?
All BMCs for cloud hosts have the -oob (out of band) suffix. For example, g05glx6u01-oob.

To perform a full power drain run the following command:

`ipmitool -C 17 -I lanplus -H <GALAXY_HOST>-oob -U root -P 0penBmc chassis power off
sleep 300
ipmitool -C 17 -I lanplus -H <GALAXY_HOST>-oob -U root -P 0penBmc chassis power on`

If this is a first time setup you must run the following command prior to the previous power reset command:

`ipmitool -C 17 -I lanplus -H <GALAXY_HOST>-oob -U root -P 0penBmc chassis bootdev disk options=efiboot,persistent`
