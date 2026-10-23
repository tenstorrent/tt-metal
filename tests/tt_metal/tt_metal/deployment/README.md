# Deployment Test Suite
Qualification tests for new systems. Run `test_runner.sh` from the repository
root after building the tests (`./build_metal.sh --build-metal-tests`); see
`test_runner.sh -h` for options.

| Step | Command | What it checks |
|------|---------|----------------|
| Ethernet tests | `eth/test_runner.py` | Link up, bandwidth and data integrity on every Ethernet link (see [eth/README.md](eth/README.md)) |
| DRAM tests | `dram/test_runner.py` | DRAM data patterns on every chip |
| NOC tests | `unit_tests_deployment --gtest_filter='*DeploymentNoc*'` | On-chip NOC data integrity from every Ethernet core (and a few Tensix cores) on both NOCs |
| PCIe read test | `build/tools/mem_bench` | Device reads from host memory |
| PCIe write test | `build/tools/mem_bench` | Device writes to host memory |
| DIDT tests | `didt/test_runner.sh` | di/dt stress, run once after all iterations |

## NOC tests
* **TensixDeploymentNoc00EthSourceIntegrity**
  Every Ethernet core (active and idle) on every chip writes data patterns
  (walking 1s/0s, checkerboard, address-as-data, PRBS) to Tensix cores in the
  first worker rows, to the other Ethernet cores' L1 and to the DRAM endpoints
  in NOC rows 0/1. It reads the data back and compares it on-device, once on
  NOC0 and once on NOC1. Mismatches and barrier timeouts are reported per chip,
  source channel, destination and NOC.

* **TensixDeploymentNoc01TensixSourceIntegrity**
  Same destinations, with four corner Tensix cores as sources. Use it to tell
  a source-side fault from a destination-side one.

Both tests need `TT_METAL_SLOW_DISPATCH_MODE=1`; the runner sets it.

Environment variables:
- `NOC_TEST_ITERATIONS`: Pattern passes per source/destination (default 1)
- `NOC_TEST_SLICE_SIZE`: Bytes per source in each destination (default 4096)
- `NOC_TEST_PACKET_SIZE`: NOC write size (default 2048)
- `NOC_TEST_BURST`: Writes of the slice in flight before each barrier (default 8)
- `NOC_TEST_VC`: Unicast write VC (default 2, as used by fabric)
- `NOC_TEST_TENSIX_ROWS`: Number of worker rows used as destinations (default 2)
- `NOC_TEST_TIMEOUT_MS`: Barrier timeout in the kernel (default 2000)
- `NOC_TEST_DRAM_ALL_NIUS`: Also target non-preferred DRAM NIUs when DRISC firmware is off (default 0)
