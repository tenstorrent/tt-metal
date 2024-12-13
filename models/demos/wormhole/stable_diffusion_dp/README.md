
# Stable Diffusion

## How To Run

### Single Device
#### Wormhole_B0 Device Performance

To obtain device performance,
1. Set the WH_ARCH_YAML environment variable: ```export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml ```
2. Run the test using
```
pytest models/demos/wormhole/stable_diffusion_dp/tests/test_perf_device_sd.py::test_perf_device
```
This will run the model for 4 times and generate CSV reports under `<this repo dir>/generated/profiler/reports/ops/<report name>`. The report file name is logged in the run output.
It will also show a sumary of the device throughput in the run output.

#### Wormhole_B0 End-to-End Performance

For end-to-end performance,
1. Set the WH_ARCH_YAML environment variable: ```export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml ```
2. Run the test using
```
pytest models/demos/wormhole/stable_diffusion_dp/tests/test_perf_e2e_sd.py::test_perf
```
 This will generate a CSV with the timings and throughputs.
**Expected end-to-end perf**: For batch = 2, it is about `13.13 fps` currently. This may vary machine to machine.
