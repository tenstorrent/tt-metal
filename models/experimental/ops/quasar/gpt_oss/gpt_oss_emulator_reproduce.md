# gpt-oss on Quasar: reproducing the passing end-to-end demo

Branch `fplavec/gpt_oss_emulator`: `origin/main` (`c947577e109`) plus the demo changes in commit `aad28cb0683`
(`DEMO_NO_PROGRAM_CACHE`, `DEMO_MAX_TOKENS`, `DEMO_TIMEOUT_S`, `DEMO_SKIP_COMPILE_PREFILL` in `demo/text_demo.py`, and
the prefill sparsity mask in L1). No local plugins are needed. Run from the tt-metal root after `./build_metal.sh` and
`./create_venv.sh`. `HF_MODEL` must point at a gpt-oss-20b checkpoint in a folder named exactly `gpt-oss-20b` (here a
symlink to `/proj_sw/user_dev/gpt-oss/gpt-oss-20b-BF16`). `TT_CACHE_PATH` keeps the ttnn weight cache out of the model
folder. `TT_METAL_WATCHER` and `TT_METAL_LLK_ASSERTS` must not be set.

## ttsim

Uses the ttsim-private `qsr` build, staged next to a `soc_descriptor.yaml` that declares 3 GiB per DRAM channel (a copy
of `tt_metal/soc_descriptors/quasar_32_arch_ttsim.yaml` with `dram_view_size` and `dram_bank_size` set to 3221225472),
on the 2x1 grid the emulator has. About 40 to 60 minutes.

```bash
source python_env/bin/activate
export TT_METAL_HOME=$PWD ARCH_NAME=quasar CHIP_ARCH=quasar PYTHONPATH=$PWD/ttnn:$PWD LD_LIBRARY_PATH=$PWD/build/lib
export TT_METAL_SIMULATOR=/localdev/fplavec/ttsim_qsr_3g/libttsim.so TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DISABLE_SFPLOADMACRO=1
export TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="3,2"
export HF_MODEL=/localdev/fplavec/temp/gpt_oss_models/gpt-oss-20b TT_CACHE_PATH=/localdev/fplavec/temp/gpt_oss_weight_cache
QSR_N_LAYERS=2 DEMO_NO_PROGRAM_CACHE=1 DEMO_MAX_TOKENS=3 DEMO_TIMEOUT_S=172800 \
  pytest -s "models/experimental/ops/quasar/gpt_oss/demo/text_demo.py::test_gpt_oss_demo[quasar-1x1-prefill_128]"
```

`DEMO_NO_PROGRAM_CACHE=1` is not needed on ttsim (its DRAM does not alias) but does no harm.

## Quasar emulator

Uses a copy of the `emu-quasar-2x3` build directory whose `soc_descriptor.yaml` declares two DRAM channels of 3 GiB
(`/proj_sw/user_dev/fplavec/emu-quasar-2x3_2ch3g/`; see `gpt_oss_emulator_hang_investigation.md`). `NNG_SOCKET_ADDR` is the reservation's physical
machine and its `P_USER_DBD_PORT`. The run takes about 6.5 hours once the farm starts it (about 5 hours with
`DEMO_SKIP_COMPILE_PREFILL=1`), plus the time in the farm queue; `DEMO_TIMEOUT_S` covers the queue wait as well,
because the device open waits for the remote job. Check that no earlier job of yours is on the ZeBu dashboard and that
port 5555 is free before launching.

```bash
source python_env/bin/activate
export TT_METAL_HOME=$PWD ARCH_NAME=quasar CHIP_ARCH=quasar PYTHONPATH=$PWD/ttnn:$PWD LD_LIBRARY_PATH=$PWD/build/lib
export TT_METAL_SIMULATOR=/proj_sw/user_dev/fplavec/emu-quasar-2x3_2ch3g/ TT_METAL_SIMULATOR_DRAM_UNCAPPED=1
export NNG_SOCKET_ADDR=tcp://yyzc-swc08:54539 NNG_SOCKET_LOCAL_PORT=5555 EMULATOR_TIMEOUT=172800
export TT_METAL_SLOW_DISPATCH_MODE=1
export HF_MODEL=/localdev/fplavec/temp/gpt_oss_models/gpt-oss-20b TT_CACHE_PATH=/localdev/fplavec/temp/gpt_oss_weight_cache
QSR_N_LAYERS=2 DEMO_NO_PROGRAM_CACHE=1 DEMO_MAX_TOKENS=3 DEMO_TIMEOUT_S=259200 \
  pytest -s "models/experimental/ops/quasar/gpt_oss/demo/text_demo.py::test_gpt_oss_demo[quasar-1x1-prefill_128]"
```

If the run is stopped early or aborts, the ZeBu job keeps running until `EMULATOR_TIMEOUT`; only a kill from the ZeBu
dashboard releases it.
