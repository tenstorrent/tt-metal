# Running LLK Tests on Versim

Versim is a functional Tensix model. It runs on a plain x86_64 Linux host, needs
no silicon and no simulator licence, and is reached through tt-exalens like the
other simulator backends.

Its one significant difference from ttsim and VCS: **the host cannot reach the
Tensix register bus.** Reads and writes from the host resolve straight into L1,
so an access to the `0xFFB.....` aperture silently aliases -- a read returns
whatever L1 holds at the aliased offset, a write lands somewhere in L1 and is
lost. Nothing reports an error. Register access from *inside* the Tensix (the
RISCs' own `reg_read`/`reg_write`) is unaffected, which is why BRISC firmware can
still drive TRISC reset and the tests work at all.

The harness handles this: `TestTargetConfig` detects the backend and exposes
`has_host_register_access`, and the affected paths (reset, ELF load, the
ebreak/callstack probe) take a different route on Versim. See
`helpers/target_config.py`.

## Prerequisites

- A Versim build directory from
  [tt-umd-simulators](https://yyz-gitlab.local.tenstorrent.com/tensix/tt-umd-simulators),
  e.g. `build/versim-wormhole-b0`. It holds the `run.sh` UMD spawns plus the
  `versim-<arch>` binary. Building needs `git-lfs` for the `third_party/versim`
  submodule.
- The LLK test venv (`tests/.venv`). tt-exalens 0.3.29 or newer is required:
  earlier versions crash on Wormhole in `WormholeDevice.__init__`.
- SFPI, as for any other backend.

## Running tests

```bash
cd tt_metal/tt-llk/tests/python_tests

export LLK_HOME=<path to>/tt-llk
export TT_METAL_SIMULATOR=<path to>/tt-umd-simulators/build/versim-wormhole-b0
export TT_SIMULATOR_LOCALHOST=1
export NNG_SOCKET_NAME=llk          # any name; distinguishes concurrent runs
export CHIP_ARCH=wormhole

../.venv/bin/python -u -m pytest --run-simulator -p no:randomly -v <test>
```

No tt-exalens server to start: Versim runs in-process, and UMD spawns the
simulator itself. `TT_UMD_SIMULATOR_PATH` works as an alias for
`TT_METAL_SIMULATOR`.

### Canonical smoke test

The smallest matmul in the suite -- one tile per operand, lowest fidelity:

```bash
../.venv/bin/python -u -m pytest --run-simulator -p no:randomly -v \
 'test_matmul.py::test_matmul[math_fidelity:LoFi-format_dest_acc_and_dims:(InputOutputFormat[A:Float16_b,B:Float16_b,out:Float16_b], <DestAccumulation.No: False>, ([32, 32], [32, 32]))]'
```

Expect roughly 1.5-3.5 minutes, depending on whether the ELF artefacts are
already built.

## Speed, and the waveform dump

Versim dumps a VCD by default. It is the dominant cost: the file reaches several
GB within the hour, and throughput roughly halves as it grows. The only switch is
the `USER` environment variable, checked in `tt_versim_device.cpp`:

```bash
USER=gitlab-ci ../.venv/bin/python -u -m pytest --run-simulator ...   # no VCD
```

Budget about 2 minutes of wall clock per output tile with the dump off. A test
much larger than a few tiles will exceed the 600 s kernel timeout
(`TestTargetConfig.kernel_timeout_s`) before it finishes.

Both the VCD and the `versim_<timestamp>_<socket>.log` are written to the current
working directory, which is inside the repo. Check `git clean -n` before
committing.

Long runs are worth detaching (`setsid nohup ... > run.log 2>&1 &`); an
interrupted shell otherwise takes the simulator with it.

## Supported architectures

| Arch | Build directory | Status |
|------|-----------------|--------|
| Wormhole B0 | `versim-wormhole-b0` | supported |
| Blackhole | `versim-blackhole` | needs the BRISC reset-PC change; rejected with a clear error without it |
| Quasar | — | Versim has no Quasar model |

## Limitations

- **BRISC boot mode only.** TRISC boot needs the host to deassert a single TRISC,
  but the reset backdoor is whole-core; EXALENS boot needs the RISC debug
  hardware to inject instructions. Both are refused with an explicit error.
- **No RISC debug hardware.** The ebreak/callstack probe is skipped, so a hang
  reports as a plain timeout with no callstack.
- **`--reset-simulator-per-test` is unsupported** -- it restarts the tt-exalens
  server, which Versim does not use. Refused rather than silently ignored.

## Troubleshooting

A run that produces no result is usually one of two things, and the log tells
them apart:

```bash
grep -c "ASSERTION FAILED" versim_*.log
```

Non-zero means the Tensix model tripped an RTL assertion -- a real problem in
what the kernel asked the hardware to do. Zero, with the log ending in a repeating
`read(1, 1, 0x0001ffb8, 12)`, means the TRISC mailboxes never reached `0xFF`: the
kernel is still running (or stuck), and the wait timed out. `0x1FFB8` is the
Unpacker/Math/Packer mailbox triple.

If the tests build for one architecture and run on another, they hang rather than
fail. Keep `CHIP_ARCH` and `TT_METAL_SIMULATOR` pointing at the same arch.
