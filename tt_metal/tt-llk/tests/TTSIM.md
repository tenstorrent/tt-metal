# Running LLK Tests on ttsim

[ttsim](https://github.com/tenstorrent/ttsim) is Tenstorrent's functional
simulator. It models a Wormhole or Blackhole chip end-to-end on a plain
x86_64 Linux host — no silicon required. The LLK test suite can target
ttsim with the same pytest invocations used for silicon, modulo a couple
of environment variables.

## Prerequisites

- `libttsim_{wh,bh}.so` downloaded from
  [ttsim releases](https://github.com/tenstorrent/ttsim/releases).
- A SoC descriptor YAML for the target arch, placed next to the `.so`
  and named `soc_descriptor.yaml` (ttsim derives its path from the `.so`
  path).
- `tt-umd` with TTSim fixes for arch / noc-translation / chip-info and
  clock advance on host writes (check `pip show tt-umd`; required
  version depends on when these land).
- `tt-exalens` with the `send_tensix_risc_reset` positional-API fix
  (0.3.16+).
- SFPI compiler set up per the main `README.md` (SFPI is required for
  kernel compilation regardless of backend).

## Setup

```bash
# One-time: assemble the simulator directory
mkdir -p ~/sim
cd ~/sim
# Pick the arch that matches the tests you want to run:
wget https://github.com/tenstorrent/ttsim/releases/latest/download/libttsim_bh.so
# Copy the arch-appropriate SOC descriptor from tt-metal:
cp $TT_METAL_HOME/tt_metal/soc_descriptors/blackhole_140_arch.yaml \
   ~/sim/soc_descriptor.yaml
```

## Running tests

```bash
export TT_METAL_SIMULATOR=~/sim/libttsim_bh.so

# ttsim does not currently support SFPLOADMACRO — disable it if your
# tests hit the SFPU:
export TT_METAL_DISABLE_SFPLOADMACRO=1

cd $TT_METAL_HOME/tt_metal/tt-llk/tests/python_tests
pytest -v --run-simulator --timeout=600 test_eltwise_unary_datacopy.py
```

`TT_METAL_SIMULATOR` is the canonical env var — the same one tt-metal
runtime and the ttsim project docs use. `TT_UMD_SIMULATOR_PATH` is
accepted as an alias for back-compat with the RTL-simulator flow; when
both are set, `TT_METAL_SIMULATOR` wins.

### Canonical smoke test

The simplest end-to-end check that the stack is wired up:

```bash
pytest -v --run-simulator --timeout=300 \
  test_eltwise_unary_datacopy.py -k "Float16_b and not tilize"
```

This exercises unpack → math (datacopy) → pack on a single tile with
the most common format and no SFPU/tilize paths.

## Supported architectures

| Arch | LLK tests on ttsim |
|---|---|
| Blackhole | Working (validated against ttsim v1.5.5). |
| Wormhole B0 | Working (validated against ttsim v1.5.5). |
| Quasar | Not validated. |

## Limitations

- **Slow dispatch only.** ttsim does not yet implement fast dispatch.
  LLK tests go through the direct exalens path and are unaffected, but
  any programming example invoked alongside needs
  `TT_METAL_SLOW_DISPATCH_MODE=1`.
- **SFPLOADMACRO** is not implemented. Set
  `TT_METAL_DISABLE_SFPLOADMACRO=1` if your test hits the SFPU.
- **Not all ISA is modeled.** Unimplemented opcodes surface as
  `UnimplementedFunctionality` errors from the simulator. See the
  [ttsim README](https://github.com/tenstorrent/ttsim) for the error
  taxonomy, and file a bug against the ttsim repo with a minimal repro
  if you hit one.
- **Performance.** ttsim is substantially slower than silicon for
  individual tests. Use it for correctness and CI coverage, not
  performance measurements.

## Troubleshooting

- `Getting NOC translation status is not supported in TTSim simulation device` —
  old tt-umd; upgrade to a release that contains the TTSimTTDevice fixes.
- `Exactly 2 or 14 ETH cores should be harvested on full Blackhole` —
  same cause; upgrade tt-umd.
- `SOFT_RESET_0=0x6F` warnings flooding the log — old tt-exalens;
  upgrade to 0.3.16+ (the reset sequence is skipped on ttsim).
- `UnimplementedFunctionality: <opcode>` from ttsim — ISA gap, not a
  test or harness issue. File against
  [tenstorrent/ttsim](https://github.com/tenstorrent/ttsim/issues) with
  a minimal repro.
- `Polling brisc command timed out` on WH — you're running an old
  tt-umd. The fix that advances the sim clock on host writes
  (TTSimTTDevice::write_to_device) is required; upgrade to a release
  that contains it.
