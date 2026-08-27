# AGENTS.md

## Cursor Cloud specific instructions

This is **tenstorrent/tt-metal**, a C++/Python SDK + model zoo for Tenstorrent AI
accelerators (Wormhole / Blackhole / Quasar). See `README.md`, `INSTALLING.md`,
`CONTRIBUTING.md`, and `tt_metal/tt-llk/tests/README.md` for the full docs.

### Hardware reality in the cloud VM

There is **no Tenstorrent device** in the Cursor Cloud VM (`/dev/tenstorrent`
does not exist). Anything that opens a real device cannot run end to end here:
the full `./build_metal.sh` C++ build + `ttnn` Python API + `models/` demos +
`tt-train` all require silicon (or a much heavier RTL/emulation setup) to
actually execute. Do not expect `python3 -m ttnn.examples.usage.run_op_on_device`
or device pytest suites to pass without hardware.

### What DOES run without hardware: LLK tests on the ttsim functional simulator

The supported silicon-free path is the low-level-kernel (LLK) test suite running
on [ttsim](https://github.com/tenstorrent/ttsim), Tenstorrent's functional
simulator. It compiles real kernels with the SFPI compiler and runs
unpack → math → pack on a modeled Blackhole/Wormhole chip. Full docs:
`tt_metal/tt-llk/tests/TTSIM.md`.

The environment set up by the update script / snapshot already provides:
- Python 3.10 venv at `tt_metal/tt-llk/tests/.venv` (deps from
  `tt_metal/tt-llk/tests/requirements.txt`, includes `tt-exalens`, `tt-umd`,
  CPU `torch`).
- SFPI toolchain at `tt_metal/tt-llk/tests/sfpi` (via `setup_testing_env.sh`).
- ttsim Blackhole library + SoC descriptor at `~/sim/` (`libttsim_bh.so`,
  `soc_descriptor.yaml`). This lives in `$HOME`, not the repo — if the snapshot
  is ever reset, re-download it per `TTSIM.md` (Blackhole/Wormhole libs from
  the ttsim GitHub releases; copy `tt_metal/soc_descriptors/blackhole_140_arch.yaml`
  to `~/sim/soc_descriptor.yaml`).

Run a test on the simulator:

```bash
source tt_metal/tt-llk/tests/.venv/bin/activate
export TT_METAL_SIMULATOR=~/sim/libttsim_bh.so
export TT_METAL_DISABLE_SFPLOADMACRO=1   # ttsim does not implement SFPLOADMACRO
cd tt_metal/tt-llk/tests/python_tests
pytest -v --run-simulator --timeout=300 \
  "test_eltwise_unary_datacopy.py::test_unary_datacopy[formats:Float16_b->Float16_b-dest_acc:No-num_faces:4-tilize:No-input_dimensions:[64, 64]]"
```

Non-obvious gotchas:
- `--run-simulator` is required; without `TT_METAL_SIMULATOR` set the suite
  exits with an error.
- **`-k` filtering is a trap here.** Every parametrized test id contains the
  substring `tilize` (as `tilize:No`/`tilize:Yes`), so the smoke-test example in
  `TTSIM.md` (`-k "Float16_b and not tilize"`) deselects *all* tests. Prefer
  running explicit node ids (as above). Also note `-k` rejects the `>`, `:`,
  and `,` characters that appear in ids, so you cannot paste a full id into `-k`.
- ttsim is slow-dispatch only and substantially slower than silicon; use it for
  correctness, not performance numbers. Unmodeled opcodes surface as
  `UnimplementedFunctionality` from the simulator (a ttsim ISA gap, not a test
  bug).

### Linting

Canonical lint is pre-commit (`.pre-commit-config.yaml` at the repo root, plus
LLK-specific hooks in the same file; `tt_metal/tt-llk/` has its own config too).
`pre-commit` is installed in the tt-llk venv.

Gotcha: `pre-commit install` fails in the cloud VM with *"Cowardly refusing to
install hooks with `core.hooksPath` set"* (the agent sets `core.hooksPath`).
This is harmless — do **not** unset `core.hooksPath`. Just run hooks directly
without installing the git hook:

```bash
source tt_metal/tt-llk/tests/.venv/bin/activate
pre-commit run --files <paths...>   # or: pre-commit run --all-files (slow on this monorepo)
```

The first `pre-commit run` downloads hook environments (clang-format, black,
isort, autoflake, pylint, yamllint, codespell, gersemi); these are cached in the
snapshot afterward.

### Building full tt-metal (optional, hardware-bound)

`./install_dependencies.sh` + `./build_metal.sh` + `./create_venv.sh` build the
full C++/Python stack (needs Clang, Ninja, submodules, SFPI). This is a large
build and, without a device, the resulting `ttnn`/models still cannot execute
kernels. Only pursue it if the task specifically needs the C++ build to compile.
