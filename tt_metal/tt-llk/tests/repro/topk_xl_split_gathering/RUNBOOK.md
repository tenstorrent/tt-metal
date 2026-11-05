# Runbook: waveform A/B of the topk_xl split failure under Blackhole instruction gathering

This runbook is written for a Claude Code session on a machine that can run the Blackhole waveform (VCS)
simulator. The goal is **two waveform dumps of the same test**, one with gathering on and one with it off.
A person can then diff them and find where the gathering-on run diverges.

Follow the steps in order. Do not improvise around a failed check: stop and report.

## 1. What this test is

`tt_metal/tt-llk/tests/python_tests/test_topk_xl_split_gathering_repro.py` runs one case of the topk_xl
split kernel (`tests/sources/topk_xl_split_test.cpp`):

- K=512, 2 chunks, a last-chunk tail of 8 elements, 1 row, ascending last-chunk rebuild.
- Each chunk's SFPU work is split between the MATH (TRISC1) and PACK (TRISC2) threads, which hand off
  with Tensix semaphores.
- The result is checked against the golden top-K. No reference kernel is run, so a simulator run stays short.

How gathering is selected:

- On is the branch default.
- Off is `TT_METAL_DISABLE_GATHERING=1`.
- Both builds come from the same kernel source.
- The off build also turns gathering off at boot. It goes to `<build-root>/no-gathering/...`.

### What is already established on Blackhole silicon (p150, bh-35, 2026-11-04)

| Arm | Result | Signature printed by the test |
|---|---|---|
| gathering on | FAIL, 3 of 3 runs (deterministic) | `gathering=on lanes=512 missing=[516, 517] extra=[3, 15] pair_errors=0` |
| gathering off | PASS, 3 of 3 | `gathering=off lanes=512 missing=[] extra=[] pair_errors=0` |

In words: with gathering on, two chunk-1 survivors (indices 516 and 517) are lost, and two smaller
chunk-0 elements (indices 3 and 15) are kept instead. Every returned value still matches its index. So the
merge/rebuild selection goes wrong; nothing is corrupted in place.

Experiments already done (do not repeat them; they constrain where to look):

- **Only the PACK thread's gathering matters.**
  - Gathering off in PACK alone: the test passes.
  - Gathering off in MATH alone, or in UNPACK alone: the same failure.
- **Timing between the threads does not matter.**
  - Delays of 1..1000 loop iterations in MATH or UNPACK leave PACK's code byte-identical and give a
    word-for-word identical wrong output.
  - With gathering off, NOP-delay perturbation of every safe instruction of every thread found no failure:
    ~180k variants, 10 repeats each, delays 0..100.
- **Every replay record in PACK is bracketed, has the right length, and loads and executes** (`Exec=1`).
- **The failure depends on where PACK's code sits in memory.**
  - Moving PACK's helper functions `record_stride2_len8_loads`, `record_stride2_len16_loads` and
    `_topk_xl_split_sort_head_512_` by 4..64 bytes keeps the failure.
  - Moving them by 128 bytes or more makes it pass.
  - Adding one RISC instruction before every gathering disable also makes it pass.
- **Consequence: any change to the code this kernel compiles may hide the bug.** That includes debug
  prints, extra checks and padding. That is why step 3 fingerprints the ELFs.

## 2. Setup

- **Branch:** `amahmud/bh-gathering-topk-split-repro`. Use it as is.
- **SFPI toolchain:** must be the version in `tt_metal/tt-llk/tests/sfpi-version` (7.94.0). A different
  compiler gives a different layout, and the fingerprints in step 3 will not match.
  - Install it with `tt_metal/tt-llk/tests/setup_testing_env.sh`, or symlink an existing install to
    `tt_metal/tt-llk/tests/sfpi`.
- **Python env:** `tt_metal/tt-llk/tests/.venv` with `tests/requirements.txt`.
- **Simulator setup:** use the internal "Run LLK tests on the simulator" page and the machine's own
  simulator install. Claude: your memory notes `llk-waveform-vcs-not-versim`, `lcm-gathering-waveform-ab`
  and `fsdb-naming-convention` have the machine-specific paths and the traps.
  - The harness needs `TT_UMD_SIMULATOR_PATH` (or `TT_METAL_SIMULATOR`) pointing at the waveform
    simulator's build directory.
  - It also needs `NNG_SOCKET_ADDR` and `NNG_SOCKET_LOCAL_PORT`.

## 3. Build both arms and check the fingerprints (no simulator needed)

```bash
cd tt_metal/tt-llk/tests && source .venv/bin/activate && cd python_tests
export CHIP_ARCH=blackhole
export RUNNER_TEMP=$PWD/../repro-build-on
pytest --compile-producer -q test_topk_xl_split_gathering_repro.py
export RUNNER_TEMP=$PWD/../repro-build-off
TT_METAL_DISABLE_GATHERING=1 pytest --compile-producer -q test_topk_xl_split_gathering_repro.py
python ../repro/topk_xl_split_gathering/elf_fingerprint.py $PWD/../repro-build-on/tt-llk-build
python ../repro/topk_xl_split_gathering/elf_fingerprint.py $PWD/../repro-build-off/tt-llk-build
```

The output must match these lines exactly. They are the executable-section hashes of the ELFs that
failed and passed on silicon:

```
gathering {'TOPK_XL_TAIL_ELEMENTS': '8', 'ASCENDING': 'true'} unpack=0f2b65815691c217 math=7a52d8f1d7d49fc8 pack=8311648c0258c5e2
no-gathering {'TOPK_XL_TAIL_ELEMENTS': '8', 'ASCENDING': 'true'} unpack=690057a2cee42165 math=6f17ba56a7ad5a89 pack=3da67c28b2290965
```

If `pack=` differs, stop. The build is not the one that reproduces. Report the sfpi version and the hash,
and do not run the simulator.

Keep the two build trees. Step 4 runs from them with `--compile-consumer`, so the simulator executes exactly
the fingerprinted ELFs.

## 4. Run each arm once under the waveform simulator

Rules (each one has cost a multi-hour run before):

- **One arm per simulator session, each in its own empty working directory.** The simulator writes its
  dump into the server's current directory and overwrites the previous one.
- **Never SIGKILL the simulator** (`kill -9`, `pkill -9`, a harness hard timeout): that leaves an invalid,
  unwrapped dump. The clean exit is the harness's own `exit` on the server's stdin; SIGTERM is the fallback.
- **Raise the harness timeouts** so the harness neither gives up on a slow simulated run nor kills the
  simulator while it is still writing the dump:
  - `TT_LLK_SIMULATOR_TIMEOUT=7200`: seconds per device operation, default 600.
  - `TT_LLK_SIM_EXIT_TIMEOUT_S=1800`: seconds to wait for a clean simulator exit before SIGTERM, default 30.
  - Both are added on this branch.
- Run arm OFF first. It must pass, which proves the simulator setup. Then run arm ON.
- Do not run anything else in the same simulator session. Several tests in one session go into one waveform.

```bash
# Arm OFF (expected: PASS)
mkdir -p ~/topk-gath-sim/off && cd ~/topk-gath-sim/off
( cd <repo>/tt_metal/tt-llk/tests/python_tests && \
  CHIP_ARCH=blackhole RUNNER_TEMP=<repo>/tt_metal/tt-llk/tests/repro-build-off TT_METAL_DISABLE_GATHERING=1 \
  TT_LLK_SIMULATOR_TIMEOUT=7200 TT_LLK_SIM_EXIT_TIMEOUT_S=1800 \
  pytest -s -p no:cacheprovider --run-simulator --compile-consumer --timeout 0 \
    test_topk_xl_split_gathering_repro.py ) 2>&1 | tee pytest.log

# Arm ON (expected on silicon: FAIL with the signature above)
mkdir -p ~/topk-gath-sim/on && cd ~/topk-gath-sim/on
( cd <repo>/tt_metal/tt-llk/tests/python_tests && \
  CHIP_ARCH=blackhole RUNNER_TEMP=<repo>/tt_metal/tt-llk/tests/repro-build-on \
  TT_LLK_SIMULATOR_TIMEOUT=7200 TT_LLK_SIM_EXIT_TIMEOUT_S=1800 \
  pytest -s -p no:cacheprovider --run-simulator --compile-consumer --timeout 0 \
    test_topk_xl_split_gathering_repro.py ) 2>&1 | tee pytest.log
```

Adjust these to the machine's simulator flow:

- Where the server is started.
- Which working directory the dump lands in.
- The NNG variables.

Keep the build trees, the env vars above and the one-arm-per-session rule unchanged.

## 5. After each arm: verify and name the dump

1. Check for exactly one wrapped dump and no fragments. Fragments mean the wrap failed and the run must be
   repeated.
2. Record the simulator log line with the final simulation time, and the test's printed `gathering=...` line.
3. Rename the dump and keep the simulator log beside it under the same stem:
   - `topk-split-gath_gathering-off_k512-c2-t8-asc_<YYYYMMDD>.fsdb`
   - `topk-split-gath_gathering-on_k512-c2-t8-asc_<YYYYMMDD>.fsdb`
4. Copy both ELF trees, both `pytest.log`s and the simulator logs next to the dumps. The ELFs are needed to
   map PCs to code.

How to classify each arm:

- **OFF must PASS.** If it fails, the simulator setup is wrong. Stop and report.
- **ON:**
  - A FAIL with the silicon signature means the simulator reproduces the bug. That is the main result.
  - A PASS is also a result: the simulator does not reproduce it. Report it. Do not change the code to
    force it.
  - A hang or a simulator error: report the log tail. The dump is still useful if it wrapped.

## 6. What to compare (software view)

The bug is in the PACK thread (TRISC2) with gathering on. Compare the two dumps around PACK's work:

- **First, the stream of Tensix instructions PACK delivers to the coprocessor, in order, with their PCs:**
  - from PACK's first token take after the pack init,
  - to the end of the epilogue (`split_indices`).
- Find the first point where the ON stream differs from the OFF stream. That means a missing, repeated,
  reordered or substituted instruction, or the same instruction at a different PC.
- Gathering bundles 2-4 adjacent Tensix instructions only when code runs a second time. Pay special
  attention to code PACK executes more than once:
  - `pack_slot` runs 3 times.
  - The helpers below run once per chunk.
- If the instruction streams are identical, compare SFPU register-file / Dst writes for the same PCs.
  - That would mean the instructions are right but their timing differs, which is a different class of bug.

PACK function map of the validated gathering-on `pack.elf` (`pack=8311648c0258c5e2`):

| Function | Start | Size |
|---|---|---|
| `main` | 0x100c0 | 0xd8 |
| `topk_xl_split::pack_slot<512>` | 0x1051c | 0x1330 |
| `run_kernel` (PACK) | 0x1184c | 0x6ac |
| `topk_xl_fused_macro::record_stride2_len8_loads` | 0x11ef8 | 0xd0 |
| `topk_xl_fused_macro::record_stride2_len16_loads` | 0x11fc8 | 0xd0 |
| `_topk_xl_split_sort_head_512_<true>` | 0x12098 | 0x668 |

Regenerate it, and the list of gathering brackets, from the copied ELF:

```bash
OD=tt_metal/tt-llk/tests/sfpi/compiler/bin/riscv-tt-elf-objdump
$OD -t <tree>/sources/topk_xl_split_test.cpp/<variant>/elf/pack.elf | grep " F .text"
$OD -d <tree>/sources/topk_xl_split_test.cpp/<variant>/elf/pack.elf | grep -n "csr[sc]\s*0x7c0\|ttreplay"
```

Notes on that disassembly:

- `csrs 0x7c0` with bit 18 set (`lui rX,0x40`) disables gathering.
- `csrc 0x7c0` with bit 18 re-enables it.
- `ttreplay a,b,1,1` is a replay record (load + execute).

## 7. Report back

- Both arms:
  - pass, fail or hang;
  - the printed `gathering=` line;
  - the final simulation time;
  - wall-clock time;
  - dump name, size and location.
- The fingerprint lines from step 3.
- If you got as far as step 6: the PC and instruction of the first divergence, and which function it is in.
