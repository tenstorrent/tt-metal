# Wormhole packer-phase repro (experiment branch, not for merge)

One Wormhole kernel, two states. The pack thread packs one Float16 32x32 tile from DEST to L1
256 times (`perf_math_matmul`, PACK_ISOLATE, LoFi, DestSync Full, loop factor 256). Unpack and
math do no work and spin until packing is over.

**Experiment 1, same code.** The math RISC-V does one L1 load (`lw` from its own profiler buffer)
at spin iteration `K` of a fixed 3,000-iteration loop. `K` is written by the host to L1
`0x16AFE0` before the launch and read in INIT, so `unpack.elf`, `math.elf` and `pack.elf` are the
same for every `K`. Silicon (bgd-lab-08, n150): `K` = 0, 225, 250, 400 give 9,071-9,072 cycles
(fast); `K` = 100..200 and 275..375 give 9,293-9,325 (slow).

**Experiment 2, a code change that does no work.** `NOPS` 4-byte nops in the Wormhole
`_llk_pack_init_`, no load. Silicon, peers idle: N=0 9,070, N=1 9,317, N=2 9,064, N=3 9,071,
N=4 9,316. Peers ending during the loop (`TAIL=0`): 11,658-11,674 for every N.

## Where the code is

| What | File |
|---|---|
| The load at runtime `K` (`REPRO_INJ == 15`) and the read of `K` in INIT | `sources/math_matmul_test.cpp`, math thread |
| The host write of `K` before each launch | `python_tests/helpers/perf/core.py` (`REPRO_RT`) |
| The nops (`LLK_PACK_INIT_NOPS`) | `tt_llk_wormhole_b0/llk_lib/llk_pack.h`, `_llk_pack_init_`; flag set in `python_tests/helpers/test_config.py` (`REPRO_PACK_NOPS`) |
| Idle peers, pack-loop position | `sources/math_matmul_test.cpp` (`REPRO_TAIL`, `REPRO_PAD`) |
| Config selection, knobs from the environment | `python_tests/perf_math_matmul.py` |
| Versim support | `python_tests/helpers/target_config.py`, `device.py` (from `lpremovic/versim-harness`), `VERSIM.md` |
| BRISC: 1 us mailbox poll in simulators, off L1 during a kernel | `helpers/src/brisc.cpp` |

## Run it

On a Linux machine with `/proj_sw` (an IRD reservation, for example), from a checkout of this
branch with the LLK venv and SFPI linked into `tt_metal/tt-llk/tests` (`.venv`, `sfpi`):

```bash
cd tt_metal/tt-llk/tests
./repro_phase.sh hw  K=0            # silicon, no load        -> about 9,071 (fast)
./repro_phase.sh hw  K=100          # silicon, load at 100    -> about 9,325 (slow)
./repro_phase.sh sim K=0            # Versim, same ELF        (about 30 min)
./repro_phase.sh sim K=100
./repro_phase.sh sim K=100 VCD=1    # keep the waveform
./repro_phase.sh hw  NOPS=1         # experiment 2
```

Each run prints the INIT, TILE_LOOP and KERNEL cycles and leaves `run.log`,
`perf_math_matmul.csv` and, for Versim, the Versim log (and VCD) in `/tmp/repro_phase/<name>/`.
TILE_LOOP is the measured pack loop. `grep -c "ASSERTION FAILED" versim_*.log` must be 0.

## Check that the code is the same for every K

The ELF files differ only in debug information (each run builds in its own directory). The code is
identical; compare the `.text` sections:

```bash
B=sfpi/compiler/bin
for t in unpack math pack; do
  for k in 0 100; do
    e=$(find /tmp/repro_phase/hw_k${k}_n0_t6000_l256_i15/build -name $t.elf | head -1)
    $B/riscv-tt-elf-objcopy -O binary --only-section=.text "$e" /tmp/text_$k.bin
  done
  cmp /tmp/text_0.bin /tmp/text_100.bin && echo "$t: same code"
done
```

## Results: silicon against Versim (2026-10-05)

Silicon: the n150 in IRD machine bgd-lab-08. Versim: `/proj_sw/user_dev/ndivnic/tt-umd-simulators/build/versim-wormhole-b0`.
For every build, the `.text` sections of `unpack.elf`, `math.elf` and `pack.elf` are identical between the silicon
and the Versim run. TILE_LOOP cycles for 256 tiles; the CSV files are in `repro_phase_results/`.

### Experiment 2: N nops in `_llk_pack_init_` (commit d8bed466332, no injected load)

| N | inner `ttpacr` | silicon, peers idle | Versim, peers idle | silicon, peers end | Versim, peers end |
|--:|---|--:|--:|--:|--:|
| 0 | 0xf1d4 | 9,070 | 9,070 | 11,674 | 11,664 |
| 1 | 0xf1dc | 9,317 | 9,313 | 11,669 | 11,418 |
| 2 | 0xf1e0 | 9,064 | 9,070 | 11,664 | 11,653 |
| 3 | 0xf1e4 | 9,071 | 9,070 | 11,668 | 11,664 |
| 4 | 0xf1e8 | 9,316 | 9,317 | 11,658 | 11,655 |

### Experiment 1: one L1 load at runtime spin iteration K, one ELF (commit 81f48070eda)

| K | silicon | Versim |
|--:|--:|--:|
| 0 (no load) | 9,070 | 9,070 |
| 100 | 9,323 | 9,323 |
| 125 | 9,320 | 9,320 |
| 150 | 9,317 | 9,317 |
| 175 | 9,314 | 9,314 |
| 200 | 9,070 | 9,070 |
| 225 | 9,309 | 9,309 |
| 250 | 9,306 | 9,306 |
| 275 | 9,303 | 9,303 |
| 300 | 9,300 | 9,300 |
| 325 | 9,297 | 9,297 |
| 350 | 9,294 | 9,294 |
| 375 | 9,070 | 9,070 |
| 400 | 9,289 | 9,289 |

All 14 values match to the cycle. The same three ELF files are fast with no load, slow with a load at most
moments (+1 cycle for each tile still to be packed), and fast again when the load lands at K = 200 or 375.
