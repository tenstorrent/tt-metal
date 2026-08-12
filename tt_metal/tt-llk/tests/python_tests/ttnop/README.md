# ttnop — timing perturbation for Tensix kernels

Races between the three Tensix threads, and between a RISC MMIO write and the
backend unit that consumes it, only show up when relative timing shifts. This
tool shifts it on purpose: it inserts a controlled delay at a sync site and
re-runs the same test against the same golden, so a window that fails once a
month in CI fails on demand.

## How a delay gets in

Kernels are never rebuilt per delay. Each thread ELF is scanned once, and the
free tail of its L1 code region becomes a **cave**:

```
cave start ->  filler          \
               filler           |  max_delay words (default 100)
               ...             /
parked     ->  <the instruction that used to live at the site>
ret        ->  jal x0, site+4
```

The site itself is overwritten with `jal x0, <somewhere in the cave>`. Executing
`n` fillers is just aiming that jump `n` words short of the parked instruction,
so changing the delay costs one word write. Delay 0 lands directly on the parked
instruction, which makes it a free control: it still detours, so if delay 0
passes and delay 8 fails, the fillers did it, not the jump.

A sweep walks the delays of one site back to back, so the injector only writes
what actually changed: the filler run when the nop changes, the parked word and
its jump back when the site changes, and always the one word at the site. The
common step — next delay, same site — is a single word. Nothing is rebuilt and
nothing is reloaded, which is what makes sweeping all 100 counts affordable.

## Which nop

A nop only widens a window if it retires on the unit whose timing is in
question. `TTI_NOP` advances the RISC and the front-end together, so it cannot
open a RISC-MMIO-vs-unpacker gap at all.

| filler | word | retires on |
| --- | --- | --- |
| `tti_nop` | `0x08000000` | RISC + front-end (default) |
| `sfpnop` | `0x3C000002` | SFPU |
| `unpacr0` | `0x0C000009` | unpacker 0 / SrcA |
| `unpacr1` | `0x0E000009` | unpacker 1 / SrcB |

Only pure `UNP_NOP` mode is ever used. `ZEROSRC`, `SET_DVALID` and `NEGINFSRC`
have side effects on the source registers and would corrupt data rather than
delay it.

Under the `auto` policy, unpack-thread sync sites get an unpacker nop, SFPU
sites get the SFPU nop, and everything else gets the generic nop.

**Picking the unpacker by counting UNPACR in `.text` does not work.** LLK
records both SrcA and SrcB UNPACRs into the MOP/replay buffer before any runtime
branch, so the text looks dual-unpacker even when only one loads real data. The
scanner instead ORs the `CntSetMask` of every `SETADCXX`, which LLK only issues
for the unpacker(s) that actually read L1. An empty census means both are swept
rather than guessed.

A site counts as SFPU-related if its opcode is in `[0x70, 0x95]`, or if it is a
`STALLWAIT` that stalls the SFPU or waits on it — those block boundaries are
often as interesting as the SFPU ops themselves.

## Running it

Breadth, one shot per variant, shardable across machines:

```bash
./ci.sh --test test_eltwise_unary_datacopy.py --k Float16_b
./ci.sh --test test_sdpa_reinits.py --splits 4 --group 2 --report-dir reports/host2
```

Depth, one case repeated until a flaky race shows a rate:

```bash
TTNOP_SITES=unpack:3 TTNOP_DELAYS=8,16 TTNOP_REPEATS=50 \
    ./focus.sh 'test_x.py::test_y[params]'
```

`TTNOP_DELAYS` takes ints and `lo-hi` ranges, comma separated: `1-100` (the
default), `1,5,10,20,40,60,80,100` for a coarse probe, `1-8,16,32` for a mix.
Sweep every count unless you have a reason not to — the races found so far live
in bands tens of counts wide, and sampling powers of two walks past most of them.

| variable | meaning |
| --- | --- |
| `TTNOP_SITE_MODE` | `sync` (default) or `all` |
| `TTNOP_THREADS` | default `unpack,math` (add `pack` to include it) |
| `TTNOP_DELAYS` | filler counts, default `1-100` |
| `TTNOP_MAX_DELAY` | cave capacity, default 100 |
| `TTNOP_FILLER` | `auto`, a name from the table, or a raw word |
| `TTNOP_SITES` | e.g. `unpack:3,math:7` |
| `TTNOP_REPEATS` | runs per variant; >1 adds the delay-0 control |
| `TTNOP_REPORT_DIR` | one per machine when sharding |
| `TTNOP_VERBOSE` | print each detour as it is applied |

A case that loses a variant goes red and names it, so a sweep reads like an
ordinary pytest run:

```
>> delays=1-100 threads=unpack,math,pack sites=sync filler=auto
>> [3/3] sweeping
⨯ test_bcast.py::test_unpack_bcast[...] 47 perturbation(s) failed: unpack ATGETM@0x05550 n=54 unpacr1 mismatch (+46 more)
>> 47 failing variant(s) -> reports/report.md
```

Output is `failures.jsonl` plus a `report.md` rendered from it, written only when
there is something to report. Each failing site lists the exact counts that broke
it, its addr2line inline chain (resolved during the sweep, while the ELF still
exists) and a copy-paste `focus.sh` line that re-runs just those counts as a rate.

`--device-jobs N` fans the device phase out over N Tensix cores (the harness maps
xdist `gwN` to core `N/8, N%8`). Default is 8. Variants call the test body
directly (`item.obj`), not `item.runtest`, so the sweep hook cannot nest.

## Scanning by hand

```bash
make && ./scan --mode sync /tmp/tt-llk-build/<test>/<hash>/elf/unpack.elf
python3 scanner.py /tmp/tt-llk-build/<test>/<hash>/elf/*.elf
```

## Notes on the LLK harness

- Both loaders are wrapped. Suites like SDPA, pack-dest-bank and streams call
  `run_elf_files()` directly and never return a result, so their own asserts are
  the golden; a `run()` that produced no result is skipped because there is
  nothing to mismatch.
- The harness only reloads ELFs when the variant directory changes, which within
  a case it never does — so the image stays put and a variant is a couple of word
  writes. `LAST_LOADED_ELFS` is cleared once at the end of the case, because the
  cave bytes outlive the restore and the next case must not inherit them.
- Stimuli come from a global RNG seeded once per test. Variants keep that stream
  going (no re-seed): some races only show up on later draws, and re-seeding to a
  fixed value made every variant share one stimulus set.
- Variants are meant to fail, so the harness's logging is muted while one runs;
  otherwise every mismatch dumps the offending tiles in colour. The baseline pass
  runs outside that, so a genuinely broken test still says why.
- A hang is a `TimeoutError`, never a substring match. Recovery soft-resets and
  continues; if the reset does not take, the finding is recorded first and the
  sweep stops rather than reporting a dead device as a wall of races.

## Metal

`./metal.sh` points the same sweep at a **ttnn op test**. The scan, sites, fillers
and cave arithmetic are shared because they are Tensix-level; two things differ.

**Where the poke lands.** The LLK harness loads each ELF once, so poking L1 sticks.
Metal rewrites all three TRISC binaries into L1 from a host-side `ll_api::memory`
on *every* slow-dispatch launch, with no "already configured" guard, so an L1 poke
is erased before the kernel runs — and there is no host-visible seam between
"binaries written" and GO. So the Metal backend pokes the **host image** and lets
metal re-apply the perturbation on each launch, through the ctypes seam in
`libttnop_metal.so`. A variant stays a few word writes: no JIT recompile, no device
reopen.

**Where the cave lives.** A metal kernel's `.text` is XIPified — linked at 0 and
packed into a rotating kernel-config ring — so there is no stable gap after it and
no `_etext`. `main.ld` therefore reserves the cave *inside* `.text`, which is
strictly better: both jumps of the detour are PC-relative within the image, so
nothing has to resolve `kernel_config_base`, the ring slot stops mattering, and
every core running the op gets the same perturbation.

```bash
make metal_cave                 # reserve the cave, once
./metal.sh path/to/test_op.py
TTNOP_SITES=unpack:3 TTNOP_DELAYS=40-60 TTNOP_REPEATS=200 \
    ./metal.sh 'path/to/test_op.py::test_case[params]'
```

`make metal_cave` regenerates the six WH/BH TRISC linker scripts with
`TTNOP_CAVE_BYTES` defined and drops the JIT kernel cache — the kernel hash is
`build_key + hlk_desc + compute_hash`, so it cannot see a linker change and stale
kernels would otherwise be reused. Re-run it after rebuilding the `hw_toolchain`
target, which regenerates those scripts without the cave. `make metal_cave_disable`
puts them back. Nothing is reserved in a normal build.

| variable | meaning |
| --- | --- |
| `TTNOP_METAL` | set by `metal.sh`; selects the Metal backend |
| `TTNOP_METAL_KERNEL` | regex picking the compute kernel, e.g. `eltwise_sfpu` |

The default kernel is the compute kernel with the most recently written XIP dump,
which is the op that just ran. Set `TTNOP_METAL_KERNEL` whenever a test drives more
than one op, or the wrong kernel gets perturbed.

### Differences worth knowing before reading a report

- **Slow dispatch is required** (`metal.sh` sets it). Fast dispatch stages the image
  into a DRAM `kernels_buffer` on the first enqueue and relays it from there, so
  later host-image writes are invisible. It is the mode you want anyway: it removes
  command-queue overlap, so a `(site, delay)` pair measures kernel timing rather
  than host scheduling. A perturbation is the *same size* as the original image, so
  the fast-dispatch port is a DRAM re-write plus a prefetcher invalidate rather than
  a re-pack — but those hooks live in tt-metal and do not exist yet.
- **A hang is not recoverable in-process.** It takes the dispatcher with it, not
  just the Tensix, so the LLK soft-reset has no equivalent. The sweep records the
  finding and stops; reset with `tt-smi -r` before continuing.
- **A site is a site × the whole core grid.** One image serves every core running
  the op, so all of them are perturbed identically. That is usually what you want,
  but it means a sweep cannot move *relative* timing between two cores.
- **`.xip.elf` must exist.** The scan reads the post-XIP dump metal writes beside
  each kernel ELF, because XIPify rewrites text-targeting `LUI` into `AUIPC` — the
  pre-XIP words would both mis-report site contents and let a now-PC-relative
  instruction past the relocatability filter. Do not set
  `TT_METAL_DISABLE_XIP_DUMP`.

Binding refuses to proceed if it *constructed* an image rather than reusing
metal's — `get_risc_binary` keys its cache on the path string, so a mismatched
spelling would build a second image nobody launches and the sweep would read 0%
everywhere. Constructing one rewrites the `.xip.elf`, so its mtime settles it;
`Injector.arm` then checks each site word as it patches. The address arithmetic
has an offline check that needs no device:

```bash
python3 metal.py
```
