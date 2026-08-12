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
cave start              ->  NOP          \
                            NOP           |  max_delay words (default 100)
                            ...          /
displaced_instruction   ->  <the instruction that used to live at the site>
ret                     ->  jal x0, site+4
end                     ->  first byte past this layout
```

The ELF only reserves `[start, limit)` (`__kernel_cave_end` or the L1 tail).
`ret` is not a symbol in that ELF — it is the trampoline word the injector
writes inside the reservation. `end` is how much of that reservation this
layout uses; `limit` is how much the linker set aside.

The site itself is overwritten with `jal x0, <somewhere in the cave>`. Executing
`n` fillers is just aiming that jump `n` words short of the displaced instruction,
so changing the delay costs one word write. Delay 0 lands directly on the displaced
instruction, which makes it a free control: it still detours, so if delay 0
passes and delay 8 fails, the fillers did it, not the jump.

A sweep walks the delays of one site back to back, so the injector only writes
what actually changed: the filler run when the nop changes, the displaced word and
its jump back when the site changes, and always the one word at the site. The
common step — next delay, same site — is a single word. Nothing is rebuilt and
nothing is reloaded, which is what makes sweeping all 100 counts affordable.

## Which nop

A nop only widens a window if it retires on the unit whose timing is in
question. `TTI_NOP` advances the RISC and the front-end together, so it cannot
open a RISC-MMIO-vs-unpacker gap at all; `risc_nop` is the one that can, because
it costs the RISC a cycle while the backend keeps draining its instruction FIFO.
The two therefore perturb in opposite directions and are both worth trying at
the same site.

| filler | word | retires on |
| --- | --- | --- |
| `tti_nop` | `0x08000000` | RISC + front-end |
| `risc_nop` | `0x00000013` | RISC alone (RV32 `addi x0, x0, 0`) |
| `sfpnop` | `0x3C000002` | SFPU |
| `unpacr0` | `0x0C000009` | unpacker 0 / SrcA |
| `unpacr1` | `0x0E000009` | unpacker 1 / SrcB |

Only pure `UNP_NOP` mode is ever used. `ZEROSRC`, `SET_DVALID` and `NEGINFSRC`
have side effects on the source registers and would corrupt data rather than
delay it.

Under the `auto` policy every site is tried with `tti_nop` and `risc_nop`, plus
whichever unit-retired nop applies: an unpacker nop on unpack-thread sync sites,
the SFPU nop on SFPU sites. So an unpack sync site costs three variants per
delay (four when the unpacker census came back empty) and a math site two, or
three at an SFPU site.

`risc_nop` is the weakest filler per count — one core cycle, against an
`UNPACR_NOP` that occupies the unpacker itself — and the instruction FIFO can
absorb the first few outright while the RISC is still running ahead of it. Expect
it to fail in a cliff rather than a band, and if a site looks immune to it,
suspect the delay range before concluding the site is clean.

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
./focus.sh --sites unpack:3 --delays 8,16 'test_x.py::test_y[params]'
```

[FOCUS.md](FOCUS.md) is the full reference for the depth runner; its flags are
aliases for the variables below.

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
| `TTNOP_REPEATS` | runs per variant (10 under `focus.sh`); >1 adds the delay-0 control |
| `TTNOP_DRIFT` | compare each variant's output to the baseline's, default on |
| `TTNOP_REPORT_DIR` | one per machine when sharding |
| `TTNOP_VERBOSE` | print each detour as it is applied |

## Drift: the failures that still pass

A delay that shifts a mantissa bit is not a failure to this suite. `passed_test`
gates on per-element tolerance and a PCC floor (0.99, lower for the block floats),
and skips PCC entirely for MX formats and for goldens with no signal — so a
variant can change the answer and still go green.

Drift asks the question the golden cannot: **did the output change at all** from
the clean run. That needs no threshold, because the baseline carries the same
approximation error as the variant, and for a race-free kernel the answer is "no
change" whatever the format. Any difference is recorded as a `drift` tag, with the
PCC *against the baseline* and the number of elements that moved.

It costs the stimulus lottery. To make the two runs the same problem, the RNG is
rewound to the baseline's state before every variant, so all variants share one
stimulus set — `TTNOP_DRIFT=0` restores the rolling stream instead, at the price
of having nothing to compare. Freezing also makes a breadth finding exactly
reproducible, which the reproduce line in `report.md` could not promise before.

Two guards keep it honest, because a case that cannot reproduce its own output
would otherwise report drift on every variant:

- The harness already knows which variants are not bit-reproducible —
  `TestConfig._bit_exact_unsupported_reason()` names l1_acc, coverage builds and
  deliberately-undefined state — and drift takes its word for it.
- Every other case runs its body one extra time, same stimuli and no detour, and
  has to reproduce itself before any drift verdict is believed. That control also
  proves the rewind worked, so a test drawing from outside torch's global RNG
  cannot invent findings.

A drift-only case stays **green**: the variant passed the test's own golden, so the
finding lives in `report.md` and `failures.jsonl` rather than in the exit code.

A case that loses a variant goes red and names it, so a sweep reads like an
ordinary pytest run:

```
>> delays=1-100 threads=unpack,math,pack sites=sync filler=auto
>> [3/3] sweeping
⨯ test_bcast.py::test_unpack_bcast[...] 47 perturbation(s) failed: unpack ATGETM@0x05550 n=54 unpacr1 mismatch (+46 more)
>> 69 recorded variant(s) (22 drift) -> reports/report.md
```

Output is `failures.jsonl` plus a `report.md` rendered from it, written only when
there is something to report. Each site lists the exact counts that broke it, its
addr2line inline chain (resolved during the sweep, while the ELF still exists) and
a copy-paste `focus.sh` line that re-runs just those counts as a rate.

A supervised run also writes `junit.xml`, covering every case it reached. The
supervisor builds it rather than pytest, because pytest writes its junit file at
session end and a wedged core gets that session killed — every result from that
attempt would be missing from exactly the runs worth reading. It comes instead
from a per-case log the workers append as they go, which survives the kill and
spans the attempts a resume is split across, and it carries the hung cases as
failures: the worker that would have reported one is stuck in a device read.

`--device-jobs N` fans the device phase out over N Tensix cores (the harness maps
xdist `gwN` to core `N/8, N%8`). Default is 8. A breadth run gives each worker
different cases; a depth run has only one, so `focus.sh` hands it to every worker
with `--dist each` and splits the variant plan between them instead. Variants call
the test body directly (`item.obj`), not `item.runtest`, so the sweep hook cannot
nest.

## Notes on the LLK harness

- Both loaders are wrapped. Suites like SDPA, pack-dest-bank and streams call
  `run_elf_files()` directly and never return a result, so their own asserts are
  the golden; a `run()` that produced no result is skipped because there is
  nothing to mismatch.
- The harness only reloads ELFs when the variant directory changes, which within
  a case it never does — so the image stays put and a variant is a couple of word
  writes. `LAST_LOADED_ELFS` is cleared once at the end of the case, because the
  cave bytes outlive the restore and the next case must not inherit them.
- Stimuli come from a global RNG seeded once per test. Under `TTNOP_DRIFT=1` (the
  default) every variant is rewound to the state the baseline drew from, so they
  share one stimulus set and their outputs are comparable. `TTNOP_DRIFT=0` lets the
  stream run on instead: different data per variant, and some races only show up on
  later draws, but no two runs can be compared.
- Variants are meant to fail, so the harness's logging is muted while one runs;
  otherwise every mismatch dumps the offending tiles in colour. The baseline pass
  runs outside that, so a genuinely broken test still says why.
- A hang is a `TimeoutError`. The finding is recorded, the case is closed, and
  the worker parks until the supervisor kills it — xdist then brings a
  replacement up on one of the card's spare Tensix, so the wedged core is simply
  never addressed again and the other seven workers keep their cases. Only a card
  with no spare cores left falls back to `tt-smi -r`. Soft reset-and-continue is
  gone: it rebooted BRISC mid-session and failed every case after the hang.

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
