<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# llk-audit — deterministic recall engine for the LLK race audits

An AST-based tool that **exhaustively enumerates the KNOWN hazard patterns** for
several LLK race-audit classes and emits a candidate list the `/*-audit` skills
consume. It is an **augmentor, never a gate**: it finds every instance of the
patterns it can encode, but a tool cannot recognize *unknown* patterns, so its
output is advisory input to the LLM/human — which still owns the verdict and the
hunt for novel hazards.

```
┌─ extractor/  (C++ / Clang libTooling) ─ parse once, emit a semantics-free FACT BASE
│     llk_extract.cpp     functions · pointer-writes (+provenance) · calls · macro expansions
│
├─ llkaudit/   (Python) ─ classify the fact base into recall candidates
│     registry.py         ← THE table that maps LLK names/signatures to meaning (edit this)
│     factbase.py         load / merge / dedup / query
│     checks/*.py         one checker per hazard class (augmentors)
│     cli.py              run checks, emit one advisory JSON envelope
│
└─ run.sh      orchestrate: extract every header for an arch → run checks
```

## Why this split

The fragile, fidelity-critical work — parsing template/macro-heavy 3-arch
headers, tracing pointer provenance, recovering macro names+args the AST discards
(`TTI_STALLWAIT`, `p_stall::TRISC_CFG`, …) — needs libTooling and only has to be
gotten right **once**. The frequently-edited work — *which* names count as *what*
— is data in `registry.py`. **When an LLK signature changes, you edit one table
in `registry.py`; you rarely touch a checker and never the C++.**

## Checks (this build)

| Check | What it recalls | Hints |
|---|---|---|
| `mmio-race` | RISC MMIO cfg/GPR write vs. consuming Tensix instruction/MOP; is an applicable ordering primitive local? | `LOCALLY_ORDERED` / `NO_LOCAL_ORDERING` |
| `cfg-word-overlap` | fields sharing one 32-bit CONFIG word (per register file) written by ≥2 threads | `CROSS_THREAD_SHARED_WORD` / `UNRESOLVED` |
| `semaphore-handshake` | mutex acquire/release imbalance; semaphore wait with no in-tree init | `MUTEX_IMBALANCE` / `WAIT_WITHOUT_INIT` |
| `reconfig-stall` | reconfig/uninit config write missing a unit-draining stall | `NO_UNIT_DRAIN` / `THCON_ONLY` |

Every finding is a **recall bucket, not a verdict**, and every check declares its
`blind_spots` in the output.

### Why these four (and not the other nine audits)
Scoped by evidence — a checker ships only where AST recall is exhaustive on a
real tt-llk pattern:
- **dataflow-cb-sync / noc-sync** — 0 CB/NoC sites in the tt-llk compute lib (they
  live in `tt_metal/hw/inc/api` + `ttnn`/`models`). A tt-llk-scoped checker finds
  nothing → dropped.
- **mailbox-sync** — exactly one functional pair in tt-llk; surface is elsewhere → dropped.
- **instruction-latency** — its surface is the SFPU files, which don't parse under
  clang (GCC vector extensions), and the verdict needs an out-of-tree version-pinned
  latency table → deferred until SFPU parsing is solved.
- **srcreg-bank-sync** — too semantic for exhaustive recall → LLM-led.

## Build & run

```bash
extractor/build.sh                 # needs Clang/LLVM >= 18 dev libs (libclang-cpp)
./run.sh wormhole                  # or blackhole | quasar   [--checks a,b] [out_dir]
python3 tests/test_checks.py       # hermetic unit tests (no clang/repo needed)
```

`run.sh` writes `out/facts.<arch>.jsonl` (fact base) and `out/audit.<arch>.json`
(advisory findings). Feed `audit.<arch>.json` to the matching `/*-audit` skill as
its pre-enumerated worklist.

## Updating when signatures change

Open `registry.py` — it is organized by concept with an `EDIT HERE` banner:
- new cfg-pointer accessor → add to `CFG_POINTER_PRODUCERS`
- new MMIO write call → `MMIO_WRITE_CALLS`
- renamed/added instruction macro → the relevant `*_SUBSTR` list
- new drain/sync function → `DRAIN_CALLS`
- new reconfig function name → `RECONFIG_FN_SUBSTR`; new latched register → `LATCHED_FIELDS`
- new semaphore/mutex wrapper → `SEMAPHORE_CALLS`

Then `python3 tests/test_checks.py` to confirm nothing regressed.

## Coverage boundaries (explicit — no silent caps)

1. **Augmentor only.** No gate, no "safe" claim, always exits 0. Green = "no new
   *known-pattern* instance," never "no bug."
2. **Interprocedural linkage is the LLM's job.** `mmio-race` consumer/guard
   association is intra-function; a caller-supplied guard shows as
   `NO_LOCAL_ORDERING`.
3. **SFPU stubbed.** `sfpi.h`/`sfpi_classes.h` are stubbed so headers that merely
   include them parse; files that structurally use `sfpi::` types fail to parse
   and are counted in `parse_errors` / logged to `out/parse.log`.
4. **cfg-word-overlap** partitions THCON vs the main config file by name prefix;
   fields that don't resolve to an ADDR32 are reported `UNRESOLVED`. Whether a
   shared word actually races (bit-disjoint masking, mutex/semaphore ordering,
   value-invariance) is deferred.
5. **Quasar** enables HW AutoTTSync; the tool still enumerates, but treat QSR
   output per that mechanism.

## Validated against ground truth

- `mmio-race` reproduces the earlier hand-validated result (WH 91: 67/24,
  BH 79: 60/19 — one BH write is `NO_LOCAL_ORDERING` because its only guard
  follows a consumer, which the "guard must precede the first consumer" rule
  correctly refuses to credit), incl. `_llk_unpack_A_`→`LOCALLY_ORDERED` and the
  `THCON_SEC0_REG3_Base_address` writes→`NO_LOCAL_ORDERING`.
- `cfg-word-overlap` finds the known shared MAIN words (ALU-format words 0/1/2 incl.
  STACC_RELU on WH; 1/2 on BH) and suppresses the THCON/main same-index false alias.
- `reconfig-stall` flags `set_packer_strides` etc. and correctly allowlists the
  latched `program_packer_destination` (`L1_Dest_addr`); exercises `THCON_ONLY` on BH.
- `semaphore-handshake` sees all ops (17 post / 20 get / 4 init / balanced mutexes
  on WH) and correctly reports no imbalance — after excluding wrapper defs + RAII.
