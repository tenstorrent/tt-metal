# Router bring-up tools

> **Temporary: drop this directory before the branch is merged** (as `tt_metal/fabric/WH_BRINGUP.md`).

These tools come from the Blackhole performance study of the new fabric router. They are packaged for the Wormhole
bring-up, to chase perf regressions against main. They are scripts, not tests: nothing here runs in CI.

Scripts take paths as arguments. Only `ab_run.sh` and `pc_sampler.py` touch a device, and only when you run them.

**Toolchain.** The ELF tools find the RISC-V binutils the way the JIT build does:
1. `$TT_METAL_HOME/runtime/sfpi`, or this tree's `runtime/sfpi`;
2. then `/opt/tenstorrent/sfpi`;
3. then `riscv-tt-elf-objdump` on `PATH`.

Override it with `ROUTER_TOOLS_TOOLCHAIN=<sfpi>/compiler/bin/riscv-tt-elf-`.

**Router images.** The JIT names them per arch:
- Wormhole: `erisc`. One ERISC per Ethernet core; the router runs `single::sender_pass` and `single::receiver_pass`
  in one loop.
- Blackhole: `active_erisc` (ERISC0, E0) and `subordinate_active_erisc` (ERISC1, E1).

The kernel is `fabric_erisc_router` (`--kernel` changes it). Its images sit in a kernel cache at
`<cache>/.../kernels/fabric_erisc_router/<config hash>/<image>/<image>.elf`. Set `TT_METAL_CACHE` to keep a cache
per run.

## Status on Wormhole

| Tool | Answers | Wormhole |
| --- | --- | --- |
| `ab_run.sh` + `ab_report.py` | main vs new router, cycles/packet per row, with pass noise | WH-ready: no arch assumptions; the clock is recovered per run or given. Not yet run on a WH machine. |
| `offline_build.py` | rebuild the router's ELFs from the JIT's logged commands, for a source or flag variant | WH-ready: arch-neutral, checked on a Blackhole capture. Not yet run on WH logs. |
| `elf_size.py` | code size and static stack depth per image | WH-ready. "free stack" needs `--entry-sp erisc=<WH value>`, which is not known here. |
| `codegen_diff.py` | how many configs' code changed, and by how much | WH-ready |
| `line_compare.py` | one function's instructions per source line, base vs new | WH-ready. On `erisc` images the default function is `single::sender_pass`; use `--function single::receiver_pass` for the other. |
| `loopvars.py` | which hot values the router loop keeps on the stack (`--headroom`: what a change spills) | WH-ready for `erisc` images (the `single::` passes); checked on Blackhole images only |
| `pc_sampler.py`, `pcs_lines.py`, `pcs_profile.py` | where an ERISC spends its cycles, from PC samples over the debug bus | **Blackhole only.** See "PC sampling: what a Wormhole port needs". |
| hang capture / decode (not shipped) | router state after a hang | **Blackhole only.** It reads Blackhole's L1 layout and the two ERISCs' state; not ported. |

## 1. Main vs new on the same rows: `ab_run.sh`, `ab_report.py`

```
ab_run.sh --tree-a <main tree> --tree-b <new tree> --out <dir> [--name-a main] [--name-b new] \
    [--env-b "TT_METAL_FABRIC_OPT_LEVEL=O3"] <bandwidth yaml> [<yaml> ...]
ab_report.py <dir> [--clock-mhz <AICLK>] [--noise 1.0] [--csv table.csv]
```

- **What it runs.** Each tree's own `build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric` (build both first), with
  `TT_METAL_HOME` set to that tree and one kernel cache per arm under `<dir>`. Every run has
  `TT_FABRIC_NO_BW_TELEMETRY=1` (router bandwidth telemetry off), `TT_METAL_CLEAR_L1=1` and
  `TT_METAL_DEVICE_PROFILER=0`.
- **Order.** Two passes, the arm order reversed in the second: pass a runs A then B per yaml, pass b runs B then A.
  A drift over time then shows as a pass difference, not as an arm difference.
- **On failure.** A failed or timed-out run stops the script; reset the device before rerunning.
- **The metric.** cycles/packet = clock / the summary CSV's `avg_packets_per_s`.
- **The clock.** By default it is recovered per run from the per-sender CSV (`packets_per_second × cycles /
  (num_packets × total_traffic_count)`), so it is the clock test_tt_fabric itself used. Pass `--clock-mhz` to force one.
  Without a per-sender CSV the report stops and asks for it. Blackhole runs at 1350 MHz; read the Wormhole device's
  AICLK, don't assume it.
- **Reading it.** `*` marks an arm whose two passes differ by more than `--noise` %: treat that row as noise-limited.
  Compare arms only within one `ab_run.sh` directory (same host, same session); cross-run differences mix in host
  and link state.
- **Rows.** Any test_tt_fabric bandwidth yaml (`benchmark_mode: true`). Keep the rows that show the regression and one
  row each that should not move (a 1-hop unicast, a size below the regression), as controls.

## 2. Code shape, offline: `offline_build.py`, `elf_size.py`, `codegen_diff.py`, `line_compare.py`, `loopvars.py`

**When to use them.** When a row regresses and you want to see what changed in the generated code, before or instead
of more device runs. They read ELFs only.

**Getting the inputs:**
1. Run one test (any yaml with the rows' configs) with `TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1` and a kept
   `TT_METAL_CACHE`. Keep its log.
2. `offline_build.py collect --out cmds.json <log>...`: the router's logged compile/link commands, per config hash and
   image.
3. `offline_build.py build --cmds cmds.json --variant <dir> --out <build dir>`: rebuild every logged config with the
   variant's files overlaid. An empty dir rebuilds the tree as logged; the code then equals the cache's.
   - `--set-opt O2` rebuilds at another -O level; `--flags` adds compiler flags; `--toolchain-root` swaps the sfpi
     release.
   - The cache the commands point into must still exist: it holds the per-config generated headers and the firmware
     objects the link uses.

**The tools:**
- `elf_size.py <cache or build dir>...`: per image, code bytes (PT_LOAD), `.text` and static stack depth from `_start`
  (min..max over configs).
  - `--entry-sp IMAGE=HEX` adds free stack.
  - `--limit BYTES` flags images over the kernel config buffer.
  - `--detail` prints the deepest call chain.
- `codegen_diff.py <base build> name=<build>...`: per image, how many configs' `.text` differ, and the change in
  instructions and in the largest frame. `--hot HASH/IMAGE` adds an instruction-level diff count for the configs of
  the rows you care about. Builds must hold the same configs: offline builds of one `cmds.json`.
- `line_compare.py <base.elf> <new.elf>`: the router loop's instructions per innermost source line (`addr2line -i`),
  base vs new, largest change first, plus loads and stores (sp-relative apart). `--inline <helper>` lists the
  instructions under a helper's inline frame. Compare by source line, not by function symbol size: the loop inlines
  its step functions.
- `loopvars.py --headroom <base.elf> <new.elf> --role sender|receiver`: the values the new image's loop stores to the
  stack that the base's does not (hot spills), the carried register counts and the loop's instruction count.
  - `loopvars.py <elf> --role ...` lists one image's spills.
  - Directories are accepted too: every image of the role in both, paired by config hash.
  - The self-test images are not shipped (`--self-test` says so).

**Caveat (all of them).** These are static counts. They say whether and where code moved, not what it costs:
- a stalled load shows at its first use, and moving code moves the stall;
- a spill on a path that runs rarely costs nothing.

Confirm any cause with an A/B that changes only that piece (`ab_run.sh` with two trees, or one tree and an `--env`
switch).

## 3. PC sampling over the debug bus: `pc_sampler.py`, `pcs_lines.py`, `pcs_profile.py` (Blackhole only)

**What it does.** `pc_sampler.py` reads each router ERISC's PC through the Ethernet core's debug bus while a test runs:
- one NoC write (signal select) and one NoC read per sample;
- no halt or step; tt-exalens in safe mode (no reset, no firmware load).

**How to run it.** Start the test with its output to a log, and start the sampler with `--start-log` on that log. The
sampler opens the device only after the test logged `Running Test:`, and it stops at `Tests: `. It writes
`<prefix>.bin` and `<prefix>.json`, which include each core's launch message and image head.

**The readers:**
- `pcs_lines.py <prefix> <cache> CHIP/CHAN/ERISC`: one ERISC's samples by source line and inline chain. The build is
  matched by image head; samples are mapped through the kernel cache's ELF (the JIT builds with `-g`).
- `pcs_profile.py <cache> <prefix> --cycles-per-packet X`: the busiest ERISC of each kind by named piece of work, as
  shares and as cycles/packet. Edit `SECTIONS` for other function names.

**Caveat.** Samples split cycles within one ERISC. They cannot say which ERISC bounds a row: routers poll without
stopping, so every ERISC's samples add up to the row's rate. That takes an A/B that changes one side's cost.

**PC sampling: what a Wormhole port needs** (from the sources, not verified on a WH machine):
- **The signal.** tt-exalens's Wormhole Ethernet block defines one PC signal, `erisc_pc` (`rd_sel 0, daisy_sel 7,
  sig_sel 18`, mask `0x7FFFFFFF`). Blackhole's are `erisc0_pc` / `erisc1_pc`. The sampler would pick the name per arch
  and sample one ERISC (`--eriscs 0`).
- **The launch message.** The load-address read uses Blackhole's `MEM_AERISC_MAILBOX_BASE` (0x100) and Blackhole's
  `launch_msg_t` offsets. Wormhole needs its own mailbox base (`tt_metal/hw/inc/internal/tt-1xx/wormhole/dev_mem_map.h`)
  and the offsets of `kernel_config_base`, `kernel_text_offset` and the text size for its ERISC.
- **The readers.** `pcs_common.PROC` maps ERISC 0/1 to the Blackhole image names; Wormhole's single ERISC runs the
  `erisc` image.
- **Check first** that the sampled PCs fall inside the kernel image's load window on a Wormhole router.

## 4. Hang capture and decode (Blackhole only, not shipped)

The study's hang capture reads the routers' L1 state and both ERISCs' mailboxes with Blackhole's layout. Porting it
means redoing that layout for Wormhole, so it is left out.
