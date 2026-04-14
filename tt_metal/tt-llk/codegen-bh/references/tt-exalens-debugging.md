# tt-exalens: On-Device Debugging Reference

tt-exalens is a command-line debugger for inspecting Tenstorrent hardware state. Use it when you need to inspect L1 memory, check RISC-V register state, or diagnose hangs/timeouts that log analysis alone can't explain.

In our repo, run it with the `tt-exalens` command.

---

## Starting tt-exalens

```bash
tt-exalens
```

Prompt format: `gdb:None device:0 loc:1-1 (0, 0) >` shows current device and core location.

Type `h` for help, `help <command>` for details on a specific command.

---

## Key Commands for LLK Debugging

### Select Core Location (`go`)

Target a specific Tensix core before reading memory or registers:

```bash
go -l <x>,<y>      # e.g., go -l 2,2
go -d <device_id>   # switch device
```

### Read L1 Memory (`brxy`)

**Most useful command for debugging DATA_MISMATCH and stimuli issues.**

```bash
brxy <core-loc> <addr>                  # read 1 word (4 bytes)
brxy <core-loc> <addr> <word-count>     # read multiple words
brxy <core-loc> <addr> --sample <N>     # sample address for N seconds (detect changes)
```

Examples for LLK debugging (addresses from `docs/tests/infra_architecture.md`):

```bash
# Check runtime arguments struct (performance layout)
brxy 0,0 0x20000 16

# Check stimuli space start
brxy 0,0 0x21000 16

# Check result data area
brxy 0,0 0x21000 64

# Sample an address to detect if kernel is writing (useful for TIMEOUT diagnosis)
brxy 0,0 0x21000 --sample 5
```

### Write to L1 Memory (`wxy`)

Write a single word to L1 (useful for manually setting runtime params or testing):

```bash
wxy <core-loc> <addr> <data>
# e.g., wxy 0,0 0x100 0x123
```

Note: writes one word (4 bytes) at a time.

### Dump RISC-V Registers (`gpr`)

**Critical for diagnosing TIMEOUT / hang issues.** Shows register state for all RISC-V cores on the selected Tensix tile:

```bash
gpr    # dumps BRISC, TRISC0, TRISC1, TRISC2 registers for current core
```

Use this to check:
- Is a TRISC core stuck? (PC not advancing)
- Which thread is hung? (helps narrow TIMEOUT to unpack/math/pack)

### Run ELF Files (`re`)

```bash
re <path-to-elf> -l <core-loc>    # run on specific core
re <path-to-elf>                   # run on all cores
```

---

## Scripting

Run multiple commands non-interactively:

```bash
tt-exalens --commands "go -l 2,2; brxy 2,2 0x20000 8; gpr; x"
```

Redirect output for analysis:

```bash
tt-exalens --commands "go -l 0,0; brxy 0,0 0x20000 16; gpr; x" > debug_dump.txt
```

---

## When to Use tt-exalens

| Scenario | What to Do |
|----------|------------|
| **TIMEOUT** — kernel hangs | `gpr` to see which TRISC is stuck. `brxy` with `--sample` to check if memory is being written. |
| **DATA_MISMATCH** — wrong output | `brxy` to read stimuli area and result area in L1. Compare against expected values. |
| **Stimuli not loaded** | `brxy` at `StimuliConfig.STIMULI_L1_ADDRESS` (default `0x21000` perf layout) to verify data was written. |
| **Runtime params wrong** | `brxy` at `0x20000` (perf layout) to inspect the `RuntimeParams` struct as the kernel sees it. |
| **Error matrix differs between runs** | Kernel not processing stimuli — use `gpr` + `brxy` to check if kernel even reaches the data. |
