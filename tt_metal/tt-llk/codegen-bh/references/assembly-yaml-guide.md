# How to Query assembly.yaml

The Blackhole ISA is documented in `tt_llk_blackhole/instructions/assembly.yaml` (3500+ lines). This guide teaches how to extract instruction information effectively.

---

## Basic Query Pattern

```bash
grep -A <lines> "^INSTRUCTION_NAME:" tt_llk_blackhole/instructions/assembly.yaml
```

- Use `^` to match instruction names at start of line
- Use `-A 50` to get ~50 lines of context (adjust as needed)
- Instruction names are ALL CAPS followed by colon

---

## Finding Instructions by Kernel Type

### SFPU Kernels

```bash
# List all SFPU instructions
grep "^SFP" tt_llk_blackhole/instructions/assembly.yaml

# Common SFPU instructions to look up:
grep -A 40 "^SFPMAD:" tt_llk_blackhole/instructions/assembly.yaml    # multiply-add
grep -A 40 "^SFPADD:" tt_llk_blackhole/instructions/assembly.yaml    # add
grep -A 40 "^SFPMUL:" tt_llk_blackhole/instructions/assembly.yaml    # multiply
grep -A 40 "^SFPLUT:" tt_llk_blackhole/instructions/assembly.yaml    # LUT operation
grep -A 40 "^SFPLOAD:" tt_llk_blackhole/instructions/assembly.yaml   # load from dest
grep -A 40 "^SFPSTORE:" tt_llk_blackhole/instructions/assembly.yaml  # store to dest
grep -A 40 "^SFPEXEXP:" tt_llk_blackhole/instructions/assembly.yaml  # extract exponent
grep -A 40 "^SFPSETEXP:" tt_llk_blackhole/instructions/assembly.yaml # set exponent
grep -A 40 "^SFPCAST:" tt_llk_blackhole/instructions/assembly.yaml   # type cast
```

### Unpack Kernels

```bash
# Core unpack instructions
grep -A 80 "^UNPACR:" tt_llk_blackhole/instructions/assembly.yaml      # main unpack
grep -A 50 "^UNPACR_NOP:" tt_llk_blackhole/instructions/assembly.yaml  # unpack NOP/clear

# Related config instructions
grep -A 50 "^CFGSHIFTMASK:" tt_llk_blackhole/instructions/assembly.yaml  # address manipulation
grep -A 30 "^SETADCXY:" tt_llk_blackhole/instructions/assembly.yaml      # set address counter
```

### Pack Kernels

```bash
grep -A 60 "^PACR:" tt_llk_blackhole/instructions/assembly.yaml       # main pack
grep -A 40 "^PACR_NOP:" tt_llk_blackhole/instructions/assembly.yaml   # pack NOP
```

### Math Kernels

```bash
grep -A 40 "^MVMUL:" tt_llk_blackhole/instructions/assembly.yaml   # matrix-vector multiply
grep -A 40 "^ELWMUL:" tt_llk_blackhole/instructions/assembly.yaml  # elementwise multiply
grep -A 40 "^ELWADD:" tt_llk_blackhole/instructions/assembly.yaml  # elementwise add
grep -A 40 "^ZEROACC:" tt_llk_blackhole/instructions/assembly.yaml # zero accumulator
```

### Config/Sync Instructions

```bash
grep -A 30 "^STALLWAIT:" tt_llk_blackhole/instructions/assembly.yaml  # stall and wait
grep -A 30 "^WRCFG:" tt_llk_blackhole/instructions/assembly.yaml      # write config
grep -A 30 "^SEMPOST:" tt_llk_blackhole/instructions/assembly.yaml    # semaphore post
grep -A 30 "^SEMGET:" tt_llk_blackhole/instructions/assembly.yaml     # semaphore get
grep -A 30 "^NOP:" tt_llk_blackhole/instructions/assembly.yaml        # no operation
```

---

## What to Extract from Results

Each instruction entry contains:

| Field | What It Tells You |
|-------|-------------------|
| `op_binary` | The instruction opcode (e.g., `0x42` for UNPACR) |
| `ex_resource` | Execution unit (SFPU, UNPACK, PACK, SYNC, etc.) |
| `instrn_type` | Instruction category |
| `arguments` | List of bit fields with positions and meanings |
| `description` | What the instruction does |

### Example: Extracting Argument Info

For `UNPACR`, the arguments section shows:
```yaml
arguments:
    - name: Last
      start_bit: 0
      description: "flush data accumulation buffers..."
    - name: SearchCacheFlush
      start_bit: 1
      description: "Flush row pointer cache..."
```

This tells you bit 0 is `Last` and bit 1 is `SearchCacheFlush`.

---

## When to Query assembly.yaml

### During Analysis (llk-analyzer)
- When you see `TTI_*` or `TT_OP_*` macros in reference code
- To understand what parameters an instruction expects

### During Debugging (llk-debugger)
- When error message mentions an instruction name
- When behavior doesn't match expectations
- To verify correct bit field values

### During Planning (llk-planner)
- To verify instruction capabilities before specifying them
- To check if instruction exists in Blackhole ISA

---

## Quick Reference: Instruction Prefixes

| Prefix | Type | Example |
|--------|------|---------|
| `SFP*` | SFPU operations | SFPMAD, SFPADD, SFPLUT |
| `UNPACR*` | Unpack operations | UNPACR, UNPACR_NOP |
| `PACR*` | Pack operations | PACR, PACR_NOP |
| `ELW*` | Elementwise math | ELWMUL, ELWADD |
| `MV*` | Matrix/vector ops | MVMUL |
| `SEM*` | Semaphores | SEMPOST, SEMGET |
| `STALL*` | Stall/sync | STALLWAIT |
| `CFG*` | Config writes | CFGSHIFTMASK |
| `SET*` | Register setup | SETADCXY, SETRWC |

---

## Troubleshooting

**Can't find instruction?**
```bash
# List all instruction names
grep "^[A-Z].*:" tt_llk_blackhole/instructions/assembly.yaml | head -100
```

**Need more context?**
```bash
# Increase -A value
grep -A 100 "^INSTRUCTION:" tt_llk_blackhole/instructions/assembly.yaml
```

**Looking for specific parameter?**
```bash
# Search within instruction definitions
grep -B5 -A5 "parameter_name" tt_llk_blackhole/instructions/assembly.yaml
```
