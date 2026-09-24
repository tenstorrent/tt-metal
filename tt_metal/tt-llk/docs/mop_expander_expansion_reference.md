# MOP Expander expansion reference

This document derives every expansion below only from the supplied MOP Expander functional model. It does not add inferred hardware behavior.

The examples use symbolic instruction names so that the expansion order is visible:

- `A0`, `A1`, `A2`, `A3`, `B`: Template 0 instructions
- `SkipA0`, `SkipB`: Template 0 skip instructions
- `S`: `StartOp`
- `E0`, `E1`: `EndOp0`, `EndOp1`
- `L0`, `L1`: `LoopOp`, `LoopOp1`
- `Last0`, `Last1`: `Loop0Last`, `Loop1Last`
- `[X, Y]`: emitted instruction stream, in order
- `NOP`: a plain NOP instruction with opcode `0x02`

`IsNop` recognizes only plain `NOP`. `DMANOP` and `SFPNOP` are not treated as NOPs by this model.

## Incoming instruction handling

Each Tensix coprocessor thread has an independent MOP Expander and independent `MaskHi` state. `MaskHi` starts at zero.

| Incoming instruction | Expander action | Emitted stream |
|---|---|---|
| `MOP_CFG` | Replace the thread's saved `MaskHi` with `Instruction.MaskHi` | `[]` |
| Template 0 `MOP` | Expand using the saved `MaskHi`, the instruction's `MaskLo` and `Count1`, and Template 0 `MopCfg` meanings | Template 0 expansion |
| `MOP` with `Template != 0` | Expand using Template 1 `MopCfg` meanings | Template 1 expansion |
| Any other instruction | Pass it through unchanged | `[Instruction]` |

The `MOP` and `MOP_CFG` instructions themselves are consumed by the expander; they are not included in the emitted stream.

## Template 0: mask-selected expansion

Template 0 uses the configuration as follows:

| Entry | Name | Used when |
|---|---|---|
| `MopCfg[0]` | Unused | Never |
| `MopCfg[1]` bit 0 | `HasB` | Selects whether the B-side instruction is emitted |
| `MopCfg[1]` bit 1 | `HasA123` | Selects whether `A1`, `A2`, and `A3` are emitted |
| `MopCfg[2]` | `B` | `HasB == 1` and the current mask bit is `0` |
| `MopCfg[3]` | `A0` | Current mask bit is `0` |
| `MopCfg[4..6]` | `A1`, `A2`, `A3` | `HasA123 == 1` and the current mask bit is `0` |
| `MopCfg[7]` | `SkipA0` | Current mask bit is `1` |
| `MopCfg[8]` | `SkipB` | `HasB == 1` and the current mask bit is `1` |

The complete mask is:

```text
Mask = (MaskHi << 16) + MaskLo
```

Expansion examines the least-significant bit first. A raw `Count1 = N` executes `N + 1` iterations and consumes mask bits `0` through `N`.

Therefore, `Count1 = 0` still executes one iteration. Only the low two bits of `Flags` are used; all higher flag bits are ignored.

### Every per-iteration situation

The current mask bit and the two flag bits completely determine one iteration:

| Mask bit | `HasA123` | `HasB` | Emitted instructions |
|---:|---:|---:|---|
| `0` | `0` | `0` | `[A0]` |
| `0` | `0` | `1` | `[A0, B]` |
| `0` | `1` | `0` | `[A0, A1, A2, A3]` |
| `0` | `1` | `1` | `[A0, A1, A2, A3, B]` |
| `1` | `0` | `0` | `[SkipA0]` |
| `1` | `0` | `1` | `[SkipA0, SkipB]` |
| `1` | `1` | `0` | `[SkipA0]` |
| `1` | `1` | `1` | `[SkipA0, SkipB]` |

`HasA123` has no effect on a mask-bit-`1` iteration. The skip path never emits `A1`, `A2`, or `A3` and has no separate skip instructions for them.

### Multi-iteration example

Given:

```text
HasA123 = 1
HasB = 1
Count1 = 3
Mask = 0b1010
```

The four bits are consumed from right to left: `0`, `1`, `0`, `1`.

```text
iteration 0, mask bit 0: [A0, A1, A2, A3, B]
iteration 1, mask bit 1: [SkipA0, SkipB]
iteration 2, mask bit 0: [A0, A1, A2, A3, B]
iteration 3, mask bit 1: [SkipA0, SkipB]

complete expansion:
[A0, A1, A2, A3, B,
 SkipA0, SkipB,
 A0, A1, A2, A3, B,
 SkipA0, SkipB]
```

### `MaskHi` situations

- After `MOP_CFG(H)`, later Template 0 expansions use `H` as mask bits `16..31` until another `MOP_CFG` replaces it.
- A Template 0 `MOP` does not clear `MaskHi` after expanding.
- A Template 1 `MOP` does not use or change `MaskHi`.
- Other instructions do not use or change `MaskHi`.

For example, with `MaskHi = 0x0001` and `MaskLo = 0x0000`, iterations `0..15` take the normal path, iteration `16` takes the skip path, and later zero bits take the normal path.

After all 32 mask bits have been shifted out, the mask is zero. Any later iteration therefore takes the mask-bit-`0` normal path.

The `ckernel_unpack_template` wrapper programs these Template 0 entries. Its `run(count, zmask)` passes `count - 1` as `Count1`, so a positive wrapper `count` requests exactly `count` iterations. The overload that does not issue `MOP_CFG` leaves the model's existing per-thread `MaskHi` unchanged.

The HAL accepts either a structural compile-time `MopConfig<Template0>` or a runtime aggregate. Both forms require `start_op` and `start_op_shadow`, require all three middle operations or none, and require the normal and shadow end operations together. Programming writes deterministic NOP values into disabled slots so a later raw field patch cannot expose stale configuration. `write<Template0Field>()` may replace an instruction slot already enabled by the complete configuration; callers must re-run `program(config)` to enable or disable the middle or end group.

## Template 1: nested-loop expansion

Template 1 uses the configuration as follows:

| Entry | Name | Meaning |
|---|---|---|
| `MopCfg[0]` | `OuterCount` | Number of outer iterations before special-case adjustment |
| `MopCfg[1]` | `InnerCount` | Number of inner iterations before optional doubling |
| `MopCfg[2]` | `StartOp` | Emitted at the start of each outer iteration unless it is plain `NOP` |
| `MopCfg[3]` | `EndOp0` | Emitted at the end of each outer iteration unless it is plain `NOP` |
| `MopCfg[4]` | `EndOp1` | Emitted after `EndOp0` only when neither end op is plain `NOP` |
| `MopCfg[5]` | `LoopOp` | Normal inner-loop instruction, or the first alternating instruction |
| `MopCfg[6]` | `LoopOp1` | Enables alternating mode unless it is plain `NOP` |
| `MopCfg[7]` | `Loop0Last` | Replaces the final inner instruction in the final outer iteration |
| `MopCfg[8]` | `Loop1Last` | Replaces the final inner instruction in every non-final outer iteration |

The `ckernel_template` wrapper programs these Template 1 entries.

### Wrapper constructor situations

Combining the wrapper defaults with the supplied model gives these expansions before any setter changes:

| Wrapper construction | Programmed loop and last slots | Inner expansion per outer iteration |
|---|---|---|
| Blackhole `ckernel_template(O, I)` | `LoopOp = NOP`, `LoopOp1 = NOP`, both last slots `NOP` | `I` yielded plain NOPs |
| `ckernel_template(O, I, op)` | `LoopOp = op`, `LoopOp1 = NOP`, both last slots `op` | `[op]` repeated `I` times |
| `ckernel_template(O, I, op0, op1)` with non-NOP `op1` | Alternating ops; both last slots `op1` | `[op0, op1]` repeated `I` times |
| `ckernel_template(O, I, op0, NOP)` | Alternation disabled; both last slots `NOP` | `[op0]` repeated `I - 1` times, then one yielded `NOP`, when `I > 0` |

The dual-op constructor with a plain-NOP `op1` is therefore not equivalent to the single-op constructor. The former yields a NOP in each final inner slot; the latter defaults that slot to `op`.

The last-instruction setters map to the model as follows:

- `set_last_inner_loop_instr(op)` sets `Loop1Last`, used by every non-final outer iteration.
- `set_last_outer_loop_instr(op)` sets `Loop0Last`, used by the final outer iteration.

The legacy wrapper and the default HAL Template 1 runner use zero instruction-level override fields, so their counts come from `MopCfg`. The Blackhole HAL also exposes compile-time and runtime per-issue overrides; a nonzero field replaces the corresponding count for one expansion, while zero preserves the `MopCfg` count.

### HAL configuration resolution

`hal::mop::MopConfig<MopTemplate::Template1>` groups fields by placement: `outer_loop` holds operations around the inner loop, while `inner_loop` holds operations emitted at inner-loop positions. The public names map to the ISA model as follows:

| HAL field | ISA model name |
|---|---|
| `outer_loop.count` | `OuterCount` |
| `inner_loop.count` | `InnerCount` |
| `outer_loop.start_op` | `StartOp` |
| `outer_loop.end_op` | `EndOp0` |
| `outer_loop.second_end_op` | `EndOp1` |
| `inner_loop.body_op` | `LoopOp` |
| `inner_loop.alternating_op` | `LoopOp1` |
| `inner_loop.last_op_on_final_outer_iteration` | `Loop0Last` |
| `inner_loop.last_op_on_nonfinal_outer_iteration` | `Loop1Last` |

The two count fields accept full 32-bit MopCfg values. The HAL writes them unchanged, and the Blackhole functional model uses their low ten bits as the effective counts. Every instruction slot is optional. Before programming hardware, the HAL resolves omitted fields as follows:

```text
StartOp = outer_loop.start_op if supplied, otherwise NOP
EndOp0 = outer_loop.end_op if supplied, otherwise NOP
EndOp1 = NOP if EndOp0 is NOP, otherwise outer_loop.second_end_op if supplied, otherwise NOP
LoopOp = inner_loop.body_op if supplied, otherwise NOP
LoopOp1 = inner_loop.alternating_op if supplied, otherwise NOP

DefaultLast = LoopOp if LoopOp1 is NOP, otherwise LoopOp1
Loop0Last = inner_loop.last_op_on_final_outer_iteration if supplied, otherwise DefaultLast
Loop1Last = inner_loop.last_op_on_nonfinal_outer_iteration if supplied, otherwise DefaultLast
```

This removes caller-side settings that merely reproduce model defaults. In particular:

- An omitted `LoopOp1` selects the single-operation model.
- An omitted last slot preserves the normal final loop operation: `LoopOp` for a single-operation loop and `LoopOp1` for an alternating loop.
- `EndOp1` is normalized to NOP when `EndOp0` is NOP because the model cannot emit it in that state.

The HAL still writes all nine resolved entries on every Template 1 program operation. `MopCfg` is write-only and persistent, so omitted fields must actively clear or replace values left by an earlier configuration.

The `program<Config>()` form resolves and validates a structural configuration at compile time. The `program(config)` form accepts runtime counts and instructions and checks runtime-dependent constraints with `LLK_ASSERT`. Both forms share the same resolver and hardware emitter. The parameterless runner takes both counts from `MopCfg`; overloads accepting `Template1CountOverrides` encode optional Blackhole instruction-level overrides.

### Count selection

On Wormhole:

```text
OuterCount = MopCfg[0] & 0x7f
InnerCount = MopCfg[1] & 0x7f
```

On Blackhole:

```text
OuterCount = MopCfg[0] & 0x3ff
InnerCount = MopCfg[1] & 0x3ff

if MOP bits[19:10] != 0:
    OuterCount = MOP bits[19:10]
if MOP bits[9:0] != 0:
    InnerCount = MOP bits[9:0]
```

The override fields are marked unsupported and weak-confidence in the ISA documentation. The HAL exposes them with ten-bit validation so callers can exercise and validate the Blackhole capability without manually splitting the outer override across legacy `TTI_MOP` arguments.

### Start and end-op situations

Start-op behavior is independent of the inner-loop count:

| `StartOp` | Start of each outer iteration |
|---|---|
| Plain `NOP` | Emit nothing |
| Anything else, including `DMANOP` or `SFPNOP` | Emit `[StartOp]` |

End-op behavior is:

| `EndOp0` | `EndOp1` | End of each outer iteration |
|---|---|---|
| Plain `NOP` | Plain `NOP` | `[]` |
| Plain `NOP` | Non-NOP | `[]` |
| Non-NOP | Plain `NOP` | `[EndOp0]` |
| Non-NOP | Non-NOP | `[EndOp0, EndOp1]` |

`EndOp1` can therefore never be emitted when `EndOp0` is plain `NOP`.

### Situation 1: one loop instruction

When `LoopOp1` is plain `NOP`, `InnerCount` is not changed and there is no alternation.

For an effective `InnerCount = I > 0`:

```text
non-final outer iteration:
  [LoopOp repeated I - 1 times, Loop1Last]

final outer iteration:
  [LoopOp repeated I - 1 times, Loop0Last]
```

Example with `OuterCount = 2` and `InnerCount = 3`:

```text
outer 0: [L0, L0, Last1]
outer 1: [L0, L0, Last0]

complete inner-loop expansion:
[L0, L0, Last1, L0, L0, Last0]
```

The last instruction is always the configured replacement. If `InnerCount = 1`, no normal `LoopOp` is emitted:

```text
OuterCount = 2, InnerCount = 1
=> [Last1, Last0]
```

### Situation 2: two alternating loop instructions

When `LoopOp1` is not plain `NOP`, the model:

1. Sets `LoopOpFlip = LoopOp ^ LoopOp1`.
2. Doubles `InnerCount`.
3. XORs the current loop instruction with `LoopOpFlip` after every effective inner iteration.

This alternates the current instruction between the original `LoopOp` and `LoopOp1`. Because the effective count is even, each outer iteration begins with the original `LoopOp`.

For a configured `InnerCount = I > 0`, the effective inner count is `2 * I`. The final alternating `LoopOp1` position is replaced by `Loop1Last` or `Loop0Last`.

Example with configured `OuterCount = 2` and configured `InnerCount = 2`:

```text
effective InnerCount = 4

outer 0: [L0, L1, L0, Last1]
outer 1: [L0, L1, L0, Last0]

complete inner-loop expansion:
[L0, L1, L0, Last1, L0, L1, L0, Last0]
```

For configured `InnerCount = 1`:

```text
OuterCount = 2
=> [L0, Last1, L0, Last0]
```

### Situation 3: full outer-iteration composition

For each outer iteration with an effective inner count greater than zero, the emitted stream is:

```text
[optional StartOp,
 inner instructions ending in Loop1Last or Loop0Last,
 optional EndOp0,
 optional EndOp1]
```

With `OuterCount = 1`, the only outer iteration is also the final one, so `Loop1Last` is never selected and the final inner slot uses `Loop0Last`. With `OuterCount > 1`, every non-final outer iteration uses `Loop1Last`, and only the final outer iteration uses `Loop0Last`.

Given:

```text
OuterCount = 2
InnerCount = 3
LoopOp1 = NOP
StartOp = S
EndOp0 = E0
EndOp1 = E1
```

the expansion is:

```text
[S, L0, L0, Last1, E0, E1,
 S, L0, L0, Last0, E0, E1]
```

### Situation 4: zero effective outer count

If the effective `OuterCount` is zero after count selection and any special-case adjustment, the expansion is empty:

```text
[]
```

No start, loop, or end instruction is emitted.

### Situation 5: zero effective inner count, normal cases

With `InnerCount = 0`, the inner loop emits nothing. `LoopOp`, `LoopOp1`, `Loop0Last`, and `Loop1Last` are not emitted.

If the special case in the next section does not change `OuterCount`, each outer iteration emits only the enabled start and end instructions. Examples:

```text
StartOp = S, EndOp0 = E0, EndOp1 = E1
OuterCount = 2, InnerCount = 0
=> [S, E0, E1, S, E0, E1]

StartOp = S, EndOp0 = NOP, EndOp1 = E1
OuterCount = 2, InnerCount = 0
=> [S, S]

StartOp = NOP, EndOp0 = NOP
OuterCount = 2, InnerCount = 0
=> []
```

### Situation 6: no-start, zero-inner, active-end special case

The documented unsupported behavior begins with these three conditions:

```text
StartOp is plain NOP
InnerCount == 0
EndOp0 is not plain NOP
```

The current Blackhole model then describes two anomalous branches: with a non-NOP `EndOp1`, `OuterCount == 1` receives `+1024`; with a plain-NOP `EndOp1`, any outer count receives `+1`. Because the ISA marks this area weak-confidence, the HAL conservatively rejects the entire three-condition shape with `static_assert` for a compile-time configuration or `LLK_ASSERT` for a runtime configuration.

### Yielded NOPs versus suppressed NOPs

The model checks `IsNop` only for `StartOp`, `EndOp0`, `EndOp1`, and `LoopOp1`:

- A plain-NOP `StartOp`, `EndOp0`, or `EndOp1` is suppressed according to the rules above.
- A plain-NOP `LoopOp1` disables alternating mode.
- `LoopOp`, `Loop0Last`, and `Loop1Last` are yielded when their positions are reached even if their value is plain `NOP`.
- Template 0 instructions are yielded according to the mask and flags without any NOP filtering.

## Configuration timing

The model presents `MopCfg` as if it were sampled at expansion entry, but the supplied description says hardware mostly samples entries when they are needed. Software must therefore not rewrite `MopCfg` while an expansion is active.

Use RISC-V `TTSync` before writing a new MOP configuration so the prior expansion has completed. Each RISC-V T0, T1, or T2 core accesses the write-only `MopCfg[0..8]` belonging to the corresponding Tensix T0, T1, or T2 thread at `TENSIX_MOP_CFG_BASE`; reading this range is undefined behavior.
