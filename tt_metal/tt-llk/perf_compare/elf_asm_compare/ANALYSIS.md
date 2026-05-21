# NC vs WC ELF analysis — worst variants from per-config gating sweep

Variant: `eltwise_binary_fpu_perf.cpp`, `tile_cnt=16`, `Elwmul`, `HiFi2`, `dest_acc=Yes`.
Build hashes (identical between NC and WC for the same variant — the perf path is
the only thing that changes):
- L1_TO_L1   `Float16_b -> Bfp8_b`   = `a562da81…f02c8af58`
- MATH_ISOLATE `Bfp8_b -> Float16`    = `f53d5b5d…f7a41118ca8`

Per-tile cycles (from the per-config sweep):

| run_type        | NC    | WC      | WC-NC   |
|-----------------|------:|--------:|--------:|
| INIT  L1_TO_L1   |  332  |   350   |   +18   |
| **TILE_LOOP L1_TO_L1** | **843** | **6147** | **+5304** |
| KERNEL L1_TO_L1  | 1354  |  12203  | +10849  |
| TILE_LOOP UNPACK_ISO |  677  |    673  |    -4   |
| TILE_LOOP MATH_ISO   |  824  |    600  |  -224   |
| TILE_LOOP PACK_ISO   |  465  |    467  |    +2   |

## 1. Inside the TILE_LOOP zone — what leaks?

**Nothing.** The instructions between `zone_scoped::ctor` (which reads
`RISCV_DEBUG_REG_WALL_CLOCK_L = 0xFFB121F0`) and `zone_scoped::dtor` (same read)
are byte-for-byte identical between NC and WC. See:
- `L1_TO_L1/pack_nc_vs_wc_insns.diff` — only register renames (a7↔a6, t3↔t1, …).
  Instruction count, opcodes, operands all match. 87 body instructions in both.

Layout for pack L1_TO_L1 (verified in `wc/L1_TO_L1/pack.asm` and `nc/.../pack.asm`):
```
  ...
  perf_counter_scoped::ctor      ; 4× sw -> 0xFFB1203C/14/38/F8 (=1u)   [WC only]
  get_zone_id()                  ; linear scan of zone_hashes[]         [WC only]
  ───── zone_scoped::ctor ─────  ; lw 0xFFB121F0 = t_start
  body (semwait, ttmop, …)       ; IDENTICAL in NC and WC
  PROFILER_SYNC = tensix_sync    ; store_blocking(pc_buf_base[1], 0) + spin
  ───── zone_scoped::dtor ─────  ; lw 0xFFB121F0 = t_end
  perf_counter_scoped::dtor      ; 4× sw stop + 169-counter read loop   [WC only]
```

`perf_dtor` happens **after** `t_end` is recorded, so its work is not inside the
zone window. Verified by addresses: `t_end` read at `pc=0x10c10`, perf_dtor
loop body at `pc=0x10cbc–0x10d10`.

## 2. Why does the TILE_LOOP measurement grow by +5304 cyc/tile then?

L1_TO_L1 is a **cross-thread** measurement (`helpers/profiler.py:224`):
```python
durations = pack_end["timestamp"] - unpack_start["timestamp"]
```
i.e. **unpack's TILE_LOOP zone start** to **pack's TILE_LOOP zone end**.

In per-config gating on L1_TO_L1, only pack arms/freezes/reads. Math and unpack
have empty perf_counter_scoped (verified in `wc/L1_TO_L1/{math,unpack}.asm` — no
`0xFFB1203C/14/38/F8` writes). Yet the cross-thread span grows by ~85K cycles
total because pack does heavy work between zones that doesn't show up inside any
single zone:

```
   pack:  INIT body  | zone_dtor t_end | perf_dtor (~7K cyc) | perf_ctor | zone_ctor t_start | TILE_LOOP body | zone_dtor t_end | perf_dtor (~7K cyc) | KERNEL zone_dtor
   unpack: INIT body | zone_dtor       | (empty perf_dtor)   |           | zone_ctor t_start | TILE_LOOP body | zone_dtor       |                     |
```

Pack accumulates two ~7K perf_dtor chunks (INIT + TILE_LOOP). That alone delays
pack's wall-clock relative to unpack's by ~14K cyc, but the observed gap is
~85K. The extra ~70K is **bus contention**: while pack is hammering the Tensix
debug-register MMIO bus (169 active config slots × 1 MMIO write + 1 MMIO read
each, plus 64 L1-mux iters with an extra MMIO read+write), every other Tensix
access (math's TT_* instructions, pack's own subsequent semwait/mop, NoC
traffic) competes for the same path and stalls.

Pack-thread-stall counter confirms this:
- WC perconfig L1_TO_L1: `pack_thread_stall_pct = 31.6 %` (≈ 31K cyc of 98K total).
- WC all-threads-stop:    `pack_thread_stall_pct =  0.0 %`.

## 3. Why is the all-threads-stop baseline NOT regressed?

When all 3 threads issue stop writes, the HW edge-detect freezes counters on
**the first** thread to write `2u`. Whichever thread happens to be idling first
(unpack, after its own pipeline drains, or math after its semaphore release)
takes the freeze hit. The 169-iter read loop then runs on every thread, but each
thread runs it on its own idle time — none of them is pack on its critical
path. The bus is still hammered, but the cost lands on idle threads, not the
bottleneck.

This is the right intuition for distributing perf overhead. Its only downside
is **value fidelity** for ISOLATE configs where threads early-return (e.g.
pack returns immediately in MATH_ISOLATE, so the first-stop fires before math
finishes its work, and the math/unpack counters under-count).

## 4. MATH_ISOLATE — same shape, smaller magnitude

For MATH_ISOLATE the active thread is math. Math arms before zone start, freezes
after zone end. The MATH_ISOLATE TILE_LOOP measurement is **same-thread**
(`math_end - math_start`), so perf_ctor/perf_dtor are outside the window.

The −224 cyc/tile (`824 → 600`) WC swing is not a stall — it's the opposite
direction. Most likely explanation: math's body is single-issue math
instructions with no bus contention against pack/unpack (both early-return for
MATH_ISOLATE). The 4 perf_ctor MMIO writes immediately before `zone_ctor`'s
wall-clock read may resolve some pipeline back-pressure that NC happens to hit
otherwise. This is within measurement noise (±240) and not a correctness
problem — just a slight measurement skew.

See `MATH_ISOLATE/math_nc_vs_wc_insns.diff` for the byte-level comparison.

## 5. Files

| File | What |
|------|------|
| `L1_TO_L1/{nc,wc}/{unpack,math,pack}.asm` | Source-interleaved disassembly |
| `L1_TO_L1/{nc,wc}/build.h`                | Compile-time config that fixes the variant |
| `L1_TO_L1/{unpack,math,pack}_nc_vs_wc.diff`        | Full source-interleaved diff |
| `L1_TO_L1/{unpack,math,pack}_nc_vs_wc_insns.diff`  | Instruction-only diff (regs+opcodes, no comments) |
| same for `MATH_ISOLATE/`                  | Worst MATH_ISOLATE variant |

## TL;DR

The instructions inside the TILE_LOOP zone are identical NC↔WC. The 5304 cyc/tile
swing is **bus contention from pack's 169-counter read loop running outside the
zone**, which delays pack's wall-clock end and inflates the cross-thread
`pack_end - unpack_start` measurement. Per-config gating concentrates all that
contention on the critical-path thread; the all-stop scheme distributes it onto
idle threads instead, at the cost of HW counter value fidelity for ISOLATE runs.
