# r03-b01-a01 result: 0.0 (runtime_error). Kernel binaries too large for the kernel config buffer on the h7168 shapes

## What happened vs expected
- r02-b03-a04's compute diff applied cleanly onto r02-b02-a04, as planned. No JIT compile error.
- Six shapes ran and passed the PCC test: kimi-k3, glm, mistral, dsv4-flash, minimax-m3, gpt-oss.
  - Their printed PCC / max_abs / mean_abs match the earlier nodes (e.g. glm 0.9802855 / 1.6094 / 0.232011).
  - The µs columns in summary.md are misaligned and unusable: the failed shapes contributed no op rows, so the
    group-by-order assignment shifted.
- Three shapes failed at program creation, before any launch:
  - deepseek-v3: `TT_FATAL: Program size (71168) too large for kernel config buffer (70656) on TENSIX`
    (program.cpp:3637, `state.offset <= max_size`).
  - kimi-k2.7: 72752 B. deepseek-v4-pro: 71440 B.
  - The overshoot is 512 to 2096 B. All three are h7168 shapes, the largest compile-time config.
- This is not a numerical or schedule failure. The merged kernels simply don't fit.

## Why
The kernel config buffer holds every RISC binary of a core's program (BRISC reader, NCRISC writer, TRISC0/1/2).
Each parent lineage was just under the 70656 B limit on h7168. The two pieces added on top of r02-b03-a03 each
grow a different part:
- r02-b02-a04's mixed phase-B planner (`adaptive_chunk.hpp`, +64 lines) is compiled 3 times:
  `plan_launch` runs on the reader (reader.cpp:390), on the writer (writer.cpp:254) and on compute UNPACK
  (fused_swiglu.cpp:1201).
- r02-b03-a04's PACK-thread pipeline adds the per-subblock hand-off K-loop to all three TRISCs and the SFPU bodies
  to TRISC2.

Compute ELF sizes (.text + .data + .rodata) from this run's JIT cache, per compiled variant:
- TRISC0 (UNPACK, includes the planner): 21.2-24.5 KB.
- TRISC1 (MATH): 9.4-10.5 KB.
- TRISC2 (PACK): 11.2-13.0 KB.
- Largest variant: 24308 + 10452 + 13024 = 47.8 KB of the 69 KB buffer.
- The reader/writer ELFs were not captured: another eval cleaned up the per-run cache dir first.

## Classification
repairable failure (bug: the combined reader + writer + 3 TRISC binaries exceed the 70656 B kernel config buffer
on h7168; the mechanism itself was never measured)

## What a child of this node should try next
Keep this code and cut at least ~2.1 KB (aim for 4 KB+ of margin) from the per-core binary set, then re-eval.
Best options first:
1. **Compute the launch plan once per core, not three times** (also the measured 5-29 µs prologue win from
   r02-b01-a04 next #1):
   - Let one RISC (e.g. the reader) run `plan_launch` and write the `LaunchPlan` into a small L1 scratch: a
     1-page CB or an L1 buffer address passed as a runtime arg.
   - It then signals the writer and compute through a CB push.
   - Writer and UNPACK read the plan instead of running the planner. That removes two copies of
     make_plan / plan_groups / model_phase (several KB from TRISC0 and NCRISC) and two of three prologue passes.
   - Watch the r02-b02-a04 phase-B per-layout fields: the struct must hold the layout index per phase plus
     whatever `layout_group` needs.
2. Cheaper: `__attribute__((noinline))` on the big planner helpers (`model_phase`, `plan_groups`, `make_plan`).
   This only helps if the compiler currently inlines several copies inside one RISC (check with
   riscv-tt-elf-size / nm on the cached ELF).
3. Cut the size of the pipelined K-loop in TRISC0/1: the last-block per-subblock path may be duplicated by
   template/constexpr branches for SiLU vs binary. Keep only the active branch with `if constexpr` on the
   activation define.
4. Don't change the compute opt level (O3 -> Os) without measuring. It would shrink the binaries but likely
   slow the matmul / SFPU inner loops.
- Expected value of the repaired combination is unchanged: ~1.58-1.59 (dsv3/glm from r02-b02-a04,
  gpt-oss/mm3/dsv4-pro/kimi-k3/mistral -1.5..-2% from the pipeline).
- To check size before a device run: sum the 5 ELFs' loadable sections for the dsv3 / kimi-k2.7 / dsv4-pro
  compile-time configs. kimi-k2.7 was the worst case (72752 B).
