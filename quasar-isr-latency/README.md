# Quasar DFB implicit-sync ISR reaction time (emulator waves)

Measures, from Zebu waveforms, the time from **the DFB tile ISR firing on DM0** to **the tile-counter
update reaching the consumer's Neo** ("data ready to be consumed"), step by step.

| File | What |
|---|---|
| `isr_tc_latency.py` | Parses an exported `waves.vcd.gz` and prints a per-ISR timeline + summary |
| `isr_tc.rc` | Signal list for the Zebu export (RoCC IRQ, DM0 retire trace, overlay + Neo tile counters) |
| `run1.vcd.gz` | `dfb_init_timing_bench --case base`, uninstrumented ISR (default PCs) |
| `run2.vcd.gz`, `run2_bench.log` | Same, with the rdcycle ISR log compiled in (needs the PC flags below) |

## Try it on an existing dump
```bash
python3 isr_tc_latency.py run1.vcd.gz --window-ns 20000
python3 isr_tc_latency.py run2.vcd.gz --window-ns 20000 --handler-pc 0xee28 --llk-write-pcs 0xefe4,0xf050 --mret-pc 0xef90
```

## Capture a new one
1. Run the test on emu-quasar-1x3 with waves on. The emu wrapper `quasar-1x3_run_dev.sh` must read
   `ENABLE_WAVEFORM=${ENABLE_WAVEFORM:-0}` (it is hard-coded to 0 in a stock wrapper):
   ```bash
   export TT_METAL_CACHE=/tmp/isr_run/cache   # fresh firmware build
   TT_METAL_RUNTIME_ROOT=$PWD TT_METAL_HOME=$PWD TT_METAL_SIMULATOR=<tt-umd-simulators>/build/emu-quasar-1x3 \
   TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_MEASURE_DFB_INIT_TIME=1 ENABLE_WAVEFORM=1 \
   ./build_Release/test/tt_metal/dfb_init_timing_bench --case base
   ```
2. Export only these signals (~1 min) on soc-l-04:
   ```bash
   Z=$(ls -dt /proj_sw/user_dev/$USER/work/aether/verification/emu/emu_out/1x3/test_umd_remote_1x3_0/zrun.* | head -1)
   cp isr_tc.rc $Z/
   ssh soc-l-04 "source /tools_soc/tt/bin/bashrc >/dev/null 2>&1; cd /proj_sw/user_dev/$USER/work/aether && \
     source bin/setup_emu_env.sh 1x3 >/dev/null 2>&1; cd $Z && start-verdi --export --rc isr_tc.rc --local --threads 8"
   VCD=$(ls -dt $Z/export.*/waves.vcd.gz | head -1)
   ```
3. Get the handler PCs of the firmware that ran (they move whenever the ISR changes):
   ```bash
   riscv-tt-elf-objdump -d $(find $TT_METAL_CACHE -path "*firmware/dm0/dm0.elf") \
     | awk '/<_Z25dfb_implicit_sync_handlerv>:/,/^$/' | grep -E "sync_handlerv>:|llk_intf_write|mret"
   ```
4. `python3 isr_tc_latency.py $VCD --window-ns 20000 --handler-pc <entry> --llk-write-pcs <posted>,<acked> --mret-pc <mret>`

Each capture leaves ~220 MB in `$Z`; delete it afterwards.

## Reading the numbers
- On the emulator **all clocks (DM, overlay, Neo) are 100 MHz / 10 ns and 1:1:1**. Cycle counts per step
  carry over to silicon; nanoseconds do not (silicon DM/overlay ~2 GHz, ai/NoC ~1.35 GHz).
- "DATA READY" = the increment is accepted at the Neo tile-counter input port (`t6[k].neo.u_t6.i_overlay_to_t6_rts`);
  Neo's internal counter register is not probed (likely +1 cycle).
- The first ISR of a program runs with cold I/D caches; later ones show the steady state.

Base-case result (uninstrumented): interrupt -> data ready = **355 cycles** (1st ISR, cold) / **74 cycles** (warm)
= 10 entering the ISR + 56 handler code until the counter write + 8 counter update reaching Neo (warm).
