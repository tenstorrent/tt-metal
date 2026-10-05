# Versim waveforms of the packer-phase repro (experiment, not for merge)

Six Versim runs of the repro in `../../REPRO_PHASE.md`, each with the full waveform recorded. These files keep only
63 signals, for the measured pack loop (simulation time 89,000 to 112,000; one clock cycle is 2 time units).
Open a file with GTKWave or Surfer; for GTKWave, `gtkwave k100.gtkw` opens it with the signals arranged.

| File | Run | TILE_LOOP cycles |
|---|---|--:|
| k0.window.vcd | same code, no L1 read | 9,070 (fast) |
| k100.window.vcd | same code, one L1 read by the math core at spin iteration 100 (cycle 46,210) | 9,323 (slow) |
| k200.window.vcd | same code, read at iteration 200 (cycle 46,610) | 9,070 (fast) |
| k225.window.vcd | same code, read at iteration 225 (cycle 46,710) | 9,309 (slow) |
| n0.window.vcd | 0 nops in `_llk_pack_init_` | 9,070 (fast) |
| n1.window.vcd | 1 nop in `_llk_pack_init_` | 9,313 (slow) |

Signals: per-packer DEST read request and grant (4-bit vectors, bit 0 = packer 0), each packer's L1 write request
(`pN_l1_wren`) and whether it is accepted (`pN_l1_ready`), the TDMA L1 arbiter, the math RISC-V core's L1 read and
address, and the pack RISC-V core's instruction pushes (`pack_riscv_instr_push`) and queue-ready.

The full waveforms (all 1.3 million signals of the tile, about 4 GB each) are compressed in
`/home/nstojic/versim-waveforms/` on the Belgrade IRD machines (`zstd -d` to unpack).
