# Tools for the Versim waveforms of the packer-phase repro (experiment, not for merge)

From a Versim run with the waveform on (`../repro_phase.sh sim K=100 VCD=1`) to the numbers and pictures on the pages.
Versim writes `1-1-core_dump.vcd` (about 4 GB) into `python_tests/`, the working directory of the test.

```bash
cd tt_metal/tt-llk/tests/repro_phase_tools
python3 extract_signals.py ../python_tests/1-1-core_dump.vcd k100.txt   # about 10 to 20 minutes for 4 GB
python3 to_cycles.py k100.cyc k100.txt                                    # one row per clock cycle
python3 analyze_waveform.py k100.cyc                                      # the numbers
python3 make_figures.py k100.cyc k100                                     # k100.cycles.svg, k100.tiles.svg
python3 make_small_vcd.py k100.window.vcd 89000 112000 k100.txt           # small VCD of the pack loop
```

`analyze_waveform.py` prints, for the measured pack loop:

- DEST reads per packer: cycles requesting and reads granted (the slow state is more cycles requesting for the same
  8,192 reads);
- wait cycles of packers 1/2/3 in each tile (where the slow state starts);
- every L1 access of the math core (TRISC1) during the loop, and the packers' L1 writes around each one;
- the pack RISC-V core's instruction pushes, and how many cycles it was blocked on a full instruction queue.

Signals used (Versim names under `TOP_01_01.tt_tensix_with_l1.tensix`): the per-packer DEST read request
`instruction_thread0.instruction_issue.tdma.tdma_dstac_regif_rden` and grant `dstac_regif_tdma_reqif_ready`; each
packer's L1 write `tdma.gen_pack_instance(N).packer.o_l1_wren` and `i_l1_req_ready`; the TDMA L1 arbiter
`tdma.l1_arbiter_packet_accept`; the RISC-V cores' L1 ports `trisc(N).u_trisc.o_l1_rden`, `o_l1_rdaddr`; and the pack
core's instruction queue `trisc(2).u_trisc.o_trisc_instrn_buf_wren`, `i_trisc_instrn_buf_req_ready`.
Run two Versim tests at the same time only from two separate checkouts: the VCD file name is fixed.
