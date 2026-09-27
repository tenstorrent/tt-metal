# Initial AutoDebug finding: selected profile abort

Fresh source diagnosis: `/root/profile_abort_diagnosis`; fuller report pending.
No implementation edits or assertion bypass are proposed.

`tt_metal/impl/profiler/profiler.cpp:2392` accepts two uint32_t arguments and
resizes its host vector using `size * num_dram_banks / sizeof(uint32_t)`.
The multiplication occurs at uint32_t width before division/promotion.
The source diagnosis finds subsequent per-bank reads use full unwrapped sizes.

The failed capture compiles12000000 bytes/RISC at250000 supported operations.
Even110 workers *5 RISCs requires6600000000 bytes per device before Ethernet
and bank rounding. uint32 multiplication wraps that lower bound to2305032704,
underallocating by4294967296 bytes. This predicts a host heap overwrite during
profiler readback. Surviving host zones finish profiler reads for chips3 and2
but no marker processing; the model already printed TP_DONE4 before SIGABRT.
The exact corruption site is not proven by a recovered crash backtrace.

The historical successful v0 count100000 compiles4800000 bytes/RISC. Its
worker-only2640000000-byte total does not wrap. A retry at100000 is a focused
configuration control; the4096/128 workload, final marker validation and all
128 complete replay-window checks must remain unchanged. It is not permission
to accept dropped markers or incomplete device rows. No C++ change is in scope.

Root CPU arithmetic reproduced the two uint32 totals using ctypes.c_uint32.
This is stronger current evidence than the prior stage's timestamp-rollover
hypothesis, which has no matching fatal timestamp evidence in this capture.
Failed artifacts: profile_selected_sliding.log and profile_selected_sliding/.
Recovery: profile_selected_failure_{list_before,reset,list_after,smoke}.log,
all exit0 and four ASICs visible. No raw failed timing enters telemetry.
