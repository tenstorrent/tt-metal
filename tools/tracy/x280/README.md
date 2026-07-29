# X280 <-> Tensix experiments

Code that runs on the Blackhole L2CPU (SiFive X280) tiles under Linux,
communicating with Tensix cores over NoC TLB windows.

The first test is a heartbeat: a kernel on Tensix logical core (0,0)
increments a counter at L1 `0x80000` forever; the X280 maps that L1 through
a 2MB NoC window and prints the counter every 20ms.

## Prerequisites

- Linux booted on L2CPU tile 0 via
  [tt-bh-linux](https://github.com/tenstorrent-riscv-software/tt-bh-linux),
  with its console tool running (it provides the X280's disk and network).
- tt-metal built with `-DBUILD_PROGRAMMING_EXAMPLES=ON`.
- `TT_METAL_SKIP_DRAM_TLBS=1` in the environment for every tt-metal process.
  This skips the per-DRAM-channel 4G TLB windows so device init doesn't
  collide with the windows held by the tt-bh-linux console.
- Never reset the chip (`tt-smi -r`) while Linux runs — X280 harts can only
  be brought out of reset once per chip reset. DRAM buffers are off-limits:
  Linux lives in D5–D7 and the allocator interleaves across all channels.

## Run

1. Launch the counter kernel from the tt-metal host (stays running):

       export TT_METAL_SKIP_DRAM_TLBS=1
       ./build/programming_examples/metal_example_loopback

   It prints the worker core for logical (0,0), e.g. `x=1 y=2`.

2. On the X280 (`ssh -p 2222 debian@localhost`, no password):

       gcc -O2 -o poll x280_poll.c
       sudo ./poll 1 2 0x80000

   Expect ~1.5M increments per 20ms line. Ctrl-C to stop.

3. Cleanup on the host: `pkill -f metal_example_loopback`.

Kernel + host launcher live in `tt_metal/programming_examples/loopback/`.
