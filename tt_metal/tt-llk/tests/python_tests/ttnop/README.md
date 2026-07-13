# ttnop

A program meant for surfacing synchronization bugs that injects NOP delays into arbitrary places in a linked kernel without moving any existing code.

It works by detouring. The instruction at each site is overwritten with a jump into a code cave grown onto the end of `.text` that holds the delay, a faithful copy of the displaced instruction, and a jump back.

## Build

```sh
make
```

## Use

```
ttnop <in.elf> -o <out.elf> [SITE ...] [options]
```

A **SITE** is `LOC=N`: inject a delay of `N` before the instruction at `LOC`:

```sh
# 5 nops before the instruction at VMA 0x6510
ttnop kernel.elf -o slow.elf 0x6510=5

# by symbol, and symbol+offset
ttnop kernel.elf -o slow.elf main=8  main+0x1c=3

# from a file (one LOC=N per line, '#' comments allowed)
ttnop kernel.elf -o slow.elf -f nop-sites

# a delay before every instruction of a class
ttnop kernel.elf -o slow.elf --every store=2 --every call=4
# classes: load store op opimm lui branch jal call jalr system fence amo custom all
```

### Large delays: `--loop`

By default a delay of `N` is `N` NOPs, which can outgrow memory. `--loop` instead emits a fixed (typically) 8-word loop of `N` iterations (`2N` cycles), so arbitrarily large delays stay compact:

```sh
ttnop kernel.elf -o slow.elf 0x6510=50000 --loop
```

The loop saves/restores `t0` on the stack, so do not target the first few instructions of a function.

### Inspect & verify

```sh
ttnop kernel.elf --list                        # classify .text + per-class tally
ttnop kernel.elf -o s.elf --every branch=3 --list   # dry-run plan + cave map
ttnop kernel.elf -o s.elf 0x6510=5 --verify    # re-open output, re-check detours
```

## How a detour looks

`bltu a4,a5,0x65ac` at `0x6590`, asking for 3 nops, becomes a jump into the cave:

```
0x6590:  j      0x6980            # was bltu; now jumps into the cave

0x6980:  nop; nop; nop            # the delay
0x698c:  bgeu   a4,a5, 0x6994     # inverted condition ...
0x6990:  j      0x65ac            #   ... taken edge -> original target
0x6994:  j      0x6594            #   ... fall-through edge -> site+4
```
