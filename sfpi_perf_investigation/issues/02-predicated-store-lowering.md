# sfpi: "pick one of two constants under a CC, then store" emits a redundant liveness copy

**Labels:** `sfpi`, `perf`
**Kind:** missed optimization
**Worth:** 1 Tensix instruction per predicated result
**Compiler:** `riscv-tt-elf-g++ (tenstorrent/sfpi:7.69.0[822]) 15.1.0`, `-O2 -mcpu=tt-bh-tensix`
**Context:** ask 2 of 7 from [the umbrella writeup](../ISSUE.md). Origin: [tt-metal#52932](https://github.com/tenstorrent/tt-metal/pull/52932).

Every comparison kernel ends the same way: pick 0.0 or 1.0 depending on a predicate, then
write it to DEST. sfpi's cost for that depends on whether you predicate the *assignment*
or the *store*, and only one of the two spellings is the obvious one.

## What we have today

The natural spelling — compute a result, then store it:

```cpp
// verified: 01_idiomatic_baseline.cc :: probe_eqz
extern "C" void probe_eqz() {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat v = dst_reg[0];
        vInt abs_bits = as<vInt>(setsgn(as<vUInt>(v), 0));
        vFloat result = 0.0f;
        v_if(abs_bits == 0) { result = 1.0f; }
        v_endif;
        dst_reg[0] = result;
        dst_reg++;
    }
}
```

8 Tensix instructions per DEST row:

```
SFPLOAD   L0, 0, 0, 7
SFPSETSGN L0, L0, 0, 1
SFPSETCC  L0, 0, 6
SFPMOV    L0, L9, 2
SFPMOV    L0, L10, 0    # LV:L0   <-- the liveness merge
SFPENCC   3, 10
SFPSTORE  L0, 0, 0, 7
TTINCRWC  0, 2, 0, 0
```

The two constants are pulled into an allocatable LReg (`L0`) and then a *second* `SFPMOV`
appears purely to merge the two definitions of `L0` across the predicate — that is the
`# LV:` annotation the compiler prints.

## What we need

Predicating the store instead of the assignment removes the merge entirely:

```cpp
// verified: 02_single_workarounds.cc :: c_eqz_predicated_store
extern "C" void c_eqz_predicated_store() {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        vFloat v = dst_reg[0];
        vInt abs_bits = as<vInt>(setsgn(as<vUInt>(v), 0));
        dst_reg[0] = 0.0f;
        v_if(abs_bits == 0) { dst_reg[0] = 1.0f; }
        v_endif;
        dst_reg++;
    }
}
```

7 instructions per row:

```
SFPLOAD   L0, 0, 0, 7
SFPSETSGN L0, L0, 0, 1
SFPSTORE  L9, 0, 0, 7    # store the constant register directly
SFPSETCC  L0, 0, 6
SFPSTORE  L10, 0, 0, 7   # ditto, under the predicate
SFPENCC   3, 10
TTINCRWC  0, 2, 0, 0
```

Note what the good form does: it stores straight out of the constant registers `L9`/`L10`
and touches no allocatable LReg at all. **sfpi already knows how to emit this** — it just
will not get there from the first spelling.

**The ask:** recognise "assign one of two constants under a CC, then store the result" and
lower it to the two-predicated-store form. Failing that, the narrower fix is enough: drop
the liveness copy when the predicated assignment's only consumer is a store.

## Why it matters

This pattern is the tail of every comparison kernel, so it is a flat 1-instruction tax on
all eight families. On `eqz`, where the kernel is only 6 instructions to begin with, one
instruction is 17%.

Things we tried that do **not** help:

- `vConst0` / `vConst1` instead of the literals `0.0f` / `1.0f`. Identical codegen, 8
  instructions either way (`e_eqz_const_reg` in the probes). They are also deprecated.
- Hoisting the constants into named `vFloat`s outside the loop.

## Repro

```sh
git fetch origin ldjurovic/sfpi-perf-gap-investigation
git checkout ldjurovic/sfpi-perf-gap-investigation
./sfpi_perf_investigation/run.sh    # probe_eqz (8) vs c_eqz_predicated_store (7)
```

See [the umbrella writeup](../ISSUE.md) §3 for the counting method.
