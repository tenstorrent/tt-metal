# Quasar hardware facts for codegen

Facts the analyzer, writer and optimizer apply directly. Each one cites its source. Ground truth is
`tt_llk_quasar/instructions/assembly.yaml`, the ISA pages and the errata tickets. In-tree kernels, earlier
codegen runs (`/proj_sw/user_dev/llk_code_gen/...`) and "a prior run did X" are **not** evidence: gcd took the
wrong SFPSWAP imm12 polarity from a topk run, and lcm took a dead NOP from gcd.

| Topic | Fact | Source |
|---|---|---|
| SFPSWAP, after | Never put a NOP after SFPSWAP. Hardware unconditionally stalls the next instruction. | TEN-4581, verified on the Quasar A0 emulator (#59439 r4205359096); in-tree NOPs are dead, cleanup #59817 |
| SFPSWAP, before | Put one NOP before SFPSWAP only when its input comes from a 2-cycle op issued on the previous cycle. | TEN-4605 |
| NOP citations | "SFPxxx is 2-cycle" alone never justifies a NOP: an interlocked consumer stalls by itself. Cite an errata ID that lists the consumer as unscoreboarded. | gcd #59258, lcm #59435 |
| 2-cycle ops | Interlock is not free: a dependent op stalls. Schedule independent instructions into each MAD/SWAP latency slot. | ema #57129: 29048 → 22910 cycles over 128 tiles |
| Replay ownership | Math-thread bank 0 (slot 0 up) is recorded by FPU inits (reduce, matmul, transpose_dest, eltwise binary) and replayed later. An SFPU replay that is not in the reference goes in bank 1, recorded once in init (`ckernel_sfpu_cumsum.h` pattern). | ema #57129, dropout #59436 |
| Replay record mode | Record only (`exec_while_loading=0`); executing while loading hangs. | TEN-4690 |
| ADDR_MOD_6 | Not owned by any op: topk and cumsum reprogram it, and `_llk_math_sfpu_init_` never restores it. Program it where the kernel can rely on it and name the owner in a `@note`. | rand #59437 r4194518067 |
| Fixed delays | A settle loop or poll-less wait needs a spec or RTL bound. PRNG seed settle is ≥1600 cycles (RTL), not the 1024 of a test kernel. | rand #59437 r4197148206 |
| PRNG seed | 0xFFFFFFFF is the XNOR-LFSR lock-up state; remap it as WH/BH `rand_init` does. | dropout #59436 r4195361154 |
| Lane select | Select or mask lanes bitwise (SFPAND/SFPOR/SFPNOT), never by multiplying by 0/1 or adding 0: 0×Inf = NaN and the mask flips -0.0. | binary_bcast; WH/BH share the flaw |
| Int32 Dest | Default encoding is two's complement (unpack-to-dest copies L1 verbatim). Sign-magnitude only behind an opt-in template parameter, as add / mul_int32 / binary_comp do. | unary_max_min #59439 r4194468751 |
| FPU datacopy | The 32-bit-Dest ELWADD datacopy drops the sign of -0.0 and -NaN on the emulator (not filed yet). | signbit #58561 r4195652497 |
