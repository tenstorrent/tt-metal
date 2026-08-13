<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

SPDX-License-Identifier: Apache-2.0
-->

# X280 ISS hello world

Run a LIM-linked hello world on a **simulated X280 cluster**, not on QEMU `sifive_u`.

The ISS is Spike (`riscv-isa-sim`) configured to match Blackhole L2CPU X280:

| Knob | Value | Source |
| --- | --- | --- |
| Harts | 4 identical | tt-llm-engine `x280/README.md` |
| ISA | `rv64gcv` | same |
| Vector | VLEN=512, ELEN=64 | `vlenb==64`, `vl==16` for e32/m1 |
| LIM | `0x08000000` size `0x1E0000` | `x280/include/x280.h` |
| Load address | `0x08001000` | active FW window |

If you have a vendor SiFive X280 ISS, set `X280_ISS=/path/to/iss` and the same `./run.sh` will use it.

This is an **ISA-level** simulator (correct vector/CSRs/4 harts/LIM map). It is not cycle-accurate and does not model Blackhole NOC/DMA.

## Run

```bash
cd tests/tt_metal/tt_metal/x280_iss_hello
./run.sh
```

Needs network on first run (clone Spike) and `riscv64-unknown-elf-gcc` (prefers tt-llm-engine’s toolchain):

```bash
X280_TOOLCHAIN=/path/to/tt-llm-engine/x280/toolchain ./run.sh
# or
TT_LLM_ENGINE=/path/to/tt-llm-engine ./run.sh
```

Success ends with: `The simulated X280 hello world ran. All checks passed.`

```bash
./run.sh --clean
```
