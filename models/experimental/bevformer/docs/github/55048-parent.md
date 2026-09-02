---
title: "☂️ BEVformer BH optimizations"
issue: "55048"
state: open
url: "https://github.com/tenstorrent/tt-metal/issues/55048"
---

## Context

Track BEVFormer encoder correctness and performance work for Blackhole. The work is split into focused child issues so each optimization can be reviewed, measured, and discussed independently.

Prototype commits and N150 profiles provide evidence for several proposals, but they do not describe the current implementation or guarantee the same result on Blackhole. Each accepted change must be validated on the target hardware.

## Shared requirements

- Measure each optimization independently against the immediately preceding configuration and report percentage changes.
- Keep host/pipeline gaps separate from device kernel time and use repeated iterations when host latency is the claim.
- Report PCC for every change; do not lower an existing threshold to make an optimization pass.
- Preserve existing MSDA, TSA, SCA, layer, and encoder coverage, including PCC >= 0.997 for the base encoder.

## Done when

- [ ] Every child issue is resolved or has a documented no-win/rejected result.
- [ ] The final Blackhole profile reports the cumulative percentage improvement and remaining bottlenecks.
- [ ] The full PCC suite passes without relaxed thresholds.
