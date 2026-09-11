# Blackhole GDDR contention verification

## Scope and safety status

`BlackholeContentionFixture.SafeGDDRContentionCharacterization` is a hardware-gated,
bank-local characterization test for the tensor-prefetcher arbitration investigation. It never
writes memory-controller registers. The test uses a DRISC GDDR-DMA plus NOC transfer as a
prefetch-source proxy and measures a sibling Tensix NOC2AXI GDDR read under four randomized modes:

- current operation only;
- prefetch proxy only;
- both using the same bank;
- both using different banks as an isolation control.

The proxy is intentionally named as such in logs and JSON. It does not claim to replace full-stack
tensor-prefetcher validation. The real DRISC tensor-prefetcher lifecycle and payload correctness
remain covered by the `test_prefetcher_BH_*` tests.

## Register-sweep gate

The repository does not contain authoritative definitions for the proposed `0xFC10_xxxx`,
`0xFC20_xxxx`, or `0xFC30_xxxx` controls. In particular, the following have not been confirmed:

- field masks and reset values;
- Blackhole revision applicability;
- bank and tile strides;
- MPFE P1/P2/P3 mapping;
- write semantics and readback behavior;
- CMFW ownership or periodic rewriting;
- a reliable restore-on-error contract.

Therefore round-robin weight, starvation timeout, reorder-priority/QoS, and xbar-regulator sweeps
are unavailable. Do not add addresses from handoff notes to this test. Enable register experiments
only after each item above is backed by an authoritative hardware or firmware definition and an
exclusive-device restoration procedure.

## Running the safe harness

Use an idle, exclusively owned Blackhole device:

```bash
TT_METAL_SLOW_DISPATCH_MODE=1 \
TT_METAL_RUN_BH_CONTENTION=1 \
BH_CONTENTION_TEST_COMMIT=$(git rev-parse HEAD) \
BH_CONTENTION_JSON=/tmp/blackhole-contention.json \
.build/default/test/tt_metal/RelWithDebInfo/unit_tests_api \
  --gtest_filter='BlackholeContentionFixture.SafeGDDRContentionCharacterization'
```

The harness uses slow-dispatch DRAM/L1 access helpers, so `TT_METAL_SLOW_DISPATCH_MODE=1` is required.

Optional environment variables:

- `BH_CONTENTION_BANK` (default `0`);
- `BH_CONTENTION_ISOLATION_BANK` (default `1`, must differ);
- `BH_CONTENTION_SENDER_SUBCHANNEL` (default NOC1 endpoint; must be one of the two
  DRISC-usable sender subchannels reported in JSON);
- `BH_CONTENTION_PREFETCH_NOC` (default `0`, accepts `0` or `1`);
- `BH_CONTENTION_CURRENT_NOC` (default `1`, accepts `0` or `1`);
- `BH_CONTENTION_BYTES` (default `16384`, 64-byte aligned);
- `BH_CONTENTION_ITERS` (default `256`);
- `BH_CONTENTION_REPETITIONS` (default `20`);
- `BH_CONTENTION_SEED` (default `0xB1AC`);
- `BH_CONTENTION_TEST_COMMIT` (set this to `git rev-parse HEAD` for archival runs);
- `BH_CONTENTION_JSON` (optional output path).

The first randomized pass is a warmup and is omitted from the output. Every measured point records
current-op and prefetch-proxy cycles and effective bandwidth, bank placement, payload correctness,
and repetition number. Modes with an active current operation also record the host-readable NOC1 NIU
read-request counter delta; this field is `null` for prefetch-proxy-only samples. The JSON also records
board/device identity, firmware versions, DRAM and AI clock speeds, physical-channel training status,
DRAM harvesting mask, selected core/NOC placement, and the register-sweep gate.

On Blackhole, a harvested physical GDDR channel has zero training and BIST status bits, which UMD
decodes as `IN_PROGRESS`; this does not mean firmware is actively training that disabled channel.
Interpret status together with `dram_harvesting_mask`. For example, mask `0x8` makes physical channel
3's persistent `IN_PROGRESS` expected while the seven logical banks remain usable. Never reset or
retrain GDDR to change this status.

## Qualification criteria

Before using results to choose a control:

1. Collect at least 20 repetitions per mode.
2. Confirm every payload check passes.
3. Confirm same-bank p95/p99 degradation is repeatable.
4. Confirm different-bank latency is materially closer to current-only than same-bank.
5. Re-run across representative prefetch banks, both usable DRAM sender subchannels, representative
   transfer sizes, and NOC placements. Keep live-hardware invocations serialized.
6. Run the full-stack tensor-prefetcher validator and bandwidth benchmark separately.

No production arbitration recommendation should be made from the proxy alone. After authoritative
register definitions become available, add one control at a time with before/applied/after
readbacks, exact restoration on all exits, and immediate abort on unexpected scope or readback.
