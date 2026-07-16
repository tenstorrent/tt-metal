# AUTOTRIAGE

## Diagnosis
- The compact-ragged metadata kernel hung because its first Blackhole NoC read violated the source/destination alignment-match contract: the DRAM source was 0 mod 64 while the destination CB began at 32 mod 64.

## Triage Evidence
- A watcher-enabled reproduction stopped device 0, worker `(0,0)`, in `compact_ragged_dispatch.cpp`.
- NCRISC reported a 32-byte NoC0 DRAM read from `0x00100080` to L1 `0x01b320` with `invalid address alignment in NOC transaction`.
- The host synchronization and later teardown waits were downstream of that rejected read.

## Source Evidence
- The kernel's first transaction reads one 32-byte compact-index row into `cb_indices`.
- The generic-op CB allocator placed that first CB at L1 `...320` while TTNN's DRAM tensor allocation was `...080`.
- Both addresses are 32-byte aligned, but their low six address bits differ (`0x20` versus `0x00`), which Blackhole's DRAM NoC alignment-match rule rejects.

## Downstream Effects
- Since the first index page never reaches `cb_indices`, the kernel never completes its input producer loop.
- `ttnn.synchronize_device` waits indefinitely without watcher; watcher turns the same condition into a concrete abort.

## Proposed Fix
- Give each CB page 63 bytes of alignment slack and compute its read/write offset from the runtime DRAM and L1 addresses.
- Keep CB page strides multiples of 64 so the selected offset remains valid across every page.
- Re-run watcher and elementwise metadata comparison before enabling the full MoE path.

## Uncertainty
- Output NoC writes use the same alignment contract; watcher validation must confirm all output tensor bases are also 0 mod 64.
