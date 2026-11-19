# Focused diagnostic timeout audit

Source-only audit, 2026-11-16. No hardware access, reset, termination, or retry was performed by this audit.

The preserved `bucket_sender_counters_mapped.txt` records a four-byte MMIO read timeout at **0xffb20228** on device 1, physical worker (1,2). UMD reports 220708 microseconds against a 2 ms per-operation budget. Triage then marks the device broken and skips it. Its top-level `pass` entry is not a successful counter measurement.

## Register identification

The failing address is **NoC0 `NIU_MST_NONPOSTED_WR_REQ_SENT`**, not a command-buffer-0 register:

- Installed `ttexalens/hardware/blackhole/functional_worker_block.py:41` sets the NoC0 register base to `0xffb20000`.
- `ttexalens/hardware/blackhole/niu_registers.py:29` declares this name as `NocStatusRegisterDescription(offset=0x28)`.
- That file's `get_niu_register_base_address_callable` adds `0x200` for a status register.
- Thus `0xffb20000 + 0x200 + 0x28 = 0xffb20228`.

The original mapped diagnostic attempts this read before its optional command-buffer inspection. The optional block was therefore **not reached**. Removing that block does not constitute a repair for this timeout.

## Read-safety contract and limits

The installed architecture's NoC memory map lists `noc0_regs` and `noc1_regs` without a read prohibition (`functional_worker_block.py:253–254`). `MemoryMapBlockInfo.is_safe_to_read` defaults to true when `safe_to_read` is unspecified (`ttexalens/memory_map.py:21–24`). The stock `tools/triage/check_noc_status.py` also reads `NIU_MST_NONPOSTED_WR_REQ_SENT` using `read_register`.

This establishes that the API recognizes the address and permits a read. It does **not** guarantee that a stalled or damaged endpoint can complete a host MMIO transaction. This diagnostic did not issue a register write, counter reset, command submission, core resume, or device reset. The observed timeout does not establish whether prior failure state, diagnostic perturbation, transport trouble, or another cause prevented the response. It does not establish the cause of the original combine write barrier hang.

The earlier `bucket_sender_counters.txt` preserved hardware values but could not read NCRISC private memory because the default memory accessor attempted unsupported NCRISC debug access. The mapped attempt fixed that API path using the installed architecture mapping, but its rows were only returned as a list. When the later MMIO read raised, triage discarded the callback's partial result. Therefore the mapped artifact contains **no recoverable counter or runtime-address values**, even if some earlier reads completed. Do not reconstruct those values from the earlier run or assume an unreported comparison passed.

## Diagnostic changes

`scripts/qwen_combine_sender.py` now prints each row as a `QWEN_SENDER_ROW` JSON line with `flush=True`, in addition to returning the normal triage table. It emits a timeout row before propagating `TimeoutDeviceRegisterError` to triage. It does not catch the timeout and continue reading the device.

The optional command-buffer reads have been removed. Software counters, shared-L1 runtime arguments, and bounded bank tables precede hardware-counter reads. Each hardware read is announced with a `PENDING` row. Future partial output therefore identifies the attempted register and preserves earlier successful measurements. This ordering is an evidence-preservation improvement, not a claim that the remaining reads are safe to retry on the currently timed-out device.

Validation was CPU-only: syntax/import/dependency discovery, and an injected exception after initial metadata to verify that JSON rows and flush calls survive without a callback return. The existing failed artifacts remain unchanged. No retry should be inferred from this edit; the owner must resolve the timed-out-device lifecycle first.
