# Native capacity fixture — closed 4K reproduction

This package runs real source prefill and transfers two selected high-end ranges into a passive buffer owner. It retains the tested range-owner command, byte-oracle and shutdown machinery. The relocated package has host validation only; device qualification is pending the separate hardware evidence.

**Scheduling boundary:** focused [4K](../../../docs/migration-prefill-capacity-4k.md), [8K](../../../docs/migration-prefill-capacity-8k.md), [16K](../../../docs/migration-prefill-capacity-16k.md) and [32K](../../../docs/migration-prefill-capacity-32k.md) selected-range gates passed. 64K device acceptance remains pending. This example stays closed at 4K. 128K remains deferred and is excluded from this fixture. This is not an automatic capacity sweep. These fixture commits and new reports remain local by the user's choice; no public push is pending.

## What one run checks

For chosen capacity C, source slot0 computes C actual tokens and slot1 computes C−32. Both use real H2D and all32 layers. The destination mapping is crossed:0→1 and1→0. Transfer ranges are `[C−1024,C)` and `[C−1056,C−32)`; the latter crosses a compute-chunk boundary.

All16 K/V-head configurations and32 layers contribute32,768 selected packed pages /142,606,336 bytes. Capture each selected chunk intersection and flush it before its first readiness ack. Check exact destination bytes, preserved source bytes, adjacent/untouched samples and both endpoints again after native shutdown. Logical endpoints remain separate from page bytes. No full-cache snapshot, HF/PCC comparison, performance remeasurement or decoder is included.

For4K, actual compute is8 full32 calls /256 post-sync acks. Compile uses2 calls; geometry warmup uses8 calls before managers/bridges start. Warmup uses distinct IDs and no native ack sink. Every selected config/layer group must change before the actual final ack. Native commands remain96 layer ranges /104 audited calls.

## Configure existing dependencies

Copy `plan.example.json` outside the repository. It is deliberately unarmed: `reviewed=false`, `launch_authorized=false`, `manager_memory_reviewed=false`; budgets, nonce, lease and pins require a fresh reviewed binding.

- Use the sibling migration `environment.example.sh`, `model-spec.example.json` and `verify_native_env.py` interfaces. Set the built source-matched repository, Python, checkpoint and native loader paths. The shell entry hashes the configured environment file before sourcing it.
- Set `allocation_owner`, the exact two-node allocation, per-role job/node/lease, rack-qualified physical lock, ports and recent independently accepted health receipts. Owner and supervisor reject wrong node/job/lock before native imports. The node entry narrows affinity to one allowed CPU before preflight and supervisor execution.
- Bind your built manager, DMK ELF, integrated source/passive bridge binaries, source manifest and reviewed54-case host receipt using `bridge-host-validation.example.json`. The receipt must bind the same manifest and both binary hashes and report no Metal/UMD linkage for the host test binary. This does not claim the source client is Metal-free.
- `accepted_gate_spec` identifies the source-matched production model and checkpoint metadata. `accepted_source_pins` hashes those production files. Include all local non-`checks_` Python helpers, shell entry, model/environment files, probe, bridge/library files, token files and health receipts in `pins`. Keep original tested-source hashes distinct from hashes of relocated/formatted publication files.
- Supply `book_manifest` using `book-manifest.example.json`. Produce two distinct token-ID JSON lists with the agreed tokenizer/checkpoint/special-token policy, exactly C IDs each, then record their SHA256s and set `validation_passed=true` only after that validation. The owner takes C−32 IDs from slot1. Filenames are relative to the manifest. CPU tests use temporary synthetic IDs only; they do not substitute for validated device inputs.
- Retain the existing resource policy:128GiB host/cgroup admission,32GiB shared free disk,16GiB RSS/HWM per manager, selected streaming captures and identity-bracketed memory observations. Resource admission is measured on the filesystem containing `run_dir`. Choose setup/warmup/supervisor budgets from actual smaller runs and the current lease. Native inbound and whole-bridge deadlines are explicit positive uint32 milliseconds. Omitted overrides retain 600,000 ms inbound and 1,500,000 ms overall; the Python transfer wait must stay at least ten seconds inside the inbound deadline. Supervisor, lease and recovery bounds still apply.

## Timeout support and source provenance

The timeout-aware clients are associated with local tt-d-gen commit
4f8847c03c888fd16a0ca1088d2ebbf7d24fcac5. Its three changed C++ files match
the source used for the independently accepted 54-case host suite. That receipt
binds the actual source/passive binaries; host tests have no Metal/UMD linkage.
This is source/host validation, not a 64K device pass. All new commits remain
local by user choice; no push is pending.

The closed example explicitly records the compatible defaults. A separately
reviewed long-context plan may override both waits. The reviewed 64K proposal
uses 1,800,000 ms inbound, 5,400,000 ms overall and 1,790 seconds for the Python
phase wait. These are operational caps, not expected runtimes or a performance
claim. Invalid types, zero/negative values, uint32 overflow and a lost
ten-second margin fail before native imports. Both role configs and their saved
resolver receipts bind the exact values. Manager and per-command timeouts,
model calls, readiness, byte checks and shutdown semantics are unchanged.

## Reproduction commands

From the tt-metal repository root, after explicit review of the configured plan and both assigned endpoints:

```bash
python3 -B models/demos/llama_3p1_8b_d_p/tests/migration/native_capacity/controller.py \
  --plan /absolute/capacity-plan.json --plan-sha256 EXACT_PLAN_SHA256
```

The existing child interface remains `supervise_owner.py --plan PLAN --plan-sha256 SHA --role source` or `passive`. Do not bypass the controller's scheduler/source/health guards or the supervisor's inherited physical lock. Its actual owner command is `owner_runner.py` with the same arguments. The controller writes `verified-result.json` only after the imported capacity verifier accepts both actual exits and reports.

Both exact managers must stop with accepted native lifecycle evidence before either cache owner releases buffers. An ambiguous teardown retains owner and lock for coordinated recovery. Lease expiry is not safe retention. No automatic reset, forced kill or retry was added.

## Host checks

The only repository pytest discovery entry is the shared subprocess wrapper. Local `checks_*` files run with child-local native/model import blockers:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest --noconftest -o addopts= -q \
  'models/demos/llama_3p1_8b_d_p/tests/migration/test_host_contracts.py::test_host_contracts[native_capacity]'
python3 -I -S -B models/demos/llama_3p1_8b_d_p/tests/migration/test_host_contracts.py --suite native_capacity
```

41 host cases retain the prior 34 checks (22 owner, 4 launcher and 8 portability) and add seven timeout checks for default/explicit serialization, integer bounds, saved-config integrity, actual validator import routes, helper pins and the closed example. They exercise streaming byte mutations, wrong slot/duplicate/missing capture, stale warmup group, ack-before-capture refusal, resource boundaries, actual repository runtime constructors via inert stand-ins, closed flags, one-CPU ordering, failed preflight, configured manifest/binary identity, transitive owner/supervisor imports, wrong host/job/lock and tampered environment/token files. None opens a device or executes a native binary.
