<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Native prefill migration: first 2K result

**The complete 2K native transfer passed on 17 September 2026.**
Two different Llama-3.1-8B-Instruct prompts produced source KV on one Galaxy.
The real tt-d-gen manager transferred that cache into passive buffers on a second
Galaxy. All **65,536 packed BFP8 pages / 285,212,672 bytes matched**, with zero
mismatches. The source cache stayed unchanged.

The receiver loaded no decoder model. This result closes the first complete-prefix
native transport case. It does not close every test in the
[migration acceptance plan](migration-prefill-tests.md).

## Tested path

~~~text
Frozen prompt IDs -> persistent H2D input -> full 32-layer prefill
  -> synchronize each compute chunk -> publish 32 layer acknowledgements
  -> production PrefillReader / KvmClient -> native managers
  -> passive destination buffers -> exact packed-byte checks
~~~

Configuration: SP=4, TP=8, two slots, 2,048-token capacity, 1,024-token compute
chunks and BFP8_B cache. Each chunk completed before its acknowledgements were
published. This test does not establish layer-by-layer compute/transfer overlap.

| Check | Result |
|---|---|
| Source call order, (slot, start, end) | (0,0,1024), (1,0,1024), (0,1024,2048), (1,1024,2048) |
| Source-to-destination slot map | 0 -> 1, 1 -> 0 |
| Layers and configurations | All 32 layers; all 16 K/V head configurations |
| Readiness | 128 ordered acknowledgements, after device synchronization |
| Native protocol | 2 registrations, 2 peer-ready events, 128 layer commands, 2 seals, 2 successful completions |
| Completed length | 2,048 tokens per slot; no reused prefix |
| Packed comparison | 65,536 pages, 4,352 bytes per page, zero mismatches |
| Source preservation | Source readback after the managers stopped matched its saved snapshot |
| Shutdown | Both managers stopped before either owner closed; all actual exits were zero |

## Why the comparison is meaningful

The prompts differ, and their packed source bytes differ in every configuration.
The receiver began with a different pattern in every configuration and slot.
The owner saved each source chunk before publishing its first acknowledgement.
After completion, it compared every selected destination page with those saved
source bytes, including BFP8 exponent and mantissa data. It also checked the
source again. A completion notification alone was not the acceptance condition.

An independent root check reconstructed the 32 configuration/slot source digests
from the four flat saved chunk files. Those digests matched the source and
destination readback reports. Root also verified source pins, JIT artifacts,
command counts, completion results and shutdown order.

The destination's raw post-transfer buffers were not retained as flat files.
The exact destination comparison ran in the device owners through the frozen,
tested comparator. The later root check verified its reports and independently
reconstructed expected source digests; it did not repeat hardware readback.

## Source identity and reproduction status

The [machine-readable result](migration-native-2k.json) records binary hashes,
source identity, counts and evidence hashes. Full evidence remains under the
shared task root in evidence/task-10-native-migration/prefill-transfer-launch-003.

The model runtime used repository head 4714492b08e31c5cb0835f92dd754c319e5ab244
and the additional pinned integration sources in that evidence. The native test
used the frozen source_client and passive_client binaries identified in the JSON.

The published [tt-d-gen source bridge](https://github.com/tenstorrent/tt-d-gen/commit/c3d7d46bc18de442a698063a29e7844d2c34b814)
includes stricter command parsing and completion accounting. Its 15 host tests
passed, but that newly built public binary was not the binary used by this
device run. The next native edge-case tests must record their actual binary
identity too.

The paired launch fixture still uses task-local setup. A portable public test
and its reproduction command remain part of the implementation work. Do not
reuse the completed launch's job IDs, nonce or output directory.

## Remaining coverage

- Partial final pages and shorter valid prefixes inside larger allocations.
- Continuation, incremental transfers and untouched destination regions.
- Ordinary slot mapping and a third prompt reusing a drained slot.
- Delayed readiness, backpressure, in-flight failure, cancellation and restart.
- Native transfers at 4K, 8K, 16K, 32K, 64K and 128K.
- Portable fixture publication and a later run of its exact published binary.

This is an exact transport result, not a new numerical-model, performance,
decoder-layout or generated-response result. The accepted 2K numerical evidence
remains the model-correctness prerequisite.
