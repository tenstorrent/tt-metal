# Direct face-segment NoC unflatten

Parent's unflatten142 v1 isolated probe passes all10 exact-bit cases. Whole-layer B32 prefill measures26.3265 ms versus integrated13426.4616 ms, only about0.5% faster. That leaves the reader's whole-tile staging and512 uint32 copies per assembled tile as a concrete next target. This v2 investigation is source-only; no device was opened.

Artifacts: standalone `candidates/prefill_unflatten_bf16_v2.py`, integrated `candidates/prefill_unflatten_v2.py`, generator/manifest, `tests/test_prefill_unflatten_v2_source.py`, and `prefill_unflatten_v2_cpu_checks.json`. The integrated candidate still starts from integrated134 and changes only the guarded final packed-input prefill unflatten. Existing selected arithmetic, recurrence, decode, B1 and aligned-prefix paths are unchanged. Policy remains `prefill_unflatten_packed=True` with optional `prefill_unflatten_cores`. The existing `tests/run_prefill_unflatten_probe.py` accepts either new candidate path with the same function name.

## Alignment-supported direct placement

Blackhole NoC requires matching source/destination low address bits, not individually aligned addresses: `internal/debug/sanitize.h:746` compares `(worker_addr & mask)` to `(noc_addr & mask)`. Architecture constants in `internal/tt-1xx/blackhole/noc/noc_parameters.h:375-380` specify64-byte DRAM read alignment,16-byte DRAM write alignment and16-byte local-L1 read alignment.

Each BF16 face row is32 bytes. If the first logical flat source row is odd, a direct read to the ordinary output tile would violate relative64-byte alignment. Instead v2 allocates two **4096-byte output-CB slots** and assembles each2048-byte tile at offset0 for an even first row or32 for an odd first row. Every valid DRAM source segment then has the same low6 bits as its destination. The writer reads from the same0/32 offset and emits the2048-byte tile to its ordinary DRAM page. Both ends have identical low4 bits, so this write is legal. The unused bytes in each slot are never written to the output tensor.

`api/tensor/noc_traits.h` supports TensorAccessor `.page_id` and `.offset_bytes`; `api/dataflow/circular_buffer.h` resolves destination offsets from the current write pointer and explicit writer READ_PTR offsets from the current read pointer. Circular-buffer configuration validates that total size is divisible by page size; it does not require a BF16 page to be exactly one tile. The existing `tests/ttnn/unit_tests/operations/debug/test_noc_read_barrier_with_trid.py:54` uses BF16 CB pages of4096 bytes. Total v2 CB storage is still8KiB/core, now entirely a double-buffered output queue; there is no scratch CB or compute kernel.

## Transfers and synchronization

Reader walks valid rows, splitting only at input and output face boundaries. At most8 DRAM segment reads cover one output tile (two faces times at most four row intervals). All reads are issued before one barrier. A full32-row tile with aligned source remains a single2048-byte direct read.

Invalid padding is filled using `noc.async_write_zeros` on only the disjoint invalid-row ranges: at most4 zero segments, all starts32-byte aligned. On Blackhole `internal/tt-1xx/noc_zero_l1.inl` implements this as loopback reads from MEM_ZEROS_BASE; `write_zeros_l1_barrier` is the same read barrier and waits for both valid DRAM reads and padding transactions. Thus there are no CPU zero stores, and no valid bytes race with zero writes. The writer waits for the filled CB slot, writes once, barriers, then releases that slot. Reader reuse is protected by ordinary CB credits. There is no per-segment barrier, mapping tensor, full source-tile staging or RISC word-copy loop.

For B32/S33, valid first sequence tiles transfer exactly2048 bytes; one-row tails transfer64 bytes rather than a whole2048-byte source tile. Padding still needs local zero traffic and the full output tile write. This is a transaction-count versus bandwidth tradeoff: v2 can issue more DRAM transactions than v1's one/two full-tile reads, but removes the512-word assembly and avoids reading unused rows. The result requires device measurement; no speedup is claimed.

## Source/CPU checks and next gates

The exhaustive test covers every first-row offset0..31 and valid-row count1..32:1024 combinations. It simulates every valid segment and zero range with unique raw-word IDs, proving exact output words, zero padding, source-page bounds, destination bounds, exactly-once coverage, no read/zero overlap, relative64-byte alignment for every DRAM read, and relative16-byte alignment for the final write. Maximum counts are8 valid segments and4 zero segments. It also verifies deterministic immutable generation/hashes, all untouched original definitions by AST, and absence of the old scratch/copy loop. Python compilation passes.

Run the unchanged10-case own-input probe first, preserving strict CPU/native-TTNN/repeat bit equality and stable output address. Then compare whole-layer B32/S33 warmed prefill against both v1 and integrated134, with identical projection precision/configs; include the queued long-sequence correctness gate and watcher run. B1 and aligned S remain the inherited paths. Any API/kernel failure needs focused diagnosis and an immutable repair; it is not evidence to reject this material movement optimization.
