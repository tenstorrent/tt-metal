# Stage04 seq4096 baseline stall: investigation

## Verdict

**No source-proven cause of the reported stall was established.** The concrete harness discrepancy worth isolating first is the explicit mesh mapper on single-chip uploads: it can execute device-side layout conversion that the completed single-chip runners avoid. This is a source-supported execution-path difference, **not a demonstrated defective kernel or confirmed root cause**.

The suggested allocation-order and mock-guard explanations do not survive comparison with the completed runners. The page table and cache dimensions are legal, including the final decode page. Do not change precision, page ownership, or decoder math on the strength of this investigation.

Inspection only: no hardware access, process attachment, signals, resets, target execution, or implementation changes. Startup contracts were read. Two xhigh inspection subagents independently examined harness preparation and paged attention. No removed Gemma implementation was read or copied. AutoDebug was not relaunched.

Paths prefixed `D/` mean `models/autoports/google_gemma_4_31b_it/`.

## Evidence and revision limits

Direct observations:

- `D/doc/multichip_decoder/sliding_rep4096.log` contains device initialization messages, ending at the subset-of-MMIO warning. It contains no `DONE 1`, exception, or operation marker. This does not locate the stall inside prefill: setup, uploads, prefill, decode, host reads, synchronization, and trace work precede `DONE 1`.
- `host_stack.txt` records process 266295's command, layer 0, seq4096, replicated mode, and the 262144-input fixture. It contains **no stack frames**.
- The initially empty `triage_baseline.txt` later acquired only system information. `triage_baseline_summary.txt` reports missing Inspector data and skipped device checks. There is no captured kernel/NoC/CB state supporting a device-side diagnosis.
- `smoke.json` reports seq33 sliding B1 prefill/decode PCC 0.999931/0.999956; `full_smoke.json` reports seq33 full B1 PCC 0.999990/0.999995 and 129 trace replays. The sliding artifact has no trace metrics, unlike the current runner; it is evidence of its recorded execution, not proof that every current harness step passed.
- Another actor edited the runner during investigation, adding faulthandler and phase markers. The original read had neither. The later `sliding_rep4096_retry.log` reaches `SETUP 1` but, at the inspected snapshot, no `PREFILL_START`. This retry is separate evidence; absence of its next marker during setup is not proof of a permanent stall or of the original run's location.

Inspected HEAD: `87140921cfc65695b74f0981b3c19b4182e46e61`; the new harness is untracked. Later instrumented harness SHA256: `7d64475cd3e9c1be37e29b48573f5425e00edc89bfe00e5d78e764750f12df7a`. Decoder SHA256: `5267f1ee57d0f077e71357445e96e15b3851b022c5846eeb10b77566eb4b6ff6`. Harness line references below refer to that instrumented version. Do not assume the live original process loaded later edits.

## Leading unverified candidate: baseline uploads take a different path

`D/tests/run_multichip_decoder.py:38-40` supplies `ReplicateTensorToMesh(mesh)` even for count1. By comparison, `run_optimized_decoder.py:92-93` and `profile_optimized_decoder.py:31-32` omit the mapper.

This is not merely a redundant distribution annotation:

1. `ttnn/ttnn/operations/core.py:397-409` forwards the unwrapped mapper into Tensor construction.
2. `ttnn/core/tensor/py_to_tt_tensor.cpp:264-289` handles explicit mapping through `create_distributed_tensor`. When device construction and capacity checks permit, it uses a **ROW_MAJOR source layout**, not the requested tiled layout.
3. The no-mapper branch explicitly excludes layout transformations from borrowed device construction (`:302-312`) and falls back to `Tensor::from_span` with the destination layout (`:320-325`).
4. Conversion then uploads and calls `to_layout` on device when needed (`:457-493`). Thus a mapped BF16 tiled request can perform device tilization; the ordinary single-device tiled upload is prepared on the host.
5. `to_layout/to_layout_op.cpp:129-162,222-228` distinguishes aligned tilize from tilize-with-padding. The4096-token residual has `[1,1,4096,5376]` geometry, 128 by168 tiles, and needs no tile padding. Seq33 has a nonaligned sequence dimension and uses a padding conversion when constructed on device. Passing seq33 therefore does not exercise an identical conversion signature.

The lower tilize path was also inspected. For a 120-core grid, the 168-by-128-tile input selects block tilize: 108 block cores versus 64 row-split cores (`tilize/device/tilize_device_operation.cpp:353-376`; `core/work_split/work_split_tilize.hpp:67-110`). The block regions are 15-by-15, 3-by-15, 15-by-8, and 3-by-8 tiles. Reader pushes the region width per row, compute consumes that width for the region height, and writer pops width times height tiles (`reader_unary_pad_multicore_both_dims.cpp:132-146`; block factory `:343-371`; `tilize_wh.cpp:27-34`; `writer_unary_interleaved_start_id_wh.cpp:37-45`). Blackhole fast tilize supports width15 as 4+4+4+3 and width3 directly (`tt_metal/hw/inc/api/compute/tilize.h:343-365`). No count mismatch was found. These are conditional source calculations, not a measured grid or captured running kernel.

This explains why upload-path parity must be established before blaming `prefill_forward`. It does **not** prove the conversion hangs. Eligibility depends on device state and dtype; no runtime op trace establishes where the reported process reached. A minimal controlled intervention is to omit the mapper for count1 only and leave all tensors, page IDs, forward calls, and precision unchanged. Separately synchronize after each upload to distinguish submission from completion. These are recommended follow-ups, not performed controls.

## Allocation, lifetime, and synchronization comparison

| Detail | New harness | Completed comparator | Assessment |
|---|---|---|---|
| Decode input allocated before prefill | Yes, lines49-50 | Profiler lines80-83 also does this | Not independently causal |
| Runtime conversion mocks | Both forwards, lines52-55 | Optimized runner lines153-165 wraps both, including an extra as_tensor guard | No demonstrated mock-induced hang |
| Retained full prefill output during first decode | Yes | Profiler deletes warm output; correctness runner retains prefill output until assigning decode output | Difference from profiler, not a proven lifetime violation |
| Read prefill before decode submission | No, reads both at line57 | Correctness runner reads prefill at lines178-189 | Important localization difference, not by itself missing synchronization |
| Full prefill host read | Yes | Correctness runner also reads all rows for seq<=4096 | Not an unsupported read size |
| Mesh shard extraction before read | get_device_tensors(t)[0] | Comparator reads mesh tensor directly | API returns a device-storage view, not an unowned copied buffer |

The decoder bounds temporary prefill work to 1024-token chunks (`optimized_decoder.py:531-553`). Each B1 chunk owns an L1 clone before in-place residual operations (`:571-575`). Returned chunk outputs are moved to DRAM (`:603-605`), then concatenated (`:554-556`). Retaining full input/output does not retain four chunks of L1 intermediates. A full BF16 residual is 44,040,192 bytes; a chunk is 11,010,048 bytes. This is DRAM output retention, not evidence of L1 exhaustion.

The harness keeps `px`, `pc`, `ps`, `dx`, `kw`, cache, page table, `pref`, and `dec` live across the first reads. No explicit deallocation or trace replay precedes those reads. `get_device_tensors` creates storage views (`ttnn/core/distributed/api.cpp:68-85`). There is no identified use-after-free in this sequence.

Both forward calls enqueue work before `host(pref)`; therefore a block in that first read can reflect **earlier decode work as well as prefill**. `to_torch` calls `from_device` (`operations/core.py:463-466`), which delegates to tensor CPU transfer (`operations/core/core.cpp:55-56`). The explicit synchronize after the two reads is not the first completion boundary. Later eager timing, trace capture/replay, and teardown introduce further waits. A host-side return from prefill or a phase print proves submission progress, not accelerator completion.

## Page/cache audit: no capacity contradiction

For B1, S4096, sliding attention:

| Quantity | Value |
|---|---|
| blocks=(S+32)//32 |129 |
| Page table |INT32 `[1,129]`, unique physical IDs3..131 |
| Each cache |BFP8 `[132,16,32,256]` |
| Prefill physical chunks |Four1024-token chunks |
| Q per chunk |`[1,32,1024,256]` |
| K/V fill input per chunk |`[1,16,1024,256]` |
| Fill table slices |columns0:32,32:64,64:96,96:128 |
| Attention table/start |First full129-entry table/start0; then64-entry slices/start1024 |
| Attention K extent |1024,2048,2048,2048 |
| Prefill Q/K chunk |128/128 |
| Decode position, K chunk |4096,32 |

`optimized_decoder.py:607-621` rebases the attention view only; cache writes and RoPE remain absolute. The scalar chunked SDPA wrapper selects the legacy scalar-start path (`sdpa/sdpa.cpp:105-142`). Its factory derives K extent from start plus Q length (`sdpa/device/sdpa_program_factory.cpp:592-618`), not129 times32 for the first chunk. Later views have all 64 required entries.

The table read uses the buffer's aligned page size (`sdpa_program_factory.cpp:167-171`); raw 516-byte row width need not itself be32-byte aligned. Row-major slicing accounts for aligned source/destination strides (`data_movement/slice/device/slice_program_factory_rm.cpp:285-306`).

Both fill and attention address physical tiles using `physical_block*128 + head*8 + tile_column`. Maximum index is `131*128 + 15*8 + 7 = 16895`, the last of 16896 cache tiles. IDs need not begin at zero. Sources: fill-cache `writer_fill_cache_interleaved.cpp:20-37` and SDPA `device/kernels/dataflow/dataflow_common.hpp:53-68`.

Decode explicitly chooses a chunk dividing owned capacity (`optimized_decoder.py:674-679`). Capacity 4128 selects 32, so the final read uses valid virtual page 128. The profiler's 132 pages give capacity 4224 and select 64; both configurations are legal, but they are different decode signatures. The earlier out-of-capacity decode finding is not applicable to current code.

For S33 the table has 2 entries and capacity 64. Prefill physically pads to 64 rows and the SDPA reader clamps/pads remaining rows rather than looking up nonexistent pages (`reader_interleaved.cpp:650-652`; shared dataflow helper `:280-284`). This reconciles the passing short case without inventing a page129 alignment defect.

## Candidate adjudication and next boundary to isolate

| Candidate | Direct contract violation? | Missing evidence / smallest useful control |
|---|---|---|
| Explicit mapper/device tilization | No; real setup-path difference | Prove upload completion; omit mapper only for baseline |
| Decode32 vs profiler64 | No; both capacities safe | If decode localized, supply 132 valid pages with initialized backing, preserving other settings |
| Early decode allocation | No; profiler shares ordering | Cannot headline without another interacting condition |
| Full output retention | No; DRAM, retained references | Phase-separated read/sync can localize, not prove a lifetime fix |
| Mocks | No; correctness runner also guards | Removing guards alone has no source-backed causal rationale |
| Invalid page IDs/rounded capacity | No in current path | Concrete dimension audit rules out simple form |
| Fabric/TP4 | Not reached before baseline completion | Cannot explain first-phase stall through TP4-only calls |
| Generic lower-level kernel failure | Unresolved | Requires identifying the selected running operation first |

Recommended sequence for the parent: preserve available failure evidence, identify the last completed setup/upload/forward stage with completion boundaries, then run the single-chip no-mapper control. Do not combine that control with changed page counts, precision, buffer deletion, or removed guards. If prefill completion is established and decode alone stalls, investigate its 32-token chunk signature next. If setup itself does not complete, decode allocation and forward guards cannot be the cause.

## Post-draft review

Rechecked the allocation-order contrast against profiler lines77-99; the mocks against correctness runner153-165; full output retention against decoder DRAM conversion; page ownership and chunk selection against the lowered factories/readers; and explicit mapper handling against the native constructor. None establishes a concrete producer/consumer violation on the reported path. The report therefore deliberately leaves root cause unresolved and labels the upload difference as a candidate. Later instrumentation and retry evidence were kept separate from the original uninstrumented run.
