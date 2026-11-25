# gpt-oss on the Quasar emulator: the layer-1 MoE hang

Working notes for review. Started 2026-11-25 01:30 UTC while the user is offline; updated as the work goes.
Nothing here was committed or pushed.

## 1. Where things stood at the start

The goal is the gpt-oss end-to-end demo (2 layers, prefill 128, 3 decode tokens) passing on the Quasar RTL emulator
(`emu-quasar-2x3`, 2 Tensix clusters). The emulator backs only 64 MiB of DRAM per channel, so all runs use
`TT_METAL_SIMULATOR_DRAM_UNCAPPED=1` and a descriptor copy that declares more DRAM; data above 64 MiB per channel
aliases, so output values are not meaningful and the run checks only that nothing hangs.

| Try | Change | Result |
|---|---|---|
| 1 | descriptor as shipped (1 bank, 1 GiB) | OOM on the 1.16 GB embedding table |
| 2 | 1 bank, 4 GiB | OOM on layer 1 `down_proj` (needed 0.53 GB, 0.33 GB free) |
| 3 | 1 bank, 8 GiB | first attempt lost to `/proj_sw` full; rerun: `TT_FATAL` "Cannot write to reserved DRAM region": `WriteToDeviceDRAMChannel` takes a `uint32_t` address, so a bank above 4 GiB cannot work |
| 4 | 2 banks x 3 GiB (`emu-quasar-2x3_2ch3g`) | all weights load; prefill trips the `sparse_matmul` nnz assert: the constant prefill sparsity mask, held in DRAM, was overwritten by aliased weight uploads |
| 5 | + prefill sparsity mask and its repeat in L1 | past the sparse matmul; trips the TEN-4746 LLK assert in the binary_ng held-operand path (`hold_dfb`) after 3 h of the compile-only prefill |
| 6 | + compile-only prefill skipped, 24 h remote limit, watcher and LLK asserts off, binary_ng fix stashed | weights 2 h 43 min; layer 0 completes in 44 min; **hangs** at layer 1's first `sparse_matmul`, in the `zeros_like` fill of its output (`wait_until_cores_done`), no progress for 2.5 h |
| 7 | try 6 + watcher on (LLK asserts off), to see the hang | running |

Local, uncommitted changes in the tree for these runs (first marked "TEMP (emulator run, do not commit)"; on
2026-11-25 the user decided to keep them, and the comments now describe them as permanent):

- `models/experimental/ops/quasar/gpt_oss/tt/experts/__init__.py`: prefill sparsity mask created in L1.
- `models/experimental/ops/quasar/gpt_oss/tt/experts/prefill.py`: `ttnn.repeat` of the mask into L1.
- `models/experimental/ops/quasar/gpt_oss/demo/text_demo.py`: `DEMO_SKIP_COMPILE_PREFILL=1` skips the compile-only prefill.
- The binary_ng TEN-4746 fix (`dummy_unpack` before the credit-only pop in `hold_dfb`) is **stashed** (`stash@{0}`)
  and saved as `/localdev/fplavec/temp/gpt_oss_ops_runs/patches_20261124/binary_ng_hold_dfb_ten4746.patch`.

## 2. What was established before try 7

| Run | Result |
|---|---|
| ttsim (ttsim-private `qsr` at `bdab0349`), 4 small-grid `sparse_matmul` cases incl. both prefill shapes, 2x1 grid | all pass |
| ttsim, full 2-layer demo, same tree and settings as try 6 | **passes** in 39 min 47 s (prefill 29 min, 3 decode tokens) |
| emulator, the layer-1 gate projection call alone (`test_sparse_matmul_small_grid[02_4x32x2880...]`) | **no hang**; fill and matmul complete in 38 min; fails only the finite-values check (aliased weights) |

So the hang needs both the emulator and the ops that ran before it in the demo.

## 3. Investigation log

### 3.1 Try 7 on the emulator (watcher on, LLK asserts off), launched 2026-11-24 22:16 UTC

Weights loaded 22:19 to 01:03 UTC (2 h 44 min); prefill started 01:03 with the compile-only prefill skipped. At 01:17
both cores ran layer 0's expert `sparse_matmul` (kernel ids 82 to 85) with waypoints changing between dumps.

**Result: lost to the emulation farm, not a device hang.** The watcher's last dump is #3150 at 01:43 UTC; the host's
main thread then sat in `RtlSimChannel::wait_for_response()` (one pending read to the emulator, inside the
`wait_until_cores_done` poll of a `permute` op). The launcher log (`emu_2026-11-24_22-16_.log` in the repo root) shows
why: the scheduler's per-minute handle refresh for the job stops at 01:39 UTC, and at 01:55
`vovsh: Timed-out on socket read ... Disconnected from the server ... Cannot find job 055667051`; the remote
`test_umd_remote` then ends. The watcher dumps up to 01:43 still show both cores' waypoints changing, so the cores were
working when the link went. The ZeBu monitor (`soc-zebu-01:8766`) also stopped answering at about 02:55 UTC. Try 7's
local process was stopped at 02:58; nothing of ours is left on soc-l-04.

**Try 6 was different:** its handle refreshes continue until 20:49 UTC (the dashboard kill), and its host process kept
using CPU (10:54 to 16:25 min of CPU time over 85 min), so it was polling and getting answers: the cores really never
reported the fill program done.

### 3.2 Two ttsim experiments in parallel (started 2026-11-25 01:39 UTC)

Neither touches the emulator or the tt-metal source tree, so try 7 (which JIT-compiles from that tree) is unaffected.

1. **ttsim demo with the watcher on** (NoC sanitize on, LLK asserts off). No earlier ttsim run had the watcher on, so a
   bad NoC address or a kernel assert could have gone unseen. Script:
   `/localdev/fplavec/temp/gpt_oss_ops_runs/20261124_demo_ttsim/watcher_run/run.sh`; watcher log under its own
   `logs/` folder (`TT_METAL_LOGS_PATH`) so it does not collide with try 7's.
2. **ttsim demo with DRAM aliased the way the emulator aliases it.** A local, unpushed ttsim-private branch
   `fplavec/dram-alias-experiment` (worktree `/localdev/fplavec/ttsim-alias`) adds `TTSIM_DRAM_ALIAS_BYTES=N`: every
   DRAM channel then backs only N bytes and addresses wrap modulo N, in `tile_rd_bytes` / `tile_wr_bytes`, the only two
   places that touch ttsim's DRAM array. Run with N = 64 MiB, watcher off, otherwise as try 6. If the hang is caused by
   aliased DRAM, this should reproduce it in under an hour with full visibility. Staged at
   `/localdev/fplavec/ttsim_qsr_3g_alias/`; script `.../20261124_demo_ttsim/alias_run/run.sh`.

Results:

- The **watcher run** was far too slow: still uploading weights after 75 min (ttsim serves the 1 s watcher poll and the
  uploads from one thread). Stopped at 02:58 UTC.
- The **aliased run fails** at 02:06 UTC, in layer 1's prefill SDPA (the last SDPA kernel compile before the error is the
  same count as in the clean run), with
  `ERROR: UnsupportedFunctionality: tile_rd_bytes: tile=T1 addr=0xff140a00 size=2048`: a kernel issued a 2 KiB NoC read
  of Tensix tile T1's register space, far outside its 4 MiB L1. The unaliased ttsim demo with the same tree passes, so
  aliased DRAM makes a kernel compute a bogus NoC address. On the RTL a read or write of register space can wedge a core
  instead of erroring, which fits try 6: it stalled in the program right after layer 1's attention. Prefill SDPA is not
  paged (`attention/prefill.py:147`); its reader (`sdpa/.../reader_interleaved.cpp`) has several tensor-fed inputs
  (`chunk_start_idx`, windowed offsets, page table) to check.
- Next: rerun the aliased demo with the watcher at a 60 s interval and NoC sanitize on, so tt-metal reports the core,
  kernel and address before the bogus read issues (`.../20261124_demo_ttsim/alias_watcher_run/run.sh`, started 02:50).

Hypotheses checked while it runs:

- **Correction on the failing op.** The last kernel compile before the error was layer 1's SDPA (01:56), but the error
  came at 02:06, and every op after layer 1's SDPA reuses kernels compiled for layer 0. So the failing op is somewhere
  after layer 1's SDPA, the same stretch where try 6 stalled, not necessarily SDPA itself. Layer 1's SDPA reader runs
  causal + attention sink, unpaged, unchunked, no sliding window, no provided mask (its compile-time args in the JIT
  cache): none of its addresses come from tensor contents.
- **Kernel code in aliased DRAM: ruled out.** Quasar DM kernels execute in place from L1 (`MEM_DM_GLOBAL_BASE`,
  `tt_metal/hw/inc/internal/tt-2xx/quasar/dev_mem_map.h`), and slow dispatch writes binaries to L1; the DRAM kernel
  binary buffer (`ProgramImpl::allocate_kernel_bin_buf_on_device`) serves fast dispatch only. The DRAM reserved region
  (`qa_hal.cpp`: barrier word, device-print ring for dispatch_s, optional DRAM-backed CQ) holds nothing a slow-dispatch
  kernel reads its addresses from.
- **Router indices in aliased DRAM: ruled out.** With 128 prefill tokens `TopKRouter` puts the router logits in L1
  (`actual_tokens <= 128`), and `topk`, `zeros_like`, `softmax` and `scatter` inherit that.
- The aliased run with the watcher (60 s interval, NoC sanitize on) never got past the weight upload: no log line from
  02:50 to 04:24 UTC. The watcher does not work for this on ttsim; stopped at 04:24.

### 3.3 Instrumented aliasing ttsim (started 2026-11-25 04:26 UTC)

The experimental ttsim branch now also prints, for any RoCC NoC read or unicast write that targets a Tensix tile outside
its 4 MiB L1 with more than 8 bytes (`rocc_cmdbuf_issue_trans` in `tile.cpp`, the path Quasar DM cores use): the
requesting tile, the RISC-V core (`riscv_id`), the command buffer, that core's PC, the target tile and both addresses
(`TTSIM_DBG bad read|write ...`). The PC maps to a kernel through the ELFs in the run's JIT cache. Run:
`.../20261124_demo_ttsim/alias_run2/run.sh`, otherwise as the first aliased run.

Result (runs `alias_run2` and `alias_run3`, deterministic at simulator cycle 106186776):

```
TTSIM_DBG bad read: requester=T1 riscv=0x80000002 (DM2) cmd_buf=1 pc=0x39000 target=T0 remote_addr=0x1f13ea00 local_addr=0x51840 len=2048
```

(`alias_run2` had target T1, `0x7f13f200`: same instruction, different garbage address.) The L1 words at the PC match
`reader_reshape_tiled/.../dm2/dm2.elf` at file offset 0xf78 (code address 0x400500), which `addr2line` resolves to a
`noc_async_read` inlined through `enhanced_noc_async_read` (`data_movement/common/kernels/common.hpp:43`) in
`ttnn/cpp/ttnn/operations/data_movement/reshape_view/device/device/dataflow/reader_reshape_tiled.cpp:40`. So the bad
read is not SDPA's: it is a **tiled `reshape`** after layer 1's attention.

**Root cause (on aliased DRAM): the tiled reshape's page-mapping tensor is host-built, kept in DRAM, and reused by the
cached program.** `reshape_tiled_program_factory.cpp` (both `data_movement/reshape_view` and the
`experimental/quasar/reshape_view` port) computes a table of `SegmentMapData` (input page index, offsets, element
count) per output page on the host, uploads it with `.to_device(device)` (default `MemoryConfig()`, DRAM interleaved),
and keeps it owned by the cached program ("device allocation outlives the cached program"). The reader reads one map page
per output page (line 40, `Max_Map_Size_Bytes` = 2048 here) and then reads the input pages the map names. In layer 0
the program is created and the map is fresh. Everything later written to DRAM aliases at 64 MiB, so by layer 1, when the
same reshape hits the program cache, its map holds garbage; the reader's input page index is then garbage, and an
L1-interleaved input's page address lands on a worker tile (T0 or T1) far outside L1. ttsim errors immediately; on the
RTL such an access can wedge the core, so the next program never completes, which is what try 6 showed (layer 0 fine,
layer 1 stalled in the program after its attention). (The read and the input-page reads share the inlined helper, so
the PC cannot tell which of the two 2 KiB reads it is; the garbage map explains the input-page one directly.)

This is the same pattern as try 4's `sparse_matmul` mask: control data stored in DRAM early and reused later. Other
candidates in the same class: the page table (created once in the demo and reused by every decode step's
`paged_update_cache` / paged SDPA decode, `models/experimental/ops/quasar/gpt_oss/tt/model.py:1133-1180`, default DRAM).

**Test without a rebuild** (the reshape fix is C++, and the user's standing rule is no rebuild without asking): an
opt-in pytest plugin `/localdev/fplavec/temp/gpt_oss_ops_runs/g_plugins/no_program_cache.py` (`DEMO_NO_PROGRAM_CACHE=1`)
calls `mesh_device.disable_and_clear_program_cache()` before the test body, so every reshape uploads a fresh map right
before it runs. Aliased ttsim run with it: `.../20261124_demo_ttsim/alias_nocache/run.sh`, started 05:22 UTC.

**Result: the aliased ttsim demo passes with the program cache off** (`alias_nocache`, 05:22 to 06:03 UTC): prefill
through both layers, the first token, and all 3 decode tokens, `1 passed in 2451.39s (0:40:51)`. With the cache on the
same run fails in layer 1 (above). So the stale reshape map is the only thing in this run that trips over aliased DRAM;
the decode page table, reused across decode steps, does not.

Two ways to apply this on the emulator:

1. **No rebuild (used for try 9):** the `no_program_cache` plugin, now wired into
   `/localdev/fplavec/temp/gpt_oss_ops_runs/20261119_emu/run_demo_emu2.sh` (`-p no_program_cache`, active only with
   `DEMO_NO_PROGRAM_CACHE=1`). Host-side only; every device kernel is unchanged.
2. **Proper fix, for review, not applied:** keep the map in L1, as the pad op already does for its host-built tables
   (`.to_device(device, MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::L1})`), in both reshape factories:
   `/localdev/fplavec/temp/gpt_oss_ops_runs/patches_20261124/reshape_mapping_tensor_l1.patch` (`git apply --check`
   passes). Needs a rebuild. Open question for review: L1 use of cached maps across many reshape shapes on the 2-core
   grid.

### 3.4 Try 8 on the emulator (queued 2026-11-25 04:25 UTC)

`soc-zebu-01` came back at about 04:20 UTC (352 jobs queued). The user killed try 7's orphaned job (still listed as
Running on soc-zebu-02 after 6 h). Try 8 = try 7's configuration (watcher on, LLK asserts off) with
`EMULATOR_TIMEOUT=129600` (36 h) to allow for the queue, job 055696254. Watching for a scheduler disconnect as well.

It connected at 04:37 UTC on soc-c-28 and is loading weights (prefill expected about 07:20). It runs with the program
cache **on**, so it is expected to stall in layer 1's tiled reshape, with the watcher showing the core in
`reader_reshape_tiled`: an RTL confirmation of the root cause. It could not be switched to the cache-off setting:
stopping its local side would orphan its farm job (only a dashboard kill releases it), and disabling the cache inside
the running process would need code injected through gdb during the weight upload.

**Result: try 8 reproduces the hang on the RTL, inside the tiled reshape.** Weights loaded by 07:36 UTC (3 h), prefill
started 07:36. Layer 0's MoE kernels compiled at 09:46 to 09:49 (including `reader_reshape_tiled`), layer 0's
`reduce_nc` at 10:54 (the point try 6 reached at 17:58, about 4x slower under the watcher), layer 1's SDPA at 10:56. From
watcher dump #19127 (about 12:49 UTC) both cores stay in one state while the dumps keep coming (37 min and counting at
13:26):

```
Device 0 worker core(x= 0,y= 0): NTW, W1, RBW, WFD, ... k_ids: 0| 0|120|121| ...
Device 0 worker core(x= 1,y= 0): NTW, W1, RBW, WFD, ... k_ids: 0| 0|120|121| ...
k_id[120]: ttnn/cpp/ttnn/operations/data_movement/reshape_view/device/device/dataflow/reader_reshape_tiled.cpp
k_id[121]: ttnn/cpp/ttnn/operations/data_movement/reshape_view/device/device/dataflow/writer_reshape_tiled.cpp
```

DM2 (the reshape reader) waits on a NoC read barrier (`RBW`) that never completes; DM3 (the writer) waits for its data
(`WFD`). This is the read that ttsim reports as a garbage-address access in the aliased run: on the RTL it never returns.
Same layer and position as try 6's stall. Logs: `.../20261119_emu/demo.log` (try 8), the watcher log in
`generated/watcher/watcher.log`.

## Why the demo hangs on the emulator, and why turning off the program cache fixes it

### The emulator has less DRAM than the model needs

The Quasar emulator used here runs the chip's hardware design (its RTL, register-transfer level description) on
emulation hardware. Its configuration, `emu-quasar-2x3`, is a 2x3 grid of tiles: two tiles hold the two DRAM channels,
two are compute cores, and its device descriptor lists no role for the other two. It stores only 64 MiB per DRAM
channel. The gpt-oss demo needs several GiB for its weights. To let the demo run at all,
we give tt-metal a copy of the emulator's device descriptor file (`soc_descriptor.yaml`) that declares each of the two
channels as 3 GiB, and set `TT_METAL_SIMULATOR_DRAM_UNCAPPED=1` so that tt-metal uses that size.

Addresses above 64 MiB wrap around: an address and that address plus any multiple of 64 MiB reach the same storage.
This is called aliasing. For example, in one channel, a write to offset 100 MiB changes what a later read of offset
36 MiB returns. (This wraparound was measured on the same emulator during earlier work on another model, Qwen3-VL:
section S1 of [QUASAR_GAPS.md](../qwen3_vl/tests/e2e/QUASAR_GAPS.md).) Each channel is declared as 3 GiB but stores
64 MiB, so each stored location is shared by 48 addresses. Once more than 64 MiB of a channel is in use, writes to one
tensor overwrite parts of others.

For most data this only makes values wrong. That is acceptable here, because the goal is only to check that the demo
runs to the end without hanging, not that its output is correct. It causes hangs when a kernel reads addresses or sizes
from DRAM: corrupted values send the kernel to wrong addresses or make it copy wrong amounts.

### The tiled reshape reads its addresses from a table in DRAM

`ttnn.reshape` on a tensor in tiled layout (stored as 32x32 blocks called tiles) moves data between tiles: each output
tile takes its elements from one or more input tiles. Which input tiles, and which elements of each, depends only on the
input and output shapes.

The host works this out once and stores the result as a table: for each output tile, a list of segments, where each
segment gives the input tile number, the offset in the input tile, the offset in the output tile, and the number of
elements. The host then copies this table to the device with `.to_device(device)`. With no memory configuration given,
the table goes to DRAM
([reshape_tiled_program_factory.cpp:299-302](../../../../../ttnn/cpp/ttnn/operations/data_movement/reshape_view/device/reshape_tiled_program_factory.cpp#L299-L302)).

On each core, two kernels use the table. They pass data through two dataflow buffers, small queues in the core's local
SRAM (called L1): the reader fills them, and the writer empties them.

1. The reader
   ([reader_reshape_tiled.cpp:38-64](../../../../../ttnn/cpp/ttnn/operations/data_movement/reshape_view/device/device/dataflow/reader_reshape_tiled.cpp#L38-L64)),
   for each output tile, reads that tile's table entry from DRAM and sends it to the writer. Then, for each segment, it
   takes the input tile number, computes that tile's address, reads the tile, and sends it to the writer.
2. The writer
   ([writer_reshape_tiled.cpp:34-70](../../../../../ttnn/cpp/ttnn/operations/data_movement/reshape_view/device/device/dataflow/writer_reshape_tiled.cpp#L34-L70)),
   for each segment in the same table entry, takes the input tile the reader sent, copies the given number of elements
   from the given offset in that tile to the given offset in a scratch tile in L1, and then writes the finished output
   tile to the output tensor.

So the table's contents decide which addresses the reader reads and which memory the writer copies to and from. If the
table holds garbage, both kernels work from garbage tile numbers, offsets and sizes.

### The program cache keeps the table for later calls

Building a program for an op (choosing kernels, compiling them, setting their arguments) is slow, so ttnn keeps built
programs in a program cache. When an op runs again with the same shapes, ttnn reuses the cached program and only
updates the addresses of its input and output tensors.

The reshape's table belongs to the cached program. The reshape's host code keeps the table allocated for as long as the
program is cached, and reuses it on every cache hit, on the reasoning that the table depends only on the shapes, which
are part of the cache key
([reshape_tiled_program_factory.cpp:314-321](../../../../../ttnn/cpp/ttnn/operations/data_movement/reshape_view/device/reshape_tiled_program_factory.cpp#L314-L321)).
That reasoning holds on real hardware, where nothing else writes to the table's memory. On the emulator it fails: the
table stays in DRAM for the rest of the run, and later writes to addresses that wrap around to the same storage
overwrite it.

### What happens in the demo

The demo runs two decoder layers, and both layers make the same reshape calls with the same shapes.

1. Layer 0 makes a reshape call for the first time. ttnn builds the program and writes the table to DRAM. The table is
   correct, and the reshape works.
2. The rest of layer 0 and the start of layer 1 write many tensors to DRAM. These writes can land on storage shared with
   the table and overwrite it. We did not observe this directly; the evidence is the failures below and the result of
   turning the cache off in the next section.
3. Layer 1 makes the same reshape call. This is a cache hit, so ttnn reuses the program and its table, which now holds
   garbage. Both kernels work from garbage.

The failure looked different in each of the three runs below. They used different settings, and the garbage differs
from run to run (two ttsim runs with identical settings failed at the same instruction with different addresses).

- **Emulator, with Watcher.** Watcher is tt-metal's debug tool that records, for each kernel, the last checkpoint
  ("waypoint") it reached. Both cores running the reshape stayed in it for over 37 minutes, after the attention block of
  layer 1. On each core the writer's last waypoint showed it had received data from the reader and then made no more
  progress; the reader's showed it was waiting for free space in a dataflow buffer that the writer never freed.
  Likely cause, not confirmed: the writer is copying with a corrupted offset or size.
- **Emulator, without Watcher.** The run stalled in the same part of layer 1, but not in a reshape. A host stack trace
  (taken with gdb) showed the host waiting for the cores to finish zeroing the output buffer of the first matrix multiply
  in the mixture-of-experts block. Likely cause, not confirmed: a reshape earlier in layer 1 ran with a garbage table,
  finished, and its garbage copies wrote over core memory that a later program needs.
- **ttsim with simulated wraparound.** ttsim is a software simulator of the chip that runs on a CPU. With a local,
  uncommitted change that makes its DRAM wrap around at 64 MiB like the emulator's, the run stopped in the same part of
  layer 1 with an error: a 2 KiB read (one tile of bfloat16 values) at an offset far beyond the 4 MiB L1 of a core.
  The read named a core, not a DRAM channel, which means this reshape's input tiles are stored in the cores' L1: there,
  an input tile's address is a core plus an offset in its L1, and the reader computed it from a garbage tile number.
  The code at the reading core's program counter matched the reshape reader. Without the wraparound change, the same demo passes on ttsim.

### Why turning off the program cache fixes it

With the program cache off, ttnn builds every op's program again on every call. For the reshape, the host then computes
the table and writes it to DRAM again, just before the kernels read it. No earlier op's writes can corrupt it, so both
kernels work from correct tile numbers, offsets and sizes. Only the host's behavior changes; the kernels are identical.

On ttsim with 64 MiB wraparound, the demo fails in layer 1 with the program cache on, and passes in full (processing the
prompt and generating 3 tokens, 40 min 51 s) with it off. On the emulator the demo passes in full with the cache off
(try 9, 4 h 45 min), where tries 6 and 8 stalled in layer 1 with it on.

Setting the environment variable `DEMO_NO_PROGRAM_CACHE=1` activates a local pytest plugin, not in the repository
(`/localdev/fplavec/temp/gpt_oss_ops_runs/g_plugins/no_program_cache.py` on the run host), that calls
`mesh_device.disable_and_clear_program_cache()` before the test runs.

### Limits of this workaround

A write made by the reshape itself, between writing the table and reading it, could still land on the table. The ttsim
run passed, so this apparently did not happen there, but nothing prevents it. The fix for the reshape is to put the
table in L1, which the emulator stores in full (the pad op already keeps its host-built tables in L1). A patch that does
this for both versions of the reshape program (`data_movement/reshape_view` and its Quasar copy in
`experimental/quasar/reshape_view`) is prepared but not applied, because testing it needs tt-metal rebuilt. Whether the
tables of all cached reshapes fit in L1 next to the other buffers is not checked. Other ops that keep host-built tables
in DRAM would still be exposed.

## 4. Next step: try 9 (program cache off)

Once try 8's job is gone from the dashboard (if it stalls as expected, it needs a dashboard kill), clear the remote run
folder and launch:

```bash
B=/localdev/fplavec/temp/gpt_oss_ops_runs/20261119_emu
DEMO_SIMULATOR=/proj_sw/user_dev/fplavec/emu-quasar-2x3_2ch3g/ DEMO_EMULATOR_TIMEOUT=129600 \
DEMO_SKIP_COMPILE_PREFILL=1 DEMO_NO_PROGRAM_CACHE=1 \
    setsid nohup $B/run_demo_emu2.sh > $B/run_demo_emu.out 2>&1 < /dev/null &
```

Watcher and LLK asserts off (the default), the 2 x 3 GiB descriptor, the L1 prefill sparsity mask, the compile-only
prefill skipped, the program cache off. The farm queue was 1487 jobs at 06:05 UTC.

The user killed try 8's job at about 14:00 UTC; try 8's local side was stopped and its logs kept as
`demo_2ch3g_try8_reshape_hang.log` and `watcher_try8_reshape_hang.log` in `/localdev/fplavec/temp/gpt_oss_ops_runs/20261119_emu/`.
**Try 9 launched at 14:02 UTC** with the command above (queue about 1186 jobs).

**Result: try 9 passes.** Connected 14:07 UTC (the log confirms "program cache disabled on the mesh device"); weights
loaded by 16:57; layer 0 done and layer 1's SDPA compiled by 17:39 (42 min, the pace of try 6); prefill finished at
18:25 (1 h 28 min) with the first token; all 3 decode tokens done by 18:48; `1 passed in 17122.47s (4:45:22)`,
`EXIT=0`. The remote job ended cleanly (nothing left on the dashboard or on soc-l-04). Log:
`/localdev/fplavec/temp/gpt_oss_ops_runs/20261119_emu/demo.log`.

Changes that the passing run depends on, all local and uncommitted:

- a descriptor copy with 2 DRAM channels of 3 GiB (`/proj_sw/user_dev/fplavec/emu-quasar-2x3_2ch3g/`) and
  `TT_METAL_SIMULATOR_DRAM_UNCAPPED=1`;
- the prefill sparsity mask and its repeat in L1 (`tt/experts/__init__.py`, `tt/experts/prefill.py`, marked TEMP);
- the program cache off (`no_program_cache` plugin, `DEMO_NO_PROGRAM_CACHE=1`);
- the compile-only prefill skipped (`DEMO_SKIP_COMPILE_PREFILL=1`, `demo/text_demo.py`, marked TEMP), to save
  emulator time;
- watcher and LLK asserts off; the binary_ng TEN-4746 fix stashed (`stash@{0}`), as it only matters with LLK asserts on.
