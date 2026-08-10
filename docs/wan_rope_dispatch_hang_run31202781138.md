# Wan2.2 T2V-A14B unit tests — RoPE dispatch hang on wh_galaxy

Job: `models-unit-tests / TT-DiT Wan2.2-T2V-A14B unit tests [wh_galaxy]`.

Two occurrences so far, **both on runner `g03glx04`**:

| # | Run | Job | Branch | Commit / wheel | Date |
|---|---|---|---|---|---|
| A | [31202781138](https://github.com/tenstorrent/tt-metal/actions/runs/31202781138) | `92992728071` | `jameslee/add_rope_attention_wan_tests` | `1851dfc71d0` | 2026-07-31 |
| B | [31355919455](https://github.com/tenstorrent/tt-metal/actions/runs/31355919455) | [`93358239171`](https://github.com/tenstorrent/tt-metal/actions/runs/31355919455/job/93358239171) | **`main`** | `0.75.0rc10.dev232+g8f29d3a80f` | 2026-08-10 |

Occurrence B is on `main` from the scheduled nightly, so **this is not caused by the Wan test
branch**. Same param, same failure mode, same host.

Other attempts on run 31202781138: attempt 1 (job `92949695003`, `g03glx04`) — 1 failed / 213
passed, PCC only, no hang; attempt 3 (job `93000295346`, `OM1-01A01-STGWH03`).

## Symptom

`test_rope.py::test_wan_rotary_pos_embed[wormhole_b0-2x4sp0tp1nl1]` — the 8th of 13 params.
The preceding 7 pass. It hangs in the **host→device tensor upload**, not in the RoPE op:

```
RuntimeError: TT_THROW @ /work/tt_metal/impl/dispatch/system_memory_manager.cpp:724
TIMEOUT: device timeout in fetch queue wait, potential hang detected
```

The timeout is 5 s (`TT_METAL_OPERATION_TIMEOUT_SECONDS=5`). In occurrence B the test reached
`patchified input shape: torch.Size([1, 75600, 40, 128])` at 04:59:40.782 and the timeout fired at
04:59:47.640.

Note the AI job summary's suggested action ("investigate the WAN RoPE kernel for a hang … check if
the kernel correctly handles the 2x4 mesh topology") is misleading — the RoPE op never executed.

## Root cause signal: the math TRISC never got the go message

Occurrence B resolved a callstack for **trisc1**, which was missing in occurrence A. That is the
new and decisive piece of evidence.

On the single stuck core — device 20, NOC `9-8`, logical `(7,6)` — the five RISCs are:

| RISC | `_start()` frame | State |
|---|---|---|
| trisc0 (unpack) | `trisck.cc:89` → `run_kernel()` | launched, blocked in `mailbox_read` |
| **trisc1 (math)** | **`trisck.cc:83` → `wait_for_go_message()`** | **never launched** |
| trisc2 (pack) | `trisck.cc:89` → `run_kernel()` | launched, blocked in packer setup |
| ncrisc (reader) | — | blocked in `cb_reserve_back` |
| brisc (writer) | — | blocked in `cb_wait_front` |

`tt_metal/hw/firmware/src/tt-1xx/trisck.cc:83` is the `wait_for_go_message()` call; `:89` is
`run_kernel()`. So **two of the three compute threads on that core observed the go message and the
third did not.** The unpacker's `mailbox_read` is the TRISC-to-TRISC handshake waiting on precisely
that missing math thread.

Everything else is downstream fallout: unpack never completes → CBs never drain → reader blocks on
`cb_reserve_back` and writer on `cb_wait_front` → dispatch on device 20 sits in `process_wait` and
prefetch in `process_stall` → every other chip's dispatcher sits in `CBReader::acquire_pages` →
host times out reserving the fetch queue.

### Device callstacks (occurrence B, op 50, core `20:9-8 (7,6)`)

**trisc1 (math)** — the anomaly:

```
#0 0x00009C80 in wait_for_go_message () at tt_metal/hw/inc/internal/firmware_common.h 203:60
#1 _start () at tt_metal/hw/firmware/src/tt-1xx/trisck.cc 83:24
```

**trisc0 (unpack)** — waiting on the mailbox trisc1 never wrote:

```
#0  0x000097C4 in ckernel::mailbox_read () at tt_llk_wormhole_b0/common/inc/ckernel.h 557:13
#1  ckernel::unpacker::set_dst_write_addr () at tt_llk_wormhole_b0/common/inc/cunpack_common.h 995:57
#2  unpack_tilize_to_dest_impl () at tt_llk_wormhole_b0/llk_lib/llk_unpack_tilize.h 182:23
#3  _llk_unpack_tilize_ () at tt_llk_wormhole_b0/llk_lib/llk_unpack_tilize.h 289:35
#4  llk_unpack_tilize () at hw/ckernels/wormhole_b0/metal/llk_api/llk_unpack_tilize_api.h 71:24
#5  llk_unpack_tilize_block () at hw/ckernels/wormhole_b0/metal/llk_api/llk_unpack_tilize_api.h 99:26
#6  ckernel::tilize_block () at tt_metal/hw/inc/api/compute/tilize.h 150:5
#7  compute_kernel_lib::tilize<4, 0, 16, InitUninitMode(0), WaitMode(0),
        ReconfigureRegisterDatatypeMode(0), Fp32Mode(1)> () at ttnn/cpp/ttnn/kernel_lib/tilize_helpers.inl 201:25
#8  kernel_main () at ./ttnn/cpp/ttnn/kernel/compute/tilize.cpp 30:19
#9  run_kernel () at hw/ckernels/wormhole_b0/metal/common/chlkc_list.h 38:16
#10 _start () at tt_metal/hw/firmware/src/tt-1xx/trisck.cc 89:15
```

**trisc2 (pack)**:

```
#0 0x0000A3DC in ckernel::packer::program_packer_destination () at tt_llk_wormhole_b0/common/inc/cpack_common.h 873:5
#1 _llk_pack_<DstSync(0), true, PackMode(0)> () at tt_llk_wormhole_b0/llk_lib/llk_pack.h 475:31
#2 llk_pack<true, true, PackMode(0)> () at hw/ckernels/wormhole_b0/metal/llk_api/llk_pack_tile_api.h 62:62
#3 ckernel::tilize_block () at tt_metal/hw/inc/api/compute/tilize.h 161:9
#4 compute_kernel_lib::tilize<...> () at ttnn/cpp/ttnn/kernel_lib/tilize_helpers.inl 201:25
#5 kernel_main () at ./ttnn/cpp/ttnn/kernel/compute/tilize.cpp 30:19
#6 run_kernel () at hw/ckernels/wormhole_b0/metal/common/chlkc_list.h 38:16
#7 _start () at tt_metal/hw/firmware/src/tt-1xx/trisck.cc 89:15
```

**ncrisc (reader)** — CBs never drain:

```
#0 0xFFC002AC in cb_reserve_back () at tt_metal/hw/inc/api/dataflow/dataflow_api.h 419:19
#1 DataflowBuffer::reserve_back_impl () at hw/inc/internal/tt-1xx/dataflow_buffer.inl 81:20
#2 DataflowBuffer::reserve_back () at hw/inc/api/dataflow/dataflow_buffer.h 142:64
#3 kernel_main::tag(19)::operator() ()
     at ./ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/kernels/dataflow/reader_unary_pad_dims_split_rows_multicore.cpp 102:29
#4 kernel_main () at .../reader_unary_pad_dims_split_rows_multicore.cpp 160:27
#5 kernel_launch () at tt_metal/hw/firmware/src/tt-1xx/ncrisck.cc 72:16
```

**brisc (writer)**:

```
#0 0x00008C00 in reg_read () at tt_metal/hw/inc/internal/tt-1xx/risc_common.h 111:5
#1 cb_wait_front () at tt_metal/hw/inc/api/dataflow/dataflow_api.h 482:45
#2 DataflowBuffer::wait_front_impl () at hw/inc/internal/tt-1xx/dataflow_buffer.inl 97:18
#3 DataflowBuffer::wait_front () at hw/inc/api/dataflow/dataflow_buffer.h 144:60
#4 kernel_main () at ./ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp 45:23
#5 _start () at tt_metal/hw/firmware/src/tt-1xx/brisck.cc 80:20
```

### Host callstack — where the throw comes from

The upload path, via `TilizeWithValPadding` inside `bf16_tensor` / `bf16_tensor_2dshard`:

```
tt::tt_metal::SystemMemoryManager::fetch_queue_reserve_back(unsigned char)
tt::tt_metal::buffer_dispatch::issue_buffer_dispatch_command_sequence<InterleavedBufferWriteDispatchParams>(...)
tt::tt_metal::buffer_dispatch::write_interleaved_buffer_to_device(...)
tt::tt_metal::buffer_dispatch::write_to_device_buffer(...)
tt::tt_metal::distributed::FDMeshCommandQueue::write_shard_to_device(...)
```

### Host callstack — the teardown abort (exit 134)

Six seconds later the same timeout fires inside a destructor, where it cannot propagate:

```
tt::tt_metal::SystemMemoryManager::fetch_queue_reserve_back(unsigned char)
tt::tt_metal::event_dispatch::issue_record_event_commands(...)
tt::tt_metal::distributed::FDMeshCommandQueue::clear_expected_num_workers_completed()
tt::tt_metal::distributed::FDMeshCommandQueue::~FDMeshCommandQueue()
tt::tt_metal::distributed::MeshDeviceImpl::close_impl(MeshDevice*)
tt::tt_metal::distributed::MeshDevice::close()
```

```
terminate called after throwing an instance of 'std::runtime_error'
Fatal Python error: Aborted
  ttnn/distributed/distributed.py:689 in close_mesh_device
  /work/conftest.py:636 in mesh_device        ← fixture teardown
```

**Collateral damage:** the step runs `bash --noprofile --norc -eo pipefail`, so the SIGABRT ends the
whole script. `test_attention_wan.py`, `test_vae_wan2_1.py`, `test_transformer_wan.py` and
`test_pipeline_wan.py` never ran in either occurrence — the job result says nothing about them.

## The stuck op

Identical in both occurrences apart from the device. `dump_running_operations.py`:

```
Op 50  TilizeWithValPaddingDeviceOperation
       logical_shape: [1, 1, 37800, 128]   dtype: FLOAT32
       config: RowMajorPageConfig()  memory_layout: INTERLEAVED  buffer_type: DRAM
       Device Cnt 1   Core Cnt 1   Devices 20   Cores 20:9-8 (7,6)
```

This is the on-device tilize inside `bf16_tensor` / `bf16_tensor_2dshard` uploading the fp32 test
input (37800 = 75600 seq len ÷ 2 sp shards). Ops 51–53 (typecast / tilize / typecast) are queued
behind it in `dump_op_window.py`.

`dump_op_mesh.py` shows **one device busy, seven idle** — in B, device 20 at mesh coord (row 0,
col 2); in A, device 17 at (row 1, col 3). Both are Tray 2 chips of `g03glx04`.

## Correction: the `.text` mismatch is weaker evidence than it first looked

An earlier revision of this doc called `check_binary_integrity` "the one hard anomaly." Occurrence B
undercuts that.

| | Occurrence A (device 17) | Occurrence B (device 20) |
|---|---|---|
| trisc0 | — | `0x00009400` mismatch |
| trisc1 | `0x00009bb0` mismatch | `0x00009bb0` mismatch |
| cache root | `16138300672933534673` | `15763904200462748101` |
| kernel hash | `1330443098452927355` | `8259936983083445092` |

Two problems with reading this as memory corruption:

1. In B the check fires for **trisc0 as well**, and trisc0 is demonstrably executing the correct
   tilize kernel (full, sensible callstack at `trisck.cc:89`). So a reported mismatch is not
   sufficient to explain a stall.
2. The trisc1 offset is **byte-identical (`0x9bb0`) across two different builds** with different
   cache roots and different kernel hashes. Random corruption does not land on the same offset
   twice; a systematic artifact of the checker does.

`check_binary_integrity` compares L1 against whatever ELF `dispatcher_data` believes is loaded at
`kernel_offset` (`tools/triage/check_binary_integrity.py:99`), so a stale or mid-load launch message
can produce a false positive. Treat the mismatch as corroborating, not causal. **The go-message miss
is the better lead.**

## Everything else on the machine is clean

Occurrence B, all passing: `check_cb_inactive`, `check_eth_status`, `check_l1_status`,
`check_noc_locations`, `check_noc_status`, `check_core_magic`, `check_arc`,
`check_broken_components`, `dump_lightweight_asserts`, `dump_risc_debug_signals`.

Telemetry across all 8 devices: AI clk 1000 MHz, ARC clk 540 MHz, ASIC 38.3–40.6 °C, board 45 °C,
ETH `0xFFFFFFFF` (32 live), DDR `0x555` @ 14000 MT/s, ARC uptime 0:05:12, FW bundle 19.12.0,
KMD 2.10.0, IOMMU disabled.

## There are no watcher logs

The workflow exports `TT_METAL_WATCHER=2`, but the first line of the test script is:

```bash
unset TT_METAL_WATCHER  # blocked by #50886
```

The runtime config dump confirms it: `rtOptions watcher_enabled = false`, and
`dump_watcher_ringbuffer.py: pass` (empty). Real watcher coverage on these hangs is blocked on
\#50886.

What *was* captured is the tt-triage dump fired by the fast-dispatch timeout hook
(`TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE`). Artifacts:

| Occurrence | Artifacts |
|---|---|
| A | `triage_output_92992728071`, `triage_llm_output_92992728071`, `ai_job_summary_92992728071`, `test_reports_fcc09d7a-…` (`hang_report_3151f9e63fc129c2.xml`) |
| B | `triage_output_93358239171`, `triage_llm_output_93358239171`, `ai_job_summary_93358239171` |

## Prior art — issue #45517

[#45517](https://github.com/tenstorrent/tt-metal/issues/45517) — *"[GPT-OSS-120B,b=1/128,WHGLX]
TilizeDeviceOperation hang during weight preprocessing — kernel .text corruption on g14glx03"*.
Same class of failure:

| | #45517 | This issue |
|---|---|---|
| Host | `g14glx03` | `g03glx04` |
| Op | `TilizeDeviceOperation` | `TilizeWithValPaddingDeviceOperation` |
| Tensor | `[1, 1, 131072, 64]` FLOAT32 DRAM INTERLEAVED | `[1, 1, 37800, 128]` FLOAT32 DRAM INTERLEAVED |
| Stuck core | logical 9-10, physical (7,8) | NOC 9-8, logical (7,6) |
| Offsets | trisc0 `0x8950`, trisc1 `0x8fa0`, stable | trisc0 `0x9400`, trisc1 `0x9bb0`, stable |
| Devices | 23, then 17 (2 runs, same host) | 17, then 20 (2 runs, same host) |

It was **closed 2026-07-30** after ~18,200 stress executions on `wh-glx6u-04` / `wh-glx6u-08`
failed to reproduce, landing on *"H3 — runner-specific to `g14glx03`"* as the best-fitting
hypothesis, with the explicit note: *"if something similar comes up, new issue will be created."*

This is that follow-up, and it is on a **second host**, which weakens H3. What #45517 did not have
is the trisc1 callstack — that investigation inferred `.text` corruption; here we can see the math
thread parked in `wait_for_go_message()`.

## Is it the test, or the machine?

Evidence it is **not** the test:

1. The hung op is the generic tilize used to upload the input tensor — nothing Wan- or
   RoPE-specific about it.
2. Plain single-core interleaved path. No fabric, no CCL, no multi-chip collectives.
3. Only 1 of 8 devices hung; the other 7 finished the same mesh op and went idle.
4. The same param passed in attempt 1 of run 31202781138, on the same commit and same host.
5. Occurrence B is on `main`, unrelated to the Wan test branch.
6. The same signature appeared in a completely different model (#45517, GPT-OSS weight
   preprocessing).

Evidence it correlates with the **host**: both occurrences are `g03glx04`, both Tray 2, both logical
core `(7,6)`. Across the 9 nightly `main` runs of this job that could be queried (2026-08-05 →
2026-08-10): 4 success, 5 failure — but only occurrence B is this hang. The other failures are
`test_wan_attention` PCC failures (job `93059991848`, `OM1-01A01-STGWH03`) and job cancellations
(jobs `93199464715` / `92995627835`, `j09glx02`), on different runners.

So: correlates with `g03glx04`, but the same class of bug has now been seen on `g14glx03` too, and
the underlying mechanism (a TRISC missing its go message) is not obviously hardware-specific.

## Actions

- [ ] File a new issue referencing #45517, headlined on the **go-message miss** (trisc1 parked at
      `trisck.cc:83` while trisc0/trisc2 reached `:89`), not on the `.text` mismatch. Attach
      `triage_output_93358239171`. Note the second host, which weakens #45517's H3 closure.
- [ ] Loosen the conv3d PCC threshold (0.999_980 → ~0.999_900) in `test_vae_wan2_1.py:558` / `:712`;
      it is failing on rounding (see below).
- [ ] Split the five pytest invocations into separate workflow steps (or collect reports past a
      failure). With `set -e`, one SIGABRT in `test_rope.py` hid four other test files in both
      occurrences.
- [ ] Watcher coverage for these hangs is blocked on #50886.

## Appendix — run 31202781138 attempt 1, the real signal about the test code

Attempt 1 got through the whole suite: **1 failed, 213 passed, 448 skipped, 681 deselected**
in 1:12:58. The single failure:

```
FAILED models/tt_dit/tests/models/wan2_2/test_vae_wan2_1.py::
  test_wan_conv3d[wormhole_b0-line-bf16-4x8h0w1nl4-0-1-cache_1-conv_1]
  - Exception: PCC = 99.9980 % >= 99.9980 %
```

That's `assert_quality(..., pcc=0.999_980, relative_rmse=0.007)` in
`models/tt_dit/tests/models/wan2_2/test_vae_wan2_1.py:558` / `:712`. Measured PCC landed a hair
below a threshold sitting right at the bf16 noise floor — neighbouring configs in the same run
log 99.9972 %–99.9989 %. The confusing message text is just how
`models/tt_dit/utils/check.py:56` formats the failure (it prints the assertion that *should*
have held, not the comparison that failed).

Also worth noting from attempt 1, non-fatal: `test_wan_conv3d` logs
`Padding is not zero, but tensor(...)` from
`test_vae_wan2_1.py:convert_to_torch_channels_first:301` on the 4x8 line configs.
