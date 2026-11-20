*** THIS TREE AND ITS BUILD ARE PATCHED -- 2026-11-02 ***

Patch: experimental/quasar/tilize/device/tilize_device_operation.cpp
       select_program_factory() -- the two returns of
       ttnn::prim::qsr::TilizeMultiCoreBlockProgramFactory (the !enough_space_height
       branch and the threshold_row_block branch) now return
       ttnn::prim::qsr::TilizeMultiCoreDefaultProgramFactory instead.
Why:   the Block factory binds g.stage as both PRODUCER and CONSUMER on a
       data-movement kernel, which Gen2/Quasar rejects at program build
       (program_spec.cpp:1502, "DataflowBuffer 'stage_*' is self-looped").

Deployed to the SHARED libraries at 2026-11-02 23:10:39 UTC:
   tt-forge-onnx/third_party/tt-mlir/build/install/lib/{libtt_metal,_ttnncpp,_ttnn}.so
   tt-metal-src/build_Release/lib/{same three}
ANY PROCESS STARTED AFTER THAT TIME LINKS THIS PATCH, whether or not its owner
knows about it. Several concurrent sessions did; see the hazard section of
quasar_models/TILIZE_FIX_RESULT.txt.

Validated: Llama-3.2-1B seq32 15/60 -> 52/60 (all 30 self-loop aborts cleared);
           ResNet-50 plan_par_ttsim unchanged at 120/120 of 122, pcc 0.999823.
NOT validated: wide WIDTH-SHARDED tilize (see TILIZE_FIX_RESULT.txt M1c).

TO RESTORE THE PRE-PATCH BUILD, one command:
   bash /proj_sw/user_dev/ctr-lelanchelian/lab_17_forge_onxx/TILIZE_FIX_BACKUP_20261102/RESTORE.sh
Full write-up: quasar_models/TILIZE_FIX_RESULT.txt


================================================================================
*** LAYER 3 ADDED -- 2026-11-03 02:07 UTC -- interleaved_to_sharded scratchpad ***
================================================================================
This tree now carries THREE independent patch layers. They revert SEPARATELY.

Patch (layer 3): two files, the SECOND producer of program_spec.cpp:1502.
  ttnn/.../data_movement/sharded/interleaved_to_sharded/device/
      interleaved_to_sharded_program_factory.cpp
        The "scratch" staging buffer was a DataflowBufferSpec bound as both
        PRODUCER and CONSUMER on the row-major 'reader' (a DM kernel). It is now
        a ScratchpadSpec with ONE ScratchpadBinding (accessor "pad"). Same bytes,
        no producer/consumer edge.
  ttnn/.../data_movement/sharded/device/kernels/dataflow/
      reader_unary_stick_layout_sharded_blocks_interleaved_start_id_metal2.cpp
        Uses Scratchpad<uint32_t>(scratch::pad) + CoreLocalMem instead of
        DataflowBuffer(dfb::scratch).  (JIT-compiled from source: a stale
        TT_METAL_CACHE can still serve the OLD kernel. Clear it.)
  Every hunk is tagged in-source with:  [Quasar bring-up: i2s scratchpad]
  Ported from the already-correct in-tree sibling
    experimental/quasar/interleaved_to_sharded/  (I2S_SCRATCH_PAD / scratch::pad).

Why:   same defect class as layer 2, DIFFERENT producer. Layer 2 fixed the
       tilize 'stage_*' DFB; this fixes the i2s 'scratch' DFB. The tilize
       selector cannot reach this one -- it is in the MAINLINE tree.

Deployed to the SHARED libraries at 2026-11-03 02:07:05-02:07:23 UTC.
  ONLY _ttnncpp.so changed:  f5c3d14c... -> 260c6c07...
  libtt_metal.so (1eae9aad), _ttnn.so (e2aa0ef6) and tt-mlir's
  libTTMLIRRuntime.so (bd96ff34) are BYTE-IDENTICAL to before.
  No other session was running at deploy time (pgrep verified 02:06:37).

MEASURED (ttsim only; ZeBu never contacted):
  PDL  186/198, pcc 0.999849  -- IDENTICAL to the layer-2 baseline.
       program_spec.cpp:1502 on PDL: 3 -> 0.  The 3 Resize segments cleared the
       self-loop and immediately hit the NEXT wall, kernel.hpp:451
       "DataMovementKernel is not supported on Quasar".  So the fault is fixed
       but buys ZERO segments today.
  RS50 120/120 of 122, pcc 0.999823, top1 610 -- BIT-IDENTICAL. No regression.
  PDL failure set: the same 12 segments, same ops; only the Resize FAULT CLASS
  changed (1502-selfloop -> 451-unported). Zero pass->fail.

TO REVERT **ONLY LAYER 3**, one command:
   bash /proj_sw/user_dev/ctr-lelanchelian/lab_17_forge_onxx/I2S_FIX_BACKUP_20261103/REVERT_I2S.sh
   (Do NOT use TILIZE_FIX_BACKUP_20261102/RESTORE.sh for this -- that reverts
    layer 2 as well.)
Full write-up: quasar_models/INTERLEAVED_TO_SHARDED_FIX.txt


================================================================================
*** LAYER 4 ADDED -- 2026-11-05 08:58:51 UTC -- reduce-W Metal 2.0 port ***
================================================================================
This tree now carries FOUR independent patch layers. They revert SEPARATELY.

Patch (layer 4): three HOST files, no device kernels.
  ttnn/.../experimental/quasar/reduction/generic/device/
      reduce_op_multi_core_w_program_factory.cpp
        The TILE path is now a Metal 2.0 ProgramSpec/KernelSpec program built from the
        three *_metal2 kernels that already existed in this directory and were already
        exercised by the sibling single-core-HW factory:
          dataflow/reader_unary_reduce_universal_start_id_metal2.cpp
          dataflow/writer_unary_interleaved_start_id_metal2.cpp
          compute/reduce_metal2.cpp
        The legacy ProgramDescriptor function is KEPT VERBATIM and renamed to
        ReduceMultiCoreWRmProgramFactory::create_descriptor.
      reduce_op_device_operation.hpp
        ReduceMultiCoreWProgramFactory now declares create_program_artifacts;
        new ReduceMultiCoreWRmProgramFactory holds the legacy create_descriptor;
        program_factory_t gains a fourth alternative.
      reduce_op_device_operation.cpp
        select_program_factory: MULTI_CORE_W picks the RM factory when
        operation_attributes.row_major_w_dense_path is set, else the Metal 2.0 one.
  Every hunk is tagged in-source with:  [Quasar bring-up: reduce-W metal2]

Why:   the legacy factory's reader/writer carried Reader/WriterConfigDescriptor{}, which
       on Gen2 reaches CreateDataMovementKernel and TT_FATALs
       "DataMovementKernel is not supported on Quasar" (kernel.hpp:451). dim == W ALWAYS
       selects MULTI_CORE_W (common.cpp:147-163), so EVERY device reduce over the last
       axis aborted. That is 12 segments across gpt-oss, Qwen and Llama.
Not ported: the row-major dense branch. Its three kernels (reader_unary_reduce_rm.cpp,
       writer_reduce_rm_scalar.cpp, compute/reduce_rm.cpp) have NO Metal 2.0 twin; that
       would be 3 device-kernel rewrites. It keeps the legacy factory and behaves exactly
       as before. In practice it is unreachable: it is opt-in via reduce_op.cpp's
       use_row_major_support, which defaults false and which no in-tree caller sets.

Deployed to the SHARED libraries at 2026-11-05 08:58:51-08:59:27 UTC.
  ONLY _ttnncpp.so changed:  260c6c07... -> 82a3ed74...
  libtt_metal.so (1eae9aad), _ttnn.so (e2aa0ef6) and tt-mlir's
  libTTMLIRRuntime.so (bd96ff34) are BYTE-IDENTICAL to before.
  Built and deployed holding flock /tmp/qsr_build.lock, 08:58:04 -> 08:59:28.

  *** NO OTHER RUN WAS AFFECTED -- CHECKED, NOT ASSUMED ***
  pgrep at 08:57:31 showed ONE run (SSR pid 2706195). By 08:59 there were five: the
  unaryact session launched gpt-oss/Qwen/Llama/RS50 at 08:57:54-08:58:03, INSIDE my
  lock window and ~50 s before the deploy. I suspected I had switched them mid-run
  (_ttnncpp.so was in /proc/2706195/maps at 08:57 and gone at 09:05, which looks like
  a per-segment dlopen/dlclose). THE OUTCOME REFUTES THAT:
    - the unaryact gpt-oss/Qwen/Llama arms overlapped the deploy by 4-24 minutes and
      their residual sets still contain ReduceMean 3 + ReduceMax 1 / ReduceMean 5 /
      ReduceMean 3 -- every reduce segment still aborted in them, while my own arms on
      the same plans ran all of them on device. They kept the pre-deploy library.
    - SSR cannot be affected at all: its plan has ZERO device-eligible Reduce ops.
  So the atomic rename (cp to .new + mv -f) DID protect running processes here, and the
  /proc/maps observation is not evidence of a switch. The real lesson is narrower: the
  lock does not stop other sessions LAUNCHING runs, so pgrep once before the build is
  not enough -- pgrep again immediately before the deploy step.

TO REVERT ONLY LAYER 4, one command:
   bash /proj_sw/user_dev/ctr-lelanchelian/lab_17_forge_onxx/REDUCEW_BACKUP_20261105_0851/REVERT_REDUCE_W.sh
Full write-up: quasar_models/REDUCE_W_PORT.txt
Pre-registration (written before any run): quasar_models/REDUCEW_PREREGISTERED.txt


================================================================================
*** LAYER 5 ADDED -- 2026-11-05 09:11 UTC -- concat Metal-2.0 (SOURCE ONLY) ***
*** NOT DEPLOYED. NO SHARED LIBRARY WAS WRITTEN. READ THE HAZARD NOTE BELOW. ***
================================================================================
Patch (layer 5): the generic interleaved concat gains a Metal-2.0 twin factory,
  arch-gated to Quasar. Four files added, three edited, all under
  ttnn/cpp/ttnn/operations/data_movement/concat/ plus one sources.cmake line.
  ADDED   device/concat_metal2_program_factory.{hpp,cpp}
          device/kernels/dataflow/reader_concat_interleaved_start_id_metal2.cpp
          device/kernels/dataflow/reader_concat_stick_layout_interleaved_start_id_metal2.cpp
  EDITED  device/concat_device_operation.{hpp,cpp}   (variant + arch gate)
          ttnn/cpp/ttnn/operations/data_movement/sources.cmake
  Every hunk is tagged in-source with:  [Quasar bring-up: concat interleaved metal2]
  The writers are NOT new: it binds the existing Metal-2.0 twins
  writer_unary_interleaved_start_id_metal2.cpp and
  writer_unary_stick_layout_interleaved_start_id_metal2.cpp.

Why:  ConcatProgramFactory emits Reader/WriterConfigDescriptor{}, which on Gen2 reaches
      CreateDataMovementKernel and TT_FATALs at kernel.hpp:451. Concat is 33 of the 89
      residual segments and the only op present in all five models.

*** STATE OF THE SHARED LIBRARIES: UNCHANGED. VERIFIED. ***
  build_Release/lib/_ttnncpp.so    82a3ed74  (layer 4's, untouched)
  build_Release/lib/_ttnn.so       e2aa0ef6  (untouched)
  build_Release/lib/libtt_metal.so 1eae9aad  (untouched)
  tt-mlir/build/install/lib/{_ttnncpp,_ttnn,libtt_metal,libTTMLIRRuntime}.so
                                             (all four untouched)
  The build target was `ninja ttnn/_ttnncpp.so`, whose OUTPUT IS build_Release/ttnn/
  _ttnncpp.so -- a different file from build_Release/lib/_ttnncpp.so. The lib/ copy is
  made by the separate deploy step, which I did not run. The measured binary was copied
  to /tmp/qsr_concat*/lib and reached via LD_LIBRARY_PATH, which the loader searches
  BEFORE libTTMLIRRuntime.so's DT_RUNPATH. Confirmed live in /proc/<pid>/maps: the run
  mapped /tmp/qsr_concat/lib/_ttnncpp.so, not a shared one.
  Four other sessions' runs were live throughout (SSR, gpt-oss, Qwen, Llama) and were
  NOT touched. Build held flock /tmp/qsr_build.lock 09:03:26 -> 09:07:32.

*** HAZARD FOR WHOEVER BUILDS NEXT -- PLEASE READ ***
  The SOURCE change is in the shared tree. Anyone who runs ninja from here will compile
  this layer in, and anyone who then deploys will ship it WITHOUT HAVING ASKED FOR IT.
  It is a behaviour change on Quasar: concat segments that used to abort will now run.
  If that is not what you want, run the revert below first -- it is source-only and
  takes a second. Note also that sources.cmake changed, so the next ninja will
  reconfigure cmake and rebuild all 24 ttnn_op_data_movement unity blobs once.

MEASURED (ttsim only; ZeBu never contacted). All three against the PRIVATE library:
  YOLOv8n, all 19 Concat segments flipped to device, everything else CPU:
       16/19 Concat on Quasar; pcc 1.000000; top1 33391 MATCH; 84.1 s.
       Baseline for that set is 0/19 (YOLOV8N_PATCHED_BANKED.txt:206).
  PDL, plan unmodified, --dev-only on its six device-backed Concat segments:
       6/6 Concat on Quasar; pcc 0.999998; top1 454322 MATCH; 42.3 s.
       Baseline for that set is 0/6 (PDL_PATCHED.txt:187,307).
  ResNet-50 REGRESSION, plan_par_ttsim.json, --conv-as-matmul:
       120/120 of 122; pcc 0.999823; top1 610 MATCH; 439.9 s; zero DEVFAILs.
       That is the required figure to all six digits. NO REGRESSION.
  The 3 YOLOv8n misses are NOT concat: the backtrace shows ttnn::concat ->
  ttnn::untilize_with_unpadding -> UntilizeWithUnpaddingDeviceOperation ->
  Program(ProgramDescriptor) -> CreateDataMovementKernel. The concat device op is never
  reached. See quasar_models/CONCAT_PORT.txt section 7.
  Row-major branch (which no model reaches, so it was probed directly with a pure-ttnn
  harness on ttsim): 6/6 cases BIT-EXACT against torch.cat -- four of them the
  WIDTH_CONCAT sub-stick assembly that carries the Quasar cache-coherence hazard,
  including a mixed-width 3-input case. Self-checking rerun:
      bash CONCAT_FIX_BACKUP_20261105/run_rm_concat_probe.sh
  (~90 s of simulator time, exits non-zero on any mismatch). Run it after ANY change to
  either concat reader. CONCAT_PORT.txt sections 5, 6A run (4), 8 and 10.

TO REVERT ONLY LAYER 5, one command:
   bash /proj_sw/user_dev/ctr-lelanchelian/lab_17_forge_onxx/CONCAT_FIX_BACKUP_20261105/REVERT_CONCAT.sh
   (Source-only: it restores three files and deletes four. It does NOT touch any .so,
    and it does NOT touch layers 1-4. Do NOT use TILIZE_FIX_BACKUP_20261102/RESTORE.sh,
    I2S_FIX_BACKUP_20261103/REVERT_I2S.sh or REDUCEW_BACKUP_20261105_0851/
    REVERT_REDUCE_W.sh for this.)
Full write-up: quasar_models/CONCAT_PORT.txt

================================================================================
*** SUPERSEDED -- 2026-11-12 -- EVERYTHING ABOVE IS NOW COMMITTED ***
================================================================================
All the layers described above are COMMITTED, not working-tree edits:

  tt-metal  branch quasar-resnet50-consolidated-20261112
            134b07c9791  inherited pre-2026-11-02 state (fill.h, permute, fill_pad,
                         tilize device factories, umd pointer) -- NOT part of this work
            bd017536f20  the six consolidated Quasar layers
  tt-mlir   branch quasar-resnet50-consolidated-20261112
            34bc1734b1   inherited pre-2026-10-21 runtime state
            7260e8bfe3   the eltwise redirects + unary-as-activation

ONE library set now carries ALL of them (md5 _ttnncpp.so bfab1e48,
libTTMLIRRuntime.so dfb4aa41), archived at
    quasar_models/resnet50_complete_20261112/lib/
and verified on ResNet-50 122/122, PDL, YOLOv8n and Llama-3.2-1B.

DO NOT RUN THE REVERT SCRIPTS REFERENCED ABOVE. They restore source files that are
now committed and libraries that are two generations behind; running one leaves the
tree inconsistent with HEAD. To drop a layer, revert its hunks from the commit.

Read quasar_models/RESNET50_COMPLETE.md instead of this file.
