# Reference diagnostic recovery and completed comparison

The original CPU reference failed before forwarding the model because
Transformers returned sets in `loading_info`; the JSON report writer rejected
them. Its exception handler then hit the same serialization error, leaving a
stale `loading_reference` report. The controller's exit code and raw traceback,
not that stale report, establish failure. The report writer now serializes sets
deterministically and rejects unsupported objects. Three regression tests pass.
The isolated local fixture first used the wrong `pytest.raises` signature; the
fixture was corrected without changing the production test expectations.

The original image import succeeded, but its checker used the OCI config digest
as a Docker image ID. This host exposes the imported manifest digest instead:
`sha256:0b11f045bf089088a62b6e3c1aeb9b64cc72b74a632935f25203609b1e023579`.
The unchanged config lookup returned `No such image`. Source labels and the
manifest descriptor match the built image, but runtime-import/startup checks
remain outstanding. The failed original layer, chunked-state and startup
followers stopped without opening a model because their dependencies failed.
All failed receipts are retained; none was rewritten as success.

A fresh reference-only CPU job and layer follower were launched after verifying
the old jobs were terminal. Nineteen native host tests passed. The CPU job
completed its eight BF16 reference positions at 09:02:34 UTC; all checkpoint
loading-key diagnostics were empty. Host-rounded BFP4 head weights produced
8.148% relative logit RMS error versus 0.532% for BFP8, with 8/8 top-1 matches
for both. This measures head-weight sensitivity on one short public prompt,
not GPQA or device arithmetic.

**Hardware-access clarification:** the frozen CPU report's `hardware_opened:
false` records that the probe does not explicitly open a TT mesh or forward the
model on an accelerator. Its host `ttnn.from_torch` conversion nevertheless
initialized the UMD driver and discovered/started devices, as the retained log
shows. It must not be described as having no hardware access. The job held the
shared device lock throughout; the Torch model forward ran on CPU. No existing
artifact was edited to hide this distinction.

The hardware follower completed one passing diagnostic test over eight
teacher-forced steps and all 64 layer inputs, with clean device shutdown. The
unchanged BFP4/LoFi decoder plus BFP8/HiFi2 head produced full-logit relative RMS
errors from **20.1% to 70.3%**, while the actual device head on matched HF hidden
states was **0.60-0.83%**. Both paths matched HF top-1 on all eight positions.
For the initial prefill, cumulative hidden-state error rose from 2.36% after
layer 0 to 25.6% at layer 32 and 79.1% after final norm. These results motivate
decoder controls but do not alone identify a kernel bug or predict GPQA.

The exact reference tensor remains on the host, SHA256
`6a27844742b8ca9caaad4e7236101a9e4b493098a6ace5bb99b2e46e2c7c1561`.
Compressed source copies, logs, JUnit and the full per-rank layer comparison
preserve exact original bytes. The hardware wrapper reset the prior dirty
device marker before execution; a marker alone was not labeled a new hang.

The fresh CPU job was bounded to 55 minutes (40-minute reference subprocess),
160 GiB and eight CPU quota. The layer follower had a three-hour outer limit
and used `/tmp/tt-device.lock`. Both completed. The next numerical experiments
are the [decoder controls](../hf-decoder-controls-v2/README.md). The original
chunked-state and image-startup jobs are still failed, awaiting separate
recovery after the current accuracy priority.
