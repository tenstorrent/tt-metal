# Open issue — MLP matmul static-CB clash with L1-resident weights

**Filed**: 2026-06-03
**Status**: Open. Blocks Option C paired-mode end-to-end `run_inference()`.
**Owner**: TBD.

## Symptom

The same static-CB clash trips on **two** paired-mode paths now:

- `Pi0_5PipelineC(layer_paired_l1=True)`: VLM prefill MLP (program ~788).
- `Pi0_5PipelineC(device_siglip=True, vision_weights_l1=True)`: SigLIP
  encoder MLP (program ~282).

Both crash inside an MLP-style matmul kernel:

```
RuntimeError: TT_THROW @ tt_metal/impl/program/program.cpp:1452: tt::exception

Statically allocated circular buffers in program 788 clash with L1
buffers on core range [0-0 - 11-7]. L1 buffer allocated at 397568 and
static circular buffer region ends at 694272
```

The numeric addresses (`397568` and `694272`) drift slightly between
runs but are stable in pattern: an L1-interleaved buffer lands inside
the `[0, 678 KB]` region that the matmul kernel's static circular
buffers want to occupy.

The crashing op's program kwargs (captured from the failure trace):

```
function_kwargs = {
  'dtype': DataType.BFLOAT16,
  'memory_config': MemoryConfig(
      memory_layout=TensorMemoryLayout::INTERLEAVED,
      buffer_type=BufferType::L1, ...),
  'program_config': MatmulMultiCoreReuseMultiCastProgramConfig(
      compute_with_storage_grid_size=12-10,
      in0_block_w=4, out_subblock_h=2, out_subblock_w=1,
      out_block_h=2, out_block_w=43,
      per_core_M=2, per_core_N=43,
      transpose_mcast=0,
      fused_activation=UnaryWithParam(op_type=UnaryOpType::GELU, params=[1]),
      fuse_batch=1, allowed_worker_cores=std::nullopt)
}
```

`out_block_w=43, per_core_N=43` and the GELU fused activation pin this
to the VLM MLP up_proj path (gate/up at width 16384 × tile 32 = 512
output tiles, sharded across cores).

## Why pipeline plumbing alone didn't fix this

The same class of issue exists for `rms_norm` and is already handled
in `models/experimental/pi0_5/tt/option_c/vlm_slice.py`:

- **vlm_slice.py:298** (replicated slice, post-stack final norm)
- **vlm_slice.py:578** (paired slice, post-stack final norm)

Both sites do the same thing:

```python
h_dram = ttnn.to_memory_config(h, ttnn.DRAM_MEMORY_CONFIG)
ttnn.deallocate(h)
h = ttnn.rms_norm(
    h_dram, weight=..., epsilon=..., memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
ttnn.deallocate(h_dram)
```

The kernel's static CB region sits at the low end of L1. As long as
its input AND output buffers are in DRAM, the allocator never tries to
place an L1 buffer in the contested range — the clash is structurally
impossible. Same trick needs to happen around the MLP matmul.

## What `l1_small_size` does (and doesn't) fix

`open_galaxy_mesh(layout, l1_small_size=N)` reserves `N` bytes per
bank for the **L1 small allocator** — a separate region from regular
L1 interleaved buffers. Static CBs do **not** live in L1_SMALL. They
live in regular L1, at the low end. Increasing `l1_small_size` does
not push the static CB region up and does not free L1 space for
interleaved buffers in that range.

The single-device pi0.5 path uses `l1_small_size=24576` (24 KB / bank)
ubiquitously — it is the standard pi0.5 value and what the probe
defaults to in paired mode. It does **not** fix this issue. Don't
chase it.

(Tried — sanity 2 of the probe sweep:
`PI0_OC_L1_PROBE_LAYER_PAIRED=1` with default 24 KB / bank → crashes
at the GELU MLP CB clash. Same crash at 1 MB / bank, just with much
less L1 cap.)

## Where to add the fix

Inside whichever VLM block code path runs the MLP matmul on the
paired prefill submesh. The candidates by file:

- `models/experimental/pi0_5/tt/option_c/vlm_slice.py` —
  `Pi0_5OptionCVLMSlicePaired.forward` (paired prefill driver). The
  per-layer block call is somewhere in the loop that walks each
  micro-submesh. The DRAM bounce wraps the matmul / fused-GELU op so
  its input and output buffers are in DRAM while it runs.
- The underlying block kernel is shared with tt-transformers and may
  be in `models/tt_dit/` or `models/tt_transformers/` — same fix
  applies wherever the matmul is built.

The same fix has to land for the expert MLP too (Option C denoise
chips at `layer_paired_l1=True` will trip the same clash for the
same reason once forward gets that far). Worth doing both at once
since they share the block pattern.

Equivalent paired-mode L1-resident path in **Option B's TP=8 expert**
already works (`tp_expert_block.py` runs with L1 weights + collective
ops). Either it doesn't trip this kernel program config (the shapes
are different — TP=8 splits the 16384 dim across 8 chips, so out_block_w
becomes ~6 not 43), or it has the bounce baked in. Worth diffing.

## How to reproduce

```bash
source python_env/bin/activate

TT_METAL_HOME=/home/tt-admin/sdawle/pi0/tt-metal \
PI0_UPSTREAM_MASKS=1 \
QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT=1 \
QWEN_NLP_CREATE_HEADS_HEAD_SPLIT=1 \
PI05_CHECKPOINT_DIR=/home/tt-admin/pi05_cache/pi05_libero_upstream \
PI05_NUM_DENOISE_STEPS=10 \
PI0_OC_L1_PROBE=1 \
PI0_OC_L1_PROBE_LAYER_PAIRED=1 \
PYTHONPATH=/home/tt-admin/sdawle/pi0/tt-metal \
python -m pytest -xvs \
  models/experimental/pi0_5/tests/test_option_c_l1_footprint_probe.py::test_oc_l1_footprint_probe_full_depth
```

Expected: `initialize()` succeeds, per-chip L1 lands at 119 / 86.5 MB
on prefill / denoise (matches plan within 3 MB), then `run_inference`
crashes at the static-CB clash.

## What's already validated

- The Option C pipeline plumbing for `layer_paired_l1` and
  `device_siglip` is correct. See commit `0f3a0b2fe94`.
- Replicated mode reproduces the baseline 2113 / 576 MB DRAM and
  9.4 MB transient L1 numbers bit-for-bit — the refactor in
  `pipeline.py::run_inference` is safe.
- The Option C smoke tests #8 / #9 / #10 already exercise paired-mode
  at the slice level with tiny single-layer payloads and pass. They
  use shorter sequences / fewer layers / different fused activations
  than the real workload, which is why they don't trip this kernel
  configuration.

## Pointers

- Crash site: `tt_metal/impl/program/program.cpp:1452`
  (`validate_circular_buffer_region`)
- Existing fix pattern: `models/experimental/pi0_5/tt/option_c/vlm_slice.py:298`
  (rms_norm DRAM bounce), `:578` (paired-mode mirror)
- Comment trail (predicts and documents this class of bug):
  `vlm_slice.py:50–63` (`_upload_replicated` docstring)
- Repro (VLM MLP path):
  `tests/test_option_c_l1_footprint_probe.py` with
  `PI0_OC_L1_PROBE_LAYER_PAIRED=1`
- Repro (SigLIP MLP path, added 2026-06-03):
  `tests/test_option_c_l1_footprint_probe.py` with
  `PI0_OC_L1_PROBE_DEVICE_SIGLIP=1 PI0_OC_L1_PROBE_VISION_WEIGHTS_L1=1`.
  Confirms the issue is general: any L1-resident matmul weight path on
  Option C trips the same kernel.
- Measurement context: `OPTION_C_L1_FOOTPRINT_PROBE.md` "Status — open"
  section.
