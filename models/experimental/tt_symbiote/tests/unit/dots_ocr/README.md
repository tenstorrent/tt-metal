# dots.ocr Bottom-Up Unit Tests

## Purpose

Bottom-up PCC test suite for the dots.ocr TTNN model. Each test exercises one
captured operation or module against a PyTorch reference using the exact
per-device shapes, dtypes, memory configs, program configs and compute-kernel
configs recorded from the production prefill + decode passes. The shapes,
dtypes, and configs come from the captured `shape_matrix/*.json` (Phase 0
output) — see `tests/test_dots_ocr.py` for the capture entry point. The suite
acts as a fast (~24 min on T3K) regression gate that can pinpoint a regression
at the op or module level without re-running the full model.

## Layout

```
tests/unit/dots_ocr/
├── README.md                       # this file
├── conftest.py                     # mesh_device_t3k_dp fixture (DP-on-T3K, fabric on)
├── e2e/                            # full-graph smoke tests
│   ├── test_text_decode_step.py    # 1 step of the text decode graph end-to-end
│   └── test_vision_tower.py        # vision tower forward end-to-end (xfail: fill_pad dtype)
├── modules/                        # per-module PCC tests (nn.Module / TTNN wrappers)
│   ├── test_lm_attention.py        # text attention block (prefill + decode)
│   ├── test_lm_decoder_layer.py    # full text decoder block
│   ├── test_lm_embedding.py
│   ├── test_lm_head.py             # final projection + argmax
│   ├── test_lm_mlp.py
│   ├── test_lm_rmsnorm.py
│   ├── test_vision_attention.py    # vision attention block (small + large S)
│   ├── test_vision_block.py
│   ├── test_vision_mlp.py
│   ├── test_vision_patch_embed.py
│   ├── test_vision_patch_merger.py # SKIPPED (see Known issues)
│   ├── test_vision_rms_norm.py
│   └── test_vision_rope.py
├── ops/                            # per-op PCC tests (one ttnn.* op each)
│   ├── test_all_gather.py
│   ├── test_argmax.py
│   ├── test_concat_slice_pad.py
│   ├── test_elementwise.py         # add / mul / where / typecast
│   ├── test_embedding.py
│   ├── test_layer_norm.py
│   ├── test_linear.py              # ttnn.linear / ttnn.matmul
│   ├── test_nlp_concat_heads.py
│   ├── test_nlp_create_qkv_heads.py
│   ├── test_reduce_scatter.py
│   ├── test_rms_norm.py
│   ├── test_rotary_embedding.py
│   └── test_sdpa.py                # 1 known fail (BFP4 V vision large-S)
├── reference/                      # CPU torch reference implementations
│   └── op_reference.py             # rms_norm, apply_rotary_emb, nlp_create_qkv_heads, ...
├── scripts/
│   ├── regression_check.sh         # convenience wrapper: tt-smi -r + full suite
│   └── regression_demo.py          # standalone regressions (math_fidelity, wrong ref)
├── shape_matrix/                   # captured per-op shape records (Phase 0 output)
│   ├── {text,vision}_ops.json
│   ├── {text,vision}_ops_dedup.json
│   ├── {text,vision}_modules.json
│   └── {text,vision}_modules_dedup.json
└── util/                           # rebuild ttnn kwargs from captured records, PCC helpers
    ├── capture.py
    ├── matrix_loader.py            # load_op_matrix, make_row_id, make_row_tags
    ├── mesh_gather.py              # replicated / DP-2d gather helpers
    ├── module_helpers.py
    ├── pcc.py                      # assert_op_pcc, op_pcc_threshold
    └── ttnn_kwargs.py              # build_compute_kernel_config / build_memory_config / build_program_config
```

## How to run

All commands assume cwd = `/home/ttuser/salnahari/tt-metal` and the in-tree
virtualenv `python_env/` is activated.

```bash
cd /home/ttuser/salnahari/tt-metal
source python_env/bin/activate
```

### Full suite

```bash
# Always reset the chip first.
unset TT_VISIBLE_DEVICES && python_env/bin/tt-smi -r

MESH_DEVICE=T3K DOTS_OCR_PARALLELISM=DP TT_SYMBIOTE_RUN_MODE=NORMAL \
pytest models/experimental/tt_symbiote/tests/unit/dots_ocr/ops/ \
       models/experimental/tt_symbiote/tests/unit/dots_ocr/modules/ \
       models/experimental/tt_symbiote/tests/unit/dots_ocr/e2e/ \
       --timeout=0 --tb=short --durations=20 -q
```

Or use the wrapper:

```bash
bash models/experimental/tt_symbiote/tests/unit/dots_ocr/scripts/regression_check.sh
```

### One group

```bash
# Ops only (~14.6 min on T3K):
MESH_DEVICE=T3K DOTS_OCR_PARALLELISM=DP TT_SYMBIOTE_RUN_MODE=NORMAL \
pytest models/experimental/tt_symbiote/tests/unit/dots_ocr/ops/ --tb=short -q

# Modules only (~5.7 min):
MESH_DEVICE=T3K DOTS_OCR_PARALLELISM=DP TT_SYMBIOTE_RUN_MODE=NORMAL \
pytest models/experimental/tt_symbiote/tests/unit/dots_ocr/modules/ --tb=short -q

# e2e only (~3.8 min, ~2 tests):
MESH_DEVICE=T3K DOTS_OCR_PARALLELISM=DP TT_SYMBIOTE_RUN_MODE=NORMAL \
pytest models/experimental/tt_symbiote/tests/unit/dots_ocr/e2e/ --tb=short -q
```

### Single file

```bash
MESH_DEVICE=T3K DOTS_OCR_PARALLELISM=DP TT_SYMBIOTE_RUN_MODE=NORMAL \
pytest models/experimental/tt_symbiote/tests/unit/dots_ocr/ops/test_rms_norm.py --tb=short -v
```

### Single row by id substring

Test ids are constructed by `util/matrix_loader.make_row_id(row)` and look like
`linear_text_b1_M14_K1536_N2048_bf16xbfp8_HiFi2_cid32`. Use `-k` to filter:

```bash
# All matmul rows with call_id ending in 32:
pytest models/experimental/tt_symbiote/tests/unit/dots_ocr/ops/test_linear.py -k cid32 --tb=short -v

# All vision rms_norm rows:
pytest models/experimental/tt_symbiote/tests/unit/dots_ocr/ops/test_rms_norm.py -k vision --tb=short -v
```

## How to regenerate the shape matrix

The shape matrix is the source of truth for every parametrize. Regenerate it
by running the model end-to-end with `DOTS_OCR_CAPTURE_SHAPES=1`:

```bash
unset TT_VISIBLE_DEVICES && python_env/bin/tt-smi -r

# Text path (text-only graph captured into text_ops.json / text_modules.json):
MESH_DEVICE=T3K DOTS_OCR_PARALLELISM=DP \
DOTS_OCR_CAPTURE_SHAPES=1 \
DOTS_OCR_CAPTURE_PHASE=text \
TT_SYMBIOTE_RUN_MODE=NORMAL \
pytest models/experimental/tt_symbiote/tests/test_dots_ocr.py --tb=short -q

# Vision path:
unset TT_VISIBLE_DEVICES && python_env/bin/tt-smi -r
MESH_DEVICE=T3K DOTS_OCR_PARALLELISM=DP \
DOTS_OCR_CAPTURE_SHAPES=1 \
DOTS_OCR_CAPTURE_PHASE=vision \
TT_SYMBIOTE_RUN_MODE=NORMAL \
pytest models/experimental/tt_symbiote/tests/test_dots_ocr.py --tb=short -q
```

The capture writes both the raw and dedup-by-shape records under
`shape_matrix/`. The dedup files are what the tests parametrize over;
re-run the test suite afterward to pick up the new rows. The capture is
driven by `util/capture.py`.

## How to interpret a failure

When a test fails the assertion looks like:

```
PCC mismatch in op_name='ttnn.linear' row_id='linear_vision_TTNNDotsVisionMLP_in1x1x12288x4224_bfp8_wbfp8_LoFi_cid32'
  threshold       = 0.985
  computed PCC    = 0.953...
  reference shape = (1, 1, 12288, 1536) dtype=torch.float32
  actual    shape = (1, 1, 12288, 1536) dtype=torch.float32
```

1. **Match `row_id` to the captured row**. The trailing `cid<N>` is the
   `call_id` field — grep the right shape_matrix file:

   ```bash
   python -c "import json; d=json.load(open('models/experimental/tt_symbiote/tests/unit/dots_ocr/shape_matrix/vision_ops_dedup.json')); \
   print(json.dumps([r for r in d if r['call_id']==32][0], indent=2))"
   ```

2. **Read the threshold margin**. `op_pcc_threshold(op, in_dtypes, fidelity)` in
   `util/pcc.py` picks the dtype-minimum table entry; e.g. `bfp4_b` is 0.965,
   `bfp8_b` is 0.985, `bfloat16` is 0.997. If the measured PCC is *just*
   below threshold the regression is likely a dtype/fidelity drift rather
   than a logic bug — check `compute_kernel_config.math_fidelity` against
   the captured value.

3. **Compare the rebuilt ttnn kwargs to the captured ones**. Run with `-s`
   and uncomment any of the `print(...)` hooks, or set a breakpoint in the
   test before the `ttnn.<op>(...)` call and inspect:
   - `memory_config` (reconstructed by `util/ttnn_kwargs.build_memory_config`)
   - `program_config` (reconstructed by `build_program_config`)
   - `compute_kernel_config` (`build_compute_kernel_config`)

   The capture record's `repr` field shows the exact production object;
   if a field is missing or coerced wrong, the reconstruction diverges and
   the PCC drops.

4. **Cross-check against the dependency layer**. An ops/test_X.py failure
   usually shows up first in a modules/test_Y.py test that consumes that
   op. If a module passes but its op fails on the same shape, the harness
   may be skipping a memory_config that the module-level wrapper restores
   internally — extend `build_memory_config` (see Phase 1 note #4 in the
   plan).

## Known issues

1. **BFP4 V-cache SDPA at large S** — `test_sdpa.py` row
   `transformer_scaled_dot_product_attention_vision_TTNNDotsVisionAttention_in1x12x12288x128_bfp8_wbfp8xbfp4_LoFi_cid22`
   FAILs with measured PCC ≈ 0.52 against threshold 0.97. The smaller-S
   vision attention SDPA rows (S ≈ 2814) and all text SDPA rows pass.
   The corresponding module-level `test_vision_attention.py` still passes
   because the module test path doesn't exercise this exact dtype +
   fidelity combination at the captured S=12288.

2. **Attention block PCC plateau ~0.70** — when intermediate Q/K/V are
   forced to match production via the captured rotary-embedding tables,
   the attention-block module test plateaus at PCC ≈ 0.70 due to a
   rotary half-half convention mismatch between the captured tables and
   the upstream `apply_rotary_emb`. Tracked separately; not a regression.

3. **`test_vision_patch_merger.py` SKIPPED** — `TTNNDotsPatchMerger`
   col-shards its `w2` weight across the TP mesh axis and returns a tensor
   with the same TP-sharded layout. The Phase 3 module harness uses
   replicated input + replicate gather, which is incompatible with that
   layout. Needs a `ShardTensor2dMesh`-aware input wrapper (Phase 4
   follow-up). Coverage today comes from `e2e/test_vision_tower.py`.

4. **`e2e/test_vision_tower.py` XFAIL** — the vision tower end-to-end
   fails with
   `TT_FATAL ... fill_pad_device_operation.cpp:19: detail::data_type_to_size.contains(input_tensor.dtype())`
   when trace is disabled. The op refuses the dtype handed to it in the
   non-traced execution path; under trace it's tolerated. Tracked as a
   known dtype-support gap in `fill_pad`.

5. **`test_nlp_concat_heads.py` two XFAILs** — the two sharded
   `nlp_concat_heads_decode` rows (text cid854, vision cid1709). The
   harness's `util/ttnn_kwargs.build_memory_config` does NOT reconstruct
   `shard_spec` from the captured record (Phase 1 note #4) so the inputs
   are placed in interleaved DRAM and the op rejects them. Fix is to
   extend `build_memory_config` to parse the captured `shard_spec`
   substring — tracked as a follow-up.

6. **`test_elementwise.py` one SKIP** — row
   `add_vision_anon_in1_i32_cid4885` is a scalar-add (one TTNN tensor
   operand + a constant). The constant is not in the capture and we
   cannot reconstruct an exact PCC, so the row self-skips with an
   explanatory message.

## Adding new tests

The standard pattern (see `ops/test_rms_norm.py`):

```python
from ...util.matrix_loader   import load_op_matrix, make_row_id, make_row_tags
from ...util.ttnn_kwargs     import parse_ttnn_dtype, build_compute_kernel_config, build_memory_config, build_program_config
from ...util.mesh_gather     import gather_to_torch
from ...util.pcc             import assert_op_pcc, op_pcc_threshold
from ...reference.op_reference import <ref_fn>

_ROWS = load_op_matrix("ttnn.<op>")

@pytest.mark.parametrize("row", _ROWS, ids=[make_row_id(r) for r in _ROWS])
def test_<op>(row, mesh_device_t3k_dp):
    tags   = make_row_tags(row)
    row_id = make_row_id(row)
    device = mesh_device_t3k_dp

    # 1. Build a torch reference at the captured per-device shape.
    x_torch = torch.randn(*row["inputs"][0]["shape"], dtype=torch.bfloat16)
    ref     = <ref_fn>(x_torch, ...)

    # 2. Build the ttnn input(s).
    x_dtype = parse_ttnn_dtype(row["inputs"][0]["dtype"])
    x_tt    = ttnn.from_torch(x_torch, dtype=x_dtype, layout=ttnn.TILE_LAYOUT,
                              device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                              mesh_mapper=ttnn.ReplicateTensorToMesh(device))

    # 3. Run the captured op with rebuilt kwargs.
    ckc = build_compute_kernel_config(*<from row["kwargs"]["compute_kernel_config"]>)
    out_tt = ttnn.<op>(x_tt, compute_kernel_config=ckc, ...)

    # 4. Gather + assert.
    out_torch = gather_to_torch(out_tt, device, strategy="replicated")
    threshold = op_pcc_threshold(row["op"], [x_dtype], tags["math_fidelity"])
    pcc       = assert_op_pcc(ref.to(torch.float32), out_torch.to(torch.float32),
                              threshold=threshold, op_name=row["op"], row_id=row_id)

    ttnn.deallocate(out_tt); ttnn.deallocate(x_tt)
```

Add the file under `ops/` or `modules/`. The collection logic in `conftest.py`
auto-resolves the `mesh_device_t3k_dp` fixture.

## Regression baseline

Last measured baseline (T3K, MESH_DEVICE=T3K, DOTS_OCR_PARALLELISM=DP):

```
1 failed, 209 passed, 2 skipped, 4 xfailed in 1443.13s (24:03)
```

Per-directory wall clock:
- `ops/` : ~14.6 min  (plan target: < 30 min — MET)
- `modules/` : ~5.7 min (plan target: combined < 60 min — MET)
- `e2e/` : ~3.8 min

Slowest 5 test files (sum of in-file durations from the slowest-20 report):
1. `e2e/test_vision_tower.py`        158 s (1 test, xfailed)
2. `ops/test_embedding.py`           108 s (4 tests)
3. `ops/test_all_gather.py`          100 s (3 tests in top-20)
4. `ops/test_linear.py`               77 s (4 tests in top-20)
5. `e2e/test_text_decode_step.py`     58 s (1 test)

`scripts/regression_demo.py` quantifies the suite's PCC sensitivity:
- **Demo 1a (rms_norm cid=8)**: HiFi4 PCC = 0.99998, LoFi PCC = 0.99977
  (delta 2.15e-4; suite threshold 0.999). RMSNorm at this shape is largely
  fidelity-insensitive; the LoFi run still passes.
- **Demo 1b (matmul bf16 × bfp4_b, M=256 K=8192 N=2048)**: HiFi4 PCC = 0.9932,
  LoFi PCC = 0.9914 (delta 1.7e-3; bfp4-derived threshold 0.965).
- **Demo 2 (ttnn.add vs torch.sub wrong reference)**: `assert_op_pcc` raises
  with computed PCC ≈ -0.02 against threshold 0.999 — proves the failure
  path is real.

Run with:

```bash
MESH_DEVICE=T3K \
python models/experimental/tt_symbiote/tests/unit/dots_ocr/scripts/regression_demo.py
```
