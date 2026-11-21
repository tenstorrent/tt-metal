# gpt-oss on the ttsim Quasar simulator: results

This report covers four runs of gpt-oss on Quasar with ttsim, the public functional simulator of Tenstorrent chips: the gpt-oss op test suite and a 2-layer gpt-oss demo, each on the full 8x4 grid and on a 2x1 grid. All four pass. The demo checks no values, so the report also covers a per-step PCC (Pearson correlation coefficient) comparison of the model against the Hugging Face reference (section "Per-step PCC against the Hugging Face reference"). The earlier reports used the craq-sim simulator: [gpt_oss_ops_analysis.md](gpt_oss_ops_analysis.md) for the full grid and [gpt_oss_small_grid_analysis.md](gpt_oss_small_grid_analysis.md) for the 2x1 grid. Problem letters (I, L) refer to [gpt_oss_ops_analysis.md](gpt_oss_ops_analysis.md).

## Results

| Run | Result |
|---|---|
| Op suite, 8x4 grid | 124 passed, 7 skipped, 0 failed |
| Op suite, 2x1 grid | 120 passed, 11 skipped, 0 failed |
| gpt-oss demo, 2 layers, 8x4 grid | passed in 2 h 6 min |
| gpt-oss demo, 2 layers, 2x1 grid | passed in 2 h 40 min |

The op suite has 131 cases: the 119 cases of the 38 generated test files in `models/experimental/ops/quasar/tests/gpt_oss_ops/`, and the 12 cases of the 5 `_small_grid.py` companion tests, which run some captured calls in the form that the model uses on a small grid. Each case ran in its own pytest process. The suite took 60 min on 8x4 and 107 min on 2x1.

Both demos ran `text_demo.py::test_gpt_oss_demo[quasar-1x1-prefill_128]` with the first 2 decoder layers (`QSR_N_LAYERS=2`), so the generated text has no meaning. They decoded 3 tokens instead of 200: a pytest plugin used for the runs, not in the repository, sets `max_generated_tokens`. Every op of the model ran on the device. The demo picks each next token with argmax on the host, because on-device sampling is not available.

| Demo step | 8x4 grid | 2x1 grid |
|---|---|---|
| Warmup prefill (128 tokens) | 52 min | 69 min |
| Prefill (128 tokens) | 51 min | 66 min |
| Decode, first token (includes kernel compiles) | 494 s | 508 s |
| Decode, next 2 tokens | 413 s per token | 429 s per token |

## Setup

- Branch `fplavec/gpt_oss_quasar_bringup_2` at `20a35a0ea79`, on `main` `840db3161d0`, with PR #63905 (head `e5f7dfff689`) applied on top. #63905 widens the implicit-sync descriptor counts of the Quasar DFBs (dataflow buffers) to 16 bits; without it, untilize fails (problem I).
- ttsim v1.16.0, the release library `libttsim_qsr.so` from [tenstorrent/ttsim](https://github.com/tenstorrent/ttsim/releases/tag/v1.16.0), renamed to `libttsim.so`. Its SoC descriptor is a copy of [quasar_32_arch_ttsim.yaml](../../../../../tt_metal/soc_descriptors/quasar_32_arch_ttsim.yaml) with 3 GiB per DRAM channel instead of 1 GiB, because the gpt-oss weights do not fit in 1 GiB per channel.
- Slow dispatch (`TT_METAL_SLOW_DISPATCH_MODE=1`). bfloat16 instead of the block float formats in the op suite (`TTNN_GRAPH_OPS_BF8_AS_BF16=1`), because Quasar has no block float formats.
- The 2x1 grid comes from `TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="3,2"`; [gpt_oss_small_grid_analysis.md](gpt_oss_small_grid_analysis.md) explains how.
- Timeouts per case: 20 min on 8x4, 50 min on 2x1. The two `lm_head` matmul cases, `test_matmul[03_...]` and `[04_...]`, took 14 to 20 min on 2x1, against 5 to 6 min on 8x4.
- The run folder is `/localdev/fplavec/temp/gpt_oss_ops_runs/20261119_run14_ttsim/`, with the run scripts, one log per run, and `summary.tsv` and one log per case for each suite.

The machine runs tests in a container with a 63 GiB memory limit. One demo used about 42 GiB while it loaded the weights (the kernel log of an out-of-memory stop on 2026-11-18), so the two demos ran one after the other; two demos that load at the same time are stopped by the limit.

## How to reproduce

Run every command from the root of a tt-metal checkout. Steps 1 to 3 are done once.

1. Check out the branch, apply PR #63905, and build:

   ```bash
   git fetch origin fplavec/gpt_oss_quasar_bringup_2 pull/63905/head:pr63905
   git checkout 20a35a0ea79
   git diff 104f3582d60 e5f7dfff689 | git apply   # #63905 at head e5f7dfff689, against its base
   ./build_metal.sh -e --enable-fake-kernels-target
   ./create_venv.sh   # only if python_env/ does not exist
   ```

2. Get ttsim and its SoC descriptor. tt-metal reads the SoC descriptor from `soc_descriptor.yaml` in the folder that holds `libttsim.so` (`TT_METAL_SIMULATOR_HOME`):

   ```bash
   SIM=/localdev/$USER/ttsim_v1.16.0_qsr
   mkdir -p $SIM
   gh release download v1.16.0 --repo tenstorrent/ttsim --pattern libttsim_qsr.so --dir $SIM
   mv $SIM/libttsim_qsr.so $SIM/libttsim.so
   cp tt_metal/soc_descriptors/quasar_32_arch_ttsim.yaml $SIM/soc_descriptor.yaml
   sed -i 's/^  1073741824$/  3221225472/' $SIM/soc_descriptor.yaml   # 3 GiB per DRAM channel
   ```

3. Write the pytest plugin that shortens the demo to `/localdev/$USER/demo_plugins/short_demo.py`. The demo marks itself with a 2 h pytest timeout, which is shorter than a demo run on ttsim, and it decodes 200 tokens:

   ```python
   """Shorten the gpt-oss demo for simulator runs: DEMO_MAX_TOKENS sets max_generated_tokens,
   and DEMO_TIMEOUT_S replaces the demo's own pytest timeout."""

   import os

   import pytest


   def pytest_collection_modifyitems(items):
       timeout_s = int(os.environ.get("DEMO_TIMEOUT_S", str(48 * 3600)))
       for item in items:
           item.add_marker(pytest.mark.timeout(timeout_s), append=False)


   @pytest.hookimpl(tryfirst=True)
   def pytest_pyfunc_call(pyfuncitem):
       args = pyfuncitem.funcargs
       if "DEMO_MAX_TOKENS" in os.environ and "max_generated_tokens" in args:
           args["max_generated_tokens"] = int(os.environ["DEMO_MAX_TOKENS"])
   ```

4. Set the environment in each new shell. For the 2x1 grid, also set the override:

   ```bash
   source python_env/bin/activate
   export SIM=/localdev/$USER/ttsim_v1.16.0_qsr
   export TT_METAL_HOME=$PWD TT_METAL_SIMULATOR=$SIM/libttsim.so TT_METAL_SIMULATOR_HOME=$SIM \
       ARCH_NAME=quasar CHIP_ARCH=quasar TT_METAL_SLOW_DISPATCH_MODE=1 \
       LD_LIBRARY_PATH=$PWD/build/lib:${LD_LIBRARY_PATH:-} PYTHONPATH=$PWD/ttnn:$PWD
   export TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="3,2"   # 2x1 grid only
   ```

5. Run the op suite, one pytest process per case. A simulator error ends the whole pytest process, so one process per case keeps one failure from stopping the other cases. Use a timeout of 20m on 8x4 and 50m on 2x1:

   ```bash
   export TTNN_GRAPH_OPS_BF8_AS_BF16=1
   T=20m   # 50m on the 2x1 grid
   for id in $(pytest -o addopts="--import-mode=importlib" --collect-only -q \
                   models/experimental/ops/quasar/tests/gpt_oss_ops | grep "::"); do
       timeout --signal=INT --kill-after=20s $T pytest -s --timeout=0 "$id" > "suite_$(echo "$id" | tr -c 'A-Za-z0-9_.-' '_' | cut -c1-150).log" 2>&1
       echo "$? $id"
   done | tee suite_results.txt
   ```

   Exit status 0 means passed or skipped (the log says which), 124 or 137 means the timeout fired (137 when the case did not stop after SIGINT and got SIGKILL 20 s later), and any other value means the case failed.

6. Run the demo. `HF_MODEL` is a local copy of the `openai/gpt-oss-20b` checkpoint, and `TT_CACHE_PATH` is any writable folder for the converted weights:

   ```bash
   export PYTHONPATH=/localdev/$USER/demo_plugins:$PYTHONPATH
   export QSR_N_LAYERS=2 DEMO_MAX_TOKENS=3 DEMO_TIMEOUT_S=172800 \
       HF_MODEL=/path/to/gpt-oss-20b TT_CACHE_PATH=/localdev/$USER/gpt_oss_weight_cache
   pytest -s -p short_demo "models/experimental/ops/quasar/gpt_oss/demo/text_demo.py::test_gpt_oss_demo[quasar-1x1-prefill_128]" > demo.log 2>&1
   ```

   The demo always runs a warmup prefill before the measured prefill. The run passed if pytest exits with 0 and `demo.log` contains "1 passed" and "GPT-OSS demo completed successfully!". Run one demo at a time: one demo used about 42 GiB while it loaded the weights, and the container of the recorded runs had a 63 GiB memory limit.


## Per-step PCC against the Hugging Face reference

The demo passes when every op runs without an error; it compares no values. To check the values, the model can log the PCC of each step against the Hugging Face reference, as resnet does with `RESNET_PCC_LOG` and llama with `LLAMA_PCC_LOG`:

- With `GPT_OSS_PCC_LOG=1`, [pcc_log.py](utils/pcc_log.py#L137-L177) reads the output of each step back to the host and logs a `[PCCLOG]` line with the shape and summary values (mean, standard deviation, minimum, maximum, mean absolute value, NaN and Inf counts, and the first 8 values). Without the variable it does nothing.
- The steps are 6 per decoder layer ([layer.py:113-149](tt/layer.py#L113-L149)): `input_layernorm`, `self_attn`, `attn_residual` (the sum after attention), `post_attention_layernorm`, `mlp` and `output`; then the final `norm` and `lm_head` (the logits, [model.py:424-428](tt/model.py#L424-L428)).
- `test_model` in [test_modules.py](tests/unit/test_modules.py#L1115-L1131) records the reference output of each step with forward hooks on the Hugging Face model (`capture_golden_intermediates`, [pcc_log.py:55-88](utils/pcc_log.py#L55-L88)). The device run then also logs a `[GOLDENPCC]` line with the PCC of each step, marked "DIVERGES" below 0.98.
- `GPT_OSS_PCC_DUMP=<dir>` also saves the output of each step to a file.

The `prefill_b1_s128` case of `test_model` runs a 128-token prefill with 1 or 2 layers (`num_layers`), and with random weights or with the first layers of the checkpoint in `HF_MODEL` (`weights=random|trained`, [test_modules.py:1213-1219](tests/unit/test_modules.py#L1213-L1219)). The `decode_b128_s1` case of `test_model` is skipped on a 1x1 mesh. The test's own check is the PCC of the logits: at least 0.95 for 1 layer and 0.85 for 2 layers.

To run it, use the environment of step 4 of "How to reproduce". For the 2-layer cases on ttsim, use `4294967296` (4 GiB per channel) in the `sed` command of step 2. For example:

```bash
GPT_OSS_PCC_LOG=1 HF_MODEL=/path/to/gpt-oss-20b pytest -s models/experimental/ops/quasar/gpt_oss/tests/unit/test_modules.py \
    -k "test_model and prefill_b1_s128 and 2_layers and trained"
```

On ttsim, the 2-layer cases need 4 GiB per DRAM channel in the SoC descriptor. With 3 GiB, the 1.5 GB `lm_head` weight does not fit: it is split across 2 banks of 755 MB each, and each bank has 935 MB free, but in blocks of at most 453 MB. The ttsim runs below used a copy of the ttsim setup with 4 GiB per channel, except the 1-layer run with random weights and the `test_decoder` runs on 8x4, which used 3 GiB. They ran on 2026-11-20, on the same tree as the other runs plus `utils/pcc_log.py` and the calls to it in `tt/layer.py`, `tt/model.py` and `tests/unit/test_modules.py`. The trained-weight cases and `test_decoder` ran on both the 8x4 and the 2x1 grid; the random-weight cases ran on 8x4 only. The logs are in `/localdev/fplavec/temp/gpt_oss_ops_runs/20261120_pcc_ttsim/`.

### Results with trained weights

| Step | 8x4, 1 layer | 8x4, 2 layers (layer 0 / layer 1) | 2x1, 1 layer | 2x1, 2 layers (layer 0 / layer 1) |
|---|---|---|---|---|
| `input_layernorm` | 0.99998 | 0.99998 / 0.9990 | 0.99998 | 0.99998 / 0.9991 |
| `self_attn` | 0.9996 | 0.9996 / 0.9945 | 0.9996 | 0.9996 / 0.9950 |
| `attn_residual` | 0.99995 | 0.99995 / 0.9990 | 0.99995 | 0.99995 / 0.9991 |
| `post_attention_layernorm` | 0.9997 | 0.9997 / 0.9972 | 0.9997 | 0.9997 / 0.9976 |
| `mlp` | 0.9984 | 0.9984 / 0.9943 | 0.9986 | 0.9986 / 0.9941 |
| `output` | 0.9991 | 0.9991 / 0.9987 | 0.9992 | 0.9992 / 0.9988 |
| `norm` | 0.9991 | 0.9987 | 0.9993 | 0.9988 |
| `lm_head` (logits) | 0.9990 | 0.9986 | 0.9992 | 0.9987 |

All four cases passed: on 8x4 in 43 min (1 layer) and 67 min (2 layers), on 2x1 in 65 min (1 layer) and 102 min (2 layers). No step is below 0.994. The two grids give nearly the same PCC at every step. These are prefill cases, so on 2x1 they cover the experts on a (2, 1) grid and the prefill SDPA (scaled dot product attention) on a 2x1 grid. The decode paths that the model takes only on a small grid (the head merge for a device with fewer cores than heads, the paged SDPA decode on 2x1, and the DRAM output of the decode down projection) are covered by the decode cases of `test_decoder` below, with random weights.

### Random weights, and a comparison with Wormhole

The same test ran with random weights on ttsim, and with both kinds of weights on a Wormhole n150 board, with the same seeds:

| Step, 2 layers | ttsim, random | Wormhole, random | ttsim, trained | Wormhole, trained |
|---|---|---|---|---|
| `layer0.mlp` | 0.9851 | 0.9551 | 0.9984 | 0.9753 |
| `layer1.self_attn` | 0.9813 | 0.9221 | 0.9945 | 0.9207 |
| `layer1.mlp` | 0.9359 | 0.7947 | 0.9943 | 0.9021 |
| `lm_head` (logits) | 0.9728 | 0.9068 | 0.9986 | 0.9813 |

- All these cases passed their logit check.
- With random weights, the `mlp` step loses the most PCC in each layer, on both machines. With trained weights, the `self_attn` step of layer 1 loses more than its `mlp` step, on both machines. In all four runs, the `mlp` output of layer 1 has the lowest PCC of all steps.
- With random weights, the loss on Quasar is large enough to mark the `mlp` step of layer 1 (0.936) and every step after it "DIVERGES"; with trained weights, no step is marked.
- Quasar is closer to the reference than Wormhole at every step. My guess is that Wormhole loses more because it stores the weights as `bfloat8_b` and `bfloat4_b`, while the Quasar tests turn them into bfloat16; this was not tested.

### Module tests

`test_decoder` in [test_modules.py](tests/unit/test_modules.py#L693-L700) compares the attention, RMSNorm, experts, the full MLP, the router (decode only) and the whole decoder layer 0 with the Hugging Face reference, with random weights. All 4 cases pass on ttsim on both grids: paged and unpaged, each with decode and with a 128-token prefill. On 2x1 they took 9 to 10 min for decode and 38 min (paged) and 96 min (unpaged) for prefill.

## Fixes that the runs depend on

All of these are on `main` `840db3161d0` except #63905:

| Change | What it fixes |
|---|---|
| #66243 (issue #66114) | Makes `sparse_matmul` turn off DFB implicit sync for its in0 and in1 kernels on Quasar, as the other matmul factories do. Without it, the firmware adds tiles to a full tile counter at kernel exit, and the simulator stops on grids such as (2, 1) and (6, 3) (problem L). |
| #63365 | Makes the Quasar math setup track the ALU formats that it has programmed. See the next section. |
| #66295 | Adds the missing `import functools` in `models/common/moe/compute_helpers.py`. Without it, the demo stops before it runs any op. |
| #63905 (open) | The 16-bit implicit-sync descriptor counts (see Setup). |

## The softmax failure that only ttsim found

Before #63365, `test_softmax[01_128x4_bf16_int-l1]` on the 2x1 grid stopped ttsim with:

```
ERROR: UnimplementedFunctionality: tensix_movd2b: read_fmt=5 write_fmt=4
```

The same case passed on 8x4 on ttsim, and on both grids on craq-sim.

- The case has 4 tile rows. On 2x1 each core handles 2 tile rows, and on 8x4 each core handles 1.
- At the end of the first tile row, the softmax multiply sets the format of the SrcB source register to Tf32 (code 4 in the Quasar format list), for its Float32 input. The reduce-MAX of the second tile row ([softmax.cpp:53-62](../../../../../ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/compute/softmax.cpp#L53-L62)) should set SrcB back to Float16_b (code 5).
- Before #63365, the math setup only recorded that the default formats had been set, not which formats, so the reduce-MAX setup skipped the change. Its MOVD2B instruction, which moves data from the destination register to SrcB ([llk_math_reduce.h:186-190](../../../../../tt_metal/tt-llk/tt_llk_quasar/llk_lib/llk_math_reduce.h#L186-L190)), then ran with a Float16_b destination and a Tf32 SrcB. ttsim stops on that pair (`UnimplementedFunctionality`); craq-sim runs it without an error.
- A variant with 1056 rows, which gives one core 2 tile rows on 8x4, failed in the same way on 8x4. A variant with 64 rows, 1 tile row per core, passed on 2x1. So the failure depends on the tile rows per core, not on the grid.

With #63365, both softmax cases pass on both grids (`softmax_1119_full.log`, `softmax_1119_2x1.log`), and all three row-count variants pass on 8x4 (`movd2b/movd2b_1119_full.log`), all in `/localdev/fplavec/temp/gpt_oss_ops_runs/20261118_run13_ttsim/`. The variants were not run on 2x1 with #63365. The tracking problem is also filed as #58588, #58592 and #59468.

## Skipped cases

The suite harness skips a case that cannot run on the device as captured. Two rules were already in place for shard grids. Two rules were added for this work in `graph_case.py`:

- A captured program config whose compute grid is larger than the device grid ([graph_case.py:326-346](../tests/gpt_oss_ops/graph_case.py#L326-L346)).
- A call to `nlp_concat_heads_decode` with more heads than the device has cores, because the op needs one core per head ([graph_case.py:1308-1334](../tests/gpt_oss_ops/graph_case.py#L1308-L1334)). The rule checks the function that the test calls, so the companion that runs the same captured case through the model's head merge is not skipped.

| Skipped case | 8x4 | 2x1 | Why | Covered by |
|---|---|---|---|---|
| `test_add[03_32x2880_bf16_ws-l1]` | yes | yes | captured SRAM shard grid needs core column 10 | nothing |
| `test_linear[00_32x4096_bf16_ws-l1]` | yes | yes | captured SRAM shard grid needs core column 10 | nothing |
| `test_nlp_concat_heads_decode[00_...]` | yes | yes | 64 heads, more than the cores of either grid | `test_nlp_concat_heads_decode_small_grid.py` |
| `test_sparse_matmul[01_...]`, `[03_...]` | yes | yes | captured compute grid 5x6 | `test_sparse_matmul_small_grid.py` |
| `test_sparse_matmul[00_...]`, `[02_...]` | no | yes | captured compute grid 3x4 | `test_sparse_matmul_small_grid.py` |
| `test_scaled_dot_product_attention[00_...]`, `[01_...]` | no | yes | captured compute grid 8x4 | `test_scaled_dot_product_attention_small_grid.py` |
| `test_paged_scaled_dot_product_attention_decode[00_...]`, `[01_...]` | no | yes | captured compute grid 8x4 | `test_paged_scaled_dot_product_attention_decode_small_grid.py` |
| `test_sparse_matmul_small_grid[00_...]`, `[02_...]` | yes | no | the captured call fits the 8x4 grid | `test_sparse_matmul.py` |

The companions run each skipped call in the form that the model uses on that grid, and they pass. The two shard-grid cases run on no Quasar grid.

## ttsim compared with craq-sim

ttsim ran the op suite faster than craq-sim: 60 min against 2 h 37 min on 8x4, and 107 min against 4 h 34 min on 2x1. The demo took about as long on 8x4 (2 h 6 min against 2 h) and less time on 2x1 (2 h 40 min against 3 h 38 min). The craq-sim times come from 2026-11-18, on the same branch before its rebase onto `840db3161d0`, so the trees were not the same. The craq-sim suite also had more cases: 138 on 8x4 and 139 on 2x1, against 131 for ttsim. ttsim needs no workaround for the TRISC data cache setting that craq-sim rejects (tenstorrent/craq-sim#549, `TT_METAL_ENABLE_L1_DATA_CACHE_CORES=all`). ttsim checks more: it found the softmax format bug, which craq-sim did not report.

## Open problems

- `test_add[03_...]` and `test_linear[00_...]` run on no Quasar grid, and no companion test covers them.
- #63905 is not on `main`.
