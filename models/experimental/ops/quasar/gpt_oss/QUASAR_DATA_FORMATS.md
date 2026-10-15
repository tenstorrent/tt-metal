# gpt-oss on Quasar: unsupported data formats

Quasar is a new Tenstorrent chip. For now it runs in a simulator. This note explains why the gpt-oss demo stops on Quasar with the error below, which errors will most likely come next, how two other models dealt with the same problem on Quasar, and which approach to use for gpt-oss. DFB and `rm_in` are explained in section 1.3.

```
TT_FATAL: DFB 'rm_in' has data format 'UInt32' which is not supported on architecture quasar
```

The model code is in `models/experimental/ops/quasar/gpt_oss/`. It is a copy of `models/demos/gpt_oss/`, kept separate so that it can be changed for Quasar.

## 1. Background

### 1.1 Data formats

Every tensor has a data type (`dtype`), for example `ttnn.bfloat16` or `ttnn.uint32`. When a kernel reads or writes the tensor, the hardware must know how the bytes are encoded. This encoding is the *data format*. Each `ttnn` dtype has a matching data format.

Not every chip supports every data format. The code keeps one list of supported formats for the current chips, Wormhole and Blackhole ([tt_backend_api_types.cpp:69-95](../../../../../tt_metal/common/tt_backend_api_types.cpp#L69-L95)), and one list for Quasar ([tt_backend_api_types.cpp:97-122](../../../../../tt_metal/common/tt_backend_api_types.cpp#L97-L122)). Three formats matter here:

| Data format | `ttnn` dtype | Wormhole, Blackhole | Quasar |
|---|---|---|---|
| `UInt32` | `ttnn.uint32` | yes | **no** (`Int32` is supported) |
| `Bfp8_b` | `ttnn.bfloat8_b` | yes | **no** |
| `Bfp4_b` | `ttnn.bfloat4_b` | yes | **no** |

### 1.2 Block float formats

`Bfp8_b` and `Bfp4_b` are *block float* formats. Normal floating point numbers store an exponent for each value. A block float format stores one exponent for a group of 16 values, and only a small mantissa (with sign) for each value. This makes the tensor much smaller.

A tensor in tile layout is stored in tiles of 32 x 32 = 1024 values. (A tensor in row-major layout is stored row by row, without tiles.) The size of one tile in each format, compared with bf16 (16-bit float, `ttnn.bfloat16`):

| Format | Per value | Exponents | Bytes per tile | Size compared to bf16 |
|---|---|---|---|---|
| bf16 | 2 bytes | (in each value) | 2048 | 1 |
| `Bfp8_b` | 1 byte | 64 bytes (one per 16 values) | 1088 | 0.53 |
| `Bfp4_b` | 0.5 bytes | 64 bytes | 576 | 0.28 |

### 1.3 Dataflow buffers and the format check

A kernel does not work on a tensor in DRAM directly. It moves data into small buffers in the SRAM of its core, works on it there, and moves the results out. On Wormhole and Blackhole these buffers are called circular buffers. On Quasar they are called *dataflow buffers* (DFBs). Each DFB has a name, chosen by the op (for example `rm_in`, the row-major input of a slice op), and a data format.

Each op runs as a *program*: its kernels plus their DFBs. Before a program runs on Quasar, the tt-metal runtime checks every DFB of the program. If a DFB has a data format that Quasar does not support, the check stops the run with the error above ([dfb.cpp:103-111](../../../../../tt_metal/impl/metal2_host_api/program_spec/validation/resource/dfb.cpp#L103-L111)).

So an op fails on Quasar as soon as it sets up a DFB in an unsupported format. In practice this happens when the op reads or writes a tensor in that format. An upload writes data from the host into DRAM without running a kernel, so the model can upload `Bfp4_b` weights without an error. The first op that reads those weights fails.

## 2. The current error: token IDs in UInt32

### 2.1 Where it happens

The demo uses the `Generator` class from `tt_transformers`, a shared library of model code in this repository. The model runs in two phases. *Prefill* processes the whole prompt in one pass. *Decode* then produces one new token per user in each step. Before prefill, the generator runs one decode step with dummy inputs, so that the decode programs are compiled in advance ([generator.py:646-668](../../../../tt_transformers/tt/generator.py#L646-L668), called from [generator.py:1237](../../../../tt_transformers/tt/generator.py#L1237-L1237)).

That decode step runs `ttnn_decode_forward` in the gpt-oss model. On Wormhole and Blackhole, `ttnn_decode_forward` does this (on Quasar it uses the host embedding described in section 6.2) ([model.py:536-547](tt/model.py#L536-L547)):

```python
actual_batch = current_pos.shape[-1]
...
if not self.users_row_sharded and tokens.shape[-1] > actual_batch:
    tokens_for_embed = tokens[:, :, :, :actual_batch]
...
input_embeds = ttnn.embedding(
    tokens_for_embed, self.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b
)
```

`tokens` holds the token IDs of the batch, in `ttnn.uint32`. Its shape is `[1, 1, 1, 32]`, because the batch is padded to 32 users. `current_pos` has one entry per real user, so `actual_batch` is 1 in the demo. The `if` branch is taken, and the code cuts `tokens` down to `[1, 1, 1, 1]`.

### 2.2 Why it fails

Indexing a `ttnn` tensor with `[...]` runs the device op `ttnn.slice` ([core.py:43-148](../../../../../ttnn/ttnn/operations/core.py#L43-L148)). The slice program has a DFB named `rm_in` for its input. The input is `UInt32`, so `rm_in` is `UInt32`, and the check in section 1.3 stops the run.

The model code has no bug: the same slice works on Wormhole and Blackhole, where `UInt32` is supported.

### 2.3 Why changing the dtype is not a fix

Quasar supports `Int32`, and token IDs (below 201088, the gpt-oss vocabulary size) fit in `Int32`. But the next op, `ttnn.embedding`, accepts only `ttnn.uint32` or `ttnn.bfloat16` token IDs ([embedding_device_operation.cpp:35](../../../../../ttnn/cpp/ttnn/operations/embedding/device/embedding_device_operation.cpp#L35-L35)). So `Int32` token IDs would fail one op later. bf16 token IDs are not an option either: bf16 stores whole numbers exactly only up to 256.

## 3. The next errors: block float formats

The embedding in section 2.1 asks for its output in `ttnn.bfloat8_b`. My guess is that its output DFB uses that format, `Bfp8_b`. If it does, the check in section 1.3 stops the embedding. The same holds for every op that sets up a DFB in `Bfp8_b` or `Bfp4_b`.

The model uses block float formats in many places, for example:

- The expert weights, in `ttnn.bfloat4_b` ([experts/weights.py:35](tt/experts/weights.py#L35-L35)). gpt-oss has 32 small feed-forward networks per layer, called *experts*, and uses 4 of them for each token.
- The `lm_head` weight, in `ttnn.bfloat8_b` ([model.py:234-242](tt/model.py#L234-L242)). `lm_head` is the last layer. It turns the model output into a score for each entry of the vocabulary.
- The attention weights and the KV cache, in `ttnn.bfloat8_b` ([attention/weights.py:33](tt/attention/weights.py#L33-L33), [attention/kv_cache.py:19](tt/attention/kv_cache.py#L19-L19)). The KV cache keeps the attention keys and values of earlier tokens.
- Ops that convert their result to `ttnn.bfloat8_b`, for example [attention/operations.py:113](tt/attention/operations.py#L113-L113).

Issue [#60132](https://github.com/tenstorrent/tt-metal/issues/60132), which tracks the work to run gpt-oss on Quasar, expects this and says to use bf16 instead.

### 3.1 The cost of bf16

bf16 needs much more DRAM (section 1.2). The simulated device has 2 DRAM banks of 3 GiB each, 6 GiB in total. The demo setting `QSR_N_LAYERS=2` builds only 2 of the 24 layers of gpt-oss-20b, to keep the simulation short. The weights then need:

| Part | Current dtype | In bf16 |
|---|---|---|
| Expert weights, per layer | 0.42 GiB (`Bfp4_b`) | 1.48 GiB |
| Expert weights, 2 layers | 0.83 GiB | 2.97 GiB |
| `lm_head` (2880 x 262144) | 0.75 GiB (`Bfp8_b`) | 1.41 GiB |
| Embedding table (201088 x 2880) | 1.08 GiB (bf16) | 1.08 GiB |
| **Total** | **2.66 GiB** | **5.45 GiB** |

The `lm_head` has 262144 columns, not 201088: the code rounds the vocabulary size up to the next power of 2, because `ttnn.topk`, the op that picks the highest scores from the `lm_head` output, can then use more than one core ([model.py:25-35](tt/model.py#L25-L35)).

The table does not include the attention weights (about 0.05 GiB per layer), the KV cache, or the activations (the temporary results that pass between layers). In bf16 the weights alone use 5.45 GiB of 6 GiB, so with the rest added the model may not fit. Without the embedding table on the device, the weights need 4.37 GiB.

## 4. Two ways to change the model for Quasar

**Direct edit.** Change the model code. The Quasar copy still runs on Wormhole and Blackhole, so Quasar-only behavior goes in an `if is_quasar():` branch. A reader of the model code sees the Quasar behavior where it applies.

**Monkeypatching.** Replace a function at run time from outside the code that uses it, without changing the source file. pytest's `monkeypatch` fixture (a helper object that pytest gives to a test or setup function) does this, and restores the original function when the test ends. For example, a setup function in `conftest.py` (a file that pytest loads for all tests in its directory) can replace `ttnn.from_torch` with a wrapper that changes the requested dtype. A patch applies to every call of the function, so one patch can change many call sites. But a patch is not visible in the model code.

## 5. What the other Quasar models did

### 5.1 llama32_1b_quasar: patches

The llama end-to-end test installs many patches when it runs on Quasar ([test_llama_e2e.py:1373-1480](../../../llama32_1b_quasar/tests/demos/llama32_1b/test_llama_e2e.py#L1373-L1480)). Two are relevant here.

**Block float to bf16.** The patch of `ttnn.from_torch` uploads `ttnn.bfloat8_b` and `ttnn.bfloat4_b` weights as bf16 ([test_llama_e2e.py:144-153](../../../llama32_1b_quasar/tests/demos/llama32_1b/test_llama_e2e.py#L144-L153)). Other patches change ops that ask for a block float result so that they produce bf16 ([test_llama_e2e.py:1334-1340](../../../llama32_1b_quasar/tests/demos/llama32_1b/test_llama_e2e.py#L1334-L1340)). The KV cache is also made bf16 ([test_llama_e2e.py:396-423](../../../llama32_1b_quasar/tests/demos/llama32_1b/test_llama_e2e.py#L396-L423)).

**Embedding on the host.** A patch replaces the forward function of the llama embedding class ([test_llama_e2e.py:496-540](../../../llama32_1b_quasar/tests/demos/llama32_1b/test_llama_e2e.py#L496-L540)). It copies the token IDs to the host, looks them up in a host copy of the embedding table with torch, and uploads only the result as bf16. The embedding table never goes to the device. The comment of the patch gives the reason: uploading the table took most of the setup time in the simulator. As a result, no device op reads the `UInt32` token IDs, so llama never needed a patch for `ttnn.slice`.

### 5.2 resnet50/quasar: direct edits

The resnet Quasar copy (`models/demos/vision/classification/resnet50/quasar/`) changes the code of its model and its end-to-end test in `if is_quasar():` branches. For example, on Quasar it uploads the input image in a different layout ([resnet50_test_infra.py:294-310](../../../../demos/vision/classification/resnet50/quasar/tests/common/resnet50_test_infra.py#L294-L310)). Only some of its single-op tests use patches.

For data formats, resnet needed no code change. The resnet model takes the dtypes of activations and weights as arguments, and the Quasar end-to-end test passes bf16 for both ([test_resnet50_e2e.py:69-76](../../../../demos/vision/classification/resnet50/quasar/tests/test_resnet50_e2e.py#L69-L76)).

Resnet has no token IDs, so it never had the `UInt32` problem.

## 6. Recommendation for gpt-oss

### 6.1 Block float formats: patches, as in llama

Resnet did not need to change each call site, because its dtypes are arguments. In gpt-oss, `ttnn.bfloat8_b` and `ttnn.bfloat4_b` appear 41 times in 18 files under `tt/`. `Attention` and `Experts` take a `weight_dtype` argument ([attention/\_\_init\_\_.py:36](tt/attention/__init__.py#L36-L36), [experts/\_\_init\_\_.py:57](tt/experts/__init__.py#L57-L57)), so their weights could be set to bf16 the way resnet does it. But the dtypes of the activations, the `ttnn.typecast` calls and the `lm_head` are fixed in the code, and they are most of the 41. Changing each one would touch many files. So a set of patches in `conftest.py` covers all of them, in the same way as in llama, and the two `weight_dtype` arguments are left unchanged. On Quasar:

- `ttnn.from_torch` and `ttnn.as_tensor` are wrapped, so that a request for `ttnn.bfloat8_b` or `ttnn.bfloat4_b` gives `ttnn.bfloat16`.
- The ops that the model calls with a block float result dtype (`ttnn.typecast`, `ttnn.matmul`, `ttnn.linear` and `ttnn.sparse_matmul`) are wrapped in the same way.

An op that is not in this list, and that sets up a block float DFB, still fails. The fix is to add it to the list. The cost of the patches is the extra DRAM in section 3.1.

The model also keeps a cache of its converted weights on disk, one file per weight, in one directory. The file name contains the dtype, so a bf16 weight never loads a block float file ([core.py:925-933](../../../../../ttnn/ttnn/operations/core.py#L925-L933)). But when the model has written every weight to the cache, it also marks the whole directory as complete ([model_config.py:315-356](tt/model_config.py#L315-L356)). On the next run, a complete directory makes the model skip loading the original weights ([common.py:69-72](tt/common.py#L69-L72)). A directory completed with block float weights has no bf16 files, so with the patches active every weight would be missing. So on Quasar the patches point the cache to a separate directory.

### 6.2 Token IDs: an embedding on the host, as a direct edit

The slice and the embedding are next to each other in `ttnn_decode_forward`, so one edit there removes both. Prefill calls `ttnn.embedding` in two more functions: `transform_and_embed_prefill_inputs_device` ([model.py:1087-1098](tt/model.py#L1087-L1098)) and `prepare_inputs_prefill` ([model.py:1147-1154](tt/model.py#L1147-L1154)). In llama the embedding is a separate class, so one patch could replace it. gpt-oss has no such class. Two smaller patches could still work: one that replaces `ttnn.slice` with a host slice for `UInt32` inputs, and one that replaces `ttnn.embedding` with a host lookup. But `ttnn.embedding` receives the table as a device tensor, so with these patches the table stays on the device, and the DRAM saving below is lost. Only a direct edit in the Quasar copy, as in resnet, keeps the table off the device. On Quasar:

- The model keeps the embedding table on the host and does not upload it.
- At each embedding call site, the model copies the token IDs to the host. This copy runs no kernel, so no DFB is set up. In decode, the model keeps only the first `actual_batch` token IDs. It looks the IDs up in the table with torch, and uploads the result as bf16.

This one change removes three problems:

1. No device op reads `UInt32` token IDs, so the slice error in section 2 does not occur.
2. The embedding output is no longer `Bfp8_b` (section 3).
3. The device needs 1.08 GiB less DRAM (section 3.1).

### 6.3 Device traces must be off

A *trace* records the device commands of a step and later replays them without the host. Host work inside the recorded step is not replayed. The Quasar run uses the demo configuration `prefill_128`, which turns on both the decode trace and the prefill trace ([text_demo.py:210-211](demo/text_demo.py#L210-L211)). The decode trace records a call of `ttnn_decode_forward` ([generator.py:2391-2415](../../../../tt_transformers/tt/generator.py#L2391-L2415)). The prefill trace records a call of `transform_and_embed_prefill_inputs_device` ([generator.py:824-827](../../../../tt_transformers/tt/generator.py#L824-L827), [generator.py:709-715](../../../../tt_transformers/tt/generator.py#L709-L715)). Both functions contain the host embedding, so a replay of either trace would skip the embedding.

So on Quasar the demo must run without traces. The demo already turns both traces off when ttnn logging is on ([text_demo.py:497-501](demo/text_demo.py#L497-L501)). The Quasar copy adds an `if is_quasar():` condition just before it, which turns both traces off on Quasar.

My guess, not checked, is that the simulator needs traces off in any case: it runs in slow dispatch mode (the host sends each program to the device one at a time), and traces need fast dispatch mode (the device fetches the programs itself).

### 6.4 Summary

| Problem | Approach | Same as | Code |
|---|---|---|---|
| `Bfp8_b` and `Bfp4_b` in many places | Patches in `conftest.py` that map them to bf16 | llama | [conftest.py:44-100](conftest.py#L44-L100) |
| `UInt32` token IDs, embedding output, DRAM used by the embedding table | Direct `if is_quasar():` edit: embedding on the host | llama's host embedding, done as a direct edit like resnet | [model.py:166-187](tt/model.py#L166-L187), [model.py:1066-1085](tt/model.py#L1066-L1085) |
| Host embedding inside traced steps | Direct `if is_quasar():` edit in the demo: traces off | the demo's own logging case | [text_demo.py:486-491](demo/text_demo.py#L486-L491) |
