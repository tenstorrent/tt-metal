# gpt-oss on Quasar: unsupported data formats

Quasar is a new Tenstorrent chip. For now it runs in a simulator. This note explains why the gpt-oss demo stopped on Quasar with the error below until 2026-11-03, which errors came next, how two other models dealt with the same problem on Quasar, and which approach gpt-oss uses. DFB and `rm_in` are explained in section 1.3.

```
TT_FATAL: DFB 'rm_in' has data format 'UInt32' which is not supported on architecture quasar
```

On Quasar, `ttnn.slice` avoids this error with a raw data format (section 2.2), so gpt-oss runs its token embedding on the device (section 6.2).

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

Each op runs as a *program*: its kernels plus their DFBs. Before a program runs on Quasar, the tt-metal runtime checks every DFB of the program. If a DFB has a data format that Quasar does not support, the check stops the run with the error above ([dfb.cpp:111-119](../../../../../tt_metal/impl/metal2_host_api/program_spec/validation/resource/dfb.cpp#L111-L119)).

So an op fails on Quasar as soon as it sets up a DFB in an unsupported format. In practice this happens when the op reads or writes a tensor in that format. An upload writes data from the host into DRAM without running a kernel, so the model can upload `Bfp4_b` weights without an error. The first op that reads those weights fails.

## 2. The first error: token IDs in UInt32

### 2.1 Where it happens

The demo uses the `Generator` class from `models/ttt_compat`, shared model code in this repository. The model runs in two phases. *Prefill* processes the whole prompt in one pass. *Decode* then produces one new token per user in each step. When device traces are on, the generator runs one decode step with mock inputs before the first prefill, so that the decode programs are compiled in advance ([generator.py:1300-1310](../../../../ttt_compat/tt/generator.py#L1300-L1310)). Without traces (section 6.3), the first decode step comes after the prefill.

Each decode step runs `ttnn_decode_forward` in the gpt-oss model, which does this ([model.py:521-529](tt/model.py#L521-L529)):

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

### 2.2 Why it failed

Indexing a `ttnn` tensor with `[...]` runs the device op `ttnn.slice` ([core.py:44-168](../../../../../ttnn/ttnn/operations/core.py#L44-L168)). The slice program has a DFB named `rm_in` for its input. Until 2026-11-03, `rm_in` had the format of the input, `UInt32`, so the check in section 1.3 stopped the run.

The model code has no bug: the same slice works on Wormhole and Blackhole, where `UInt32` is supported.

On Quasar, the slice declares a `UInt16` or `UInt32` DFB as `RawUInt16` or `RawUInt32`, raw formats of the same element size that Quasar supports ([slice_device_operation.cpp:95-107](../../../../../ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_device_operation.cpp#L95-L107)). The slice only copies bytes, so the result is the same, and the error above does not occur.

### 2.3 Why changing the dtype is not a fix

Quasar supports `Int32`, and token IDs (below 201088, the gpt-oss vocabulary size) fit in `Int32`. But the next op, `ttnn.embedding`, accepts only `ttnn.uint32` or `ttnn.bfloat16` token IDs ([embedding_device_operation.cpp:35](../../../../../ttnn/cpp/ttnn/operations/embedding/device/embedding_device_operation.cpp#L35-L35)). So `Int32` token IDs would fail one op later. bf16 token IDs are not an option either: bf16 stores whole numbers exactly only up to 256.

## 3. The next errors: block float formats

The embedding in section 2.1 asks for its output in `ttnn.bfloat8_b`. Without the patches of section 6.1, its output DFB `out` has the format `Bfp8_b`, and the check in section 1.3 stops the embedding (tested on 2026-11-03 with the two `test_embedding` cases of `models/experimental/ops/quasar/tests/gpt_oss_ops/` that ask for a `bfloat8_b` output). The same holds for every op that sets up a DFB in `Bfp8_b` or `Bfp4_b`.

The model code asks for block float formats in many places, for example:

- The expert weights, in `ttnn.bfloat4_b` ([experts/weights.py:35](tt/experts/weights.py#L35-L35)). gpt-oss has 32 small feed-forward networks per layer, called *experts*, and uses 4 of them for each token.
- The `lm_head` weight, in `ttnn.bfloat8_b` ([model.py:220-228](tt/model.py#L220-L228)). `lm_head` is the last layer. It turns the model output into a score for each entry of the vocabulary.
- The attention weights and the KV cache, in `ttnn.bfloat8_b` ([attention/weights.py:33](tt/attention/weights.py#L33-L33), [attention/kv_cache.py:19](tt/attention/kv_cache.py#L19-L19)). The KV cache keeps the attention keys and values of earlier tokens.
- Ops that convert their result to `ttnn.bfloat8_b`, for example [attention/operations.py:113](tt/attention/operations.py#L113-L113).

Issue [#60132](https://github.com/tenstorrent/tt-metal/issues/60132), which tracks the work to run gpt-oss on Quasar, expects this and says to use bf16 instead.

### 3.1 The cost of bf16

bf16 needs much more DRAM (section 1.2). The simulated device has 2 DRAM banks of 3 GiB each, 6 GiB in total. The demo setting `QSR_N_LAYERS=2` builds only 2 of the 24 layers of gpt-oss-20b, to keep the simulation short. The weights then need:

| Part | Dtype in the model code | In bf16 |
|---|---|---|
| Expert weights, per layer | 0.42 GiB (`Bfp4_b`) | 1.48 GiB |
| Expert weights, 2 layers | 0.83 GiB | 2.97 GiB |
| `lm_head` (2880 x 262144) | 0.75 GiB (`Bfp8_b`) | 1.41 GiB |
| Embedding table (201088 x 2880) | 1.08 GiB (bf16) | 1.08 GiB |
| **Total** | **2.66 GiB** | **5.45 GiB** |

The `lm_head` has 262144 columns, not 201088: the code rounds the vocabulary size up to the next power of 2, because `ttnn.topk`, the op that picks the highest scores from the `lm_head` output, can then use more than one core ([model.py:25-35](tt/model.py#L25-L35)).

The table does not include the attention weights (about 0.05 GiB per layer), the KV cache, or the activations (the temporary results that pass between layers). In bf16 the weights alone use 5.45 GiB of 6 GiB, so with the rest added the model may not fit. Without the embedding table on the device, the weights need 4.37 GiB.

A demo run on craq-sim on 2026-11-03, with `QSR_N_LAYERS=2` and the embedding table on the device, finished the model setup without an allocation error, so the weights, the table and the KV cache fit. The run stopped in the first layer on the multicast NoC check of the mainline `ttnn.matmul` (issue #63758), so whether the activations also fit was not tested.

## 4. Two ways to change the model for Quasar

**Direct edit.** Change the model code. The Quasar copy still runs on Wormhole and Blackhole, so Quasar-only behavior goes in an `if is_quasar():` branch. A reader of the model code sees the Quasar behavior where it applies.

**Monkeypatching.** Replace a function at run time from outside the code that uses it, without changing the source file. pytest's `monkeypatch` fixture (a helper object that pytest gives to a test or setup function) does this, and restores the original function when the test ends. For example, a setup function in `conftest.py` (a file that pytest loads for all tests in its directory) can replace `ttnn.from_torch` with a wrapper that changes the requested dtype. A patch applies to every call of the function, so one patch can change many call sites. But a patch is not visible in the model code.

## 5. What the other Quasar models did

### 5.1 llama32_1b_quasar: patches

The llama end-to-end test installs many patches when it runs on Quasar ([test_llama_e2e.py:1262-1348](../../../llama32_1b_quasar/tests/demos/llama32_1b/test_llama_e2e.py#L1262-L1348)). Two are relevant here.

**Block float to bf16.** The patch of `ttnn.from_torch` uploads `ttnn.bfloat8_b` and `ttnn.bfloat4_b` weights as bf16 ([test_llama_e2e.py:142-153](../../../llama32_1b_quasar/tests/demos/llama32_1b/test_llama_e2e.py#L142-L153)). Other patches change ops that ask for a block float result so that they produce bf16, for example the eltwise ops ([test_llama_e2e.py:1205-1210](../../../llama32_1b_quasar/tests/demos/llama32_1b/test_llama_e2e.py#L1205-L1210)). The KV cache is also made bf16 ([test_llama_e2e.py:271-300](../../../llama32_1b_quasar/tests/demos/llama32_1b/test_llama_e2e.py#L271-L300)).

**Embedding on the host.** A patch replaces the forward function of the llama embedding class ([test_llama_e2e.py:367-411](../../../llama32_1b_quasar/tests/demos/llama32_1b/test_llama_e2e.py#L367-L411)). It copies the token IDs to the host, looks them up in a host copy of the embedding table with torch, and uploads only the result as bf16. The embedding table never goes to the device. The comments give the reason: uploading the table took most of the setup time in the simulator, and the device embedding would work ("device-viable, host is a perf choice", [test_llama_e2e.py:1309-1310](../../../llama32_1b_quasar/tests/demos/llama32_1b/test_llama_e2e.py#L1309-L1310)). As a result, no device op reads the `UInt32` token IDs, so llama never needed a patch for `ttnn.slice`.

### 5.2 resnet50/quasar: direct edits

The resnet Quasar copy (`models/demos/vision/classification/resnet50/quasar/`) changes the code of its model and its end-to-end test in `if is_quasar():` branches. For example, on Quasar it uploads the input image in a different layout ([resnet50_test_infra.py:294-310](../../../../demos/vision/classification/resnet50/quasar/tests/common/resnet50_test_infra.py#L294-L310)). Only some of its single-op tests use patches.

For data formats, resnet needed no code change. The resnet model takes the dtypes of activations and weights as arguments, and the Quasar end-to-end test passes bf16 for both ([test_resnet50_e2e.py:73-78](../../../../demos/vision/classification/resnet50/quasar/tests/test_resnet50_e2e.py#L73-L78)).

Resnet has no token IDs, so it never had the `UInt32` problem.

## 6. Recommendation for gpt-oss

### 6.1 Block float formats: patches, as in llama

Resnet did not need to change each call site, because its dtypes are arguments. In gpt-oss, `ttnn.bfloat8_b` and `ttnn.bfloat4_b` appear 34 times in 18 files under `tt/`. `Attention` and `Experts` take a `weight_dtype` argument ([attention/\_\_init\_\_.py:36](tt/attention/__init__.py#L36-L36), [experts/\_\_init\_\_.py:57](tt/experts/__init__.py#L57-L57)), so their weights could be set to bf16 the way resnet does it. But the dtypes of the activations, the `ttnn.typecast` calls and the `lm_head` are fixed in the code, and they are most of the 34. Changing each one would touch many files. So a set of patches in `conftest.py` covers all of them, in the same way as in llama, and the two `weight_dtype` arguments are left unchanged. On Quasar:

- `ttnn.from_torch` and `ttnn.as_tensor` are wrapped, so that a request for `ttnn.bfloat8_b` or `ttnn.bfloat4_b` gives `ttnn.bfloat16`.
- The ops that the model calls with a block float result dtype (`ttnn.typecast`, `ttnn.embedding`, `ttnn.matmul`, `ttnn.linear` and `ttnn.sparse_matmul`) are wrapped in the same way.

An op that is not in this list, and that sets up a block float DFB, still fails. The fix is to add it to the list. The cost of the patches is the extra DRAM in section 3.1.

The model also keeps a cache of its converted weights on disk, one file per weight, in one directory. The file name contains the dtype, so a bf16 weight never loads a block float file ([core.py:972](../../../../../ttnn/ttnn/operations/core.py#L972-L972)). But when the model has written every weight to the cache, it also marks the whole directory as complete ([model_config.py:315-356](tt/model_config.py#L315-L356)). On the next run, a complete directory makes the model skip loading the original weights ([common.py:69-72](tt/common.py#L69-L72)). A directory completed with block float weights has no bf16 files, so with the patches active every weight would be missing. So on Quasar the patches point the cache to a separate directory.

### 6.2 Token IDs: the embedding on the device

gpt-oss runs its token embedding on the device on Quasar, with the same model code as on Wormhole and Blackhole: the slice of the token IDs in `ttnn_decode_forward` (section 2.1), and `ttnn.embedding` there and in the two prefill functions `transform_and_embed_prefill_inputs_device` ([model.py:1056](tt/model.py#L1056-L1056)) and `prepare_inputs_prefill` ([model.py:1106](tt/model.py#L1106-L1106)). Four things make this work on Quasar:

1. The slice declares its `UInt32` DFBs as `RawUInt32` (section 2.2).
2. `ttnn.embedding` accepts `UInt32` token IDs. It keeps them in a scratchpad (SRAM that only one kernel uses), not in a DFB, so the check in section 1.3 does not apply to them. This holds for both of its factories that the model uses: the row-major factory, for a number of IDs that is not a multiple of 32 ([embeddings_rm_program_factory.cpp:181-183](../../../../../ttnn/cpp/ttnn/operations/embedding/device/embeddings_rm_program_factory.cpp#L181-L183)), and the fused factory, which also tilizes the output ([embeddings_fused_program_factory.cpp:211-213](../../../../../ttnn/cpp/ttnn/operations/embedding/device/embeddings_fused_program_factory.cpp#L211-L213)).
3. The patches of section 6.1 change the `bfloat8_b` output of `ttnn.embedding` to bf16.
4. The bf16 table fits in DRAM next to the weights of 2 layers (section 3.1).

On 2026-11-03, a test on craq-sim ran these steps with a table of the real size (201088 x 2880): the decode steps (the slice of 32 token IDs to 1, then `ttnn.embedding`), the embedding of all 32 decode IDs, and the embedding of 128 prefill IDs. All outputs were bit-exact with a torch lookup.

Llama runs its embedding on the host instead (section 5.1). gpt-oss does not do this: an op that runs on the host only hides a problem, and the model run would not show whether the op works on Quasar.

### 6.3 Device traces must be off in slow dispatch

A *trace* records the device commands of a step and later replays them without the host. The craq-sim runs of gpt-oss use slow dispatch mode (`TT_METAL_SLOW_DISPATCH_MODE=1`): the host sends each program to the device one at a time. In slow dispatch, the command queue rejects trace capture and replay ([sd_mesh_command_queue.cpp:487-493](../../../../../tt_metal/distributed/sd_mesh_command_queue.cpp#L487-L493)). The Quasar run uses the demo configuration `prefill_128`, which turns on both the decode trace and the prefill trace ([text_demo.py:210-211](demo/text_demo.py#L210-L211)).

So the demo turns both traces off when `TT_METAL_SLOW_DISPATCH_MODE` is set ([text_demo.py:486-490](demo/text_demo.py#L486-L490)), just before it does the same when ttnn logging is on.

### 6.4 Summary

| Problem | Approach | Same as | Code |
|---|---|---|---|
| `Bfp8_b` and `Bfp4_b` in many places, including the embedding output | Patches in `conftest.py` that map them to bf16 | llama | [conftest.py:44-104](conftest.py#L44-L104) |
| `UInt32` token IDs in the decode slice | None in the model: the slice declares `RawUInt32` DFBs on Quasar | | [slice_device_operation.cpp:95-107](../../../../../ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_device_operation.cpp#L95-L107) |
| Traces in slow dispatch | Traces off when `TT_METAL_SLOW_DISPATCH_MODE` is set | the demo's own logging case | [text_demo.py:486-490](demo/text_demo.py#L486-L490) |
