# metal_BERT_large 11

## Platforms:
    Wormhole (n150, n300), Blackhole (P100)

## Introduction
This directory runs BERT-Large for extractive question answering on device, at a sequence length
of 384: the embeddings, the 24 encoder layers and the span-prediction head that gives each
token's start and end logits.

BERT is a bidirectional transformer encoder, pretrained on unlabeled text by conditioning on both
left and right context in every layer and fine-tuned for a task with one output layer.
https://arxiv.org/abs/1810.04805

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

The tests and the demo load `phiyodr/bert-large-finetuned-squad2` from Hugging Face on first use.

## Demo

From the repository root, answer the questions in `demo/input_data.json`, replacing `BATCH_SIZE`
with 7, 8 or 12 (7 or 8 on Wormhole):

```sh
pytest --disable-warnings models/demos/metal_BERT_large_11/demo/demo.py::test_demo -k BATCH_SIZE
```

To answer other questions, replace the entries of `demo/input_data.json`; the demo takes the first
`BATCH_SIZE` of them.

`test_demo_squadv2` scores the model on SQuAD v2 questions. It reads the dataset with
`datasets.load_from_disk` from `/mnt/MLPerf/tt_dnn-models/squad_v2`, a Tenstorrent-internal path
(`models/datasets/dataset_squadv2.py`); its test in `test_demo.py` runs on Wormhole only for now:

```sh
pytest --disable-warnings models/demos/metal_BERT_large_11/demo/demo.py::test_demo_squadv2 -k BATCH_SIZE
```

## Performance

- End-to-end throughput at batch 8; its targets are Wormhole ones, so it runs on Wormhole only:

```sh
pytest models/demos/metal_BERT_large_11/tests/test_perf_bert11.py::test_perf_bare_metal_wh
```
- Device kernel throughput of `test_bert.py` at batch 8:

```sh
pytest models/demos/metal_BERT_large_11/tests/test_perf_device_bert.py::test_perf_device_bare_metal_wh
```

## Testing

Each test runs a set of batch sizes and model configs (bfloat16 or bfloat8_b, activations in
DRAM, L1 or sharded) and checks the outputs against the Hugging Face model:

- `test_bert_batch_dram.py` and `test_bert.py`: the whole model, on the start and end logits
- `test_bert_encoder.py`, `test_mha.py`, `test_ffn.py`: one encoder layer, its self-attention and
  its feed-forward block

```sh
pytest --disable-warnings models/demos/metal_BERT_large_11/tests/test_bert_batch_dram.py
```

### Reduced grid

The model checks each of its hand-sized plans against the device's grid (`tt/grid.py`):

- The interleaved matmul program configs take one core row per image and 8 to 12 core columns. On
  a grid with fewer rows than the batch, or fewer than 8 columns, ttnn chooses each matmul's
  program config.
- The L1 and mixed-precision configs keep their activations in L1 on a grid with a core row per
  image and at least 11 columns, and in DRAM otherwise.
- `BFLOAT8_B-SHARDED` needs a grid of at least 8 columns by `batch` rows, or `batch` columns by 8
  rows. On a smaller grid the model runs the `BFLOAT8_B-DRAM` config instead; on Wormhole that
  test is skipped.

The tests and the demo pass on two worker cores:

```sh
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=1,0 pytest --disable-warnings models/demos/metal_BERT_large_11/tests/test_bert_batch_dram.py
```

The override is the inclusive end coordinate of the worker grid, so `1,0` is a 2x1 grid.

### Reduced layer count

`BERT_NUM_LAYERS=N` (1 to 24) keeps the first `N` encoder layers, in the Hugging Face reference and
in the TT model. The whole-model tests (`test_bert_batch_dram.py`, `test_bert.py`), the end-to-end
perf test and the demo read it:

```sh
BERT_NUM_LAYERS=1 pytest --disable-warnings models/demos/metal_BERT_large_11/tests/test_bert_batch_dram.py
```

## Project Structure

```
models/demos/metal_BERT_large_11/
├── demo/
│   ├── demo.py             # Question answering on input_data.json and on SQuAD v2
│   └── input_data.json     # Default demo questions and contexts
├── tests/
│   ├── test_bert_batch_dram.py  # E2E PCC, with and without the program cache
│   ├── test_bert.py             # E2E PCC; the model the device perf test profiles
│   ├── test_bert_encoder.py     # One encoder layer
│   ├── test_mha.py              # Self-attention of one layer
│   ├── test_ffn.py              # Feed-forward block of one layer
│   ├── test_demo.py             # Demo answers and SQuAD v2 scores
│   ├── test_perf_bert11.py      # End-to-end throughput
│   └── test_perf_device_bert.py # Device kernel throughput
└── tt/
    ├── bert_model.py       # TtBertBatchDram: embeddings, encoder layers, QA head
    ├── bert_encoder.py     # One encoder layer
    ├── mha.py              # Self-attention
    ├── ffn.py              # Feed-forward block
    ├── embeddings.py       # Word, position and token-type embeddings
    ├── custom_matmuls.py   # Matmuls and their program configs, from the device's grid
    ├── model_config.py     # Memory configs and dtypes per model config, BERT_NUM_LAYERS
    └── tensor_utils.py     # Weight caching
```

## Model Structure

`BertForQuestionAnswering` from Hugging Face, as `print(model)` shows it:

```
BertForQuestionAnswering(
  (bert): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(30522, 1024, padding_idx=0)
      (position_embeddings): Embedding(512, 1024)
      (token_type_embeddings): Embedding(2, 1024)
      (LayerNorm): LayerNorm((1024,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0-23): 24 x BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=1024, out_features=1024, bias=True)
              (key): Linear(in_features=1024, out_features=1024, bias=True)
              (value): Linear(in_features=1024, out_features=1024, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=1024, out_features=1024, bias=True)
              (LayerNorm): LayerNorm((1024,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=1024, out_features=4096, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=4096, out_features=1024, bias=True)
            (LayerNorm): LayerNorm((1024,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
  )
  (qa_outputs): Linear(in_features=1024, out_features=2, bias=True)
)
```

Indented under each file are the modules from the tree it implements, with the class or function
in parentheses.

```
tt/bert_model.py
    BertForQuestionAnswering (TtBertBatchDram), qa_outputs included

tt/embeddings.py
    embeddings (TtEmbeddings)

tt/bert_encoder.py
    BertLayer (TtBertEncoder)
    attention.output.dense (TtBertEncoder.__call__)
    attention.output.LayerNorm, with its residual add (TtBertEncoder.op8_add_layernorm)
    output.LayerNorm, with its residual add (TtBertEncoder.op11_add_layernorm)

tt/mha.py
    BertSelfAttention (TtMultiHeadAttentionModel), query, key and value fused into one matmul

tt/ffn.py
    intermediate and output dense layers (TtFeedForwardModel)
```

## Details
- Entry point is `TtBertBatchDram` in `tt/bert_model.py`, given the model config
  `get_model_config(batch, device.compute_with_storage_grid_size(), model_config_str)` returns.
- Batch size: 7, 8, 9 or 12, by model config; sequence length 384.
- Inputs: token ids, token type ids and attention mask; outputs: start and end logits per token.
- Parameters: 334M.
