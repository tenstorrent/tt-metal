# SentenceBERT T3000

### Platforms:

T3000

**Note:** This demo is specifically designed for T3000 devices and uses optimized device parameters for maximum performance.

To obtain the perf reports through profiler, please build with following command:
```
./build_metal.sh -p
```

### Introduction

**bert-base-turkish-cased-mean-nli-stsb-tr** is a SentenceBERT-based model fine-tuned for semantic textual similarity and natural language inference tasks in Turkish. Built on a cased BERT architecture, it leverages mean pooling to generate dense sentence embeddings, enabling efficient and accurate sentence-level understanding. Optimized for performance in real-world NLP applications such as semantic search, clustering, and question answering.

Resource link - [source](https://huggingface.co/emrecan/bert-base-turkish-cased-mean-nli-stsb-tr)

### Details

- The entry point to the SentenceBERT model is located at: `models/demos/sentence_bert/ttnn/ttnn_sentence_bert.py`
- Batch size: 8 (configurable via device_batch_size parameter)
- Sequence Length: 384
- Data Types: bfloat16 (activation), bfloat8_b (weights)
- Device Parameters: Optimized for T3000 with L1 small size: 79104, trace region size: 23887872, num command queues: 2

### How to Run:

Use the following command to run the end-to-end performant model with Trace+2CQs (without mean-pooling):

```
pytest --disable-warnings models/demos/t3000/sentence_bert/tests/test_sentence_bert_e2e_performant.py::test_e2e_performant_sentencebert_data_parallel
```

### Performant Model with Trace+2CQ

> **Note:** SentenceBERT uses BERT-base as its backbone model.
- End-to-end performance without mean-pooling post-processing is **3073 sentences per second**
- Uses data parallel execution across multiple devices when available
- Optimized for T3000 architecture with specific device parameters

### Test Features

- **Data Parallel Execution**: Automatically scales batch size based on number of available devices
- **Trace Optimization**: Uses TTNN trace capture and execution for optimal performance
- **Memory Optimization**: Configured with T3000-specific memory parameters for optimal resource utilization

### Device Configuration

The test uses the following optimized device parameters for T3000:
- L1 small size: 79104
- Trace region size: 23887872
- Number of command queues: 2

### Performance Metrics

The test outputs detailed performance information including:
- Batch size (scaled by number of devices)
- Average inference time per iteration
- Sentences processed per second
