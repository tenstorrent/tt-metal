# SentenceBERT

## Platforms:
    Wormhole (n150, n300), Blackhole (p150), QuietBox, LoudBox, Galaxy

## Introduction
**bert-base-turkish-cased-mean-nli-stsb-tr** is a SentenceBERT-based model fine-tuned for semantic textual similarity and natural language inference tasks in Turkish. Built on a cased BERT architecture, it leverages mean pooling to generate dense sentence embeddings, enabling efficient and accurate sentence-level understanding. Optimized for performance in real-world NLP applications such as semantic search, clustering, and question answering.

Resource link - [source](https://huggingface.co/emrecan/bert-base-turkish-cased-mean-nli-stsb-tr)

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run

Find sentence_bert instructions for the following device implementations:

- Wormhole: [demos/wormhole/sentence_bert/README](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/wormhole/sentence_bert/README.md)

- Blackhole:[demos/blackhole/sentence_bert/README](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/blackhole/sentence_bert/README.md)

- QuietBox / LoudBox: [demos/t3000/sentence_bert/README](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/t3000/sentence_bert/README.md)

- Galaxy: [demos/tg/sentence_bert/README](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/tg/sentence_bert/README.md)
